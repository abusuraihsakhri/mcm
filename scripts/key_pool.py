"""A pool of API keys with independent quotas, rotated as limits are hit.

One free-tier key allows a few requests per minute and a bounded number per day.
Several keys are several independent quotas, so the pool exists to spend whichever
one is available rather than queueing behind whichever one was used last.

Three rules shape the design.

**Quotas are per key, not per pool.** Each key carries its own minute spacing and
its own daily counter. A key that has spent its day is skipped while the others
keep working, and the run only stops when every key is done.

**A 429 is information, not an error to retry through.** The key that produced it
goes into cooldown and the next request goes to a different key. Retrying the
same key faster is how a free tier turns into a blocked one.

**Keys never touch disk or logs.** Daily counters have to persist, so the state
file is keyed by a short fingerprint derived from a hash of the key. A leaked
state file reveals how much quota was spent and nothing else.

    $env:GEMINI_API_KEYS = "key1,key2,key3"
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from dataclasses import dataclass, field


class AllKeysExhausted(RuntimeError):
    """Every key has spent its daily budget. The run should checkpoint and stop."""


#: First backoff, in seconds, when a 429 arrives with no delay attached. It
#: doubles from here, so a provider that keeps refusing is still backed away
#: from quickly. Ten rather than sixty because sixty is a guess shaped by
#: Gemini, which at least says when to return; NVIDIA sends `{"status": 429,
#: "title": "Too Many Requests"}` and nothing else, and assuming a minute for it
#: cost fifteen of eighteen minutes of one run in cooldown on a single key with
#: nothing to rotate to.
BLIND_BACKOFF = 10.0


def fingerprint(key: str) -> str:
    """A stable, non-reversible label for logs and state files."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


@dataclass
class PooledKey:
    value: str
    rpm: int
    rpd: int
    #: Which service this key authenticates against. Keys from different
    #: providers sit in one pool for throughput, but the model that answers is
    #: recorded per trial, because a comparison that silently mixes models is
    #: measuring the models as much as the conditions.
    provider: str = "gemini"
    model: str = ""
    used_today: int = 0
    #: Monotonic time before which this key must not be used again.
    available_at: float = 0.0
    consecutive_429: int = 0

    @property
    def label(self) -> str:
        return fingerprint(self.value)

    @property
    def interval(self) -> float:
        return 60.0 / max(self.rpm, 1)

    @property
    def spent(self) -> bool:
        return self.used_today >= self.rpd

    def note_success(self) -> None:
        self.used_today += 1
        self.consecutive_429 = 0
        self.available_at = time.monotonic() + self.interval

    def note_rate_limited(self, retry_after: float | None) -> float:
        """Back this key off. Returns how long it will be unavailable.

        When the server says when to come back, that is the answer: Gemini's free
        tier is a 20-per-minute window and it names the seconds left in it, so the
        doubling backoff below was waiting sixty for a thirteen-second refusal and
        then a hundred and twenty, until every key was cooling and the trial ran
        out of attempts.

        Doubling is for the case where nothing was said: a key that keeps refusing
        without explanation is telling you something a fixed delay would ignore.
        """
        self.consecutive_429 += 1
        if retry_after is not None:
            wait = max(retry_after + 1.0, 2.0)
        else:
            wait = min(BLIND_BACKOFF * (2 ** (self.consecutive_429 - 1)), 900.0)
        self.available_at = time.monotonic() + wait
        return wait


@dataclass
class KeyPool:
    keys: list[PooledKey]
    state_path: pathlib.Path | None = None
    _today: str = ""
    verbose: bool = True
    rotations: int = 0
    waits: float = 0.0
    _order: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.keys:
            raise RuntimeError("no API keys supplied")
        self._today = time.strftime("%Y-%m-%d")
        self._load()

    # --- construction -------------------------------------------------------

    @classmethod
    def from_providers(cls, specs: list[tuple[str, tuple[str, ...], str, int, int]],
                       *, state_path: pathlib.Path | None = None) -> "KeyPool":
        """Build one pool spanning several providers.

        Each spec is (provider, env var names, model, rpm, rpd). Every key found
        becomes an independent quota, so three Gemini keys and one NVIDIA key is
        four quotas rotating rather than one queue.
        """
        keys: list[PooledKey] = []
        seen: set[str] = set()
        for provider, names, model, rpm, rpd in specs:
            for value in cls._collect(names):
                if value in seen:
                    continue
                seen.add(value)
                keys.append(PooledKey(value=value, rpm=rpm, rpd=rpd,
                                      provider=provider, model=model))
        if not keys:
            raise RuntimeError("no API keys found in the environment")
        return cls(keys=keys, state_path=state_path)

    @staticmethod
    def _collect(names: tuple[str, ...]) -> list[str]:
        found: list[str] = []
        candidates: list[str] = []
        for name in names:
            candidates += [name, f"{name}S", f"{name}_POOL"]
            candidates += [f"{name}_{i}" for i in range(2, 9)]
        for name in candidates:
            for part in os.environ.get(name, "").split(","):
                part = part.strip()
                if part and part not in found:
                    found.append(part)
        return found

    @classmethod
    def from_env(cls, names: tuple[str, ...], *, rpm: int = 10, rpd: int = 1200,
                 state_path: pathlib.Path | None = None) -> "KeyPool":
        """Collect keys from `NAME`, `NAME_2`, `NAME_3`, ... and `NAME` plural forms.

        Comma-separated values are split, so one variable can hold the whole pool.
        Duplicates are dropped: two names pointing at one key is one quota, and
        treating it as two would plan around throughput that does not exist.
        """
        found: list[str] = []
        candidates: list[str] = []
        for name in names:
            candidates += [name, f"{name}S", f"{name}_POOL"]
            candidates += [f"{name}_{i}" for i in range(2, 9)]
        for name in candidates:
            raw = os.environ.get(name, "")
            for part in raw.split(","):
                part = part.strip()
                if part and part not in found:
                    found.append(part)
        if not found:
            raise RuntimeError(f"set {names[0]} (or {names[0]}S, comma separated)")
        return cls(keys=[PooledKey(value=k, rpm=rpm, rpd=rpd) for k in found],
                   state_path=state_path)

    # --- persistence --------------------------------------------------------

    def _load(self) -> None:
        if not (self.state_path and self.state_path.exists()):
            return
        data = json.loads(self.state_path.read_text())
        if data.get("date") != self._today:
            return
        counts = data.get("used", {})
        for key in self.keys:
            key.used_today = int(counts.get(key.label, 0))

    def _save(self) -> None:
        if not self.state_path:
            return
        self.state_path.write_text(json.dumps({
            "date": self._today,
            "used": {k.label: k.used_today for k in self.keys},
        }, indent=2), encoding="utf-8")

    # --- acquiring ----------------------------------------------------------

    def _roll_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._today:
            self._today = today
            for key in self.keys:
                key.used_today = 0
            self._save()

    @property
    def remaining_today(self) -> int:
        return sum(max(k.rpd - k.used_today, 0) for k in self.keys)

    def acquire(self, provider: str | None = None) -> PooledKey:
        """Return a usable key, sleeping only as long as the soonest one needs.

        Picks the key that becomes available earliest rather than round-robin, so
        a key in cooldown is stepped over instead of waited on while another sits
        idle. That is the whole point of holding more than one.

        `provider` is not optional in practice. A key authenticates against one
        service, so handing a request a key from a different provider than the URL
        it is about to post to fails as an auth error and looks exactly like a
        dead key. Callers that know the endpoint must say so.
        """
        self._roll_day()
        live = [k for k in self.keys
                if not k.spent and (provider is None or k.provider == provider)]
        if not live:
            scope = f" for {provider}" if provider else ""
            raise AllKeysExhausted(
                f"all keys{scope} have spent their daily budget; "
                f"the run is checkpointed and resumes with the same command")

        now = time.monotonic()
        chosen = min(live, key=lambda k: k.available_at)
        wait = chosen.available_at - now
        if wait > 0:
            self.waits += wait
            time.sleep(wait)

        if self._order and self._order[-1] != chosen.label:
            self.rotations += 1
        self._order.append(chosen.label)
        return chosen

    def note_success(self, key: PooledKey) -> None:
        key.note_success()
        self._save()

    def note_rate_limited(self, key: PooledKey, retry_after: float | None) -> None:
        wait = key.note_rate_limited(retry_after)
        if self.verbose:
            others = sum(1 for k in self.keys
                         if k is not key and not k.spent
                         and k.available_at <= time.monotonic())
            print(f"    key {key.label} rate limited, cooling {wait:.0f}s; "
                  f"{others} other key(s) ready", flush=True)

    def summary(self) -> str:
        rows = [f"  {k.label}  {k.provider:<8} {k.used_today:>5}/{k.rpd} used today"
                + ("  (spent)" if k.spent else "") for k in self.keys]
        return (f"{len(self.keys)} keys, {self.remaining_today} requests left today, "
                f"{self.rotations} rotations, {self.waits:.0f}s spent waiting\n"
                + "\n".join(rows))


def preflight(pool: "KeyPool", probe) -> "KeyPool":
    """Drop keys that fail authentication, before the run starts.

    A dead key in the pool is worse than a missing one: it is selected, it fails,
    and the trial it was serving is recorded as an error that has nothing to do
    with the experiment. One cheap request per key at the start is the whole cost
    of never having to wonder whether a result was a model failure or an auth
    failure.

    `probe(key)` should return None when the key works and a reason when it does
    not. Transient failures are not disqualifying: only auth is.
    """
    live, dead = [], []
    for key in pool.keys:
        reason = probe(key)
        (live if reason is None else dead).append((key, reason))
    for key, reason in dead:
        print(f"  dropping key {key.label} ({key.provider}): {reason}")
    if not live:
        raise RuntimeError("no working keys: every one failed preflight")
    pool.keys = [k for k, _ in live]
    for key, _ in live:
        print(f"  key {key.label} ({key.provider}) ready")
    return pool
