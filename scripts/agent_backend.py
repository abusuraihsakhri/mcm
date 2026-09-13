"""An OpenAI-compatible agent backend with rate limiting, for the tool experiment.

Gemini, NVIDIA NIM, Groq, Cerebras and GitHub Models all expose an
OpenAI-compatible chat-completions endpoint, so one client covers every free tier
worth using and the experiment can be replicated across providers. An effect that
shows up on three different models is about the information; an effect on one
could be about that model.

**The free tiers are the binding constraint, so the limiter is not optional.**
Gemini's free tier allows roughly 15 requests per minute and a bounded number per
day, and one agent trial is several requests because every tool call is another
round trip. A naive loop hits 429 within a minute, and retrying a 429 by
hammering it is how a free key gets throttled harder. So:

- a minimum interval between requests, set from a requests-per-minute budget
- a daily counter that persists across runs, because the day limit outlives the
  process
- exponential backoff that honours `Retry-After` when the server sends it
- checkpointing after every trial, so hitting the daily limit pauses the
  experiment instead of losing it

Set the key in the environment. Never in a file, never in an argument, because
arguments end up in shell history and process listings.

    $env:GEMINI_API_KEY = "..."
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import time
from dataclasses import dataclass, field

import httpx

# Machines running TLS-inspecting antivirus present a locally-issued certificate
# that certifi's bundle does not contain, so every HTTPS call fails verification.
# truststore reads the operating system's certificate store, where that CA is
# installed, which fixes it without the obvious and much worse alternative of
# turning verification off.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # not installed: certifi's bundle is used as normal
    pass

#: provider -> (base url, default model, env var holding the key)
PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai",
               "gemini-2.0-flash", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "nvidia": ("https://integrate.api.nvidia.com/v1",
               "meta/llama-3.3-70b-instruct", ("NVIDIA_API_KEY",)),
    "groq": ("https://api.groq.com/openai/v1",
             "llama-3.3-70b-versatile", ("GROQ_API_KEY",)),
    "github": ("https://models.inference.ai.azure.com",
               "gpt-4o-mini", ("GITHUB_TOKEN",)),
}


def api_key(provider: str) -> str:
    """The first key for this provider. Single-shot checks only; the agent loop
    uses the pool so that quotas rotate."""
    for name in PROVIDERS[provider][2]:
        value = os.environ.get(name)
        if value:
            return value.split(",")[0].strip()
    names = " or ".join(PROVIDERS[provider][2])
    raise RuntimeError(f"set {names} in the environment")


# --- rate limiting is delegated to the key pool ------------------------------
#
# One key is just a pool of one, so there is no separate single-key path to keep
# in step. See key_pool.py for the quota, cooldown and persistence rules.

from key_pool import AllKeysExhausted, KeyPool, PooledKey  # noqa: E402


def build_pool(provider: str, *, rpm: int = 10, rpd: int = 1200,
               state_path: pathlib.Path | None = None) -> KeyPool:
    return KeyPool.from_env(PROVIDERS[provider][2], rpm=rpm, rpd=rpd,
                            state_path=state_path)


# --- tools the agent may call ----------------------------------------------

TOOL_SCHEMAS = {
    "list_files": {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List Python files in the repository, optionally under a subdirectory.",
            "parameters": {"type": "object", "properties": {
                "subdirectory": {"type": "string"}}},
        },
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository by its path.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}}, "required": ["path"]},
        },
    },
    "grep": {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search the repository for a regular expression. "
                           "Returns matching lines with their file and line number.",
            "parameters": {"type": "object", "properties": {
                "pattern": {"type": "string"}}, "required": ["pattern"]},
        },
    },
    "mcm_impact": {
        "type": "function",
        "function": {
            "name": "mcm_impact",
            "description": "Given a definition, report which other definitions "
                           "must be updated if its signature changes, derived from "
                           "the call graph rather than from text matching.",
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string"}}, "required": ["target"]},
        },
    },
}

#: The experiment's independent variable. grep is in both on purpose: the
#: comparison worth making is against an agent doing what agents do today.
TOOLS_CONTROL = ["read_file", "list_files", "grep"]
TOOLS_TREATMENT = TOOLS_CONTROL + ["mcm_impact"]

MAX_TOOL_OUTPUT = 4000


@dataclass
class ToolBox:
    """Executes the agent's tool calls against one repository and MCM index."""

    repo: pathlib.Path
    db: pathlib.Path
    calls: int = 0
    used: set[str] = field(default_factory=set)
    read_already: set[str] = field(default_factory=set)
    #: (tool, output) in order, so a final answer can be asked for outside the
    #: tool-calling conversation without losing what was found.
    transcript: list = field(default_factory=list)

    def run(self, name: str, args: dict) -> str:
        self.calls += 1
        self.used.add(name)
        try:
            handler = getattr(self, f"_{name}")
        except AttributeError:
            return f"no such tool: {name}"
        try:
            output = handler(args)[:MAX_TOOL_OUTPUT]
        except Exception as exc:  # noqa: BLE001 - the agent should see the failure
            output = f"{type(exc).__name__}: {exc}"
        self.transcript.append((name, output))
        return output

    def _list_files(self, args: dict) -> str:
        root = self.repo / args.get("subdirectory", "")
        files = [str(p.relative_to(self.repo).as_posix())
                 for p in root.rglob("*.py")
                 if ".git" not in p.parts][:200]
        return "\n".join(files) or "no Python files there"

    def _read_file(self, args: dict) -> str:
        requested = args["path"]
        if requested in self.read_already:
            return (f"You already read {requested} earlier in this conversation; "
                    f"scroll up rather than spending another call on it.")
        self.read_already.add(requested)
        path = (self.repo / requested).resolve()
        if self.repo.resolve() not in path.parents and path != self.repo.resolve():
            return "refused: path escapes the repository"
        if not path.exists():
            return f"no such file: {args['path']}"
        return path.read_text(encoding="utf-8", errors="replace")

    def _grep(self, args: dict) -> str:
        try:
            # Decode as UTF-8 and replace what will not decode, rather than
            # `text=True`, which uses the machine's locale codec: on Windows that
            # is cp1252, and one byte of a binary test fixture in the search
            # output kills the reader thread. The control condition is the one
            # that depends on grep, so a crash here would quietly handicap the
            # arm this experiment is comparing against.
            done = subprocess.run(
                ["git", "grep", "-n", "-E", args["pattern"], "--", "*.py"],
                cwd=self.repo, capture_output=True, timeout=30,
                encoding="utf-8", errors="replace")
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"grep failed: {exc}"
        return done.stdout[:MAX_TOOL_OUTPUT] or "no matches"

    def _mcm_impact(self, args: dict) -> str:
        from mcm.core.change import Change, ChangeKind
        from mcm.reasoning.change_propagation import propagate
        from mcm.retrieval.symbolic import resolve_one
        from mcm.storage.sqlite_store import SQLiteStore

        store = SQLiteStore(self.db)
        try:
            obj = resolve_one(store, args["target"])
            result = propagate(store, Change(obj.id, ChangeKind.SIGNATURE),
                               max_depth=6)
            must = [f"  {p.object.id}\n"
                    f"      confidence {p.confidence:.2f} ({p.band}), "
                    f"{'direct caller' if p.is_direct else 'indirect'}\n"
                    f"      because: {p.why}"
                    for p in result.must_update]
            may = [f"  {p.object.id}  (confidence {p.confidence:.2f})"
                   for p in result.may_differ[:10]]
            out = ["MUST UPDATE - these will break unless edited:"]
            out += must or ["  (none found)"]
            if may:
                out += ["", "MAY DIFFER - behaviour may change, check but no edit forced:"]
                out += may
            return "\n".join(out)
        except KeyError as exc:
            return f"not in the index: {exc}"
        finally:
            store.close()


# --- the agent loop ---------------------------------------------------------

SYSTEM = (
    "You are repairing a Python repository after a function's signature changed. "
    "Find every definition that must now be updated. Use the tools; do not guess. "
    "When finished, list each definition to edit on its own line as:\n"
    "EDIT: <file path>::<definition name>\n"
    "List nothing you have not verified."
)


def run_agent(*, prompt: str, tools: list[str], toolbox: ToolBox,
              pool: KeyPool, provider: str, model: str,
              max_steps: int = 8, timeout: float = 120.0) -> tuple[str, int, str | None]:
    """Drive one trial. Returns (final text, tool calls made, error).

    The key is chosen per request rather than per trial, so a trial that starts
    on one key and hits a limit mid-way continues on another instead of failing.
    """
    base, default_model, _ = PROVIDERS[provider]
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
    schemas = [TOOL_SCHEMAS[t] for t in tools]

    with httpx.Client(timeout=timeout) as client:
        for _ in range(max_steps):
            payload = {"model": model or default_model, "messages": messages,
                       "tools": schemas, "temperature": 0.0}
            data = _post(client, f"{base}/chat/completions", payload, pool, provider)
            if isinstance(data, str):
                return "", toolbox.calls, data

            choice = data["choices"][0]["message"]
            messages.append(choice)
            calls = choice.get("tool_calls") or []
            if not calls:
                return message_text(choice), toolbox.calls, None

            for call in calls:
                fn = call["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": toolbox.run(fn["name"], args)})

        # Out of steps. Asking for the answer inside the same conversation does
        # not work: both providers stay in tool-calling mode once the history
        # contains tool calls, returning finish_reason="tool_calls" and empty
        # content no matter what tool_choice says. So the final answer is asked
        # in a *fresh* conversation containing a plain-text digest of what the
        # tools returned, where there is no tool state to be sticky about.
        digest = (chr(10) * 2).join(
            f"{name} returned:" + chr(10) + output[:1200]
            for name, output in toolbox.transcript[-6:]) or "(no findings)"
        target_line = prompt.splitlines()[0] if prompt else "the target"
        closing = [
            {"role": "system", "content":
             "Answer only in lines of the form 'EDIT: <file path>::<definition>'. "
             "No prose, no explanation. If nothing needs updating, write "
             "'EDIT: none'."},
            {"role": "user", "content":
             target_line + chr(10) * 2
             + "Findings from investigating the repository:" + chr(10) * 2
             + digest + chr(10) * 2
             + "Which definitions must be updated? EDIT: lines only."},
        ]
        data = _post(client, f"{base}/chat/completions",
                     {"model": model or default_model, "messages": closing,
                      "temperature": 0.0}, pool, provider)
        if isinstance(data, str):
            return "", toolbox.calls, data
        return message_text(data["choices"][0]["message"]), toolbox.calls, None


#: Gemini states its delay in the message body rather than in a header:
#: "Please retry in 13.274118104s". Missing it costs the difference between
#: waiting the thirteen seconds the server asked for and the doubling backoff the
#: pool falls back to, which on a 20-per-minute limit is the difference between a
#: run that proceeds and one that exhausts its retries.
RETRY_IN = re.compile(r"retry in ([0-9.]+)s", re.I)


def retry_delay(response: httpx.Response) -> float | None:
    """How long the server asked us to wait, from the header or the body."""
    header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    found = RETRY_IN.search(response.text or "")
    return float(found.group(1)) if found else None


def _post(client: httpx.Client, url: str, payload: dict, pool: KeyPool,
          provider: str, attempts: int = 6):
    """POST, rotating keys rather than retrying through a rate limit.

    A 429 cools the key that produced it and the next attempt takes whichever key
    the pool says is ready, which is usually a different one. Retrying the same
    key harder is how a free tier becomes a blocked one.
    """
    delay = 2.0
    for attempt in range(attempts):
        # Same provider as the URL: a key is only valid against its own service.
        key = pool.acquire(provider)
        headers = {"Authorization": f"Bearer {key.value}",
                   "Content-Type": "application/json"}
        try:
            response = client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            if attempt == attempts - 1:
                return f"transport: {exc}"
            time.sleep(delay)
            delay *= 2
            continue

        if response.status_code == 200:
            pool.note_success(key)
            return response.json()

        if response.status_code == 429:
            pool.note_rate_limited(key, retry_delay(response))
            continue                       # straight to another key, no sleep here

        if response.status_code in (500, 502, 503, 504):
            pool.note_success(key)         # the key is fine; the service is not
            if attempt == attempts - 1:
                return f"http {response.status_code} after {attempts} attempts"
            time.sleep(delay)
            delay *= 2
            continue

        pool.note_success(key)
        return f"http {response.status_code}: {response.text[:200]}"
    return "exhausted retries across every key"


def message_text(message: dict) -> str:
    """The model's answer, wherever this model happens to put it.

    Reasoning models return an empty `content` and place their output in
    `reasoning_content`. Reading only `content` scores every one of their answers
    as silence, which looks exactly like a model that could not do the task.
    """
    for field_name in ("content", "reasoning_content"):
        value = message.get(field_name)
        if value and value.strip():
            return value
    return ""


EDIT_LINE = re.compile(r"^\s*EDIT:\s*(.+?)\s*$", re.M)


def parse_edits(text: str) -> list[str]:
    return sorted({m.strip() for m in EDIT_LINE.findall(text or "") if m.strip()})
