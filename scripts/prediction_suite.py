"""Measure impact prediction against real Git history, pooled across repositories.

`mcm accuracy` answers this for one repository and says, correctly, that one
repository cannot separate a good propagation model from a lucky one. This
replays several and pools them, because the quantity that matters is rare.

**Why pooling is necessary.** On real history roughly 93% of changes to an
existing definition are behaviour changes, which propagate as MAY_DIFFER. A
commit diff records edits, not behaviour, so those predictions are neither
confirmed nor refuted by history: they are unfalsifiable *by this ground truth*,
not unfalsifiable in principle. Only SIGNATURE and REMOVE changes produce
MUST_UPDATE, the prediction a diff can actually check, and they are about 7% of
changes. A 161-observation repository therefore yields around a dozen testable
cases, which is nothing.

**Why co-change is a weak ground truth, stated up front.** Commits here touch a
mean of 33 definitions. Most of that is unrelated work batched together, so a
low "reached" figure mostly measures how much unrelated code ships in the same
commit. The number to read is *precision on MUST_UPDATE*: when the model says an
edit is required, was one made. Recall against co-change is not a fair target and
is reported only because hiding it would be worse.

    python scripts/prediction_suite.py --repos repos/ --out prediction/
    python scripts/prediction_suite.py --out prediction/ --summarise
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

from mcm.core.change import ChangeKind
from mcm.ingestion.history import HistoryIngestor
from mcm.reasoning.prediction import evaluate, observations_from_history
from mcm.storage.sqlite_store import SQLiteStore

#: Replaying history costs one full ingest per commit, so this is ordered by
#: what is affordable rather than by what is most interesting.
REPOS = [
    ("itsdangerous", 200), ("python-dotenv", 200), ("requests", 150),
    ("flask", 150), ("click", 150), ("rich", 100),
]


def replay(name: str, path: pathlib.Path, db: pathlib.Path, commits: int) -> bool:
    if db.exists():
        print(f"  {name}: already replayed")
        return True
    store = SQLiteStore(db)
    try:
        started = time.time()
        HistoryIngestor(store).ingest(path, name=name, max_commits=commits)
        print(f"  {name}: {commits} commits in {time.time() - started:.0f}s")
        return True
    except Exception as exc:  # noqa: BLE001 - one repository failing is not fatal
        print(f"  {name}: FAILED {type(exc).__name__}: {exc}")
        db.unlink(missing_ok=True)
        return False
    finally:
        store.close()


def measure(name: str, db: pathlib.Path) -> dict | None:
    store = SQLiteStore(db)
    try:
        observations = observations_from_history(store)
        if not observations:
            return None
        report = evaluate(store)
        kinds = collections.Counter(o.kind.value for o in observations)

        # Testable means the model claimed an edit was required. Everything else
        # is a behaviour prediction a diff cannot settle.
        predicted = confirmed = 0
        for tally in report.by_edge.values():
            predicted += tally.predicted_edits
            confirmed += tally.confirmed_edits

        return {
            "repository": name,
            "observations": len(observations),
            "kinds": dict(kinds),
            "testable_kinds": kinds.get("SIGNATURE", 0) + kinds.get("REMOVE", 0),
            "predicted_edits": predicted,
            "confirmed_edits": confirmed,
            "mean_co_changed": round(
                sum(len(o.co_changed) for o in observations) / len(observations), 1),
            "by_edge": {k.value: {"predicted": v.predicted_edits,
                                  "confirmed": v.confirmed_edits}
                        for k, v in report.by_edge.items() if v.predicted_edits},
        }
    finally:
        store.close()


def summarise(out: pathlib.Path) -> None:
    rows = [json.loads(f.read_text()) for f in sorted(out.glob("*.json"))]
    if not rows:
        print("no results yet")
        return

    print(f"\n{'repository':<16}{'obs':>6}{'behaviour':>11}{'testable':>10}"
          f"{'predicted':>11}{'confirmed':>11}{'precision':>11}")
    for r in rows:
        b = r["kinds"].get("BEHAVIOUR", 0)
        p, c = r["predicted_edits"], r["confirmed_edits"]
        prec = f"{c / p:.2f}" if p else "n/a"
        print(f"{r['repository']:<16}{r['observations']:>6}{b:>11}"
              f"{r['testable_kinds']:>10}{p:>11}{c:>11}{prec:>11}")

    obs = sum(r["observations"] for r in rows)
    beh = sum(r["kinds"].get("BEHAVIOUR", 0) for r in rows)
    testable = sum(r["testable_kinds"] for r in rows)
    predicted = sum(r["predicted_edits"] for r in rows)
    confirmed = sum(r["confirmed_edits"] for r in rows)

    print(f"\n{'POOLED':<16}{obs:>6}{beh:>11}{testable:>10}"
          f"{predicted:>11}{confirmed:>11}"
          f"{(f'{confirmed / predicted:.2f}' if predicted else 'n/a'):>11}")

    print(f"\n  {beh / obs * 100:.0f}% of changes to existing definitions are behaviour")
    print(f"  changes, which propagate as MAY_DIFFER and which a commit diff cannot")
    print(f"  confirm or refute. Only {testable} observations ({testable / obs * 100:.0f}%)"
          f" are the kind that")
    print(f"  produces a checkable MUST_UPDATE claim.")

    if predicted:
        print(f"\n  Precision on those: when the model said an edit was required,")
        print(f"  one was made {confirmed / predicted * 100:.0f}% of the time"
              f" ({confirmed}/{predicted}).")
    else:
        print(f"\n  The model made no MUST_UPDATE predictions at all across these")
        print(f"  repositories, so its checkable claim is untested rather than wrong.")

    per_edge: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        for edge, v in r["by_edge"].items():
            per_edge[edge][0] += v["predicted"]
            per_edge[edge][1] += v["confirmed"]
    if per_edge:
        print("\n  By edge type:")
        for edge, (p, c) in sorted(per_edge.items(), key=lambda kv: -kv[1][0]):
            print(f"    {edge:<14} {c:>4}/{p:<4} confirmed  ({c / p:.2f})")

    print("\n  Recall against co-change is not reported as a headline: commits here")
    print("  touch a mean of", round(sum(r["mean_co_changed"] for r in rows) / len(rows), 1),
          "definitions, most of them unrelated work in the")
    print("  same commit, so recall would mostly measure commit hygiene.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repos", type=pathlib.Path, default=pathlib.Path("repos"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("prediction"))
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--summarise", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.summarise:
        summarise(args.out)
        return 0

    for name, commits in REPOS:
        if args.only and name not in args.only:
            continue
        path = args.repos / name
        if not path.exists():
            print(f"  {name}: not cloned")
            continue
        result_path = args.out / f"{name}.json"
        if result_path.exists():
            print(f"  {name}: already measured")
            continue
        db = args.out / f"{name}.db"
        if not replay(name, path, db, commits):
            continue
        result = measure(name, db)
        if result:
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"    {result['testable_kinds']} testable of "
                  f"{result['observations']} observations, "
                  f"{result['predicted_edits']} edit predictions")

    summarise(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
