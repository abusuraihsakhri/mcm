"""Run the benchmark across a set of repositories and collect the results.

Section 49 asks for four size tiers, and the evaluation reported two repositories
of 16 and 83 files for a long time because anything larger was too slow to
attempt. It no longer is, so the constraint on breadth is now patience rather
than throughput.

Breadth matters here more than it would for a fixed-dataset benchmark, because
the section 3 verdict has already moved with the repository: `graph > vector-rag`
failed on itsdangerous and flask, held on django under feature hashing, and
failed again on django under a trained embedding. Three points cannot
characterise that. The suite exists to say how often each link holds rather than
whether it held once.

The set below spans size, ecosystem and domain, since all three plausibly change
the answer. `httpx` stays out of it: it is the tuning repository and scoring it
would report the quality of the fit.

    python scripts/evaluation_suite.py --clone --out results/
    python scripts/evaluation_suite.py --out results/ --provider sentence-transformers:all-MiniLM-L6-v2@cuda
    python scripts/evaluation_suite.py --out results/ --summarise
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Repo:
    name: str
    url: str
    #: Commits back from HEAD to split history at. Smaller repositories need a
    #: deeper horizon to yield enough tasks; larger ones yield plenty from fewer.
    horizon: int
    tier: str
    ecosystem: str
    note: str


#: Scored repositories. Chosen before any of them was run, and not revised
#: afterwards: picking the set once the scores are visible is how a benchmark
#: quietly becomes a demonstration.
SUITE: tuple[Repo, ...] = (
    Repo("itsdangerous", "https://github.com/pallets/itsdangerous", 250,
         "tiny", "pallets", "signing primitives, 16 files"),
    Repo("click", "https://github.com/pallets/click", 400,
         "small", "pallets", "CLI framework, argument parsing"),
    Repo("flask", "https://github.com/pallets/flask", 400,
         "small", "pallets", "web microframework"),
    Repo("python-dotenv", "https://github.com/theskumar/python-dotenv", 250,
         "tiny", "independent", "config loading, very small surface"),
    Repo("requests", "https://github.com/psf/requests", 400,
         "small", "psf", "HTTP client, heavy public API"),
    Repo("rich", "https://github.com/Textualize/rich", 400,
         "medium", "textualize", "terminal rendering, deep class hierarchy"),
    Repo("pydantic", "https://github.com/pydantic/pydantic", 400,
         "medium", "independent", "validation, heavy metaprogramming"),
    Repo("scrapy", "https://github.com/scrapy/scrapy", 400,
         "medium", "independent", "crawler, plugin and middleware graph"),
    Repo("sqlalchemy", "https://github.com/sqlalchemy/sqlalchemy", 400,
         "large", "independent", "ORM, very deep inheritance"),
    Repo("django", "https://github.com/django/django", 400,
         "large", "django", "web framework, 2932 files"),
)

#: Fitted on, never scored. Kept here so the split is visible in one place.
TUNING = Repo("httpx", "https://github.com/encode/httpx", 400,
              "small", "encode", "tuning only, never reported")


def clone(repo: Repo, into: Path) -> Path:
    """Clone at a depth that covers the horizon plus the tasks drawn after it.

    Deliberately *not* a blob-filtered clone. `--filter=blob:none` makes the
    clone fast and then makes every file read a network round trip, which turns
    a benchmark into a download. This costs more disk and finishes sooner.
    """
    target = into / repo.name
    if target.exists():
        print(f"  {repo.name}: already present")
        return target
    depth = repo.horizon + 400
    print(f"  {repo.name}: cloning depth {depth} ...", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", str(depth), repo.url, str(target)],
        check=True, capture_output=True)
    return target


def run_one(repo: Repo, path: Path, out: Path, provider: str | None,
            max_tasks: int) -> dict | None:
    json_path = out / f"{repo.name}.json"
    log_path = out / f"{repo.name}.log"
    if json_path.exists():
        print(f"  {repo.name}: already scored, skipping")
        return json.loads(json_path.read_text())

    cmd = [sys.executable, "-m", "mcm.cli", "benchmark", str(path),
           "--name", repo.name, "--horizon", str(repo.horizon),
           "--max-tasks", str(max_tasks), "--json", str(json_path)]
    if provider:
        cmd += ["--provider", provider]
    # A large repository's graph does not fit in RAM alongside a GPU embedding
    # model on a 16GB machine. Paying disk for it is what makes the tier run.
    if repo.tier in ("medium", "large"):
        cmd.append("--on-disk")

    print(f"  {repo.name}: running ...", flush=True)
    started = time.time()
    completed = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started
    log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")

    if completed.returncode != 0 or not json_path.exists():
        print(f"  {repo.name}: FAILED in {elapsed:.0f}s, see {log_path.name}")
        return None
    print(f"  {repo.name}: done in {elapsed:.0f}s")
    return json.loads(json_path.read_text())


def verdicts(doc: dict) -> dict[str, str]:
    out = {}
    for link in doc.get("chain", []):
        out[f"{link['left']}>{link['right']}"] = link["verdict"]
    for link in doc.get("mcm_vs_baselines", []):
        out[f"mcm>{link['baseline']}"] = link["verdict"]
    return out


def summarise(out: Path) -> None:
    """Report how often each link holds, which is the point of running a set."""
    docs = {}
    for repo in SUITE:
        f = out / f"{repo.name}.json"
        if f.exists():
            docs[repo.name] = json.loads(f.read_text())
    if not docs:
        print("no results yet")
        return

    print(f"\n{'repository':<16}{'tier':<8}{'tasks':>6}{'mcm eff':>10}"
          f"{'best baseline':>22}{'margin':>9}")
    for repo in SUITE:
        d = docs.get(repo.name)
        if not d:
            continue
        overall = d["overall"]
        mcm = overall["mcm"]["efficiency"]
        score, name = max((v["efficiency"], k) for k, v in overall.items() if k != "mcm")
        print(f"{repo.name:<16}{repo.tier:<8}{d['tasks']['generated']:>6}{mcm:>10.3f}"
              f"{name + ' ' + format(score, '.3f'):>22}{mcm - score:>+9.3f}")

    print("\nHow often each link holds, across the repositories scored so far")
    links = ["mcm>vector-rag", "mcm>graph", "mcm>hybrid", "graph>vector-rag"]
    tally: dict[str, dict[str, int]] = {k: {} for k in links}
    for name, d in docs.items():
        v = verdicts(d)
        for link in links:
            verdict = v.get(link, "not reported")
            tally[link][verdict] = tally[link].get(verdict, 0) + 1
    for link in links:
        total = sum(tally[link].values())
        parts = ", ".join(f"{v} {k}" for k, v in sorted(tally[link].items()))
        print(f"  {link:<18} n={total}  {parts}")

    print("\n  A link that holds on some repositories and not others is a result")
    print("  about when it holds, not a result that it does.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--repos", type=Path, default=Path("repos"),
                        help="where clones live")
    parser.add_argument("--clone", action="store_true", help="clone, do not run")
    parser.add_argument("--provider", default=None,
                        help="embedding provider every system shares")
    parser.add_argument("--max-tasks", type=int, default=120)
    parser.add_argument("--only", nargs="*", help="restrict to these repositories")
    parser.add_argument("--summarise", action="store_true",
                        help="report what is already in --out and exit")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    args.repos.mkdir(parents=True, exist_ok=True)

    if args.summarise:
        summarise(args.out)
        return 0

    selected = [r for r in SUITE if not args.only or r.name in args.only]

    print(f"Cloning {len(selected)} repositories into {args.repos}")
    paths = {}
    for repo in selected:
        try:
            paths[repo.name] = clone(repo, args.repos)
        except subprocess.CalledProcessError as exc:
            print(f"  {repo.name}: clone failed: "
                  f"{exc.stderr.decode('utf-8', 'replace')[:200]}")
    if args.clone:
        return 0

    print(f"\nScoring, provider={args.provider or 'default feature hashing'}")
    for repo in selected:
        if repo.name in paths:
            run_one(repo, paths[repo.name], args.out, args.provider, args.max_tasks)

    summarise(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
