"""Does `mcm_impact` change what an agent edits, when the tests decide the answer?

`agent_experiment.py` asks the same question against co-change ground truth, and
the ceiling measured there says why that cannot answer it: the definitions
`mcm_impact` names cover a mean of 17% of what a commit edits, because commits
touch things that are not downstream of anything. Scoring against all of them
charges propagation for being right.

Here the ground truth comes from `oracle_tasks.py`, where a file counts as
required only because removing it made the suite fail and restoring it made the
suite pass. Nothing in the answer is there by association.

## What the agent sees

Not the repository at HEAD. Each task is materialised as its own worktree at the
parent commit with the target file alone advanced to the child, which is the
broken state the task describes -- callers stale, suite red. The MCM index is
built from that same worktree, so the tool and the agent are looking at one
revision. Pointing either at HEAD would let the treatment answer from code the
control cannot see.

## The conditions

Identical model, prompt, temperature and task; the tool list is the only
difference.

    control     read_file, list_files, grep
    treatment   read_file, list_files, grep, mcm_impact

## What is measured

Recall and precision over *files*, because files are the granularity the tests
adjudicated. The target file is excluded from both sides: it is given in the
prompt, and counting it would hand every agent a free hit.

Tasks are split by whether `mcm_impact` names any required file at all. The
split is computed before the run and reported separately, so the treatment's
gain where it has something to say is not averaged away by the tasks where it
has nothing -- and so the frequency of having nothing to say is visible rather
than hidden.

    python scripts/oracle_experiment.py --prepare
    python scripts/oracle_experiment.py --run --trials 3
    python scripts/oracle_experiment.py --report
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from agent_backend import (PROVIDERS, TOOLS_CONTROL, TOOLS_TREATMENT, ToolBox,
                           parse_edits, run_agent)
from key_pool import AllKeysExhausted, KeyPool, preflight

SEED = 20260913


@dataclass
class Task:
    task_id: str
    repository: str
    commit: str
    parent: str
    target_file: str
    #: Definitions inside the target file the commit rewrote, named in the
    #: prompt. Empty when the commit changed module-level code instead, which is
    #: a real kind of breaking change and not a reason to discard the task.
    changed_defs: list[str]
    #: Files the test suite proved were required. The answer.
    truth: list[str]
    #: Files `mcm_impact` names for the changed definitions, resolved offline.
    mcm_files: list[str] = field(default_factory=list)
    #: True when at least one of those is a file the tests proved required.
    mcm_informative: bool = False
    #: Which way the dependency runs between the target and what the tests
    #: required: "downstream" when a required file depends on the target, which
    #: is the case impact analysis exists to predict; "upstream" when the target
    #: depends on the required file, which is a co-requirement and not a
    #: consequence; "none" when the index records no edge either way.
    direction: str = "none"
    worktree: str = ""
    index: str = ""


@dataclass
class Trial:
    task_id: str
    condition: str
    trial: int
    edited: list[str] = field(default_factory=list)
    tool_calls: int = 0
    used_mcm: bool = False
    error: str | None = None
    model: str = ""
    tools_used: list[str] = field(default_factory=list)


# --- comparing file references ----------------------------------------------

def normalise(reference: str) -> str:
    """Reduce a file reference to a comparable path.

    The agent writes ``src/click/core.py``, ``./src/click/core.py`` or
    ``src/click/core.py::Group.invoke`` for what ground truth calls
    ``src/click/core.py``. Everything after ``::`` is a definition name, which
    this experiment does not score, and a leading ``./`` or backslash is a
    spelling difference rather than a different file.
    """
    text = (reference or "").strip().strip("`\"'*- ")
    text = text.partition("::")[0].partition("#")[0].strip()
    text = text.replace("\\", "/").lstrip("./")
    return text.lower()


def same_file(left: str, right: str) -> bool:
    """True when two normalised paths name the same file.

    Suffix matching, not equality, because an agent that answers ``click/core.py``
    for ``src/click/core.py`` has identified the file. Requiring a directory
    boundary stops ``core.py`` from matching ``mycore.py``.
    """
    if not left or not right:
        return False
    if left == right:
        return True
    longer, shorter = (left, right) if len(left) > len(right) else (right, left)
    return longer.endswith("/" + shorter)


def score(edited: list[str], truth: list[str]) -> tuple[float, float]:
    claimed = [c for c in {normalise(e) for e in edited} if c]
    wanted = [normalise(t) for t in truth]
    if not wanted:
        return (0.0, 0.0)
    hits = [w for w in wanted if any(same_file(w, c) for c in claimed)]
    recall = len(hits) / len(wanted)
    if not claimed:
        return (recall, 0.0)
    right = [c for c in claimed if any(same_file(w, c) for w in wanted)]
    return (recall, len(right) / len(claimed))


# --- task preparation -------------------------------------------------------

def probe_ids(store, repository: str, target_file: str,
              changed_defs: list[str]) -> list[str]:
    """The object ids to propagate from, built rather than searched for.

    `resolve_one` matches on a bare name, which is ambiguous exactly where these
    tasks live: `__init__.py` matches the package and the tests package,
    `__getattr__` matches every class that defines one, and an ambiguous name
    raises rather than guessing. Since the task already says which file changed,
    the id can be constructed and checked instead.

    A file with no changed definition is itself the subject: MCM indexes files as
    objects, and propagating from one follows IMPORTS, which is the dependency a
    module-level change breaks.
    """
    base = f"repo://{repository}/{target_file}"
    ids = [oid for name in changed_defs
           for kind in ("function", "method", "class")
           if store.get_object(oid := f"{base}#{kind}:{name}") is not None]
    if not ids and store.get_object(f"{base}#file") is not None:
        ids = [f"{base}#file"]
    return ids


def mcm_reach(index: pathlib.Path, repository: str, target_file: str,
              changed_defs: list[str]) -> list[str]:
    """Files `mcm_impact` would name for this task's change."""
    from mcm.core.change import Change, ChangeKind
    from mcm.reasoning.change_propagation import propagate
    from mcm.storage.sqlite_store import SQLiteStore

    store = SQLiteStore(index)
    files: set[str] = set()
    try:
        for object_id in probe_ids(store, repository, target_file, changed_defs):
            try:
                result = propagate(store, Change(object_id, ChangeKind.SIGNATURE),
                                   max_depth=6)
            except (KeyError, ValueError):
                continue
            # Both bands, because both are what the tool prints. Counting only
            # MUST_UPDATE understated what the agent is shown: an importer of a
            # changed module lands in MAY_DIFFER, since importing something is
            # not by itself a reason to edit, and dropping it scored the tool as
            # silent on tasks where it had named the right file.
            for reached in list(result.must_update) + list(result.may_differ[:10]):
                path = reached.object.id.partition("#")[0]
                files.add(path.split("//", 1)[-1].partition("/")[2])
    finally:
        store.close()
    return sorted(files)


def portable(path: pathlib.Path) -> str:
    """A path to record in `tasks.json`, relative to where the run was started.

    `materialise` works in absolute paths because `git worktree add` resolves a
    relative one against the clone rather than against the caller. What gets
    written down should not be absolute, though: the file is committed, and an
    absolute path both leaks the operator's home directory and makes the record
    useless on any other machine. Anything outside the working directory is kept
    as-is, since there is nothing better to say about it.
    """
    try:
        return path.relative_to(pathlib.Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def dependency_direction(index: pathlib.Path, repository: str,
                         target_file: str, truth: list[str]) -> str:
    """Which way the edges run between the target file and the required files.

    Worth recording because it bounds what the treatment could possibly do. A
    file the tests proved necessary is not always a file that broke: a commit
    that changes `main.py` to call a new helper makes `compat.py` required, and
    the suite cannot tell that apart from a caller that went stale. Propagation
    predicts consequences, so scoring it on co-requirements measures the wrong
    thing -- and the split says how much of the task set is which.
    """
    from mcm.storage.sqlite_store import SQLiteStore

    def in_file(object_id: str, relpath: str) -> bool:
        return object_id.partition("#")[0].endswith("/" + relpath)

    store = SQLiteStore(index)
    downstream = upstream = False
    try:
        for relation in store.all_relations():
            if relation.relation_type not in ("CALLS", "IMPORTS", "USES", "INHERITS"):
                continue
            subject, obj = relation.arguments[0], relation.arguments[1]
            for required in truth:
                if in_file(subject, required) and in_file(obj, target_file):
                    downstream = True
                if in_file(subject, target_file) and in_file(obj, required):
                    upstream = True
    finally:
        store.close()
    if downstream:
        return "downstream"
    return "upstream" if upstream else "none"


def materialise(repos: pathlib.Path, raw: dict, into: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Build the broken worktree for one task, and index that worktree.

    The worktree is the parent commit with the target file alone advanced, which
    is the state the prompt describes. The index is built from it rather than
    from history, so `mcm_impact` answers about the code in front of the agent.
    """
    from mcm.ingestion.repository import RepositoryIngestor
    from mcm.storage.sqlite_store import SQLiteStore

    # Absolute, because `git worktree add` runs with cwd set to the clone and
    # would otherwise create the worktree *inside* it, at a path this process
    # then cannot find.
    root = (repos / raw["repository"]).resolve()
    stem = raw["task_id"].replace(":", "-")
    worktree = (into / "worktrees" / stem).resolve()
    index = (into / "index" / f"{stem}.db").resolve()
    index.parent.mkdir(parents=True, exist_ok=True)
    worktree.parent.mkdir(parents=True, exist_ok=True)

    if not worktree.exists():
        subprocess.run(["git", "worktree", "add", "--detach", "--force",
                        str(worktree), raw["parent"]],
                       cwd=root, capture_output=True, encoding="utf-8", errors="replace", check=True)
        subprocess.run(["git", "checkout", raw["commit"], "--", raw["target_file"]],
                       cwd=worktree, capture_output=True, encoding="utf-8", errors="replace", check=True)

    if not index.exists():
        store = SQLiteStore(index)
        try:
            RepositoryIngestor(store).ingest(worktree, name=raw["repository"])
        finally:
            store.close()
    return worktree, index


def prepare(oracle: pathlib.Path, repos: pathlib.Path,
            out: pathlib.Path) -> list[Task]:
    tasks: list[Task] = []
    for path in sorted(oracle.glob("*-tasks.json")):
        for raw in json.loads(path.read_text(encoding="utf-8")):
            truth = [f for f in raw["required_files"] if f != raw["target_file"]]
            if not truth:
                continue
            worktree, index = materialise(repos, raw, out)
            # With no definition to ask about, ask about the file. MCM indexes a
            # file as an object in its own right, and propagating from it follows
            # IMPORTS -- which is exactly the dependency a module-level change
            # breaks. Discarding these tasks threw away four of python-dotenv's
            # six, and they are the ones where the tool has most to say.
            reach = [f for f in mcm_reach(index, raw["repository"],
                                          raw["target_file"],
                                          raw["changed_defs"])
                     if not same_file(normalise(f), normalise(raw["target_file"]))]
            informative = any(any(same_file(normalise(t), normalise(r))
                                  for r in reach) for t in truth)
            direction = dependency_direction(index, raw["repository"],
                                             raw["target_file"], truth)
            tasks.append(Task(
                task_id=raw["task_id"], repository=raw["repository"],
                commit=raw["commit"], parent=raw["parent"],
                target_file=raw["target_file"], changed_defs=raw["changed_defs"],
                truth=truth, mcm_files=reach, mcm_informative=informative,
                direction=direction, worktree=portable(worktree),
                index=portable(index)))
            print(f"  {raw['task_id']}: {len(truth)} required ({direction}), "
                  f"mcm names {len(reach)}"
                  f"{' (hits)' if informative else ''}", flush=True)
    return tasks


# --- running ----------------------------------------------------------------

# The first line has to stand on its own. When the agent runs out of steps,
# `run_agent` asks for the final answer in a fresh conversation seeded with
# `prompt.splitlines()[0]` plus a digest of the tool output, so a question split
# across two lines arrives there as a sentence fragment. It showed: every answer
# in the first probe named the target file's own definitions, which is what a
# model does when the half of the question saying "which *other* files" was cut.
PROMPT = """{what} changed, so which OTHER files in this repository must now be updated? Do not name `{target_file}` itself.

The test suite fails as things stand. Find the files that must change with it,
and do not guess: check, using the tools.

When you are done, list the files you would edit, one per line, each on its own
line prefixed with EDIT: and written as a path relative to the repository root.

Available tools: {tools}
"""


def run_one(task: Task, condition: str, trial: int, pool: KeyPool) -> Trial:
    key = pool.acquire()          # chooses the model; _post re-acquires per request
    tools = TOOLS_TREATMENT if condition == "treatment" else TOOLS_CONTROL
    toolbox = ToolBox(repo=pathlib.Path(task.worktree), db=pathlib.Path(task.index))
    what = (f"The definitions {', '.join(f'`{d}`' for d in task.changed_defs)} "
            f"in `{task.target_file}`"
            if task.changed_defs else f"Module-level code in `{task.target_file}`")
    prompt = PROMPT.format(what=what, target_file=task.target_file,
                           tools=", ".join(tools))
    try:
        text, calls, error = run_agent(prompt=prompt, tools=tools, toolbox=toolbox,
                                       pool=pool, provider=key.provider,
                                       model=key.model)
    except AllKeysExhausted:
        raise
    except Exception as exc:  # noqa: BLE001 - one trial failing is not the run
        return Trial(task_id=task.task_id, condition=condition, trial=trial,
                     error=f"{type(exc).__name__}: {exc}", model=key.model)
    edited = [e for e in parse_edits(text)
              if not same_file(normalise(e), normalise(task.target_file))
              and normalise(e) not in ("none", "")]
    return Trial(task_id=task.task_id, condition=condition, trial=trial,
                 edited=edited, tool_calls=calls,
                 used_mcm="mcm_impact" in toolbox.used, error=error,
                 model=key.model, tools_used=sorted(toolbox.used))


def build_pool(out: pathlib.Path) -> KeyPool:
    import httpx

    # One model, so that a difference between the conditions cannot be a
    # difference between models. Gemini rather than the NVIDIA endpoint the
    # earlier experiment used, measured rather than assumed: NVIDIA returned 429
    # on roughly every other request and spent 325 seconds of cooldown to land 20
    # of them, which at this run's size is hours of waiting. Three Gemini keys
    # sustained 12 requests in 27 seconds with no rate limiting at all.
    # One model, so that a difference between the conditions cannot be a
    # difference between models.
    #
    # NVIDIA rather than Gemini, measured rather than assumed. Gemini's free tier
    # allows 20 requests a day for its flash models across a whole project, which
    # three keys share and which this run spent in its first few trials: after
    # that every request was refused and eleven of twelve trials recorded nothing
    # but "exhausted retries". `gemini-flash-latest` is the same bucket. NVIDIA
    # landed 8 of 10 requests in 19 seconds on the same probe.
    #
    # `gpt-oss-20b` rather than `nemotron-3-super-120b`, on latency. Once the
    # backoff stopped wasting minutes it became clear the quota was never the
    # binding constraint: the 120b reasoning model answers in about 30 seconds,
    # which is ten hours for this plan. The 20b answers in 3.4 and sustained 10 of
    # 10 requests at 23 a minute.
    #
    # The cost of that is honest and belongs in any write-up: this measures what a
    # small model does with the tools. Both conditions run the same model, so the
    # comparison holds, but it is not evidence about a strong agent.
    #
    # Twenty a minute against the 23 observed: one key has nothing to rotate to,
    # so pacing under the limit is cheaper than discovering it.
    pool = KeyPool.from_providers(
        [("nvidia", ("NVIDIA_API_KEY",), "openai/gpt-oss-20b", 20, 2000)],
        state_path=out / "key_state.json")

    def probe(key):
        base = PROVIDERS[key.provider][0]
        try:
            r = httpx.post(f"{base}/chat/completions",
                           headers={"Authorization": f"Bearer {key.value}"},
                           json={"model": key.model,
                                 "messages": [{"role": "user", "content": "hi"}],
                                 "max_tokens": 5}, timeout=45)
        except Exception:      # noqa: BLE001 - a transient failure is not a dead key
            return None
        if r.status_code in (401, 403) or (r.status_code == 400 and "API key" in r.text):
            return f"{r.status_code} auth"
        return None

    return preflight(pool, probe)


# --- reporting --------------------------------------------------------------

def report(out: pathlib.Path) -> None:
    tasks = {t["task_id"]: Task(**t)
             for t in json.loads((out / "tasks.json").read_text(encoding="utf-8"))}
    path = out / "trials.json"
    if not path.exists():
        print("no trials yet")
        return
    trials = [Trial(**t) for t in json.loads(path.read_text(encoding="utf-8"))]
    errored = [t for t in trials if t.error]
    print(f"{len(tasks)} tasks, {len(trials)} trials, {len(errored)} errored\n")

    for label, wanted in (("mcm_impact names a required file", True),
                          ("mcm_impact names nothing required", False)):
        ids = {k for k, v in tasks.items() if v.mcm_informative is wanted}
        if not ids:
            continue
        print(f"{label}  ({len(ids)} tasks)")
        print(f"  {'condition':<12}{'recall':>9}{'precision':>11}"
              f"{'tool calls':>12}{'trials':>9}")
        for condition in ("control", "treatment"):
            rows = [t for t in trials if t.task_id in ids
                    and t.condition == condition and not t.error]
            if not rows:
                continue
            scores = [score(t.edited, tasks[t.task_id].truth) for t in rows]
            recall = sum(s[0] for s in scores) / len(scores)
            prec = sum(s[1] for s in scores) / len(scores)
            calls = sum(t.tool_calls for t in rows) / len(rows)
            print(f"  {condition:<12}{recall:>9.3f}{prec:>11.3f}"
                  f"{calls:>12.1f}{len(rows):>9}")
        used = [t for t in trials if t.task_id in ids
                and t.condition == "treatment" and not t.error]
        if used:
            print(f"  mcm_impact actually called in "
                  f"{sum(1 for t in used if t.used_mcm)}/{len(used)} treatment trials")
        print()

    # Paired, per task. With a handful of tasks a difference of means is a
    # summary of very little; what a reader can check is which tasks moved, in
    # which direction, and by how much.
    print("per task (recall, averaged over trials)")
    print(f"  {'task':<22}{'direction':>11}{'required':>9}{'mcm':>5}"
          f"{'control':>9}{'treatment':>11}{'delta':>8}")
    deltas = []
    for task_id, task in sorted(tasks.items()):
        means = {}
        for condition in ("control", "treatment"):
            rows = [t for t in trials if t.task_id == task_id
                    and t.condition == condition and not t.error]
            means[condition] = (sum(score(t.edited, task.truth)[0] for t in rows)
                                / len(rows)) if rows else None
        if means["control"] is None or means["treatment"] is None:
            continue
        delta = means["treatment"] - means["control"]
        deltas.append(delta)
        print(f"  {task_id:<22}{task.direction:>11}{len(task.truth):>9}"
              f"{len(task.mcm_files):>5}{means['control']:>9.2f}"
              f"{means['treatment']:>11.2f}{delta:>+8.2f}")
    if deltas:
        better = sum(1 for d in deltas if d > 0)
        worse = sum(1 for d in deltas if d < 0)
        print(f"\n  treatment better on {better}, worse on {worse}, tied on "
              f"{len(deltas) - better - worse} of {len(deltas)} tasks; "
              f"mean paired delta {sum(deltas) / len(deltas):+.3f}")

    print("\n  Recall and precision are over files, the granularity the tests")
    print("  adjudicated. The target file is excluded from both.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--oracle", type=pathlib.Path, default=pathlib.Path("oracle"))
    p.add_argument("--repos", type=pathlib.Path, default=pathlib.Path("repos"))
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("oracle-experiment"))
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--clean", action="store_true",
                   help="drop prepared worktrees and indexes before preparing")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for repo in sorted(args.repos.iterdir()) if args.repos.exists() else []:
            subprocess.run(["git", "worktree", "prune"], cwd=repo,
                           capture_output=True, encoding="utf-8", errors="replace")
        shutil.rmtree(args.out / "worktrees", ignore_errors=True)
        shutil.rmtree(args.out / "index", ignore_errors=True)

    if args.prepare:
        tasks = prepare(args.oracle, args.repos, args.out)
        (args.out / "tasks.json").write_text(
            json.dumps([asdict(t) for t in tasks], indent=2), encoding="utf-8")
        informative = sum(1 for t in tasks if t.mcm_informative)
        print(f"\n{len(tasks)} tasks: {informative} where mcm_impact names a file "
              f"the tests proved required, {len(tasks) - informative} where it "
              f"does not")
        return 0

    if args.run:
        tasks = [Task(**t) for t in
                 json.loads((args.out / "tasks.json").read_text(encoding="utf-8"))]
        # Interleaved and shuffled: a drift in the model or the machine during a
        # long run must not land preferentially on one condition.
        plan = [(t, c, i) for t in tasks
                for c in ("control", "treatment")
                for i in range(args.trials)]
        random.Random(SEED).shuffle(plan)
        if args.limit:
            plan = plan[:args.limit]

        pool = build_pool(args.out)
        path = args.out / "trials.json"
        results = ([Trial(**t) for t in json.loads(path.read_text(encoding="utf-8"))]
                   if path.exists() else [])
        done = {(t.task_id, t.condition, t.trial) for t in results if not t.error}

        for index, (task, condition, trial) in enumerate(plan, start=1):
            if (task.task_id, condition, trial) in done:
                continue
            try:
                results.append(run_one(task, condition, trial, pool))
            except AllKeysExhausted as exc:
                print(f"\n  {exc}")
                break
            # Checkpoint every trial: an interrupted run keeps what it paid for.
            path.write_text(json.dumps([asdict(t) for t in results], indent=2),
                            encoding="utf-8")
            if index % 5 == 0:
                print(f"  {index}/{len(plan)}", flush=True)

        print(f"{len(results)} trials recorded")
        print(pool.summary())

    if args.report or args.run:
        report(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
