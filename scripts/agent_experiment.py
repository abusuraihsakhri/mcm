"""Does giving an agent `mcm_impact` change what it edits?

Everything else in this repository measures what MCM puts in front of an agent.
Nothing measures what an agent then does with it, which is `evaluation.md`
limitation 8 and, after the retrieval results, the only claim left that is worth
making.

## The task

Git history supplies both the task and the answer. Take a commit that changed a
function's signature *and* updated its callers. Check out the parent, apply only
the signature change, and the repository is now internally inconsistent in
exactly the way an agent's own edit would leave it. Ask the agent to repair it.
The callers that the real commit updated are the answer, and the agent never sees
them.

## The conditions

Identical model, prompt, temperature and task. The only difference is the tool
list:

    control     read_file, list_files, grep
    treatment   read_file, list_files, grep, mcm_impact

`grep` is in both deliberately. The interesting comparison is not against an
agent with no tools, it is against an agent doing what agents actually do today.
If grep finds the callers just as well, the treatment has nothing to offer and
this experiment should say so.

## What is measured

Recall of the caller set: of the definitions the real commit updated, how many
did the agent edit. Precision is reported beside it, because an agent that edits
everything scores perfect recall and is useless.

Both conditions run `n` trials per task, because a language model is not
deterministic and a single run of each is a coin flip dressed as a result.

## The ceiling, which bounds what this can ever show

Measured on the task set this builds: across the informative tasks, the
definitions `mcm_impact` names cover a **mean of 17% of what the commit actually
edited** (median 7%, best case 67%). A treatment agent that uses the tool
perfectly and does nothing else therefore tops out near 17% recall.

That is not a flaw in the harness, it is the same property the prediction
measurement already found from the other side: propagation is conservative. It
names the callers it can justify, which is roughly one of four definitions a
commit touches, because commits also touch things that are not causally
downstream of anything.

Two consequences, and both belong in any write-up:

- **The effect size available to detect is small.** Detecting a difference that
  is bounded at 17% against agent-to-agent variance needs far more trials than
  18 tasks x 2 conditions can provide. Running it at this size and reporting a
  null would be reporting the sample size, not the tool.
- **Recall against the full co-change set is the wrong dependent variable.** A
  fairer one is whether the agent found the callers that *genuinely break*,
  which co-change does not isolate. Getting that needs a test oracle -- revert
  the caller fixes, run the suite, count failures -- rather than a diff.

## What this cannot show

Tasks are drawn from the ~10% of signature changes where MCM's propagation names
at least one real caller, plus a matched sample where it names none. The first
group is where the treatment could help; the second is there so the report can
say how often it has nothing to say. Neither is a random sample of an agent's
working day, and the write-up has to state that.

    python scripts/agent_experiment.py --build   --out experiment/
    python scripts/agent_experiment.py --dry-run --out experiment/
    python scripts/agent_experiment.py --run --trials 3 --out experiment/
    python scripts/agent_experiment.py --report  --out experiment/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from agent_backend import (PROVIDERS, TOOLS_CONTROL, TOOLS_TREATMENT, ToolBox,
                           parse_edits, run_agent)
from key_pool import AllKeysExhausted, KeyPool, preflight

from mcm.core.change import Change, ChangeKind
from mcm.reasoning.change_propagation import propagate
from mcm.reasoning.prediction import observations_from_history
from mcm.storage.sqlite_store import SQLiteStore

SEED = 20260913


@dataclass
class Task:
    task_id: str
    repository: str
    commit: str
    #: The definition whose signature changed.
    target: str
    #: Definitions the real commit edited alongside it. The answer.
    truth: list[str]
    #: What MCM's propagation would name. Empty for the matched control group.
    mcm_names: list[str]
    #: True when MCM names at least one definition the commit really edited.
    mcm_informative: bool


@dataclass
class Trial:
    task_id: str
    condition: str
    trial: int
    edited: list[str] = field(default_factory=list)
    tool_calls: int = 0
    used_mcm: bool = False
    error: str | None = None
    #: Which model answered. Recorded because the pool spans providers, and a
    #: comparison that silently mixes models measures the models too.
    model: str = ""
    tools_used: list[str] = field(default_factory=list)

    def recall(self, truth: list[str]) -> float:
        if not truth:
            return 0.0
        return len(self._hits(truth)) / len(truth)

    def precision(self, truth: list[str]) -> float:
        claimed = {normalise(e) for e in self.edited} - {("", "")}
        if not claimed:
            return 0.0
        return len(self._hits(truth)) / len(claimed)

    def _hits(self, truth: list[str]) -> set:
        claimed = {normalise(e) for e in self.edited}
        return {t for t in truth if normalise(t) in claimed}


def normalise(reference: str) -> tuple[str, str]:
    """Reduce an identifier to (file stem, definition name) for comparison.

    The agent writes `src/dotenv/main.py::DotEnv.dict`; ground truth is
    `repo://python-dotenv/src/dotenv/main.py#function:dotenv_values`. Comparing
    those as strings scores every correct answer as wrong, so both sides collapse
    to the two parts that actually identify a definition: which file, and which
    name. The leading path and the `#type:` tag carry no information the other
    side has.
    """
    text = (reference or "").strip().strip("`\"'")
    if text.lower() in ("", "none"):
        return ("", "")

    if "#" in text:                       # repo://path/to/file.py#function:name
        path, _, tail = text.partition("#")
        name = tail.partition(":")[2] or tail
    elif "::" in text:                    # path/to/file.py::Class.method
        path, _, name = text.partition("::")
    else:
        path, name = text, ""

    stem = path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if stem.endswith(".py"):
        stem = stem[:-3]
    # A method is identified by its own name; Class.method and method are the
    # same definition seen from two naming conventions.
    leaf = name.replace("()", "").strip().rsplit(".", 1)[-1]
    return (stem, leaf)


# --- task construction ------------------------------------------------------

def build_tasks(pred_dir: pathlib.Path) -> list[Task]:
    """Draw tasks from the history databases the prediction suite already built."""
    informative: list[Task] = []
    uninformative: list[Task] = []
    seen: set[tuple[str, str]] = set()

    for db in sorted(pred_dir.glob("*.db")):
        store = SQLiteStore(db)
        try:
            for obs in observations_from_history(store):
                if obs.kind is not ChangeKind.SIGNATURE or not obs.co_changed:
                    continue
                key = (db.stem, obs.target_id)
                if key in seen:            # the same definition changes repeatedly;
                    continue               # one task per definition keeps it honest
                seen.add(key)
                try:
                    result = propagate(store, Change(obs.target_id, obs.kind),
                                       max_depth=6)
                except KeyError:
                    continue
                names = sorted({p.object.id for p in result.must_update})
                hit = bool(set(names) & obs.co_changed)
                task = Task(
                    task_id=f"{db.stem}:{obs.target_id.rsplit('#', 1)[-1]}",
                    repository=db.stem, commit=obs.commit_id,
                    target=obs.target_id, truth=sorted(obs.co_changed),
                    mcm_names=names, mcm_informative=hit,
                )
                (informative if hit else uninformative).append(task)
        finally:
            store.close()

    # Matched control group, same size, so the report can say how often the
    # treatment has nothing to offer without that group swamping the comparison.
    random.Random(SEED).shuffle(uninformative)
    return informative + uninformative[:len(informative)]


# --- agent backends ---------------------------------------------------------

PROMPT = """The signature of `{target}` has changed in this repository.

Find every definition that must be updated as a consequence, and say which ones.
Do not guess: check. When you are done, list the definitions you would edit, one
per line, each on its own line prefixed with EDIT:

Available tools: {tools}
"""


def run_stub(task: Task, condition: str, trial: int) -> Trial:
    """A deterministic stand-in used to test the harness without spending money.

    It is not a model and its output is not evidence. It exists so the plumbing --
    task construction, scoring, aggregation -- can be exercised and debugged
    before any real run, and so a broken pipeline is found for free rather than
    after an API bill.
    """
    rng = random.Random(f"{task.task_id}:{condition}:{trial}")
    if condition == "treatment" and task.mcm_informative:
        found = [d for d in task.truth if d in task.mcm_names]
        found += [d for d in task.truth if d not in found and rng.random() < 0.3]
    else:
        found = [d for d in task.truth if rng.random() < 0.4]
    return Trial(task_id=task.task_id, condition=condition, trial=trial,
                 edited=sorted(found), tool_calls=rng.randint(2, 9),
                 used_mcm=(condition == "treatment"))


def run_api(task: Task, condition: str, trial: int, repo_root: pathlib.Path,
            db: pathlib.Path, pool: KeyPool) -> Trial:
    """One trial against a pooled provider."""
    key = pool.acquire()          # only to choose the model; _post re-acquires
    tools = TOOLS_TREATMENT if condition == "treatment" else TOOLS_CONTROL
    toolbox = ToolBox(repo=repo_root, db=db)
    try:
        text, calls, error = run_agent(
            prompt=PROMPT.format(target=task.target, tools=", ".join(tools)),
            tools=tools, toolbox=toolbox, pool=pool,
            provider=key.provider, model=key.model)
    except AllKeysExhausted:
        raise
    except Exception as exc:  # noqa: BLE001 - one trial failing is not the run
        return Trial(task_id=task.task_id, condition=condition, trial=trial,
                     error=f"{type(exc).__name__}: {exc}", model=key.model)
    return Trial(task_id=task.task_id, condition=condition, trial=trial,
                 edited=parse_edits(text), tool_calls=calls,
                 used_mcm="mcm_impact" in toolbox.used, error=error,
                 model=key.model, tools_used=sorted(toolbox.used))


def run_claude_cli(task: Task, condition: str, trial: int,
                   repo_root: pathlib.Path, db: pathlib.Path) -> Trial:
    """Drive Claude Code headlessly. Bills the operator's account; opt in only."""
    tools = TOOLS_TREATMENT if condition == "treatment" else TOOLS_CONTROL
    prompt = PROMPT.format(target=task.target, tools=", ".join(tools))
    cmd = ["claude", "-p", prompt, "--output-format", "json"]
    try:
        done = subprocess.run(cmd, cwd=repo_root, capture_output=True,
                              text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Trial(task_id=task.task_id, condition=condition, trial=trial,
                     error=f"{type(exc).__name__}: {exc}")
    edited = re.findall(r"^EDIT:\s*(\S+)", done.stdout, re.M)
    return Trial(task_id=task.task_id, condition=condition, trial=trial,
                 edited=sorted(set(edited)),
                 used_mcm=(condition == "treatment"))


BACKENDS = {"stub": run_stub, "api": run_api, "claude-cli": run_claude_cli}


# --- reporting --------------------------------------------------------------

def build_pool(pred: pathlib.Path) -> KeyPool:
    """Every key found, across providers, validated before the run starts."""
    import httpx

    pool = KeyPool.from_providers(
        # Gemini keys minted in one project share that project's quota: they
        # rate limit together rather than independently, which the first run
        # showed plainly. Paced low for that, and NVIDIA is a genuinely separate
        # budget so it carries more of the load.
        # One model, deliberately. Three Gemini keys minted in one project share
        # that project's quota -- they rate limit together, which the first run
        # showed -- so they buy throughput that does not exist. Running a single
        # model also removes the confound of different trials being answered by
        # different models, which matters more than the throughput would have.
        [("nvidia", ("NVIDIA_API_KEY",), "nvidia/nemotron-3-super-120b-a12b", 40, 900)],
        state_path=pred.parent / "key_state.json")

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


def report(out: pathlib.Path) -> None:
    tasks = {t["task_id"]: t for t in json.loads((out / "tasks.json").read_text())}
    trials_path = out / "trials.json"
    if not trials_path.exists():
        print("no trials yet")
        return
    trials = [Trial(**t) for t in json.loads(trials_path.read_text())]

    print(f"{len(tasks)} tasks, {len(trials)} trials, "
          f"{sum(1 for t in trials if t.error)} errored\n")

    for group, wanted in (("MCM names a real caller", True),
                          ("MCM names nothing useful", False)):
        ids = {k for k, v in tasks.items() if v["mcm_informative"] is wanted}
        if not ids:
            continue
        print(f"{group}  ({len(ids)} tasks)")
        print(f"  {'condition':<12}{'recall':>9}{'precision':>11}{'tool calls':>12}")
        for condition in ("control", "treatment"):
            rows = [t for t in trials
                    if t.task_id in ids and t.condition == condition and not t.error]
            if not rows:
                continue
            recall = sum(t.recall(tasks[t.task_id]["truth"]) for t in rows) / len(rows)
            prec = sum(t.precision(tasks[t.task_id]["truth"]) for t in rows) / len(rows)
            calls = sum(t.tool_calls for t in rows) / len(rows)
            print(f"  {condition:<12}{recall:>9.3f}{prec:>11.3f}{calls:>12.1f}")
        print()

    print("  Recall is of the callers the real commit updated. Precision is beside")
    print("  it because an agent that edits everything scores perfect recall.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("experiment"))
    p.add_argument("--pred", type=pathlib.Path, default=pathlib.Path("prediction"),
                   help="directory of history databases from prediction_suite.py")
    p.add_argument("--repos", type=pathlib.Path, default=pathlib.Path("repos"))
    p.add_argument("--build", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="run the stub backend: exercises the pipeline, spends nothing")
    p.add_argument("--report", action="store_true")
    p.add_argument("--backend", choices=sorted(BACKENDS), default="claude-cli")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--limit", type=int, default=0,
                   help="stop after this many trials; use for the floor check")
    p.add_argument("--resume", action="store_true", default=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.build:
        tasks = build_tasks(args.pred)
        (args.out / "tasks.json").write_text(
            json.dumps([asdict(t) for t in tasks], indent=2), encoding="utf-8")
        informative = sum(1 for t in tasks if t.mcm_informative)
        print(f"{len(tasks)} tasks: {informative} where MCM names a real caller, "
              f"{len(tasks) - informative} matched controls")
        return 0

    if args.run or args.dry_run:
        backend = BACKENDS["stub" if args.dry_run else args.backend]
        tasks = [Task(**t) for t in json.loads((args.out / "tasks.json").read_text())]
        # Interleaved and shuffled, so a drift in the model or the machine over
        # the run cannot land preferentially on one condition.
        plan = [(t, c, i) for t in tasks
                for c in ("control", "treatment")
                for i in range(args.trials)]
        random.Random(SEED).shuffle(plan)

        pool = None
        if backend is run_api:
            pool = build_pool(args.pred)
        if args.limit:
            plan = plan[:args.limit]

        # Resume rather than restart: a run stopped by a daily quota should
        # continue tomorrow, not begin again and spend the budget twice.
        trials_path = args.out / "trials.json"
        results: list[Trial] = ([Trial(**t) for t in
                                 json.loads(trials_path.read_text())]
                                if trials_path.exists() and args.resume else [])
        done = {(t.task_id, t.condition, t.trial) for t in results}

        for index, (task, condition, trial) in enumerate(plan, start=1):
            if (task.task_id, condition, trial) in done:
                continue
            try:
                if backend is run_stub:
                    results.append(backend(task, condition, trial))
                elif backend is run_api:
                    results.append(backend(task, condition, trial,
                                           args.repos / task.repository,
                                           args.pred / f"{task.repository}.db", pool))
                else:
                    results.append(backend(task, condition, trial,
                                           args.repos / task.repository,
                                           args.pred / f"{task.repository}.db"))
            except AllKeysExhausted as exc:
                print(f"\n  {exc}")
                break
            # Checkpoint every trial: an interrupted run keeps what it paid for.
            trials_path.write_text(
                json.dumps([asdict(t) for t in results], indent=2), encoding="utf-8")
            if index % 5 == 0:
                print(f"  {index}/{len(plan)}", flush=True)

        print(f"{len(results)} trials recorded")
        if pool is not None:
            print(pool.summary())

    if args.report or args.run or args.dry_run:
        report(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
