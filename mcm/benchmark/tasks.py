"""Benchmark tasks generated from held-out Git history (spec sections 46, 49).

Spec section 49 says to use real repositories and not to cherry-pick examples.
Both are properties of where the questions come from, so that is what this module
controls.

**The split.** A repository's history is cut at a *horizon* commit H. Every system
under test indexes the repository exactly as it stood at H. Tasks are drawn from
commits strictly after H, and the ground truth for a task is what that commit
actually changed. A system cannot retrieve the answer because the answer had not
been written yet at the revision it indexed. This is an ordinary train/test split
with time as the splitting variable, and it is the reason no human had to label
anything.

**The ceiling.** A commit after H usually touches definitions that did not exist at
H - new functions, new files. No system can retrieve those, so counting them as
misses would measure how much the repository grew rather than how well anything
retrieves. Each task therefore records both the full truth set and the subset that
existed at H, scores against the reachable subset, and reports ``ceiling`` so a
reader can see how much was set aside. A task with a low ceiling is a task about a
new feature, and the number says so instead of hiding in an average.

**The filters.** Merge commits are skipped: a merge diff is the union of other
work and has no single intent. Commits above ``max_files`` are skipped because a
repository-wide rename is not a retrieval question. Commits whose subject is
boilerplate are skipped because "bump version to 2.1.4" is not a query. Every
filter is counted and reported in ``TaskSuite.rejected``, so the gap between the
commits considered and the tasks generated is visible rather than implied.

Four of spec section 46's nine families are generated here. The rest are absent
for reasons given in ``docs/evaluation.md``; they have no ground truth in a diff.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..ingestion.git import GitCommit, GitReader
from ..ingestion.parser import SymbolDef, parse_python

#: Spec section 46 families this module can generate ground truth for.
RETRIEVAL = "retrieval"     # A. Find the function responsible for X.
IMPACT = "impact"           # C. What could break if X changes?
PLANNING = "planning"       # H. What files should be changed to implement X?
DEBUGGING = "debugging"     # I. Given this error, identify relevant components.

FAMILIES = (RETRIEVAL, IMPACT, PLANNING, DEBUGGING)

#: Subjects that are not questions about the code.
_BOILERPLATE = re.compile(
    r"^\s*(merge\b|revert\b|bump\b|release\b|v?\d+\.\d+|"
    r"update changelog|changelog\b|\[?ci skip\]?|typo\b|"
    r"update (the )?(readme|docs?|copyright)\b|pin \w+|"
    r"regenerate\b|re-?format\b|lint\b|black\b|isort\b)",
    re.IGNORECASE)

#: Housekeeping on the development environment rather than on the program.
#:
#: "fix mypy findings" and "update dev dependencies" edit real definitions, so the
#: diff-based ground truth accepts them, but the subject describes the tool that
#: complained rather than anything the code does. No retriever can connect that
#: query to that answer, and a suite full of them measures noise while looking
#: like it measures retrieval.
#:
#: This filter is defined on the query text alone and fixed before any system was
#: run. A filter tuned on which commits the systems answered badly would be
#: cherry-picking with extra steps, which spec section 49 rules out. Rejections
#: are counted separately so the cost of this rule stays visible.
_CHORE = re.compile(
    r"\b(mypy|flake8|pyright|ruff|pylint|pre-?commit|tox|nox|coverage|"
    r"dev dependencies|dependencies|requirements|type (hints?|annotations?)|"
    r"CI|workflows?|github actions|setup\.(py|cfg)|pyproject)\b",
    re.IGNORECASE)

#: A fix commit, for spec section 46's debugging family. The query an agent gets
#: in that family is a report of something being wrong, so the family is defined
#: by the commit claiming to repair something.
_FIX = re.compile(
    r"\b(fix(e[sd])?|bug|crash|regression|broken|error|exception|"
    r"traceback|fail(s|ed|ing)?|incorrect|wrong|leak|race)\b",
    re.IGNORECASE)


#: Fraction of a definition's lines a retrieved window must contain before the
#: definition counts as delivered. See ``RepoSnapshot.covering``.
COVERAGE_FLOOR = 0.5


def unit_id(relpath: str, qualname: str) -> str:
    """The granularity-neutral name for one definition.

    Deliberately not an MCM object ID. Baselines A and B have no reason to know
    the spec section 20 ID grammar, and making them speak it would be the first
    step towards making them speak MCM generally.
    """
    return f"{relpath}:{qualname}"


@dataclass(frozen=True)
class Horizon:
    """The revision every system indexes. Nothing after this is visible."""

    repo: str
    root: Path
    sha: str
    when: datetime

    @property
    def short(self) -> str:
        return self.sha[:8]


@dataclass
class RepoSnapshot:
    """The repository as it stood at the horizon.

    Shared by every system so that no system is measured against a different
    corpus. Systems differ in what they *do* with this; they do not differ in what
    they were given.
    """

    horizon: Horizon
    sources: dict[str, bytes] = field(default_factory=dict)
    definitions: dict[str, list[SymbolDef]] = field(default_factory=dict)
    #: unit id -> token cost of that definition's own source text.
    unit_tokens: dict[str, int] = field(default_factory=dict)
    #: Overridable so the sensitivity of the headline metric to this rule can be
    #: measured rather than asserted. It is not a neutral knob: raising it from 0
    #: to 0.5 moved one link of the spec section 3 chain on the first repository
    #: tested, which is exactly why it is exposed instead of buried.
    coverage_floor: float = COVERAGE_FLOOR

    @property
    def unit_ids(self) -> set[str]:
        return set(self.unit_tokens)

    @property
    def files(self) -> list[str]:
        return sorted(self.sources)

    def text_of(self, relpath: str, symbol: SymbolDef) -> str:
        source = self.sources.get(relpath, b"")
        return source[symbol.start_byte:symbol.end_byte].decode(
            "utf-8", errors="replace")

    def covering(self, relpath: str, start_line: int, end_line: int) -> dict[str, int]:
        """Definitions a line window actually delivers, and their cost.

        A chunked retriever hands over line ranges, not definitions, so the harness
        has to decide when a range counts as having delivered a definition. The
        rule is ``COVERAGE_FLOOR``: at least half the definition's lines must be
        inside the window.

        Any overlap at all is the tempting rule and it is wrong. A window clipping
        the last three lines of a forty-line function would be credited with the
        whole function, and since useful tokens are counted at the definition's
        full size, chunk retrieval would collect credit for code it never showed.
        The floor removes that without inventing a penalty: MCM and the graph
        baseline deliver whole definitions and always clear it, so the rule
        constrains only the system that can actually deliver fragments.
        """
        found: dict[str, int] = {}
        for symbol in self.definitions.get(relpath, ()):
            if symbol.end_line < start_line or symbol.start_line > end_line:
                continue
            span = max(1, symbol.end_line - symbol.start_line + 1)
            shown = (min(end_line, symbol.end_line)
                     - max(start_line, symbol.start_line) + 1)
            if shown / span < self.coverage_floor:
                continue
            key = unit_id(relpath, symbol.qualname)
            found[key] = self.unit_tokens.get(key, 0)
        return found

    def definitions_in(self, relpath: str) -> dict[str, int]:
        """Every definition in a file, for a system that answers at file level."""
        out: dict[str, int] = {}
        for symbol in self.definitions.get(relpath, ()):
            key = unit_id(relpath, symbol.qualname)
            out[key] = self.unit_tokens.get(key, 0)
        return out


@dataclass(frozen=True)
class BenchmarkTask:
    """One question with an answer nobody wrote by hand."""

    task_id: str
    family: str
    repo: str
    query: str
    commit_sha: str
    #: Every definition the commit changed, including ones that did not exist at
    #: the horizon.
    truth_definitions: frozenset[str]
    #: The subset that existed at the horizon. Scoring uses this.
    reachable_definitions: frozenset[str]
    truth_files: frozenset[str]
    reachable_files: frozenset[str]
    #: Impact family only: the definition the question is asked *about*.
    seed: str | None = None
    #: Commits between the horizon and this task's commit. Drift, reported so a
    #: reader can check whether scores decay with distance.
    distance: int = 0

    @property
    def ceiling(self) -> float:
        """Fraction of the true answer that was reachable at the horizon.

        1.0 means the commit only touched code that already existed. A task
        scoring 0.4 has 60% of its answer in code no system could have retrieved,
        and its recall must be read against that.
        """
        if not self.truth_definitions:
            return 0.0
        return len(self.reachable_definitions) / len(self.truth_definitions)

    @property
    def targets(self) -> frozenset[str]:
        """What a system is scored against for this family."""
        if self.family == PLANNING:
            return self.reachable_files
        if self.family == IMPACT:
            return frozenset(self.reachable_definitions - {self.seed})
        return self.reachable_definitions

    @property
    def granularity(self) -> str:
        return "file" if self.family == PLANNING else "definition"


@dataclass
class TaskSuite:
    """Generated tasks plus the accounting for what was thrown away."""

    horizon: Horizon
    snapshot: RepoSnapshot
    tasks: list[BenchmarkTask] = field(default_factory=list)
    considered: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def of_family(self, family: str) -> list[BenchmarkTask]:
        return [t for t in self.tasks if t.family == family]

    @property
    def families(self) -> list[str]:
        return [f for f in FAMILIES if self.of_family(f)]

    def summary(self) -> str:
        counts = ", ".join(f"{f}={len(self.of_family(f))}" for f in self.families)
        dropped = ", ".join(f"{k}={v}" for k, v in sorted(self.rejected.items()))
        ceilings = [t.ceiling for t in self.tasks]
        mean_ceiling = sum(ceilings) / len(ceilings) if ceilings else 0.0
        return (f"{len(self.tasks)} tasks from {self.considered} commits after "
                f"{self.horizon.short} ({counts}); mean ceiling "
                f"{mean_ceiling:.2f}; rejected: {dropped or 'none'}")


def build_snapshot(reader: GitReader, horizon: Horizon, *,
                   include: str = ".py",
                   coverage_floor: float = COVERAGE_FLOOR) -> RepoSnapshot:
    """Read and parse the repository at the horizon revision.

    Reads blobs out of Git rather than the working tree, so the snapshot is the
    horizon revision even when the checkout is somewhere else entirely.
    """
    from .metrics import token_estimate

    snapshot = RepoSnapshot(horizon=horizon, coverage_floor=coverage_floor)
    wanted = [relpath for relpath in reader.files_at(horizon.sha)
              if relpath.endswith(include)]
    # One git invocation rather than one per file; see GitReader.blobs.
    blobs = reader.blobs(horizon.sha, wanted)
    for relpath in wanted:
        blob = blobs.get(relpath)
        if blob is None:
            continue
        snapshot.sources[relpath] = blob
        try:
            parsed = parse_python(blob, relpath)
        except Exception:  # noqa: BLE001 - an unparsable revision contributes no symbols
            continue
        snapshot.definitions[relpath] = list(parsed.symbols)
        for symbol in parsed.symbols:
            text = blob[symbol.start_byte:symbol.end_byte].decode(
                "utf-8", errors="replace")
            snapshot.unit_tokens[unit_id(relpath, symbol.qualname)] = token_estimate(text)
    return snapshot


def generate_tasks(root: Path | str, *, repo: str | None = None,
                   horizon_back: int = 60, max_tasks: int = 40,
                   max_files: int = 8, min_subject: int = 12,
                   include: str = ".py",
                   coverage_floor: float = COVERAGE_FLOOR) -> TaskSuite:
    """Split a repository's history and turn the held-out half into questions.

    ``horizon_back`` commits back from HEAD is the horizon. Everything older is
    what the systems get to see; the ``horizon_back`` commits after it are the
    candidate task pool, oldest first, so ``max_tasks`` truncates the *far* end
    rather than silently selecting the easiest.
    """
    root = Path(root)
    reader = GitReader(root)
    if not reader.is_repository():
        raise ValueError(f"{root} is not a Git repository")

    name = repo or root.name
    # GitReader.commits() is oldest-first, so HEAD is the last entry and the
    # horizon counts back from the end. Reading this the other way round puts the
    # horizon near the start of history and draws every task from the repository's
    # first weeks, which is a different experiment that looks identical in the
    # output.
    history = list(reader.commits())
    if len(history) < horizon_back + 2:
        raise ValueError(
            f"{name} has {len(history)} commits; need at least "
            f"{horizon_back + 2} to split at {horizon_back} back from HEAD")

    horizon_commit = history[-(horizon_back + 1)]
    horizon = Horizon(repo=name, root=root, sha=horizon_commit.sha,
                      when=horizon_commit.date)
    snapshot = build_snapshot(reader, horizon, include=include,
                              coverage_floor=coverage_floor)
    suite = TaskSuite(horizon=horizon, snapshot=snapshot)

    # Everything after the horizon, oldest first, so truncating at max_tasks drops
    # the commits furthest from the horizon rather than selecting for difficulty.
    candidates = history[-horizon_back:]
    for distance, commit in enumerate(candidates, start=1):
        suite.considered += 1
        made = _tasks_for(reader, snapshot, name, commit, distance,
                          max_files=max_files, min_subject=min_subject,
                          include=include, rejected=suite.rejected)
        suite.tasks.extend(made)
        if len({t.commit_sha for t in suite.tasks}) >= max_tasks:
            break
    return suite


def _tasks_for(reader: GitReader, snapshot: RepoSnapshot, repo: str,
               commit: GitCommit, distance: int, *, max_files: int,
               min_subject: int, include: str,
               rejected: dict[str, int]) -> list[BenchmarkTask]:
    """Turn one commit into zero or more tasks, or record why it produced none."""

    def reject(reason: str) -> list[BenchmarkTask]:
        rejected[reason] = rejected.get(reason, 0) + 1
        return []

    if len(commit.parents) != 1:
        return reject("merge or root commit")

    subject = (commit.subject or "").strip()
    if len(subject) < min_subject:
        return reject("subject too short")
    if _BOILERPLATE.match(subject):
        return reject("boilerplate subject")
    if _CHORE.search(subject):
        return reject("tooling or dependency chore")

    changed = {path: status for path, status in reader.changed_paths(commit).items()
               if path.endswith(include)}
    if not changed:
        return reject("no source files changed")
    if len(changed) > max_files:
        return reject("too many files changed")

    parent = commit.parents[0]
    truth_defs: set[str] = set()
    # Two git invocations per commit rather than two per changed file. Over a few
    # hundred commits that is the difference between minutes and seconds, because
    # the cost is process creation rather than reading.
    paths = list(changed)
    before_blobs = reader.blobs(parent, paths)
    after_blobs = reader.blobs(commit.sha, paths)
    for relpath in paths:
        before = _defs_in(before_blobs.get(relpath), relpath)
        after = _defs_in(after_blobs.get(relpath), relpath)
        for qualname in _differing(before, after):
            truth_defs.add(unit_id(relpath, qualname))
    if not truth_defs:
        return reject("no definitions changed")

    reachable_defs = frozenset(truth_defs & snapshot.unit_ids)
    if not reachable_defs:
        return reject("no changed definition existed at the horizon")

    truth_files = frozenset(changed)
    reachable_files = frozenset(f for f in truth_files if f in snapshot.sources)

    common = dict(
        repo=repo, query=subject, commit_sha=commit.sha,
        truth_definitions=frozenset(truth_defs),
        reachable_definitions=reachable_defs,
        truth_files=truth_files, reachable_files=reachable_files,
        distance=distance,
    )
    short = commit.sha[:8]
    tasks = [
        BenchmarkTask(task_id=f"{repo}:{short}:{RETRIEVAL}",
                      family=RETRIEVAL, **common),
    ]
    if reachable_files:
        tasks.append(BenchmarkTask(task_id=f"{repo}:{short}:{PLANNING}",
                                   family=PLANNING, **common))
    if _FIX.search(subject):
        tasks.append(BenchmarkTask(task_id=f"{repo}:{short}:{DEBUGGING}",
                                   family=DEBUGGING, **common))
    if len(reachable_defs) >= 2:
        # The seed is chosen by sort order rather than by which choice makes the
        # task easiest. Any rule that looked at the answer would be cherry-picking
        # inside a single task.
        seed = sorted(reachable_defs)[0]
        tasks.append(BenchmarkTask(task_id=f"{repo}:{short}:{IMPACT}",
                                   family=IMPACT, seed=seed, **common))
    return tasks


def _defs_in(blob: bytes | None, relpath: str) -> dict[str, bytes]:
    """qualname -> definition source, for one file's content at one revision.

    Takes the content rather than fetching it, so the caller can read a whole
    commit's files in one batch.
    """
    if not blob:
        return {}
    try:
        parsed = parse_python(blob, relpath)
    except Exception:  # noqa: BLE001 - unparsable revision contributes nothing
        return {}
    return {s.qualname: blob[s.start_byte:s.end_byte] for s in parsed.symbols}


def _differing(before: dict[str, bytes], after: dict[str, bytes]) -> set[str]:
    """Definitions added, removed, or textually changed between two revisions."""
    changed = {q for q, text in after.items() if before.get(q) != text}
    changed |= {q for q in before if q not in after}
    return changed


def query_for(task: BenchmarkTask) -> str:
    """The text a system is actually given for this task.

    Every family but impact asks the commit subject as written. Impact asks about
    a named definition instead, because spec section 46 C is a question about a
    symbol rather than about an intent.
    """
    if task.family == IMPACT and task.seed:
        name = task.seed.split(":", 1)[-1]
        return f"what could break if {name} changes"
    return task.query
