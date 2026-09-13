"""Running the experiment and reporting it (spec sections 49, 3).

Spec section 49's procedure, in order: ingest, generate questions, run each
system, record metrics, repeat across tasks, compare statistically. This module
is that loop plus the reporting, and the reporting is the part that has to be
careful.

Spec section 3 requires the experiment to be able to produce ``MCM > Graph``,
``MCM ~ Graph`` or ``MCM < Graph`` depending on the evidence. ``verdict`` returns
whichever of those the numbers support, including the two that would embarrass the
premise. It reaches "greater than" only when the paired bootstrap interval for the
difference excludes zero *and* the difference is larger than one percent of the
means being compared; otherwise the answer is "approximately equal", and a
negative interval clearing both bars reads as "less than". The size test is there
because with 149 paired tasks a gap in the fifth decimal separates cleanly, and
reporting that as one system beating another is true and worthless. Both
thresholds are fixed constants in ``metrics``, not values chosen after seeing a
result.

The ordering claim in the hypothesis is a chain, ``MCM > Graph > Vector``, so it
is reported as a chain: each link is decided on its own evidence, and a chain with
one broken link is reported broken rather than rounded off to the nearest
agreeable summary.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .metrics import (METRICS, MIN_N, Aggregate, Comparison, ContextEfficiency,
                      TaskScore, aggregate, mean, paired_bootstrap,
                      paired_values, score_ranking)
from .systems import DEFAULT_BUDGET, BenchmarkSystem, SystemAnswer
from .tasks import FAMILIES, BenchmarkTask, TaskSuite

#: The spec section 3 chain, strongest first.
HYPOTHESIS_CHAIN = ("mcm", "graph", "vector-rag")

GREATER = "greater than"
SIMILAR = "approximately equal to"
LESS = "less than"


@dataclass
class BenchmarkRun:
    """Everything one repository's experiment produced."""

    repo: str
    horizon: str
    suite: TaskSuite
    systems: list[str] = field(default_factory=list)
    scores: list[TaskScore] = field(default_factory=list)
    budget: int = DEFAULT_BUDGET
    k: int = 10
    seconds: float = 0.0
    prepare_seconds: dict[str, float] = field(default_factory=dict)

    def aggregates(self) -> list[Aggregate]:
        out: list[Aggregate] = []
        for family in FAMILIES:
            if not self.suite.of_family(family):
                continue
            for system in self.systems:
                out.append(aggregate(self.scores, system, family))
        return out

    def overall(self, system: str, metric: str) -> float:
        return mean([s.value(metric) for s in self.scores if s.system == system])


def score_answer(task: BenchmarkTask, answer: SystemAnswer, *,
                 k: int = 10) -> TaskScore:
    """Grade one answer. The system had no part in this.

    Planning is scored on files because spec section 46 H asks which files should
    change; every other family is scored on definitions.
    """
    targets = task.targets
    granularity = task.granularity
    # The symbol an impact question names is not an answer to it, and no system
    # should be credited or charged a rank slot for returning it.
    exclude = frozenset({task.seed}) if task.seed else frozenset()
    ranked = (answer.ranked_files() if granularity == "file"
              else answer.ranked_definitions(exclude=exclude))
    useful = answer.useful_tokens(targets, granularity)
    return TaskScore(
        task_id=task.task_id,
        family=task.family,
        system=answer.system,
        retrieval=score_ranking(ranked, targets, k=k),
        efficiency=ContextEfficiency(useful_tokens=useful,
                                     total_tokens=answer.total_tokens,
                                     units=len(answer.units)),
        ceiling=task.ceiling,
        latency_ms=answer.latency_ms,
    )


def run_suite(suite: TaskSuite, systems: list[BenchmarkSystem], *,
              budget: int = DEFAULT_BUDGET, k: int = 10,
              progress: bool = False) -> BenchmarkRun:
    """Prepare every system once, then ask every system every question."""
    started = time.perf_counter()
    run = BenchmarkRun(repo=suite.horizon.repo, horizon=suite.horizon.short,
                       suite=suite, budget=budget, k=k,
                       systems=[s.name for s in systems])

    # Systems are prepared, asked and released in groups rather than all at once.
    # Holding four indexes simultaneously is what made the large tier fail:
    # MCM's in-memory store alone reaches 1.8GB on django. Scoring order does not
    # affect any result, because `paired_values` lines scores up by task id.
    for group in _dependency_groups(systems):
        for system in group:
            mark = time.perf_counter()
            system.prepare(suite.snapshot)
            run.prepare_seconds[system.name] = time.perf_counter() - mark
            if progress:
                print(f"  prepared {system.name} in "
                      f"{run.prepare_seconds[system.name]:.1f}s", flush=True)

        names = "+".join(s.name for s in group)
        for index, task in enumerate(suite.tasks, start=1):
            for system in group:
                run.scores.append(
                    score_answer(task, system.answer(task, budget=budget), k=k))
            if progress and index % 20 == 0:
                print(f"  {names}: {index}/{len(suite.tasks)} tasks", flush=True)

        for system in group:
            release = getattr(system, "release", None)
            if callable(release):
                release()

    run.seconds = time.perf_counter() - started
    return run


def _dependency_groups(systems: list[BenchmarkSystem]) -> list[list[BenchmarkSystem]]:
    """Partition systems so that nothing is released while something still needs it.

    Baseline D reuses the *same instances* as baselines A and B rather than
    building its own, which is deliberate: it fuses exactly what they return. It
    also means releasing A before D has answered would empty D's inputs. Systems
    that share an instance therefore travel together, and everything else goes
    alone.
    """
    owned: dict[int, set[int]] = {}
    for system in systems:
        for value in vars(system).values():
            if any(value is other for other in systems):
                owned.setdefault(id(system), set()).add(id(value))

    groups: list[list[BenchmarkSystem]] = []
    placed: set[int] = set()
    # Dependants first, so a group is created by the system that needs the others.
    for system in sorted(systems, key=lambda s: -len(owned.get(id(s), ()))):
        if id(system) in placed:
            continue
        group = [system]
        placed.add(id(system))
        for other in systems:
            if id(other) not in placed and id(other) in owned.get(id(system), ()):
                group.append(other)
                placed.add(id(other))
        groups.append(group)
    return groups


def compare(run: BenchmarkRun, left: str, right: str, metric: str,
            family: str | None = None) -> Comparison:
    a, b = paired_values(run.scores, left, right, metric, family)
    return paired_bootstrap(a, b, metric=metric, left_name=left, right_name=right)


def verdict(comparison: Comparison) -> str:
    """One of the three spec section 3 outcomes.

    A difference has to both separate from zero and be large enough to mean
    something. With 149 paired tasks a gap in the fifth decimal separates cleanly,
    and calling that "greater than" would be true and worthless.
    """
    if comparison.negligible or not comparison.separated:
        return SIMILAR
    return GREATER if comparison.difference > 0 else LESS


def chain_verdict(run: BenchmarkRun, metric: str,
                  chain: tuple[str, ...] = HYPOTHESIS_CHAIN
                  ) -> list[tuple[str, str, str, Comparison]]:
    """Each link of the spec section 3 ordering, judged separately."""
    links = []
    for left, right in zip(chain, chain[1:]):
        if left not in run.systems or right not in run.systems:
            continue
        comparison = compare(run, left, right, metric)
        links.append((left, right, verdict(comparison), comparison))
    return links


def report(run: BenchmarkRun, *, metric: str = "efficiency") -> str:
    """The human-readable result. Every number that qualifies a claim is present."""
    lines: list[str] = []
    add = lines.append

    add(f"Benchmark: {run.repo} @ {run.horizon}")
    add(f"  {run.suite.summary()}")
    add(f"  budget {run.budget} tokens/task, K={run.k}, "
        f"{len(run.scores)} scores in {run.seconds:.1f}s")
    prep = ", ".join(f"{name} {seconds:.1f}s"
                     for name, seconds in run.prepare_seconds.items())
    add(f"  index build: {prep}")
    add("")

    add("Per family (spec section 47)")
    add(f"  {'system':<14} {'family':<10} {'n':<5} " +
        " ".join(f"{name}" for name in METRICS))
    for item in run.aggregates():
        add("  " + item.summary())
    add("")

    add("Overall")
    for system in run.systems:
        values = " ".join(
            (f"{name}={run.overall(system, name):.3f}" if name != "tokens"
             else f"tokens={run.overall(system, name):.0f}")
            for name in METRICS)
        add(f"  {system:<14} {values}")
    add("")

    add(f"MCM against each baseline on {metric}")
    reference = HYPOTHESIS_CHAIN[0]
    if reference in run.systems:
        for system in run.systems:
            if system == reference:
                continue
            comparison = compare(run, reference, system, metric)
            add(f"  vs {system:<12} {verdict(comparison)}")
            add(f"    {comparison.summary()}")
    else:
        add("  MCM was not among the systems run")
    add("")

    add(f"Spec section 3 chain on {metric} (paired bootstrap, 95% interval)")
    links = chain_verdict(run, metric)
    if not links:
        add("  not evaluable: the chain's systems were not all run")
    for left, right, outcome, comparison in links:
        add(f"  {left} is {outcome} {right}")
        add(f"    {comparison.summary()}")
    broken = [f"{left} {outcome} {right}"
              for left, right, outcome, _ in links if outcome != GREATER]
    add("")
    if not links:
        add("  Hypothesis: not evaluated.")
    elif broken:
        add(f"  Hypothesis H1 NOT supported on {metric}: " + "; ".join(broken))
    else:
        add(f"  Hypothesis H1 supported on {metric} for this repository.")
    add("")
    thin = sorted({a.family for a in run.aggregates() if a.n < MIN_N})
    if thin:
        add(f"  Under-powered families (n < {MIN_N}): {', '.join(thin)}. Their")
        add("  per-family rows are printed but should not be read as results.")
    add("  One repository is one data point. A chain that holds here is not a")
    add("  result about MCM in general, and the ceiling and rejection counts")
    add("  above bound what these tasks could have shown at all.")
    return "\n".join(lines)


def run_json(run: BenchmarkRun, *, metric: str = "efficiency") -> dict:
    """Machine-readable results, for re-analysis without a re-run."""
    return {
        "repo": run.repo,
        "horizon": run.horizon,
        "budget": run.budget,
        "k": run.k,
        "seconds": round(run.seconds, 2),
        "systems": run.systems,
        "tasks": {
            "generated": len(run.suite.tasks),
            "considered": run.suite.considered,
            "rejected": run.suite.rejected,
            "per_family": {f: len(run.suite.of_family(f))
                           for f in run.suite.families},
            "mean_ceiling": round(
                mean([t.ceiling for t in run.suite.tasks]), 4),
        },
        "aggregates": [
            {"system": a.system, "family": a.family, "n": a.n,
             "values": {k: round(v, 6) for k, v in a.values.items()}}
            for a in run.aggregates()
        ],
        "overall": {
            system: {name: round(run.overall(system, name), 6) for name in METRICS}
            for system in run.systems
        },
        "chain": [
            {"left": left, "right": right, "metric": metric, "verdict": outcome,
             "difference": round(c.difference, 6),
             "interval": [round(c.low, 6), round(c.high, 6)],
             "win_rate": round(c.win_rate, 4), "n": c.n,
             "separated": c.separated}
            for left, right, outcome, c in chain_verdict(run, metric)
        ],
        "mcm_vs_baselines": [
            {"baseline": system, "metric": metric,
             "verdict": verdict(compare(run, HYPOTHESIS_CHAIN[0], system, metric)),
             "difference": round(
                 compare(run, HYPOTHESIS_CHAIN[0], system, metric).difference, 6)}
            for system in run.systems if system != HYPOTHESIS_CHAIN[0]
        ] if HYPOTHESIS_CHAIN[0] in run.systems else [],
        "scores": [
            {"task_id": s.task_id, "family": s.family, "system": s.system,
             "recall": round(s.retrieval.recall, 6),
             "precision": round(s.retrieval.precision, 6),
             "mrr": round(s.retrieval.mrr, 6),
             "useful_tokens": s.efficiency.useful_tokens,
             "total_tokens": s.efficiency.total_tokens,
             "units": s.efficiency.units,
             "efficiency": round(s.efficiency.ratio, 6),
             "ceiling": round(s.ceiling, 6),
             "latency_ms": round(s.latency_ms, 3)}
            for s in run.scores
        ],
    }


def write_json(run: BenchmarkRun, path: Path | str, *,
               metric: str = "efficiency") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run_json(run, metric=metric), indent=2),
                    encoding="utf-8")
    return path
