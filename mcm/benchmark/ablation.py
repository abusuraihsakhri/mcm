"""Component ablations (spec section 50).

Spec section 50 lists seven configurations::

    MCM, MCM - graph, MCM - vector, MCM - symbolic reasoning,
    MCM - provenance, MCM - temporal, MCM - constraints

and gives the reason for running them: "This determines which components actually
provide value." An ablation answers a different question from a baseline. A
baseline asks whether a rival architecture does better; an ablation asks whether a
part of this one is earning its place. Removing a channel from MCM leaves MCM with
a channel missing, which is not a vector RAG system and must not be reported as
one - the distinction ``systems.py`` exists to preserve.

**Four of the seven are real here, and three are not.** The benchmark measures a
retrieval and impact path over a single ingested revision. Temporal reasoning and
constraint checking do not participate in that path, so removing them would change
nothing and the resulting "no effect" would be an artefact of what is measured
rather than a finding about the component. They are reported as not exercised.
Printing a null result for them would be the more impressive-looking option and
the dishonest one.

Spec section 50's "symbolic reasoning" is ambiguous between the exact-resolution
retrieval channel and the inference engine on top of the graph. Both are ablated,
separately and under distinct names, because they are different components and
collapsing them would lose whichever one matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..retrieval.hybrid import RetrievalWeights
from .metrics import METRICS, Comparison, paired_bootstrap, paired_values
from .runner import BenchmarkRun
from .systems import BenchmarkSystem, MCMSystem

#: Components spec section 50 names that this benchmark's task families never
#: exercise, with the reason. Reported rather than silently skipped.
NOT_EXERCISED = {
    "temporal": (
        "the benchmark ingests one revision, so no relation has a validity "
        "interval to ignore; removing temporal reasoning changes nothing this "
        "harness can observe"),
    "constraints": (
        "no task family asks whether a change violates a requirement (spec "
        "section 46 E), so the constraint checker is never called"),
}


@dataclass(frozen=True)
class Ablation:
    """One configuration of MCM with a component removed."""

    name: str
    description: str
    build: Callable[[], BenchmarkSystem]

    def system(self) -> BenchmarkSystem:
        return self.build()


def _weights(**overrides: float) -> RetrievalWeights:
    return RetrievalWeights(**overrides)


def ablations() -> list[Ablation]:
    """The spec section 50 line-up that this harness can actually run."""
    return [
        Ablation("mcm", "the full system, as the baseline for every row below",
                 lambda: MCMSystem(name="mcm")),
        Ablation("mcm-no-graph",
                 "graph channel off: relevance by proximity to an anchor removed",
                 lambda: MCMSystem(name="mcm-no-graph",
                                   weights=_weights(graph=0.0))),
        Ablation("mcm-no-vector",
                 "vector channel off: embedding similarity removed",
                 lambda: MCMSystem(name="mcm-no-vector",
                                   weights=_weights(vector=0.0))),
        Ablation("mcm-no-symbolic",
                 "symbolic channel off: exact name resolution removed",
                 lambda: MCMSystem(name="mcm-no-symbolic",
                                   weights=_weights(symbolic=0.0))),
        Ablation("mcm-no-provenance",
                 "provenance channel off: extractor trust no longer ranks",
                 lambda: MCMSystem(name="mcm-no-provenance",
                                   weights=_weights(provenance=0.0))),
        Ablation("mcm-no-reasoning",
                 "inference off: impact answered by retrieval, not propagation",
                 lambda: MCMSystem(name="mcm-no-reasoning", use_reasoning=False)),
    ]


def ablation_systems() -> list[BenchmarkSystem]:
    return [item.system() for item in ablations()]


def contributions(run: BenchmarkRun, metric: str = "efficiency",
                  reference: str = "mcm") -> list[Comparison]:
    """What each removed component was worth, as full-minus-ablated.

    A positive difference means the full system beat the ablated one, so the
    component contributed. A negative difference means removing it *helped*, which
    is a real possible outcome and the reason spec section 50 exists: a component
    that costs accuracy should be found out rather than assumed useful.
    """
    found: list[Comparison] = []
    for system in run.systems:
        if system == reference:
            continue
        left, right = paired_values(run.scores, reference, system, metric)
        found.append(paired_bootstrap(left, right, metric=metric,
                                      left_name=reference, right_name=system))
    return sorted(found, key=lambda c: -c.difference)


def report(run: BenchmarkRun, *, metric: str = "efficiency",
           reference: str = "mcm") -> str:
    """Ranked component contributions, plus what could not be tested."""
    lines: list[str] = []
    add = lines.append
    described = {a.name: a.description for a in ablations()}

    add(f"Ablations (spec section 50): {run.repo} @ {run.horizon}")
    add(f"  {len(run.suite.tasks)} tasks, budget {run.budget} tokens, K={run.k}")
    add("")

    add("Scores")
    for system in run.systems:
        values = " ".join(
            (f"{name}={run.overall(system, name):.3f}" if name != "tokens"
             else f"tokens={run.overall(system, name):.0f}")
            for name in METRICS)
        add(f"  {system:<20} {values}")
    add("")

    add(f"Contribution of each component on {metric} "
        f"(full minus ablated, 95% interval)")
    for comparison in contributions(run, metric=metric, reference=reference):
        if comparison.negligible:
            worth = "no measurable effect"
        elif comparison.difference > 0:
            worth = "contributes" if comparison.separated else "contributes (unresolved)"
        elif comparison.difference < 0:
            worth = "costs" if comparison.separated else "costs (unresolved)"
        else:
            worth = "no effect"
        add(f"  {comparison.right:<20} {worth}")
        add(f"    {comparison.summary()}")
        note = described.get(comparison.right)
        if note:
            add(f"    {note}")
    add("")

    add("Not exercised by this benchmark")
    for component, reason in NOT_EXERCISED.items():
        add(f"  {component}: {reason}")
    add("")
    add("  An interval that includes zero means this task set could not resolve")
    add("  the component's value, not that the component is worthless.")
    return "\n".join(lines)
