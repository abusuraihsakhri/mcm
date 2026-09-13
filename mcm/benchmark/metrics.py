"""Benchmark metrics (spec sections 47 and 48).

Spec section 47 asks for Recall@K, Precision@K and MRR on retrieval, and section
48 names context efficiency the most important metric of the study::

    ContextEfficiency = Useful Information / Total Retrieved Information

Two decisions turn that formula into something comparable across architectures,
and both are judgement calls worth stating plainly rather than burying.

**Units are tokens, not items.** A chunk retriever returns 40-line windows; MCM
returns individual definitions and relations. Counting *items* would hand MCM the
metric for free, because thirty facts beats thirty chunks on item count no matter
what is in them. Tokens are what an agent's context window actually spends, so
tokens are the denominator. ``token_estimate`` is used by every system, so any
error in it is common-mode and cancels in the comparison.

**Useful means the part that answers, not the item containing it.** When a system
delivers a 900-token file to convey a 40-token function, the useful information is
40 tokens. This is the reading spec section 48 argues for when it says a system
retrieving 1000 relevant-looking chunks is not necessarily better than one
retrieving 30 explanatory facts. The alternative - crediting the whole containing
item - would make coarse retrieval look efficient precisely when it is not. The
consequence is that efficiency is low for everyone in absolute terms; it is the
ratio *between* systems that carries the claim.

**Statistics.** Spec section 49 step 9 asks for a statistical comparison. Systems
are run on identical tasks, so the comparison is paired, and the test here is a
paired bootstrap over tasks: resample the task list with replacement, recompute the
mean difference, and report the interval. It assumes nothing about the shape of the
score distribution, which matters because these scores are bounded, skewed, and
nothing like normal. No SciPy dependency; the resampling is six lines.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

#: Word, identifier fragment, number, or single punctuation mark.
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\sA-Za-z_0-9]")


def token_estimate(text: str) -> int:
    """Approximate the context cost of a piece of text.

    Not a real BPE tokenizer, and deliberately not one: adding ``tiktoken`` would
    pin the study to one vendor's vocabulary for a number that only ever appears
    in a ratio. Identifiers count as one token here where a BPE tokenizer would
    split ``validate_token`` into several, so absolute efficiency figures run
    optimistic. Every system is measured with this same function, so the bias is
    identical on both sides of every comparison drawn from it.
    """
    return len(_TOKEN.findall(text))


@dataclass(frozen=True)
class RetrievalMetrics:
    """Spec section 47's retrieval block, for one task."""

    k: int
    recall: float
    precision: float
    mrr: float
    hits: int
    targets: int

    @property
    def found_anything(self) -> bool:
        return self.hits > 0


@dataclass(frozen=True)
class ContextEfficiency:
    """Spec section 48, for one task."""

    useful_tokens: int
    total_tokens: int
    units: int

    @property
    def ratio(self) -> float:
        """Useful over total. Zero total means the system returned nothing.

        A system that returns nothing has retrieved no useless information, which
        would be an efficiency of 1.0 under a naive reading. It is scored 0.0:
        efficiency is a property of information delivered, and delivering none is
        a failure to answer rather than a perfectly efficient answer.
        """
        return self.useful_tokens / self.total_tokens if self.total_tokens else 0.0


@dataclass(frozen=True)
class TaskScore:
    """Everything recorded for one system on one task."""

    task_id: str
    family: str
    system: str
    retrieval: RetrievalMetrics
    efficiency: ContextEfficiency
    ceiling: float
    latency_ms: float = 0.0

    def value(self, metric: str) -> float:
        if metric == "recall":
            return self.retrieval.recall
        if metric == "precision":
            return self.retrieval.precision
        if metric == "mrr":
            return self.retrieval.mrr
        if metric == "efficiency":
            return self.efficiency.ratio
        if metric == "tokens":
            return float(self.efficiency.total_tokens)
        if metric == "answered":
            return 1.0 if self.efficiency.units else 0.0
        if metric == "latency":
            return self.latency_ms
        raise ValueError(f"unknown metric {metric!r}")


#: The metrics reported per family.
#:
#: ``tokens`` is not a score to maximise; spec section 47 asks for tokens consumed,
#: and an efficiency ratio means little without the scale it is a ratio of.
#:
#: ``answered`` is the fraction of tasks where the system returned anything at all.
#: Without it a mean of 0.0 is ambiguous between "answered every task wrongly" and
#: "declined most of them", which are different architectures failing in different
#: ways. Spec section 44 asks for confidence-aware answers; a system that returns
#: nothing when it knows nothing is behaving correctly, and averaging that
#: together with wrong answers would hide it.
METRICS = ("recall", "precision", "mrr", "efficiency", "answered", "tokens")


def score_ranking(ranked: list[str], targets: frozenset[str], *,
                  k: int = 10) -> RetrievalMetrics:
    """Recall@K, Precision@K and MRR for one ranked answer.

    ``ranked`` must already be deduplicated and in rank order. Recall is capped
    below 1.0 whenever a task has more targets than K, which is correct rather
    than unfair: a system given ten slots cannot deliver twelve answers.
    """
    if not targets:
        return RetrievalMetrics(k=k, recall=0.0, precision=0.0, mrr=0.0,
                                hits=0, targets=0)
    top = ranked[:k]
    hits = [item for item in top if item in targets]
    mrr = 0.0
    for position, item in enumerate(ranked, start=1):
        if item in targets:
            mrr = 1.0 / position
            break
    return RetrievalMetrics(
        k=k,
        recall=len(hits) / len(targets),
        precision=len(hits) / len(top) if top else 0.0,
        mrr=mrr,
        hits=len(hits),
        targets=len(targets),
    )


#: Below this many tasks, a family mean is printed but flagged as unreadable.
#: Spec section 49 asks for a statistical comparison, and a mean of two tasks is
#: not one. The threshold is arbitrary but fixed in advance, and it suppresses no
#: data: the number is still shown, with a warning attached to it.
MIN_N = 5

#: A difference smaller than this fraction of the compared means is reported as
#: negligible rather than as a win, however cleanly it separates. See
#: ``Comparison.negligible``.
NEGLIGIBLE_FRACTION = 0.01


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class Comparison:
    """A paired bootstrap comparison of two systems on the same tasks."""

    metric: str
    left: str
    right: str
    n: int
    left_mean: float
    right_mean: float
    difference: float
    low: float
    high: float
    #: Fraction of resamples in which left beat right. Reported instead of a
    #: p-value because it answers the question asked: how often does this ordering
    #: survive resampling the tasks.
    win_rate: float

    @property
    def separated(self) -> bool:
        """Whether the interval excludes zero.

        Not significance in the hypothesis-testing sense, and not called that. It
        says the sign of the difference was stable under resampling.
        """
        return (self.low > 0.0) or (self.high < 0.0)

    @property
    def negligible(self) -> bool:
        """Whether the difference is too small to be worth a verdict.

        Separation and size are different questions, and with enough paired tasks
        a difference of 0.00005 separates cleanly. Reporting that as one system
        beating another is technically true and useless: it says the systems
        differ somewhere in the fifth decimal, not that anything was learned.

        The threshold is one percent of the larger mean, so it scales with
        whatever metric is being compared instead of assuming a [0, 1] range.
        """
        scale = max(abs(self.left_mean), abs(self.right_mean))
        return abs(self.difference) < NEGLIGIBLE_FRACTION * scale

    def summary(self) -> str:
        if self.negligible:
            verdict = "negligible"
        else:
            verdict = "separated" if self.separated else "overlapping"
        return (f"{self.left} {self.left_mean:.3f} vs {self.right} "
                f"{self.right_mean:.3f} on {self.metric}: "
                f"diff {self.difference:+.3f} "
                f"[{self.low:+.3f}, {self.high:+.3f}] "
                f"win {self.win_rate:.0%}, n={self.n}, {verdict}")


def paired_bootstrap(left: list[float], right: list[float], *, metric: str = "",
                     left_name: str = "left", right_name: str = "right",
                     resamples: int = 2000, seed: int = 20250911,
                     alpha: float = 0.05) -> Comparison:
    """Bootstrap the mean paired difference between two systems.

    Pairs are held together through resampling. Both systems answered the same
    task, so the per-task difference is the observation and breaking the pair
    would throw away the design's main source of power.

    ``seed`` is fixed so a reported interval can be reproduced exactly.
    """
    if len(left) != len(right):
        raise ValueError(f"unpaired inputs: {len(left)} vs {len(right)}")
    n = len(left)
    if n == 0:
        return Comparison(metric=metric, left=left_name, right=right_name, n=0,
                          left_mean=0.0, right_mean=0.0, difference=0.0,
                          low=0.0, high=0.0, win_rate=0.0)

    differences = [a - b for a, b in zip(left, right)]
    observed = mean(differences)

    rng = random.Random(seed)
    means: list[float] = []
    wins = 0
    for _ in range(resamples):
        sample = [differences[rng.randrange(n)] for _ in range(n)]
        value = mean(sample)
        means.append(value)
        if value > 0.0:
            wins += 1
    means.sort()
    low = means[int(alpha / 2 * resamples)]
    high = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]

    return Comparison(
        metric=metric, left=left_name, right=right_name, n=n,
        left_mean=mean(left), right_mean=mean(right),
        difference=observed, low=low, high=high,
        win_rate=wins / resamples,
    )


@dataclass
class Aggregate:
    """Mean scores for one system over a set of tasks."""

    system: str
    family: str
    n: int
    values: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        parts = " ".join(
            f"{name}={self.values.get(name, 0.0):.3f}"
            if name != "tokens" else f"tokens={self.values.get(name, 0.0):.0f}"
            for name in METRICS)
        thin = "  (n too small to read)" if self.n < MIN_N else ""
        return f"{self.system:<14} {self.family:<10} n={self.n:<3} {parts}{thin}"


def aggregate(scores: list[TaskScore], system: str, family: str) -> Aggregate:
    selected = [s for s in scores if s.system == system and s.family == family]
    return Aggregate(
        system=system, family=family, n=len(selected),
        values={name: mean([s.value(name) for s in selected]) for name in METRICS},
    )


def paired_values(scores: list[TaskScore], left: str, right: str,
                  metric: str, family: str | None = None
                  ) -> tuple[list[float], list[float]]:
    """Line up two systems' scores task by task.

    A task missing from either system is dropped from both rather than filled with
    a zero. Imputing a zero would be inventing an observation, and the bootstrap
    would treat the invention as evidence.
    """
    def index(system: str) -> dict[str, TaskScore]:
        return {s.task_id: s for s in scores
                if s.system == system and (family is None or s.family == family)}

    a, b = index(left), index(right)
    shared = sorted(set(a) & set(b))
    return ([a[t].value(metric) for t in shared],
            [b[t].value(metric) for t in shared])
