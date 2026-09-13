"""Prediction against observation (spec sections 61 and 62, development step 18).

    compare Predicted(Δ) against Observed(Δ), calculate prediction error, and
    update confidence.

**Where observations come from.** Spec section 61 lists the actual diff, the actual
test result and the actual runtime behaviour. Only the first exists here: Git
history, already ingested commit by commit, records which definitions each commit
changed and - since step 18 - what kind of change each one was. So an observation is
a commit, and the prediction is made against the repository as it stood one instant
before that commit landed.

There is no test runner and no agent execution, so two of section 61's three
channels are absent. That is not a detail: it decides what this evaluation can and
cannot measure, which the next paragraph is about.

**Co-change measures edits, not behaviour.** If a commit changes A and also changes
B, that is evidence B had to be edited. It is *not* evidence that B's behaviour
changed, and the absence of an edit to B is not evidence that B was unaffected -
B may have kept working, or the work may have been deferred to a later commit.

This maps onto the two consequences from spec section 30 very unevenly:

* ``MUST_UPDATE`` predicts a source edit, and a commit records source edits. The
  comparison is sound: rename a function and the commit that renames it must fix
  its callers or ship broken code.
* ``MAY_DIFFER`` predicts a behaviour change, and no diff can confirm or deny one.
  A MAY_DIFFER "false alarm" is not evidence the prediction was wrong. It is
  evidence that this observation channel cannot see the thing being predicted.

So the report scores them separately and refuses to combine them. Folding both into
one accuracy number would produce a figure that looks like a measurement and is not
one, which spec section 64 is largely about avoiding.

**What gets updated.** Spec section 62 says to reduce confidence in ``A → B`` when
changing A does not affect B. In this system that number is not one number.
``relation.confidence`` is the belief that the edge is real - an AST-observed call
is certain, and no amount of co-change data makes it less so. What the evidence
actually bears on is ``EDGE_DECAY``: how much of a change survives crossing an edge
of that type. ``confidence.py`` already separates belief from attenuation and says
why. This module learns the second and never touches the first.

Nothing is applied. The learned table is returned for inspection beside the
hand-set one, because silently retuning the constants that every impact answer
depends on, from a handful of commits, is the sort of thing that makes a benchmark
unfalsifiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..algebra.confidence import DEFAULT_EDGE_DECAY, EDGE_DECAY
from ..algebra.dependency import dependencies_of
from ..core.change import ADDED, COSMETIC, Change, ChangeKind
from ..core.objects import ObjectType
from ..core.relations import RelationType as RT
from ..storage.database import Store
from .change_propagation import Consequence, propagate

#: How far before a commit's timestamp to read the repository. The relations a
#: commit opens are valid from its exact date, so stepping back one microsecond
#: gives the state its author was looking at.
_INSTANT = timedelta(microseconds=1)

#: Pseudo-count behind the section 62 update. The hand-set EDGE_DECAY value acts as
#: a prior worth this many observations, so a single surprising commit nudges the
#: rate instead of replacing it. Transparent by construction: at zero observations
#: the posterior *is* the prior, and it takes this many agreeing observations to
#: move halfway to the observed rate.
PRIOR_STRENGTH = 5.0


@dataclass(frozen=True)
class Observation:
    """What one commit actually did to one definition."""

    commit_id: str
    commit_name: str
    date: datetime
    target_id: str
    kind: ChangeKind
    #: Other definitions the same commit edited, excluding additions.
    co_changed: frozenset[str]
    additions: frozenset[str] = frozenset()
    #: Edits that changed text without changing the parse tree. Excluded for the
    #: same reason as additions: a reformat propagates nowhere, so treating one as
    #: a co-change would score the formatter rather than the model.
    cosmetic: frozenset[str] = frozenset()

    @property
    def before(self) -> datetime:
        return self.date - _INSTANT


@dataclass
class Comparison:
    """Predicted(Δ) against Observed(Δ) for one observation."""

    observation: Observation
    must_update: frozenset[str]
    may_differ: frozenset[str]
    #: Co-changed objects the target *depends on*, rather than the reverse.
    upstream: frozenset[str] = frozenset()

    @property
    def predicted(self) -> frozenset[str]:
        return self.must_update | self.may_differ

    @property
    def edit_hits(self) -> frozenset[str]:
        return self.must_update & self.observation.co_changed

    @property
    def edit_false_alarms(self) -> frozenset[str]:
        return self.must_update - self.observation.co_changed

    @property
    def missed(self) -> frozenset[str]:
        """Edited alongside, reached by nothing.

        Counted against the whole prediction rather than against MUST_UPDATE
        alone: an object the system placed in MAY_DIFFER and that did in fact get
        edited was at least reached, and calling that a miss would be scoring the
        traversal for something the label got wrong.

        Every miss is by construction an object with no dependency path *from* the
        change site, since anything reachable is predicted. So this measures graph
        coverage, not propagation labelling, and it has two quite different causes
        - separated by ``upstream_missed`` and ``unexplained_missed``.
        """
        return self.observation.co_changed - self.predicted

    @property
    def upstream_missed(self) -> frozenset[str]:
        """Misses that are upstream of the change site.

        Not a failure of anything. Co-change is symmetric and propagation is
        directional: when one commit changes a callee and its caller, the callee's
        observation records the caller as co-changed and predicts it, while the
        caller's observation records the callee and cannot - nothing downstream of
        the caller is the callee. Scoring these as errors would penalise the model
        for not running backwards.
        """
        return self.missed & self.upstream

    @property
    def unexplained_missed(self) -> frozenset[str]:
        """Misses with no dependency path in either direction.

        These are the interesting ones. Either the two edits were unrelated - one
        commit doing two jobs - or the extractor never saw the edge that connects
        them, which is a gap in the graph rather than in the propagation model.
        """
        return self.missed - self.upstream


@dataclass
class EdgeTally:
    """Per relation type: how often a predicted edit was actually made."""

    predicted_edits: int = 0
    confirmed_edits: int = 0
    predicted_behaviour: int = 0
    co_changed_behaviour: int = 0

    @property
    def observed_rate(self) -> float | None:
        if not self.predicted_edits:
            return None
        return self.confirmed_edits / self.predicted_edits


@dataclass
class AccuracyReport:
    observations: list[Observation] = field(default_factory=list)
    comparisons: list[Comparison] = field(default_factory=list)
    by_edge: dict[RT, EdgeTally] = field(default_factory=dict)
    additions_excluded: int = 0
    #: Observations whose target is not an object in the store. A definition can
    #: be visible to the differ and absent from the graph -- a class nested in a
    #: function body, for one -- and that is a gap in extraction rather than a
    #: prediction that was wrong. Counted, not silently dropped, and not scored.
    unresolved_targets: int = 0

    @property
    def edit_precision(self) -> float | None:
        """Of the edits predicted, how many were made.

        ``None`` rather than 1.0 when nothing was predicted: a system that predicts
        nothing has not achieved perfect precision, it has abstained.
        """
        predicted = sum(len(c.must_update) for c in self.comparisons)
        if not predicted:
            return None
        return sum(len(c.edit_hits) for c in self.comparisons) / predicted

    @property
    def edit_recall(self) -> float | None:
        """Of the edits made alongside, how many were predicted as required."""
        observed = sum(len(c.observation.co_changed) for c in self.comparisons)
        if not observed:
            return None
        return sum(len(c.edit_hits) for c in self.comparisons) / observed

    @property
    def reach_recall(self) -> float | None:
        """Of the edits made alongside, how many were reached at all.

        Separated from ``edit_recall`` because reaching an object and labelling it
        correctly are different failures with different fixes: one is the
        traversal, the other is the propagation table.
        """
        observed = sum(len(c.observation.co_changed) for c in self.comparisons)
        if not observed:
            return None
        hits = sum(len(c.predicted & c.observation.co_changed)
                   for c in self.comparisons)
        return hits / observed

    @property
    def usable(self) -> bool:
        """Whether any observation could test an edit prediction at all.

        A corpus of pure behaviour changes cannot, however many commits it has.
        """
        return any(c.must_update or c.observation.co_changed
                   for c in self.comparisons)


def observations_from_history(store: Store, *,
                              since: datetime | None = None) -> list[Observation]:
    """Every commit-level change to an existing definition, oldest first.

    Additions are excluded as targets: nothing can depend on a definition that did
    not exist at the previous revision, so there is no prediction to test. They are
    also excluded from the co-changed set, because an object with no relations in
    the earlier state could not have been predicted by any traversal, and counting
    it as a miss would measure the wrong thing. The count is reported so the
    exclusion is visible rather than quietly improving the numbers.
    """
    out: list[Observation] = []
    for commit in store.find_objects(type=ObjectType.COMMIT):
        date = _commit_date(commit)
        if date is None or (since is not None and date <= since):
            continue
        edits: dict[str, str] = {}
        for relation in store.relations_for(commit.id, direction="out",
                                            types=[RT.TRANSFORMS],
                                            include_historical=True):
            target = relation.arguments[1]
            kind = relation.properties.get("change_kind")
            if kind is None:  # a file-level TRANSFORMS carries no definition kind
                continue
            edits[target] = kind

        ignored = {ADDED, COSMETIC}
        changed = {oid for oid, kind in edits.items() if kind not in ignored}
        added = {oid for oid, kind in edits.items() if kind == ADDED}
        cosmetic = {oid for oid, kind in edits.items() if kind == COSMETIC}
        for target_id, kind in sorted(edits.items()):
            if kind in ignored:
                continue
            out.append(Observation(
                commit_id=commit.id, commit_name=commit.name, date=date,
                target_id=target_id, kind=ChangeKind(kind),
                co_changed=frozenset(changed - {target_id}),
                additions=frozenset(added),
                cosmetic=frozenset(cosmetic),
            ))
    out.sort(key=lambda o: (o.date, o.target_id))
    return out


def evaluate(store: Store, observations: list[Observation] | None = None, *,
             max_depth: int = 6) -> AccuracyReport:
    """Compare what the model would have predicted against what happened."""
    observations = (observations if observations is not None
                    else observations_from_history(store))
    report = AccuracyReport(observations=list(observations))

    for observation in observations:
        report.additions_excluded += len(observation.additions)
        try:
            result = propagate(store, Change(observation.target_id, observation.kind),
                               max_depth=max_depth, as_of=observation.before)
            closure = dependencies_of(store, observation.target_id,
                                      max_depth=max_depth, as_of=observation.before)
        except KeyError:
            # The differ saw a definition the extractor never turned into an
            # object. Scoring it either way would be wrong: there is no model
            # here to be right or mistaken.
            report.unresolved_targets += 1
            continue
        comparison = Comparison(
            observation=observation,
            must_update=frozenset(p.object.id for p in result.must_update),
            may_differ=frozenset(p.object.id for p in result.may_differ),
            upstream=frozenset(closure.paths) & observation.co_changed,
        )
        report.comparisons.append(comparison)

        for predicted in result.predicted:
            edge = predicted.path.relations[0].relation_type
            tally = report.by_edge.setdefault(edge, EdgeTally())
            edited = predicted.object.id in observation.co_changed
            if predicted.consequence is Consequence.MUST_UPDATE:
                tally.predicted_edits += 1
                tally.confirmed_edits += int(edited)
            else:
                tally.predicted_behaviour += 1
                tally.co_changed_behaviour += int(edited)
    return report


def learned_decay(report: AccuracyReport) -> dict[RT, tuple[float, float, int]]:
    """Spec section 62's update, as a transparent heuristic.

        Confidence_{t+1} = (confirmed + k * prior) / (predicted + k)

    A Beta-Binomial posterior mean with the hand-set ``EDGE_DECAY`` value as the
    prior and ``PRIOR_STRENGTH`` as its weight in observations. Two properties make
    it honest to publish: with no observations the posterior equals the prior
    exactly, and a single commit cannot move a constant that every impact answer
    depends on.

    Returns ``edge -> (prior, posterior, observations)``. Nothing is applied. The
    point of computing it is to see whether the constants guessed in
    ``confidence.py`` survive contact with a real repository, and that comparison is
    only worth making on a corpus far larger than any this prototype has been run
    against.
    """
    out: dict[RT, tuple[float, float, int]] = {}
    for edge, tally in sorted(report.by_edge.items(), key=lambda item: item[0].value):
        if not tally.predicted_edits:
            continue
        prior = EDGE_DECAY.get(edge, DEFAULT_EDGE_DECAY)
        posterior = ((tally.confirmed_edits + PRIOR_STRENGTH * prior)
                     / (tally.predicted_edits + PRIOR_STRENGTH))
        out[edge] = (prior, posterior, tally.predicted_edits)
    return out


def explain(report: AccuracyReport) -> str:
    """Render the comparison, with what it cannot measure stated first."""
    lines = [f"Observations: {len(report.observations)} commit-level changes to "
             "existing definitions"]
    if report.unresolved_targets:
        lines.append(
            f"  {report.unresolved_targets} observations skipped: the differ saw a "
            f"definition the\n    extractor never made an object, so there is no "
            f"prediction to score")
    if report.additions_excluded:
        lines.append(f"  {report.additions_excluded} added definitions excluded: "
                     "nothing could depend on them yet")
    lines.append("")

    if not report.observations:
        lines.append("No history is ingested, so there is nothing to compare "
                     "against. Run `mcm history <path>` first.")
        return "\n".join(lines)

    if not report.usable:
        lines.append("Every observed change is a behaviour change with no "
                     "co-edits, so no edit prediction can be tested.")
        lines.append("A diff records edits; it cannot confirm or deny a behaviour "
                     "change. That needs test results, which this phase does not")
        lines.append("collect - so the numbers below would measure nothing and "
                     "are not reported.")
        return "\n".join(lines)

    lines.append("Edit predictions (MUST_UPDATE) - testable against a diff")
    lines.append(f"  precision   {_pct(report.edit_precision)}"
                 "   of predicted edits, how many were made")
    lines.append(f"  recall      {_pct(report.edit_recall)}"
                 "   of edits made alongside, how many were predicted")
    lines.append(f"  reached     {_pct(report.reach_recall)}"
                 "   of edits made alongside, how many were reached at all")
    lines.append("")

    upstream = sum(len(c.upstream_missed) for c in report.comparisons)
    unexplained = sum(len(c.unexplained_missed) for c in report.comparisons)
    if upstream or unexplained:
        lines.append("Edits alongside that nothing reached")
        if upstream:
            lines.append(f"  {upstream} upstream of the change: co-change is "
                         "symmetric, propagation is not, so these")
            lines.append("    are not errors - a callee cannot predict its own "
                         "caller's callee")
        if unexplained:
            lines.append(f"  {unexplained} with no dependency path either way: "
                         "unrelated work in the same commit,")
            lines.append("    or an edge the extractor never saw")
        lines.append("")

    behaviour = sum(t.predicted_behaviour for t in report.by_edge.values())
    if behaviour:
        lines.append(f"Behaviour predictions (MAY_DIFFER): {behaviour} made, "
                     "0 testable.")
        lines.append("  A commit records edits, not behaviour. These are neither "
                     "confirmed nor refuted here.")
        lines.append("")

    learned = learned_decay(report)
    if learned:
        lines.append("Section 62 update - propagation strength, not relation belief")
        lines.append(f"  {'edge':14} {'prior':>7} {'posterior':>10} {'n':>4}")
        for edge, (prior, posterior, count) in learned.items():
            lines.append(f"  {edge.value:14} {prior:>7.2f} {posterior:>10.2f} "
                         f"{count:>4}")
        lines.append("")

    lines.append(f"Sample size: {len(report.comparisons)}. "
                 "Nothing here is a result.")
    lines.append("A repository this size cannot separate a good propagation model "
                 "from a lucky one;")
    lines.append("the evaluation framework in sections 45 to 50 is what would. "
                 "Nothing was applied.")
    return "\n".join(lines)


def accuracy_json(report: AccuracyReport) -> dict:
    return {
        "mode": "accuracy",
        "observations": len(report.observations),
        "additions_excluded": report.additions_excluded,
        "unresolved_targets": report.unresolved_targets,
        "usable": report.usable,
        "edit_precision": report.edit_precision,
        "edit_recall": report.edit_recall,
        "reach_recall": report.reach_recall,
        "by_edge": {
            edge.value: {
                "predicted_edits": tally.predicted_edits,
                "confirmed_edits": tally.confirmed_edits,
                "predicted_behaviour": tally.predicted_behaviour,
                "observed_rate": tally.observed_rate,
            }
            for edge, tally in sorted(report.by_edge.items(),
                                      key=lambda item: item[0].value)
        },
        "learned_decay": {
            edge.value: {"prior": prior, "posterior": posterior,
                         "observations": count}
            for edge, (prior, posterior, count) in learned_decay(report).items()
        },
        "applied": False,
        "comparisons": [
            {
                "commit": c.observation.commit_name,
                "target": c.observation.target_id,
                "kind": c.observation.kind.value,
                "must_update": sorted(c.must_update),
                "may_differ": sorted(c.may_differ),
                "co_changed": sorted(c.observation.co_changed),
                "edit_hits": sorted(c.edit_hits),
                "edit_false_alarms": sorted(c.edit_false_alarms),
                "missed": sorted(c.missed),
                "upstream_missed": sorted(c.upstream_missed),
                "unexplained_missed": sorted(c.unexplained_missed),
            }
            for c in report.comparisons
        ],
    }


def _commit_date(commit) -> datetime | None:
    raw = commit.properties.get("date")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _pct(value: float | None) -> str:
    return "    n/a" if value is None else f"{value:>7.2f}"
