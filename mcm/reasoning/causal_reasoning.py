"""Regression candidates (spec sections 37 and 66).

Section 66 asks: "Why did authentication start failing after commit X?" and says
to answer it by combining Git history, changed functions, dependencies, test
failures, temporal relations and causal inference.

This module does the first four and the fifth. It deliberately does not do the
sixth, and that restraint is the point.

Section 37 is explicit: do not infer a causal relation merely because a
dependency exists. "For causal claims require evidence such as: runtime
experiment, test result, commit history, explicit documentation, human
assertion." Commit history alone is one item on that list, and on its own it
supports correlation, not causation. So this module emits no ``CAUSES`` relation.
It ranks *candidates*, scores them by how tightly the target depends on what each
commit changed, and says in its own output that the ranking is correlational.

Turning a candidate into a ``CAUSES`` relation needs a second, independent source
of evidence - a test that passed before the commit and fails after it, or a
person confirming it. That evidence is not available in this phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..algebra.confidence import confidence_band
from ..algebra.dependency import DependencyPath, dependencies_of
from ..core.objects import MCMObject, ObjectType
from ..core.relations import MCMRelation, RelationType as RT
from ..storage.database import Store


@dataclass
class RegressionCandidate:
    """A commit that changed something the target depends on."""

    commit: MCMObject
    changed: MCMObject
    #: How the target reaches the changed object. None when the target itself
    #: was changed, which is depth 0 and needs no path.
    path: DependencyPath | None
    relation: MCMRelation
    #: Strength of the dependency between target and changed object. NOT a
    #: probability that this commit caused the failure (spec section 37).
    correlation: float

    @property
    def date(self) -> datetime:
        return datetime.fromisoformat(self.commit.properties["date"])

    @property
    def depth(self) -> int:
        return self.path.depth if self.path else 0

    @property
    def band(self) -> str:
        return confidence_band(self.correlation)

    @property
    def subject(self) -> str:
        return self.commit.properties.get("subject", "")


@dataclass
class RegressionReport:
    target: MCMObject
    candidates: list[RegressionCandidate] = field(default_factory=list)
    since: datetime | None = None
    truncated: bool = False
    #: Tests that cover the target, which is what would settle the question.
    covering_tests: list[MCMObject] = field(default_factory=list)


def regression_candidates(store: Store, target_id: str, *, since: datetime | None = None,
                          until: datetime | None = None,
                          max_depth: int = 6) -> RegressionReport:
    """Commits that changed anything the target depends on, best candidate first.

    The search runs over the target's *dependencies*, not its dependents. If
    ``authenticate`` started failing, the cause lies in what it relies on.

    ``since`` is normally the last commit known to work, which narrows the
    candidates to changes that landed after it.
    """
    target = store.get_object(target_id)
    if target is None:
        raise KeyError(f"unknown object: {target_id}")

    closure = dependencies_of(store, target_id, max_depth=max_depth, as_of=until)
    reachable: dict[str, DependencyPath | None] = {target_id: None}
    reachable.update(closure.paths)

    report = RegressionReport(target=target, since=since, truncated=closure.truncated)
    for object_id, path in reachable.items():
        changed = store.get_object(object_id)
        if changed is None:
            continue
        for relation in store.relations_for(object_id, direction="in",
                                            types=[RT.TRANSFORMS]):
            commit = store.get_object(relation.arguments[0])
            if commit is None or commit.type is not ObjectType.COMMIT:
                continue
            candidate = RegressionCandidate(
                commit=commit, changed=changed, path=path, relation=relation,
                correlation=path.confidence if path else 1.0,
            )
            if since is not None and candidate.date <= since:
                continue
            if until is not None and candidate.date > until:
                continue
            report.candidates.append(candidate)

    # Most recent first, then tightest dependency. Recency leads because a
    # regression is bounded by when it appeared, and the dependency score
    # separates commits that landed in the same window.
    report.candidates.sort(key=lambda c: (c.date, c.correlation, c.changed.id),
                           reverse=True)
    report.covering_tests = _covering_tests(store, target_id, max_depth)
    return report


def _covering_tests(store: Store, target_id: str, max_depth: int) -> list[MCMObject]:
    from ..algebra.dependency import dependents_of

    closure = dependents_of(store, target_id, max_depth=max_depth)
    tests = []
    for object_id in closure.paths:
        obj = store.get_object(object_id)
        if obj is not None and obj.type is ObjectType.TEST:
            tests.append(obj)
    return sorted(tests, key=lambda o: o.id)


def explain(store: Store, report: RegressionReport, limit: int = 8) -> str:
    """Render the ranked candidates and their evidence (spec section 43)."""
    lines: list[str] = []
    target = report.target
    lines.append(f"Regression candidates for {target.name}  [{target.type.value}]")
    lines.append(f"  {target.id}")
    if report.since is not None:
        lines.append(f"  restricted to changes after {report.since.isoformat()}")
    lines.append("")

    if not report.candidates:
        lines.append("No commit in range changed anything this object depends on.")
        return "\n".join(lines)

    for rank, candidate in enumerate(report.candidates[:limit], start=1):
        lines.append(f"{rank}. {candidate.commit.name}  {candidate.subject}")
        lines.append(f"     {candidate.commit.properties.get('author')}  "
                     f"{candidate.date.date().isoformat()}")
        lines.append(f"     changed: {candidate.changed.name} "
                     f"({candidate.changed.properties.get('relpath', '')})")
        if candidate.path is None:
            lines.append("     the target itself was changed by this commit")
        else:
            chain = " <- ".join(_display(store, oid)
                                for oid in reversed(candidate.path.objects))
            lines.append(f"     {target.name} depends on it: {chain}")
            lines.append(f"     via {'/'.join(candidate.path.edge_types)}, "
                         f"dependency strength {candidate.correlation:.2f} "
                         f"({candidate.band})")
        for evidence_id in candidate.relation.evidence_ids:
            evidence = store.get_evidence(evidence_id)
            if evidence is not None:
                lines.append(f"     evidence: {evidence.source_ref} "
                             f"[{evidence.source_type.value}] {evidence.content}")
        lines.append("")

    if len(report.candidates) > limit:
        lines.append(f"... and {len(report.candidates) - limit} more")
        lines.append("")

    if report.covering_tests:
        names = ", ".join(t.name for t in report.covering_tests)
        lines.append(f"To settle this, bisect with: {names}")
    else:
        lines.append("No test covers this object, so no automatic check can "
                     "confirm which candidate is responsible.")
    lines.append("")
    lines.append("These are correlations, not causes. A commit is listed because it "
                 "changed something")
    lines.append("the target depends on, which is not evidence that it caused a "
                 "failure (spec section 37).")
    return "\n".join(lines)


def _display(store: Store, object_id: str) -> str:
    obj = store.get_object(object_id)
    return obj.name if obj is not None else object_id
