"""Agent context generation (spec section 33, development step 16).

Spec section 33 gives a task and a pipeline:

    TASK -> relevant objects -> relevant relations -> constraints -> recent
    changes -> historical decisions -> potential impact -> relevant evidence
    -> uncertainties

and asks for a compact context package at the end. This module runs that
pipeline; ``minimisation.py`` is what makes the package compact.

**Everything in the package is a claim with a justification.** A context item that
supports no claim is not context, it is padding, and spec section 34 exists to
remove it. So assembly records, for every statement the package makes, the objects,
relations and evidence that produced it. That record is what the minimiser reduces
and what spec section 43 requires an agent to be able to show.

**There is no LLM here.** Spec section 59 puts the model *after* this step, and the
package is what gets handed to it. Task parsing is hybrid retrieval over the task
text (step 15), not intent classification, and ``candidate_actions`` contains only
actions the semantic model can justify - run these tests, check this constraint,
bisect these commits. Anything else would be the package inventing work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from ..algebra.confidence import confidence_band
from ..algebra.dependency import DependencyPath, dependencies_of
from ..algebra.specs import dependency_edge_types
from ..core.constraints import Constraint, Verdict
from ..core.objects import MCMObject, ObjectType
from ..reasoning.causal_reasoning import RegressionCandidate, regression_candidates
from ..reasoning.constraint_checker import (ConstraintResult, check_constraints,
                                            load_constraints)
from ..reasoning.dependency_propagation import ImpactResult, analyse_impact
from ..retrieval.hybrid import HybridRetriever, RetrievalResult
from ..retrieval.symbolic import resolve_one
from ..storage.database import Store

#: Claim kinds. Each is a different sort of statement with a different warrant,
#: and the package keeps them apart so a reader never has to guess which is which.
IMPACT = "impact"
DEPENDENCY = "dependency"
TEST = "test"
CONSTRAINT = "constraint"
CHANGE = "change"


@dataclass(frozen=True)
class Justification:
    """What a claim rests on. This is the unit spec section 34 minimises."""

    object_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Claim:
    """One statement the package makes, and why it holds.

    ``is_fact`` separates an observed relation from an inference over several
    (spec section 54). The package never flattens the two into "context".
    """

    kind: str
    statement: str
    confidence: float
    justification: Justification
    is_fact: bool = False
    subject_id: str = ""

    @property
    def band(self) -> str:
        return confidence_band(self.confidence)


@dataclass(frozen=True)
class CandidateAction:
    """An action the model can justify, with the claim that warrants it."""

    action: str
    because: str


@dataclass
class ContextPackage:
    """The spec section 33 package.

    Spec section 33's JSON example lists eight fields and its pipeline lists nine
    stages; ``impact`` is the stage the example elides, and it is kept as a field
    of its own because "what could break" is the question the rest of the package
    is assembled to answer.
    """

    goal: str
    focus: MCMObject
    claims: list[Claim] = field(default_factory=list)
    entities: list[MCMObject] = field(default_factory=list)
    constraints: list[ConstraintResult] = field(default_factory=list)
    recent_changes: list[RegressionCandidate] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    actions: list[CandidateAction] = field(default_factory=list)
    retrieved: RetrievalResult | None = None
    minimised: bool = False
    #: Set by the minimiser: what it removed and whether the result verified.
    minimisation: object | None = None

    @property
    def object_ids(self) -> set[str]:
        return {oid for claim in self.claims for oid in claim.justification.object_ids}

    @property
    def relation_ids(self) -> set[str]:
        return {rid for claim in self.claims for rid in claim.justification.relation_ids}

    @property
    def evidence_ids(self) -> set[str]:
        return {eid for claim in self.claims for eid in claim.justification.evidence_ids}

    @property
    def size(self) -> int:
        """|C|: the objects, relations and evidence records an agent would read.

        Provenance is not counted. It is a property of the relation record rather
        than an independent thing to read, and spec section 34 asks for the
        smallest evidence and relationship substructure.
        """
        return len(self.object_ids) + len(self.relation_ids) + len(self.evidence_ids)

    @property
    def gathered_size(self) -> int:
        """Everything the pipeline surfaced, justified or not.

        The denominator of spec section 48's ratio: retrieval hits that no claim
        ever used are counted here and not in ``size``, and the gap between the
        two is what minimisation removes.
        """
        objects = set(self.object_ids) | {self.focus.id}
        if self.retrieved is not None:
            objects |= {c.object_id for c in self.retrieved.candidates}
        return len(objects) + len(self.relation_ids) + len(self.evidence_ids)

    def claims_of(self, kind: str) -> list[Claim]:
        return [claim for claim in self.claims if claim.kind == kind]


def build_context(store: Store, task: str, *, focus: str | None = None,
                  limit: int = 8, max_depth: int = 6, floor: float = 0.0,
                  constraints: list[Constraint] | None = None,
                  as_of: datetime | None = None,
                  retriever: HybridRetriever | None = None) -> ContextPackage:
    """Run the spec section 33 pipeline for a task statement.

    ``focus`` names the object the task is about. Left out, it is the top hybrid
    retrieval result for the task text - which is the honest reading of spec
    section 59's "PARSE TASK -> IDENTIFY OBJECTS" without an LLM in the loop.

    ``floor`` is spec section 34's "acceptable confidence": claims below it are
    not carried. It defaults to 0.0, so nothing is silently dropped, and raising
    it is how a caller trades completeness for certainty.
    """
    retriever = retriever or HybridRetriever(store)
    retrieved = retriever.retrieve(task, limit=max(limit, 10), as_of=as_of)

    target = (resolve_one(store, focus) if focus is not None
              else _focus_from(store, retrieved, task))
    impact = analyse_impact(store, target.id, max_depth=max_depth, as_of=as_of)
    closure = dependencies_of(store, target.id, max_depth=max_depth, as_of=as_of)

    package = ContextPackage(goal=task, focus=target, retrieved=retrieved)
    package.claims.extend(_impact_claims(store, impact))
    package.claims.extend(_dependency_claims(store, target, closure.ordered()))

    scope = {target.id} | package.object_ids | {c.object_id for c in retrieved.candidates}
    package.constraints = _relevant_constraints(store, constraints, scope, as_of)
    package.claims.extend(_constraint_claims(package.constraints))

    package.recent_changes = _recent_changes(store, target.id, max_depth, as_of)
    package.claims.extend(_change_claims(package.recent_changes))

    package.claims = [claim for claim in package.claims if claim.confidence >= floor]
    package.entities = _entities(store, package, retrieved)
    package.uncertainties = _uncertainties(store, package, impact, closure.truncated,
                                           floor,
                                           guessed=_focus_is_a_guess(retrieved, target,
                                                                     explicit=focus))
    package.actions = _actions(package, impact)
    return package


# --- pipeline stages ------------------------------------------------------

def _focus_from(store: Store, retrieved: RetrievalResult, task: str) -> MCMObject:
    """The best-ranked retrieval hit that the reasoning engine can actually work on.

    Relevance alone picks the wrong thing. "Authentication started failing"
    retrieves the commit whose message says *authentication*, and a commit has no
    dependency edges - it participates in TRANSFORMS and PRECEDES - so a package
    focused on one contains no impact, no dependencies and nothing to check.

    The test is therefore structural rather than a list of favoured types: prefer
    the highest-ranked object that takes part in at least one dependency relation.
    A commit stays in the package as a recent change, which is the role it can
    actually fill.
    """
    edge_types = dependency_edge_types()
    fallback: MCMObject | None = None
    for candidate in retrieved.candidates:
        if candidate.object is None:
            continue
        if fallback is None:
            fallback = candidate.object
        if store.relations_for(candidate.object_id, direction="any", types=edge_types):
            return candidate.object
    if fallback is not None:
        return fallback
    raise KeyError(f"nothing in the repository matches {task!r}; "
                   "name a focus explicitly, or ingest the repository first")


def _impact_claims(store: Store, impact: ImpactResult) -> list[Claim]:
    """"Changing X may affect Y", one claim per affected object.

    Depth 1 is an observed relation and says so; beyond that the claim is an
    inference carrying the derived relation's confidence (spec section 54).
    """
    claims = []
    for affected in impact.all_affected:
        is_test = affected.object.type is ObjectType.TEST
        verb = "is exercised by" if is_test else "may be affected by a change to"
        claims.append(Claim(
            kind=TEST if is_test else IMPACT,
            statement=f"{affected.object.name} {verb} {impact.target.name}",
            confidence=affected.confidence,
            justification=_from_path(store, affected.path),
            is_fact=affected.path.depth == 1,
            subject_id=affected.object.id,
        ))
    return claims


def _dependency_claims(store: Store, target: MCMObject,
                       paths: Iterable[DependencyPath]) -> list[Claim]:
    """"X depends on Y". The direction impact analysis does not cover."""
    claims = []
    for path in paths:
        dependency = store.get_object(path.object_id)
        if dependency is None:
            continue
        claims.append(Claim(
            kind=DEPENDENCY,
            statement=f"{target.name} depends on {dependency.name}",
            confidence=path.confidence,
            justification=_from_path(store, path),
            is_fact=path.depth == 1,
            subject_id=dependency.id,
        ))
    return claims


def _relevant_constraints(store: Store, constraints: list[Constraint] | None,
                          scope: set[str],
                          as_of: datetime | None) -> list[ConstraintResult]:
    """Constraints that touch the context, checked against it.

    ``restrict_to`` keeps the check scoped, so a package about authentication does
    not report on every constraint in the repository.
    """
    constraints = constraints if constraints is not None else load_constraints()
    results = check_constraints(store, constraints, as_of=as_of, restrict_to=scope)
    return [result for result in results
            if result.verdict is not Verdict.SATISFIED or result.checked]


def _constraint_claims(results: list[ConstraintResult]) -> list[Claim]:
    """Only violations are claims. A satisfied constraint is context an agent
    should see, but it asserts nothing that needs justifying, and an
    UNEVALUATABLE one is an uncertainty rather than a claim."""
    claims = []
    for result in results:
        if result.verdict is not Verdict.VIOLATED:
            continue
        for violation in result.violations:
            claims.append(Claim(
                kind=CONSTRAINT,
                statement=f"constraint {result.constraint.name} is violated: "
                          f"{violation.message}",
                confidence=1.0,
                justification=Justification(
                    object_ids=tuple(violation.objects),
                    relation_ids=tuple(violation.relations),
                    evidence_ids=tuple(violation.evidence_ids),
                ),
                is_fact=True,
                subject_id=violation.objects[0] if violation.objects else "",
            ))
    return claims


def _recent_changes(store: Store, target_id: str, max_depth: int,
                    as_of: datetime | None) -> list[RegressionCandidate]:
    """Commits that touched anything the focus depends on, most recent first.

    Empty unless Git history has been ingested, which is a fact about the
    repository rather than about the task, and is reported as an uncertainty.
    """
    report = regression_candidates(store, target_id, until=as_of, max_depth=max_depth)
    return report.candidates


def _change_claims(candidates: list[RegressionCandidate]) -> list[Claim]:
    """Correlation, never causation.

    Spec section 37 forbids inferring causality from dependency. These claims say
    a commit touched something the focus depends on, which is all the evidence
    supports, and the score is named accordingly.
    """
    claims = []
    for candidate in candidates:
        # Ordered focus -> ... -> changed object -> commit, so the justification
        # reads as the route that reached the commit. A set union here would put
        # the commit in the middle of the dependency chain.
        path = (_from_path(None, candidate.path) if candidate.path is not None
                else Justification(object_ids=(candidate.changed.id,)))
        justification = Justification(
            object_ids=_ordered(*path.object_ids, candidate.commit.id),
            relation_ids=_ordered(*path.relation_ids, candidate.relation.id),
            evidence_ids=_ordered(*path.evidence_ids,
                                  *candidate.relation.evidence_ids),
        )
        claims.append(Claim(
            kind=CHANGE,
            statement=f"commit {candidate.commit.name} changed {candidate.changed.name}, "
                      f"which the focus depends on (correlation "
                      f"{candidate.correlation:.2f})",
            confidence=candidate.correlation,
            justification=justification,
            is_fact=True,
            subject_id=candidate.commit.id,
        ))
    return claims


def _entities(store: Store, package: ContextPackage,
              retrieved: RetrievalResult) -> list[MCMObject]:
    """Objects the package talks about: the focus, then everything a claim rests
    on, then retrieval hits that no claim reached.

    The third group is what minimisation removes. Keeping it visible before
    minimisation is the point - it is the difference between what was gathered and
    what was justified, which is the ratio spec section 48 is about.
    """
    ordered: list[MCMObject] = [package.focus]
    seen = {package.focus.id}
    for object_id in sorted(package.object_ids):
        if object_id in seen:
            continue
        obj = store.get_object(object_id)
        if obj is not None:
            ordered.append(obj)
            seen.add(object_id)
    for candidate in retrieved.candidates:
        if candidate.object is not None and candidate.object_id not in seen:
            ordered.append(candidate.object)
            seen.add(candidate.object_id)
    return ordered


def _focus_is_a_guess(retrieved: RetrievalResult, focus: MCMObject,
                      explicit: str | None) -> bool:
    """True when nothing but fuzzy similarity picked the focus.

    A score threshold would be a tuned constant, and a constant tuned on the
    handful of queries at hand is exactly what this project refuses to do to the
    retrieval weights. The structural question is answerable without one: did any
    channel other than the vector channel support this object? Lexical and
    symbolic retrieval match terms that are actually present. If neither of them
    said anything, the focus rests on morphological similarity alone, and the
    package should say so whether the task was a reasonable one or nonsense.
    """
    if explicit is not None:
        return False
    for candidate in retrieved.candidates:
        if candidate.object_id == focus.id:
            return candidate.lexical == 0.0 and candidate.symbolic == 0.0
    return True


def _uncertainties(store: Store, package: ContextPackage, impact: ImpactResult,
                   truncated: bool, floor: float, guessed: bool = False) -> list[str]:
    """What the package does not know.

    Spec section 33 asks for this section, and it is the one that keeps the rest
    honest: an agent handed only what MCM is sure of would not know what it is
    missing.
    """
    out: list[str] = []
    if guessed:
        out.append(f"nothing in the repository lexically matches this task; the "
                   f"focus {package.focus.name!r} was chosen by similarity alone "
                   "and may be the wrong object - name one with --focus")

    for result in package.constraints:
        if result.verdict is Verdict.UNEVALUATABLE:
            out.append(f"constraint {result.constraint.name} could not be decided: "
                       f"{result.reason}")

    low = [claim for claim in package.claims if claim.band == "LOW"]
    if low:
        out.append(f"{len(low)} claims are LOW confidence and should not be acted on "
                   "without checking the source")

    inferred = [claim for claim in package.claims if not claim.is_fact]
    if inferred:
        out.append(f"{len(inferred)} of {len(package.claims)} claims are inferences "
                   "over a dependency chain, not observed relations")

    if not any(obj.type is ObjectType.COMMIT for obj in store.find_objects(
            type=ObjectType.COMMIT)[:1]):
        out.append("no Git history is ingested, so recent changes are unknown; "
                   "run `mcm history <path>`")
    elif not package.recent_changes:
        out.append("no commit in the ingested history touched anything the focus "
                   "depends on")

    if not store.find_objects(type=ObjectType.DECISION):
        out.append("no historical decisions are recorded: ingestion creates no "
                   "Decision objects, so the rationale behind this code is absent")

    if truncated or impact.truncated:
        out.append("the dependency search hit its depth bound, so the picture is "
                   "incomplete rather than empty")

    if floor > 0.0:
        out.append(f"claims below confidence {floor} were dropped from this package")
    return out


def _actions(package: ContextPackage, impact: ImpactResult) -> list[CandidateAction]:
    """Actions the model can justify. Nothing here is invented.

    Spec section 37's restraint applies: the change claims name commits to examine
    and tests to run, because running them is the evidence this phase cannot
    gather on its own.
    """
    actions: list[CandidateAction] = []
    tests = [claim for claim in package.claims_of(TEST)]
    if tests:
        names = ", ".join(sorted({claim.statement.split(" is exercised")[0]
                                  for claim in tests}))
        actions.append(CandidateAction(
            action=f"run the covering tests: {names}",
            because=f"{len(tests)} tests reach the focus through the dependency graph",
        ))
    violated = [claim for claim in package.claims_of(CONSTRAINT)]
    for claim in violated:
        actions.append(CandidateAction(
            action=f"resolve or waive: {claim.statement}",
            because="the constraint is violated by the current relation graph",
        ))
    changes = package.claims_of(CHANGE)
    if changes:
        actions.append(CandidateAction(
            action="bisect the correlated commits against the covering tests",
            because=f"{len(changes)} commits touched dependencies of the focus; "
                    "dependency is not causation, so the tests are the evidence",
        ))
    if not impact.all_affected:
        actions.append(CandidateAction(
            action="widen the search or name a different focus",
            because="nothing in the repository depends on the focus, so a change "
                    "here is either safe or the focus is wrong",
        ))
    return actions


def package_json(store: Store, package: ContextPackage) -> dict:
    """The spec section 33 package shape.

    ``impact`` is the ninth pipeline stage that section 33's JSON example elides.
    ``claims`` is not in the example either: it is where the justifications live,
    and without it the package would assert things a reader cannot check.
    """
    report = package.minimisation
    return {
        "goal": package.goal,
        "focus": _object_json(package.focus),
        "entities": [_object_json(obj) for obj in package.entities],
        "dependencies": [_claim_json(c) for c in package.claims_of(DEPENDENCY)],
        "impact": [_claim_json(c) for c in
                   [*package.claims_of(IMPACT), *package.claims_of(TEST)]],
        "constraints": [
            {"name": r.constraint.name, "type": r.constraint.type.value,
             "verdict": r.verdict.value, "reason": r.reason, "checked": r.checked,
             "violations": [v.message for v in r.violations]}
            for r in package.constraints
        ],
        "recent_changes": [
            {"commit": c.commit.name, "sha": c.commit.properties.get("sha"),
             "date": c.commit.properties.get("date"), "changed": c.changed.name,
             "correlation": round(c.correlation, 4), "band": c.band}
            for c in package.recent_changes
        ],
        "evidence": [_evidence_json(store, eid) for eid in sorted(package.evidence_ids)],
        "uncertainties": list(package.uncertainties),
        "candidate_actions": [{"action": a.action, "because": a.because}
                              for a in package.actions],
        "claims": [_claim_json(c) for c in package.claims],
        "size": package.size,
        "minimised": package.minimised,
        "minimisation": None if report is None else {
            "gathered": report.gathered, "justified": report.before,
            "retained": report.after, "efficiency": round(report.efficiency, 4),
            "ablations": report.ablations, "bound_hit": report.bound_hit,
            "already_minimal": report.already_minimal,
            "verified": report.verified, "verification": report.verification,
            "removed": [{"kind": kind, "id": item} for kind, item in report.removed],
        },
    }


def explain(store: Store, package: ContextPackage) -> str:
    """The package as an agent-readable briefing (spec sections 33, 43, 44)."""
    lines = [f'Task: "{package.goal}"',
             f"Focus: {package.focus.name}  [{package.focus.type.value}] "
             f"{package.focus.properties.get('relpath') or ''}".rstrip()]
    report = package.minimisation
    if report is not None:
        lines.append(f"Context: {report.summary()}")
    else:
        lines.append(f"Context: {package.size} items, not minimised")
    lines.append("")

    impact = [*package.claims_of(IMPACT), *package.claims_of(TEST)]
    _section(lines, "Potential impact", impact)
    _section(lines, "Dependencies", package.claims_of(DEPENDENCY))
    _section(lines, "Violated constraints", package.claims_of(CONSTRAINT))
    _section(lines, "Recent changes (correlation, not cause)",
             package.claims_of(CHANGE))

    if package.constraints:
        lines.append("Constraints checked")
        for result in package.constraints:
            lines.append(f"  [{result.verdict.value:13}] {result.constraint.name}"
                         + (f" - {result.reason}" if result.reason else ""))
        lines.append("")

    if package.uncertainties:
        lines.append("Uncertainties")
        lines.extend(f"  - {item}" for item in package.uncertainties)
        lines.append("")

    if package.actions:
        lines.append("Candidate actions")
        for action in package.actions:
            lines.append(f"  - {action.action}")
            lines.append(f"    because: {action.because}")
    return "\n".join(lines)


def _section(lines: list[str], title: str, claims: list[Claim]) -> None:
    if not claims:
        return
    lines.append(f"{title} ({len(claims)})")
    for claim in sorted(claims, key=lambda c: -c.confidence):
        kind = "FACT " if claim.is_fact else "INFER"
        lines.append(f"  [{kind}] {claim.statement}  "
                     f"({claim.confidence:.2f} {claim.band})")
        support = claim.justification
        lines.append(f"          via {' -> '.join(support.object_ids)}"
                     if support.object_ids else "          via (no path)")
        if support.evidence_ids:
            lines.append(f"          evidence: {', '.join(support.evidence_ids)}")
    lines.append("")


def _claim_json(claim: Claim) -> dict:
    return {
        "kind": claim.kind,
        "statement": claim.statement,
        "subject": claim.subject_id,
        "confidence": round(claim.confidence, 4),
        "band": claim.band,
        "type": "fact" if claim.is_fact else "inference",
        "justification": {
            "objects": list(claim.justification.object_ids),
            "relations": list(claim.justification.relation_ids),
            "evidence": list(claim.justification.evidence_ids),
        },
    }


def _evidence_json(store: Store, evidence_id: str) -> dict:
    evidence = store.get_evidence(evidence_id)
    if evidence is None:
        return {"id": evidence_id, "missing": True}
    return {"id": evidence.id, "source_type": evidence.source_type.value,
            "source_ref": evidence.source_ref, "content": evidence.content,
            "confidence": evidence.confidence}


def _object_json(obj: MCMObject) -> dict:
    return {"id": obj.id, "type": obj.type.value, "name": obj.name,
            "relpath": obj.properties.get("relpath"),
            "qualname": obj.properties.get("qualname")}


def _ordered(*ids: str) -> tuple[str, ...]:
    """Deduplicate while keeping first-seen order. Justifications are paths."""
    seen: dict[str, None] = {}
    for item in ids:
        seen.setdefault(item, None)
    return tuple(seen)


def _from_path(store: Store | None, path: DependencyPath) -> Justification:
    """Everything one reasoning path rests on: its objects, its edges, and the
    evidence each edge cites (spec section 43)."""
    return Justification(
        object_ids=tuple(path.objects),
        relation_ids=tuple(relation.id for relation in path.relations),
        evidence_ids=tuple({eid for relation in path.relations
                            for eid in relation.evidence_ids}),
    )
