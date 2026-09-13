"""Change propagation (spec section 30, development step 17).

    X_{t+1} = F(X_t, Δ)        estimate  ΔA ⇒ {ΔB, ΔC, ΔD}

Impact analysis (spec section 29) answers "what could break if A changes" with the
change left opaque, so every dependent comes back wearing the same label. That is
the right answer to a question nobody asks: an agent does not propose "something
about `validate_token` will be different", it proposes a rename, or a new
parameter, or a different result for the same inputs, and those have different
consequences for different dependents.

This module takes a typed ``Change`` and returns, for each affected object, what
*kind* of consequence follows and whether its source actually has to be edited.

**The propagation table is the whole idea.** Which dependents must be edited is a
function of the change kind and of the relation type that reaches them, and both
are declared rather than inferred from the shape of the data - the same discipline
`RELATION_SPECS` applies to traversal. Renaming a function forces an edit at every
site that names it, including the import statement. Adding a parameter forces an
edit only where it is *called*: a module that imports it and never calls it is
untouched, and the table says so rather than a chain of conditionals saying so.

**Edits do not propagate; behaviour does.** A caller of a caller of a renamed
function has no reference to the old name anywhere in its source. Its behaviour may
still change, so it is reported - at the attenuated confidence the dependency
algebra gives it - as MAY_DIFFER rather than as work to do. Conflating the two is
what turns an impact report into a list an agent cannot act on.

Nothing here is persisted. A predicted change is an inference about a future state
(spec Rule 2), and comparing it against what actually happened is spec section 61,
which needs observations this phase cannot gather.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..algebra.confidence import confidence_band
from ..algebra.dependency import DependencyPath
from ..core.change import Change, ChangeKind
from ..core.objects import MCMObject, ObjectType
from ..core.relations import RelationType as RT
from ..storage.database import Store
from .dependency_propagation import AffectedObject, ImpactResult, analyse_impact


class Consequence(str, Enum):
    """What a change does to one dependent."""

    #: Its source has to be edited before it works again.
    MUST_UPDATE = "MUST_UPDATE"
    #: Its source is still valid; what it computes may change. Tests are the
    #: evidence, which is why the report ends by naming them.
    MAY_DIFFER = "MAY_DIFFER"


@dataclass(frozen=True)
class EdgeResponse:
    """How one relation type transmits a change to the object at its other end."""

    #: Does a REMOVE or RENAME force an edit here? True when the edge means the
    #: dependent's source contains the target's name.
    on_reference_break: bool
    #: Does a SIGNATURE change force an edit here? True when the edge means the
    #: dependent's source contains a *call* with an argument list.
    on_call_break: bool
    why: str


#: The propagation table (spec section 30).
#:
#: Read it as: "if the dependent reaches the change site by this edge, does this
#: class of change land in its source?" Every entry is a claim about what the edge
#: means, and each one is written down so it can be argued with.
EDGE_RESPONSES: dict[RT, EdgeResponse] = {
    RT.CALLS: EdgeResponse(
        True, True, "the call site names the callee and passes its arguments"),
    RT.USES: EdgeResponse(
        True, True, "the use site names the target and may pass arguments to it"),
    RT.TESTS: EdgeResponse(
        True, True, "a test names its subject and calls it"),
    RT.INHERITS: EdgeResponse(
        True, True, "a subclass names its base, and an override carries the "
                    "signature it overrides"),
    RT.IMPLEMENTS: EdgeResponse(
        True, True, "an implementation names its interface and matches its shape"),
    RT.READS: EdgeResponse(
        True, False, "a read names the target but does not call it, so a "
                     "signature change does not reach it"),
    RT.IMPORTS: EdgeResponse(
        True, False, "the import statement names the module; it says nothing "
                     "about any signature inside it"),
}

#: Used for a dependency edge with no entry above.
#:
#: Deliberately claims nothing. Reporting MUST_UPDATE for an edge whose meaning is
#: not modelled would be asserting that the dependent's source names the target,
#: which is exactly what is not known. The dependent still appears in the report as
#: MAY_DIFFER, so it is never silently dropped - this is the same stance the
#: constraint engine takes when it refuses to return SATISFIED for something it did
#: not check.
DEFAULT_EDGE_RESPONSE = EdgeResponse(
    False, False, "this edge type's reference semantics are not modelled, so no "
                  "source edit is claimed")


@dataclass
class PredictedChange:
    """One ΔB in ``ΔA ⇒ {ΔB, ΔC, ΔD}``."""

    object: MCMObject
    change: Change
    consequence: Consequence
    path: DependencyPath
    #: Where the dependency on the change site was observed, from the evidence on
    #: the first hop. For MUST_UPDATE these are the places to edit.
    sites: list[str] = field(default_factory=list)
    why: str = ""

    @property
    def confidence(self) -> float:
        return self.path.confidence

    @property
    def band(self) -> str:
        return confidence_band(self.confidence)

    @property
    def is_direct(self) -> bool:
        return self.path.depth == 1


@dataclass
class PropagationResult:
    change: Change
    target: MCMObject
    predicted: list[PredictedChange] = field(default_factory=list)
    truncated: bool = False
    as_of: datetime | None = None

    @property
    def must_update(self) -> list[PredictedChange]:
        return [p for p in self.predicted if p.consequence is Consequence.MUST_UPDATE]

    @property
    def may_differ(self) -> list[PredictedChange]:
        return [p for p in self.predicted if p.consequence is Consequence.MAY_DIFFER]

    @property
    def affected_tests(self) -> list[PredictedChange]:
        return [p for p in self.predicted if p.object.type is ObjectType.TEST]

    @property
    def confidence(self) -> float:
        """Confidence of the weakest prediction in the report."""
        return min((p.confidence for p in self.predicted), default=1.0)

    @property
    def edit_confidence(self) -> float:
        """Confidence in the "these sites must be edited" claim.

        Reported separately from ``confidence`` for the same reason impact analysis
        separates ``test_confidence``: it is the claim an agent acts on, every
        MUST_UPDATE rests on a directly observed reference, and a long
        low-confidence behaviour tail should not drag it down.
        """
        return min((p.confidence for p in self.must_update), default=1.0)


def propagate(store: Store, change: Change, *, max_depth: int = 6,
              as_of: datetime | None = None) -> PropagationResult:
    """Estimate the changes that follow from ``change``.

    Built on ``analyse_impact`` rather than beside it, so there is one dependency
    traversal in the system and propagation cannot disagree with impact about who
    is affected. What this adds is the *kind* of consequence and where the work is.
    """
    impact = analyse_impact(store, change.target_id, max_depth=max_depth, as_of=as_of)
    result = PropagationResult(change=change, target=impact.target,
                               truncated=impact.truncated, as_of=as_of)
    for affected in impact.all_affected:
        result.predicted.append(_predict(store, change, affected))
    result.predicted.sort(key=_ranking)
    return result


def propagate_from(store: Store, impact: ImpactResult,
                   change: Change) -> PropagationResult:
    """Label an impact result that has already been computed.

    For callers holding an ``ImpactResult`` - the context package builds one - so
    the traversal is not repeated.
    """
    result = PropagationResult(change=change, target=impact.target,
                               truncated=impact.truncated, as_of=impact.as_of)
    for affected in impact.all_affected:
        result.predicted.append(_predict(store, change, affected))
    result.predicted.sort(key=_ranking)
    return result


def _predict(store: Store, change: Change,
             affected: AffectedObject) -> PredictedChange:
    """What follows for one dependent."""
    first_hop = affected.path.relations[0]
    response = EDGE_RESPONSES.get(first_hop.relation_type, DEFAULT_EDGE_RESPONSE)

    if affected.path.depth != 1:
        # An indirect dependent holds no reference to the change site, so there is
        # nothing in its source to edit no matter what kind of change this is.
        consequence = Consequence.MAY_DIFFER
        why = (f"reached through {len(affected.path.relations)} dependency edges; "
               "its source does not name the change site")
    elif change.breaks_reference and response.on_reference_break:
        consequence = Consequence.MUST_UPDATE
        why = response.why
    elif change.breaks_call and response.on_call_break:
        consequence = Consequence.MUST_UPDATE
        why = response.why
    elif change.kind is ChangeKind.BEHAVIOUR:
        consequence = Consequence.MAY_DIFFER
        why = "the interface is unchanged, so this reference stays valid"
    else:
        consequence = Consequence.MAY_DIFFER
        why = (f"a {change.kind.value} change does not land in this edge: "
               f"{response.why}")

    return PredictedChange(
        object=affected.object,
        change=Change(
            target_id=affected.object.id,
            # A dependent's own interface does not change because its dependency
            # did: editing a call site or computing a different result are both
            # changes to a body. The one case this misses is an override that has
            # to be renamed with its base, which needs member-level modelling the
            # extractor does not have.
            kind=ChangeKind.BEHAVIOUR,
            details=f"follows {change.kind.value} of {change.target_id}",
        ),
        consequence=consequence,
        path=affected.path,
        sites=_sites(store, first_hop),
        why=why,
    )


def _sites(store: Store, relation) -> list[str]:
    """Source locations behind one observed dependency, from its evidence."""
    out: list[str] = []
    for evidence_id in relation.evidence_ids:
        evidence = store.get_evidence(evidence_id)
        if evidence is not None and evidence.source_ref not in out:
            out.append(evidence.source_ref)
    return out


def _ranking(predicted: PredictedChange) -> tuple:
    """Work first, then certainty. MUST_UPDATE is what an agent acts on."""
    return (predicted.consequence is not Consequence.MUST_UPDATE,
            -predicted.confidence, predicted.object.id)


def explain(result: PropagationResult) -> str:
    """Render the estimate (spec sections 43, 44)."""
    lines = [f"Change: {result.change.describe()}",
             f"  {result.target.name}  [{result.target.type.value}]"]
    if result.as_of is not None:
        lines.append(f"  as of {result.as_of.isoformat()}")
    lines.append("")

    if not result.predicted:
        lines.append("Nothing in the repository depends on this object, so this "
                     "change propagates nowhere.")
        return "\n".join(lines)

    must = result.must_update
    if must:
        lines.append(f"Must be updated ({len(must)}) - "
                     f"confidence {result.edit_confidence:.2f} "
                     f"{confidence_band(result.edit_confidence)}:")
        for predicted in must:
            lines.extend(_render(predicted))
        lines.append("")
    else:
        lines.append("No source edits are required by this change.")
        lines.append("")

    differ = result.may_differ
    if differ:
        lines.append(f"Behaviour may differ ({len(differ)}) - no source edit:")
        for predicted in differ:
            lines.extend(_render(predicted))
        lines.append("")

    tests = result.affected_tests
    if tests:
        names = ", ".join(sorted({p.object.name for p in tests}))
        lines.append(f"Run these tests: {names}")
        lines.append("  They are the evidence for every MAY_DIFFER above, which "
                     "this phase cannot settle statically.")
    else:
        lines.append("No test reaches this change site, so nothing here can be "
                     "confirmed by running the suite.")
    return "\n".join(lines)


def _render(predicted: PredictedChange) -> list[str]:
    kind = "FACT " if predicted.is_direct else "INFER"
    lines = [f"  [{kind}] {predicted.object.name}  "
             f"({predicted.object.type.value}, depth {predicted.path.depth}, "
             f"{predicted.confidence:.2f} {predicted.band})"]
    lines.append(f"          {' -> '.join(predicted.path.edge_types)}")
    lines.append(f"          why: {predicted.why}")
    if predicted.sites and predicted.consequence is Consequence.MUST_UPDATE:
        lines.append(f"          edit: {', '.join(predicted.sites)}")
    return lines


def propagation_json(result: PropagationResult) -> dict:
    """Serialise in the style of the spec section 42 response."""
    return {
        "mode": "propagate",
        "change": {"target": result.change.target_id,
                   "kind": result.change.kind.value,
                   "details": result.change.details},
        "target": {"id": result.target.id, "type": result.target.type.value,
                   "name": result.target.name},
        "must_update": [_predicted_json(p) for p in result.must_update],
        "may_differ": [_predicted_json(p) for p in result.may_differ],
        "affected_tests": [p.object.name for p in result.affected_tests],
        "edit_confidence": round(result.edit_confidence, 4),
        "edit_confidence_band": confidence_band(result.edit_confidence),
        "weakest_claim_confidence": round(result.confidence, 4),
        "truncated": result.truncated,
        "as_of": result.as_of.isoformat() if result.as_of else None,
    }


def _predicted_json(predicted: PredictedChange) -> dict:
    return {
        "object": {"id": predicted.object.id, "name": predicted.object.name,
                   "type": predicted.object.type.value,
                   "relpath": predicted.object.properties.get("relpath")},
        "predicted_change": {"kind": predicted.change.kind.value,
                             "details": predicted.change.details},
        "consequence": predicted.consequence.value,
        "confidence": round(predicted.confidence, 4),
        "band": predicted.band,
        "kind": "fact" if predicted.is_direct else "inference",
        "depth": predicted.path.depth,
        "reasoning_path": list(predicted.path.objects),
        "relations": list(predicted.path.edge_types),
        "sites": list(predicted.sites),
        "why": predicted.why,
        "evidence_ids": [eid for relation in predicted.path.relations
                         for eid in relation.evidence_ids],
    }
