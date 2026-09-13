"""Change-impact analysis (spec sections 29, 30, 43, 44).

Given a change site, compute what may need to change with it, and produce an
explanation that names the relations and evidence behind every claim. Spec
section 43: the agent must never simply say "I know this."

Everything this module returns beyond depth 1 is an *inference*. Derived
POSSIBLY_AFFECTS relations are built here and handed back to the caller; they are
deliberately not written to the store by this function. Persisting an inference is
a separate, explicit decision (spec Rule 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..algebra.confidence import confidence_band
from ..algebra.dependency import DependencyPath, dependents_of
from ..core.ids import relation_id
from ..core.objects import MCMObject, ObjectType
from ..core.provenance import ExtractionMethod
from ..core.relations import Inference, MCMRelation, RelationType as RT
from ..storage.database import Store

IMPACT_RULE = "dependency_impact"


@dataclass
class AffectedObject:
    object: MCMObject
    path: DependencyPath
    #: None at depth 1, where the relation is an asserted fact rather than an
    #: inference. Beyond depth 1 this is the derived POSSIBLY_AFFECTS relation.
    derived: MCMRelation | None

    @property
    def confidence(self) -> float:
        return self.path.confidence

    @property
    def band(self) -> str:
        return confidence_band(self.confidence)

    @property
    def is_fact(self) -> bool:
        """True when the dependency on the change site was directly observed."""
        return self.path.depth == 1


@dataclass
class ImpactResult:
    target: MCMObject
    direct: list[AffectedObject] = field(default_factory=list)
    indirect: list[AffectedObject] = field(default_factory=list)
    affected_tests: list[AffectedObject] = field(default_factory=list)
    truncated: bool = False
    as_of: datetime | None = None

    @property
    def all_affected(self) -> list[AffectedObject]:
        return [*self.direct, *self.indirect]

    @property
    def affected_apis(self) -> list[AffectedObject]:
        """Spec section 29 requires "APIs affected" in the return.

        A view over the affected set, not a separate analysis: an API is affected
        exactly when it is affected. Empty on every corpus this system can ingest
        today, because Python ingestion emits no ``API`` objects - naming one needs
        a framework-route extractor that does not exist. The view is correct the
        moment one does, which is the difference between a projection and a stub.
        """
        return [a for a in self.all_affected if a.object.type is ObjectType.API]

    @property
    def affected_configuration(self) -> list[AffectedObject]:
        """Spec section 29 requires "configuration affected". Same reasoning as
        ``affected_apis``: no configuration parser exists yet, so no
        ``Configuration`` objects exist to be affected."""
        return [a for a in self.all_affected
                if a.object.type is ObjectType.CONFIGURATION]

    @property
    def confidence(self) -> float:
        """Confidence of the weakest claim in the report.

        A min rather than a mean: a single low-confidence tail should stay
        visible instead of being averaged out of sight. Note that this describes
        the *report*, not its recommendation - a loose file-level import chain
        will dominate it. Use ``test_confidence`` for the actionable number.
        """
        return min((a.confidence for a in self.all_affected), default=1.0)

    @property
    def test_confidence(self) -> float:
        """Confidence in the "run these tests" recommendation.

        Reported separately because it is the claim an agent acts on, and because
        it can be HIGH while ``confidence`` is LOW.
        """
        return min((a.confidence for a in self.affected_tests), default=1.0)

    @property
    def weakest(self) -> "AffectedObject | None":
        return min(self.all_affected, key=lambda a: a.confidence, default=None)


def analyse_impact(store: Store, target_id: str, *, max_depth: int = 6,
                   as_of: datetime | None = None) -> ImpactResult:
    """What could break if ``target_id`` changes?

    ``as_of`` answers the question against a past state of the repository, which
    is what makes "what would this change have broken last week" a different
    query from "what would it break now".
    """
    target = store.get_object(target_id)
    if target is None:
        raise KeyError(f"unknown object: {target_id}")

    closure = dependents_of(store, target_id, max_depth=max_depth, as_of=as_of)
    result = ImpactResult(target=target, truncated=closure.truncated, as_of=as_of)

    for path in closure.ordered():
        obj = store.get_object(path.object_id)
        if obj is None:
            continue
        affected = AffectedObject(
            object=obj,
            path=path,
            derived=None if path.depth == 1 else _derive(target_id, path),
        )
        if path.depth == 1:
            result.direct.append(affected)
        else:
            result.indirect.append(affected)
        if obj.type is ObjectType.TEST:
            result.affected_tests.append(affected)

    return result


def _derive(target_id: str, path: DependencyPath) -> MCMRelation:
    """Build the POSSIBLY_AFFECTS relation justifying a multi-hop claim."""
    arguments = [target_id, path.object_id]
    return MCMRelation(
        id=relation_id(RT.POSSIBLY_AFFECTS.value, arguments,
                       ExtractionMethod.SYMBOLIC_INFERENCE.value),
        relation_type=RT.POSSIBLY_AFFECTS,
        arguments=arguments,
        confidence=path.confidence,
        evidence_ids=[eid for r in path.relations for eid in r.evidence_ids],
        inference=Inference(
            rule=IMPACT_RULE,
            path=list(path.objects),
            premise_relation_ids=[r.id for r in path.relations],
        ),
    )


def explain(store: Store, result: ImpactResult) -> str:
    """Render the reasoning path and its evidence (spec section 43)."""
    lines: list[str] = []
    target = result.target
    lines.append(f"Change site: {target.name}  [{target.type.value}]")
    lines.append(f"  {target.id}")
    if result.as_of is not None:
        lines.append(f"  as of {result.as_of.isoformat()}")
    lines.append("")

    if not result.all_affected:
        lines.append("Nothing in the repository depends on this object.")
        return "\n".join(lines)

    lines.append(f"Direct dependents ({len(result.direct)}) - observed facts:")
    for affected in result.direct:
        lines.extend(_render(store, affected, indent="  "))

    if result.indirect:
        lines.append("")
        lines.append(f"Indirect dependents ({len(result.indirect)}) - inferred:")
        for affected in result.indirect:
            lines.extend(_render(store, affected, indent="  "))

    lines.append("")
    if result.affected_tests:
        names = ", ".join(a.object.name for a in result.affected_tests)
        lines.append(f"Tests to run after the change ({len(result.affected_tests)}): {names}")
        lines.append(f"  confidence {result.test_confidence:.2f} "
                     f"({confidence_band(result.test_confidence)})")
    else:
        lines.append("No tests cover the affected objects.")

    weakest = result.weakest
    if weakest is not None:
        lines.append(f"Weakest claim in report: {weakest.object.name} at "
                     f"{weakest.confidence:.2f} ({weakest.band}) via "
                     f"{'/'.join(weakest.path.edge_types)}")
    if result.truncated:
        lines.append("NOTE: traversal hit the depth limit; this closure is incomplete.")
    return "\n".join(lines)


def _render(store: Store, affected: AffectedObject, indent: str) -> list[str]:
    kind = "FACT " if affected.is_fact else "INFER"
    header = (f"{indent}[{kind}] {affected.object.name}  "
              f"({affected.object.type.value}, depth {affected.path.depth}, "
              f"confidence {affected.confidence:.2f} {affected.band})")
    lines = [header]

    # The reasoning path, target-first, annotated with the relation crossed.
    names = [_display(store, oid) for oid in affected.path.objects]
    for i, edge_type in enumerate(affected.path.edge_types):
        lines.append(f"{indent}    {names[i]}")
        lines.append(f"{indent}      <-{edge_type}-")
    lines.append(f"{indent}    {names[-1]}")

    for relation in affected.path.relations:
        for evidence_id in relation.evidence_ids:
            evidence = store.get_evidence(evidence_id)
            if evidence is not None:
                lines.append(f"{indent}    evidence: {evidence.source_ref} "
                             f"[{evidence.source_type.value}] {evidence.content}")
    return lines


def _display(store: Store, object_id: str) -> str:
    obj = store.get_object(object_id)
    if obj is None:
        return object_id
    qualname = obj.properties.get("qualname")
    relpath = obj.properties.get("relpath")
    if qualname and relpath:
        return f"{relpath}:{qualname}"
    return obj.name
