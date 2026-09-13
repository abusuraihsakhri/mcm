"""The composition operator (spec section 11).

``compose(r1, r2)`` is licensed by the RelationSpec table, never by the shape of
the data. Spec section 11: "the engine must NOT blindly assume transitivity."
"""

from __future__ import annotations

from ..core.ids import relation_id
from ..core.provenance import ExtractionMethod
from ..core.relations import Inference, MCMRelation, RelationType as RT
from .confidence import path_confidence
from .specs import implies_dependency, spec_for

TRANSITIVE_DEPENDENCY_RULE = "transitive_dependency"


def compose(r1: MCMRelation, r2: MCMRelation) -> MCMRelation | None:
    """Compose two relations into a derived relation, or return None.

    Returns None when composition is not licensed: mismatched endpoints, a
    relation type declared non-composable, or a pair that the algebra has no rule
    for. Returning None rather than raising lets the closure walk skip edges it
    cannot reason about.
    """
    if len(r1.arguments) != 2 or len(r2.arguments) != 2:
        return None
    if r1.arguments[1] != r2.arguments[0]:
        return None

    s1, s2 = spec_for(r1.relation_type), spec_for(r2.relation_type)
    if not (s1.composable and s2.composable):
        return None

    # The only rule V1 declares. Both operands must be dependency edges under
    # subsumption; the result is POSSIBLY_DEPENDS_ON, never DEPENDS_ON, because
    # extracted dependency graphs are incomplete (spec sections 35 and 58).
    if not (implies_dependency(r1.relation_type) and implies_dependency(r2.relation_type)):
        return None

    source, middle, target = r1.arguments[0], r1.arguments[1], r2.arguments[1]
    path = [source, middle, target]
    return MCMRelation(
        id=relation_id(RT.POSSIBLY_DEPENDS_ON.value, [source, target],
                       ExtractionMethod.SYMBOLIC_INFERENCE.value),
        relation_type=RT.POSSIBLY_DEPENDS_ON,
        arguments=[source, target],
        confidence=path_confidence([r1, r2]),
        evidence_ids=[*r1.evidence_ids, *r2.evidence_ids],
        inference=Inference(
            rule=TRANSITIVE_DEPENDENCY_RULE,
            path=path,
            premise_relation_ids=[r1.id, r2.id],
        ),
    )
