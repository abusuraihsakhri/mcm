"""Confidence arithmetic for derived relations.

Spec section 17 is explicit: "Do not automatically multiply these numbers unless a
mathematically justified model is defined." So the combination rule lives here, in
one replaceable function, rather than being scattered through the reasoning engine.

Three uncertainty dimensions are kept distinct (spec section 17):

    confidence          on MCMRelation  - belief in this specific claim
    evidence_strength   on Evidence     - how strongly the artifact supports it
    source_reliability  on Provenance   - how much the extractor is trusted

Only ``confidence`` is combined along a path. The other two remain attached to
their records so that a caller can inspect *why* a path is weak.
"""

from __future__ import annotations

from ..core.relations import MCMRelation, RelationType as RT

#: How much impact survives crossing one edge of each type.
#:
#: This is not a belief about whether the edge is real - AST edges are certain.
#: It is a belief about how tightly a change at one end constrains the other:
#:
#: * ``CALLS`` / ``USES`` are tight. Changing a callee's behaviour reaches its
#:   caller directly.
#: * ``IMPORTS`` is loose and file-granular. "auth.py imports jwt_provider" says
#:   almost nothing about which functions in auth.py a given change touches, so
#:   long import chains are damped hard rather than competing with call chains.
#: * ``TESTS`` sits between the two: a test does exercise its subject, but the
#:   edge itself is already a heuristic assertion.
EDGE_DECAY: dict[RT, float] = {
    RT.CALLS: 0.95,
    RT.USES: 0.95,
    RT.DEPENDS_ON: 0.95,
    RT.READS: 0.95,
    RT.INHERITS: 0.9,
    RT.IMPLEMENTS: 0.9,
    RT.TESTS: 0.9,
    RT.IMPORTS: 0.6,
}

#: Applied to any dependency edge type without an entry above. Deliberately
#: pessimistic: an unmodelled edge should not outrank a modelled one.
DEFAULT_EDGE_DECAY = 0.9


def path_confidence(edges: list[MCMRelation]) -> float:
    """Confidence of a derived relation spanning a chain of asserted edges.

    ``edges`` is ordered source-to-target and always has at least one entry (a
    one-hop "path" is the asserted edge itself).

    The model has two independent factors:

    * **Belief**: ``min(edge confidences)``. Weakest-link semantics - a chain is
      no more believable than its least believable edge. Unlike a product, this
      does not punish a long chain of certainties for being long.
    * **Attenuation**: the product of ``EDGE_DECAY`` over every edge *after the
      first*. This is not about edge belief. It encodes how far a change actually
      propagates across each kind of relation, so that a five-hop call chain and a
      five-hop import chain do not arrive at the same number.

    The first edge is exempt from attenuation, which keeps a direct dependent at
    the confidence its asserted relation carries. Beyond depth 1 the result is an
    inference and is scored as one.

    A plain product would report a five-hop chain of AST facts as 1.0, which
    overstates what static analysis supports. A plain ``min`` would do the same.
    Keeping belief and attenuation separate lets each be tuned - or replaced -
    without disturbing the other, which is what the ablation studies in spec
    section 50 need.
    """
    if not edges:
        raise ValueError("path_confidence requires at least one edge")
    belief = min(edge.confidence for edge in edges)
    attenuation = 1.0
    for edge in edges[1:]:
        attenuation *= EDGE_DECAY.get(edge.relation_type, DEFAULT_EDGE_DECAY)
    return belief * attenuation


def confidence_band(confidence: float) -> str:
    """Map a score onto the categories from spec section 44.

    Spec section 44: "Do not present uncertain inferences as facts." Callers
    render the band, not the raw float, in agent-facing output.
    """
    if confidence >= 0.85:
        return "HIGH"
    if confidence >= 0.6:
        return "MEDIUM"
    if confidence > 0.0:
        return "LOW"
    return "UNKNOWN"
