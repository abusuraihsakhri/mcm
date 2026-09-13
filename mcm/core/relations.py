"""N-ary typed relations with evidence, provenance and temporal validity.

Spec section 8 states this as a fundamental architectural requirement: a relation
is NOT ``(source, target, label)``. It carries its own confidence, the evidence
that supports it, where it came from, and when it was true.

Spec section 58 adds the second requirement enforced here: an *asserted* relation
(read off an AST) and a *derived* relation (produced by the inference engine) are
different kinds of record. A derived relation carries an ``Inference`` describing
the rule and path that produced it, and can never be silently promoted to a fact
(spec Rule 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .objects import utcnow


class RelationType(str, Enum):
    """Primitive relation families from spec section 9.

    TESTS / TESTED_BY are additions. Spec section 9 omits them, but the section 65
    proof-of-concept requires ``test_auth TESTS authenticate``. See
    docs/relation-algebra.md for the rationale.
    """

    # Structural
    CONTAINS = "CONTAINS"
    PART_OF = "PART_OF"
    DECLARES = "DECLARES"
    INHERITS = "INHERITS"
    IMPLEMENTS = "IMPLEMENTS"

    # Dependency
    IMPORTS = "IMPORTS"
    DEPENDS_ON = "DEPENDS_ON"
    CALLS = "CALLS"
    USES = "USES"
    READS = "READS"
    WRITES = "WRITES"
    TESTS = "TESTS"
    TESTED_BY = "TESTED_BY"

    # Semantic
    REPRESENTS = "REPRESENTS"
    DESCRIBES = "DESCRIBES"
    REFERS_TO = "REFERS_TO"
    SIMILAR_TO = "SIMILAR_TO"

    # Causal
    CAUSES = "CAUSES"
    CONTRIBUTES_TO = "CONTRIBUTES_TO"
    PREVENTS = "PREVENTS"
    TRIGGERS = "TRIGGERS"

    # Temporal
    PRECEDES = "PRECEDES"
    FOLLOWS = "FOLLOWS"
    ACTIVE_DURING = "ACTIVE_DURING"
    SUPERSEDES = "SUPERSEDES"

    # Agent knowledge
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    ASSUMED = "ASSUMED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"

    # Transformation
    TRANSFORMS = "TRANSFORMS"
    REFACTORS = "REFACTORS"
    REPLACES = "REPLACES"
    FIXES = "FIXES"
    BREAKS = "BREAKS"

    # Equivalence
    EQUIVALENT_TO = "EQUIVALENT_TO"
    SEMANTICALLY_EQUIVALENT_TO = "SEMANTICALLY_EQUIVALENT_TO"

    # Derived-only. Never asserted by ingestion (spec sections 35 and 58).
    POSSIBLY_DEPENDS_ON = "POSSIBLY_DEPENDS_ON"
    POSSIBLY_AFFECTS = "POSSIBLY_AFFECTS"


@dataclass(frozen=True)
class Inference:
    """How a derived relation was produced (spec section 58)."""

    rule: str
    path: list[str]
    premise_relation_ids: list[str] = field(default_factory=list)


@dataclass
class MCMRelation:
    id: str
    relation_type: RelationType
    arguments: list[str]
    properties: dict = field(default_factory=dict)
    confidence: float = 1.0
    evidence_ids: list[str] = field(default_factory=list)
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    provenance_id: str | None = None
    inference: Inference | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.relation_type, RelationType):
            self.relation_type = RelationType(self.relation_type)
        if not self.arguments:
            raise ValueError("MCMRelation requires at least one argument")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")

    @property
    def is_derived(self) -> bool:
        """True if this relation was inferred rather than observed."""
        return self.inference is not None

    def is_valid_at(self, moment: datetime | None = None) -> bool:
        """Temporal validity (spec section 18). Historical relations are retained,
        not deleted, so every read is implicitly a read at a point in time."""
        moment = moment or utcnow()
        if self.valid_from and moment < self.valid_from:
            return False
        if self.valid_until and moment >= self.valid_until:
            return False
        return True
