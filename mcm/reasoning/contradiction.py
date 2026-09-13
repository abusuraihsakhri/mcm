"""Contradiction detection (spec section 31).

    A => B   and   A => ¬B

Spec section 31 instructions:
1. Do not overwrite either relation.
2. Create a CONTRADICTION record and retain both evidence chains.
3. Evaluate:
   - Which evidence is stronger?
   - Which is newer?
   - Which source is more reliable?

A contradiction arises when two active relations make mutually incompatible
claims about the same entities at the same time, or when an asserted update
directly negates an existing claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..core.evidence import Evidence
from ..core.provenance import Provenance
from ..core.relations import MCMRelation, RelationType as RT
from ..storage.database import Store


class ConflictKind(str, Enum):
    OPPOSING_PREDICATES = "OPPOSING_PREDICATES"
    EXPLICIT_NEGATION = "EXPLICIT_NEGATION"
    EXCLUSIVE_STATE = "EXCLUSIVE_STATE"


#: Incompatible relation pairs that directly oppose each other when sharing arguments.
OPPOSING_PAIRS: dict[RT, RT] = {
    RT.CAUSES: RT.PREVENTS,
    RT.PREVENTS: RT.CAUSES,
    RT.FIXES: RT.BREAKS,
    RT.BREAKS: RT.FIXES,
    RT.CONFIRMED: RT.REJECTED,
    RT.REJECTED: RT.CONFIRMED,
}


@dataclass
class Contradiction:
    """A detected contradiction between two relations with retained evidence chains."""

    relation_a: MCMRelation
    relation_b: MCMRelation
    kind: ConflictKind
    evidence_a: list[Evidence] = field(default_factory=list)
    evidence_b: list[Evidence] = field(default_factory=list)
    provenance_a: Provenance | None = None
    provenance_b: Provenance | None = None

    @property
    def stronger(self) -> str:
        """Which evidence/relation is stronger in confidence?"""
        c_a = self.relation_a.confidence
        c_b = self.relation_b.confidence
        if abs(c_a - c_b) < 1e-6:
            return "TIE"
        return "A" if c_a > c_b else "B"

    @property
    def newer(self) -> str:
        """Which claim is newer in time?"""
        t_a = self._timestamp_for(self.relation_a, self.evidence_a)
        t_b = self._timestamp_for(self.relation_b, self.evidence_b)
        if t_a is None or t_b is None or t_a == t_b:
            return "TIE"
        return "A" if t_a > t_b else "B"

    @property
    def more_reliable(self) -> str:
        """Which source has higher provenance reliability?"""
        r_a = self.provenance_a.source_reliability if self.provenance_a else 1.0
        r_b = self.provenance_b.source_reliability if self.provenance_b else 1.0
        if abs(r_a - r_b) < 1e-6:
            return "TIE"
        return "A" if r_a > r_b else "B"

    @property
    def resolution_verdict(self) -> str:
        """Summary of which evidence chain dominates."""
        votes = [self.stronger, self.newer, self.more_reliable]
        count_a = votes.count("A")
        count_b = votes.count("B")
        if count_a > count_b:
            return "FAVOR_A"
        if count_b > count_a:
            return "FAVOR_B"
        return "UNRESOLVED_DISPUTE"

    @property
    def preferred_relation(self) -> MCMRelation | None:
        verdict = self.resolution_verdict
        if verdict == "FAVOR_A":
            return self.relation_a
        if verdict == "FAVOR_B":
            return self.relation_b
        return None

    @staticmethod
    def _timestamp_for(rel: MCMRelation, evidences: list[Evidence]) -> datetime | None:
        if rel.valid_from:
            return rel.valid_from
        if evidences:
            return max(e.timestamp for e in evidences if e.timestamp)
        return None


def detect_contradictions(store: Store, as_of: datetime | None = None) -> list[Contradiction]:
    """Scan the store for active contradictory relations at the given moment.

    Returns a list of Contradiction records.
    Neither relation is removed or altered in the store (spec section 31).
    """
    # Fetch relations valid at this moment
    all_relations = [r for r in store.all_relations() if r.is_valid_at(as_of)]
    contradictions: list[Contradiction] = []
    seen_pairs: set[frozenset[str]] = set()

    for i, rel_a in enumerate(all_relations):
        for rel_b in all_relations[i + 1:]:
            pair_key = frozenset({rel_a.id, rel_b.id})
            if pair_key in seen_pairs:
                continue

            conflict = _check_pair(rel_a, rel_b)
            if conflict is not None:
                seen_pairs.add(pair_key)
                ev_a = [store.get_evidence(eid) for eid in rel_a.evidence_ids]
                ev_b = [store.get_evidence(eid) for eid in rel_b.evidence_ids]
                ev_a_clean = [e for e in ev_a if e is not None]
                ev_b_clean = [e for e in ev_b if e is not None]

                prov_a = store.get_provenance(rel_a.provenance_id) if rel_a.provenance_id else None
                prov_b = store.get_provenance(rel_b.provenance_id) if rel_b.provenance_id else None

                contradictions.append(Contradiction(
                    relation_a=rel_a,
                    relation_b=rel_b,
                    kind=conflict,
                    evidence_a=ev_a_clean,
                    evidence_b=ev_b_clean,
                    provenance_a=prov_a,
                    provenance_b=prov_b,
                ))

    return contradictions


def _check_pair(a: MCMRelation, b: MCMRelation) -> ConflictKind | None:
    if a.arguments != b.arguments:
        return None

    # Check opposing predicates (e.g. CAUSES vs PREVENTS)
    opposed = OPPOSING_PAIRS.get(a.relation_type)
    if opposed and b.relation_type == opposed:
        return ConflictKind.OPPOSING_PREDICATES

    # Check explicit negation on same predicate
    if a.relation_type == b.relation_type:
        neg_a = bool(a.properties.get("negated", False))
        neg_b = bool(b.properties.get("negated", False))
        if neg_a != neg_b:
            return ConflictKind.EXPLICIT_NEGATION

        # Check exclusive property assertions
        if a.properties and b.properties:
            for k in ("state", "value", "expected_outcome", "return_type"):
                if k in a.properties and k in b.properties and a.properties[k] != b.properties[k]:
                    return ConflictKind.EXCLUSIVE_STATE

    return None


def explain(contradictions: list[Contradiction]) -> str:
    """Format contradiction reports with evidence comparisons."""
    if not contradictions:
        return "No contradictions detected. The semantic store is coherent."

    lines = [f"Found {len(contradictions)} contradiction(s) (spec section 31):"]
    for idx, c in enumerate(contradictions, 1):
        a = c.relation_a
        b = c.relation_b
        args = " -> ".join(a.arguments)
        lines.append(f"\n{idx}. [{c.kind.value}] Arguments: {args}")
        lines.append(f"   Claim A: {a.relation_type.value} (conf={a.confidence:.2f}, "
                     f"rel={c.provenance_a.source_reliability if c.provenance_a else 1.0:.2f})")
        lines.append(f"   Claim B: {b.relation_type.value} (conf={b.confidence:.2f}, "
                     f"rel={c.provenance_b.source_reliability if c.provenance_b else 1.0:.2f})")
        lines.append(f"   Comparison:")
        lines.append(f"     Stronger evidence:  {c.stronger}")
        lines.append(f"     Newer claim:        {c.newer}")
        lines.append(f"     More reliable:      {c.more_reliable}")
        lines.append(f"     Resolution verdict: {c.resolution_verdict}")
        lines.append(f"   Evidence retained: Claim A has {len(c.evidence_a)} record(s), "
                     f"Claim B has {len(c.evidence_b)} record(s)")
    return "\n".join(lines)


def contradiction_json(contradictions: list[Contradiction]) -> list[dict]:
    """JSON-serializable representation for API and CLI."""
    out = []
    for c in contradictions:
        out.append({
            "kind": c.kind.value,
            "arguments": c.relation_a.arguments,
            "claim_a": {
                "id": c.relation_a.id,
                "type": c.relation_a.relation_type.value,
                "confidence": c.relation_a.confidence,
                "evidence_count": len(c.evidence_a),
            },
            "claim_b": {
                "id": c.relation_b.id,
                "type": c.relation_b.relation_type.value,
                "confidence": c.relation_b.confidence,
                "evidence_count": len(c.evidence_b),
            },
            "stronger": c.stronger,
            "newer": c.newer,
            "more_reliable": c.more_reliable,
            "verdict": c.resolution_verdict,
        })
    return out
