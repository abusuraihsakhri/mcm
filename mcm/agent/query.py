"""Query dispatcher (spec sections 27, 28, 42).

Four of the ten query types from spec section 28: LOOKUP, DEPENDENCY, IMPACT and
EQUIVALENCE. The remaining six (CAUSE, TRACE, CONSTRAINT, TEMPORAL, PROVENANCE,
CONTRADICTION) need machinery that later phases add; they are absent rather than
stubbed, so an unsupported query fails loudly instead of returning an empty result
that reads like an answer.

EQUIVALENCE answers under a stated domain (spec section 10) and reports UNKNOWN
rather than NOT_EQUIVALENT where the question is undecidable, which is the whole
point of the domain being stated.

Results are plain dicts shaped like the response in spec section 42, ready for the
API layer to serialise.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from ..algebra.confidence import confidence_band
from ..algebra.dependency import dependencies_of
from ..core.objects import MCMObject
from ..equivalence.engine import (Domain, EquivalenceEngine, Verdict,
                                  fingerprint_from)
from ..reasoning.dependency_propagation import ImpactResult, analyse_impact
from ..retrieval.symbolic import resolve, resolve_one
from ..storage.database import Store


class QueryType(str, Enum):
    LOOKUP = "lookup"
    DEPENDENCY = "dependency"
    IMPACT = "impact"
    EQUIVALENCE = "equivalence"


def run_query(store: Store, mode: QueryType | str, target: str, *,
              max_depth: int = 6, as_of: datetime | None = None,
              domain: Domain | str = Domain.NORMAL_FORM) -> dict:
    mode = QueryType(mode)
    if mode is QueryType.LOOKUP:
        return _lookup(store, target)
    if mode is QueryType.DEPENDENCY:
        return _dependency(store, target, max_depth, as_of)
    if mode is QueryType.EQUIVALENCE:
        return _equivalence(store, target, Domain(domain))
    return _impact(store, target, max_depth, as_of)


def _equivalence(store: Store, target: str, domain: Domain) -> dict:
    """What is equivalent to this, under a stated domain.

    Every other definition sharing the target's form. The comparison runs through
    the engine rather than off a raw digest match, so the answer carries the
    verdict and the domain that witnessed it instead of a bare list.
    """
    obj = resolve_one(store, target)
    engine = EquivalenceEngine()
    fingerprint = fingerprint_from(obj)
    matches = []
    for candidate in store.all_objects():
        if candidate.id == obj.id:
            continue
        result = engine.compare(fingerprint, fingerprint_from(candidate),
                                domain=domain)
        if result.verdict is Verdict.EQUIVALENT:
            matches.append({"object": _object_json(candidate),
                            "verdict": result.verdict.value,
                            "witness": result.witness.value if result.witness else None,
                            "reason": result.reason})
    return {
        "mode": QueryType.EQUIVALENCE.value,
        "target": _object_json(obj),
        "domain": domain.value,
        "equivalent_to": matches,
        # Spec section 44: an empty list under a decidable domain means "none
        # found"; under a domain that cannot be decided it would mean nothing at
        # all, which is why the domain travels with the answer.
        "decidable": domain is not Domain.SEMANTIC or bool(matches),
    }


def _lookup(store: Store, target: str) -> dict:
    matches = resolve(store, target)
    return {
        "mode": QueryType.LOOKUP.value,
        "target": target,
        "matches": [_object_json(obj) for obj in matches],
    }


def _dependency(store: Store, target: str, max_depth: int,
                as_of: datetime | None = None) -> dict:
    obj = resolve_one(store, target)
    closure = dependencies_of(store, obj.id, max_depth=max_depth, as_of=as_of)
    return {
        "mode": QueryType.DEPENDENCY.value,
        "target": _object_json(obj),
        "depends_on": [
            {
                "object": _object_json(store.get_object(path.object_id)),
                "depth": path.depth,
                "path": list(path.objects),
                "relations": list(path.edge_types),
                "confidence": round(path.confidence, 4),
                "asserted": path.depth == 1,
            }
            for path in closure.ordered()
        ],
        "truncated": closure.truncated,
        "as_of": as_of.isoformat() if as_of else None,
    }


def _impact(store: Store, target: str, max_depth: int,
            as_of: datetime | None = None) -> dict:
    obj = resolve_one(store, target)
    result = analyse_impact(store, obj.id, max_depth=max_depth, as_of=as_of)
    return impact_json(result)


def impact_json(result: ImpactResult) -> dict:
    """Serialise an impact result in the shape of spec section 42."""
    return {
        "mode": QueryType.IMPACT.value,
        "target": _object_json(result.target),
        "direct_dependents": [_affected_json(a) for a in result.direct],
        "indirect_dependents": [_affected_json(a) for a in result.indirect],
        "affected_tests": [a.object.name for a in result.affected_tests],
        "test_confidence": round(result.test_confidence, 4),
        "test_confidence_band": confidence_band(result.test_confidence),
        "weakest_claim_confidence": round(result.confidence, 4),
        "truncated": result.truncated,
        "as_of": result.as_of.isoformat() if result.as_of else None,
    }


def _affected_json(affected) -> dict:
    return {
        "object": _object_json(affected.object),
        "depth": affected.path.depth,
        "confidence": round(affected.confidence, 4),
        "band": affected.band,
        # Spec section 54: an asserted fact and an inference are different kinds
        # of claim, and the response says which one this is.
        "kind": "fact" if affected.is_fact else "inference",
        "reasoning_path": list(affected.path.objects),
        "relations": list(affected.path.edge_types),
        "evidence_ids": [
            eid for relation in affected.path.relations for eid in relation.evidence_ids
        ],
        "derived_relation_id": affected.derived.id if affected.derived else None,
    }


def _object_json(obj: MCMObject | None) -> dict | None:
    if obj is None:
        return None
    return {
        "id": obj.id,
        "type": obj.type.value,
        "name": obj.name,
        "relpath": obj.properties.get("relpath"),
        "qualname": obj.properties.get("qualname"),
        "start_line": obj.properties.get("start_line"),
    }
