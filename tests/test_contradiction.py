"""Tests for Contradiction Detection (spec section 31)."""

from datetime import datetime, timezone, timedelta
import pytest

from mcm.core.evidence import Evidence, EvidenceType
from mcm.core.objects import MCMObject, ObjectType, utcnow
from mcm.core.provenance import ExtractionMethod, Provenance
from mcm.core.relations import MCMRelation, RelationType as RT
from mcm.reasoning.contradiction import (
    ConflictKind,
    Contradiction,
    detect_contradictions,
    explain,
    contradiction_json,
)
from mcm.storage.sqlite_store import SQLiteStore


@pytest.fixture
def store():
    s = SQLiteStore()
    obj_a = MCMObject(id="fn://auth", type=ObjectType.FUNCTION, name="authenticate")
    obj_b = MCMObject(id="fn://jwt", type=ObjectType.FUNCTION, name="validate_jwt")
    s.put_object(obj_a)
    s.put_object(obj_b)
    yield s
    s.close()


class TestContradictionDetection:
    def test_clean_store_has_no_contradictions(self, store):
        contradictions = detect_contradictions(store)
        assert contradictions == []
        assert "No contradictions detected" in explain(contradictions)

    def test_opposing_predicates_causes_and_prevents(self, store):
        prov_ast = Provenance.create(ExtractionMethod.AST, "ast-parser", "auth.py")
        prov_llm = Provenance.create(ExtractionMethod.LLM_INFERENCE, "agent", "prompt")
        store.put_provenance(prov_ast)
        store.put_provenance(prov_llm)

        ev1 = Evidence.create(EvidenceType.SOURCE_CODE, "auth.py:10", "validate_jwt()", "AST", confidence=0.95)
        ev2 = Evidence.create(EvidenceType.USER, "chat", "prevents jwt", "USER", confidence=0.60)
        store.put_evidence(ev1)
        store.put_evidence(ev2)

        now = utcnow()
        rel1 = MCMRelation(
            id="rel://1",
            relation_type=RT.CAUSES,
            arguments=["fn://auth", "fn://jwt"],
            confidence=0.95,
            evidence_ids=[ev1.id],
            provenance_id=prov_ast.id,
            valid_from=now - timedelta(days=2),
        )
        rel2 = MCMRelation(
            id="rel://2",
            relation_type=RT.PREVENTS,
            arguments=["fn://auth", "fn://jwt"],
            confidence=0.60,
            evidence_ids=[ev2.id],
            provenance_id=prov_llm.id,
            valid_from=now,
        )
        store.put_relation(rel1)
        store.put_relation(rel2)

        contradictions = detect_contradictions(store)
        assert len(contradictions) == 1
        c = contradictions[0]
        assert c.kind == ConflictKind.OPPOSING_PREDICATES

        # Spec section 31 requirements:
        # 1. Do not overwrite either: Both relations must remain in the store
        assert store.get_relation("rel://1") is not None
        assert store.get_relation("rel://2") is not None

        # 2. Retain both evidence chains
        assert len(c.evidence_a) == 1
        assert len(c.evidence_b) == 1
        assert c.evidence_a[0].id == ev1.id
        assert c.evidence_b[0].id == ev2.id

        # 3. Which evidence is stronger? (0.95 vs 0.60 -> A)
        assert c.stronger == "A"

        # 4. Which is newer? (rel2 is now, rel1 is 2 days ago -> B)
        assert c.newer == "B"

        # 5. Which source is more reliable? (AST 1.0 vs LLM 0.6 -> A)
        assert c.more_reliable == "A"

        # Overall resolution
        assert c.resolution_verdict == "FAVOR_A"
        assert c.preferred_relation.id == "rel://1"

    def test_explicit_negation_contradiction(self, store):
        now = utcnow()
        r1 = MCMRelation(
            id="rel://calls-1",
            relation_type=RT.CALLS,
            arguments=["fn://auth", "fn://jwt"],
            properties={"negated": False},
            valid_from=now,
        )
        r2 = MCMRelation(
            id="rel://calls-2",
            relation_type=RT.CALLS,
            arguments=["fn://auth", "fn://jwt"],
            properties={"negated": True},
            valid_from=now,
        )
        store.put_relation(r1)
        store.put_relation(r2)

        contradictions = detect_contradictions(store)
        assert len(contradictions) == 1
        assert contradictions[0].kind == ConflictKind.EXPLICIT_NEGATION

    def test_expired_relation_does_not_contradict_active_relation(self, store):
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t1 = datetime(2026, 2, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 3, 1, tzinfo=timezone.utc)

        # Relation 1 was true between Jan and Feb
        r1 = MCMRelation(
            id="rel://hist-1",
            relation_type=RT.CAUSES,
            arguments=["fn://auth", "fn://jwt"],
            valid_from=t0,
            valid_until=t1,
        )
        # Relation 2 started in Feb
        r2 = MCMRelation(
            id="rel://hist-2",
            relation_type=RT.PREVENTS,
            arguments=["fn://auth", "fn://jwt"],
            valid_from=t1,
            valid_until=t2,
        )
        store.put_relation(r1)
        store.put_relation(r2)

        # Checking at t2: r1 was expired, so no contradiction
        contradictions = detect_contradictions(store, as_of=datetime(2026, 2, 15, tzinfo=timezone.utc))
        assert len(contradictions) == 0

    def test_json_and_explain_formatting(self, store):
        r1 = MCMRelation(id="rel://1", relation_type=RT.FIXES, arguments=["fn://auth", "fn://jwt"])
        r2 = MCMRelation(id="rel://2", relation_type=RT.BREAKS, arguments=["fn://auth", "fn://jwt"])
        store.put_relation(r1)
        store.put_relation(r2)

        contradictions = detect_contradictions(store)
        assert len(contradictions) == 1
        output = explain(contradictions)
        assert "Found 1 contradiction(s)" in output
        assert "Claim A: FIXES" in output
        assert "Claim B: BREAKS" in output

        data = contradiction_json(contradictions)
        assert len(data) == 1
        assert data[0]["kind"] == "OPPOSING_PREDICATES"
        assert data[0]["claim_a"]["type"] == "FIXES"
        assert data[0]["claim_b"]["type"] == "BREAKS"
