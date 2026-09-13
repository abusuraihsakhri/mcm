"""Tests for Memory Update and Reconciliation (spec section 32)."""

from datetime import datetime, timezone
import pytest

from mcm.agent.memory_update import (
    EpistemicStatus,
    ItemCategory,
    MemoryItem,
    MemoryUpdater,
)
from mcm.core.objects import ObjectType
from mcm.core.provenance import ExtractionMethod
from mcm.core.relations import RelationType as RT
from mcm.retrieval.hybrid import HybridRetriever
from mcm.storage.sqlite_store import SQLiteStore


@pytest.fixture
def store():
    s = SQLiteStore()
    yield s
    s.close()


class TestMemoryUpdate:
    def test_apply_structured_memory_items(self, store):
        updater = MemoryUpdater(store, agent_name="agent-42")

        items = [
            MemoryItem(
                category=ItemCategory.DECISION,
                status=EpistemicStatus.CONFIRMED,
                subject="Adopt SQLite storage",
                content="Use SQLite instead of PostgreSQL for single-file portability.",
                evidence_text="Spec decision: Section 5 allows SQLite for prototype.",
                method=ExtractionMethod.HUMAN,
                confidence=1.0,
            ),
            MemoryItem(
                category=ItemCategory.CONSTRAINT,
                status=EpistemicStatus.OBSERVED,
                subject="Max query depth",
                content="Graph traversal depth must not exceed 6.",
                confidence=0.95,
            ),
            MemoryItem(
                category=ItemCategory.ERROR,
                status=EpistemicStatus.OBSERVED,
                subject="Token count discrepancy",
                content="Identified mismatch between character count and token count.",
                target="Adopt SQLite storage",
                relation_type=RT.BREAKS,
                confidence=0.85,
                evidence_text="Log trace showed token count drift.",
                method=ExtractionMethod.STATIC_ANALYSIS,
            ),
        ]

        report = updater.apply(items, rebuild=True)

        assert report.items_processed == 3
        assert report.objects_created >= 3
        assert report.relations_created >= 3
        assert report.evidence_created >= 2
        assert report.projections_rebuilt is True

        # Verify structured objects exist in canonical store
        decision_obj = store.get_object("decision://agent/adopt_sqlite_storage")
        assert decision_obj is not None
        assert decision_obj.type == ObjectType.DECISION

        # Verify searchability in hybrid retriever after projection rebuild
        retriever = HybridRetriever(store)
        search_result = retriever.retrieve("SQLite portability", limit=5)
        assert len(search_result.candidates) > 0
        assert any("sqlite" in c.object_id for c in search_result.candidates)

    def test_contradiction_surfaces_in_report(self, store):
        updater = MemoryUpdater(store, agent_name="agent-42")

        # Initial confirmed claim
        updater.apply([
            MemoryItem(
                category=ItemCategory.FACT,
                status=EpistemicStatus.CONFIRMED,
                subject="fn://auth",
                target="fn://jwt",
                relation_type=RT.CAUSES,
                content="auth causes jwt validation",
                confidence=0.90,
            )
        ])

        # Conflicting claim arrives later (newer, but lower confidence)
        report2 = updater.apply([
            MemoryItem(
                category=ItemCategory.FACT,
                status=EpistemicStatus.CONFIRMED,
                subject="fn://auth",
                target="fn://jwt",
                relation_type=RT.PREVENTS,
                content="auth prevents jwt validation",
                confidence=0.70,
            )
        ])

        assert len(report2.contradictions) == 1
        c = report2.contradictions[0]
        # Opposing predicates detected
        assert c.kind.value == "OPPOSING_PREDICATES"
        # Criteria accurately reflect: stronger on one, newer on the other
        assert {c.stronger, c.newer} == {"A", "B"}
        assert c.resolution_verdict == "UNRESOLVED_DISPUTE"
