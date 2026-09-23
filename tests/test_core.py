"""Object, relation, evidence, provenance and temporal-validity tests (spec section 71)."""

from datetime import datetime, timedelta, timezone

import pytest

from mcm.core.evidence import Evidence, EvidenceType
from mcm.core.ids import relation_id, symbol_id
from mcm.core.objects import MCMObject, ObjectType
from mcm.core.provenance import ExtractionMethod, Provenance
from mcm.core.relations import Inference, MCMRelation, RelationType


class TestIdentity:
    def test_symbol_ids_are_deterministic(self):
        first = symbol_id("app", "auth.py", "function", "authenticate")
        second = symbol_id("app", "auth.py", "function", "authenticate")
        assert first == second == "repo://app/auth.py#function:authenticate"

    def test_identity_ignores_line_numbers(self):
        """Spec section 20: identity must survive a symbol moving within a file."""
        assert (symbol_id("app", "auth.py", "function", "authenticate")
                == symbol_id("app", "auth.py", "function", "authenticate"))

    def test_windows_separators_normalise(self):
        assert (symbol_id("app", "tests\\test_auth.py", "test", "test_x")
                == symbol_id("app", "tests/test_auth.py", "test", "test_x"))

    def test_relation_id_separates_extraction_methods(self):
        """The same claim from two extractors stays two records (spec section 54)."""
        args = ["a", "b"]
        assert (relation_id("TESTS", args, "AST")
                != relation_id("TESTS", args, "STATIC_ANALYSIS"))

    def test_relation_id_is_order_sensitive(self):
        assert relation_id("CALLS", ["a", "b"], "AST") != relation_id("CALLS", ["b", "a"], "AST")


class TestObjects:
    def test_requires_an_id(self):
        with pytest.raises(ValueError):
            MCMObject(id="", type=ObjectType.FUNCTION, name="f")

    def test_string_type_is_coerced(self):
        obj = MCMObject(id="x", type="Function", name="f")
        assert obj.type is ObjectType.FUNCTION


class TestRelations:
    def test_requires_arguments(self):
        with pytest.raises(ValueError):
            MCMRelation(id="r", relation_type=RelationType.CALLS, arguments=[])

    def test_rejects_confidence_outside_unit_interval(self):
        with pytest.raises(ValueError):
            MCMRelation(id="r", relation_type=RelationType.CALLS,
                        arguments=["a", "b"], confidence=1.5)

    def test_asserted_relation_is_not_derived(self):
        relation = MCMRelation(id="r", relation_type=RelationType.CALLS, arguments=["a", "b"])
        assert relation.is_derived is False

    def test_relation_with_inference_is_derived(self):
        """Spec Rule 2: an inference is a different kind of record from a fact."""
        relation = MCMRelation(
            id="r", relation_type=RelationType.POSSIBLY_DEPENDS_ON, arguments=["a", "c"],
            inference=Inference(rule="transitive_dependency", path=["a", "b", "c"]),
        )
        assert relation.is_derived is True

    def test_supports_more_than_two_arguments(self):
        """Spec section 8: relations are n-ary, not source/target/label triples."""
        relation = MCMRelation(id="r", relation_type=RelationType.CALLS,
                               arguments=["a", "b", "c"])
        assert len(relation.arguments) == 3


class TestTemporalValidity:
    """Spec section 18: facts change, and history is retained rather than deleted."""

    def _at(self, offset_days: int) -> datetime:
        return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=offset_days)

    def test_open_ended_relation_is_always_valid(self):
        relation = MCMRelation(id="r", relation_type=RelationType.CALLS, arguments=["a", "b"])
        assert relation.is_valid_at(self._at(0))

    def test_relation_is_invalid_before_valid_from(self):
        relation = MCMRelation(id="r", relation_type=RelationType.CALLS,
                               arguments=["a", "b"], valid_from=self._at(10))
        assert not relation.is_valid_at(self._at(5))
        assert relation.is_valid_at(self._at(15))

    def test_superseded_relation_stays_readable_at_its_old_time(self):
        relation = MCMRelation(id="r", relation_type=RelationType.DEPENDS_ON,
                               arguments=["a", "b"],
                               valid_from=self._at(0), valid_until=self._at(10))
        assert relation.is_valid_at(self._at(5))
        assert not relation.is_valid_at(self._at(11))


class TestEvidenceAndProvenance:
    def test_evidence_id_is_content_addressed(self):
        first = Evidence.create(EvidenceType.AST, "auth.py:1", "x calls y", "AST")
        second = Evidence.create(EvidenceType.AST, "auth.py:1", "x calls y", "AST")
        assert first.id == second.id

    def test_provenance_defaults_reliability_by_method(self):
        """Spec section 17: source reliability is its own dimension, not confidence."""
        ast = Provenance.create(ExtractionMethod.AST, "test", "auth.py:1")
        llm = Provenance.create(ExtractionMethod.LLM_INFERENCE, "test", "auth.py:1")
        assert ast.source_reliability == 1.0
        assert llm.source_reliability < ast.source_reliability


class TestStorageRoundTrip:
    def test_object_round_trip(self, empty_store):
        obj = MCMObject(id="x", type=ObjectType.FUNCTION, name="f",
                        properties={"relpath": "a.py", "start_line": 3})
        empty_store.put_object(obj)
        assert empty_store.get_object("x").properties["start_line"] == 3

    def test_relation_round_trip_preserves_inference(self, empty_store):
        relation = MCMRelation(
            id="r", relation_type=RelationType.POSSIBLY_DEPENDS_ON, arguments=["a", "c"],
            confidence=0.8, inference=Inference(rule="transitive_dependency",
                                                path=["a", "b", "c"],
                                                premise_relation_ids=["r1", "r2"]),
        )
        empty_store.put_relation(relation)
        loaded = empty_store.get_relation("r")
        assert loaded.is_derived
        assert loaded.inference.rule == "transitive_dependency"
        assert loaded.inference.path == ["a", "b", "c"]

    def test_reads_exclude_derived_relations_by_default(self, empty_store):
        """Spec Rule 2: an inference must be asked for, never returned as a fact."""
        empty_store.put_relation(MCMRelation(
            id="fact", relation_type=RelationType.CALLS, arguments=["a", "b"]))
        empty_store.put_relation(MCMRelation(
            id="guess", relation_type=RelationType.POSSIBLY_DEPENDS_ON, arguments=["a", "c"],
            inference=Inference(rule="transitive_dependency", path=["a", "b", "c"])))

        assert {r.id for r in empty_store.all_relations()} == {"fact"}
        assert {r.id for r in empty_store.all_relations(include_derived=True)} == {"fact", "guess"}

    def test_writes_are_idempotent(self, empty_store):
        relation = MCMRelation(id="r", relation_type=RelationType.CALLS, arguments=["a", "b"])
        empty_store.put_relation(relation)
        empty_store.put_relation(relation)
        assert len(list(empty_store.all_relations())) == 1
        assert len(empty_store.relations_for("a", direction="out")) == 1

    def test_invalid_relation_direction_is_rejected(self, empty_store):
        with pytest.raises(ValueError, match="invalid relation direction"):
            empty_store.relations_for("a", direction="incoming")
