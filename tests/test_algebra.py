"""Relation algebra tests (spec sections 11, 12, 17, 71)."""

import pytest

from mcm.algebra.composition import TRANSITIVE_DEPENDENCY_RULE, compose
from mcm.algebra.confidence import (DEFAULT_EDGE_DECAY, EDGE_DECAY, confidence_band,
                                    path_confidence)
from mcm.algebra.specs import (dependency_edge_types, implies_dependency, spec_for)
from mcm.core.relations import MCMRelation, RelationType as RT


def rel(relation_type: RT, a: str, b: str, confidence: float = 1.0) -> MCMRelation:
    return MCMRelation(id=f"{relation_type.value}:{a}:{b}", relation_type=relation_type,
                       arguments=[a, b], confidence=confidence)


class TestRelationProperties:
    def test_calls_is_not_transitive(self):
        """Spec section 12: A calls B and B calls C does not mean A calls C."""
        assert spec_for(RT.CALLS).transitive is False

    def test_equivalence_is_an_equivalence_relation(self):
        spec = spec_for(RT.EQUIVALENT_TO)
        assert (spec.symmetric, spec.reflexive, spec.transitive) == (True, True, True)

    def test_dependency_is_transitive(self):
        assert spec_for(RT.DEPENDS_ON).transitive is True

    def test_inverse_pairs_are_consistent(self):
        for relation_type, spec in [(RT.CONTAINS, spec_for(RT.CONTAINS)),
                                    (RT.TESTS, spec_for(RT.TESTS))]:
            assert spec.inverse is not None
            assert spec_for(spec.inverse).inverse is relation_type

    def test_undeclared_relation_type_raises(self, monkeypatch):
        """The algebra refuses to guess properties it was never given."""
        from mcm.algebra import specs

        patched = dict(specs.RELATION_SPECS)
        patched.pop(RT.CAUSES)
        monkeypatch.setattr(specs, "RELATION_SPECS", patched)
        with pytest.raises(KeyError, match="No RelationSpec declared"):
            specs.spec_for(RT.CAUSES)

    def test_every_relation_type_has_a_spec(self):
        """Spec section 12: every relation type should have a specification."""
        from mcm.algebra.specs import undeclared_types

        assert undeclared_types() == set()

    def test_causality_is_not_a_dependency_edge(self):
        """Spec section 37: do not treat every dependency as causality, and do
        not let a causal claim propagate impact as though it were one."""
        assert implies_dependency(RT.CAUSES) is False

    def test_similarity_is_symmetric_but_not_transitive(self):
        spec = spec_for(RT.SIMILAR_TO)
        assert spec.symmetric is True
        assert spec.transitive is False

    def test_tested_by_does_not_propagate_impact(self):
        """A function does not depend on the test that covers it."""
        assert implies_dependency(RT.TESTED_BY) is False


class TestSubsumption:
    def test_depends_on_premise_matches_concrete_edges(self):
        from mcm.algebra.specs import subsumed_by

        matched = subsumed_by(RT.DEPENDS_ON)
        assert {RT.CALLS, RT.USES, RT.IMPORTS, RT.TESTS, RT.DEPENDS_ON} <= matched
        assert RT.CONTAINS not in matched

    def test_concrete_premise_matches_only_itself(self):
        from mcm.algebra.specs import subsumed_by

        assert subsumed_by(RT.CALLS) == {RT.CALLS}

    def test_derived_types_are_not_subsumed_under_depends_on(self):
        """Otherwise the closure over asserted facts would consume inferences."""
        from mcm.algebra.specs import subsumed_by

        assert RT.POSSIBLY_DEPENDS_ON not in subsumed_by(RT.DEPENDS_ON)


class TestDependencySubsumption:
    def test_call_and_use_edges_are_dependency_edges(self):
        assert implies_dependency(RT.CALLS)
        assert implies_dependency(RT.USES)
        assert implies_dependency(RT.TESTS)

    def test_containment_is_not_a_dependency_edge(self):
        """A file does not depend on the functions it contains."""
        assert implies_dependency(RT.CONTAINS) is False
        assert RT.CONTAINS not in dependency_edge_types()

    def test_dependency_edge_types_include_depends_on_itself(self):
        assert RT.DEPENDS_ON in dependency_edge_types()


class TestComposition:
    def test_composing_two_calls_yields_a_possible_dependency(self):
        """Spec section 35: the conclusion is MAY depend, not depends."""
        result = compose(rel(RT.CALLS, "a", "b"), rel(RT.CALLS, "b", "c"))
        assert result is not None
        assert result.relation_type is RT.POSSIBLY_DEPENDS_ON
        assert result.arguments == ["a", "c"]

    def test_composition_never_produces_a_fact(self):
        result = compose(rel(RT.CALLS, "a", "b"), rel(RT.CALLS, "b", "c"))
        assert result.is_derived
        assert result.relation_type is not RT.CALLS
        assert result.inference.rule == TRANSITIVE_DEPENDENCY_RULE
        assert result.inference.path == ["a", "b", "c"]

    def test_composition_requires_matching_endpoints(self):
        assert compose(rel(RT.CALLS, "a", "b"), rel(RT.CALLS, "x", "c")) is None

    def test_containment_does_not_compose_into_dependency(self):
        """CONTAINS is transitive but is not a dependency edge, so this is refused."""
        assert compose(rel(RT.CONTAINS, "a", "b"), rel(RT.CONTAINS, "b", "c")) is None

    def test_composition_carries_both_evidence_chains(self):
        first = rel(RT.CALLS, "a", "b")
        first.evidence_ids = ["ev1"]
        second = rel(RT.USES, "b", "c")
        second.evidence_ids = ["ev2"]
        result = compose(first, second)
        assert result.evidence_ids == ["ev1", "ev2"]


class TestPathConfidence:
    def test_single_edge_is_unattenuated(self):
        """A direct dependent keeps the confidence its asserted relation carries."""
        assert path_confidence([rel(RT.CALLS, "a", "b")]) == 1.0
        assert path_confidence([rel(RT.TESTS, "a", "b", 0.9)]) == pytest.approx(0.9)

    def test_attenuation_applies_from_the_second_edge(self):
        edges = [rel(RT.USES, "a", "b"), rel(RT.CALLS, "b", "c")]
        assert path_confidence(edges) == pytest.approx(EDGE_DECAY[RT.CALLS])

    def test_import_chains_attenuate_faster_than_call_chains(self):
        """A file-level import says much less about impact than a call does."""
        calls = [rel(RT.CALLS, "a", "b"), rel(RT.CALLS, "b", "c"), rel(RT.CALLS, "c", "d")]
        imports = [rel(RT.IMPORTS, "a", "b"), rel(RT.IMPORTS, "b", "c"),
                   rel(RT.IMPORTS, "c", "d")]
        assert path_confidence(imports) < path_confidence(calls)

    def test_weakest_edge_bounds_the_path(self):
        edges = [rel(RT.CALLS, "a", "b", 0.5), rel(RT.CALLS, "b", "c", 1.0)]
        assert path_confidence(edges) <= 0.5

    def test_longer_chains_are_never_more_confident(self):
        short = [rel(RT.CALLS, "a", "b"), rel(RT.CALLS, "b", "c")]
        long = [*short, rel(RT.CALLS, "c", "d")]
        assert path_confidence(long) < path_confidence(short)

    def test_unmodelled_edge_type_uses_the_pessimistic_default(self):
        edges = [rel(RT.CALLS, "a", "b"), rel(RT.WRITES, "b", "c")]
        assert path_confidence(edges) == pytest.approx(DEFAULT_EDGE_DECAY)

    def test_empty_path_is_an_error_not_a_default(self):
        with pytest.raises(ValueError):
            path_confidence([])


class TestConfidenceBands:
    @pytest.mark.parametrize("value,band", [
        (1.0, "HIGH"), (0.85, "HIGH"), (0.84, "MEDIUM"),
        (0.6, "MEDIUM"), (0.59, "LOW"), (0.0, "UNKNOWN"),
    ])
    def test_bands(self, value, band):
        assert confidence_band(value) == band
