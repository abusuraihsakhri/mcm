"""Rule engine tests (spec sections 35, 36).

The load-time validation tests matter as much as the inference tests: a rule that
would produce a fact, or bind nothing, should be rejected when the file is read
rather than when it first fires against a repository that happens to trigger it.
"""

import pytest

from mcm.algebra.dependency import dependents_of
from mcm.core.relations import RelationType as RT
from mcm.reasoning.engine import derive
from mcm.reasoning.rules import (DEFAULT_RULES, Rule, RuleError, load_rules,
                                 rules_producing)

from conftest import AUTHENTICATE, AUTH_FILE, JWT, LOGIN, TEST_AUTHENTICATE, USER_CLASS


@pytest.fixture(scope="module")
def rules():
    return load_rules()


@pytest.fixture(scope="module")
def derivation(store, rules):
    return derive(store, rules)


def write_rules(tmp_path, body: str):
    path = tmp_path / "r.yaml"
    path.write_text(body, encoding="utf-8")
    return path


class TestRuleLoading:
    def test_ships_a_valid_default_rule_set(self, rules):
        assert DEFAULT_RULES.exists()
        assert {r.name for r in rules} >= {"transitive_dependency",
                                           "container_inherits_member_dependency"}
        assert all(isinstance(r, Rule) for r in rules)

    def test_abstract_premise_expands_to_concrete_types(self, rules):
        rule = next(r for r in rules if r.name == "transitive_dependency")
        assert {RT.CALLS, RT.USES, RT.TESTS} <= rule.premises[0].types

    def test_rules_producing_answers_the_composition_rules_question(self, rules):
        producers = rules_producing(RT.POSSIBLY_DEPENDS_ON, rules)
        assert {r.name for r in producers} == {"transitive_dependency",
                                               "container_inherits_member_dependency"}

    def test_unbound_conclusion_variable_is_rejected_at_load(self, tmp_path):
        path = write_rules(tmp_path, """
rules:
  - name: bad
    premises:
      - [CALLS, "?a", "?b"]
    conclusion: [POSSIBLY_DEPENDS_ON, "?a", "?unbound"]
""")
        with pytest.raises(RuleError, match="unbound variables"):
            load_rules(path)

    def test_unknown_relation_type_is_rejected_at_load(self, tmp_path):
        path = write_rules(tmp_path, """
rules:
  - name: bad
    premises:
      - [NOT_A_RELATION, "?a", "?b"]
    conclusion: [POSSIBLY_DEPENDS_ON, "?a", "?b"]
""")
        with pytest.raises(RuleError, match="unknown relation type"):
            load_rules(path)

    def test_a_rule_may_not_conclude_a_fact(self, tmp_path):
        """Spec Rule 2, enforced when the file is read."""
        path = write_rules(tmp_path, """
rules:
  - name: bad
    premises:
      - [CALLS, "?a", "?b"]
      - [CALLS, "?b", "?c"]
    conclusion: [CALLS, "?a", "?c"]
""")
        with pytest.raises(RuleError, match="also matches as a premise"):
            load_rules(path)

    def test_empty_rule_file_is_rejected(self, tmp_path):
        with pytest.raises(RuleError, match="declares no rules"):
            load_rules(write_rules(tmp_path, "rules: []\n"))


class TestForwardChaining:
    def test_reaches_a_fixpoint(self, derivation):
        assert derivation.complete
        assert derivation.iterations < 12
        assert derivation.derived

    def test_every_derived_relation_is_marked_as_an_inference(self, derivation):
        assert all(r.is_derived for r in derivation.relations)
        assert all(r.inference.rule for r in derivation.relations)

    def test_derivations_cite_asserted_premises_only(self, store, derivation):
        """Confidence is scored over original observations, not intermediates."""
        asserted = {r.id for r in store.all_relations()}
        for relation in derivation.relations:
            assert set(relation.inference.premise_relation_ids) <= asserted

    def test_nothing_is_written_to_the_store(self, store, derivation):
        stored = {r.id for r in store.all_relations(include_derived=True)}
        assert {r.id for r in derivation.relations}.isdisjoint(stored)

    def test_transitivity_chains_through_multiple_iterations(self, derivation):
        """login is four hops from jwt, so it needs the fixpoint, not one pass."""
        reached = derivation.between(LOGIN, JWT)
        assert reached
        assert reached[0].properties["depth"] == 4

    def test_distinct_clause_blocks_self_dependency(self, derivation):
        assert not any(r.arguments[0] == r.arguments[1] for r in derivation.relations)

    def test_bounds_are_reported_when_hit(self, store, rules):
        bounded = derive(store, rules, max_iterations=1)
        assert bounded.hit_iteration_limit is True
        assert bounded.complete is False


class TestCrossCheckAgainstClosure:
    """The BFS closure and the rule fixpoint are independent implementations of
    the same reasoning. Where both apply they must agree."""

    def test_reachability_agrees_beyond_depth_one(self, store, derivation):
        bfs = dependents_of(store, JWT)
        bfs_indirect = {oid for oid, path in bfs.paths.items() if path.depth > 1}
        engine = {r.arguments[0] for r in derivation.of_type(RT.POSSIBLY_DEPENDS_ON)
                  if r.arguments[1] == JWT}
        assert bfs_indirect <= engine

    def test_confidence_agrees_where_both_reach(self, store, derivation):
        bfs = dependents_of(store, JWT)
        by_source = {r.arguments[0]: r for r in derivation.of_type(RT.POSSIBLY_DEPENDS_ON)
                     if r.arguments[1] == JWT}
        compared = 0
        for object_id, path in bfs.paths.items():
            if path.depth > 1 and object_id in by_source:
                assert by_source[object_id].confidence == pytest.approx(path.confidence)
                compared += 1
        assert compared >= 5

    def test_engine_reaches_containers_the_closure_cannot(self, store, derivation):
        """CONTAINS is not a dependency edge, so the closure never crosses it.
        A rule that reads both families can, which is what the engine adds."""
        bfs = dependents_of(store, JWT)
        engine = {r.arguments[0] for r in derivation.of_type(RT.POSSIBLY_DEPENDS_ON)
                  if r.arguments[1] == JWT}
        assert "repo://app" in engine
        assert "repo://app" not in bfs.paths

    def test_class_inherits_its_methods_dependencies(self, store):
        """A class depends on what its methods depend on."""
        derivation = derive(store)
        reached = derivation.between(USER_CLASS, USER_CLASS)
        assert not reached  # no self-dependency
        by_rule = [r for r in derivation.relations
                   if r.properties.get("rule") == "container_inherits_member_dependency"]
        assert by_rule


class TestInverseRules:
    def test_tested_by_is_derived_from_tests(self, derivation):
        inverses = derivation.of_type(RT.TESTED_BY)
        assert any(r.arguments == [AUTHENTICATE, TEST_AUTHENTICATE] for r in inverses)

    def test_part_of_is_derived_from_contains(self, derivation):
        inverses = derivation.of_type(RT.PART_OF)
        assert any(r.arguments == [AUTHENTICATE, AUTH_FILE] for r in inverses)

    def test_inverse_derivations_are_one_hop(self, derivation):
        for relation in derivation.of_type(RT.TESTED_BY):
            assert relation.properties["depth"] == 1
