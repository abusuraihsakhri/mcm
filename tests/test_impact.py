"""Impact analysis and the spec section 65 acceptance scenario.

Section 65 defines the minimum proof-of-concept:

    Query: "What could break if I replace the JWT implementation?"

    JWT -> validate_token -> authenticate -> test_auth

    "JWT replacement directly affects validate_token. validate_token is called by
    authenticate. authenticate is covered by test_auth. Therefore test_auth should
    be run after the change."

These tests assert that chain, its direction, its explanation and its
fact/inference labelling.
"""

import pytest

from mcm.agent.query import QueryType, run_query
from mcm.algebra.dependency import dependencies_of, dependents_of
from mcm.core.objects import ObjectType
from mcm.core.relations import RelationType as RT
from mcm.reasoning.dependency_propagation import IMPACT_RULE, analyse_impact, explain

from conftest import (AUTHENTICATE, CREATE_TOKEN, DECODE_CLAIMS, ENCODE_CLAIMS, JWT,
                      LOGIN, TEST_AUTHENTICATE, TEST_LOGIN, VALIDATE_TOKEN)


@pytest.fixture(scope="module")
def impact(store):
    return analyse_impact(store, JWT)


class TestSection65Scenario:
    """The minimum proof-of-concept from spec section 65."""

    def test_jwt_replacement_directly_affects_the_provider_functions(self, impact):
        direct = {a.object.id for a in impact.direct}
        assert DECODE_CLAIMS in direct
        assert ENCODE_CLAIMS in direct

    def test_the_full_chain_is_reachable(self, impact):
        affected = {a.object.id for a in impact.all_affected}
        for object_id in (DECODE_CLAIMS, VALIDATE_TOKEN, AUTHENTICATE, LOGIN,
                          TEST_AUTHENTICATE, TEST_LOGIN):
            assert object_id in affected

    def test_the_reasoning_path_follows_the_spec_narrative(self, impact):
        """JWT -> decode_claims -> validate_token -> authenticate."""
        by_id = {a.object.id: a for a in impact.all_affected}
        assert by_id[AUTHENTICATE].path.objects == (
            JWT, DECODE_CLAIMS, VALIDATE_TOKEN, AUTHENTICATE)
        assert by_id[AUTHENTICATE].path.edge_types == ("USES", "CALLS", "CALLS")

    def test_tests_are_identified_for_rerun(self, impact):
        names = {a.object.name for a in impact.affected_tests}
        assert names == {"test_authenticate_returns_username",
                         "test_login_rejects_inactive_user"}
        assert all(a.object.type is ObjectType.TEST for a in impact.affected_tests)

    def test_test_recommendation_is_reported_with_its_own_confidence(self, impact):
        """The actionable claim is scored separately from the weakest claim in the
        report, which is dominated by loose file-level import chains."""
        assert impact.test_confidence > impact.confidence


class TestFactVersusInference:
    """Spec Rule 2 and section 54."""

    def test_direct_dependents_are_facts(self, impact):
        assert all(a.is_fact for a in impact.direct)
        assert all(a.derived is None for a in impact.direct)

    def test_indirect_dependents_are_inferences(self, impact):
        assert all(not a.is_fact for a in impact.indirect)
        assert all(a.derived is not None for a in impact.indirect)

    def test_derived_relations_are_possibly_affects(self, impact):
        for affected in impact.indirect:
            assert affected.derived.relation_type is RT.POSSIBLY_AFFECTS
            assert affected.derived.is_derived
            assert affected.derived.inference.rule == IMPACT_RULE

    def test_derived_relations_cite_their_premises(self, impact):
        by_id = {a.object.id: a for a in impact.indirect}
        derived = by_id[AUTHENTICATE].derived
        assert len(derived.inference.premise_relation_ids) == 3
        assert derived.inference.path == [JWT, DECODE_CLAIMS, VALIDATE_TOKEN, AUTHENTICATE]

    def test_inferences_are_not_written_to_the_store(self, store, impact):
        """analyse_impact returns derivations; persisting them is a separate act."""
        derived_ids = {a.derived.id for a in impact.indirect}
        stored = {r.id for r in store.all_relations(include_derived=True)}
        assert derived_ids.isdisjoint(stored)


class TestConfidenceDecay:
    def test_confidence_falls_with_depth_along_a_call_chain(self, impact):
        by_id = {a.object.id: a for a in impact.all_affected}
        chain = [DECODE_CLAIMS, VALIDATE_TOKEN, AUTHENTICATE, LOGIN]
        scores = [by_id[object_id].confidence for object_id in chain]
        assert scores == sorted(scores, reverse=True)
        assert len(set(scores)) == len(scores)

    def test_call_chains_outrank_import_chains_at_the_same_depth(self, impact):
        by_id = {a.object.id: a for a in impact.all_affected}
        auth_file = next(a for a in impact.all_affected
                         if a.object.name == "auth.py")
        assert by_id[VALIDATE_TOKEN].confidence > auth_file.confidence
        assert by_id[VALIDATE_TOKEN].path.depth == auth_file.path.depth


class TestClosureDirection:
    def test_impact_is_the_reverse_of_dependency(self, store):
        """A > B means A's behaviour is a function of B's, so impact runs backwards."""
        forward = dependencies_of(store, AUTHENTICATE)
        backward = dependents_of(store, AUTHENTICATE)
        assert VALIDATE_TOKEN in forward.paths
        assert LOGIN in backward.paths
        assert LOGIN not in forward.paths
        assert VALIDATE_TOKEN not in backward.paths

    def test_containment_does_not_propagate_impact(self, store):
        """Changing a function does not imply changing every sibling in its file."""
        affected = dependents_of(store, CREATE_TOKEN).paths
        assert VALIDATE_TOKEN not in affected

    def test_depth_limit_is_reported_not_hidden(self, store):
        shallow = dependents_of(store, JWT, max_depth=2)
        assert shallow.truncated is True
        assert dependents_of(store, JWT, max_depth=10).truncated is False

    def test_unknown_object_raises(self, store):
        with pytest.raises(KeyError):
            analyse_impact(store, "repo://app/nope.py#function:nope")


class TestExplanation:
    """Spec section 43: every inferred answer must be traceable."""

    def test_explanation_names_the_relations_crossed(self, store, impact):
        text = explain(store, impact)
        assert "<-CALLS-" in text
        assert "<-USES-" in text

    def test_explanation_cites_source_locations(self, store, impact):
        text = explain(store, impact)
        assert "jwt_provider.py:13" in text
        assert "auth.py:" in text

    def test_explanation_labels_facts_and_inferences(self, store, impact):
        text = explain(store, impact)
        assert "[FACT ]" in text
        assert "[INFER]" in text

    def test_explanation_states_the_test_recommendation(self, store, impact):
        text = explain(store, impact)
        assert "Tests to run after the change" in text
        assert "test_authenticate_returns_username" in text

    def test_object_with_no_dependents_says_so(self, store):
        text = explain(store, analyse_impact(store, LOGIN))
        assert "test_login_rejects_inactive_user" in text


class TestQueryInterface:
    def test_impact_query_returns_the_section_42_shape(self, store):
        payload = run_query(store, QueryType.IMPACT, "validate_token")
        assert payload["mode"] == "impact"
        assert payload["target"]["name"] == "validate_token"
        assert payload["affected_tests"]
        for entry in payload["direct_dependents"]:
            assert entry["kind"] == "fact"
        for entry in payload["indirect_dependents"]:
            assert entry["kind"] == "inference"
            assert entry["derived_relation_id"]

    def test_dependency_query_runs_forward(self, store):
        payload = run_query(store, QueryType.DEPENDENCY, "authenticate")
        names = {entry["object"]["name"] for entry in payload["depends_on"]}
        assert {"validate_token", "create_token", "decode_claims"} <= names
        assert "login" not in names

    def test_lookup_returns_every_match(self, store):
        payload = run_query(store, QueryType.LOOKUP, "authenticate")
        assert payload["matches"]
        assert payload["matches"][0]["qualname"] == "authenticate"

    def test_ambiguous_reference_raises_instead_of_choosing(self, empty_store):
        """Two objects can share a bare name. Picking one silently would put an
        arbitrary choice underneath every downstream inference."""
        from mcm.core.objects import MCMObject

        for relpath in ("a.py", "b.py"):
            empty_store.put_object(MCMObject(
                id=f"repo://x/{relpath}#function:handler", type=ObjectType.FUNCTION,
                name="handler", properties={"relpath": relpath, "qualname": "handler"}))

        with pytest.raises(KeyError, match="ambiguous"):
            run_query(empty_store, QueryType.IMPACT, "handler")

    def test_unknown_reference_raises(self, store):
        with pytest.raises(KeyError, match="no object matches"):
            run_query(store, QueryType.IMPACT, "does_not_exist")
