"""Change propagation and the section 29 return shape.

Spec sections 29 and 30, development step 17.

Impact analysis answers "what could break if A changes" with the change left
opaque, so every dependent comes back wearing the same label. Spec section 30 asks
for ``ΔA ⇒ {ΔB, ΔC, ΔD}``: a *typed* change, and per-dependent consequences that
differ according to what actually changed.

The properties under test are the ones that make this more than impact with
labels:

* the same impact set produces different work for different change kinds
* the propagation table, not a chain of conditionals, decides who must be edited
* edits do not propagate past the first hop; behaviour does
* an edge whose meaning is not modelled claims no edit rather than guessing
* every MUST_UPDATE names the source location to edit, from stored evidence
"""

import pytest

from mcm.core.change import Change, ChangeKind
from mcm.core.objects import MCMObject, ObjectType
from mcm.core.relations import MCMRelation, RelationType as RT
from mcm.reasoning.change_propagation import (DEFAULT_EDGE_RESPONSE, EDGE_RESPONSES,
                                              Consequence, explain, propagate,
                                              propagate_from, propagation_json)
from mcm.reasoning.dependency_propagation import analyse_impact

from conftest import (AUTHENTICATE, DECODE_CLAIMS, ENCODE_CLAIMS, JWT, LOGIN,
                      TEST_AUTHENTICATE, VALIDATE_TOKEN)

JWT_PROVIDER_FILE = "repo://app/jwt_provider.py#file"


def change(target, kind, details=""):
    return Change(target_id=target, kind=kind, details=details)


def by_id(result):
    return {p.object.id: p for p in result.predicted}


class TestChangeModel:
    def test_a_change_needs_a_target(self):
        with pytest.raises(ValueError, match="requires a target"):
            Change(target_id="", kind=ChangeKind.BEHAVIOUR)

    def test_kind_accepts_its_string_form(self):
        assert Change("x", "RENAME").kind is ChangeKind.RENAME

    def test_removal_and_rename_break_references(self):
        assert change("x", ChangeKind.REMOVE).breaks_reference
        assert change("x", ChangeKind.RENAME).breaks_reference
        assert not change("x", ChangeKind.SIGNATURE).breaks_reference

    def test_a_signature_change_breaks_calls_but_not_references(self):
        """The distinction the whole table rests on: the name still resolves, the
        call no longer type-checks."""
        signature = change("x", ChangeKind.SIGNATURE)
        assert signature.breaks_call
        assert not signature.breaks_reference

    def test_a_behaviour_change_breaks_neither(self):
        behaviour = change("x", ChangeKind.BEHAVIOUR)
        assert not behaviour.breaks_call
        assert not behaviour.breaks_reference


class TestKindDiscrimination:
    """The same dependents, different work, according to what changed."""

    def test_a_signature_change_requires_editing_the_call_site(self, store):
        result = propagate(store, change(DECODE_CLAIMS, ChangeKind.SIGNATURE))
        assert by_id(result)[VALIDATE_TOKEN].consequence is Consequence.MUST_UPDATE

    def test_a_behaviour_change_requires_editing_nothing(self, store):
        """Identical impact set, zero work: the interface is intact, so every
        consequence is a behaviour question that only the tests can settle."""
        result = propagate(store, change(DECODE_CLAIMS, ChangeKind.BEHAVIOUR))
        assert result.must_update == []
        assert by_id(result)[VALIDATE_TOKEN].consequence is Consequence.MAY_DIFFER

    def test_both_kinds_reach_the_same_objects(self, store):
        """Propagation labels the impact set; it does not change who is in it."""
        signature = propagate(store, change(DECODE_CLAIMS, ChangeKind.SIGNATURE))
        behaviour = propagate(store, change(DECODE_CLAIMS, ChangeKind.BEHAVIOUR))
        assert set(by_id(signature)) == set(by_id(behaviour))

    def test_a_rename_reaches_the_import_statement(self, store):
        """Renaming a module means the import that names it has to change."""
        result = propagate(store, change(JWT, ChangeKind.RENAME))
        predicted = by_id(result)[JWT_PROVIDER_FILE]
        assert predicted.consequence is Consequence.MUST_UPDATE
        assert predicted.path.edge_types == ("IMPORTS",)

    def test_a_signature_change_does_not_reach_the_import_statement(self, store):
        """An import says nothing about any signature inside the module. This is
        the row of the table that a one-size impact report cannot express."""
        result = propagate(store, change(JWT, ChangeKind.SIGNATURE))
        assert by_id(result)[JWT_PROVIDER_FILE].consequence is Consequence.MAY_DIFFER

    def test_a_signature_change_still_reaches_the_call_sites(self, store):
        result = propagate(store, change(JWT, ChangeKind.SIGNATURE))
        must = {p.object.id for p in result.must_update}
        assert {DECODE_CLAIMS, ENCODE_CLAIMS} <= must


class TestEditsDoNotPropagate:
    def test_only_direct_dependents_are_ever_edited(self, store):
        """An indirect caller holds no reference to the change site anywhere in
        its source, whatever the change was."""
        for kind in ChangeKind:
            result = propagate(store, change(DECODE_CLAIMS, kind))
            assert all(p.is_direct for p in result.must_update), kind

    def test_behaviour_still_propagates_down_the_chain(self, store):
        result = propagate(store, change(DECODE_CLAIMS, ChangeKind.RENAME))
        predicted = by_id(result)
        assert predicted[AUTHENTICATE].consequence is Consequence.MAY_DIFFER
        assert predicted[LOGIN].consequence is Consequence.MAY_DIFFER

    def test_indirect_predictions_keep_the_attenuated_confidence(self, store):
        result = propagate(store, change(DECODE_CLAIMS, ChangeKind.RENAME))
        predicted = by_id(result)
        assert predicted[AUTHENTICATE].confidence < predicted[VALIDATE_TOKEN].confidence

    def test_the_explanation_says_why_no_edit_is_needed(self, store):
        result = propagate(store, change(DECODE_CLAIMS, ChangeKind.RENAME))
        assert "does not name the change site" in by_id(result)[AUTHENTICATE].why


class TestPropagationTable:
    def test_an_unmodelled_edge_claims_no_edit(self, empty_store):
        """A dependency edge whose reference semantics are not in the table still
        surfaces the dependent, but never asserts that its source must change.
        Same stance as the constraint engine: what was not checked is not claimed.
        """
        assert RT.DEPENDS_ON not in EDGE_RESPONSES
        assert not DEFAULT_EDGE_RESPONSE.on_reference_break
        assert not DEFAULT_EDGE_RESPONSE.on_call_break

        for object_id, name in ((("a://x"), "x"), (("a://y"), "y")):
            empty_store.put_object(MCMObject(id=object_id, type=ObjectType.FUNCTION,
                                             name=name))
        empty_store.put_relation(MCMRelation(
            id="rel:unmodelled", relation_type=RT.DEPENDS_ON,
            arguments=["a://y", "a://x"]))

        result = propagate(empty_store, change("a://x", ChangeKind.REMOVE))
        assert [p.consequence for p in result.predicted] == [Consequence.MAY_DIFFER]
        assert "not modelled" in result.predicted[0].why

    def test_every_modelled_edge_states_its_reasoning(self):
        for relation_type, response in EDGE_RESPONSES.items():
            assert response.why, relation_type

    def test_a_call_bearing_edge_also_carries_references(self):
        """Nothing can break on a signature change without also breaking on a
        rename: if the source calls it, the source names it."""
        for relation_type, response in EDGE_RESPONSES.items():
            if response.on_call_break:
                assert response.on_reference_break, relation_type


class TestEditSites:
    def test_a_required_edit_names_its_source_location(self, store):
        """Spec section 43 in its most practical form: not "something must change"
        but "line 12 of auth.py"."""
        result = propagate(store, change(DECODE_CLAIMS, ChangeKind.SIGNATURE))
        predicted = by_id(result)[VALIDATE_TOKEN]
        assert predicted.sites
        assert all(":" in site for site in predicted.sites)

    def test_sites_come_from_stored_evidence(self, store):
        result = propagate(store, change(DECODE_CLAIMS, ChangeKind.SIGNATURE))
        predicted = by_id(result)[VALIDATE_TOKEN]
        refs = {store.get_evidence(eid).source_ref
                for relation in predicted.path.relations
                for eid in relation.evidence_ids
                if store.get_evidence(eid) is not None}
        assert set(predicted.sites) <= refs


class TestReportShape:
    def test_work_is_ranked_before_certainty(self, store):
        result = propagate(store, change(JWT, ChangeKind.RENAME))
        consequences = [p.consequence for p in result.predicted]
        assert consequences == sorted(
            consequences, key=lambda c: c is not Consequence.MUST_UPDATE)

    def test_edit_confidence_is_separate_from_the_weakest_claim(self, store):
        """The actionable number should not be dragged down by a long behaviour
        tail, the same way impact separates test_confidence."""
        result = propagate(store, change(JWT, ChangeKind.RENAME))
        assert result.edit_confidence >= result.confidence

    def test_tests_are_named_as_the_evidence_for_may_differ(self, store):
        text = explain(propagate(store, change(DECODE_CLAIMS, ChangeKind.BEHAVIOUR)))
        assert "Run these tests" in text
        assert "test_authenticate_returns_username" in text

    def test_a_change_that_propagates_nowhere_says_so(self, store):
        result = propagate(store, change(TEST_AUTHENTICATE, ChangeKind.REMOVE))
        assert result.predicted == []
        assert "propagates nowhere" in explain(result)

    def test_an_unknown_target_fails_loudly(self, store):
        with pytest.raises(KeyError, match="unknown object"):
            propagate(store, change("repo://app/nope.py#function:nope",
                                    ChangeKind.REMOVE))

    def test_json_carries_the_consequence_and_the_sites(self, store):
        payload = propagation_json(
            propagate(store, change(DECODE_CLAIMS, ChangeKind.SIGNATURE)))
        assert payload["mode"] == "propagate"
        assert payload["change"]["kind"] == "SIGNATURE"
        first = payload["must_update"][0]
        assert first["consequence"] == "MUST_UPDATE"
        assert first["sites"]
        assert first["predicted_change"]["kind"] == "BEHAVIOUR"


class TestSharedTraversal:
    def test_propagation_and_impact_agree_on_who_is_affected(self, store):
        """One traversal in the system. Propagation adds the consequence; it must
        not be able to disagree with impact about the set."""
        impact = analyse_impact(store, DECODE_CLAIMS)
        result = propagate(store, change(DECODE_CLAIMS, ChangeKind.SIGNATURE))
        assert {a.object.id for a in impact.all_affected} == set(by_id(result))

    def test_an_existing_impact_result_can_be_labelled_without_re_walking(self, store):
        impact = analyse_impact(store, DECODE_CLAIMS)
        from_impact = propagate_from(store, impact,
                                     change(DECODE_CLAIMS, ChangeKind.SIGNATURE))
        fresh = propagate(store, change(DECODE_CLAIMS, ChangeKind.SIGNATURE))
        assert ([(p.object.id, p.consequence) for p in from_impact.predicted]
                == [(p.object.id, p.consequence) for p in fresh.predicted])


class TestSection29ReturnShape:
    """Spec section 29 lists what impact analysis must return."""

    def test_apis_and_configuration_are_views_over_the_affected_set(self, store):
        """Empty on this corpus because Python ingestion emits no API or
        Configuration objects. The view is correct by construction the moment an
        extractor does, which is what separates a projection from a stub."""
        impact = analyse_impact(store, JWT)
        assert impact.affected_apis == []
        assert impact.affected_configuration == []

    def test_the_views_filter_the_affected_set_by_type(self, empty_store):
        empty_store.put_object(MCMObject(id="api://login", type=ObjectType.API,
                                         name="POST /login"))
        empty_store.put_object(MCMObject(id="fn://handler",
                                         type=ObjectType.FUNCTION, name="handler"))
        empty_store.put_relation(MCMRelation(
            id="rel:api", relation_type=RT.CALLS,
            arguments=["api://login", "fn://handler"]))
        impact = analyse_impact(empty_store, "fn://handler")
        assert [a.object.id for a in impact.affected_apis] == ["api://login"]
        assert impact.affected_configuration == []
