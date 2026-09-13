"""Agent context generation and context minimisation.

Spec sections 33 and 34, development step 16.

The properties under test:

* every statement the package makes carries the objects, relations and evidence
  that produced it (spec section 43) - a claim with no justification is padding
* minimisation never grows the context, never loses a claim, and the result is
  *verified* by replaying the query against a store holding only the reduced
  context rather than argued from the construction (spec section 34)
* the three entailment clauses each bite: replay catches a missing relation,
  presence catches a missing object, traceability catches missing evidence
* nothing in the package is invented - spec section 37's restraint survives into
  the candidate actions
"""

import math

import pytest

from mcm.agent.context import (CHANGE, CONSTRAINT, DEPENDENCY, IMPACT, TEST,
                               build_context, explain, package_json)
from mcm.agent.minimisation import (_items_of, _sufficient, minimise)
from mcm.core.constraints import Verdict
from mcm.core.evidence import Evidence, EvidenceType
from mcm.core.objects import ObjectType
from mcm.core.relations import RelationType as RT
from mcm.ingestion.repository import RepositoryIngestor
from mcm.retrieval.hybrid import build_indexes
from mcm.storage.sqlite_store import SQLiteStore

from conftest import (AUTHENTICATE, DECODE_CLAIMS, DEMO_REPO, JWT, LOGIN,
                      TEST_AUTHENTICATE, VALIDATE_TOKEN)

TASK = "Fix authentication failure"


@pytest.fixture(scope="module")
def ctx_store():
    store = SQLiteStore()
    RepositoryIngestor(store).ingest(DEMO_REPO, name="app")
    build_indexes(store)
    yield store
    store.close()


@pytest.fixture(scope="module")
def package(ctx_store):
    return build_context(ctx_store, TASK)


@pytest.fixture(scope="module")
def minimal(ctx_store, package):
    return minimise(ctx_store, package)


@pytest.fixture
def fresh_store():
    store = SQLiteStore()
    RepositoryIngestor(store).ingest(DEMO_REPO, name="app")
    build_indexes(store)
    yield store
    store.close()


class TestFocus:
    def test_the_task_text_alone_identifies_the_focus(self, package):
        """Spec section 59's PARSE TASK -> IDENTIFY OBJECTS, with no LLM: the
        focus is hybrid retrieval over the task statement."""
        assert package.focus.id == AUTHENTICATE

    def test_an_explicit_focus_overrides_retrieval(self, ctx_store):
        assert build_context(ctx_store, TASK, focus="jwt").focus.id == JWT

    def test_the_focus_must_be_something_the_engine_can_reason_about(self, ctx_store):
        """Relevance alone is not enough. A File is retrievable and nameable but
        the focus has to participate in dependency relations, or the package has
        no impact, no dependencies and nothing to check."""
        from mcm.algebra.specs import dependency_edge_types
        assert ctx_store.relations_for(package_focus_id(ctx_store),
                                       direction="any",
                                       types=dependency_edge_types())

    def test_a_focus_chosen_by_similarity_alone_is_flagged(self, ctx_store):
        """No term of the task appears in the repository, so nothing but the
        vector channel had an opinion. A score threshold would be a constant tuned
        on the handful of queries at hand; the structural question - did any
        channel that matches real terms support this? - needs no constant."""
        package = build_context(ctx_store, "kubernetes helm chart rollout")
        assert any("similarity alone" in item for item in package.uncertainties)

    def test_a_task_naming_real_symbols_is_not_flagged(self, ctx_store):
        package = build_context(ctx_store, "replace the jwt library")
        assert not any("similarity alone" in item for item in package.uncertainties)

    def test_an_explicit_focus_is_never_second_guessed(self, ctx_store):
        package = build_context(ctx_store, "kubernetes helm", focus="authenticate")
        assert not any("similarity alone" in item for item in package.uncertainties)

    def test_an_empty_repository_fails_loudly(self):
        """Spec section 64: an unsupported query fails rather than returning an
        empty result that reads like an answer."""
        empty = SQLiteStore()
        try:
            with pytest.raises(KeyError, match="nothing in the repository matches"):
                build_context(empty, TASK)
        finally:
            empty.close()


def package_focus_id(store) -> str:
    return build_context(store, TASK).focus.id


class TestClaims:
    def test_every_claim_names_what_produced_it(self, package):
        """Spec section 43. A claim that cannot be traced is not context."""
        for claim in package.claims:
            assert claim.justification.object_ids, claim.statement
            assert claim.justification.relation_ids, claim.statement

    def test_impact_and_dependency_are_separate_claims(self, package):
        assert package.claims_of(IMPACT)
        assert package.claims_of(DEPENDENCY)

    def test_covering_tests_are_their_own_claim_kind(self, package):
        subjects = {claim.subject_id for claim in package.claims_of(TEST)}
        assert TEST_AUTHENTICATE in subjects

    def test_depth_one_is_a_fact_and_deeper_is_an_inference(self, package):
        """Spec section 54: an observed relation and an inference over a chain are
        different kinds of claim, and the package never flattens them."""
        by_subject = {claim.subject_id: claim
                      for claim in package.claims_of(DEPENDENCY)}
        assert by_subject[VALIDATE_TOKEN].is_fact is True
        assert by_subject[DECODE_CLAIMS].is_fact is False

    def test_confidence_carries_its_band(self, package):
        for claim in package.claims:
            assert claim.band in ("HIGH", "MEDIUM", "LOW", "UNKNOWN")

    def test_the_confidence_floor_drops_claims(self, ctx_store, package):
        raised = build_context(ctx_store, TASK, floor=0.96)
        assert len(raised.claims) < len(package.claims)
        assert all(claim.confidence >= 0.96 for claim in raised.claims)

    def test_a_dropped_claim_takes_its_support_with_it(self, ctx_store, package):
        raised = build_context(ctx_store, TASK, floor=0.96)
        assert raised.size < package.size


class TestConstraintsAndUncertainty:
    def test_constraints_are_scoped_to_the_context(self, package):
        assert package.constraints

    def test_undecidable_constraints_become_uncertainties(self, package):
        """A constraint that was never checked never reports SATISFIED, and the
        package says out loud that it could not be decided."""
        undecided = [r for r in package.constraints
                     if r.verdict is Verdict.UNEVALUATABLE]
        assert undecided
        for result in undecided:
            assert any(result.constraint.name in item
                       for item in package.uncertainties)

    def test_absent_history_is_stated_not_implied(self, package):
        assert any("no Git history" in item for item in package.uncertainties)

    def test_absent_decisions_are_stated(self, package):
        """Spec section 33 asks for historical decisions. Ingestion creates no
        Decision objects, so the section is empty - and says why."""
        assert any("Decision objects" in item for item in package.uncertainties)

    def test_inferences_are_counted_in_the_uncertainties(self, package):
        assert any("inferences" in item for item in package.uncertainties)


class TestCandidateActions:
    def test_actions_name_the_tests_to_run(self, package):
        actions = " ".join(action.action for action in package.actions)
        assert "test_authenticate_returns_username" in actions

    def test_every_action_states_its_warrant(self, package):
        for action in package.actions:
            assert action.because


class TestMinimisation:
    def test_minimisation_never_grows_the_context(self, package, minimal):
        assert minimal.size <= package.size

    def test_the_reduced_context_is_verified_by_replay(self, minimal):
        """Spec section 34's C ⊨ Q, tested rather than asserted: the query is
        re-run against a store holding only the reduced context."""
        assert minimal.minimisation.verified
        assert minimal.minimisation.verification == ""

    def test_no_claim_is_lost(self, package, minimal):
        assert ({c.statement for c in minimal.claims}
                == {c.statement for c in package.claims})

    def test_unjustified_retrieval_hits_do_not_survive(self, package, minimal):
        """The gap between what was gathered and what was justified. Retrieval
        returns ten candidates; the ones no claim rests on are not context."""
        assert minimal.minimisation.gathered > minimal.size
        assert len(minimal.entities) < len(package.entities)

    def test_efficiency_is_retained_over_gathered(self, minimal):
        report = minimal.minimisation
        assert math.isclose(report.efficiency, report.after / report.gathered)
        assert 0.0 < report.efficiency <= 1.0

    def test_the_justification_closure_is_already_minimal_here(self, minimal):
        """Not a failure: ablation tried every item and found nothing removable,
        which is a result about the construction rather than a missed reduction.
        The fixture has exactly one evidence record per relation, so there is no
        redundancy to find."""
        assert minimal.minimisation.already_minimal
        assert minimal.minimisation.ablations > 0

    def test_redundant_evidence_is_actually_removed(self, fresh_store):
        """A relation observed twice cites two records; one is enough to trace it.
        This is what the ablation search is for, and it has to bite when there is
        something to bite on."""
        relation = next(r for r in fresh_store.all_relations()
                        if r.relation_type is RT.CALLS)
        extra = Evidence.create(EvidenceType.SOURCE_CODE, "auth.py:2",
                                "second sighting", "ast")
        fresh_store.put_evidence(extra)
        relation.evidence_ids = [*relation.evidence_ids, extra.id]
        fresh_store.put_relation(relation)

        reduced = minimise(fresh_store, build_context(fresh_store, TASK))
        assert ("evidence", extra.id) in reduced.minimisation.removed
        assert reduced.minimisation.verified

    def test_minimising_twice_removes_nothing_more(self, ctx_store, minimal):
        again = minimise(ctx_store, minimal)
        assert again.size == minimal.size
        assert again.minimisation.already_minimal

    def test_the_search_is_bounded_and_says_so(self, ctx_store, package):
        reduced = minimise(ctx_store, package, max_ablations=3)
        assert reduced.minimisation.bound_hit
        assert reduced.minimisation.ablations == 3

    def test_reasoning_paths_keep_their_order(self, minimal):
        """A justification is a path, not a set. Sorting it would turn the section
        43 explanation into a list of names in alphabetical order."""
        for claim in minimal.claims_of(DEPENDENCY):
            assert claim.justification.object_ids[0] == minimal.focus.id


class TestEntailmentClauses:
    """Each clause of C ⊨ Q has to reject something, or it is decoration."""

    def test_replay_rejects_a_relation_the_answer_needs(self, ctx_store, package):
        items = _items_of(package)
        needed = next(r for r in sorted(items.relations)
                      if ctx_store.get_relation(r).relation_type is RT.CALLS)
        verdict = _sufficient(ctx_store, package, items.without("relation", needed),
                              6, None)
        assert not verdict.ok
        assert "replay" in verdict.reason

    def test_presence_rejects_an_object_a_claim_names(self, ctx_store, package):
        """Traversal runs on the argument index and never loads an object, so
        replay alone is blind to this: the closure walks straight through the gap
        at the same confidence while the claim names something that is gone."""
        items = _items_of(package)
        verdict = _sufficient(ctx_store, package,
                              items.without("object", VALIDATE_TOKEN), 6, None)
        assert not verdict.ok
        assert "lost its support" in verdict.reason

    def test_traceability_rejects_a_relation_with_no_evidence_left(
            self, ctx_store, package):
        """Impact analysis never reads evidence, so without this clause ablation
        would strip all of it and the package would still 'answer' the query while
        being unable to show a single source."""
        items = _items_of(package)
        relation = ctx_store.get_relation(next(iter(sorted(items.relations))))
        stripped = items
        for evidence_id in relation.evidence_ids:
            stripped = stripped.without("evidence", evidence_id)
        verdict = _sufficient(ctx_store, package, stripped, 6, None)
        assert not verdict.ok
        assert "evidence" in verdict.reason

    def test_the_focus_itself_is_required(self, ctx_store, package):
        items = _items_of(package)
        verdict = _sufficient(ctx_store, package,
                              items.without("object", package.focus.id), 6, None)
        assert not verdict.ok

    def test_the_full_context_satisfies_every_clause(self, ctx_store, package):
        verdict = _sufficient(ctx_store, package, _items_of(package), 6, None)
        assert verdict.ok
        assert verdict.reproduced == len(
            [c for c in package.claims
             if c.kind in (IMPACT, TEST, DEPENDENCY)])


class TestSerialisation:
    def test_the_package_has_the_section_33_shape(self, ctx_store, minimal):
        payload = package_json(ctx_store, minimal)
        for key in ("goal", "entities", "dependencies", "constraints",
                    "recent_changes", "evidence", "uncertainties",
                    "candidate_actions"):
            assert key in payload

    def test_the_minimisation_is_reported_not_hidden(self, ctx_store, minimal):
        report = package_json(ctx_store, minimal)["minimisation"]
        assert report["verified"] is True
        assert report["gathered"] >= report["justified"] >= report["retained"]

    def test_evidence_is_resolved_to_its_content(self, ctx_store, minimal):
        payload = package_json(ctx_store, minimal)
        assert payload["evidence"]
        assert all("content" in item for item in payload["evidence"])

    def test_the_briefing_labels_facts_and_inferences(self, ctx_store, minimal):
        text = explain(ctx_store, minimal)
        assert "[FACT ]" in text
        assert "[INFER]" in text
        assert "Uncertainties" in text
