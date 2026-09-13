"""Pre-action simulation, and prediction against observation.

Spec sections 60, 61 and 62. Development step 18.

Two halves. Section 60 builds the state a change would produce and checks it;
sections 61 and 62 take the predictions the model would have made before each
historical commit and compare them with what the commit did.

The properties under test are mostly about restraint:

* a simulation reports and never decides - spec section 60 leaves that to the agent
* a change that closes no relation cannot move a constraint verdict, and the report
  says that rather than printing an empty diff that reads like an all-clear
* co-change can test an edit prediction and cannot test a behaviour prediction, and
  the report refuses to score what it cannot measure
* the section 62 update moves propagation strength, never relation belief, and is
  never applied
"""

import shutil
import sys
from pathlib import Path

import pytest

from mcm.algebra.confidence import EDGE_DECAY
from mcm.core.change import ADDED, Change, ChangeKind
from mcm.core.constraints import ConstraintType, Verdict
from mcm.core.relations import RelationType as RT
from mcm.ingestion.history import HistoryIngestor
from mcm.ingestion.repository import RepositoryIngestor
from mcm.reasoning.prediction import (PRIOR_STRENGTH, accuracy_json, evaluate,
                                      learned_decay, observations_from_history)
from mcm.reasoning.prediction import explain as accuracy_explain
from mcm.reasoning.simulation import simulate, simulation_json
from mcm.reasoning.simulation import explain as simulation_explain
from mcm.storage.sqlite_store import SQLiteStore

from conftest import (AUTHENTICATE, CREATE_TOKEN, DEMO_REPO, ENCODE_CLAIMS,
                      TEST_AUTHENTICATE, VALIDATE_TOKEN)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from prediction_fixture import build as build_prediction_repo  # noqa: E402

GEO_AREA = "repo://geo/geometry.py#function:area"
GEO_SUMMARISE = "repo://geo/report.py#function:summarise"


def change(target, kind, details=""):
    return Change(target_id=target, kind=kind, details=details)


# --- section 60 ------------------------------------------------------------

class TestSimulationStructure:
    def test_a_removal_closes_the_targets_relations(self, store):
        result = simulate(store, change(ENCODE_CLAIMS, ChangeKind.REMOVE))
        assert result.structural
        assert result.closed_relations > 0

    def test_a_behaviour_change_closes_nothing(self, store):
        """K(t+1) is structurally identical, so no predicate over the graph can
        move. That is a result, not a gap."""
        result = simulate(store, change(ENCODE_CLAIMS, ChangeKind.BEHAVIOUR))
        assert not result.structural
        assert result.closed_relations == 0
        assert result.verdict_changes == []

    def test_a_rename_is_structural_because_identity_is_the_name(self, store):
        """Object identity is built from the qualname (spec section 20), so a
        renamed definition is a different object and every edge into the old
        identity stops being true."""
        assert simulate(store, change(ENCODE_CLAIMS, ChangeKind.RENAME)).structural

    def test_the_original_store_is_never_modified(self, store):
        before = len(list(store.all_relations()))
        simulate(store, change(ENCODE_CLAIMS, ChangeKind.REMOVE))
        assert len(list(store.all_relations())) == before

    def test_an_unknown_target_fails_loudly(self, store):
        with pytest.raises(KeyError, match="unknown object"):
            simulate(store, change("repo://app/nope.py#function:x", ChangeKind.REMOVE))


class TestSimulationChecks:
    """Spec section 60 lists five checks."""

    def test_a_removal_can_break_a_constraint_that_currently_holds(self, store):
        """Inlining encode_claims into create_token means create_token no longer
        depends on jwt_provider, which is exactly what the constraint forbids."""
        result = simulate(store, change(ENCODE_CLAIMS, ChangeKind.REMOVE))
        names = {v.name for v in result.regressions}
        assert "token_creation_goes_through_provider" in names

    def test_a_regression_names_the_verdict_it_moved_from(self, store):
        result = simulate(store, change(ENCODE_CLAIMS, ChangeKind.REMOVE))
        regression = next(v for v in result.regressions
                          if v.name == "token_creation_goes_through_provider")
        assert regression.before is Verdict.SATISFIED
        assert regression.after is Verdict.VIOLATED
        assert regression.messages

    def test_architectural_regressions_are_reported_separately(self, store):
        result = simulate(store, change(ENCODE_CLAIMS, ChangeKind.REMOVE))
        assert all(v.type is ConstraintType.ARCHITECTURAL_CONSTRAINT
                   for v in result.architectural_regressions)

    def test_dependencies_name_the_edit_sites(self, store):
        result = simulate(store, change(ENCODE_CLAIMS, ChangeKind.REMOVE))
        must = {p.object.id: p for p in result.propagation.must_update}
        assert CREATE_TOKEN in must
        assert must[CREATE_TOKEN].sites

    def test_tests_and_apis_are_reported(self, store):
        result = simulate(store, change(ENCODE_CLAIMS, ChangeKind.REMOVE))
        assert {a.object.id for a in result.affected_tests}
        assert result.affected_apis == []

    def test_unchanged_verdicts_are_not_reported(self, store):
        """The diff is the point. A constraint that held before and after is not
        news."""
        result = simulate(store, change(ENCODE_CLAIMS, ChangeKind.REMOVE))
        assert "auth_must_not_import_jwt_directly" not in {
            v.name for v in result.verdict_changes}


class TestSimulationReporting:
    def test_no_recommendation_is_offered(self, store):
        """Spec section 60: "then let the agent decide whether to proceed". There
        is no should_proceed field to read."""
        result = simulate(store, change(ENCODE_CLAIMS, ChangeKind.REMOVE))
        assert not hasattr(result, "should_proceed")
        assert not hasattr(result, "safe")

    def test_a_structurally_inert_change_says_why_the_diff_is_empty(self, store):
        text = simulation_explain(simulate(store,
                                           change(ENCODE_CLAIMS, ChangeKind.BEHAVIOUR)))
        assert "closes no" in text
        assert "tests below" in text

    def test_the_report_states_that_nothing_was_written(self, store):
        text = simulation_explain(simulate(store, change(ENCODE_CLAIMS,
                                                         ChangeKind.REMOVE)))
        assert "Nothing was written" in text

    def test_json_carries_the_verdict_diff(self, store):
        payload = simulation_json(simulate(store, change(ENCODE_CLAIMS,
                                                         ChangeKind.REMOVE)))
        assert payload["mode"] == "simulate"
        assert payload["structural"] is True
        assert "token_creation_goes_through_provider" in payload["regressions"]


# --- sections 61 and 62 -----------------------------------------------------

pytestmark_git = pytest.mark.skipif(shutil.which("git") is None,
                                    reason="git not on PATH")


@pytest.fixture(scope="module")
def predicted():
    """A repository whose history contains a signature change and its call-site fix."""
    import tempfile

    tmp = tempfile.TemporaryDirectory()
    store = SQLiteStore()
    HistoryIngestor(store).ingest(build_prediction_repo(Path(tmp.name) / "repo"),
                                  name="geo")
    yield store
    store.close()
    tmp.cleanup()


@pytestmark_git
class TestObservations:
    def test_history_records_what_kind_of_change_each_commit_made(self, predicted):
        kinds = {o.kind for o in observations_from_history(predicted)}
        assert ChangeKind.SIGNATURE in kinds
        assert ChangeKind.BEHAVIOUR in kinds

    def test_additions_are_not_observations(self, predicted):
        """Nothing could depend on a definition that did not exist, so there is no
        prediction to test."""
        assert all(o.kind != ADDED for o in observations_from_history(predicted))

    def test_the_prediction_reads_the_state_before_the_commit(self, predicted):
        for observation in observations_from_history(predicted):
            assert observation.before < observation.date

    def test_observations_are_oldest_first(self, predicted):
        dates = [o.date for o in observations_from_history(predicted)]
        assert dates == sorted(dates)


@pytestmark_git
class TestPredictionError:
    def test_a_signature_change_predicts_the_call_site_edit_that_happened(
            self, predicted):
        """The commit that added a parameter also fixed its caller, because a
        signature change that does not is broken code. This is the one comparison
        a diff can settle."""
        report = evaluate(predicted)
        signature = next(c for c in report.comparisons
                         if c.observation.kind is ChangeKind.SIGNATURE)
        assert GEO_SUMMARISE in signature.must_update
        assert GEO_SUMMARISE in signature.edit_hits
        assert signature.edit_false_alarms == frozenset()

    def test_edit_precision_is_computed_over_must_update_only(self, predicted):
        report = evaluate(predicted)
        assert report.edit_precision == 1.0

    def test_an_upstream_co_change_is_not_counted_as_an_error(self, predicted):
        """Co-change is symmetric; propagation is directional. When one commit
        edits a callee and its caller, the caller's own observation records the
        callee as co-changed and cannot predict it - nothing downstream of the
        caller is the callee."""
        report = evaluate(predicted)
        caller = next(c for c in report.comparisons
                      if c.observation.target_id == GEO_SUMMARISE)
        assert GEO_AREA in caller.upstream_missed
        assert caller.unexplained_missed == frozenset()

    def test_every_miss_is_unreachable_by_construction(self, predicted):
        """Anything the traversal reaches is predicted as something, so a miss is
        always a gap in the graph rather than a mislabelling."""
        for comparison in evaluate(predicted).comparisons:
            assert not (comparison.missed & comparison.predicted)

    def test_abstaining_is_not_scored_as_perfect(self, empty_store):
        """A system that predicts nothing has not achieved precision 1.0."""
        report = evaluate(empty_store)
        assert report.edit_precision is None
        assert report.edit_recall is None


@pytestmark_git
class TestWhatCannotBeMeasured:
    def test_a_corpus_of_pure_behaviour_changes_is_refused(self, tmp_path):
        """The original history fixture has one change to an existing definition,
        a behaviour change with no co-edits. A diff cannot confirm or deny a
        behaviour change, so no number is reported at all."""
        from history_fixture import build as build_history_repo

        store = SQLiteStore()
        HistoryIngestor(store).ingest(build_history_repo(tmp_path / "repo"),
                                      name="app")
        report = evaluate(store)
        assert report.observations
        assert not report.usable
        text = accuracy_explain(report)
        assert "cannot confirm or deny" in text
        assert "precision" not in text
        store.close()

    def test_behaviour_predictions_are_reported_as_untestable(self, predicted):
        text = accuracy_explain(evaluate(predicted))
        assert "neither confirmed nor refuted" in text

    def test_the_sample_size_is_stated(self, predicted):
        assert "Nothing here is a result" in accuracy_explain(evaluate(predicted))

    def test_an_uningested_history_says_so(self, empty_store):
        assert "nothing to compare" in accuracy_explain(evaluate(empty_store))


@pytestmark_git
class TestConfidenceUpdate:
    """Spec section 62, as a transparent heuristic."""

    def test_the_update_moves_propagation_strength_not_relation_belief(self,
                                                                       predicted):
        """An AST-observed call is certainly a call, whatever the co-change data
        says. What the evidence bears on is how much of a change survives the
        edge, which is EDGE_DECAY."""
        before = dict(EDGE_DECAY)
        learned = learned_decay(evaluate(predicted))
        assert learned
        assert EDGE_DECAY == before
        for relation in predicted.all_relations(include_historical=True):
            if relation.relation_type is RT.CALLS:
                assert relation.confidence == 1.0

    def test_the_posterior_sits_between_the_prior_and_the_observation(self,
                                                                      predicted):
        for edge, (prior, posterior, count) in learned_decay(evaluate(predicted)).items():
            observed = evaluate(predicted).by_edge[edge].observed_rate
            assert min(prior, observed) <= posterior <= max(prior, observed)

    def test_with_no_observations_the_posterior_is_the_prior(self, predicted):
        """The property that makes the heuristic safe to publish: evidence has to
        exist before it moves anything."""
        report = evaluate(predicted)
        for edge, tally in report.by_edge.items():
            tally.predicted_edits = 0
            tally.confirmed_edits = 0
        assert learned_decay(report) == {}

    def test_one_observation_cannot_overturn_the_prior(self, predicted):
        """PRIOR_STRENGTH is what stops a single commit from retuning a constant
        every impact answer depends on."""
        for edge, (prior, posterior, count) in learned_decay(evaluate(predicted)).items():
            assert count <= PRIOR_STRENGTH
            assert abs(posterior - prior) < 0.1

    def test_nothing_is_applied(self, predicted):
        payload = accuracy_json(evaluate(predicted))
        assert payload["applied"] is False

    def test_json_reports_both_the_prior_and_the_posterior(self, predicted):
        payload = accuracy_json(evaluate(predicted))
        assert payload["learned_decay"]
        for entry in payload["learned_decay"].values():
            assert {"prior", "posterior", "observations"} == set(entry)


class TestUnresolvableTargets:
    """A definition can be visible to the differ and absent from the graph.

    Real history hits this: `flask/tests/conftest.py` defines a class inside a
    function body, which the diff reports as a changed definition and the symbol
    extractor never turns into an object. Before this was handled, `evaluate`
    raised KeyError and took the whole run with it, which is how it went
    unnoticed on a fixture where every target resolves.
    """

    def _observation(self, target_id):
        from datetime import datetime, timezone

        from mcm.reasoning.prediction import Observation
        return Observation(
            commit_id="commit://nope", commit_name="a commit",
            date=datetime(2026, 1, 1, tzinfo=timezone.utc),
            target_id=target_id, kind=ChangeKind.SIGNATURE,
            co_changed=frozenset(),
        )

    def test_an_unknown_target_is_skipped_rather_than_raising(self, predicted):
        ghost = self._observation("repo://app/nowhere.py#class:f.Inner")
        report = evaluate(predicted, [ghost])
        assert report.unresolved_targets == 1
        assert report.comparisons == []

    def test_known_targets_are_still_scored_alongside_an_unknown_one(self, predicted):
        """One unresolvable observation must not discard the rest of the run."""
        ghost = self._observation("repo://app/nowhere.py#class:f.Inner")
        # Taken from the store rather than written down, so the test does not
        # depend on what the fixture happens to name its repository.
        real = self._observation(observations_from_history(predicted)[0].target_id)
        report = evaluate(predicted, [ghost, real])
        assert report.unresolved_targets == 1
        assert len(report.comparisons) == 1

    def test_skipped_observations_are_reported_not_hidden(self, predicted):
        ghost = self._observation("repo://app/nowhere.py#class:f.Inner")
        report = evaluate(predicted, [ghost])
        assert accuracy_json(report)["unresolved_targets"] == 1
        assert "skipped" in accuracy_explain(report)
