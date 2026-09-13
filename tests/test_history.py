"""Git ingestion and regression analysis (spec sections 19 steps 7-9, 37, 66).

The fixture repository in examples/history_fixture.py has four commits and one
planted regression: commit 4 edits ``decode_claims``, which nothing in auth.py
names. Finding it means following the dependency chain rather than searching for
the symptom, which is the point of the section 66 query.
"""

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mcm.core.objects import ObjectType
from mcm.core.provenance import ExtractionMethod
from mcm.core.relations import RelationType as RT
from mcm.ingestion.git import GitError, GitReader
from mcm.ingestion.history import (HistoryIngestor, _Definition,
                                   _changed_definitions)
from mcm.ingestion.sources import GitRevisionProvider, directories_for, skipped
from mcm.reasoning.causal_reasoning import explain, regression_candidates
from mcm.storage.sqlite_store import SQLiteStore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from history_fixture import build  # noqa: E402

AUTHENTICATE = "repo://app/auth.py#function:authenticate"
VALIDATE_TOKEN = "repo://app/auth.py#function:validate_token"
DECODE_CLAIMS = "repo://app/jwt_provider.py#function:decode_claims"
LOGIN = "repo://app/auth.py#function:login"
USER_CLASS = "repo://app/user.py#class:User"
TEST_AUTHENTICATE = "repo://app/tests/test_auth.py#test:test_authenticate_returns_username"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    return build(tmp_path_factory.mktemp("history") / "repo")


@pytest.fixture(scope="module")
def reader(repo):
    return GitReader(repo)


@pytest.fixture(scope="module")
def history(repo):
    store = SQLiteStore()
    report = HistoryIngestor(store).ingest(repo, name="app")
    yield store, report
    store.close()


@pytest.fixture(scope="module")
def after_tests(history):
    """The moment the tests landed: the last point known to work."""
    _, report = history
    return report.commits[2].commit.date


def commit_by_subject(report, fragment):
    return next(e.commit for e in report.commits if fragment in e.commit.subject)


def _def(digest, kind="function", parameters=(), ast="tree"):
    return _Definition(kind=kind, digest=digest, parameters=tuple(parameters),
                       ast_digest=ast)


class TestGitReader:
    def test_detects_a_repository(self, reader, tmp_path):
        assert reader.is_repository()
        assert not GitReader(tmp_path).is_repository()

    def test_commits_are_oldest_first(self, reader):
        commits = reader.commits()
        assert len(commits) == 4
        assert commits[0].subject == "add token creation and validation"
        assert commits[-1].subject == "rename claim key returned by decode_claims"
        assert [c.date for c in commits] == sorted(c.date for c in commits)

    def test_root_commit_has_no_parents(self, reader):
        commits = reader.commits()
        assert commits[0].is_root
        assert commits[1].parents == (commits[0].sha,)

    def test_max_count_limits_the_window(self, reader):
        assert len(reader.commits(max_count=2)) == 2

    def test_files_at_a_revision(self, reader):
        commits = reader.commits()
        first = set(reader.files_at(commits[0].sha))
        last = set(reader.files_at(commits[-1].sha))
        assert first == {"auth.py", "jwt_provider.py"}
        assert "user.py" in last and "tests/test_auth.py" in last

    def test_blob_returns_content_at_that_revision(self, reader):
        commits = reader.commits()
        before = reader.blob(commits[0].sha, "jwt_provider.py")
        after = reader.blob(commits[-1].sha, "jwt_provider.py")
        assert b"subject" not in before
        assert b"subject" in after

    def test_blob_of_a_missing_path_is_none(self, reader):
        assert reader.blob(reader.commits()[0].sha, "user.py") is None

    def test_batched_blobs_match_reading_them_one_at_a_time(self, reader):
        """The batch path replaces per-file reads, so it must agree byte for byte."""
        sha = reader.commits()[-1].sha
        paths = reader.files_at(sha)
        batched = reader.blobs(sha, paths)
        assert batched == {p: reader.blob(sha, p) for p in paths}

    def test_batched_blobs_omit_paths_absent_at_that_revision(self, reader):
        first = reader.commits()[0].sha
        batched = reader.blobs(first, ["jwt_provider.py", "user.py"])
        assert "jwt_provider.py" in batched
        assert "user.py" not in batched  # added by a later commit

    def test_batched_blobs_stay_aligned_after_a_missing_path(self, reader):
        """A miss returns no body; mistaking that for one would shift every result."""
        sha = reader.commits()[-1].sha
        batched = reader.blobs(sha, ["does_not_exist.py", "jwt_provider.py"])
        assert batched == {"jwt_provider.py": reader.blob(sha, "jwt_provider.py")}

    def test_batched_blobs_of_nothing_is_empty(self, reader):
        assert reader.blobs(reader.commits()[0].sha, []) == {}

    def test_root_commit_changes_are_additions(self, reader):
        commits = reader.commits()
        changed = reader.changed_paths(commits[0])
        assert changed == {"auth.py": "A", "jwt_provider.py": "A"}

    def test_later_commit_reports_modifications(self, reader):
        commits = reader.commits()
        assert reader.changed_paths(commits[-1]) == {"jwt_provider.py": "M"}

    def test_a_failing_command_raises(self, reader):
        with pytest.raises(GitError):
            reader.files_at("not-a-sha")


class TestRevisionProvider:
    def test_yields_only_python_files_at_that_revision(self, reader):
        commits = reader.commits()
        provider = GitRevisionProvider(reader, commits[0].sha)
        assert {f.relpath for f in provider.files()} == {"auth.py", "jwt_provider.py"}

    def test_label_names_the_revision(self, reader):
        provider = GitRevisionProvider(reader, reader.commits()[0].sha)
        assert provider.label.startswith("commit ")

    def test_directories_are_derived_from_paths(self):
        assert directories_for(["a.py", "x/b.py", "x/y/c.py"]) == ["x", "x/y"]

    def test_vendored_directories_are_skipped(self):
        assert skipped("src/__pycache__/x.py")
        assert not skipped("src/pkg/x.py")


class TestCommitObjects:
    def test_a_commit_becomes_an_object(self, history):
        store, report = history
        commit = commit_by_subject(report, "rename claim key")
        obj = store.get_object(f"commit://app/{commit.sha}")
        assert obj.type is ObjectType.COMMIT
        assert obj.properties["author"] == "Fixture Author"
        assert obj.properties["subject"] == "rename claim key returned by decode_claims"

    def test_commits_are_ordered_by_precedes(self, history):
        store, report = history
        first, second = report.commits[0].commit, report.commits[1].commit
        edges = store.relations_for(f"commit://app/{first.sha}", direction="out",
                                    types=[RT.PRECEDES])
        assert [e.arguments[1] for e in edges] == [f"commit://app/{second.sha}"]

    def test_parents_outside_the_window_are_not_invented(self, repo):
        """A --max-commits window must not produce edges to absent objects."""
        store = SQLiteStore()
        report = HistoryIngestor(store).ingest(repo, name="app", max_commits=2)
        assert len(report.commits) == 2
        for relation in store.all_relations(include_historical=True):
            if relation.relation_type is RT.PRECEDES:
                assert store.get_object(relation.arguments[0]) is not None
        store.close()


class TestChangeAttribution:
    """Spec section 19 step 9."""

    def test_the_regression_commit_touches_exactly_one_definition(self, history):
        _, report = history
        entry = next(e for e in report.commits if "rename claim key" in e.commit.subject)
        assert entry.changed_symbols == [DECODE_CLAIMS]

    def test_attribution_is_by_definition_text_not_line_span(self, history):
        """encode_claims sits above the edit and shifts no lines; either way a
        line-based attribution would be tempted to include it."""
        _, report = history
        entry = next(e for e in report.commits if "rename claim key" in e.commit.subject)
        assert "encode_claims" not in " ".join(entry.changed_symbols)

    def test_commits_transform_the_files_they_touch(self, history):
        store, report = history
        commit = commit_by_subject(report, "rename claim key")
        edges = store.relations_for(f"commit://app/{commit.sha}", direction="out",
                                    types=[RT.TRANSFORMS])
        targets = {e.arguments[1] for e in edges}
        assert "repo://app/jwt_provider.py#file" in targets
        assert DECODE_CLAIMS in targets

    def test_first_commit_attributes_every_definition_as_added(self, history):
        _, report = history
        entry = report.commits[0]
        assert set(entry.changed_symbols) == {
            "repo://app/auth.py#function:authenticate",
            "repo://app/auth.py#function:create_token",
            "repo://app/auth.py#function:validate_token",
            DECODE_CLAIMS,
            "repo://app/jwt_provider.py#function:encode_claims",
        }

    def test_transform_relations_carry_git_provenance(self, history):
        store, report = history
        commit = commit_by_subject(report, "rename claim key")
        relation = store.relations_for(f"commit://app/{commit.sha}", direction="out",
                                       types=[RT.TRANSFORMS])[0]
        assert store.get_provenance(relation.provenance_id).method is ExtractionMethod.GIT
        assert store.get_evidence(relation.evidence_ids[0]) is not None

    def test_changed_definitions_detects_all_three_kinds(self):
        before = {"a": _def("d1", ast="t1"), "b": _def("d2", ast="t2")}
        after = {"a": _def("d1", ast="t1"), "b": _def("CHANGED", ast="t2-changed"),
                 "c": _def("d3", ast="t3")}
        assert _changed_definitions(before, after) == [
            ("b", "function", "BEHAVIOUR"), ("c", "function", "ADDED")]

    def test_removed_definition_is_reported(self):
        assert _changed_definitions({"gone": _def("d")}, {}) == [
            ("gone", "function", "REMOVE")]

    def test_a_changed_parameter_list_is_a_signature_change(self):
        """Spec section 61 compares a predicted change against an observed one, so
        the observation has to say which kind of change it was. Both revisions are
        parsed here already, so the classification is one comparison."""
        before = {"f": _def("d1", parameters=("user",))}
        after = {"f": _def("d2", parameters=("user", "leeway"))}
        assert _changed_definitions(before, after) == [("f", "function", "SIGNATURE")]

    def test_an_edited_body_with_a_stable_signature_is_a_behaviour_change(self):
        before = {"f": _def("d1", parameters=("user",), ast="t1")}
        after = {"f": _def("d2", parameters=("user",), ast="t2")}
        assert _changed_definitions(before, after) == [("f", "function", "BEHAVIOUR")]

    def test_a_reformat_is_cosmetic_not_a_behaviour_change(self):
        """The text moved and the parse tree did not, so the code means exactly
        what it meant before. Without this, a repository-wide reformat enters the
        spec section 61 evaluation as hundreds of behaviour changes whose predicted
        consequences no commit ever confirms."""
        before = {"f": _def("d1", parameters=("user",), ast="same")}
        after = {"f": _def("d2", parameters=("user",), ast="same")}
        assert _changed_definitions(before, after) == [("f", "function", "COSMETIC")]

    def test_a_signature_change_outranks_an_unchanged_tree(self):
        """Parameters cannot change without the tree changing, but the ordering is
        asserted so a future normaliser that drops parameter names cannot silently
        turn a signature change into a cosmetic one."""
        before = {"f": _def("d1", parameters=("user",), ast="same")}
        after = {"f": _def("d2", parameters=("user", "leeway"), ast="same")}
        assert _changed_definitions(before, after) == [("f", "function", "SIGNATURE")]

    def test_an_unparseable_revision_is_not_called_cosmetic(self):
        """No parse tree means no evidence that nothing changed."""
        before = {"f": _def("d1", parameters=("user",), ast=None)}
        after = {"f": _def("d2", parameters=("user",), ast=None)}
        assert _changed_definitions(before, after) == [("f", "function", "BEHAVIOUR")]

    def test_the_observed_kind_is_recorded_on_the_transforms_relation(self, history):
        store, _ = history
        relations = store.relations_for(DECODE_CLAIMS, direction="in",
                                        types=[RT.TRANSFORMS],
                                        include_historical=True)
        assert relations
        assert all("change_kind" in r.properties for r in relations)


class TestHistoricalValidity:
    """Relation validity carries commit dates, not ingest times (spec section 18)."""

    def test_relations_open_at_their_commit_date(self, history):
        store, report = history
        login_edges = store.relations_for(LOGIN, direction="out", types=[RT.CALLS],
                                          include_historical=True)
        second_commit_date = report.commits[1].commit.date
        assert all(e.valid_from == second_commit_date for e in login_edges)

    def test_a_query_before_a_feature_landed_does_not_see_it(self, history):
        store, report = history
        before_login = report.commits[0].commit.date
        assert store.relations_for(LOGIN, direction="out", types=[RT.CALLS])
        assert not store.relations_for(LOGIN, direction="out", types=[RT.CALLS],
                                       as_of=before_login)

    def test_objects_from_later_commits_are_absent_earlier(self, history):
        store, report = history
        first = report.commits[0].commit.date
        assert not store.relations_for(USER_CLASS, direction="in", as_of=first)

    def test_history_opens_relations_across_several_commits(self, history):
        _, report = history
        opened_per_commit = [e.opened for e in report.commits]
        assert sum(1 for count in opened_per_commit if count > 0) >= 3


class TestRegressionCandidates:
    """Spec section 66, under the section 37 restriction."""

    def test_finds_the_regression_commit(self, history, after_tests):
        store, _ = history
        result = regression_candidates(store, AUTHENTICATE, since=after_tests)
        assert len(result.candidates) == 1
        assert "rename claim key" in result.candidates[0].subject

    def test_the_candidate_names_what_it_changed(self, history, after_tests):
        store, _ = history
        candidate = regression_candidates(store, AUTHENTICATE,
                                          since=after_tests).candidates[0]
        assert candidate.changed.id == DECODE_CLAIMS
        assert candidate.depth == 2

    def test_the_dependency_path_is_reported(self, history, after_tests):
        store, _ = history
        candidate = regression_candidates(store, AUTHENTICATE,
                                          since=after_tests).candidates[0]
        assert candidate.path.objects == (AUTHENTICATE, VALIDATE_TOKEN, DECODE_CLAIMS)

    def test_search_runs_over_dependencies_not_dependents(self, history):
        """A commit touching the tests that call authenticate is not a candidate
        for authenticate failing."""
        store, _ = history
        result = regression_candidates(store, AUTHENTICATE)
        changed = {c.changed.id for c in result.candidates}
        assert DECODE_CLAIMS in changed
        assert TEST_AUTHENTICATE not in changed
        assert LOGIN not in changed

    def test_since_narrows_the_window(self, history, after_tests):
        store, _ = history
        assert len(regression_candidates(store, AUTHENTICATE).candidates) > 1
        assert len(regression_candidates(store, AUTHENTICATE,
                                         since=after_tests).candidates) == 1

    def test_candidates_are_most_recent_first(self, history):
        store, _ = history
        dates = [c.date for c in regression_candidates(store, AUTHENTICATE).candidates]
        assert dates == sorted(dates, reverse=True)

    def test_covering_tests_are_offered_for_bisection(self, history, after_tests):
        store, _ = history
        result = regression_candidates(store, AUTHENTICATE, since=after_tests)
        assert {t.name for t in result.covering_tests} == {
            "test_authenticate_returns_username", "test_login_rejects_inactive_user"}

    def test_an_empty_window_says_so(self, history):
        store, _ = history
        far_future = datetime(2030, 1, 1, tzinfo=timezone.utc)
        result = regression_candidates(store, AUTHENTICATE, since=far_future)
        assert result.candidates == []
        assert "No commit in range" in explain(store, result)

    def test_unknown_target_raises(self, history):
        store, _ = history
        with pytest.raises(KeyError):
            regression_candidates(store, "repo://app/nope.py#function:nope")


class TestSection37Restraint:
    """Dependency is not causality, and this module must not blur them."""

    def test_no_causal_relation_is_created(self, history):
        store, _ = history
        regression_candidates(store, AUTHENTICATE)
        causal = [r for r in store.all_relations(include_historical=True)
                  if r.relation_type in (RT.CAUSES, RT.CONTRIBUTES_TO, RT.TRIGGERS)]
        assert causal == []

    def test_the_explanation_states_that_these_are_correlations(self, history):
        store, _ = history
        text = explain(store, regression_candidates(store, AUTHENTICATE))
        assert "correlations, not causes" in text

    def test_the_score_is_dependency_strength_not_probability(self, history, after_tests):
        store, _ = history
        candidate = regression_candidates(
            store, AUTHENTICATE, since=after_tests).candidates[0]
        assert candidate.correlation == pytest.approx(candidate.path.confidence)


class TestCliSurface:
    def test_history_demo_runs(self, capsys):
        from mcm.cli import main

        assert main(["history-demo"]) == 0
        out = capsys.readouterr().out
        assert "rename claim key" in out
        assert "correlations, not causes" in out
