"""Repository ingestion integration tests (spec sections 19, 71)."""

from mcm.core.objects import ObjectType
from mcm.core.provenance import ExtractionMethod
from mcm.core.relations import RelationType as RT
from mcm.ingestion.dependencies import TESTS_HEURISTIC_CONFIDENCE
from mcm.ingestion.repository import RepositoryIngestor
from mcm.ingestion.sources import WorkingTreeProvider
from mcm.storage.sqlite_store import SQLiteStore

from conftest import (AUTH_FILE, AUTHENTICATE, CREATE_TOKEN, DECODE_CLAIMS, DEMO_REPO,
                      ENCODE_CLAIMS, JWT, LOGIN, TEST_AUTHENTICATE, USER_CLASS,
                      USER_IS_ACTIVE, VALIDATE_TOKEN)


def relation_between(store, source, target, relation_type=None):
    return [
        r for r in store.relations_for(source, direction="out")
        if r.arguments[1] == target and (relation_type is None or r.relation_type is relation_type)
    ]


class TestObjectExtraction:
    def test_creates_typed_objects_for_every_symbol_kind(self, store):
        assert store.get_object(AUTHENTICATE).type is ObjectType.FUNCTION
        assert store.get_object(USER_CLASS).type is ObjectType.CLASS
        assert store.get_object(USER_IS_ACTIVE).type is ObjectType.METHOD
        assert store.get_object(TEST_AUTHENTICATE).type is ObjectType.TEST
        assert store.get_object(AUTH_FILE).type is ObjectType.FILE
        assert store.get_object(JWT).type is ObjectType.MODULE

    def test_line_numbers_are_properties_not_identity(self, store):
        """Spec section 20: identity must not be positional."""
        obj = store.get_object(AUTHENTICATE)
        assert obj.properties["start_line"] > 0
        assert str(obj.properties["start_line"]) not in obj.id

    def test_captures_signature_and_docstring(self, store):
        assert store.get_object(AUTHENTICATE).properties["parameters"] == ["user"]
        assert store.get_object(USER_IS_ACTIVE).properties["parameters"] == ["self"]

    def test_external_library_becomes_a_module_object(self, store):
        assert store.get_object(JWT).properties["external"] is True

    def test_methods_are_contained_by_their_class(self, store):
        assert relation_between(store, USER_CLASS, USER_IS_ACTIVE, RT.CONTAINS)


class TestRelationExtraction:
    def test_extracts_the_call_graph(self, store):
        assert relation_between(store, AUTHENTICATE, VALIDATE_TOKEN, RT.CALLS)
        assert relation_between(store, AUTHENTICATE, CREATE_TOKEN, RT.CALLS)
        assert relation_between(store, LOGIN, AUTHENTICATE, RT.CALLS)
        assert relation_between(store, VALIDATE_TOKEN, DECODE_CLAIMS, RT.CALLS)

    def test_library_member_access_becomes_uses(self, store):
        assert relation_between(store, DECODE_CLAIMS, JWT, RT.USES)
        assert relation_between(store, ENCODE_CLAIMS, JWT, RT.USES)

    def test_imports_are_file_level(self, store):
        assert relation_between(store, AUTH_FILE, "repo://app/jwt_provider.py#file", RT.IMPORTS)

    def test_test_coverage_is_asserted_at_lower_confidence(self, store):
        """Calling is not the same as testing, so the TESTS edge is weaker than
        the CALLS fact it accompanies (spec section 54)."""
        calls = relation_between(store, TEST_AUTHENTICATE, AUTHENTICATE, RT.CALLS)
        tests = relation_between(store, TEST_AUTHENTICATE, AUTHENTICATE, RT.TESTS)
        assert calls and tests
        assert calls[0].confidence == 1.0
        assert tests[0].confidence == TESTS_HEURISTIC_CONFIDENCE
        assert calls[0].id != tests[0].id

    def test_ingestion_asserts_no_derived_relations(self, store):
        """Spec Rule 5: ingestion observes, it does not infer."""
        assert all(not r.is_derived for r in store.all_relations(include_derived=True))


class TestProvenance:
    def test_every_relation_has_provenance_and_evidence(self, store):
        """Spec Rule 4."""
        for relation in store.all_relations():
            assert relation.provenance_id, f"{relation.id} has no provenance"
            assert relation.evidence_ids, f"{relation.id} has no evidence"
            assert store.get_provenance(relation.provenance_id) is not None

    def test_evidence_points_at_a_source_location(self, store):
        relation = relation_between(store, AUTHENTICATE, VALIDATE_TOKEN, RT.CALLS)[0]
        evidence = store.get_evidence(relation.evidence_ids[0])
        assert evidence.source_ref.startswith("auth.py:")
        assert "authenticate calls validate_token" in evidence.content

    def test_ast_facts_and_heuristics_record_different_methods(self, store):
        calls = relation_between(store, TEST_AUTHENTICATE, AUTHENTICATE, RT.CALLS)[0]
        tests = relation_between(store, TEST_AUTHENTICATE, AUTHENTICATE, RT.TESTS)[0]
        assert store.get_provenance(calls.provenance_id).method is ExtractionMethod.AST
        assert (store.get_provenance(tests.provenance_id).method
                is ExtractionMethod.STATIC_ANALYSIS)


class TestUnresolvedNames:
    def test_reports_rather_than_guesses_untyped_attribute_calls(self, report):
        """``user.is_active()`` needs type inference. V1 says so instead of guessing."""
        assert any("user.is_active" in item for item in report.unresolved)
        assert any("claims.get" in item for item in report.unresolved)

    def test_no_relation_invents_the_unresolved_edge(self, store):
        assert not relation_between(store, LOGIN, USER_IS_ACTIVE)

    def test_ingestion_reports_no_parse_errors(self, report):
        assert report.parse_errors == []


class TestIdempotence:
    def test_reingesting_produces_the_same_store(self):
        """Deterministic IDs mean a second ingest updates rather than duplicates."""
        store = SQLiteStore()
        first = RepositoryIngestor(store).ingest(DEMO_REPO, name="app")
        objects_after_first = len(list(store.all_objects()))
        relations_after_first = len(list(store.all_relations()))

        RepositoryIngestor(store).ingest(DEMO_REPO, name="app")
        assert len(list(store.all_objects())) == objects_after_first
        assert len(list(store.all_relations())) == relations_after_first
        assert first.files == 5
        store.close()


def test_working_tree_does_not_follow_symlinked_python_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inside.py").write_text("VALUE = 1\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = 'outside'\n", encoding="utf-8")
    link = repo / "leak.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are not available on this platform")

    paths = {source.relpath for source in WorkingTreeProvider(repo).files()}

    assert paths == {"inside.py"}
