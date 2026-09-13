"""Vector projection, lexical projection and hybrid retrieval.

Spec sections 22 to 26, development steps 14 and 15. The properties under test are
the ones the architecture depends on, not the ranking itself:

* the projections are *derived* - rebuildable, prunable, and worth nothing if
  deleted (spec section 21 applies to all three projections, not only the graph)
* the vector channel never gets to decide an exact match (spec section 22)
* every weight in the section 26 scoring model actually moves the ranking, which
  is what makes the section 50 ablations possible
* a score can be traced back to the document that produced it (spec section 43)
"""

import math

import pytest

from mcm.core.evidence import Evidence, EvidenceType
from mcm.core.objects import MCMObject, ObjectType, utcnow
from mcm.core.relations import Inference, MCMRelation, RelationType as RT
from mcm.ingestion.repository import RepositoryIngestor
from mcm.retrieval import documents as doc
from mcm.retrieval.embedding import (HashedTokenProvider, cosine, get_provider,
                                     tokenize)
from mcm.retrieval.hybrid import (GRAPH, LEXICAL, SYMBOLIC, VECTOR, HybridRetriever,
                                  RetrievalWeights, build_indexes, explain,
                                  result_json)
from mcm.retrieval.lexical import LexicalProjection
from mcm.retrieval import vector as vector_module
from mcm.retrieval.vector import SQLiteVectorIndex, VectorProjection
from mcm.storage.sqlite_store import SQLiteStore

from conftest import (AUTHENTICATE, DECODE_CLAIMS, JWT, USER_IS_ACTIVE,
                      VALIDATE_TOKEN, DEMO_REPO)


@pytest.fixture(scope="module")
def indexed():
    """A store with examples/app ingested and both text indexes built.

    Module-scoped and separate from the shared ``store`` fixture: these tests add
    derived tables and rebuild them, and nothing else should have to care.
    """
    store = SQLiteStore()
    RepositoryIngestor(store).ingest(DEMO_REPO, name="app")
    build_indexes(store)
    yield store
    store.close()


@pytest.fixture
def fresh():
    store = SQLiteStore()
    RepositoryIngestor(store).ingest(DEMO_REPO, name="app")
    yield store
    store.close()


class TestTokenizer:
    def test_snake_case_yields_parts_and_the_whole(self):
        assert tokenize("validate_token") == ["validate_token", "validate", "token"]

    def test_camel_case_yields_the_same_parts(self):
        camel = set(tokenize("validateToken"))
        assert {"validate", "token"} <= camel

    def test_dotted_paths_split_on_the_dot(self):
        assert tokenize("auth.validate_token")[0] == "auth"


class TestHashedProvider:
    def test_vectors_are_unit_length(self):
        vector = HashedTokenProvider().embed_one("authenticate a user")
        assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, abs_tol=1e-9)

    def test_identical_text_embeds_identically_across_instances(self):
        """Stability across processes, not just calls: the index is on disk, and
        Python's built-in hash() is randomised per process."""
        a = HashedTokenProvider().embed_one("validate_token")
        b = HashedTokenProvider().embed_one("validate_token")
        assert a == b

    def test_related_text_scores_above_unrelated_text(self):
        provider = HashedTokenProvider()
        anchor = provider.embed_one("validate_token authentication jwt")
        near = provider.embed_one("validate_tokens authentication jwt")
        far = provider.embed_one("render html template markup")
        assert cosine(anchor, near) > cosine(anchor, far)

    def test_morphological_variants_are_close(self):
        """The character n-grams earn their place here. A whole-term index cannot
        match a plural to its singular, which is the one thing this default
        provider does that the lexical channel cannot."""
        provider = HashedTokenProvider()
        assert cosine(provider.embed_one("validate_token"),
                      provider.embed_one("validate_tokens")) > 0.6

    def test_empty_text_gives_a_zero_vector_and_no_similarity(self):
        provider = HashedTokenProvider()
        assert cosine(provider.embed_one(""), provider.embed_one("anything")) == 0.0

    def test_dimension_is_configurable_through_the_spec_string(self):
        assert get_provider("hashed:64").dimension == 64

    def test_configuration_is_part_of_provider_identity(self):
        """Two configurations produce incomparable geometries, so they must not
        share a name - the name is what the index stores to keep them apart."""
        assert get_provider("hashed:64").name != get_provider("hashed:128").name

    def test_unknown_provider_is_refused_by_name(self):
        with pytest.raises(ValueError, match="unknown embedding provider"):
            get_provider("nonesuch")

    def test_cosine_refuses_mismatched_dimensions(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine([1.0, 0.0], [1.0, 0.0, 0.0])


class TestDocuments:
    def test_object_document_carries_name_path_and_signature(self, indexed):
        obj = indexed.get_object(AUTHENTICATE)
        text = doc.object_document(obj).text
        assert "authenticate" in text
        assert "auth.py" in text
        assert text.startswith("Function authenticate")

    def test_relation_document_reads_as_the_assertion(self, indexed):
        relation = next(r for r in indexed.all_relations()
                        if r.relation_type is RT.CALLS
                        and r.arguments[0] == AUTHENTICATE)
        text = doc.relation_document(relation, indexed).text
        assert text.startswith("CALLS")
        assert "authenticate" in text

    def test_digest_tracks_the_text(self, indexed):
        obj = indexed.get_object(AUTHENTICATE)
        first = doc.object_document(obj).digest
        assert doc.object_document(obj).digest == first
        obj.properties["docstring"] = "changed"
        assert doc.object_document(obj).digest != first

    def test_derived_relations_are_not_indexed(self, fresh):
        """Spec Rule 2. Indexing an inference beside its premises would let a
        later search return the conclusion as though it were a source."""
        fresh.put_relation(MCMRelation(
            id="rel:derived-for-test", relation_type=RT.POSSIBLY_AFFECTS,
            arguments=[JWT, AUTHENTICATE], confidence=0.5,
            inference=Inference(rule="test", path=[JWT, AUTHENTICATE]),
        ))
        keys = {d.key for d in doc.documents(fresh)}
        assert doc.document_key(doc.RELATION, "rel:derived-for-test") not in keys

    def test_unknown_kind_is_refused(self, indexed):
        with pytest.raises(ValueError, match="unknown document kinds"):
            list(doc.documents(indexed, kinds=("object", "nonsense")))


class TestVectorProjection:
    def test_rebuild_covers_objects_relations_and_evidence(self, fresh):
        report = VectorProjection(fresh).rebuild()
        assert set(report.by_kind) == {doc.OBJECT, doc.RELATION, doc.EVIDENCE}
        assert report.embedded == sum(report.by_kind.values())

    def test_rebuilding_an_unchanged_core_embeds_nothing(self, fresh):
        projection = VectorProjection(fresh)
        first = projection.rebuild()
        second = projection.rebuild()
        assert second.embedded == 0
        assert second.unchanged == first.embedded

    def test_an_edited_object_is_re_embedded_alone(self, fresh):
        projection = VectorProjection(fresh)
        projection.rebuild()
        obj = fresh.get_object(AUTHENTICATE)
        obj.properties["docstring"] = "now with a docstring"
        fresh.put_object(obj)
        report = projection.rebuild()
        assert report.embedded == 1

    def test_a_closed_relation_drops_out_of_the_index(self, fresh):
        """The temporal model closes relations rather than deleting them (spec
        section 18), so a fact can stop being true without anything being removed.
        Digest-based pruning catches that; a 'modified since' rebuild would not."""
        projection = VectorProjection(fresh)
        projection.rebuild()
        relation = next(r for r in fresh.all_relations()
                        if r.relation_type is RT.CALLS)
        fresh.close_relation(relation.id, utcnow())
        report = projection.rebuild()
        assert report.removed == 1

    def test_exact_name_is_the_nearest_neighbour(self, indexed):
        matches = VectorProjection(indexed).search("validate_token",
                                                   kinds=[doc.OBJECT], limit=1)
        assert matches[0].source_id == VALIDATE_TOKEN

    def test_a_near_miss_still_retrieves(self, indexed):
        matches = VectorProjection(indexed).search("validate_tokens",
                                                   kinds=[doc.OBJECT], limit=1)
        assert matches[0].source_id == VALIDATE_TOKEN

    def test_results_are_ordered_by_similarity(self, indexed):
        matches = VectorProjection(indexed).search("jwt token", limit=8)
        assert matches == sorted(matches, key=lambda m: -m.similarity)

    def test_vectors_of_different_models_never_mix(self, fresh):
        """A provider swap invalidates the index rather than silently comparing
        two geometries that share no basis."""
        index = SQLiteVectorIndex.for_store(fresh)
        small = VectorProjection(fresh, index=index, provider=get_provider("hashed:64"))
        large = VectorProjection(fresh, index=index, provider=get_provider("hashed:128"))
        small.rebuild()
        assert large.search("validate_token") == []
        assert large.rebuild().embedded > 0
        assert all(len(e.vector) == 64 for e in index.entries(small.model))

    def test_similarity_of_an_unindexed_key_is_zero_not_an_error(self, indexed):
        projection = VectorProjection(indexed)
        query = projection.provider.embed_one("anything")
        assert projection.similarity(query, "object::nope") == 0.0

    def test_the_dense_path_ranks_exactly_as_the_scan_it_replaced(self, indexed):
        """NumPy is an accelerator here, not a second ranking function.

        ``search`` does one matrix-vector product when NumPy is importable and a
        per-entry ``cosine`` scan when it is not. Two implementations of one
        ranking is the shape of a silent divergence, so the scan is written out
        here and both are held to the same answer.

        Similarities are compared with a tolerance rather than for equality:
        summing 256 terms in a different order is a different float, and pinning
        the bits would be pinning BLAS's blocking strategy.
        """
        projection = VectorProjection(indexed)
        if vector_module._np is None:                       # nothing to compare
            pytest.skip("NumPy is not installed; only the scan exists")

        for query in ("jwt token", "validate_token", "decode claims"):
            vector = projection.provider.embed_one(query)
            scan = [(entry.key, cosine(vector, entry.vector))
                    for entry in projection.index.entries(projection.model)]
            scan = sorted((pair for pair in scan if pair[1] > 0.0),
                          key=lambda pair: (-pair[1], pair[0]))[:8]
            dense = projection.search(query, limit=8)

            assert [m.key for m in dense] == [key for key, _ in scan]
            for match, (_, similarity) in zip(dense, scan):
                assert match.similarity == pytest.approx(similarity, abs=1e-12)

    def test_rebuilding_drops_the_stacked_form(self, indexed):
        """The cache is keyed on nothing, so only invalidation keeps it honest."""
        if vector_module._np is None:                       # no cache to drop
            pytest.skip("NumPy is not installed; the scan holds no state")
        projection = VectorProjection(indexed)
        projection.search("jwt token", limit=1)
        assert projection._dense                            # populated by search
        projection.rebuild()
        assert not projection._dense


class TestLexicalProjection:
    def test_exact_symbol_lookup_ranks_the_definition_first(self, indexed):
        matches = LexicalProjection(indexed).search("validate_token",
                                                    kinds=[doc.OBJECT], limit=5)
        assert matches[0].source_id == VALIDATE_TOKEN
        assert matches[0].relevance == 1.0

    def test_underscored_and_split_forms_both_match(self, indexed):
        projection = LexicalProjection(indexed)
        found = {m.source_id for m in projection.search("is_active", kinds=[doc.OBJECT])}
        assert USER_IS_ACTIVE in found
        found = {m.source_id for m in projection.search("active", kinds=[doc.OBJECT])}
        assert USER_IS_ACTIVE in found

    def test_query_syntax_is_treated_as_text(self, indexed):
        """A user searching for NOT, a trailing star or a quote gets those
        characters, not an FTS5 operator and not a syntax error."""
        projection = LexicalProjection(indexed)
        for query in ('NOT user*', '"unbalanced', 'user AND (jwt', 'a OR b NEAR c'):
            assert isinstance(projection.search(query, limit=2), list)

    def test_an_unmatched_query_returns_nothing_rather_than_everything(self, indexed):
        assert LexicalProjection(indexed).search("kubernetes helm chart") == []

    def test_rebuilding_an_unchanged_core_indexes_nothing(self, fresh):
        projection = LexicalProjection(fresh)
        projection.rebuild()
        assert projection.rebuild().indexed == 0

    def test_evidence_content_is_searchable(self, fresh):
        """Spec section 23 names error messages as a lexical target, and an error
        message reaches MCM as evidence content, not as an object name."""
        failure = Evidence.create(
            EvidenceType.TEST, "tests/test_auth.py::test_login",
            "AssertionError: inactive user was granted a session", "pytest")
        fresh.put_evidence(failure)
        LexicalProjection(fresh).rebuild()
        matches = LexicalProjection(fresh).search("inactive granted session",
                                                  kinds=[doc.EVIDENCE])
        assert matches[0].source_id == failure.id


class TestHybridRetrieval:
    def test_an_exact_name_is_ranked_first_and_scores_symbolically(self, indexed):
        result = HybridRetriever(indexed).retrieve("validate_token")
        top = result.candidates[0]
        assert top.object_id == VALIDATE_TOKEN
        assert top.symbolic == 1.0

    def test_the_top_result_is_corroborated_by_several_channels(self, indexed):
        """Spec section 25 exists because a result several mechanisms agree on is
        a different claim from one only similarity liked."""
        top = HybridRetriever(indexed).retrieve("validate_token").candidates[0]
        assert {VECTOR, LEXICAL, SYMBOLIC} <= set(top.channels)

    def test_the_graph_channel_reaches_what_the_query_never_names(self, indexed):
        """decode_claims is not mentioned by the query and shares no vocabulary
        with it. It is in the pool because validate_token calls it."""
        result = HybridRetriever(indexed).retrieve("validate_token", limit=25)
        found = {c.object_id: c for c in result.candidates}
        assert DECODE_CLAIMS in found
        assert found[DECODE_CLAIMS].graph > 0.0

    def test_an_anchor_is_not_credited_on_the_graph_channel(self, indexed):
        """An object found by name has no graph relevance of its own; crediting it
        for the round trip through the file that contains it would count one piece
        of evidence under two weights."""
        top = HybridRetriever(indexed).retrieve("validate_token").candidates[0]
        assert top.graph == 0.0

    def test_retrieval_works_with_no_text_indexes_at_all(self, fresh):
        """Symbolic and graph retrieval read the core directly. An unindexed
        repository degrades to exact lookup plus neighbourhood, and says so,
        rather than returning an empty result that reads like an answer."""
        result = HybridRetriever(fresh).retrieve("validate_token")
        assert result.indexed is False
        assert result.candidates[0].object_id == VALIDATE_TOKEN
        assert result.candidates[0].lexical == 0.0

    def test_similarity_cannot_outrank_an_exact_match(self, indexed):
        """Spec section 22: the vector store must not be responsible for exact
        matching. Even with every other channel switched off, the symbolic channel
        still decides."""
        weights = RetrievalWeights.parse("v=0.0,l=0.0,g=0.0,s=1.0,p=0.0")
        result = HybridRetriever(indexed, weights=weights).retrieve("validate_token")
        assert result.candidates[0].object_id == VALIDATE_TOKEN
        assert result.candidates[0].score == 1.0

    def test_every_weight_moves_the_score(self, indexed):
        """What the section 50 ablation studies need: no term is decorative.

        The query names a real symbol, so all five channels have something to
        say about it. A query that matches nothing exactly would leave the
        symbolic weight with nothing to scale, which is correct behaviour and
        would make this a test of the query rather than of the model.
        """
        retriever = HybridRetriever(indexed)
        baseline = {c.object_id: c.score for c in retriever.retrieve(
            "validate_token", limit=25).candidates}
        for term in ("v", "l", "g", "s", "p"):
            weights = RetrievalWeights.parse(f"{term}=0.9")
            moved = {c.object_id: c.score for c in HybridRetriever(
                indexed, weights=weights).retrieve("validate_token",
                                                   limit=25).candidates}
            assert moved != baseline, f"weight {term} changed nothing"

    def test_scores_stay_in_range_under_default_weights(self, indexed):
        result = HybridRetriever(indexed).retrieve("authenticate", limit=25)
        assert math.isclose(result.weights.total, 1.0)
        assert all(0.0 <= c.score <= 1.0 for c in result.candidates)

    def test_every_candidate_can_name_what_produced_its_score(self, indexed):
        """Spec section 43: an answer that cannot be traced is not an answer."""
        result = HybridRetriever(indexed).retrieve("jwt", limit=10)
        for candidate in result.candidates:
            assert candidate.routes
            assert all(route.channel in (VECTOR, LEXICAL, GRAPH, SYMBOLIC)
                       for route in candidate.routes)

    def test_relation_and_evidence_hits_are_routes_not_results(self, indexed):
        """Everything returned is an object. A relation is how you got there."""
        result = HybridRetriever(indexed).retrieve("CALLS authenticate", limit=25)
        assert all(indexed.get_object(c.object_id) is not None
                   for c in result.candidates)
        kinds = {route.kind for c in result.candidates for route in c.routes}
        assert doc.RELATION in kinds

    def test_provenance_scores_the_extractor_not_the_claim(self, indexed):
        """Spec section 17 keeps source reliability and evidence strength apart,
        so P reports reliability alone and never their product."""
        result = HybridRetriever(indexed).retrieve("authenticate", limit=25)
        scored = [c for c in result.candidates if c.provenance > 0.0]
        assert scored and all(c.provenance <= 1.0 for c in scored)

    def test_an_object_with_no_provenance_scores_zero(self, fresh):
        fresh.put_object(MCMObject(id="orphan://thing", type=ObjectType.CONCEPT,
                                   name="orphan"))
        result = HybridRetriever(fresh).retrieve("orphan")
        assert result.candidates[0].object_id == "orphan://thing"
        assert result.candidates[0].provenance == 0.0


class TestWeightParsing:
    def test_letters_alias_the_five_scoring_terms(self):
        weights = RetrievalWeights.parse("v=0.4,l=0.3,g=0.2,s=0.1,p=0.0")
        assert (weights.vector, weights.lexical, weights.graph,
                weights.symbolic, weights.provenance) == (0.4, 0.3, 0.2, 0.1, 0.0)

    def test_unnamed_terms_keep_their_defaults(self):
        assert RetrievalWeights.parse("g=0.0").vector == RetrievalWeights().vector

    def test_traversal_settings_are_weights_too(self):
        weights = RetrievalWeights.parse("graph_depth=4,graph_decay=0.25")
        assert (weights.graph_depth, weights.graph_decay) == (4, 0.25)

    def test_an_unknown_weight_is_refused(self):
        with pytest.raises(ValueError, match="unknown weight"):
            RetrievalWeights.parse("popularity=0.5")


class TestReporting:
    def test_the_report_shows_the_channel_breakdown(self, indexed):
        text = explain(HybridRetriever(indexed).retrieve("validate_token"), limit=3)
        assert "validate_token" in text
        for channel in ("v=", "l=", "g=", "s=", "p="):
            assert channel in text

    def test_an_empty_index_is_announced_not_hidden(self, fresh):
        text = explain(HybridRetriever(fresh).retrieve("validate_token"))
        assert "mcm index" in text

    def test_json_carries_scores_weights_and_routes(self, indexed):
        payload = result_json(HybridRetriever(indexed).retrieve("jwt", limit=3))
        assert payload["mode"] == "retrieve"
        assert payload["weights"]["vector"] == RetrievalWeights().vector
        first = payload["results"][0]
        assert set(first["channels"]) == {"vector", "lexical", "graph",
                                          "symbolic", "provenance"}
        assert first["routes"]
