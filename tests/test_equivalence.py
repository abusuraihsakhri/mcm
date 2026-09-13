"""The equivalence engine.

Spec sections 38 and 10. Development step 19.

The properties under test are almost all about the domain hierarchy:

    syntactic  ⟹  AST  ⟹  normal-form  ⟹  semantic

Equality propagates rightwards along it; *inequality never does*. Two functions
with different normal forms may compute exactly the same thing, so a semantic
question with no saturation backend must answer UNKNOWN and never NOT_EQUIVALENT.
Getting that backwards would let a refactoring agent conclude two implementations
differ because they were spelled differently, which is the one genuinely damaging
mistake available here.
"""

import pytest

from mcm.core.relations import RelationType as RT
from mcm.equivalence.engine import (Domain, EquivalenceEngine, Fingerprint,
                                    NoBackend, Verdict, equivalence_classes,
                                    equivalence_relation, explain,
                                    fingerprint_from, fingerprint_of)
from mcm.equivalence.normalise import forms_of
from mcm.ingestion.repository import RepositoryIngestor
from mcm.storage.sqlite_store import SQLiteStore

from conftest import AUTHENTICATE, VALIDATE_TOKEN

BASE = """def area(w, h):
    # multiply them
    return w * h
"""
REFORMATTED = """def area(w, h):
    return w*h
"""
RENAMED_PARAMS = """def area(width, height):
    return width * height
"""
RENAMED_FUNCTION = """def surface(w, h):
    return w * h
"""
DIFFERENT_OPERATOR = """def area(w, h):
    return w + h
"""
SAME_RESULT_DIFFERENT_CODE = """def area(w, h):
    total = 0
    for _ in range(h):
        total += w
    return total
"""
DIFFERENT_CALLEE = """def run(x):
    return other(x)
"""
CALLS_HELPER = """def run(x):
    return helper(x)
"""


@pytest.fixture
def engine():
    return EquivalenceEngine()


@pytest.fixture
def duplicated(tmp_path):
    """A repository containing the same function written twice.

    Built here rather than added to examples/app, whose exact shape the section 65
    acceptance tests assert.
    """
    (tmp_path / "a.py").write_text("def area(w, h):\n    return w * h\n",
                                   encoding="utf-8")
    (tmp_path / "b.py").write_text("def surface(width, height):\n"
                                   "    return width * height\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("def volume(w, h, d):\n    return w * h * d\n",
                                   encoding="utf-8")
    store = SQLiteStore()
    RepositoryIngestor(store).ingest(tmp_path, name="dup")
    yield store
    store.close()


class TestNormalisation:
    def test_comments_and_spacing_do_not_change_the_ast_form(self):
        assert forms_of(BASE).ast == forms_of(REFORMATTED).ast

    def test_renaming_parameters_changes_the_ast_but_not_the_normal_form(self):
        base, renamed = forms_of(BASE), forms_of(RENAMED_PARAMS)
        assert base.ast != renamed.ast
        assert base.normal_form == renamed.normal_form

    def test_renaming_the_function_leaves_the_normal_form(self):
        assert forms_of(BASE).normal_form == forms_of(RENAMED_FUNCTION).normal_form

    def test_a_different_operator_is_a_different_form(self):
        """The case that rules out skipping anonymous nodes: tree-sitter records
        the operator in ``a + b`` as an anonymous child, so dropping those would
        make addition and multiplication identical."""
        base, other = forms_of(BASE), forms_of(DIFFERENT_OPERATOR)
        assert base.ast != other.ast
        assert base.normal_form != other.normal_form

    def test_free_identifiers_are_never_renamed(self):
        """Renaming a callee would erase the reference the rest of the semantic
        model is built on. Two functions calling different things are not
        equivalent under any domain implemented here."""
        assert forms_of(CALLS_HELPER).normal_form != forms_of(DIFFERENT_CALLEE).normal_form

    def test_an_indented_method_normalises_like_a_function(self):
        indented = "    def area(w, h):\n        return w * h\n"
        assert forms_of(indented).ast == forms_of(
            "def area(w, h):\n    return w * h\n").ast

    def test_unparseable_source_has_no_form(self):
        """Not a digest of the broken text: that would put two unparseable
        definitions in the same equivalence class for the wrong reason."""
        assert forms_of("def broken(:") is None
        assert forms_of("   ") is None


class TestDomains:
    def test_syntactic_equivalence_is_identical_text(self, engine):
        assert engine.compare_sources(BASE, BASE,
                                      domain=Domain.SYNTACTIC).equivalent
        assert not engine.compare_sources(BASE, REFORMATTED,
                                          domain=Domain.SYNTACTIC).equivalent

    def test_ast_equivalence_ignores_formatting(self, engine):
        assert engine.compare_sources(BASE, REFORMATTED, domain=Domain.AST).equivalent

    def test_normal_form_equivalence_ignores_bound_names(self, engine):
        assert engine.compare_sources(BASE, RENAMED_PARAMS,
                                      domain=Domain.NORMAL_FORM).equivalent

    def test_a_missing_form_is_unknown_not_different(self, engine):
        """Absence of a digest is not evidence of difference."""
        result = engine.compare(Fingerprint(text="a"), Fingerprint(text="b"),
                                domain=Domain.NORMAL_FORM)
        assert result.verdict is Verdict.UNKNOWN
        assert "did not parse" in result.reason


class TestTheHierarchy:
    """Equality propagates rightwards; inequality must not."""

    def test_matching_normal_forms_settle_the_semantic_question(self, engine):
        """Two definitions differing only in the names they bind compute the same
        thing, so semantics follow from the weaker domain."""
        result = engine.compare_sources(BASE, RENAMED_PARAMS, domain=Domain.SEMANTIC)
        assert result.verdict is Verdict.EQUIVALENT
        assert result.witness is Domain.NORMAL_FORM

    def test_differing_normal_forms_leave_semantics_undecided(self, engine):
        """The load-bearing test. These two really do compute the same thing, and
        the engine must not say otherwise merely because they are spelled
        differently."""
        result = engine.compare_sources(BASE, SAME_RESULT_DIFFERENT_CODE,
                                        domain=Domain.SEMANTIC)
        assert result.verdict is Verdict.UNKNOWN
        assert result.verdict is not Verdict.NOT_EQUIVALENT

    def test_no_semantic_comparison_ever_returns_not_equivalent(self, engine):
        """Without a backend able to prove inequivalence, the verdict is not
        available at all."""
        for left, right in ((BASE, DIFFERENT_OPERATOR), (BASE, DIFFERENT_CALLEE),
                            (BASE, SAME_RESULT_DIFFERENT_CODE), (BASE, BASE)):
            result = engine.compare_sources(left, right, domain=Domain.SEMANTIC)
            assert result.verdict is not Verdict.NOT_EQUIVALENT

    def test_the_weaker_domains_do_answer_negatively(self, engine):
        """The refusal is specific to semantics, not a blanket timidity: "these
        parse trees differ" is a decidable claim and is made."""
        result = engine.compare_sources(BASE, DIFFERENT_OPERATOR, domain=Domain.AST)
        assert result.verdict is Verdict.NOT_EQUIVALENT

    def test_the_refusal_names_what_would_settle_it(self, engine):
        result = engine.compare_sources(BASE, SAME_RESULT_DIFFERENT_CODE,
                                        domain=Domain.SEMANTIC)
        assert "equality saturation" in result.reason


class TestSaturationBackend:
    def test_the_default_backend_refuses_and_says_why(self):
        verdict, reason = NoBackend().compare(BASE, SAME_RESULT_DIFFERENT_CODE)
        assert verdict is Verdict.UNKNOWN
        assert "no saturation backend" in reason

    def test_a_backend_is_consulted_when_the_forms_differ(self):
        """The adapter seam spec section 5 asks for: an egg or egglog integration
        is this protocol over that library, and nothing else changes."""

        class Oracle:
            name = "oracle"

            def compare(self, left, right):
                return Verdict.EQUIVALENT, "proved by saturation"

        result = EquivalenceEngine(backend=Oracle()).compare_sources(
            BASE, SAME_RESULT_DIFFERENT_CODE, domain=Domain.SEMANTIC)
        assert result.verdict is Verdict.EQUIVALENT
        assert "oracle" in result.reason

    def test_a_backend_is_not_consulted_when_the_forms_already_match(self):
        """Cheap answers first: identical normal forms settle it without paying
        for saturation."""

        class Explode:
            name = "explode"

            def compare(self, left, right):
                raise AssertionError("should not be consulted")

        assert EquivalenceEngine(backend=Explode()).compare_sources(
            BASE, RENAMED_PARAMS, domain=Domain.SEMANTIC).equivalent


class TestIngestedDigests:
    def test_ingestion_records_all_three_digests(self, store):
        obj = store.get_object(AUTHENTICATE)
        for prop in ("text_digest", "ast_digest", "normal_form_digest"):
            assert obj.properties.get(prop)

    def test_objects_can_be_compared_without_their_source(self, store):
        """Equivalence answers over any ingested database, including one whose
        repository is no longer on disk."""
        result = EquivalenceEngine().compare_objects(store, AUTHENTICATE,
                                                     VALIDATE_TOKEN)
        assert result.verdict is Verdict.NOT_EQUIVALENT

    def test_comparing_an_unknown_object_fails_loudly(self, store):
        with pytest.raises(KeyError, match="unknown object"):
            EquivalenceEngine().compare_objects(store, AUTHENTICATE, "repo://nope")

    def test_digests_do_not_leak_into_retrieval_text(self, store):
        """They are identity-free bookkeeping. Putting them in the rendered
        document would dilute every embedding with 64 hex characters."""
        from mcm.retrieval.documents import object_document

        text = object_document(store.get_object(AUTHENTICATE)).text
        assert "digest" not in text


class TestEquivalenceClasses:
    def test_duplicates_are_grouped(self, duplicated):
        classes = equivalence_classes(duplicated, domain=Domain.NORMAL_FORM)
        assert len(classes) == 1
        assert {m.name for m in classes[0].members} == {"area", "surface"}

    def test_a_different_function_is_not_grouped_with_them(self, duplicated):
        classes = equivalence_classes(duplicated, domain=Domain.NORMAL_FORM)
        assert "volume" not in {m.name for c in classes for m in c.members}

    def test_the_ast_domain_separates_what_the_normal_form_joins(self, duplicated):
        """area and surface differ in their parameter names, so they share a
        normal form and not a parse tree."""
        assert equivalence_classes(duplicated, domain=Domain.AST) == []

    def test_classes_below_the_minimum_are_dropped(self, duplicated):
        assert equivalence_classes(duplicated, minimum=3) == []

    def test_semantic_classes_are_refused_not_approximated(self, duplicated):
        """Two members of a semantic class may share no form at all, so grouping
        cannot produce them. Answering at NORMAL_FORM instead would be a different
        question wearing the requested name."""
        with pytest.raises(ValueError, match="cannot be computed by grouping"):
            equivalence_classes(duplicated, domain=Domain.SEMANTIC)

    def test_an_empty_result_is_not_reported_as_proof_of_uniqueness(self, store):
        text = explain(equivalence_classes(store))
        assert "rules out duplication, not similarity" in text


class TestRelationConstruction:
    def test_the_relation_carries_its_domain_in_properties(self):
        """Spec section 10 requires an equivalence to name its domain, and
        RELATION_SPECS says it belongs in properties rather than in the type."""
        result = EquivalenceEngine().compare_sources(BASE, RENAMED_PARAMS,
                                                     domain=Domain.NORMAL_FORM)
        relation = equivalence_relation("a://x", "a://y", result)
        assert relation.relation_type is RT.EQUIVALENT_TO
        assert relation.properties["domain"] == "normal_form"

    def test_a_semantic_result_uses_the_semantic_relation_type(self):
        result = EquivalenceEngine().compare_sources(BASE, RENAMED_PARAMS,
                                                     domain=Domain.SEMANTIC)
        relation = equivalence_relation("a://x", "a://y", result)
        assert relation.relation_type is RT.SEMANTICALLY_EQUIVALENT_TO

    def test_arguments_are_ordered_so_the_relation_has_one_identity(self):
        """EQUIVALENT_TO is symmetric, so the same pair must not produce two
        different relation ids depending on argument order."""
        result = EquivalenceEngine().compare_sources(BASE, RENAMED_PARAMS)
        assert (equivalence_relation("a://y", "a://x", result).id
                == equivalence_relation("a://x", "a://y", result).id)

    def test_only_an_equivalent_result_can_be_asserted(self):
        result = EquivalenceEngine().compare_sources(BASE, DIFFERENT_OPERATOR)
        with pytest.raises(ValueError, match="only an EQUIVALENT result"):
            equivalence_relation("a://x", "a://y", result)
