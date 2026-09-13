"""Constraint engine tests (spec sections 13, 14, 27)."""

import pytest

from mcm.core.constraints import (Constraint, ConstraintType, Selector, Verdict,
                                  constraint_id)
from mcm.core.objects import MCMObject, ObjectType
from mcm.core.relations import MCMRelation, RelationType as RT
from mcm.reasoning.constraint_checker import (ConstraintError, check_constraints,
                                              load_constraints, report)

from conftest import AUTHENTICATE, USER_IS_ACTIVE


@pytest.fixture(scope="module")
def constraints():
    return load_constraints()


@pytest.fixture(scope="module")
def results(store, constraints):
    return check_constraints(store, constraints)


def by_name(results, name):
    return next(r for r in results if r.constraint.name == name)


class TestConstraintModel:
    def test_requires_exactly_one_of_predicate_or_expression(self):
        with pytest.raises(ValueError, match="exactly one"):
            Constraint(id="c", name="x", type=ConstraintType.INVARIANT)
        with pytest.raises(ValueError, match="exactly one"):
            Constraint(id="c", name="x", type=ConstraintType.INVARIANT,
                       predicate="require_test", expression="a > b")

    def test_free_names_parses_attribute_paths(self):
        constraint = Constraint(id="c", name="x", type=ConstraintType.POSTCONDITION,
                                expression="token.expiry > now")
        assert constraint.free_names() == ["now", "token.expiry"]

    def test_free_names_prefers_the_longest_path(self):
        constraint = Constraint(id="c", name="x", type=ConstraintType.INVARIANT,
                                expression="user.profile.email != user.backup")
        assert constraint.free_names() == ["user.backup", "user.profile.email"]

    def test_constraint_ids_are_deterministic(self):
        assert constraint_id("x", "INVARIANT") == constraint_id("x", "INVARIANT")
        assert constraint_id("x", "INVARIANT") != constraint_id("x", "PRECONDITION")


class TestSelector:
    def test_type_and_glob_are_conjunctive(self):
        obj = MCMObject(id="repo://app/auth.py#function:login", type=ObjectType.FUNCTION,
                        name="login")
        assert Selector(type=ObjectType.FUNCTION, id_glob="repo://app/auth.py#*").matches(obj)
        assert not Selector(type=ObjectType.CLASS, id_glob="repo://app/auth.py#*").matches(obj)
        assert not Selector(type=ObjectType.FUNCTION, id_glob="repo://app/user.py#*").matches(obj)

    def test_empty_selector_matches_anything(self):
        obj = MCMObject(id="x", type=ObjectType.FILE, name="x.py")
        assert Selector().matches(obj)

    def test_bare_string_is_an_id_glob(self):
        assert Selector.from_dict("lib://*").id_glob == "lib://*"


class TestDecidableConstraints:
    def test_architectural_layering_holds(self, results):
        assert by_name(results, "production_must_not_depend_on_tests").ok

    def test_forbidden_direct_import_holds(self, results):
        assert by_name(results, "auth_must_not_import_jwt_directly").ok

    def test_transitive_test_coverage_is_satisfied(self, results):
        """create_token has no direct test, but a test reaches it via authenticate."""
        assert by_name(results, "entry_points_are_tested").ok

    def test_required_dependency_holds(self, results):
        assert by_name(results, "token_creation_goes_through_provider").ok

    def test_method_coverage_violation_traces_to_the_type_inference_gap(self, results):
        """user.is_active() is unresolvable without type inference, so no test
        edge reaches User.is_active. The constraint reports a real gap."""
        result = by_name(results, "methods_are_tested")
        assert result.verdict is Verdict.VIOLATED
        assert USER_IS_ACTIVE in {oid for v in result.violations for oid in v.objects}

    def test_violations_are_counted_against_what_was_checked(self, results):
        result = by_name(results, "entry_points_are_tested")
        assert result.checked == 4


class TestUnevaluatable:
    """Spec section 44: do not present an unchecked constraint as satisfied."""

    def test_value_level_postcondition_is_unevaluatable(self, results):
        result = by_name(results, "token_expiry_is_in_the_future")
        assert result.verdict is Verdict.UNEVALUATABLE
        assert result.verdict is not Verdict.SATISFIED

    def test_reason_names_what_is_missing(self, results):
        result = by_name(results, "token_expiry_is_in_the_future")
        assert "token.expiry" in result.reason
        assert "now" in result.reason

    def test_unevaluatable_produces_no_violations(self, results):
        assert by_name(results, "authenticated_user_is_valid").violations == []


class TestConstructedViolations:
    """The fixture is well-formed, so layering violations are built explicitly."""

    def test_forbid_dependency_catches_an_indirect_route(self, empty_store):
        for object_id, name in [("ui", "view"), ("mid", "service"), ("db", "conn")]:
            empty_store.put_object(MCMObject(id=object_id, type=ObjectType.FILE, name=name))
        empty_store.put_relation(MCMRelation(id="r1", relation_type=RT.IMPORTS,
                                             arguments=["ui", "mid"]))
        empty_store.put_relation(MCMRelation(id="r2", relation_type=RT.IMPORTS,
                                             arguments=["mid", "db"]))

        constraint = Constraint(
            id="c", name="no_ui_to_db", type=ConstraintType.ARCHITECTURAL_CONSTRAINT,
            predicate="forbid_dependency",
            parameters={"from": {"id_glob": "ui"}, "to": {"id_glob": "db"}},
        )
        result = check_constraints(empty_store, [constraint])[0]
        assert result.verdict is Verdict.VIOLATED
        assert result.violations[0].objects == ["ui", "mid", "db"]

    def test_violation_carries_the_evidence_for_every_hop(self, empty_store):
        for object_id in ("a", "b"):
            empty_store.put_object(MCMObject(id=object_id, type=ObjectType.FILE,
                                             name=object_id))
        empty_store.put_relation(MCMRelation(id="r", relation_type=RT.IMPORTS,
                                             arguments=["a", "b"], evidence_ids=["ev1"]))
        constraint = Constraint(
            id="c", name="no_a_to_b", type=ConstraintType.ARCHITECTURAL_CONSTRAINT,
            predicate="forbid_dependency",
            parameters={"from": {"id_glob": "a"}, "to": {"id_glob": "b"}},
        )
        result = check_constraints(empty_store, [constraint])[0]
        assert result.violations[0].evidence_ids == ["ev1"]


class TestScopeRestriction:
    def test_restrict_to_narrows_the_check(self, store, constraints):
        """Spec section 27's VIOLATIONS query: check only what a change touches."""
        full = check_constraints(store, constraints)
        narrowed = check_constraints(store, constraints, restrict_to={AUTHENTICATE})
        assert by_name(narrowed, "entry_points_are_tested").checked == 1
        assert by_name(full, "entry_points_are_tested").checked == 4
        assert by_name(narrowed, "methods_are_tested").checked == 0


class TestLoading:
    def test_unknown_predicate_is_rejected_at_load(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("""
constraints:
  - name: bad
    type: INVARIANT
    predicate: does_not_exist
""", encoding="utf-8")
        with pytest.raises(ConstraintError, match="unknown predicate"):
            load_constraints(path)

    def test_unknown_type_is_rejected_at_load(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("""
constraints:
  - name: bad
    type: NOT_A_TYPE
    predicate: require_test
""", encoding="utf-8")
        with pytest.raises(ConstraintError, match="unknown type"):
            load_constraints(path)


class TestReport:
    def test_report_distinguishes_the_three_verdicts(self, results):
        text = report(results)
        assert "SATISFIED" in text
        assert "VIOLATED" in text
        assert "UNEVALUATABLE" in text
        assert "satisfied," in text and "violated," in text and "unevaluatable" in text
