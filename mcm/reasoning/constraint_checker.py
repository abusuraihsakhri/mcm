"""Constraint evaluation (spec sections 13, 14, 27).

Decides the constraint types that are claims about the relation graph, and
reports the rest as UNEVALUATABLE with the names they would need. Spec section 27
names ``VIOLATIONS(change_123)``; ``check_constraints`` takes an optional impact
result so the same machinery answers "which constraints does this change break?"
without inventing a change format.

Every violation carries the objects involved, the relation path that produced it,
and the evidence behind each hop, so a violation is as traceable as an inference
(spec section 43).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from ..algebra.dependency import DependencyPath, dependencies_of
from ..core.constraints import (Constraint, ConstraintType, Selector, Verdict,
                                constraint_id)
from ..core.objects import MCMObject
from ..core.relations import RelationType as RT
from ..storage.database import Store

DEFAULT_CONSTRAINTS = Path(__file__).resolve().parent.parent / "constraints" / "core.yaml"


class ConstraintError(ValueError):
    """A constraint file is malformed or names an unimplemented predicate."""


@dataclass
class Violation:
    message: str
    objects: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class ConstraintResult:
    constraint: Constraint
    verdict: Verdict
    violations: list[Violation] = field(default_factory=list)
    #: Set on UNEVALUATABLE: why the constraint could not be decided.
    reason: str = ""
    checked: int = 0

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.SATISFIED


def check_constraints(store: Store, constraints: list[Constraint], *,
                      as_of: datetime | None = None,
                      restrict_to: set[str] | None = None) -> list[ConstraintResult]:
    """Evaluate every constraint against the store.

    ``restrict_to`` limits the scope to a set of object IDs, which is how spec
    section 27's VIOLATIONS query narrows a check to the objects an impact
    analysis says a change would touch.
    """
    objects = list(store.all_objects())
    if restrict_to is not None:
        objects = [obj for obj in objects if obj.id in restrict_to]
    return [_check(store, constraint, objects, as_of) for constraint in constraints]


def _check(store: Store, constraint: Constraint, objects: list[MCMObject],
           as_of: datetime | None) -> ConstraintResult:
    if not constraint.is_decidable:
        names = constraint.free_names()
        return ConstraintResult(
            constraint=constraint,
            verdict=Verdict.UNEVALUATABLE,
            reason=("needs runtime values: " + ", ".join(names)) if names
                   else "no static interpretation for this expression",
        )

    checker = _PREDICATES.get(constraint.predicate)
    if checker is None:
        raise ConstraintError(
            f"constraint {constraint.name} names unknown predicate "
            f"{constraint.predicate!r}; known: {sorted(_PREDICATES)}"
        )
    return checker(store, constraint, objects, as_of)


# --- predicates -----------------------------------------------------------

def _forbid_dependency(store: Store, constraint: Constraint, objects: list[MCMObject],
                       as_of: datetime | None) -> ConstraintResult:
    """No object matching ``from`` may depend on any object matching ``to``.

    Follows the transitive closure, so an indirect route through a third module
    is caught. This is the architectural-layering check.
    """
    source_selector = Selector.from_dict(constraint.parameters.get("from"))
    target_selector = Selector.from_dict(constraint.parameters.get("to"))
    max_depth = int(constraint.parameters.get("max_depth", 6))

    sources = [obj for obj in objects if source_selector.matches(obj)]
    targets = {obj.id for obj in store.all_objects() if target_selector.matches(obj)}
    result = ConstraintResult(constraint=constraint, verdict=Verdict.SATISFIED,
                              checked=len(sources))
    if not targets:
        return result

    for source in sources:
        closure = dependencies_of(store, source.id, max_depth=max_depth, as_of=as_of)
        for object_id, path in closure.paths.items():
            if object_id in targets:
                result.violations.append(_violation_from_path(
                    f"{source.name} depends on {object_id}", path))
    if result.violations:
        result.verdict = Verdict.VIOLATED
    return result


def _forbid_relation(store: Store, constraint: Constraint, objects: list[MCMObject],
                     as_of: datetime | None) -> ConstraintResult:
    """A direct edge of a given type is forbidden between two selections."""
    relation_type = RT(constraint.parameters["relation"])
    source_selector = Selector.from_dict(constraint.parameters.get("from"))
    target_selector = Selector.from_dict(constraint.parameters.get("to"))

    sources = [obj for obj in objects if source_selector.matches(obj)]
    by_id = {obj.id: obj for obj in store.all_objects()}
    result = ConstraintResult(constraint=constraint, verdict=Verdict.SATISFIED,
                              checked=len(sources))

    for source in sources:
        for relation in store.relations_for(source.id, direction="out",
                                            types=[relation_type], as_of=as_of):
            target = by_id.get(relation.arguments[1])
            if target is not None and target_selector.matches(target):
                result.violations.append(Violation(
                    message=(f"{source.name} -{relation_type.value}-> {target.name}"),
                    objects=[source.id, target.id],
                    relations=[relation.id],
                    evidence_ids=list(relation.evidence_ids),
                ))
    if result.violations:
        result.verdict = Verdict.VIOLATED
    return result


def _require_test(store: Store, constraint: Constraint, objects: list[MCMObject],
                  as_of: datetime | None) -> ConstraintResult:
    """Every object in scope must be the subject of at least one TESTS edge.

    Coverage is checked transitively when ``transitive: true``, since a function
    reached only through a helper is still exercised by the test that calls it.
    """
    transitive = bool(constraint.parameters.get("transitive", False))
    max_depth = int(constraint.parameters.get("max_depth", 4))

    in_scope = [obj for obj in objects if constraint.scope.matches(obj)]
    result = ConstraintResult(constraint=constraint, verdict=Verdict.SATISFIED,
                              checked=len(in_scope))

    for obj in in_scope:
        if _has_direct_test(store, obj.id, as_of):
            continue
        if transitive and _has_transitive_test(store, obj.id, max_depth, as_of):
            continue
        result.violations.append(Violation(
            message=f"{obj.name} has no test coverage",
            objects=[obj.id],
        ))
    if result.violations:
        result.verdict = Verdict.VIOLATED
    return result


def _require_dependency(store: Store, constraint: Constraint, objects: list[MCMObject],
                        as_of: datetime | None) -> ConstraintResult:
    """Every object matching ``from`` must depend on something matching ``to``."""
    source_selector = Selector.from_dict(constraint.parameters.get("from"))
    target_selector = Selector.from_dict(constraint.parameters.get("to"))
    max_depth = int(constraint.parameters.get("max_depth", 6))

    sources = [obj for obj in objects if source_selector.matches(obj)]
    targets = {obj.id for obj in store.all_objects() if target_selector.matches(obj)}
    result = ConstraintResult(constraint=constraint, verdict=Verdict.SATISFIED,
                              checked=len(sources))

    for source in sources:
        closure = dependencies_of(store, source.id, max_depth=max_depth, as_of=as_of)
        if not (set(closure.paths) & targets):
            result.violations.append(Violation(
                message=(f"{source.name} does not depend on any "
                         f"{target_selector.describe()}"),
                objects=[source.id],
            ))
    if result.violations:
        result.verdict = Verdict.VIOLATED
    return result


_PREDICATES = {
    "forbid_dependency": _forbid_dependency,
    "forbid_relation": _forbid_relation,
    "require_test": _require_test,
    "require_dependency": _require_dependency,
}


# --- helpers --------------------------------------------------------------

def _has_direct_test(store: Store, object_id: str, as_of: datetime | None) -> bool:
    return any(store.relations_for(object_id, direction="in", types=[RT.TESTS], as_of=as_of))


def _has_transitive_test(store: Store, object_id: str, max_depth: int,
                         as_of: datetime | None) -> bool:
    """True when any Test object transitively depends on this one.

    Checking for a TESTS edge in the chosen path would give a wrong answer. Both
    CALLS and TESTS exist between a test and its subject, and the closure keeps
    the higher-confidence route, which is always the AST-derived CALLS edge. The
    question is whether a test reaches the object at all, so the object type of
    the dependents is what settles it.
    """
    from ..algebra.dependency import dependents_of
    from ..core.objects import ObjectType

    closure = dependents_of(store, object_id, max_depth=max_depth, as_of=as_of)
    for dependent_id in closure.paths:
        dependent = store.get_object(dependent_id)
        if dependent is not None and dependent.type is ObjectType.TEST:
            return True
    return False


def _violation_from_path(message: str, path: DependencyPath) -> Violation:
    return Violation(
        message=message,
        objects=list(path.objects),
        relations=[r.id for r in path.relations],
        evidence_ids=[eid for r in path.relations for eid in r.evidence_ids],
    )


# --- loading --------------------------------------------------------------

def load_constraints(path: Path | str | None = None) -> list[Constraint]:
    path = Path(path) if path else DEFAULT_CONSTRAINTS
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = document.get("constraints")
    if not entries:
        raise ConstraintError(f"{path} declares no constraints")
    return [_build(entry, path) for entry in entries]


def _build(entry: dict, path: Path) -> Constraint:
    name = entry.get("name")
    if not name:
        raise ConstraintError(f"{path}: a constraint has no name")
    raw_type = entry.get("type")
    try:
        constraint_type = ConstraintType(raw_type)
    except ValueError:
        raise ConstraintError(
            f"{path}: constraint {name} has unknown type {raw_type!r}") from None

    predicate = entry.get("predicate")
    if predicate is not None and predicate not in _PREDICATES:
        raise ConstraintError(
            f"{path}: constraint {name} names unknown predicate {predicate!r}; "
            f"known: {sorted(_PREDICATES)}"
        )

    return Constraint(
        id=constraint_id(name, constraint_type.value),
        name=name,
        type=constraint_type,
        scope=Selector.from_dict(entry.get("scope")),
        predicate=predicate,
        parameters=entry.get("parameters") or {},
        expression=entry.get("expression"),
        description=(entry.get("description") or "").strip(),
    )


def report(results: list[ConstraintResult]) -> str:
    """Render verdicts with their violations (spec section 43)."""
    lines: list[str] = []
    for result in results:
        constraint = result.constraint
        lines.append(f"{constraint.type.value:24} {constraint.name:28} "
                     f"{result.verdict.value}")
        if result.verdict is Verdict.UNEVALUATABLE:
            lines.append(f"    {result.reason}")
            lines.append(f"    scope: {constraint.scope.describe()}")
            if constraint.expression:
                lines.append(f"    expression: {constraint.expression}")
        for violation in result.violations:
            lines.append(f"    {violation.message}")
            for evidence_id in violation.evidence_ids[:2]:
                lines.append(f"      evidence: {evidence_id}")
    violated = sum(1 for r in results if r.verdict is Verdict.VIOLATED)
    undecided = sum(1 for r in results if r.verdict is Verdict.UNEVALUATABLE)
    lines.append("")
    lines.append(f"{len(results)} constraints: {len(results) - violated - undecided} satisfied, "
                 f"{violated} violated, {undecided} unevaluatable")
    return "\n".join(lines)
