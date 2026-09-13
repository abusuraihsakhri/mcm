"""Declarative inference rules (spec section 36).

Rules live in YAML so the algebra is extensible without touching Python. Spec
section 36: "This makes the algebra extensible."

A rule is a conjunction of premises and a single conclusion. Names beginning with
``?`` are variables, unified across premises. A premise naming an abstract
relation type matches every type subsumed by it, so a rule written against
``DEPENDS_ON`` matches the ``CALLS`` and ``USES`` edges ingestion actually wrote.

Rules are validated at load time rather than at match time. A rule with an
unbound conclusion variable, an undeclared relation type, or a conclusion that
would produce a fact is rejected when the file is read, not when it first fires.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..algebra.specs import spec_for, subsumed_by
from ..core.relations import RelationType as RT

DEFAULT_RULES = Path(__file__).resolve().parent.parent / "rules" / "core.yaml"


class RuleError(ValueError):
    """A rule file is malformed or declares something the algebra forbids."""


@dataclass(frozen=True)
class Premise:
    """One conjunct. ``types`` is the expanded set this premise matches."""

    declared: tuple[RT, ...]
    arguments: tuple[str, ...]
    types: frozenset[RT]

    def variables(self) -> set[str]:
        return {a for a in self.arguments if is_variable(a)}


@dataclass(frozen=True)
class Rule:
    name: str
    premises: tuple[Premise, ...]
    conclusion_type: RT
    conclusion_arguments: tuple[str, ...]
    #: Variable pairs that must not unify to the same object. Without this,
    #: transitive_dependency derives that everything in a cycle depends on itself.
    distinct: tuple[tuple[str, str], ...] = ()
    description: str = ""

    def variables(self) -> set[str]:
        return set().union(*(p.variables() for p in self.premises))


def is_variable(token: str) -> bool:
    return isinstance(token, str) and token.startswith("?")


def load_rules(path: Path | str | None = None) -> list[Rule]:
    """Load and validate a rule file."""
    path = Path(path) if path else DEFAULT_RULES
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = document.get("rules")
    if not entries:
        raise RuleError(f"{path} declares no rules")
    return [_build(entry, path) for entry in entries]


def _build(entry: dict, path: Path) -> Rule:
    name = entry.get("name")
    if not name:
        raise RuleError(f"{path}: a rule has no name")

    raw_premises = entry.get("premises") or []
    if not raw_premises:
        raise RuleError(f"{path}: rule {name} has no premises")
    premises = tuple(_premise(p, name, path) for p in raw_premises)

    raw_conclusion = entry.get("conclusion")
    if not raw_conclusion or len(raw_conclusion) < 2:
        raise RuleError(f"{path}: rule {name} has no usable conclusion")
    conclusion_type = _relation_type(raw_conclusion[0], name, path)
    conclusion_arguments = tuple(raw_conclusion[1:])

    bound = set().union(*(p.variables() for p in premises))
    unbound = {a for a in conclusion_arguments if is_variable(a)} - bound
    if unbound:
        raise RuleError(
            f"{path}: rule {name} concludes with unbound variables {sorted(unbound)}"
        )

    _reject_fact_producing(conclusion_type, premises, name, path)

    distinct = tuple(tuple(pair) for pair in entry.get("distinct") or [])
    for pair in distinct:
        if len(pair) != 2:
            raise RuleError(f"{path}: rule {name} has a malformed distinct clause")

    return Rule(
        name=name,
        premises=premises,
        conclusion_type=conclusion_type,
        conclusion_arguments=conclusion_arguments,
        distinct=distinct,
        description=(entry.get("description") or "").strip(),
    )


def _premise(raw, rule_name: str, path: Path) -> Premise:
    if not isinstance(raw, list) or len(raw) < 2:
        raise RuleError(f"{path}: rule {rule_name} has a malformed premise {raw!r}")
    head, *arguments = raw
    declared = tuple(_relation_type(t, rule_name, path)
                     for t in (head if isinstance(head, list) else [head]))
    types: set[RT] = set()
    for relation_type in declared:
        types |= subsumed_by(relation_type)
    return Premise(declared=declared, arguments=tuple(arguments), types=frozenset(types))


def _relation_type(token: str, rule_name: str, path: Path) -> RT:
    try:
        relation_type = RT(token)
    except ValueError:
        raise RuleError(f"{path}: rule {rule_name} names unknown relation type {token!r}") from None
    spec_for(relation_type)  # raises if the algebra has no spec for it
    return relation_type


def _reject_fact_producing(conclusion_type: RT, premises: tuple[Premise, ...],
                           rule_name: str, path: Path) -> None:
    """Spec Rule 2, enforced at load time.

    A rule may not conclude a relation type that ingestion also asserts, unless
    that type is declared derived-only. Otherwise a derivation and an observation
    would be indistinguishable by type, and the fact/inference split would depend
    on every caller remembering to check ``is_derived``.

    Inverse relations are the exception this permits: TESTED_BY and PART_OF are
    never asserted by ingestion, so deriving them cannot shadow an observation.
    """
    spec = spec_for(conclusion_type)
    if spec.derived_only:
        return
    if any(conclusion_type in premise.types for premise in premises):
        raise RuleError(
            f"{path}: rule {rule_name} concludes {conclusion_type.value}, which it "
            f"also matches as a premise. That would let an inference be consumed "
            f"as a fact on the next iteration."
        )


def rules_producing(relation_type: RT, rules: list[Rule] | None = None) -> list[Rule]:
    """The rules that can derive a given relation type (spec section 12).

    Section 12 lists ``composition_rules`` on RelationSpec. Answering it by
    querying the loaded rule set keeps each rule declared in one place.
    """
    rules = rules if rules is not None else load_rules()
    return [rule for rule in rules if rule.conclusion_type is relation_type]
