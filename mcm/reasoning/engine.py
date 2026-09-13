"""Forward-chaining inference engine (spec sections 35, 36).

Runs a rule set to a fixpoint over the asserted relations in the store, and
returns the derivations. Nothing is written back. Spec Rule 2 holds by
construction rather than by discipline: a derivation cannot go stale because it
does not outlive the query that asked for it.

Confidence is computed over the *flattened* chain of asserted edges behind a
derivation, not over its immediate premises. Composing a two-hop derivation with
a third edge scores the three original AST edges, so a derived relation and the
equivalent path found by ``mcm.algebra.dependency`` carry the same number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator

from ..algebra.confidence import path_confidence
from ..core.ids import relation_id
from ..core.provenance import ExtractionMethod
from ..core.relations import Inference, MCMRelation, RelationType as RT
from ..storage.database import Store
from .rules import Rule, load_rules

#: Bounds on the fixpoint. A rule set over a large repository can derive a
#: quadratic number of relations, so both are stated rather than implicit, and
#: hitting either is reported instead of silently truncating.
MAX_ITERATIONS = 12
MAX_DERIVED = 20_000

Bindings = dict[str, str]


@dataclass
class Derivation:
    """The result of running a rule set to fixpoint."""

    derived: dict[tuple[RT, tuple[str, ...]], MCMRelation] = field(default_factory=dict)
    iterations: int = 0
    hit_iteration_limit: bool = False
    hit_size_limit: bool = False
    #: Rule name -> how many relations it contributed. Useful for ablation.
    rule_counts: dict[str, int] = field(default_factory=dict)

    @property
    def relations(self) -> list[MCMRelation]:
        return sorted(self.derived.values(), key=lambda r: (r.relation_type.value, r.arguments))

    @property
    def complete(self) -> bool:
        """False when a bound stopped the fixpoint before it settled."""
        return not (self.hit_iteration_limit or self.hit_size_limit)

    def of_type(self, relation_type: RT) -> list[MCMRelation]:
        return [r for r in self.relations if r.relation_type is relation_type]

    def between(self, source: str, target: str) -> list[MCMRelation]:
        return [r for r in self.relations if r.arguments[:2] == [source, target]]


def derive(store: Store, rules: list[Rule] | None = None, *,
           as_of: datetime | None = None,
           max_iterations: int = MAX_ITERATIONS,
           max_derived: int = MAX_DERIVED) -> Derivation:
    """Run rules to a fixpoint over the relations valid at ``as_of``."""
    rules = rules if rules is not None else load_rules()
    asserted = list(store.all_relations(as_of=as_of))
    by_id = {relation.id: relation for relation in asserted}
    result = Derivation()

    working: list[MCMRelation] = list(asserted)
    for iteration in range(1, max_iterations + 1):
        result.iterations = iteration
        index = _index_by_type(working)
        new: list[MCMRelation] = []

        for rule in rules:
            for bindings, premises in _match(rule, index):
                conclusion = _conclude(rule, bindings, premises, by_id)
                key = (conclusion.relation_type, tuple(conclusion.arguments))
                incumbent = result.derived.get(key)
                if incumbent is not None and not _prefer(conclusion, incumbent):
                    continue
                if incumbent is None:
                    result.rule_counts[rule.name] = result.rule_counts.get(rule.name, 0) + 1
                result.derived[key] = conclusion
                new.append(conclusion)

        if len(result.derived) > max_derived:
            result.hit_size_limit = True
            break
        if not new:
            break
        working = asserted + list(result.derived.values())
    else:
        result.hit_iteration_limit = True

    return result


# --- matching -------------------------------------------------------------

def _index_by_type(relations: list[MCMRelation]) -> dict[RT, list[MCMRelation]]:
    index: dict[RT, list[MCMRelation]] = {}
    for relation in relations:
        index.setdefault(relation.relation_type, []).append(relation)
    return index


def _match(rule: Rule, index: dict[RT, list[MCMRelation]]
           ) -> Iterator[tuple[Bindings, list[MCMRelation]]]:
    """Every consistent assignment of the rule's variables.

    A nested-loop join. Fine at prototype scale; a real implementation would
    index premises on their bound arguments and join in selectivity order.
    """
    yield from _match_from(rule, index, 0, {}, [])


def _match_from(rule: Rule, index: dict[RT, list[MCMRelation]], position: int,
                bindings: Bindings, matched: list[MCMRelation]
                ) -> Iterator[tuple[Bindings, list[MCMRelation]]]:
    if position == len(rule.premises):
        if _distinct_holds(rule, bindings):
            yield dict(bindings), list(matched)
        return

    premise = rule.premises[position]
    for relation_type in premise.types:
        for relation in index.get(relation_type, ()):
            extended = _unify(premise.arguments, relation.arguments, bindings)
            if extended is None:
                continue
            matched.append(relation)
            yield from _match_from(rule, index, position + 1, extended, matched)
            matched.pop()


def _unify(pattern: tuple[str, ...], arguments: list[str],
           bindings: Bindings) -> Bindings | None:
    """Extend bindings so ``pattern`` matches ``arguments``, or return None."""
    if len(pattern) != len(arguments):
        return None
    extended = dict(bindings)
    for token, value in zip(pattern, arguments):
        if token.startswith("?"):
            if extended.setdefault(token, value) != value:
                return None
        elif token != value:
            return None
    return extended


def _distinct_holds(rule: Rule, bindings: Bindings) -> bool:
    return all(bindings.get(left) != bindings.get(right) for left, right in rule.distinct)


# --- conclusion construction ---------------------------------------------

def _conclude(rule: Rule, bindings: Bindings, premises: list[MCMRelation],
              by_id: dict[str, MCMRelation]) -> MCMRelation:
    arguments = [bindings.get(token, token) for token in rule.conclusion_arguments]
    chain = _flatten(premises, by_id)
    objects = _path_objects(premises)

    return MCMRelation(
        id=relation_id(rule.conclusion_type.value, arguments,
                       ExtractionMethod.SYMBOLIC_INFERENCE.value),
        relation_type=rule.conclusion_type,
        arguments=arguments,
        properties={"rule": rule.name, "depth": len(chain)},
        confidence=path_confidence(chain),
        evidence_ids=[eid for relation in chain for eid in relation.evidence_ids],
        inference=Inference(
            rule=rule.name,
            path=objects,
            premise_relation_ids=[relation.id for relation in chain],
        ),
    )


def _flatten(premises: list[MCMRelation], by_id: dict[str, MCMRelation]) -> list[MCMRelation]:
    """Resolve premises down to the asserted edges beneath them.

    A derived premise contributes the asserted chain it was built from, so
    confidence is always scored over original observations rather than over
    intermediate conclusions. Without this, a two-step derivation would be scored
    with the attenuation of POSSIBLY_DEPENDS_ON instead of the CALLS edges that
    actually justify it.
    """
    chain: list[MCMRelation] = []
    for premise in premises:
        if premise.is_derived:
            chain.extend(by_id[rid] for rid in premise.inference.premise_relation_ids
                         if rid in by_id)
        else:
            chain.append(premise)
    return chain or list(premises)


def _path_objects(premises: list[MCMRelation]) -> list[str]:
    """The object walk implied by a chain of premises, for the explanation."""
    objects: list[str] = []
    for premise in premises:
        source = premise.inference.path[0] if premise.is_derived else premise.arguments[0]
        tail = premise.inference.path[1:] if premise.is_derived else premise.arguments[1:]
        if not objects:
            objects.append(source)
        elif objects[-1] != source:
            objects.append(source)
        objects.extend(tail)
    return objects


def _prefer(candidate: MCMRelation, incumbent: MCMRelation) -> bool:
    """Shortest derivation wins, then highest confidence.

    Deliberately the same ordering as ``mcm.algebra.dependency._prefer``. The
    closure walk and the rule fixpoint are independent implementations of the
    same reasoning, and they are only cross-checkable if they break ties the same
    way.
    """
    candidate_depth = candidate.properties.get("depth", 0)
    incumbent_depth = incumbent.properties.get("depth", 0)
    if candidate_depth != incumbent_depth:
        return candidate_depth < incumbent_depth
    if candidate.confidence != incumbent.confidence:
        return candidate.confidence > incumbent.confidence
    return candidate.inference.path < incumbent.inference.path
