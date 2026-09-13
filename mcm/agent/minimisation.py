"""Context minimisation (spec section 34, development step 16).

Spec section 34:

    C* = argmin |C|  subject to  C ⊨ Q

    "Find the smallest evidence/relationship substructure that allows the agent to
    answer the query with acceptable confidence."

Two halves, and the second is the one that usually gets skipped.

**argmin |C|.** Greedy ablation. Every item in the context is removed in turn, and
the removal is kept if the package still answers. Greedy, not optimal: minimum set
cover is NP-hard, and a prototype that claimed the true minimum would be claiming
something it did not compute. The search is bounded and says when it hit the bound,
like every other fixpoint in this system.

**C ⊨ Q.** Replay. A store is built containing *only* the reduced context, the
query is re-run against it, and every surviving claim has to come back with the
same confidence it had against the full repository. This is what makes ⊨ a test
rather than a word. A construction argument - "each claim's justification is
present, therefore it holds" - would be circular, because the justification is
exactly what the construction chose to keep.

Entailment here means three things, and all three are checked:

1. **Reproduction.** Impact, test and dependency claims are recomputed on the
   reduced store and must match. These are the claims the reasoning engine can
   reproduce from a relation graph, so they get the real test.
2. **Traceability.** Every retained relation keeps at least one of its evidence
   records. Spec section 43 requires an inferred answer to be traceable to its
   sources; a context that reproduces the numbers but has thrown away the evidence
   answers the query and fails the requirement. Without this clause ablation strips
   every evidence record, because impact analysis never reads one.
3. **Presence.** Constraint violations and commit correlations keep their cited
   objects and relations. The reasoning that produced them is not a graph walk from
   the focus, so replay cannot regenerate them, and presence is the honest weaker
   check. This is stated rather than hidden behind the word "verified".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from ..algebra.dependency import dependencies_of
from ..core.relations import MCMRelation
from ..reasoning.dependency_propagation import analyse_impact
from ..storage.database import Store
from ..storage.sqlite_store import SQLiteStore
from .context import (CHANGE, CONSTRAINT, DEPENDENCY, IMPACT, TEST, Claim,
                      ContextPackage, Justification)

#: Confidence equality tolerance when comparing a replayed claim with the
#: original. Confidences are products of small floats, so exact equality would
#: fail on arithmetic ordering rather than on anything meaningful.
TOLERANCE = 1e-9


@dataclass(frozen=True)
class ContextItems:
    """The three sets that make up C."""

    objects: frozenset[str]
    relations: frozenset[str]
    evidence: frozenset[str]

    @property
    def size(self) -> int:
        return len(self.objects) + len(self.relations) + len(self.evidence)

    def without(self, kind: str, item_id: str) -> "ContextItems":
        if kind == "object":
            return ContextItems(self.objects - {item_id}, self.relations, self.evidence)
        if kind == "relation":
            return ContextItems(self.objects, self.relations - {item_id}, self.evidence)
        return ContextItems(self.objects, self.relations, self.evidence - {item_id})


@dataclass
class Sufficiency:
    """Whether C ⊨ Q, and if not, which clause failed."""

    ok: bool
    reason: str = ""
    reproduced: int = 0


@dataclass
class MinimisationReport:
    gathered: int
    before: int
    after: int
    removed: list[tuple[str, str]] = field(default_factory=list)
    ablations: int = 0
    bound_hit: bool = False
    verified: bool = False
    verification: str = ""

    @property
    def efficiency(self) -> float:
        """Retained over gathered.

        Spec section 48 defines ContextEfficiency as useful over total retrieved
        information, and this is that ratio computed the only way this phase can
        compute it: useful means "appears in the justification of a claim that
        survives replay". Section 48's metric is about task outcomes and needs the
        evaluation framework in sections 45 to 50, which does not exist yet. This
        number measures the reduction, not the benefit.
        """
        return self.after / self.gathered if self.gathered else 1.0

    @property
    def already_minimal(self) -> bool:
        """True when ablation found nothing to remove.

        Not a failure. It means the justification closure was already minimal
        under the entailment test, which is a result about the construction rather
        than a missed opportunity.
        """
        return not self.removed

    def summary(self) -> str:
        state = ("verified" if self.verified
                 else f"NOT VERIFIED: {self.verification}")
        return (f"{self.gathered} gathered -> {self.before} justified -> "
                f"{self.after} retained "
                f"(efficiency {self.efficiency:.2f}, {self.ablations} ablations"
                f"{', bound hit' if self.bound_hit else ''}, {state})")


def minimise(store: Store, package: ContextPackage, *, max_depth: int = 6,
             max_ablations: int = 500,
             as_of: datetime | None = None) -> ContextPackage:
    """Reduce a package to the smallest context that still answers it.

    Returns a new package. The original is not modified, so a caller can report
    what was dropped by comparing the two.
    """
    items = _items_of(package)
    gathered = package.gathered_size
    report = MinimisationReport(gathered=gathered, before=items.size, after=items.size)

    for kind, item_id in _ablation_order(package, items):
        if report.ablations >= max_ablations:
            report.bound_hit = True
            break
        report.ablations += 1
        reduced = items.without(kind, item_id)
        if _sufficient(store, package, reduced, max_depth, as_of).ok:
            items = reduced
            report.removed.append((kind, item_id))

    final = _sufficient(store, package, items, max_depth, as_of)
    report.verified = final.ok
    report.verification = final.reason
    report.after = items.size

    reduced_package = _rebuild(store, package, items)
    reduced_package.minimised = True
    reduced_package.minimisation = report
    return reduced_package


# --- entailment -----------------------------------------------------------

def _sufficient(store: Store, package: ContextPackage, items: ContextItems,
                max_depth: int, as_of: datetime | None) -> Sufficiency:
    """Does C entail the package's claims? The three clauses in the module docstring."""
    if package.focus.id not in items.objects:
        return Sufficiency(False, "the focus object is not in the context")

    presence = _presence_holds(package, items)
    if presence is not None:
        return Sufficiency(False, presence)

    traceability = _traceability_holds(store, items)
    if traceability is not None:
        return Sufficiency(False, traceability)

    return _replay_holds(store, package, items, max_depth, as_of)


def _replay_holds(store: Store, package: ContextPackage, items: ContextItems,
                  max_depth: int, as_of: datetime | None) -> Sufficiency:
    """Re-run the query against a store holding only C.

    This is the clause that makes the whole thing worth doing. If a relation that
    the answer actually depends on has been dropped, the replayed closure either
    loses the claim or reaches it by a longer path at a lower confidence, and
    either way the comparison fails.
    """
    replay = _replay_store(store, items)
    try:
        impact = analyse_impact(replay, package.focus.id, max_depth=max_depth,
                                as_of=as_of)
        closure = dependencies_of(replay, package.focus.id, max_depth=max_depth,
                                  as_of=as_of)
        affected = {a.object.id: a.confidence for a in impact.all_affected}
        depends = {object_id: path.confidence
                   for object_id, path in closure.paths.items()}

        reproduced = 0
        for claim in package.claims:
            if claim.kind in (IMPACT, TEST):
                actual = affected.get(claim.subject_id)
            elif claim.kind == DEPENDENCY:
                actual = depends.get(claim.subject_id)
            else:
                continue
            if actual is None:
                return Sufficiency(
                    False, f"claim lost on replay: {claim.statement}")
            if not math.isclose(actual, claim.confidence, abs_tol=TOLERANCE):
                return Sufficiency(
                    False, f"confidence changed on replay for {claim.subject_id}: "
                           f"{claim.confidence:.4f} -> {actual:.4f}")
            reproduced += 1
        return Sufficiency(True, "", reproduced)
    finally:
        replay.close()


def _presence_holds(package: ContextPackage, items: ContextItems) -> str | None:
    """What replay cannot see has to be checked directly.

    **Objects, for every claim.** Traversal runs on the relation argument index and
    never loads an object, so a replay is entirely blind to whether the objects
    exist: drop ``validate_token`` and the closure still walks straight through the
    gap at the same confidence. But the claim *names* it, and spec section 43
    renders the reasoning path as names. A context that names an object it does not
    contain has not answered the query, it has produced a dangling pointer.

    **Relations, for constraint and change claims.** Replay cannot regenerate
    these: a constraint violation comes from evaluating a predicate over the graph,
    a commit correlation from the history, and neither is a dependency walk from the
    focus. Presence is the weaker check, and calling it what it is costs nothing.
    """
    for claim in package.claims:
        missing = {*claim.justification.object_ids} - items.objects
        if claim.kind in (CONSTRAINT, CHANGE):
            missing |= {*claim.justification.relation_ids} - items.relations
        if missing:
            return (f"{claim.kind} claim lost its support: {claim.statement} "
                    f"(missing {sorted(missing)[0]})")
    return None


def _traceability_holds(store: Store, items: ContextItems) -> str | None:
    """Every retained relation keeps at least one evidence record (spec section 43).

    Impact analysis never reads evidence, so without this clause the ablation
    would remove all of it and the package would still 'answer' the query - while
    being unable to show a single source for any of it.
    """
    for relation_id in items.relations:
        relation = store.get_relation(relation_id)
        if relation is None or not relation.evidence_ids:
            continue
        if not ({*relation.evidence_ids} & items.evidence):
            return f"relation {relation_id} has no evidence left to cite"
    return None


def _replay_store(store: Store, items: ContextItems) -> SQLiteStore:
    """An in-memory store containing exactly C.

    Building a real store rather than filtering reads keeps the replay honest: the
    query runs through the same code path as any other query, against a repository
    in which the removed items were never observed.
    """
    replay = SQLiteStore()
    for object_id in items.objects:
        obj = store.get_object(object_id)
        if obj is not None:
            replay.put_object(obj)
    for evidence_id in items.evidence:
        evidence = store.get_evidence(evidence_id)
        if evidence is not None:
            replay.put_evidence(evidence)
    for relation_id in items.relations:
        relation = store.get_relation(relation_id)
        if relation is None:
            continue
        if relation.provenance_id:
            provenance = store.get_provenance(relation.provenance_id)
            if provenance is not None:
                replay.put_provenance(provenance)
        replay.put_relation(_restricted(relation, items))
    return replay


def _restricted(relation: MCMRelation, items: ContextItems) -> MCMRelation:
    """A relation as it would be with the dropped evidence never recorded."""
    kept = [eid for eid in relation.evidence_ids if eid in items.evidence]
    if kept == relation.evidence_ids:
        return relation
    return MCMRelation(
        id=relation.id, relation_type=relation.relation_type,
        arguments=list(relation.arguments), properties=dict(relation.properties),
        confidence=relation.confidence, evidence_ids=kept,
        valid_from=relation.valid_from, valid_until=relation.valid_until,
        provenance_id=relation.provenance_id, inference=relation.inference,
    )


# --- search ---------------------------------------------------------------

def _ablation_order(package: ContextPackage,
                    items: ContextItems) -> list[tuple[str, str]]:
    """What to try removing, and in what order.

    Evidence first: it is the only category that is routinely redundant, because a
    relation observed twice cites two records and one is enough to trace it.
    Relations and objects follow, rarest support first, so the greedy pass spends
    its budget where a removal is most likely to stick. The focus is never offered.
    """
    support: dict[str, int] = {}
    for claim in package.claims:
        for item_id in (*claim.justification.object_ids,
                        *claim.justification.relation_ids,
                        *claim.justification.evidence_ids):
            support[item_id] = support.get(item_id, 0) + 1

    order: list[tuple[str, str]] = []
    order.extend(("evidence", eid) for eid in sorted(
        items.evidence, key=lambda i: (support.get(i, 0), i)))
    order.extend(("relation", rid) for rid in sorted(
        items.relations, key=lambda i: (support.get(i, 0), i)))
    order.extend(("object", oid) for oid in sorted(
        items.objects - {package.focus.id}, key=lambda i: (support.get(i, 0), i)))
    return order


def _items_of(package: ContextPackage) -> ContextItems:
    return ContextItems(
        objects=frozenset(package.object_ids | {package.focus.id}),
        relations=frozenset(package.relation_ids),
        evidence=frozenset(package.evidence_ids),
    )


def _rebuild(store: Store, package: ContextPackage,
             items: ContextItems) -> ContextPackage:
    """The package as it stands on the reduced context.

    Justifications are pruned to what survived, so ``size`` reports the reduced
    context rather than recomputing the original from claims that still name
    removed items.
    """
    claims = [
        Claim(kind=claim.kind, statement=claim.statement,
              confidence=claim.confidence, is_fact=claim.is_fact,
              subject_id=claim.subject_id,
              # Filtered in place, never re-sorted: a justification's objects are
              # a reasoning path in traversal order, and sorting them turns the
              # section 43 explanation into a set of names in alphabetical order.
              justification=Justification(
                  object_ids=tuple(i for i in claim.justification.object_ids
                                   if i in items.objects),
                  relation_ids=tuple(i for i in claim.justification.relation_ids
                                     if i in items.relations),
                  evidence_ids=tuple(i for i in claim.justification.evidence_ids
                                     if i in items.evidence),
              ))
        for claim in package.claims
    ]
    entities = [obj for obj in package.entities if obj.id in items.objects]
    return ContextPackage(
        goal=package.goal, focus=package.focus, claims=claims, entities=entities,
        constraints=list(package.constraints),
        recent_changes=list(package.recent_changes),
        uncertainties=list(package.uncertainties),
        actions=list(package.actions), retrieved=package.retrieved,
    )
