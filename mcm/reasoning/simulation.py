"""Pre-action simulation (spec section 60, development step 18).

    K_t + Δ  →  K_{t+1}^predicted

then check dependencies, constraints, tests, APIs and architectural rules against
that predicted state, "and then let the agent decide whether to proceed".

That last clause is a design instruction, so this module returns a report and never
a verdict. There is no ``should_proceed`` field: whether a newly violated
architectural constraint is a blocker or the entire point of the change is not a
question the relation graph can answer.

**How Δ is applied.** The successor state is built by forking the store and closing
the relations the change destroys - closing, not deleting, because that is what the
temporal model does when a fact stops being true (spec section 18). The simulation
therefore uses the same mechanism a real re-ingestion would, and a constraint
evaluated against `K_{t+1}` runs through exactly the code path it runs through
normally.

**What a behaviour change does to the graph: nothing.** `SIGNATURE` and `BEHAVIOUR`
changes leave every relation standing, so `K_{t+1}` is structurally identical to
`K_t` and no constraint verdict can move. That is a real result rather than a gap -
it says that this class of change cannot be cleared or condemned statically, and
that the tests are the only evidence available. The report says so instead of
printing an empty diff that reads like an all-clear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..core.change import Change, ChangeKind
from ..core.constraints import Constraint, ConstraintType, Verdict
from ..core.objects import MCMObject, utcnow
from ..storage.database import Store
from ..storage.sqlite_store import SQLiteStore
from .change_propagation import PropagationResult, propagate
from .constraint_checker import ConstraintResult, check_constraints, load_constraints
from .dependency_propagation import AffectedObject, analyse_impact

#: Change kinds that remove the target's relations from the graph. A rename is
#: here because object identity is built from the qualname (spec section 20): the
#: renamed definition is a *different object*, and every edge into the old identity
#: stops being true. The new identity has no dependents yet, which is exactly the
#: state the simulation should show.
STRUCTURAL: frozenset[ChangeKind] = frozenset({ChangeKind.REMOVE, ChangeKind.RENAME})


@dataclass(frozen=True)
class VerdictChange:
    """One constraint whose verdict moves between K_t and K_{t+1}."""

    name: str
    type: ConstraintType
    before: Verdict
    after: Verdict
    messages: tuple[str, ...] = ()

    @property
    def is_regression(self) -> bool:
        return self.after is Verdict.VIOLATED and self.before is not Verdict.VIOLATED


@dataclass
class Simulation:
    change: Change
    target: MCMObject
    propagation: PropagationResult
    verdict_changes: list[VerdictChange] = field(default_factory=list)
    affected_tests: list[AffectedObject] = field(default_factory=list)
    affected_apis: list[AffectedObject] = field(default_factory=list)
    closed_relations: int = 0
    as_of: datetime | None = None

    @property
    def structural(self) -> bool:
        """Did the change alter the relation graph at all?"""
        return self.change.kind in STRUCTURAL

    @property
    def regressions(self) -> list[VerdictChange]:
        return [v for v in self.verdict_changes if v.is_regression]

    @property
    def architectural_regressions(self) -> list[VerdictChange]:
        """Spec section 60 lists architectural rules as their own check."""
        return [v for v in self.regressions
                if v.type is ConstraintType.ARCHITECTURAL_CONSTRAINT]

    @property
    def resolved(self) -> list[VerdictChange]:
        return [v for v in self.verdict_changes
                if v.before is Verdict.VIOLATED and v.after is not Verdict.VIOLATED]

    @property
    def newly_undecidable(self) -> list[VerdictChange]:
        """Constraints that stop being checkable.

        Worth separating from a regression and from an all-clear: a constraint
        that was SATISFIED and becomes UNEVALUATABLE has not been cleared, it has
        gone dark, and the constraint engine's whole stance is that those are
        different (spec section 13).
        """
        return [v for v in self.verdict_changes
                if v.after is Verdict.UNEVALUATABLE
                and v.before is not Verdict.UNEVALUATABLE]


def simulate(store: Store, change: Change, *, max_depth: int = 6,
             constraints: list[Constraint] | None = None,
             as_of: datetime | None = None) -> Simulation:
    """Build ``K_{t+1}^predicted`` and check it (spec section 60)."""
    target = store.get_object(change.target_id)
    if target is None:
        raise KeyError(f"unknown object: {change.target_id}")

    propagation = propagate(store, change, max_depth=max_depth, as_of=as_of)
    impact = analyse_impact(store, change.target_id, max_depth=max_depth, as_of=as_of)

    constraints = constraints if constraints is not None else load_constraints()
    before = check_constraints(store, constraints, as_of=as_of)

    successor, closed = _apply(store, change, as_of)
    try:
        after = check_constraints(successor, constraints, as_of=as_of)
    finally:
        successor.close()

    return Simulation(
        change=change, target=target, propagation=propagation,
        verdict_changes=_diff(before, after),
        affected_tests=impact.affected_tests,
        affected_apis=impact.affected_apis,
        closed_relations=closed, as_of=as_of,
    )


def _apply(store: Store, change: Change,
           as_of: datetime | None) -> tuple[SQLiteStore, int]:
    """``K_t + Δ``: fork the store and apply the change to the fork.

    Only relations are touched. The object record survives a REMOVE for the same
    reason a deleted definition's history survives re-ingestion: the temporal model
    keeps what was true and marks when it stopped being true (spec section 18).
    """
    successor = _fork(store)
    if change.kind not in STRUCTURAL:
        return successor, 0

    moment = as_of or utcnow()
    closed = 0
    for relation in successor.relations_for(change.target_id, direction="any",
                                            as_of=as_of):
        successor.close_relation(relation.id, moment)
        closed += 1

    obj = successor.get_object(change.target_id)
    if obj is not None:
        obj.state = {**obj.state, "simulated": change.kind.value}
        successor.put_object(obj)
    return successor, closed


def _fork(store: Store) -> SQLiteStore:
    """An in-memory copy of the store, safe to modify.

    A copy rather than a filtered read, so that everything downstream - the
    constraint engine especially - runs against a real store through its ordinary
    code path and cannot accidentally see the original.
    """
    fork = SQLiteStore()
    for obj in store.all_objects():
        fork.put_object(obj)
    for evidence in store.all_evidence():
        fork.put_evidence(evidence)
    for relation in store.all_relations(include_derived=False,
                                        include_historical=True):
        if relation.provenance_id:
            provenance = store.get_provenance(relation.provenance_id)
            if provenance is not None:
                fork.put_provenance(provenance)
        fork.put_relation(relation)
    return fork


def _diff(before: list[ConstraintResult],
          after: list[ConstraintResult]) -> list[VerdictChange]:
    """Constraints whose verdict moved. Unchanged verdicts are not reported."""
    previous = {result.constraint.name: result for result in before}
    changes: list[VerdictChange] = []
    for result in after:
        was = previous.get(result.constraint.name)
        if was is None or was.verdict is result.verdict:
            continue
        changes.append(VerdictChange(
            name=result.constraint.name,
            type=result.constraint.type,
            before=was.verdict,
            after=result.verdict,
            messages=tuple(v.message for v in result.violations),
        ))
    return sorted(changes, key=lambda v: (not v.is_regression, v.name))


def explain(simulation: Simulation) -> str:
    """Render the five section 60 checks. No recommendation: section 60 leaves the
    decision to the agent."""
    change = simulation.change
    lines = [f"Simulating: {change.describe()}",
             f"  {simulation.target.name}  [{simulation.target.type.value}]"]
    if simulation.structural:
        lines.append(f"  K(t+1): {simulation.closed_relations} relations close")
    else:
        lines.append("  K(t+1): structurally identical - a "
                     f"{change.kind.value} change closes no relation")
    lines.append("")

    must = simulation.propagation.must_update
    lines.append(f"Dependencies ({len(must)} require an edit)")
    for predicted in must:
        sites = f"  {', '.join(predicted.sites)}" if predicted.sites else ""
        lines.append(f"  {predicted.object.name}{sites}")
    if not must:
        lines.append("  no source edit is forced by this change")
    lines.append("")

    lines.append("Constraints")
    if not simulation.structural:
        lines.append("  unchanged, and necessarily so: this change closes no")
        lines.append("  relation, so no predicate over the graph can move. Only")
        lines.append("  the tests below can catch a behaviour regression.")
    elif not simulation.verdict_changes:
        lines.append("  every verdict holds")
    else:
        for verdict in simulation.verdict_changes:
            marker = "REGRESSION" if verdict.is_regression else "changed"
            lines.append(f"  [{marker}] {verdict.name}: "
                         f"{verdict.before.value} -> {verdict.after.value}")
            for message in verdict.messages[:3]:
                lines.append(f"      {message}")
    lines.append("")

    lines.append("Architectural rules")
    architectural = simulation.architectural_regressions
    if architectural:
        for verdict in architectural:
            lines.append(f"  VIOLATED: {verdict.name}")
    else:
        lines.append("  none newly violated")
    lines.append("")

    tests = simulation.affected_tests
    lines.append(f"Tests ({len(tests)})")
    if tests:
        for affected in tests:
            lines.append(f"  {affected.object.name}  ({affected.confidence:.2f} "
                         f"{affected.band})")
    else:
        lines.append("  no test reaches this change site")
    lines.append("")

    lines.append(f"APIs ({len(simulation.affected_apis)})")
    for affected in simulation.affected_apis:
        lines.append(f"  {affected.object.name}")
    if not simulation.affected_apis:
        lines.append("  none - no API objects exist in this repository")
    lines.append("")
    lines.append("This is a prediction about a state that does not exist. "
                 "Nothing was written.")
    return "\n".join(lines)


def simulation_json(simulation: Simulation) -> dict:
    from .change_propagation import propagation_json  # noqa: PLC0415 - cyclic at import

    return {
        "mode": "simulate",
        "change": {"target": simulation.change.target_id,
                   "kind": simulation.change.kind.value,
                   "details": simulation.change.details},
        "structural": simulation.structural,
        "closed_relations": simulation.closed_relations,
        "propagation": propagation_json(simulation.propagation),
        "constraint_changes": [
            {"name": v.name, "type": v.type.value, "before": v.before.value,
             "after": v.after.value, "regression": v.is_regression,
             "messages": list(v.messages)}
            for v in simulation.verdict_changes
        ],
        "regressions": [v.name for v in simulation.regressions],
        "architectural_regressions": [v.name for v in
                                      simulation.architectural_regressions],
        "newly_undecidable": [v.name for v in simulation.newly_undecidable],
        "affected_tests": [a.object.name for a in simulation.affected_tests],
        "affected_apis": [a.object.name for a in simulation.affected_apis],
        "as_of": simulation.as_of.isoformat() if simulation.as_of else None,
    }
