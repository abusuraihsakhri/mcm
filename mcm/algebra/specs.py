"""Algebraic metadata for relation types (spec sections 11 and 12).

Every relation type declares its formal properties. The reasoning engine consults
this table instead of assuming that traversal equals inference. Spec section 12:
"This prevents invalid reasoning."

Argument-order convention for the dependency family::

    arguments[0] depends on arguments[1]

so ``CALLS(authenticate, validate_token)`` means authenticate's behaviour is a
function of validate_token's - matching ``A = f(B)`` from spec section 10.

Section 12 also lists ``composition_rules`` on the spec. Those live in
``rules/*.yaml`` and are looked up with ``mcm.reasoning.rules.rules_producing``
rather than being written here, so that a rule is declared in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.relations import RelationType as RT


@dataclass(frozen=True)
class RelationSpec:
    name: RT
    arity: int
    symmetric: bool
    reflexive: bool
    transitive: bool
    composable: bool
    inverse: RT | None = None
    #: Supertypes this relation is subsumed by. ``CALLS`` implies ``DEPENDS_ON``,
    #: which is how the dependency algebra reasons over heterogeneous edges
    #: without materialising redundant DEPENDS_ON rows (spec Rule 5: keep
    #: deterministic analysis separate from inference).
    implies: tuple[RT, ...] = ()
    #: True when this relation type may only be produced by the inference engine.
    derived_only: bool = False
    semantics: str = ""


def _spec(name, arity=2, symmetric=False, reflexive=False, transitive=False,
          composable=False, inverse=None, implies=(), derived_only=False,
          semantics="") -> RelationSpec:
    return RelationSpec(name, arity, symmetric, reflexive, transitive, composable,
                        inverse, implies, derived_only, semantics)


D = (RT.DEPENDS_ON,)

RELATION_SPECS: dict[RT, RelationSpec] = {
    # --- Structural -------------------------------------------------------
    RT.CONTAINS: _spec(
        RT.CONTAINS, transitive=True, composable=True, inverse=RT.PART_OF,
        semantics="arguments[0] structurally contains arguments[1]. Containment "
                  "is NOT dependency: a file does not depend on its functions.",
    ),
    RT.PART_OF: _spec(
        RT.PART_OF, transitive=True, composable=True, inverse=RT.CONTAINS,
        semantics="Inverse of CONTAINS.",
    ),
    RT.DECLARES: _spec(
        RT.DECLARES, composable=True,
        semantics="arguments[0] introduces the name arguments[1] into scope. "
                  "Narrower than CONTAINS, which is purely positional.",
    ),
    RT.INHERITS: _spec(
        RT.INHERITS, transitive=True, composable=True, implies=D,
        semantics="arguments[0] is a subclass of arguments[1]. Transitive: a "
                  "grandchild class really does inherit from its grandparent.",
    ),
    RT.IMPLEMENTS: _spec(
        RT.IMPLEMENTS, composable=True, implies=D,
        semantics="arguments[0] satisfies the interface arguments[1]. Not "
                  "transitive: implementing an interface says nothing about what "
                  "that interface itself implements.",
    ),

    # --- Dependency -------------------------------------------------------
    RT.DEPENDS_ON: _spec(
        RT.DEPENDS_ON, transitive=True, composable=True,
        semantics="A > B: state(B) -> state(A). A's behaviour is a function of B's.",
    ),
    RT.IMPORTS: _spec(
        RT.IMPORTS, composable=True, implies=D,
        semantics="arguments[0] imports module arguments[1]. Not transitive: "
                  "importing a module does not import that module's imports into "
                  "the importer's namespace.",
    ),
    RT.CALLS: _spec(
        RT.CALLS, composable=True, implies=D,
        semantics="Execution of arguments[0] can invoke arguments[1]. NOT "
                  "transitive: A calls B and B calls C does not mean A calls C.",
    ),
    RT.USES: _spec(
        RT.USES, composable=True, implies=D,
        semantics="arguments[0] references a member of arguments[1].",
    ),
    RT.READS: _spec(
        RT.READS, composable=True, implies=D,
        semantics="arguments[0] reads state owned by arguments[1].",
    ),
    RT.WRITES: _spec(
        RT.WRITES, composable=True,
        semantics="arguments[0] mutates state owned by arguments[1]. Deliberately "
                  "not a dependency edge: a writer constrains its target rather "
                  "than being constrained by it.",
    ),
    RT.TESTS: _spec(
        RT.TESTS, composable=True, implies=D, inverse=RT.TESTED_BY,
        semantics="arguments[0] is a test exercising arguments[1]. A test depends "
                  "on its subject, so impact propagates into tests.",
    ),
    RT.TESTED_BY: _spec(
        RT.TESTED_BY, composable=True, inverse=RT.TESTS,
        semantics="Inverse of TESTS. Not a dependency edge in this direction: a "
                  "function does not depend on the test that covers it.",
    ),

    # --- Semantic ---------------------------------------------------------
    RT.REPRESENTS: _spec(RT.REPRESENTS, composable=True,
                         semantics="arguments[0] is a representation of the concept arguments[1]."),
    RT.DESCRIBES: _spec(RT.DESCRIBES, composable=True,
                        semantics="arguments[0] is documentation about arguments[1]."),
    RT.REFERS_TO: _spec(RT.REFERS_TO, composable=True,
                        semantics="arguments[0] mentions arguments[1] without depending on it."),
    RT.SIMILAR_TO: _spec(
        RT.SIMILAR_TO, symmetric=True, reflexive=True,
        semantics="Similarity is symmetric and reflexive but NOT transitive: "
                  "a~b and b~c does not give a~c at any useful threshold.",
    ),

    # --- Causal -----------------------------------------------------------
    RT.CAUSES: _spec(
        RT.CAUSES, composable=True,
        semantics="Evidence supports a causal link from arguments[0] to "
                  "arguments[1]. Not transitive and not implied by DEPENDS_ON: "
                  "spec section 37 requires separate evidence for causal claims.",
    ),
    RT.CONTRIBUTES_TO: _spec(RT.CONTRIBUTES_TO, composable=True,
                             semantics="Partial causal contribution, weaker than CAUSES."),
    RT.PREVENTS: _spec(RT.PREVENTS,
                       semantics="arguments[0] blocks arguments[1] from occurring."),
    RT.TRIGGERS: _spec(RT.TRIGGERS, composable=True,
                       semantics="arguments[0] initiates arguments[1]."),

    # --- Temporal ---------------------------------------------------------
    RT.PRECEDES: _spec(
        RT.PRECEDES, transitive=True, composable=True, inverse=RT.FOLLOWS,
        semantics="arguments[0] happens before arguments[1]. Transitive as a "
                  "strict ordering.",
    ),
    RT.FOLLOWS: _spec(RT.FOLLOWS, transitive=True, composable=True, inverse=RT.PRECEDES),
    RT.ACTIVE_DURING: _spec(
        RT.ACTIVE_DURING,
        semantics="arguments[0] holds throughout the interval arguments[1].",
    ),
    RT.SUPERSEDES: _spec(
        RT.SUPERSEDES, transitive=True, composable=True,
        semantics="arguments[0] replaces arguments[1] as the current knowledge. "
                  "The superseded record is retained (spec section 18).",
    ),

    # --- Agent knowledge --------------------------------------------------
    RT.OBSERVED: _spec(RT.OBSERVED, semantics="An agent directly observed arguments[1]."),
    RT.INFERRED: _spec(RT.INFERRED, semantics="An agent concluded arguments[1] by reasoning."),
    RT.ASSUMED: _spec(RT.ASSUMED, semantics="An agent proceeded on arguments[1] without evidence."),
    RT.CONFIRMED: _spec(RT.CONFIRMED, semantics="arguments[1] was checked and held."),
    RT.REJECTED: _spec(RT.REJECTED, semantics="arguments[1] was checked and failed."),

    # --- Transformation ---------------------------------------------------
    RT.TRANSFORMS: _spec(RT.TRANSFORMS, composable=True,
                         semantics="arguments[0] rewrites arguments[1] into another form."),
    RT.REFACTORS: _spec(RT.REFACTORS, composable=True,
                        semantics="Behaviour-preserving change from arguments[1] to arguments[0]."),
    RT.REPLACES: _spec(RT.REPLACES, transitive=True, composable=True,
                       semantics="arguments[0] takes over the role of arguments[1]."),
    RT.FIXES: _spec(RT.FIXES, composable=True,
                    semantics="arguments[0] resolves the defect arguments[1]."),
    RT.BREAKS: _spec(RT.BREAKS, composable=True,
                     semantics="arguments[0] causes arguments[1] to stop working."),

    # --- Equivalence ------------------------------------------------------
    RT.EQUIVALENT_TO: _spec(
        RT.EQUIVALENT_TO, symmetric=True, reflexive=True, transitive=True,
        composable=True, inverse=RT.EQUIVALENT_TO,
        semantics="Equivalence under a stated domain (spec section 10). The "
                  "domain belongs in relation properties, not in the type.",
    ),
    RT.SEMANTICALLY_EQUIVALENT_TO: _spec(
        RT.SEMANTICALLY_EQUIVALENT_TO, symmetric=True, reflexive=True,
        transitive=True, composable=True, inverse=RT.SEMANTICALLY_EQUIVALENT_TO,
    ),

    # --- Derived only -----------------------------------------------------
    RT.POSSIBLY_DEPENDS_ON: _spec(
        RT.POSSIBLY_DEPENDS_ON, transitive=False, composable=False, derived_only=True,
        semantics="Inferred dependency. Never promoted to DEPENDS_ON automatically "
                  "(spec Rule 2). Not subsumed under DEPENDS_ON, so the dependency "
                  "closure over asserted facts never silently consumes inferences.",
    ),
    RT.POSSIBLY_AFFECTS: _spec(
        RT.POSSIBLY_AFFECTS, transitive=False, composable=False, derived_only=True,
        semantics="Inferred impact: a change to arguments[0] may require a change "
                  "to arguments[1].",
    ),
}


def spec_for(relation_type: RT) -> RelationSpec:
    """Look up algebraic metadata, refusing to guess for undeclared types."""
    try:
        return RELATION_SPECS[relation_type]
    except KeyError:
        raise KeyError(
            f"No RelationSpec declared for {relation_type.value}. The algebra "
            f"refuses to assume properties for undeclared relation types."
        ) from None


def implies_dependency(relation_type: RT) -> bool:
    """True if an edge of this type is a dependency edge under subsumption."""
    if relation_type is RT.DEPENDS_ON:
        return True
    return RT.DEPENDS_ON in spec_for(relation_type).implies


def dependency_edge_types() -> set[RT]:
    """Every declared relation type that counts as a dependency edge."""
    return {rt for rt in RELATION_SPECS if implies_dependency(rt)}


def subsumed_by(relation_type: RT) -> set[RT]:
    """Every declared type that a premise naming ``relation_type`` should match.

    A rule premise of ``DEPENDS_ON(?a, ?b)`` must match the ``CALLS`` and ``USES``
    edges that ingestion actually wrote. This is that expansion, and it is what
    keeps rules written against the abstraction rather than against whichever
    concrete edge types the current extractor happens to emit.
    """
    return {rt for rt, spec in RELATION_SPECS.items()
            if rt is relation_type or relation_type in spec.implies}


def undeclared_types() -> set[RT]:
    """Relation types in the enum with no spec. Empty is the healthy state."""
    return {rt for rt in RT if rt not in RELATION_SPECS}
