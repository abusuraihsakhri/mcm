# Inference

## Rules are data

Spec section 36 asks for declarative rules, and gives the reason: "This makes the
algebra extensible." Rules live in `mcm/rules/*.yaml`, loaded by
`mcm/reasoning/rules.py` and run by `mcm/reasoning/engine.py`.

```yaml
- name: transitive_dependency
  premises:
    - [[DEPENDS_ON, POSSIBLY_DEPENDS_ON], "?a", "?b"]
    - [[DEPENDS_ON, POSSIBLY_DEPENDS_ON], "?b", "?c"]
  conclusion: [POSSIBLY_DEPENDS_ON, "?a", "?c"]
  distinct: [["?a", "?c"]]
```

Names beginning with `?` are variables, unified across premises. Anything else is
a literal object ID, so a rule can be scoped to a specific part of a repository.

## Premises match through subsumption

A premise naming `DEPENDS_ON` matches every type the RelationSpec table subsumes
under it: `CALLS`, `USES`, `IMPORTS`, `INHERITS`, `IMPLEMENTS`, `READS`, `TESTS`.
Ingestion never writes a `DEPENDS_ON` row, so without this a rule written against
the abstraction would match nothing.

The effect is that rules are written against the algebra rather than against
whichever concrete edge types today's extractor emits. Adding a `DECORATES` edge
that declares `implies: [DEPENDS_ON]` makes every existing dependency rule apply
to it without editing any rule.

Listing several types in a premise, as `transitive_dependency` does, is how a
rule declares that it chains on its own output. That is what turns one pass into
a fixpoint.

## Validation happens at load time

A rule file is checked when it is read, not when a rule first fires:

- every conclusion variable must be bound by some premise
- every relation type must exist and have a RelationSpec
- `distinct` clauses must be pairs
- a rule may not conclude a type it also matches as a premise, unless that type
  is declared `derived_only`

The last check is spec Rule 2 made structural. A rule concluding `CALLS` from two
`CALLS` premises would put an inference into the same type as an observation, and
the next iteration would consume it as a fact. `RuleError` names the rule and the
file.

Inverse rules pass this check because `TESTED_BY` and `PART_OF` are never
asserted by ingestion. Deriving them cannot shadow an observation.

## The fixpoint

`derive()` runs every rule against the working set, adds what is new, and repeats
until nothing changes. On the fixture that settles in four iterations.

Both bounds are stated rather than implicit. `MAX_ITERATIONS` is 12 and
`MAX_DERIVED` is 20,000, and hitting either sets a flag that makes
`Derivation.complete` false. A truncated derivation says so instead of looking
like a settled one.

Deduplication keeps the shortest derivation, then the highest confidence, then
the lexicographically smaller path. That is deliberately the same ordering as
`_prefer` in `mcm/algebra/dependency.py`, for the reason in the next section.

## Confidence is scored over asserted edges

When a two-step derivation is composed with a third edge, `_flatten` resolves the
derived premise back to the asserted edges beneath it, so the conclusion is
scored over the three original AST observations. Scoring it over the intermediate
`POSSIBLY_DEPENDS_ON` would apply that type's attenuation instead of the
`CALLS` attenuation that actually justifies the claim.

This is what makes the engine and the closure walk directly comparable. On the
fixture they produce identical confidence for every object both reach:

```
validate_token  0.950     authenticate  0.902
login           0.857     test_auth.py  0.360
```

`tests/test_rules.py::TestCrossCheckAgainstClosure` asserts that agreement. Two
independently written implementations landing on the same numbers is stronger
evidence than either one passing its own tests.

## What the engine adds over the closure walk

Honestly: for pure dependency reasoning, not much. `dependents_of` already
computes the transitive closure by breadth-first search, and does it faster.

The engine earns its place in two ways.

**Cross-family rules.** The dependency closure never crosses `CONTAINS`, because
a file does not depend on its functions. But the converse holds: a class depends
on what its methods depend on. `container_inherits_member_dependency` expresses
that, and it is unreachable by any traversal over dependency edges alone. On the
fixture it derives that `repo://app` and `tests/` depend on `lib://jwt`, which
the closure cannot reach at all.

**Extensibility.** A new derivation is a YAML entry, not a code change. That is
the property spec section 36 is asking for, and it is what makes the ablation
studies in section 50 practical: removing a rule from the set is a one-line edit.

## Nothing is persisted

`derive()` returns its derivations and writes none of them. Spec Rule 2 holds by
construction rather than by discipline: an inference cannot go stale because it
does not outlive the query that asked for it, and there is no invalidation logic
to get wrong.

The cost is recomputation on every query. That is irrelevant at fixture scale and
will not be at repository scale. The alternative, materialising derivations with
per-premise invalidation, was rejected for this phase because getting it wrong
produces silently stale inferences carrying legitimate-looking provenance, which
is the worst failure mode available to a system whose point is traceable
reasoning.
