# Constraints

## Three verdicts, not two

Spec section 13 requires constraints to be evaluatable. Section 14 lists eight
types. Those two requirements are in tension, because the eight types are not the
same kind of claim.

Four are claims about the relation graph. `ARCHITECTURAL_CONSTRAINT`,
`DEPENDENCY_CONSTRAINT`, `SECURITY_CONSTRAINT` and `TEST_CONSTRAINT` ask whether
one part of a repository reaches another, and the store already holds everything
needed to settle them.

Four are claims about runtime values. `PRECONDITION`, `POSTCONDITION`,
`INVARIANT` and `TYPE_CONSTRAINT` say things like `token.expiry > now`. Deciding
those statically needs symbolic execution.

So `Verdict` has three members:

```
SATISFIED       checked and holds
VIOLATED        checked and fails, with the objects and evidence
UNEVALUATABLE   well-formed, but not decidable from what MCM knows
```

`UNEVALUATABLE` names what it would need:

```
POSTCONDITION  token_expiry_is_in_the_future  UNEVALUATABLE
    needs runtime values: now, token.expiry
    scope: repo://app/auth.py#function:authenticate
    expression: token.expiry > now
```

That list comes from parsing the expression with `ast` and collecting the free
names, so it names exactly what is missing rather than echoing the expression
back. Reporting `SATISFIED` for a constraint that was never checked would be the
one outcome worse than reporting nothing.

## Predicates

A decidable constraint names a predicate the checker implements:

| Predicate | Question |
|---|---|
| `forbid_dependency` | Does anything matching `from` reach anything matching `to`? Transitive, so an indirect route through a third module is caught. |
| `forbid_relation` | Does a direct edge of a named type exist between two selections? |
| `require_test` | Is every object in scope exercised by a test? `transitive: true` counts a test that reaches it through a helper. |
| `require_dependency` | Does everything matching `from` depend on something matching `to`? |

An unknown predicate is rejected when the constraint file loads, listing the ones
that exist.

## Selectors

A `Selector` picks a set of objects, and every field it sets is an additional
restriction:

```yaml
scope:
  type: Function
  id_glob: "repo://app/auth.py#function:*"
  name_glob: "test_*"
```

Globs run against the object ID, which is why the ID grammar in
`docs/semantic-model.md` is structured rather than opaque. `repo://app/auth.py#*`
selects everything in one file because the identity scheme puts the path in the
ID.

## Violations are traceable

A violation carries the objects involved, the relation IDs crossed, and the
evidence behind every hop, so it is as traceable as an inference (spec section
43). A `forbid_dependency` violation records the whole path:

```python
Violation(
    message="view depends on db/conn.py",
    objects=["ui/view.py", "mid/service.py", "db/conn.py"],
    relations=["rel:...", "rel:..."],
    evidence_ids=["ev:...", "ev:..."],
)
```

## A bug the constraints found

`require_test` with `transitive: true` first checked whether a `TESTS` edge
appeared anywhere in the dependency path returned by the closure. It reported
`create_token` as untested, which is wrong: `test_authenticate` calls
`authenticate`, which calls `create_token`.

The cause is worth recording. Ingestion writes *both* a `CALLS` edge and a
`TESTS` edge between a test and its subject, from two different extractors. The
closure keeps the higher-confidence route, which is always the AST-derived
`CALLS` edge at 1.0 rather than the heuristic `TESTS` edge at 0.9. So the `TESTS`
edge was never in the path being inspected.

The fix asks the right question: is any *Test-typed object* among the dependents?
`docs/provenance.md` explains why the two edges coexist; this is the first place
that design had a consequence for a caller.

## The failing constraint is the informative one

`methods_are_tested` fails on the fixture:

```
TEST_CONSTRAINT  methods_are_tested  VIOLATED
    __init__ has no test coverage
    is_active has no test coverage
```

That is correct, and it traces to a real gap rather than a bad constraint. The
only call to `User.is_active` is `user.is_active()`, where `user` is a local bound
to the result of `load_user`. Ingestion refuses to resolve that without type
inference and records it as unresolved. No `TESTS` edge reaches the method
because no edge of any kind reaches it.

The constraint engine is reporting the limit of what MCM currently knows, which
is the behaviour to want from it.

## Scoping a check to a change

Spec section 27 names `VIOLATIONS(change_123)`. `check_constraints` takes
`restrict_to`, a set of object IDs, so the objects an impact analysis says a
change would touch can be passed straight in:

```python
impact = analyse_impact(store, target)
touched = {a.object.id for a in impact.all_affected} | {impact.target.id}
results = check_constraints(store, constraints, restrict_to=touched)
```

This answers section 27's query without inventing a change format, which belongs
with the pre-action simulation of section 60.
