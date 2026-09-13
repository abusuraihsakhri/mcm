# Relation algebra

## Traversal is not inference

A graph library will happily walk any path you give it. The specification's
section 11 warning is that a path existing in the data does not mean the
conclusion at the end of it follows. `RELATION_SPECS` in `mcm/algebra/specs.py`
records, per relation type, what the algebra is allowed to conclude.

```python
RelationSpec(
    name, arity, symmetric, reflexive, transitive, composable,
    inverse, implies, derived_only, semantics,
)
```

`spec_for` raises on an undeclared relation type rather than returning defaults.
An undeclared type is a gap in the model, and defaulting it to non-transitive and
non-composable would hide that gap behind behaviour that looks reasonable.

## Argument order

For the dependency family, `arguments[0]` depends on `arguments[1]`. So
`CALLS(authenticate, validate_token)` reads as "authenticate's behaviour is a
function of validate_token's", matching `A = f(B)` from section 10. This
convention is what makes impact a reverse traversal and dependency a forward one,
with no special-casing per relation type.

## Subsumption instead of materialisation

`CALLS`, `USES`, `IMPORTS`, `INHERITS` and `TESTS` each declare
`implies = (DEPENDS_ON,)`. The dependency closure walks the union of every edge
type that is subsumed under `DEPENDS_ON`.

The alternative is to materialise a `DEPENDS_ON` row alongside every `CALLS` row.
That was rejected: writing "authenticate depends on validate_token" as a stored
fact when what was actually observed is a call site duplicates the store and
blurs what the extractor saw. Subsumption keeps the store to observations and puts
the abstraction in the algebra, where it can be inspected and changed.

`CONTAINS` deliberately does not imply dependency. A file does not depend on the
functions it contains, and if it did, changing any function would report every
sibling in the file as impacted.

## Composition

`compose(r1, r2)` returns a derived relation or `None`. It returns `None` for
mismatched endpoints, for a type declared non-composable, and for any pair the
algebra has no rule for. Returning `None` rather than raising lets the closure walk
skip what it cannot reason about.

V1 declares one composition rule:

```yaml
rule:
  name: transitive_dependency
  premises:
    - DEPENDS_ON(A, B)      # or any type subsumed by it
    - DEPENDS_ON(B, C)
  conclusion:
    - POSSIBLY_DEPENDS_ON(A, C)
```

The conclusion is `POSSIBLY_DEPENDS_ON`, never `DEPENDS_ON`, even though
`DEPENDS_ON` is declared transitive. Transitivity holds for real dependency;
what the store contains is *extracted* dependency, which is incomplete. Dynamic
dispatch, reflection and indirection all produce real edges that static analysis
never sees, so each additional hop is another chance for the extracted graph to
diverge from the real one. Section 35 makes the same point with the word `MAY`.

Composing two `CALLS` relations therefore never produces a `CALLS` relation. That
is asserted directly in `tests/test_algebra.py`.

## Confidence along a path

`path_confidence` in `mcm/algebra/confidence.py` combines two independent factors:

```
confidence = min(edge confidences) * prod(EDGE_DECAY[type] for edges after the first)
```

The first factor is belief: a chain is no more believable than its least
believable edge. Unlike a product, `min` does not punish a long chain of
certainties for being long.

The second factor is attenuation, and it is not about belief. Every AST edge has
confidence 1.0, so belief alone would report a five-hop chain as certain.
Attenuation encodes how far a change actually propagates across each kind of
relation:

| Edge | Decay | Reasoning |
|---|---|---|
| `CALLS`, `USES`, `DEPENDS_ON`, `READS` | 0.95 | Tight. Changing a callee reaches its caller directly. |
| `INHERITS`, `IMPLEMENTS`, `TESTS` | 0.9 | Real but looser. `TESTS` is itself a heuristic assertion. |
| `IMPORTS` | 0.6 | File-granular and loose. "auth.py imports jwt_provider" says almost nothing about which functions a change touches. |

The first edge is exempt from attenuation, so a direct dependent keeps the
confidence its asserted relation carries. Beyond depth 1 the result is an
inference and is scored as one.

The practical effect on the section 65 scenario: `test_authenticate` arrives at
0.86 (HIGH) through a call chain, while `test_auth.py` arrives at 0.36 (LOW)
through an import chain of the same shape. A single global decay constant ranked
those two equally, which put file-level import noise alongside real call paths.

Keeping belief and attenuation separate means either can be tuned or replaced
without disturbing the other, which is what the ablation studies in section 50
need.

## `TESTS` and `TESTED_BY`

Section 9 lists no relation for test coverage. Section 65 requires
`test_auth TESTS authenticate`. Section 43 shows `TESTED_BY` in a reasoning path.
The relation is required by the specification's own acceptance scenario and absent
from its relation vocabulary, so it was added to the dependency family with
`INHERITS`-level attenuation and `TESTED_BY` as its inverse.

It is asserted by a different extractor from the `CALLS` fact it accompanies.
Calling is not testing: a test may call a helper, a fixture, or an assertion
utility. So `TESTS` carries `STATIC_ANALYSIS` provenance at confidence 0.9, while
the `CALLS` edge between the same two objects carries `AST` provenance at 1.0.
Both are stored. Relation IDs include the extraction method, so the two coexist
rather than overwrite.
