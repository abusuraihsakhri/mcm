# Evidence, provenance and the fact/inference split

## Rule 4 as a structural property

Section 53 Rule 4 requires every important relation to have provenance. Making
that a convention would mean it holds until someone forgets. Instead
`assert_relation` in `mcm/ingestion/dependencies.py` is the only convenient way to
build a relation, and it returns a relation, its evidence and its provenance
together. A relation without provenance requires deliberately constructing
`MCMRelation` by hand.

`tests/test_ingestion.py` asserts the property directly: every relation in the
store has a non-empty `provenance_id`, a non-empty `evidence_ids`, and a
provenance record that actually resolves.

## What evidence looks like

```
CALLS(authenticate, validate_token)
  provenance: method=AST, agent=mcm.ingestion, source_ref=auth.py:18
  evidence:   AST, auth.py:18, "authenticate calls validate_token"
  confidence: 1.0
```

`source_ref` is a location a person can open. The impact explanation prints it for
every step of every reasoning path, which is what section 43 asks for when it says
the agent must never simply say "I know this."

## Two extractors, two confidences

The same pair of objects can carry claims from different extractors. In the
fixture:

```
CALLS(test_authenticate_returns_username, authenticate)
  method=AST, confidence=1.0
  evidence: AST, tests/test_auth.py:6

TESTS(test_authenticate_returns_username, authenticate)
  method=STATIC_ANALYSIS, confidence=0.9
  evidence: TEST, tests/test_auth.py:6,
            "test function ... invokes authenticate; treated as test coverage"
```

The first is what the syntax tree says. The second is an interpretation of it, and
a fallible one: a test calling a helper is not testing the helper. Both are stored
because both are true statements about different things, and relation IDs include
the extraction method so they do not collide.

Note that both are *asserted*, not derived. Each is read from an artifact by a
named extractor. Neither is composed from other relations. A weaker confidence
does not make something an inference.

## Fact against inference

Section 54 calls this distinction fundamental, and section 53 Rule 2 forbids
promoting an inference to a fact automatically. Three mechanisms enforce it.

**At the model level**, `MCMRelation.inference` is `None` on a fact and populated
on a derivation, and `is_derived` reads it.

**At the storage level**, reads exclude derived relations by default. Asking for
an inference is explicit.

**At the reasoning level**, `analyse_impact` returns its derived
`POSSIBLY_AFFECTS` relations to the caller and does not write them to the store.
Persisting an inference is a separate decision that nothing in V1 makes
automatically. `tests/test_impact.py` asserts that the derived relation IDs are
disjoint from what is stored.

## Where the line falls in an impact report

Depth 1 is a fact. Something that directly calls or uses the change site was
observed to do so, and the report prints it as `[FACT ]` with the confidence of
its asserted relation.

Depth 2 and beyond is an inference. It is composed from a chain of observations,
carries a `POSSIBLY_AFFECTS` relation naming the rule and the premises, prints as
`[INFER]`, and is scored by the attenuation model in `docs/relation-algebra.md`.

## Unresolved names are recorded, not guessed

The ingestion report lists every name the extractor could not resolve:

```
auth.py:13 attribute call claims.get (base claims needs type inference)
auth.py:23 attribute call user.is_active (base user needs type inference)
```

These appear in the report and produce no relation. An invented edge would carry
AST provenance pointing at a real line of source, which makes it
indistinguishable from a correct one on inspection. The cost of a missing edge is
a gap. The cost of a wrong edge is a gap that lies about itself.
