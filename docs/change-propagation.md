# Change propagation

Spec sections 29 and 30. Development step 17.

## The gap section 30 is pointing at

Impact analysis answers:

> What could break if `decode_claims` changes?

with the change left opaque. Every dependent comes back wearing the same label,
which is the correct answer to a question nobody asks. An agent does not propose
"something about `decode_claims` will be different". It proposes a rename, or a new
parameter, or the same interface computing a different result, and those have
different consequences for different dependents.

Spec section 30 asks for the typed version:

    X_{t+1} = F(X_t, Δ)        estimate  ΔA ⇒ {ΔB, ΔC, ΔD}

`mcm/core/change.py` is the Δ. `mcm/reasoning/change_propagation.py` is F.

## Four change kinds

```
REMOVE      the definition is no longer in the file
RENAME      the definition's name changed, its body did not
SIGNATURE   the parameter list changed
BEHAVIOUR   the body changed, the interface did not
```

Spec Rule 8 forbids a large ontology before the hypothesis is tested, but there is
a sharper constraint. Spec section 61 will compare a *predicted* change against an
*observed* one, so a change kind that cannot be read off a diff by the parser this
system already has can never be checked. Each of these four can be.

"Performance regression", "semantics-preserving refactor" and "stricter validation"
are all things an agent might reasonably want to say, and none is decidable from an
AST, so none is here.

Two derived properties are what the table actually reads:

```
breaks_reference   REMOVE, RENAME              dependents can no longer name it
breaks_call        REMOVE, RENAME, SIGNATURE   existing call sites stop being correct
```

A signature change breaks calls without breaking references. That single
distinction is what the rest of this document is about.

## The propagation table

Which dependents must be edited is a function of the change kind and of the
relation type that reaches them. Both are declared, never inferred from the shape
of the data - the same discipline `RELATION_SPECS` applies to traversal.

| edge | rename or remove | signature | why |
|---|---|---|---|
| `CALLS` | edit | edit | the call site names the callee and passes its arguments |
| `USES` | edit | edit | the use site names the target and may pass arguments |
| `TESTS` | edit | edit | a test names its subject and calls it |
| `INHERITS` | edit | edit | a subclass names its base; an override carries the signature |
| `IMPLEMENTS` | edit | edit | an implementation names its interface and matches its shape |
| `READS` | edit | — | a read names the target but does not call it |
| `IMPORTS` | edit | — | the import names the module, not anything inside it |
| anything else | — | — | not modelled, so nothing is claimed |

The `IMPORTS` row is where this earns its keep:

```bash
mcm propagate jwt --kind rename      # jwt_provider.py:3 must change
mcm propagate jwt --kind signature   # jwt_provider.py is untouched
```

Renaming the module forces an edit at the import statement. Changing a signature
inside it does not, because an import statement says nothing about any signature.
A single-label impact report cannot express that difference: it marks the file
affected either way and leaves the agent to work out whether there is anything to
do.

One invariant holds across the table, and the test suite asserts it: nothing
requires an edit on a signature change without also requiring one on a rename. If
the source calls it, the source names it. A new edge type cannot be added with an
incoherent pair of flags.

## Unmodelled edges claim nothing

A dependency edge with no row gets `DEFAULT_EDGE_RESPONSE`, which requires no edit
for any change kind. The dependent still appears in the report as `MAY_DIFFER`, so
it is never silently dropped, but no claim is made that its source must change.

Claiming `MUST_UPDATE` for an edge whose meaning is not modelled would be asserting
that the dependent's source names the target, which is exactly what is not known.
This is the same stance the constraint engine takes when it refuses to return
`SATISFIED` for something it did not check.

## Edits do not propagate; behaviour does

```
Change: SIGNATURE decode_claims

Must be updated (1) - confidence 1.00 HIGH:
  [FACT ] validate_token  (Function, depth 1, 1.00 HIGH)
          CALLS
          why: the call site names the callee and passes its arguments
          edit: auth.py:12

Behaviour may differ (4) - no source edit:
  [INFER] authenticate  (Function, depth 2, 0.95 HIGH)
          CALLS -> CALLS
          why: reached through 2 dependency edges; its source does not name the change site
  ...
```

`authenticate` calls `validate_token`, which calls `decode_claims`. Adding a
parameter to `decode_claims` forces an edit in `validate_token` and *nowhere else*:
`authenticate`'s source contains no reference to `decode_claims` at all. Its
behaviour may still change, so it is reported, at the attenuated confidence the
dependency algebra gives it, as something to verify rather than as work to do.

Conflating those two is what turns an impact report into a list an agent cannot
act on. Only depth 1 is ever `MUST_UPDATE`, for every change kind.

## Two confidence numbers

`edit_confidence` is the confidence in "these sites must be edited". Every
`MUST_UPDATE` rests on a directly observed reference, so it is usually 1.00.

`confidence` is the weakest prediction anywhere in the report, which a long
behaviour tail will dominate.

They are separate for the reason impact analysis separates `test_confidence`: the
actionable claim should not be dragged down by the speculative one.

## What a prediction is

Every `PredictedChange` carries a `Change` of its own, so `ΔA ⇒ {ΔB, ΔC}` is
literally a change mapping to changes. The predicted kind is always `BEHAVIOUR`:
editing a call site and computing a different result are both changes to a body,
and a dependent's own interface does not change because its dependency did.

The one case this misses is an override that has to be renamed along with the base
method it overrides, which is genuinely a `RENAME` on the subclass member. Catching
it needs member-level modelling the extractor does not have, and it is named here
rather than quietly mispredicted.

Nothing is persisted. A predicted change is an inference about a state that does
not exist yet (spec Rule 2).

## What section 29 required and was missing

Spec section 29 lists the impact return: direct, indirect, tests affected, **APIs
affected**, **configuration affected**, confidence. The last two were absent.
They are now views over the affected set, filtered by object type.

Both are empty on every corpus this system can ingest, because Python ingestion
emits no `API` or `Configuration` objects - naming an API endpoint needs a
framework-route extractor, and configuration needs a config parser. Neither exists.
The views are correct by construction the moment one does, which is the difference
between a projection and a stub, and there is a test that populates them by hand to
prove the filter works.

## What is still absent

Spec section 60's pre-action simulation - actually constructing `K_{t+1}^predicted`
and re-checking constraints, tests and architectural rules against it - is step 18,
not this one. This module estimates consequences; it does not build the successor
state.

Spec section 61's prediction error, comparing `Predicted(Δ)` against `Observed(Δ)`,
needs observations of what a change actually did. That is what the typed `Change`
above exists to make possible, and it is the reason every change kind here is one a
diff can settle.
