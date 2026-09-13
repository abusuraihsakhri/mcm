# Temporal model

## The requirement

Spec section 18: facts change, every relation can have `valid_from`, `valid_until`
and `observed_at`, and

> Never simply delete historical knowledge.

with the worked example of `DEPENDS(A,B)` at t1 and `¬DEPENDS(A,B)` at t2 after a
refactor.

V1 had the fields and honoured them on read, but nothing ever wrote a
`valid_until`, so the model was present and inert. What makes it live is
re-ingestion diffing.

## Every read is a read at a moment

`Store.relations_for` and `Store.all_relations` take `as_of`, defaulting to now:

```sql
AND (valid_from  IS NULL OR valid_from  <= ?)
AND (valid_until IS NULL OR valid_until >  ?)
```

`include_historical=True` ignores validity and returns every interval on record.
`as_of` threads through `GraphProjection.neighbours`, both closure directions,
`analyse_impact`, `run_query` and the CLI's `--as-of`, so a whole impact analysis
can be run against a past state.

Timestamps are stored as ISO-8601 strings and compared lexicographically. That is
correct only because every timestamp is produced by `utcnow()` and therefore
carries the same `+00:00` offset. A Postgres backend would use `timestamptz` and
not depend on the property.

## Re-ingestion diffing

`RepositoryIngestor.ingest` reads the currently-valid relations for the
repository *before* writing anything, then compares:

```
opened     = extracted - previously_valid    valid_from = now
unchanged  = extracted & previously_valid    untouched
closed     = previously_valid - extracted    valid_until = now
```

The order matters. Extracted relations are written first, then absent ones are
closed. A relation that was closed and is observed again is reopened by
`put_relation`, whose upsert writes `valid_until` back to NULL.

Ownership is decided on the first argument, which for every extracted relation is
an object inside the repository. The boundary check stops repository `app` from
claiming relations belonging to `app_extras`.

## What it looks like

```
$ mcm ingest examples/app --db repo.db
  38 relations ...
  0 unchanged, 38 opened, 0 closed

# authenticate stops calling create_token

$ mcm ingest examples/app --db repo.db
  37 unchanged, 0 opened, 1 closed
    closed: CALLS repo://app/auth.py#function:authenticate
                -> repo://app/auth.py#function:create_token
```

and the same question then gives two answers:

```python
CREATE_TOKEN in dependencies_of(store, AUTHENTICATE).paths                  # False
CREATE_TOKEN in dependencies_of(store, AUTHENTICATE, as_of=before).paths    # True
```

with 38 rows still on record and 37 currently valid. Nothing was deleted.

## One interval per relation

A relation record holds a single validity interval. A relation that is removed
and later restored has its `valid_until` cleared rather than gaining a second
interval, so the gap between the two observations is not represented.

This is a real limitation, and the cause is the identity scheme: relation IDs are
content-addressed over type, arguments and extraction method, so the restored
relation hashes to the same ID as the original. Modelling disjoint intervals needs
a row per interval and an identity scheme that tolerates more than one row per
claim.

Section 18's stated requirement is met. Full bitemporal history is not, and the
distinction is worth keeping straight: this system can tell you what it believed
at a moment, but not how many times a belief flipped.

## What still writes nothing

`observed_at` is not modelled separately from `valid_from`. For AST extraction the
two coincide, since the observation is the fact. They diverge once Git ingestion
lands: a commit from three months ago is observed now but was valid then, and
distinguishing those is what section 66's "why did authentication start failing
after commit X" needs.

`SUPERSEDES` has a RelationSpec and no producer. Contradiction detection (section
31) is where it earns one.
