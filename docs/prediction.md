# Prediction

Spec sections 60, 61 and 62. Development step 18.

## Two halves

**Section 60, before the change.** Build `K_t + Δ → K_{t+1}^predicted` and check
dependencies, constraints, tests, APIs and architectural rules against it.

**Sections 61 and 62, after the change.** Compare `Predicted(Δ)` with
`Observed(Δ)`, and update confidence from the error.

```bash
mcm simulate encode_claims --kind remove --db repo.db
mcm accuracy --db repo.db
```

## Simulation reports; it does not decide

Spec section 60 ends with "then let the agent decide whether to proceed", so
`Simulation` has no `should_proceed` field and no `safe` field, and the test suite
asserts their absence. Whether a newly violated architectural constraint is a
blocker or the entire point of the change is not a question the relation graph can
answer.

```
Simulating: REMOVE encode_claims (inline it into create_token)
  K(t+1): 3 relations close

Dependencies (1 require an edit)
  create_token  auth.py:8

Constraints
  [REGRESSION] token_creation_goes_through_provider: SATISFIED -> VIOLATED
      create_token does not depend on any repo://app/jwt_provider.py#*
```

`K_{t+1}` is built by forking the store and *closing* the relations the change
destroys, not deleting them, because closing is what the temporal model does when a
fact stops being true (`docs/temporal-model.md`). The constraint engine then runs
against a real store through its ordinary code path, and cannot accidentally see
the original.

A rename is structural for a reason worth stating: object identity is built from
the qualname (spec section 20), so a renamed definition is a *different object* and
every edge into the old identity stops being true. The new identity has no
dependents yet, which is exactly the state the simulation should show.

## A behaviour change cannot move a constraint

`SIGNATURE` and `BEHAVIOUR` changes close no relation, so `K_{t+1}` is structurally
identical to `K_t` and no predicate over the graph can move. The report says this
rather than printing an empty diff:

```
Constraints
  unchanged, and necessarily so: this change closes no
  relation, so no predicate over the graph can move. Only
  the tests below can catch a behaviour regression.
```

An empty diff and a diff that *cannot* be non-empty look identical and mean
completely different things. The first is an all-clear; the second is a statement
about the limits of static checking.

## Where observations come from

Spec section 61 lists three channels: the actual diff, the actual test result, the
actual runtime behaviour. Only the first exists here.

Git history is already ingested commit by commit, and since step 18 each
`TRANSFORMS` relation records *what kind* of change the commit made to that
definition - `REMOVE`, `SIGNATURE`, `BEHAVIOUR` or `ADDED` - classified by comparing
parameter lists between the two revisions, both of which the ingestion already
parses. So an observation is a commit, and the prediction is made against the
repository as it stood one microsecond before that commit landed.

`RENAME` is absent from the observed kinds. A rename appears as one definition
disappearing and another arriving, because identity is the qualname, and telling
that pair apart from an unrelated delete and add needs body matching this does not
attempt.

## What co-change can and cannot test

If a commit changes A and also changes B, that is evidence B had to be **edited**.
It is not evidence that B's *behaviour* changed, and the absence of an edit to B is
not evidence that B was unaffected - B may have kept working, or the work may have
been deferred.

That maps onto the two consequences from section 30 very unevenly:

| prediction | what it claims | can a diff test it? |
|---|---|---|
| `MUST_UPDATE` | this source must be edited | yes - a commit records edits |
| `MAY_DIFFER` | this behaviour may change | no - needs test results |

So the report scores them separately and refuses to combine them. A `MAY_DIFFER`
"false alarm" is not evidence the prediction was wrong; it is evidence that this
observation channel cannot see the thing being predicted. Folding both into one
accuracy figure would produce something that looks like a measurement and is not
one.

When a corpus contains no testable prediction at all, no number is printed:

```
Every observed change is a behaviour change with no co-edits, so no edit
prediction can be tested. A diff records edits; it cannot confirm or deny a
behaviour change.
```

That is the case for `examples/history_fixture.py`, whose only change to an
existing definition is a behaviour change. `examples/prediction_fixture.py` exists
because an evaluation needs a commit that changes a signature *and* fixes its call
site, which is the case a diff can settle.

## Co-change is symmetric; propagation is not

This is the trap in using co-change as ground truth.

Commit 3 of the prediction fixture changes `area`'s signature and updates
`summarise`, which calls it. That is **two** observations:

- target `area`: co-changed `{summarise}`. Predicted, and confirmed. Correct.
- target `summarise`: co-changed `{area}`. `area` is *upstream*, and no downstream
  traversal can reach it. Impossible to predict, by construction.

Naive recall scores the second as a failure, and reports 0.50 for a model that did
nothing wrong. So misses are split:

```
Edits alongside that nothing reached
  1 upstream of the change: co-change is symmetric, propagation is not, so these
    are not errors
```

There is a stronger statement available. Every miss is *by construction* an object
with no dependency path from the change site, because anything the traversal
reaches is predicted as something. So misses never measure whether the propagation
table labelled things correctly - they measure graph coverage, and have two causes:
the upstream artefact above, or an edge the extractor never saw. Those need
different fixes, so the report separates them.

## Measured against real history

Everything above is reasoning about the fixture. This is the first measurement on
real repositories, produced by `scripts/prediction_suite.py`, which replays
history into a store and scores what propagation would have predicted against
what the commits actually did.

Five repositories, 2403 observations of changes to definitions that already
existed:

| repository | observations | behaviour | testable | predicted | confirmed | precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| click | 951 | 536 | 415 | 9 | 9 | **1.00** |
| flask | 542 | 395 | 147 | 10 | 10 | **1.00** |
| requests | 342 | 307 | 35 | 11 | 11 | **1.00** |
| python-dotenv | 407 | 330 | 77 | 18 | 12 | 0.67 |
| itsdangerous | 161 | 150 | 11 | 0 | 0 | n/a |
| **pooled** | **2403** | **1718** | **685** | **48** | **42** | **0.88** |

**When the model says an edit is required, one was made 88% of the time.** Three
of the five repositories are at 1.00. All 48 predictions came through `CALLS`
edges.

### The two numbers that qualify it

**The model fires rarely.** 48 edit predictions across 685 testable observations,
about 7%. It is not answering "what else changed" for most commits; it is
answering "I can justify that this specific caller must change" and otherwise
staying quiet. High precision with a low firing rate is a different product from
high recall, and it is worth being explicit about which one this is.

**Most real changes are not the kind this can check.** 71% of changes to existing
definitions are behaviour changes, which propagate as `MAY_DIFFER`. A commit
records edits, not behaviour, so those predictions are neither confirmed nor
refuted *by this ground truth*. They are not unfalsifiable in principle: a test
run would settle them, which is what the observation channel in section 62
exists for and what has not been run at scale.

### Why recall is not the headline

Commits in this sample touch a mean of 62 definitions. Most of that is unrelated
work batched into the same commit, so recall against co-change would mostly
measure commit hygiene. The section above already explains the structural half of
this: co-change is symmetric and propagation is not, so an edit upstream of the
change site can never be reached by any downstream traversal and counting it as a
miss scores the metric rather than the model.

### What this does not show

One horizon per repository, all Python, all under 250 commits of replayed
history, and 48 predictions is a small number to rest a precision figure on. The
`python-dotenv` row at 0.67 is the only one below 1.00 and it is also the one with
the most predictions, which is the pattern you would expect if the others are
simply under-sampled rather than better.

Replaying history costs one full ingest per commit, so larger repositories were
not run. That is the same cost constraint as the retrieval evaluation and it has
the same fix: patience, or a machine with more memory.

## What section 62 updates

Spec section 62 says to reduce confidence in `A → B` when changing A does not
affect B. In this system that is not one number:

- `relation.confidence` is the belief that the edge is **real**. An AST-observed
  call is certainly a call, and no amount of co-change data makes it less so.
- `EDGE_DECAY` is how much of a change **survives crossing** an edge of that type.

The evidence bears on the second. `confidence.py` already separates belief from
attenuation and says why; this module learns the attenuation and never touches the
belief. The test suite asserts that relation confidences are untouched after an
evaluation.

The update is the transparent heuristic section 62 asks for:

```
Confidence_{t+1} = (confirmed + k * prior) / (predicted + k)
```

a Beta-Binomial posterior mean, with the hand-set `EDGE_DECAY` value as the prior
and `PRIOR_STRENGTH = 5` as its weight in observations. Two properties make it
honest to publish: with no observations the posterior *is* the prior, and a single
commit cannot overturn a constant that every impact answer in the system depends
on.

```
  edge             prior  posterior    n
  CALLS             0.95       0.96    1
```

**Nothing is applied.** The learned table is printed beside the hand-set one so the
guesses in `confidence.py` can be checked against a real repository. Silently
retuning them from a handful of commits would make the benchmark unfalsifiable.

## What one repository still cannot tell you

`mcm accuracy` ends every single-repository run by saying so:

```
Sample size: 3. Nothing here is a result.
A repository this size cannot separate a good propagation model from a lucky one;
the evaluation framework in sections 45 to 50 is what would. Nothing was applied.
```

That warning still stands and the tool still prints it. The pooled measurement
above exists precisely because of it: the quantity that can be checked is rare,
so a single repository yields a handful of testable cases and `itsdangerous`
yields none at all. Five repositories pooled get to 48 predictions, which is
enough to report a precision figure and still small enough that the figure should
be read with the sample size attached.

## What is still absent

Two of section 61's three observation channels. There is no test runner, so a
`MAY_DIFFER` prediction cannot be confirmed or refuted by anything; and no runtime
tracing, so nothing observes behaviour directly. Adding a test-result channel is
the single change that would make the larger half of the predictions measurable.

Section 59's loop still stops at the LLM. Simulation is the stage before it and
accuracy is the stage after; proposing and executing the change in between is not
here.
