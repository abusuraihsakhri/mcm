# Agent loop

Spec sections 33, 34 and the first half of 59. Development step 16.

## What is implemented

Spec section 59 gives thirteen stages. Step 16 is the ones before the model runs:

```
USER TASK            mcm context "<task>"
  ↓
PARSE TASK           hybrid retrieval over the task text  (no LLM, no intent model)
  ↓
IDENTIFY OBJECTS     the focus: best hit that has dependency edges
  ↓
QUERY MCM            impact + dependency closure from the focus
  ↓
RETRIEVE EVIDENCE    the evidence each relation on each path cites
  ↓
SYMBOLIC REASONING   the closure is the reasoning; every hop is licensed by the algebra
  ↓
CHECK CONSTRAINTS    scoped to the objects in the context
  ↓
BUILD MINIMAL CONTEXT  spec section 34, verified by replay
  ↓
LLM                  ← not here. This is where the package is handed over.
```

Everything after `LLM` - propose, simulate, execute, observe, update - is steps 17
and 18 and is absent. Spec section 60's pre-action simulation needs a change
format; spec section 61's prediction error needs observations. Neither exists yet,
so neither is stubbed.

## Every item is a claim with a justification

The package is not a pile of retrieved material. It is a set of statements, each
carrying the objects, relations and evidence that produced it:

```
[INFER] test_login_rejects_inactive_user is exercised by authenticate  (0.95 HIGH)
        via authenticate -> login -> test_login_rejects_inactive_user
        evidence: ev:0af32191d2ab7140, ev:f018f546e88a8f87
```

Five claim kinds, kept apart because their warrants differ: `impact`, `dependency`,
`test`, `constraint` and `change`. Each is labelled `FACT` or `INFER` - an observed
relation and a conclusion drawn over a chain of them are different sorts of thing
(spec section 54) - and carries a confidence band (spec section 44).

A context item that supports no claim is not context, it is padding. That is the
whole premise of the minimiser below.

## Picking the focus

Relevance alone picks the wrong object. `"authentication started failing"` retrieves
the commit whose message says *authentication*, and a commit participates in
`TRANSFORMS` and `PRECEDES`, not in dependency relations - so a package focused on
one contains no impact, no dependencies and nothing to check.

The rule is structural rather than a list of favoured types: the focus is the
highest-ranked hit that takes part in at least one dependency relation. The commit
stays in the package as a recent change, which is the role it can fill.

When nothing but the vector channel supported the focus - no lexical match, no
exact resolution - the package says so:

```
nothing in the repository lexically matches this task; the focus 'load_user' was
chosen by similarity alone and may be the wrong object - name one with --focus
```

A score threshold would have been a constant tuned on the handful of queries at
hand, which is what this project refuses to do to the retrieval weights. The
structural question needs no constant.

## Context minimisation

Spec section 34:

    C* = argmin |C|  subject to  C ⊨ Q

### argmin |C|

Greedy ablation. Every item is removed in turn and the removal is kept if the
package still answers. Greedy, not optimal: minimum set cover is NP-hard, and a
prototype claiming the true minimum would be claiming something it did not
compute. The search is bounded and reports when it hit the bound.

### C ⊨ Q

A store is built containing *only* the reduced context, the query is re-run against
it, and every surviving claim has to come back at the same confidence. Arguing
sufficiency from the construction - "each claim's justification is present,
therefore it holds" - would be circular, because the justification is exactly what
the construction chose to keep.

Entailment is three clauses, and each one exists because something slips past the
others:

**Reproduction.** Impact, test and dependency claims are recomputed on the reduced
store and compared. Drop a relation the answer needs and the claim either
disappears or arrives by a longer path at a lower confidence.

**Presence.** Every object a surviving claim names must be present. Replay cannot
see this: traversal runs on the relation argument index and never loads an
`MCMObject`, so deleting `validate_token` leaves the closure walking straight
through the gap at an unchanged confidence while the claim names something that is
gone. Spec section 43 renders a reasoning path as names, and a context that names
what it does not contain has produced a dangling pointer, not an answer.

**Traceability.** Every retained relation keeps at least one of its evidence
records. Impact analysis never reads evidence, so without this clause ablation
strips all of it and the package still "answers" the query while being unable to
show a single source.

Constraint violations and commit correlations are checked by presence rather than
replay, because neither is a dependency walk from the focus and replay cannot
regenerate them. That is a weaker check, and it is named as one.

### When nothing is removable

On `examples/app` the ablation tries every item and removes none, because every
relation there cites exactly one evidence record and there is no redundancy to
find. That is reported as `already_minimal`, and it is a result rather than a
failure: the justification closure was *proved* minimal under the entailment test
instead of assumed to be. Give a relation a second evidence record and the search
removes it.

## What the numbers mean

```
30 gathered -> 25 justified -> 25 retained (efficiency 0.83, 24 ablations, verified)
```

`gathered` counts everything the pipeline surfaced, including retrieval hits no
claim ever used. `justified` is the closure of the surviving claims. `retained` is
what is left after ablation. The reduction from 30 to 25 is the part minimisation
gets for free by construction: five retrieved objects that nothing rested on.

`efficiency` is `retained / gathered`. Spec section 48 defines context efficiency
as useful over total retrieved information, and this computes it the only way this
phase can: *useful* means "appears in the justification of a claim that survives
replay". Section 48's metric is about task outcomes and needs the evaluation
framework of sections 45 to 50, which does not exist. **This number measures the
reduction, not the benefit.** It says how much of what was gathered was justified.
It says nothing about whether the agent then does better work, and it must not be
quoted as though it did.

## Candidate actions are derived, not imagined

Spec section 33 asks for `candidate_actions`. With no LLM in this step, the only
honest source is the model itself, so each action comes from a claim:

- covering tests to run, from the test claims
- constraints to resolve or waive, from the violations
- commits to bisect, from the change claims

Spec section 37's restraint carries through. A change claim says a commit touched
something the focus depends on, scores it `correlation`, and the action it
generates ends by pointing at the tests - because running them is the evidence this
phase cannot gather.

## Uncertainties

The section that keeps the rest honest. An agent handed only what MCM is sure of
would not know what it is missing, so the package enumerates:

- constraints that could not be decided, and what they would need
- how many claims are inferences rather than observed relations
- claims in the LOW band
- absent Git history, or history that touched nothing relevant
- absent decisions: ingestion creates no `Decision` objects, so the rationale
  behind the code is not in the package and never was
- a depth bound that was hit, so the picture is incomplete rather than empty
- a focus chosen by similarity alone

## Usage

```bash
mcm context "Fix authentication failure" --db repo.db
mcm context "Fix authentication failure" --focus authenticate --db repo.db
mcm context "replace the JWT library" --floor 0.9 --db repo.db   # acceptable confidence
mcm context "replace the JWT library" --full --db repo.db        # skip minimisation
mcm context "replace the JWT library" --json --db repo.db
```

`--floor` is spec section 34's "acceptable confidence". It defaults to 0, so
nothing is dropped silently; raising it trades completeness for certainty, and the
package records that claims were dropped.
