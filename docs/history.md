# Git ingestion and historical reasoning

Spec section 19 steps 7 to 9: extract Git history, create commit objects,
associate changes with source objects. Section 66 is what it is for.

## History is the same pipeline, replayed

`HistoryIngestor` walks commits oldest-first and runs the ordinary extraction
pipeline once per revision. There is no second implementation of extraction.

What made that possible is `mcm/ingestion/sources.py`. Ingestion used to walk the
filesystem directly; now it takes a `SourceProvider`, and there are two:
`WorkingTreeProvider` reads from disk, `GitRevisionProvider` reads
`git ls-tree` and `git show <sha>:<path>`. Everything above that layer is unaware
of the difference.

The payoff is that the temporal diffing built for re-ingestion works unchanged.
Each revision is ingested with `moment=commit.date`, so relations open and close
at real historical dates:

```
42967b01 add token creation and validation            +15  -0   5 symbols
92459e70 add login and the user model                 +10  -0   5 symbols
ec74420b add authentication tests                     +13  -0   2 symbols
f9c4cc06 rename claim key returned by decode_claims   +0   -0   1 symbols
```

A `CALLS` edge introduced by the second commit has `valid_from` set to that
commit's date, not to when the ingest ran. Querying the repository as it stood
before that commit does not see it.

## Step 9 by definition text, not by line numbers

Associating a commit with the source objects it changed looks like a
diff-hunk-to-line-span mapping problem. It is not, and treating it as one
produces wrong attributions: hunk line numbers refer to the file at that
revision, so overlapping them with the line spans in the store attributes changes
to whatever symbol now occupies those lines.

Instead, each changed file is parsed at the commit and at its parent, and the
byte span of every definition is hashed. A definition is attributed to the commit
when its digest differs, when it is new, or when it is gone.

This works because identity is already line-independent (spec section 20). A
definition that moved forty lines but did not change hashes the same and is
correctly not attributed. On the fixture, the regression commit is attributed to
exactly one symbol:

```
f9c4cc06  changed: repo://app/jwt_provider.py#function:decode_claims
```

`encode_claims` sits in the same file and is not listed, because its text did not
change.

Nested definitions are attributed to both the member and its container. Editing a
method genuinely does change the text of its class, so a commit touching
`User.is_active` also transforms `User`.

## What Git ingestion writes

| Relation | Meaning |
|---|---|
| `PRECEDES(parent, child)` | commit ordering, transitive per the RelationSpec table |
| `TRANSFORMS(commit, file)` | the commit added, modified, deleted or renamed the file |
| `TRANSFORMS(commit, symbol)` | the definition's text changed at this commit |

All three carry `ExtractionMethod.GIT` provenance and `EvidenceType.COMMIT`
evidence naming the commit and path.

They open at the commit date and never close. A commit changing a definition is a
permanent historical fact once it happens, unlike a call edge, which stops being
true when the call is removed. Leaving `valid_from` unset would make "commit 4
changed `decode_claims`" answer true for moments before commit 4 existed, which a
test caught.

## Section 66: why did this start failing?

```bash
mcm history path/to/repo --db repo.db
mcm why authenticate --db repo.db --since ec74420b
```

`regression_candidates` searches the target's **dependencies**, not its
dependents. If `authenticate` started failing, the cause lies in what it relies
on. A commit touching the tests that call `authenticate` is not a candidate for
`authenticate` breaking, and the test suite asserts that it is excluded.

For each object the target depends on, the commits that transformed it become
candidates, filtered by the time window and ranked most-recent-first with
dependency strength as the tiebreak. Recency leads because a regression is
bounded by when it appeared.

The fixture's regression is planted so that finding it requires the dependency
chain: commit 4 edits `decode_claims`, a name that appears nowhere in `auth.py`.

```
1. f9c4cc06  rename claim key returned by decode_claims
     Fixture Author  2026-03-04
     changed: decode_claims (jwt_provider.py)
     authenticate depends on it: decode_claims <- validate_token <- authenticate
     via CALLS/CALLS, dependency strength 0.95 (HIGH)
     evidence: f9c4cc06:jwt_provider.py [COMMIT] f9c4cc06 changed the definition
               of decode_claims in jwt_provider.py

To settle this, bisect with: test_authenticate_returns_username, ...
```

## What this deliberately does not conclude

Section 37 is explicit:

> Do not infer A → B as causal merely because A → B is a dependency. For causal
> claims require evidence such as: runtime experiment, test result, commit
> history, explicit documentation, human assertion.

Commit history is one item on that list. On its own it supports correlation. So
`causal_reasoning.py` creates no `CAUSES`, `CONTRIBUTES_TO` or `TRIGGERS`
relation, the score is named `correlation` and documented as dependency strength
rather than a probability, and the output says so in its last two lines. A test
asserts the store contains no causal relation after the query runs.

What would license a causal claim is a second, independent source of evidence: a
test that passed at the parent commit and fails at the child. That is why the
report ends by naming the covering tests to bisect with. Running them is the
missing evidence, and running them is not something this phase does.

## Bounds and their honesty

`--max-commits N` selects the most *recent* N commits, then reverses them. So a
window starts partway through history, and its first revision opens every
relation that existed by then rather than at the commit that introduced each one.
Validity dates under a window are upper bounds: no later than, not exactly when.
Without a window they are exact.

A parent commit outside the window is skipped rather than invented, so a
`PRECEDES` edge never points at an object that does not exist.

Replaying history parses every Python file at every commit. For the four-commit
fixture that is negligible; on a large repository it is the honest cost of exact
historical validity, and `--max-commits` is the control.

## Still missing

`observed_at` remains collapsed into `valid_from`. Replaying history sets
`valid_from` from commit dates, which is the important half, but the moment MCM
*learned* a fact is not recorded separately from the moment it *became* true.
Distinguishing them matters once facts arrive from sources that disagree, which
is contradiction detection (spec section 31).

Merge commits are handled as ordinary commits diffed against their first parent.
Changes that arrived only through the second parent are attributed to the merge
rather than to the commit that made them.
