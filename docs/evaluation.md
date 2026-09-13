# Evaluation

Spec sections 45 to 51, development step 20. This is the component that tests the
section 3 hypothesis, and the only one whose output is a claim about the rest of
the system.

Section 3 states the hypothesis and then constrains how it may be tested:

```
H1: MCM + hybrid retrieval + inference > Graph memory > Vector RAG

Do NOT assume this hierarchy is true.
```

The experiment must be able to produce `MCM > Graph`, `MCM ≈ Graph` or
`MCM < Graph` depending on the evidence. Everything below follows from taking that
literally.

## Definitions

Let `H` be the horizon commit, `C` a commit after it, and `D(C)` the set of
definitions `C` changed. A system indexes the repository at `H` and answers a
query drawn from `C`.

**Reachable truth.** Only definitions that existed at the horizon can be
retrieved:

```
T(C) = D(C) ∩ Defs(H)
ceiling(C) = |T(C)| / |D(C)|
```

Scoring uses `T(C)`. `ceiling(C)` is reported per task, because a commit that
mostly adds new code has an answer no system could reach, and averaging that into
recall would measure repository growth.

**Retrieval**, for a ranked answer `R` and cutoff `K`:

```
Recall@K    = |R[1..K] ∩ T| / |T|
Precision@K = |R[1..K] ∩ T| / |R[1..K]|
MRR         = 1 / min{ i : R[i] ∈ T }
```

**Context efficiency** (section 48), the headline metric:

```
ContextEfficiency = UsefulTokens / TotalTokens

TotalTokens  = Σ tokens(u) for every unit u delivered
UsefulTokens = Σ tokens(t) for each distinct t ∈ T delivered
```

Both numerators and denominators are in tokens, not items, and `UsefulTokens`
counts the size of the *target*, not the size of the container that delivered it.
A 900-token file delivering a 40-token function contributes 900 to the denominator
and 40 to the numerator. This is the reading section 48 argues for when it says a
system retrieving 1000 relevant-looking chunks is not necessarily better than one
retrieving 30 explanatory facts. It makes absolute efficiency low for everyone;
the claim is carried by the ratio between systems.

An empty answer scores 0.0, not 1.0. Retrieving nothing is a failure to answer,
not perfect efficiency.

**Comparison.** Systems answer identical tasks, so differences are paired. The
test is a paired bootstrap over tasks: resample the per-task difference with
replacement 2000 times and report the 95% interval. The seed is fixed, so any
reported interval reproduces exactly. An interval excluding zero is reported as
`separated`; this is not called significance and no p-value is computed.

A difference must also be large enough to mean something. With 149 paired tasks a
gap in the fifth decimal separates cleanly, so a second bar applies: the
difference must exceed one percent of the means being compared.

```
separated  = interval excludes 0
negligible = |difference| < 0.01 * max(|left mean|, |right mean|)

verdict = greater than    if separated and not negligible and difference > 0
          less than       if separated and not negligible and difference < 0
          approximately equal to   otherwise
```

Both thresholds are fixed constants in `mcm/benchmark/metrics.py`. Neither was
chosen after seeing a result, and the negligibility rule was added because it
caught a false positive in this report's own ablations, not because it improved
one.

## Where ground truth comes from

Held-out Git history, and nothing else.

A repository's history is cut at a horizon commit `H`. Every system indexes the
repository exactly as it stood at `H`, read out of Git blobs rather than the
working tree. Tasks come from commits after `H`, and the answer to "what should
change to do X" is what the commit that did X actually changed.

No system can retrieve the answer, because the answer had not been written at the
revision it indexed. Nobody labelled anything, so nobody's judgement about what
MCM ought to find is in the ground truth. Re-running on a different horizon is one
argument.

`tests/test_benchmark.py::TestNoLeak` asserts the property directly: a definition
added after the horizon is absent from the snapshot, and every task's targets are
a subset of what existed at the horizon.

### Filters, and their cost

Commits are rejected before any system runs, on rules fixed in advance:

| Rule | Why |
| --- | --- |
| merge or root commit | a merge diff is the union of other work and has no single intent |
| more than 8 files | a repository-wide rename is not a retrieval question |
| subject under 12 characters | not a query |
| boilerplate subject | "bump version to 2.1.4" is not a question about the code |
| tooling or dependency chore | "fix mypy findings" edits real code but the subject names the tool that complained, not anything the program does |
| no definitions changed | nothing to retrieve |
| no changed definition existed at the horizon | ceiling would be 0 |

Every rejection is counted and printed. On flask, 400 commits considered yielded
52 usable ones; the gap is visible in the report rather than implied by it.

The chore filter is the one that deserves scrutiny, because it was added after
seeing that queries like "fix mypy/flake8 findings" defeat every system. It is
defined on the query text alone and applies to all systems equally. A filter
defined on *which commits the systems answered badly* would be cherry-picking, and
section 49 rules that out.

### Families covered, and families not

Section 46 lists nine task classes. Four have ground truth in a diff:

| Family | Question | Truth | Granularity |
| --- | --- | --- | --- |
| A retrieval | the commit subject | definitions the commit changed | definition |
| C impact | "what could break if X changes" | the other definitions the commit changed | definition |
| H planning | the commit subject | files the commit changed | file |
| I debugging | the subject of a fix commit | definitions the commit changed | definition |

Five are absent, and the reason is the same in each case: a diff does not contain
the answer.

- **B dependency** ("what depends on X") has truth in the import and call graph,
  not in history. Computing it from an AST pass would score MCM near-perfectly by
  construction, because MCM's graph is built from that same analysis. The result
  would be a restatement of the extractor, not a finding.
- **E constraint** needs declared architectural requirements, which these
  repositories do not carry.
- **F causal** needs labelled errors with known causes.
- **G equivalence** needs pairs known to be behaviourally identical. Refactoring
  commits are a candidate source and were not pursued.
- **D historical** ("why was X changed") is answerable from MCM's commit
  ingestion, but baselines A and B have no temporal representation at all and
  would score zero structurally. That is a real architectural difference and a
  foregone conclusion, so reporting it as a result would be misleading.

## The systems

Section 45 asks for three baselines and an optional fourth. They are **independent
implementations**, not MCM with retrieval channels zeroed.

This is the decision the whole study rests on. A baseline assembled from
`RetrievalWeights(vector=1.0, ...)` would inherit MCM's symbol-aware document
construction, its route map from relations and evidence back to objects, and its
exact-resolution channel. It would lose to MCM by construction, and section 3's
requirement that `MCM < Graph` be reachable would be unmeetable. Ablations of that
kind are a different question, asked under section 50 and labelled as ablations.

| System | What it is |
| --- | --- |
| A `vector-rag` | overlapping 40-line windows, embedded whole, cosine nearest-neighbour. No symbol table, no graph. |
| B `graph` | definitions as nodes, calls and co-location as edges, name-match seeding, breadth-first spread with decay. No embeddings. |
| D `hybrid` | reciprocal rank fusion of A and B. RRF rather than a weighted sum, because the two rankings are not on a common scale and normalising would add a parameter with nothing to tune it against. |
| C `mcm` | the real pipeline: ingest at the horizon, both text projections, `HybridRetriever` over four channels. The impact family is answered by dependency propagation from the named symbol. |

What the baselines share with MCM is deliberate: the **same corpus** and the
**same embedding provider**. A different corpus makes the comparison meaningless,
and a weaker embedding model for Baseline A would rig the headline result in the
crudest available way. What differs is that A chunks raw text on line windows the
way a RAG pipeline does, instead of embedding symbol documents.

### The budget

Every system fills the same token budget, 4000 by default, and spends it however
its architecture prefers. A budget in tokens is what an agent actually has, and it
is the only framing under which section 48's "30 explanatory facts" and "1000
relevant-looking chunks" are comparable.

### Systems do not grade themselves

`answer` returns what the system would put in front of an agent, in rank order,
with each unit's token cost and the definitions it contains. Whether any of it was
useful is decided by the harness against truth the system never sees.

### The coverage floor

A chunk retriever hands over line ranges, so the harness must decide when a range
counts as having delivered a definition. The rule is that at least half the
definition's lines must be inside the window.

Crediting any overlap at all is the tempting rule and it is wrong: a window
clipping the last three lines of a forty-line function would be credited with the
whole function, and since useful tokens are counted at the definition's full size,
chunk retrieval would collect credit for code it never showed.

This rule is not neutral, and it is exposed as `--coverage-floor` because it moved
a result. On itsdangerous:

| floor | vector-rag efficiency | mcm efficiency | graph vs vector-rag |
| --- | --- | --- | --- |
| 0.0 | 0.355 | 0.240 | less than |
| 0.5 | 0.130 | 0.240 | approximately equal |
| 0.9 | 0.097 | 0.240 | approximately equal |

MCM and the graph baseline deliver whole definitions, so their numbers do not move
at all; the floor constrains only the system that can deliver fragments. The
`MCM > graph` link is stable across the whole sweep. The `graph > vector-rag` link
is not, and it fails at every setting.

## Results with default hyperparameters

These are the untuned numbers. The tuned, and more important, ones are further
down under "The headline result changes"; both are reported because the difference
between them is itself a finding.

Three repositories are used. Two are evaluated and never fitted on; the third is
fitted on and never evaluated. Large and very large from section 49 were not run,
and the reason is cost rather than principle: ingestion runs about 0.4s per file,
and the commit-by-commit path multiplies that by commit count.

| repo | role | Python files | horizon | tasks | scores |
| --- | --- | --- | --- | --- | --- |
| itsdangerous | evaluation | 16 | 250 back from HEAD | 34 | 136 |
| flask | evaluation | 83 | 400 back from HEAD | 149 | 596 |
| httpx | tuning only | 61 | 400 back from HEAD | 237 | n/a |

### itsdangerous

| system | recall | precision | MRR | efficiency | answered | tokens |
| --- | --- | --- | --- | --- | --- | --- |
| vector-rag | 0.363 | 0.179 | 0.325 | 0.130 | 1.00 | 3965 |
| graph | 0.243 | 0.114 | 0.288 | 0.114 | 0.65 | 2494 |
| hybrid | 0.465 | 0.201 | 0.395 | 0.127 | 1.00 | 3997 |
| **mcm** | **0.518** | **0.261** | **0.625** | **0.240** | 1.00 | 3885 |

### flask

| system | recall | precision | MRR | efficiency | answered | tokens |
| --- | --- | --- | --- | --- | --- | --- |
| vector-rag | 0.348 | 0.092 | 0.278 | 0.057 | 1.00 | 3997 |
| graph | 0.198 | 0.077 | 0.179 | 0.034 | 1.00 | 3998 |
| hybrid | 0.326 | 0.094 | 0.273 | 0.046 | 1.00 | 3998 |
| **mcm** | **0.371** | **0.114** | **0.324** | **0.069** | 0.99 | 3774 |

### The hypothesis

On context efficiency, paired bootstrap, 95% interval:

| repo | link | verdict | difference |
| --- | --- | --- | --- |
| itsdangerous | mcm > graph | greater than | +0.126 [+0.026, +0.221] |
| itsdangerous | graph > vector-rag | **approximately equal** | −0.016 [−0.071, +0.048] |
| flask | mcm > graph | greater than | +0.035 [+0.019, +0.051] |
| flask | graph > vector-rag | **less than** | −0.023 [−0.036, −0.010] |

**H1 is not supported on either repository.** The first link holds on both. The
second fails on both: graph memory does not beat vector RAG on these tasks, and on
flask it loses with a separated interval.

On context efficiency, MCM against each baseline directly on itsdangerous is
`greater than` in all three cases: vs vector-rag +0.110, vs graph +0.126, vs
hybrid +0.113, all separated.

That does not hold on every metric, and the difference is worth stating rather
than leaving to whoever re-runs it. On **recall**, on the same repository, MCM
beats vector-rag (+0.155) and graph (+0.275) with separated intervals but is
**approximately equal to hybrid** (+0.054, interval spanning zero). The hybrid
baseline is the hardest of the three to beat, and on recall this task set cannot
separate it from MCM. The efficiency margin over hybrid is real; the recall margin
over hybrid is not established.

So the ordering that the evidence supports is:

```
MCM > { Vector RAG ≈ Hybrid ≈ Graph }
```

rather than section 3's chain. The part of the hypothesis that survives is that
MCM is ahead. The part that fails is the claim that graph memory is the
second-best architecture; on this evidence it is the weakest of the three
baselines, and its advantage is confined to the impact family, where structure is
the question being asked.

### What these numbers do not show

One horizon per repository, two repositories, and both from the same maintainer's
Python ecosystem. Absolute recall is low for every system, which is expected when
the query is a commit subject and the answer is a definition set, but it means
these tasks are hard rather than that any system is broken. The debugging family
is under-powered on both repositories (n=1 and n=7) and its rows are printed with
a warning attached rather than treated as results.

The efficiency metric is sensitive to the coverage floor, as the sweep above
shows. The recall and MRR columns are not.

## Tuning on a third repository

The numbers above were produced with **unfitted** hyperparameters: section 26
leaves every weight configurable and declines to choose, so the defaults were
guesses. That left the ablations unable to distinguish "this component does not
help" from "the default weights spend score on it badly".

A third repository, **httpx** (`encode/httpx`, 61 files, 237 tasks), was added to
settle it. It is fitted on and never reported on. itsdangerous and flask are never
fitted on, so every evaluation number in this document stays a held-out score.

httpx is also outside the `pallets` ecosystem that both evaluation repositories
belong to, which removes one confound from the tuning at least; the two evaluation
repositories still share a maintainer, and that limitation stands.

### Every system is tuned, not just MCM

Fitting MCM's weights while leaving the baselines at whatever constants were typed
first would turn the comparison into tuned-versus-untuned. That is the same class
of error as building the baselines out of MCM's own retriever, and it is avoided
the same way: on the same tasks, against the same objective, by the same search,

| system | fitted |
| --- | --- |
| A vector-rag | chunk window and stride, over 6 geometries |
| B graph | traversal decay and depth, over 9 combinations |
| D hybrid | RRF constant `k`, over 5 values, on top of the tuned A and B |
| C mcm | the five section 26 channel weights, plus graph decay and depth |

The hybrid is given the *tuned* vector and graph baselines rather than the
defaults, because the fair version of "graph plus vector" is the best graph plus
the best vector.

### Why the search is affordable

Section 26's model is linear in the channels:

```
Score(x|q) = w_v*V + w_l*L + w_g*G + w_s*S + w_p*P
```

The channel values do not depend on the weights, so retrieval runs **once** per
task and any number of weight vectors are scored against the captured values
offline. Only `graph_decay` and `graph_depth` change the channel values
themselves, so those form a 3x3 outer loop that does pay for re-retrieval, with a
156-vector sweep inside each step.

The search is random over the 5-simplex rather than a grid: five continuous
dimensions make a grid of useful resolution unaffordable, and random search covers
a simplex better at equal cost. The current defaults and all five single-channel
corners are included explicitly, so the report can say whether tuning beat the
defaults or merely matched them, and so a degenerate optimum cannot be missed.

`TestTuning::test_reweighting_captured_channels_matches_the_real_system` asserts
that the offline re-scoring reproduces what the real system returns. Without that,
the search would be optimising something the evaluation never runs.

### Running it

```
mcm benchmark ../httpx --horizon 400 --tune --json tuned.json
mcm benchmark ../flask --horizon 400 --config tuned.json
```

The configuration fitted for this document is checked in at
`examples/tuned-on-httpx.json`, so the held-out runs reproduce without repeating
the search. It is **not** the shipped default: the defaults stay in
`RetrievalWeights` because, as the next section shows, the fitted values do not
beat them on held-out data.

### What the search chose

| system | default | fitted on httpx |
| --- | --- | --- |
| mcm weights | v .25 l .25 g .25 s .15 p .10, decay .5 depth 2 | **v .41 l .07 g .28 s .04 p .20, decay .7 depth 3** |
| vector-rag | chunk 40, stride 30 | chunk 80, stride 60 |
| graph | decay .5, depth 2 | decay .3, depth 1 |
| hybrid | k 60 | k 10 |

MCM's fitted weights are not a degenerate corner: the top five trials all sit in
the same basin, with the vector channel largest, the graph channel second, the
symbolic and lexical channels near zero, and provenance carrying a surprising
0.20. Every system improved on the tuning repository:

| system | httpx default | httpx tuned | gain |
| --- | --- | --- | --- |
| vector-rag | 0.0585 | 0.0776 | +0.0190 |
| graph | 0.0850 | 0.0994 | +0.0144 |
| hybrid | 0.0749 | 0.0926 | +0.0177 |
| **mcm** | 0.0994 | **0.1457** | **+0.0463** |

Those are training scores, and MCM's is the largest of them. The held-out numbers
are what decides whether any of it is real.

### The tuning did not transfer for MCM

Efficiency on the two evaluation repositories, before and after, with the paired
bootstrap on the tuned-minus-untuned difference:

| repo | system | untuned | tuned | change |
| --- | --- | --- | --- | --- |
| itsdangerous | vector-rag | 0.130 | 0.200 | **+0.070** separated |
| itsdangerous | graph | 0.114 | 0.165 | **+0.051** separated |
| itsdangerous | hybrid | 0.127 | 0.176 | +0.049 overlapping |
| itsdangerous | mcm | 0.240 | 0.246 | +0.006 overlapping |
| flask | vector-rag | 0.057 | 0.071 | **+0.015** separated |
| flask | graph | 0.034 | 0.035 | +0.001 overlapping |
| flask | hybrid | 0.046 | 0.050 | +0.004 overlapping |
| flask | mcm | 0.0688 | 0.0690 | +0.0002 negligible |

**MCM gained +0.046 on the repository it was fitted on and nothing on either
repository it was not.** Baseline A gained on both, with separated intervals.

The asymmetry has an obvious mechanical explanation. MCM's search covered seven
parameters across 1404 configurations and took the maximum; Baseline A's covered
two parameters across six. Selection bias scales with the size of the search, and
a seven-dimensional argmax over 237 tasks finds repository-specific structure that
a six-point sweep cannot. This is the expected failure mode, it is why the tuning
repository is held separate from the evaluation ones, and it is what the
tuning-set column exists to make visible.

A second cost is visible in the recall column, because the objective was
efficiency and nothing else. MCM's recall on itsdangerous fell from 0.518 to
0.408, and on flask from 0.371 to 0.341. Fitting for context efficiency bought
essentially no efficiency on held-out data and gave up recall to do it.
`--metric recall` would fit a different point; that has not been run.

### The headline result changes

Held-out scores with every system tuned:

| repo | system | recall | precision | MRR | efficiency |
| --- | --- | --- | --- | --- | --- |
| itsdangerous | vector-rag | 0.320 | 0.157 | 0.388 | 0.200 |
| itsdangerous | graph | 0.238 | 0.139 | 0.288 | 0.165 |
| itsdangerous | hybrid | 0.395 | 0.178 | 0.454 | 0.176 |
| itsdangerous | **mcm** | **0.408** | **0.228** | **0.512** | **0.246** |
| flask | vector-rag | 0.279 | 0.085 | 0.250 | 0.071 |
| flask | graph | 0.195 | 0.078 | 0.176 | 0.035 |
| flask | hybrid | 0.311 | 0.095 | 0.277 | 0.050 |
| flask | **mcm** | **0.341** | **0.110** | **0.315** | 0.069 |

MCM against each baseline on efficiency, after tuning:

| repo | vs vector-rag | vs graph | vs hybrid |
| --- | --- | --- | --- |
| itsdangerous | approximately equal | approximately equal | approximately equal |
| flask | approximately equal | **greater than** | **greater than** |

Compare that with the untuned run, where MCM was `greater than` all three
baselines on itsdangerous with separated intervals.

**MCM's context-efficiency advantage over vector RAG does not survive tuning the
baselines on equal terms.** What survives is an advantage over graph memory and
over the hybrid on flask, and a lead on recall, precision and MRR everywhere that
no longer separates cleanly on the smaller repository.

Had only MCM's weights been fitted, this document would be reporting tuned MCM at
0.246 against untuned vector RAG at 0.130 and calling it a win. That comparison
would have been meaningless, and it is the reason every system is tuned.

The spec section 3 chain is unchanged by tuning: `graph > vector-rag` fails on
both repositories, before and after.

It is not unchanged by repository size. See the next section.

## Django, and why the second link is about the baseline, not the size

Everything above is drawn from two repositories of 16 and 83 Python files. The
size tiers section 49 asks for were skipped for cost, and the cost was real:
ingestion committed once per write, and reading a revision spawned one `git`
process per file, so Django was a multi-hour proposition. Both are fixed, Django
indexes in 44 seconds, and the run is no longer expensive enough to skip.

Two runs on **django** (`django/django`, 2932 Python files), identical except for
the embedding provider every system shares:

| | horizon | tasks | provider |
| --- | --- | --- | --- |
| run A | 400 back | 426 | `hashed-token/256/4` (offline feature hashing) |
| run B | 250 back | 218 | `all-MiniLM-L6-v2` on CUDA |

**Run A, feature hashing:**

| system | recall | precision | MRR | efficiency | tokens |
| --- | --- | --- | --- | --- | --- |
| vector-rag | 0.200 | 0.058 | 0.192 | 0.042 | 3999 |
| graph | 0.148 | 0.044 | 0.141 | 0.060 | 3992 |
| hybrid | 0.237 | 0.066 | 0.236 | 0.060 | 3998 |
| **mcm** | **0.347** | **0.120** | **0.410** | **0.125** | 3568 |

**Run B, a trained embedding model:**

| system | recall | precision | MRR | efficiency | tokens |
| --- | --- | --- | --- | --- | --- |
| vector-rag | 0.285 | 0.079 | 0.253 | 0.053 | 3999 |
| graph | 0.135 | 0.037 | 0.140 | 0.061 | 3987 |
| hybrid | 0.315 | 0.085 | 0.324 | 0.068 | 3998 |
| **mcm** | **0.454** | **0.143** | **0.496** | **0.114** | 3641 |

### MCM's lead holds under both

| link | run A (hashing) | run B (trained) |
| --- | --- | --- |
| mcm > vector-rag | +0.083 [+0.065, +0.101] | +0.061 [+0.040, +0.086] |
| mcm > graph | +0.065 [+0.048, +0.083] | +0.054 [+0.032, +0.076] |
| mcm > hybrid | +0.066 [+0.048, +0.084] | +0.047 [+0.028, +0.068] |

All six separated. This is the first repository where MCM beats all three
baselines on efficiency with intervals this tight, and it holds whichever
embedding every system is given. The margin narrows under the better model,
which is the pattern to expect and the pattern the tuning section already found.

### The second link is an artefact of a weak embedding

| link | run A (hashing) | run B (trained) |
| --- | --- | --- |
| graph > vector-rag | **greater than** +0.018 [+0.006, +0.030] | **approximately equal** +0.007 [−0.011, +0.029] |

Run A supports H1 on django. Run B does not.

Nothing about the repository changed between them. What changed is that vector
RAG got an embedding model that works: its efficiency went 0.042 → 0.053 and its
recall 0.200 → 0.285, which closes the gap to graph memory and returns the second
link to `approximately equal`. Graph memory barely moved (0.060 → 0.061), which is
correct, because it does not embed anything.

So the reading is **not** "graph memory beats vector RAG on large repositories".
It is that graph memory beats *feature hashing* on large repositories, and feature
hashing is not a vector RAG system anyone would deploy. Run A on its own would
have been a flattering and wrong conclusion, and it is the exact mistake the
tuning section was written to avoid, arriving through a different door.

Across all three repositories and both providers, the second link of section 3's
chain holds in exactly one configuration out of five, and that configuration is
the one with the weakest baseline. **`graph > vector-rag` is not supported.**

A caveat on the comparison: the two runs use different horizons (400 and 250) and
therefore different task sets, so they are not paired. The direction is consistent
with the small-repository result and with an 8-task pilot, but a controlled
provider ablation at a fixed horizon has not been run and would settle it properly.

### The impact family answers a different question

Django is the first repository large enough to make this visible:

| system | impact recall | impact efficiency | tokens spent |
| --- | --- | --- | --- |
| vector-rag | 0.117 | 0.033 | 3999 |
| graph | 0.169 | 0.068 | 3974 |
| hybrid | **0.219** | 0.062 | 3998 |
| mcm | 0.128 | **0.090** | **2359** |

MCM has the *worst* recall of the four on the family built to showcase it, and
the best efficiency, while spending 41% fewer tokens than anyone else. Those are
the same fact. Dependency propagation returns the causal set it can justify and
then stops; the baselines fill the budget. Recall@10 against a commit's entire
changeset rewards filling the budget, because a commit touches things that are
not causally downstream of anything.

So the impact row is not evidence that propagation retrieves badly. It is
evidence that this metric and this ground truth disagree about what the impact
task is asking for. A precision-oriented or budget-normalised score would read
differently, and neither has been run. Reporting the recall column alone, in
either direction, would be misleading.

### Index build cost

Not previously reported, and it is a real cost:

| system | run A (hashing) | run B (trained, CUDA) |
| --- | --- | --- |
| vector-rag | 14.7s | 35.2s |
| graph | 9.3s | 11.5s |
| hybrid | 0.0s (reuses both) | 0.0s |
| **mcm** | **175.0s** | **272.6s** |

MCM is roughly 12x the most expensive baseline to index, because it parses,
resolves symbols, extracts typed relations and computes equivalence fingerprints
where a chunker splits on line counts. Whole runs: 1704 scores in 4443s (A) and
872 scores in 2684s (B).

The trade is that indexing is per-revision and retrieval is per-query. Whether
12x at index time is worth the efficiency margin at query time depends on how
many queries a revision serves, which this harness does not measure.

## The evaluation suite, run twice

Three scored repositories was thin for the claim, and the composition was thinner
than the count: `itsdangerous` and `flask` are both Pallets projects with the same
maintainer, idioms and test style. The sharper problem was that the section 3
verdict moved every time evidence arrived, which three points cannot characterise.

`scripts/evaluation_suite.py` fixes a set of ten repositories across four size
tiers and six ecosystems. It is fixed in source and was chosen before any of it
ran, because selecting repositories once the scores are visible is how an
evaluation quietly becomes a demonstration. `httpx` is not in it: it is the
tuning repository.

The suite was run twice, differing only in the embedding provider every system
shares.

### Sweep A, feature hashing (10 repositories)

The default provider. Loads no model and is not a trained embedding.

| repository | tier | files | tasks | MCM | best baseline | margin |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| itsdangerous | tiny | 15 | 34 | **0.240** | vector-rag 0.130 | **+0.110** |
| python-dotenv | tiny | 20 | 34 | **0.088** | hybrid 0.082 | **+0.006** |
| requests | tiny | 37 | 130 | **0.029** | vector-rag 0.039 | **-0.010** |
| flask | small | 83 | 149 | **0.069** | vector-rag 0.057 | **+0.012** |
| click | small | 90 | 306 | **0.099** | graph 0.082 | **+0.017** |
| rich | medium | 213 | 234 | **0.079** | graph 0.071 | **+0.008** |
| pydantic | medium | 449 | 312 | **0.064** | graph 0.056 | **+0.008** |
| scrapy | medium | 491 | 329 | **0.129** | graph 0.100 | **+0.030** |
| sqlalchemy | large | 673 | 314 | **0.100** | graph 0.039 | **+0.061** |
| django | large | 2,932 | 107 | **0.173** | hybrid 0.090 | **+0.083** |

| link | n | ahead | level | behind |
| --- | ---: | ---: | ---: | ---: |
| `mcm > vector-rag` | 10 | **7** | 3 | **0** |
| `mcm > graph` | 10 | **6** | 4 | **0** |
| `mcm > hybrid` | 10 | **7** | 3 | **0** |
| `graph > vector-rag` | 10 | **4** | 5 | **1** |

### Sweep B, a trained embedding on GPU (8 repositories)

`all-MiniLM-L6-v2` on CUDA, shared by every system that embeds anything.

| repository | tier | files | tasks | MCM | best baseline | margin |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| itsdangerous | tiny | 15 | 34 | **0.205** | hybrid 0.124 | **+0.081** |
| python-dotenv | tiny | 20 | 34 | **0.093** | vector-rag 0.102 | **-0.008** |
| requests | tiny | 37 | 130 | **0.031** | vector-rag 0.040 | **-0.010** |
| flask | small | 83 | 149 | **0.067** | vector-rag 0.063 | **+0.004** |
| click | small | 90 | 306 | **0.100** | graph 0.082 | **+0.018** |
| rich | medium | 213 | 234 | **0.086** | graph 0.071 | **+0.015** |
| pydantic | medium | 449 | 312 | **0.068** | hybrid 0.057 | **+0.011** |
| scrapy | medium | 491 | 329 | **0.127** | hybrid 0.100 | **+0.027** |

| link | n | ahead | level | behind |
| --- | ---: | ---: | ---: | ---: |
| `mcm > vector-rag` | 8 | **5** | 3 | **0** |
| `mcm > graph` | 8 | **3** | 5 | **0** |
| `mcm > hybrid` | 8 | **4** | 4 | **0** |
| `graph > vector-rag` | 8 | **2** | 5 | **1** |

The large tier is missing from sweep B for a hardware reason rather than a
methodological one: torch plus a CUDA context costs roughly 1.5GB of host RAM
before any work starts, and sqlalchemy and django could not be scored alongside
it on a 16GB machine that was not otherwise idle. Two real memory reductions were
landed first and were not enough; see `--on-disk` and `_dependency_groups`. Sweep
A has no such constraint because feature hashing loads nothing, which is why it
is the complete one.

**The margin tracks repository size.** In sweep A the two large repositories give
+0.083 (django, 2932 files) and +0.061 (sqlalchemy, 673), the second and third
largest margins in the set, while the only negative is `requests` at 37 files.
That is the direction the design predicts: a 4000-token budget reaches most of a
small repository, so there is little for structure to discard, and reaches
proportionally less as the repository grows.

`itsdangerous` is the counterexample and should not be explained away. At 15 files
it gives the largest margin in the suite (+0.110). Sixteen files and 34 tasks is a
different problem shape from 2932 files and 107 tasks, and the honest reading is
that size is one factor rather than the factor. The hypothesis was not stated
before these runs, so it is something to test next, not a conclusion.

### The paired provider comparison

The 8 repositories scored under both sweeps are paired: same
repositories, same horizons, same task caps, one variable. Mean change in context
efficiency from hashing to a trained embedding, with the paired bootstrap:

| system | mean change | interval | |
| --- | ---: | --- | --- |
| vector-rag | +0.0060 | [-0.0071, +0.0168] | overlapping |
| graph | **+0.0000** | [+0.0000, +0.0000] | **negative control** |
| hybrid | +0.0053 | [+0.0006, +0.0106] | **separated** |
| mcm | -0.0035 | [-0.0166, +0.0045] | overlapping |

**Graph memory changed by exactly zero in every repository.** Baseline B embeds
nothing, so switching the embedding provider must leave it untouched, and it did,
to the last decimal, in all 8. That is a working negative control: it
says the provider really was the only thing that varied, and a comparison without
one cannot be told apart from a bug.

With that established: **a better embedding helps the systems that depend on
embeddings and does not help MCM.** The hybrid improves with a separated
interval, vector RAG improves on average, MCM is flat to slightly negative.

This is the mechanism behind three earlier observations that looked separate: the
django chain flipping when the provider changed, itsdangerous losing its margin
over graph memory, and the tuning section's finding that fitting the baselines on
equal terms erased MCM's efficiency lead. All three are the same effect. MCM's
advantage does not come from retrieval quality, so anything that raises retrieval
quality across the board narrows it.

Reproduce with:

```
python scripts/evaluation_suite.py --out results-hashing/
python scripts/evaluation_suite.py --out results-trained/ \
    --provider sentence-transformers:all-MiniLM-L6-v2@cuda
python scripts/compare_providers.py results-hashing/ results-trained/
```

## Ablations

Section 50 lists seven configurations. Four map onto components this benchmark
actually exercises, plus a fifth for the inference path:

| Ablation | What is removed |
| --- | --- |
| `mcm-no-graph` | graph channel: relevance by proximity to an anchor |
| `mcm-no-vector` | vector channel: embedding similarity |
| `mcm-no-symbolic` | symbolic channel: exact name resolution |
| `mcm-no-provenance` | provenance channel: extractor trust as a ranking term |
| `mcm-no-reasoning` | impact answered by retrieval rather than dependency propagation |

Section 50's "symbolic reasoning" is ambiguous between the exact-resolution
retrieval channel and the inference engine above the graph. Both are ablated,
separately, because collapsing them would lose whichever one matters.

Two of the seven are **not exercised** and are reported as such rather than run:

- **temporal**: the benchmark ingests one revision, so no relation has a validity
  interval to ignore. Removing temporal reasoning changes nothing this harness can
  observe.
- **constraints**: no task family asks whether a change violates a requirement
  (section 46 E), so the constraint checker is never called.

Printing a null result for either would look like a finding and be an artefact of
what is measured.

A contribution interval that includes zero means this task set could not resolve
the component's value, not that the component is worthless. A negative difference
means removing the component *helped*, which is a real possible outcome and the
reason section 50 exists.

### Negligible differences

A difference has to clear two bars before it is called a win: the bootstrap
interval must exclude zero, and the difference must exceed one percent of the
means being compared. The second bar is not decoration. On flask, with 149 paired
tasks, removing the provenance channel produced a difference of −0.0000002 with an
interval that excluded zero, and the first version of this report duly announced
that provenance "costs". It does not; the two configurations are the same system
to five decimal places. `Comparison.negligible` is that check.

### Ablation scores

| configuration | itsdangerous recall / eff | flask recall / eff |
| --- | --- | --- |
| mcm | 0.518 / 0.240 | 0.371 / 0.069 |
| mcm-no-graph | 0.464 / 0.173 | 0.366 / 0.068 |
| mcm-no-vector | 0.509 / 0.253 | 0.329 / 0.061 |
| mcm-no-symbolic | 0.518 / 0.243 | 0.374 / 0.070 |
| mcm-no-provenance | 0.518 / 0.239 | 0.371 / 0.069 |
| mcm-no-reasoning | 0.521 / 0.244 | **0.406** / 0.071 |

Contribution on efficiency, full minus ablated:

| component removed | itsdangerous | flask |
| --- | --- | --- |
| graph channel | **contributes** +0.0666 | negligible +0.0007 |
| vector channel | unresolved −0.0134 | **contributes** +0.0078 |
| symbolic channel | unresolved −0.0030 | unresolved −0.0015 |
| provenance channel | negligible +0.0006 | negligible −0.0000 |
| inference | unresolved −0.0043 | **costs** −0.0025 |

Three things follow, and none of them is the comfortable one.

**The two repositories disagree about which channel matters.** On itsdangerous the
graph channel is the only component that resolves as contributing, and the vector
channel nominally hurts. On flask that is exactly reversed. Whatever these numbers
measure, it is not a stable property of the architecture, and no single-repository
ablation of this system should be believed.

**Provenance contributes nothing measurable on either repository.** It is the one
component whose result is consistent, and the result is that ranking by extractor
trust changes nothing here. That is unsurprising given that a single ingestion run
assigns nearly uniform reliability, so the term has almost no variance to rank on.
It would take a corpus with mixed-quality extractors to test properly.

**Inference did not pay off on the impact family.** `mcm-no-reasoning` answers
impact by text retrieval instead of dependency propagation, and on flask it is
*better*, with a separated interval on efficiency and a recall of 0.406 against
MCM's 0.371. On itsdangerous the point estimate also favours removing it, though
unresolved. This is a negative result about MCM's headline reasoning path and it is
reported as one. The design predicts the opposite.

These ablations were run with the default weights, and the obvious objection was
that the defaults simply misallocate score, so the ablations measure bad weights
rather than useless components. That objection has now been tested, and it does
not hold up. Weights fitted on httpx bought MCM **+0.046 on httpx and nothing on
either held-out repository**. There is no evidence that a better weighting was
sitting there waiting to be found; the search found one, and it did not
generalise. Re-running the ablations under the fitted weights would therefore be
re-running them under a configuration that is not better on this data.

What remains is the summary the evidence supports: only the graph channel and the
vector channel have ever resolved as contributing, never on the same repository,
provenance measures as nothing on both, and the reasoning path has not shown the
advantage it was built for.

## Running it

```
mcm benchmark <repo> [--horizon N] [--max-tasks N] [--budget N]
                     [--k N] [--metric M] [--coverage-floor F]
                     [--provider SPEC] [--ablate] [--json PATH]
```

The repository must be a Git checkout with more than `--horizon` commits.

```
mcm benchmark ../flask --name flask --horizon 400 --max-tasks 60 --json out.json
mcm benchmark ../flask --name flask --horizon 400 --ablate
mcm benchmark ../flask --metric recall
mcm benchmark ../django --provider sentence-transformers:all-MiniLM-L6-v2@cuda
```

`--provider` selects the embedding model **every** system shares. The default is
offline feature hashing, which needs nothing installed and is not a trained
model; the django section above is the argument for not treating its results as
the last word. A trained model needs `pip install -e ".[embeddings]"`, and `@cuda`
runs it on the GPU, which on an RTX 3060 is about 6.9x the CPU throughput
(3798 against 551 code-sized texts per second) and is what makes a trained model
practical across a repository this size.

`--metric` selects what the section 3 chain is judged on; `efficiency` is the
default because section 48 names it the most important metric, but `recall`,
`precision` and `mrr` are all available and the verdict is recomputed for
whichever is chosen. The full per-task scores go to `--json` so a result can be
re-analysed without a re-run.

## An agent in the loop

Everything above measures what MCM puts in front of an agent. This measures what
an agent then does with it, which was limitation 8.

### The ground truth is the test suite, not the diff

The first attempt scored an agent against the definitions a commit happened to
touch, and the ceiling measurement killed it: the definitions `mcm_impact` names
cover a mean of 17% of a commit's edits, because commits touch things that are
not downstream of anything. Scoring against all of them charges propagation for
being right.

`scripts/oracle_tasks.py` builds tasks the tests adjudicate instead. Check out a
commit's parent; apply only the commit's change to one source file; the suite
now fails. Apply each other changed file alone and re-run. A file that reduces
the failure count is required, one that does not is incidental, and the required
set is the answer. Commit `333c28d7` touches ten files and the tests say one of
them was needed.

### What the agent sees

Each task is its own worktree at the parent commit with the target file alone
advanced -- the broken state, callers stale, suite red. The MCM index is built
from that same worktree, so the tool and the agent look at one revision. The
conditions differ only in the tool list:

    control     read_file, list_files, grep
    treatment   read_file, list_files, grep, mcm_impact

`grep` is in both deliberately. The comparison worth making is against an agent
doing what agents do today. Recall and precision are over files, the granularity
the tests adjudicated; the target file is excluded from both sides.

### The tasks are rare, and mostly not about impact

Across 520 candidate commits in five repositories, 23 produced a cross-file break
at all. flask produced none in 161 candidates. Whatever impact analysis is worth,
the situation it addresses is uncommon in these histories.

More limiting still is what the surviving tasks are made of. Classifying each by
which way the dependency runs between the target and the file the tests required:

| direction | tasks | what it is |
|---|---|---|
| downstream | 9 | a required file depends on the target -- what impact analysis predicts |
| upstream | 11 | the target now needs a helper that had to grow -- a co-requirement |
| none | 3 | the index records no edge either way |

A test suite cannot tell a consequence from a co-requirement; it only says the
file was needed. Propagation predicts consequences, so 14 of 23 tasks are outside
what it claims to do. `mcm_impact` names a required file on 4 of the 9 downstream
tasks and 0 of the other 14.

### The result

23 tasks, 3 trials per condition, `openai/gpt-oss-20b` at temperature 0,
interleaved and shuffled. 138 trials, 6 errored.

| group | condition | recall | precision | trials |
|---|---|---|---|---|
| `mcm_impact` names a required file (4 tasks) | control | 0.273 | 0.121 | 11 |
| | treatment | 0.750 | 0.375 | 12 |
| `mcm_impact` names nothing required (19 tasks) | control | 0.055 | 0.015 | 55 |
| | treatment | 0.037 | 0.005 | 54 |

Paired by task, the treatment is better on 5, worse on 2 and tied on 16, for a
mean delta of +0.072. A two-sided sign-flip permutation test over the 23 paired
deltas gives **p = 0.28**.

### Why the promising subgroup is not evidence

The first table invites the reading that the tool triples recall where it has
something to say. It does not survive the obvious check. Splitting those 12
treatment trials by whether the agent actually called `mcm_impact`:

| trials on the 4 informative tasks | n | mean recall |
|---|---|---|
| called `mcm_impact` | 3 | 0.667 |
| did not call it | 9 | 0.778 |

The treatment's advantage sits in the trials that never touched the tool. An
effect attributed to a tool that most of the winning trials did not use is not an
effect of the tool; it is four tasks and a lucky split. The subgroup is reported
because hiding it would be worse, and it should be read as noise.

### What would make this answerable

Not more trials. The binding constraint is nine downstream tasks, and the way to
get more is repositories with suites fast enough to adjudicate hundreds of
commits, since 4% of candidates survive. Two smaller things also bound it: the
agent is a 20b model, chosen because a 120b reasoning model answers in 30 seconds
and would have made the run ten hours; and the treatment agent called the tool in
46 of 66 trials, so part of what is being measured is whether a model reaches for
an unfamiliar tool at all.

## Honest limitations

1. **One horizon per repository.** All four size tiers are now reachable and a
   ten-repository suite is defined (see "The evaluation suite"), but every entry
   is still a single horizon, so within-repository variance is unmeasured. The
   early results in this document came from two `pallets` repositories, which is
   why the suite deliberately spans six ecosystems.
2. **MCM's fitted weights overfit the tuning repository** and buy nothing on
   held-out data. A smaller search, or the centroid of the top basin rather than
   its argmax, would likely transfer better; neither has been tried.
3. **Tuning used context efficiency as the sole objective**, which cost recall.
   `--metric recall` fits a different point and has not been run.
4. **Python only.** The parser and every extractor are Python-specific.
5. **Commit subjects are noisy queries.** They are what a maintainer wrote, not
   what an agent would ask, and the filters remove the worst of them rather than
   all of them.
6. **The default embedding provider is feature hashing**, not a trained model.
   Baseline A and MCM's vector channel share it, so the comparison is fair, but
   this limitation predicted that "a real sentence embedding would change both
   sides and possibly not equally", and the django runs confirm it: it changes
   them unequally and it flips the section 3 chain. `--provider` now selects a
   trained model, so any result reported from the default should be read as a
   result about feature hashing. Every number above except django run B is.
7. **`token_estimate` is not a BPE tokenizer.** Identifiers count as one token
   where a real tokenizer would split them, so absolute efficiency runs
   optimistic. The bias is common-mode.
8. **Section 47's agent-performance block is measured now, and the answer is a
   null.** Task completion, test pass rate and number of tool calls need an agent
   in the loop; this harness otherwise measures what is put in front of one.
   Latency is recorded; tokens consumed is the `tokens` column. Impact prediction
   on its own scores 0.88 precision over five repositories at a 7% firing rate
   (`docs/prediction.md`), which says the predictions are trustworthy when made.
   Whether an agent given them does better work is answered below, in
   "An agent in the loop", and the answer is that this experiment cannot show
   that it does.
9. **Memory quality metrics from section 47 are unmeasured.** Fact retention,
   contradiction rate and stale-fact rate need the contradiction detection of
   section 31 and the memory update of section 32, neither of which is built.
