# Retrieval

Spec sections 22 to 26, development steps 14 and 15.

## Four channels, one pool

Spec section 25 requires that a query potentially use several retrieval
mechanisms, pooled and reranked before anything reasons over the result.

```
query
  ├── lexical    FTS5 over rendered documents          mcm/retrieval/lexical.py
  ├── vector     cosine over embeddings                mcm/retrieval/vector.py
  ├── graph      proximity to what the others found    mcm/storage/projections.py
  └── symbolic   exact resolution of names and IDs     mcm/retrieval/symbolic.py
            │
            ↓  candidate pool
            ↓  rerank: Score(x|q) = w_v V + w_l L + w_g G + w_s S + w_p P
       ranked objects, each with the routes that produced its score
```

Everything returned is an `MCMObject`. Relations and evidence are indexed, as
spec section 22 requires, but they are *routes*: a hit on
`CALLS(authenticate, validate_token)` is a reason to surface both endpoints, and a
hit on an error string in an evidence record is a way to reach the code it came
from. Spec section 43 asks for evidence shown in support of an answer, which is
the opposite of returning it instead of one.

## The projections are derived, and prove it

The graph projection is a pure view over the store. The vector and lexical
projections cannot be, because embedding costs money and inverted indexes cost
time. They are caches, and the property that keeps them from becoming a second
source of truth is that they are rebuilt from a content digest:

```
for each document rendered from the core:
    if the stored digest matches, skip it
delete every stored entry the core no longer produces
```

Both directions matter. The second one is subtle: the temporal model closes a
relation rather than deleting it (`docs/temporal-model.md`), so a fact can stop
being true while every row survives. It drops out of `documents()`, and the prune
step removes it from both indexes. A rebuild keyed on "modified since T" would
leave it retrievable forever.

Deleting `vector_index` and `lexical_index` loses no knowledge. `mcm index`
rebuilds them. That is the section 21 rule applied to all three projections
rather than only to the graph.

## One rendering, two indexes

`mcm/retrieval/documents.py` decides what text stands for an object, a relation or
an evidence record, and both projections read it. If they rendered separately they
would drift, and a lexical hit and a vector hit would stop referring to the same
thing, which would make their scores incomparable at the point where section 26
adds them together.

Spec section 22 also lists documents and code summaries. There is no summariser, so
there are no code summaries. A generated summary is an LLM inference, and indexing
one beside AST facts would erase the distinction spec section 54 exists to keep.

## The embedding provider is not a model

`HashedTokenProvider` is the default because the alternative was a 90MB download
between `pip install` and the first query. It is feature hashing over tokens and
character n-grams: deterministic, offline, and not trained on anything.

What it delivers is morphological similarity. `validate_tokens` retrieves
`validate_token` at 0.72 while unrelated code sits at 0.13, and a typo still finds
its target. What it does not deliver is spec section 22's "conceptual matching": it
has no idea that `session` and `login` are related, because nothing taught it.

This is the honest state of the vector channel. The section 26 weights are not
tuned to hide it, and `MCM_EMBEDDING_PROVIDER=sentence-transformers` swaps in a
real model against the same interface. The provider's name is stored with every
vector, so changing providers invalidates the index instead of silently comparing
two geometries that share no basis.

## What each channel contributes

**V, vector.** Cosine similarity, already in [0, 1] for a unit-normalised
provider.

**L, lexical.** BM25, which is negative and unbounded below, min-max normalised
*within one result set*. That makes it comparable to the other channels for one
query and not comparable across queries. This is a real limitation of using BM25
in a weighted sum, and it is written down rather than smoothed over.

**G, graph.** Breadth-first from anchors, decaying by `graph_decay` per hop. Two
rules keep it honest. An anchor scores nothing on this channel, because it is an
anchor by virtue of the text channels and crediting it twice would double-count one
piece of evidence. And a path that returns to the anchor it started from is
discarded, or every function would score on the graph channel by way of the file
that contains it.

Anchors are the exact symbolic matches when the query names something real, and
otherwise the three strongest text hits. Deferring to symbolic matches matters:
walking from a vaguely similar function drags in a neighbourhood that has nothing
to do with the question.

**S, symbolic.** Exact resolution of the query and of each of its tokens. This is
the channel spec section 22 is protecting when it says the vector store must not be
responsible for exact matching. If the query names something that exists, it is in
the pool at 1.0 regardless of what similarity thought.

**P, provenance.** Mean `source_reliability` over the provenance records of the
asserted relations touching the object. Evidence strength is deliberately not
folded in: spec section 17 forbids multiplying the uncertainty dimensions without a
justified model, and there is none here. An object with no provenance scores 0
rather than a flattering default, because an unsourced object is what this term
exists to rank down.

## The weights are guesses

```
vector 0.25   lexical 0.25   graph 0.25   symbolic 0.15   provenance 0.10
```

They sum to 1.0, so a default-weighted score reads as a fraction. They are not
tuned. Tuning them before the benchmark in spec sections 45 to 50 would mean
choosing numbers that flatter whatever example was at hand, and spec section 26 is
explicit that graph retrieval must not be assumed superior.

Every weight is settable:

```bash
mcm search "validate_token" --weights v=0.0,l=0.5,g=0.5
mcm search "validate_token" --weights graph_depth=4,graph_decay=0.25
```

The test suite asserts that each of the five terms changes the ranking, which is
the property an ablation study needs: no term is decorative.

## Every score names its source

```
1. validate_token  [Function] auth.py
     score 0.7321  v=0.928 l=1.000 g=0.000 s=1.000 p=1.000
     found by: vector, lexical, symbolic
       via lexical object object::repo://app/auth.py#function:validate_token (1.000)
       via symbolic object repo://app/auth.py#function:validate_token (1.000)
       via vector object object::repo://app/auth.py#function:validate_token (0.928)
```

A result four channels agree on and a result only similarity liked are different
claims, and a reader who cannot tell them apart has no way to distrust the ranking.
`g=0.000` on the top hit is the anchor rule above, visible rather than hidden.

## A worked failure

```
mcm search "how are users authenticated"
```

The lexical channel returns nothing. The corpus contains `authenticate` and `user`;
the query contains `authenticated` and `users`, and an inverted index over whole
terms does not bridge that. The vector channel does, through its character n-grams,
and `authenticate` still ranks first with `l=0.000`.

This is the case hybrid retrieval exists for, and it is worth stating that the
channel doing the work here is the weakest one in the system. A real embedding
model would do it better. A lexical index with a stemmer would also have caught it.
Neither has been tried, so neither is claimed.

## What is not here

No learned reranker. There is no training data and no relevance judgements, so a
cross-encoder would be an untested guess wearing a model's reputation. The weighted
sum is the "initial scoring model" spec section 26 asks for, and it is what the
ablations are meant to attack.

No approximate nearest neighbour search. `SQLiteVectorIndex` scans every vector,
which is the right amount of machinery for a corpus of a few thousand documents and
the reason the `VectorIndex` interface exists: pgvector is a new implementation of
four methods, not a change to any caller.

The scan is exhaustive but no longer per-entry. `VectorProjection` stacks the
index into one float64 matrix on first search and answers each query with a single
matrix-vector product, keeping the stacked form until `rebuild` drops it. NumPy is
optional: without it the original per-entry `cosine` loop runs instead, and a test
holds both to the same ranking. The loop was 63% of a benchmark run on an 83-file
repository, executing its arithmetic at roughly four megaflops, and removing it is
what makes a repository the size of Django measurable at all. It buys nothing
asymptotically - every query still touches every vector, and a corpus large enough
to need sublinear search still needs a different `VectorIndex`.

No retrieval-driven query type. Spec section 28 lists ten query types and none of
them is "search"; retrieval feeds the query engine rather than answering in its own
right, so `QueryType` is unchanged and `mcm search` is its own command.
