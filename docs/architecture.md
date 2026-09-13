# Architecture

## The inversion

Most repository-memory systems make a graph or a vector index the thing that
knows. MCM makes a typed semantic record the thing that knows, and treats every
index as a view computed from it.

```
                     MCM SEMANTIC CORE
        Objects        Relations        Evidence / Provenance
                            |
                            v
                    Derived representations
             Graph         Vector        Symbolic index
                            |
                            v
                   Retrieval + Inference
```

In this milestone the canonical core is four SQLite tables: `objects`,
`relations`, `evidence`, `provenance`. One projection exists, the
`relation_arguments` adjacency index, and `GraphProjection` reads it. Vector and
lexical projections are not built yet, but they occupy the same structural slot.

The test of whether something is canonical is simple. If you could drop the table
and rebuild it from what remains, it is an index. `relation_arguments` passes that
test. `relations` does not.

## Why this matters for the research question

The ablation studies in section 50 require running MCM with the graph removed,
with the vector store removed, with symbolic reasoning removed. That is only
possible if none of those components holds knowledge the others cannot recover.
An architecture where the graph is the source of truth cannot answer the question
the project is asking.

## Layers

**`mcm/core`** defines the semantic model and has no dependencies on storage,
parsing or retrieval. Objects, relations, evidence, provenance, and the identity
scheme.

**`mcm/storage`** holds the `Store` interface and one implementation. Nothing
above this layer knows that SQLite exists. `projections.py` is the graph view and
holds no state of its own.

**`mcm/ingestion`** turns source code into objects and relations. It is
deterministic. Every fact it emits is readable off a syntax tree, which is what
lets it mark them AST evidence at confidence 1.0. When it cannot resolve a name it
records the failure in the ingestion report rather than guessing.

**`mcm/algebra`** holds relation properties, the composition operator, the
confidence model and the dependency closure. This layer decides what traversal is
*licensed*, which is not the same as what edges exist.

**`mcm/reasoning`** produces inferences and their explanations. Everything it
returns beyond depth 1 is marked as an inference and carries the relations and
evidence that produced it.

**`mcm/retrieval`** and **`mcm/agent`** are the query surface.

## The two-pass ingestion

Ingestion parses every file before resolving any name. A single pass cannot
resolve a reference to a module it has not read, and resolving forward references
by re-reading files would make ingestion order significant. The first pass builds
the symbol table; the second resolves against it.

## What ingestion refuses to do

`login` contains `user.is_active()`, where `user` is a local bound to the result of
`load_user`. Knowing that this reaches `User.is_active` needs type inference, which
V1 does not have. Rather than emit a plausible edge, ingestion reports:

```
auth.py:23 attribute call user.is_active (base user needs type inference)
```

A wrong dependency edge is worse than a missing one, because everything
downstream inherits it silently and the impact analysis presents it with
provenance that looks legitimate. Section 64 lists using an LLM for deterministic
parsing among the things not to do; guessing in the parser is the same mistake
without the LLM.

## Storage substitution

`Store` is an abstract base class with fourteen methods. `SQLiteStore` implements
it. A PostgreSQL implementation would change the JSON columns to JSONB and add
pgvector for the embedding projection, and nothing above `mcm/storage` would
change. This is section 53 Rule 9 as a structural property rather than an
intention.
