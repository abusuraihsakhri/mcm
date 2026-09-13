# Semantic model

## Objects

Every knowledge element is an `MCMObject` with a type from the section 7
vocabulary. The full vocabulary is declared in `ObjectType`, but V1 ingestion
emits only the subset deterministic Python analysis can justify: `Repository`,
`Directory`, `File`, `Module`, `Class`, `Function`, `Method`, `Test`.

Section 53 Rule 8 says not to build a large ontology before testing the basic
hypothesis. Declaring the enum costs nothing; emitting types nothing can populate
would produce an ontology that looks richer than the evidence behind it.

`properties` holds descriptive facts that take no part in identity: line span,
signature, docstring, language. `state` holds mutable status that later phases
update. They are separate so a state change never looks like a redefinition.

## Identity

```
repo://<repo>                                  Repository
repo://<repo>/<relpath>#directory              Directory
repo://<repo>/<relpath>#file                   File
repo://<repo>/<relpath>#function:<qualname>    Function
repo://<repo>/<relpath>#class:<qualname>       Class
repo://<repo>/<relpath>#method:<Class.name>    Method
repo://<repo>/<relpath>#test:<qualname>        Test
lib://<name>                                   External module
```

Section 20 requires identity to survive refactoring where possible, and forbids
line numbers as identity. Start and end lines are properties. Moving a function
within its file changes its properties and not its ID, so every relation pointing
at it stays valid.

Derived records are content-addressed. Relation IDs hash the type, the arguments
and the extraction method, so re-ingesting an unchanged repository updates rows in
place instead of duplicating them. Including the extraction method is what lets
the same claim from two extractors stay two records with separate evidence, which
section 54 requires.

## Relations

Section 8 states as a fundamental architectural requirement that a relation is not
`(source, target, label)`. `MCMRelation` carries:

```python
id, relation_type, arguments, properties, confidence,
evidence_ids, valid_from, valid_until, provenance_id, inference
```

`arguments` is a list, not a pair. Nothing in V1 emits a relation with more than
two arguments, but the storage layer and the model accept them, which is what lets
the scoped constraints of section 13 and n-ary causal claims land later without a
schema migration.

`inference` is the fact/inference marker. It is `None` on an asserted relation and
holds the rule, path and premises on a derived one. `Store.all_relations` and
`Store.relations_for` exclude derived relations unless `include_derived=True` is
passed, so an inference has to be asked for and can never be returned in place of
a fact.

## Temporal validity

`valid_from` and `valid_until` are on the relation, and `is_valid_at` reads them.
Section 18 requires that historical knowledge is never simply deleted: a
dependency that disappears in a refactor is closed off with `valid_until` rather
than removed, so a query about a past state still answers.

V1 stores and honours these fields but no ingestion path writes them yet. Nothing
in the current pipeline observes a fact ceasing to hold, because that requires Git
ingestion. The field is present rather than retrofitted, because retrofitting
temporal validity onto a store that has been deleting rows means the history is
already gone.

## Uncertainty

Section 17 requires distinguishing several dimensions rather than collapsing them
into one number, and warns against multiplying them without a justified model.
They live on different records:

| Dimension | Where | Meaning |
|---|---|---|
| `confidence` | `MCMRelation` | Belief in this specific claim |
| `confidence` | `Evidence` | How strongly the artifact supports it |
| `source_reliability` | `Provenance` | How much the extractor is trusted |

Only relation confidence is combined along a path. Evidence strength and source
reliability stay attached to their records, where a caller inspecting a weak
result can see which one is responsible. `DEFAULT_SOURCE_RELIABILITY` gives AST
extraction 1.0 and LLM inference 0.6; these are not multiplied into relation
confidence anywhere in V1.

The fourth dimension section 17 names, `probability`, is not modelled. Nothing in
V1 produces a calibrated probability, and a field holding a number that is not one
would invite exactly the arithmetic the section warns against.
