# Equivalence

Spec sections 38 and 10. Development step 19.

## What section 5 decides for you

Section 38 asks for an `EquivalenceEngine` supporting syntactic, AST, normal-form
and semantic equivalence, and adds: *for code transformations, investigate
integration with e-graphs / equality saturation rather than building a new
implementation from scratch.* Section 5 repeats it: *design an adapter for e-graph
functionality. Do not reimplement a mature e-graph engine unless necessary.*

Taking that seriously settles the whole design. Three of the four domains need no
e-graph at all - they are digest comparisons over a parse tree the ingestion
pipeline already produces. The fourth is undecidable, and an e-graph is where it
*would* be decided.

So the saturation backend is a seam with an honest `UNKNOWN` behind it. Nothing
here tries to be egg.

## The four domains

| domain | equal when | decidable |
|---|---|---|
| `syntactic` | the source text is identical | yes |
| `ast` | the parse trees match, comments and whitespace aside | yes |
| `normal_form` | the trees match after renaming the definition's own name and its parameters | yes |
| `semantic` | they compute the same thing | no |

Each is a digest stored on the object at ingestion (`text_digest`, `ast_digest`,
`normal_form_digest`), so a comparison needs no access to the repository. An
equivalence question can be answered against a database whose source tree is long
gone.

## The hierarchy, and the direction that matters

```
syntactic  ⟹  AST  ⟹  normal-form  ⟹  semantic
```

Equality propagates rightwards. Two definitions with identical text have identical
trees, and so on up to semantics.

**Inequality never propagates rightwards, and this is the load-bearing rule.** Two
functions with different normal forms may compute exactly the same thing - that is
the ordinary case, not a corner case:

```python
def area(w, h):            def area(w, h):
    return w * h               total = 0
                               for _ in range(h):
                                   total += w
                               return total
```

So `SEMANTIC` may return `EQUIVALENT` on the strength of matching normal forms, and
must **never** return `NOT_EQUIVALENT` without a backend able to prove
inequivalence. Without one the answer is `UNKNOWN`, naming what would settle it.

Getting this backwards is the one genuinely damaging mistake available here: it
would let a refactoring agent conclude two implementations differ because they were
spelled differently. `test_no_semantic_comparison_ever_returns_not_equivalent`
exists to stop that regression.

Verdicts are three-way for the same reason the constraint engine's are: a question
that was not decided must not answer as though it were.

## Normalisation

Normal forms come from tree-sitter, the parser ingestion already uses.

**Every node is included, named and anonymous.** Skipping anonymous tokens is the
usual shortcut for AST comparison and it is wrong here: tree-sitter records the
operator in `a + b` as an anonymous child, so dropping those would make `a + b` and
`a - b` identical. That would be a false *positive* - claiming equivalence that does
not hold - which the hierarchy then propagates all the way to a semantic
`EQUIVALENT`.

Keeping them means the opposite error is possible: an added trailing comma reads as
a difference. That error is one-directional and safe. These digests can say two
texts differ when they are cosmetically the same; they can never say two texts match
when they do not.

**Only bound names are renamed** - the definition's own name and its parameters.
Every other identifier is a free name, a call to another function or an imported
symbol, and renaming those would erase the references the rest of the semantic model
is built on. Two functions that call different things are not equivalent under any
domain implemented here. The consequence is that this is a *bounded*
alpha-equivalence: two implementations differing only in the name of a local
temporary are not reported as normal-form equivalent, because deciding that needs
scope analysis this phase does not have.

A definition that does not parse standalone has no form at all, rather than a digest
of its broken text. Otherwise two unparseable definitions would land in the same
equivalence class for the wrong reason.

## Equivalence classes

```bash
mcm equivalence --db repo.db                    # normal-form duplicates
mcm equivalence --domain ast --db repo.db
mcm equivalent authenticate --db repo.db
```

A grouping by stored digest, so it costs one pass rather than comparing every pair.

`--domain semantic` is refused rather than quietly answered at `normal_form`:
semantic classes are exactly what cannot be computed by grouping, since two members
of one can have nothing syntactic in common. Answering the easier question under the
requested name would be the kind of substitution section 64 warns about.

An empty result says what it means:

> No two definitions share a form. Note that this rules out duplication, not
> similarity: semantic equivalence is not decided here.

## What equivalence is actually used for

Not duplicate-hunting. The reason this step earns its place is that it repairs a
measurement in step 18.

History ingestion classifies every commit's edit to a definition so that section 61
can compare a prediction against an observation. Before this step the classification
was: parameters changed means `SIGNATURE`, otherwise `BEHAVIOUR`. That makes a
reformat a behaviour change.

A repository-wide `black` run would therefore enter the accuracy report as hundreds
of behaviour changes whose predicted consequences no commit ever confirms, and the
report would be measuring the formatter. With an AST digest on both revisions the
edit is classified `COSMETIC`, and `prediction.py` excludes it from observations
alongside `ADDED`.

`COSMETIC` is not a `ChangeKind`, for the same reason `ADDED` is not: there is
nothing to propagate. The code means exactly what it meant before.

## The adapter

```python
class SaturationBackend(Protocol):
    @property
    def name(self) -> str: ...
    def compare(self, left: str, right: str) -> tuple[Verdict, str]: ...
```

An `egglog` or `egg` integration is this protocol over that library, and nothing
else in the system changes. None is bundled: it cannot be exercised offline, and an
untested backend in the default install would be a worse claim than no backend. This
follows the same decision made for embedding providers in step 14.

A configured backend is consulted only when the normal forms already differ. Cheap
answers first - identical normal forms settle the question without paying for
saturation - and there is a test asserting the backend is not called in that case.

## Relations are built, not written

Section 10 requires an equivalence relation to name its domain, and `RELATION_SPECS`
already said where it goes: *"the domain belongs in relation properties, not in the
type."* `equivalence_relation` builds a relation carrying `domain`, `witness` and
`reason`, with arguments sorted so a symmetric relation has one identity rather than
two.

Nothing is persisted. An equivalence found here is decidable and could legitimately
be asserted, but writing it is a separate explicit decision - the same stance
`derive` takes with the rule engine.

## What is absent

Section 38's fourth domain, genuinely decided. There is no solver, no equality
saturation and no differential testing, so semantic equivalence is answered only
where a weaker domain settles it.

Normal-form equivalence modulo local variable renaming, statement reordering, or any
rewrite rule at all. A rewrite system is precisely what an e-graph would bring, and
section 5 says to adapt one rather than write one.
