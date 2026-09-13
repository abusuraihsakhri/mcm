"""What text stands for a piece of the semantic core (spec sections 22 and 23).

The vector projection and the lexical projection both have to answer "what text
represents this object?". If they answered it separately they would drift, and a
lexical hit and a vector hit would stop referring to the same thing - which would
make the per-channel scores in ``hybrid.py`` incomparable. So they answer it once,
here.

A document is a *rendering*, never a record. It holds a digest of its own text so
that rebuilding an index can skip everything the core has not changed, and so that
an entry whose source was edited is re-embedded rather than left stale.

Spec section 22 asks for embeddings of objects, relations, evidence, documents and
code summaries. The first three exist in V1. There is no summariser, so there are
no code summaries: a generated summary would be an LLM inference indexed as if it
were a fact, and spec section 54 keeps those apart.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

from ..core.evidence import Evidence
from ..core.objects import MCMObject
from ..core.relations import MCMRelation
from ..storage.database import Store

#: The kinds of thing that get indexed. Used to scope a rebuild or a search.
OBJECT = "object"
RELATION = "relation"
EVIDENCE = "evidence"
KINDS = (OBJECT, RELATION, EVIDENCE)

#: Object properties worth putting in front of a retriever, in reading order.
#: Anything not listed is descriptive detail that would dilute the signal
#: (line numbers, byte sizes, parent SHAs).
_PROPERTY_FIELDS: tuple[tuple[str, str], ...] = (
    ("qualname", "qualname"),
    ("relpath", "path"),
    ("parameters", "parameters"),
    ("docstring", ""),
    ("subject", ""),
    ("body", ""),
    ("author", "author"),
)


@dataclass(frozen=True)
class RetrievalDocument:
    """One indexable unit of the core, rendered as text."""

    key: str
    kind: str
    source_id: str
    text: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def document_key(kind: str, source_id: str) -> str:
    return f"{kind}::{source_id}"


def object_document(obj: MCMObject) -> RetrievalDocument:
    """Render an object: its type and name first, then the properties that carry
    meaning rather than location."""
    lines = [f"{obj.type.value} {obj.name}"]
    for key, label in _PROPERTY_FIELDS:
        value = obj.properties.get(key)
        if not value:
            continue
        if isinstance(value, list):
            value = " ".join(str(item) for item in value)
        lines.append(f"{label} {value}".strip())
    return RetrievalDocument(document_key(OBJECT, obj.id), OBJECT, obj.id,
                             "\n".join(lines))


def relation_document(relation: MCMRelation, store: Store) -> RetrievalDocument:
    """Render a relation as the sentence it asserts.

    Spec section 22 gives ``embedding(CALLS(authenticate, validate_token))`` as an
    example, so relations are indexed in their own right rather than being folded
    into their endpoints. Argument names are resolved where the objects exist;
    an unresolvable argument keeps its ID, because dropping it would make two
    different relations render identically.
    """
    parts = [relation.relation_type.value]
    for argument in relation.arguments:
        obj = store.get_object(argument)
        if obj is None:
            parts.append(argument)
            continue
        qualname = obj.properties.get("qualname") or obj.name
        relpath = obj.properties.get("relpath")
        parts.append(f"{qualname} ({relpath})" if relpath else qualname)
    return RetrievalDocument(document_key(RELATION, relation.id), RELATION,
                             relation.id, " ".join(parts))


def evidence_document(evidence: Evidence) -> RetrievalDocument:
    """Render an evidence record. Its content is source text, test output or a
    commit message - spec section 23 names exactly these as lexical targets."""
    text = f"{evidence.source_type.value} {evidence.source_ref}\n{evidence.content}"
    return RetrievalDocument(document_key(EVIDENCE, evidence.id), EVIDENCE,
                             evidence.id, text)


def documents(store: Store, *, kinds: tuple[str, ...] = KINDS,
              as_of: datetime | None = None) -> Iterator[RetrievalDocument]:
    """Every document the core currently supports.

    Derived relations are excluded. They are not persisted (spec Rule 2), and
    indexing an inference beside the facts it came from would let a later search
    return the conclusion as though it were a source.
    """
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown document kinds: {sorted(unknown)}")

    if OBJECT in kinds:
        for obj in store.all_objects():
            yield object_document(obj)
    if RELATION in kinds:
        for relation in store.all_relations(include_derived=False, as_of=as_of):
            yield relation_document(relation, store)
    if EVIDENCE in kinds:
        for evidence in store.all_evidence():
            yield evidence_document(evidence)
