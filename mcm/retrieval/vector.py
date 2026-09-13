"""Vector projection (spec section 22, development step 14).

Embeddings of objects, relations and evidence, derived from the canonical core and
rebuildable from it at any time. Spec section 22 states the boundary this module
lives inside: the vector store is for semantic similarity, fuzzy retrieval and
conceptual matching, and "must not be responsible for exact dependency reasoning".
So nothing here returns a relation, walks a dependency, or produces a confidence.
It returns similarity, and ``hybrid.py`` decides what that is worth.

Two interfaces, because spec Rule 9 requires every component to be replaceable:

``VectorIndex``
    Where vectors live. ``SQLiteVectorIndex`` scans; a pgvector implementation of
    the same four methods is the production answer and changes no caller.
``EmbeddingProvider`` (in ``embedding.py``)
    What produces them.

Rebuilds are incremental against a content digest, not a timestamp. Re-indexing an
unchanged repository embeds nothing, which matters once the provider is a paid API
rather than a hash function. Changing provider invalidates every vector, because
the provider's name is part of what is stored and searches never mix models.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from array import array
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Iterator

from ..core.objects import utcnow
from ..storage.database import Store
from .documents import KINDS, RetrievalDocument, documents
from .embedding import EmbeddingProvider, Vector, cosine, get_provider

try:                                    # optional: the pure-Python scan below is
    import numpy as _np                 # what runs when NumPy is not installed
except ImportError:                     # pragma: no cover - depends on the install
    _np = None

_SCHEMA = """
-- Derived index (spec section 22). Rebuildable from objects, relations and
-- evidence; never a source of truth. Dropping this table loses no knowledge.
CREATE TABLE IF NOT EXISTS vector_index (
    key       TEXT NOT NULL,
    model     TEXT NOT NULL,
    kind      TEXT NOT NULL,
    source_id TEXT NOT NULL,
    digest    TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    vector    BLOB NOT NULL,
    built_at  TEXT NOT NULL,
    PRIMARY KEY (key, model)
);
CREATE INDEX IF NOT EXISTS idx_vector_model_kind ON vector_index(model, kind);
"""


@dataclass(frozen=True)
class VectorEntry:
    """One stored vector and the document version it was built from."""

    key: str
    model: str
    kind: str
    source_id: str
    digest: str
    vector: Vector


@dataclass(frozen=True)
class VectorMatch:
    key: str
    kind: str
    source_id: str
    similarity: float


@dataclass
class VectorBuildReport:
    model: str
    dimension: int
    embedded: int = 0
    unchanged: int = 0
    removed: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (f"{self.model} dim={self.dimension}: {self.embedded} embedded, "
                f"{self.unchanged} unchanged, {self.removed} removed")


class VectorIndex(ABC):
    """Storage for vectors. Knows nothing about what they mean."""

    @abstractmethod
    def upsert(self, entries: Iterable[VectorEntry]) -> None: ...

    @abstractmethod
    def delete(self, model: str, keys: Iterable[str]) -> None: ...

    @abstractmethod
    def digests(self, model: str) -> dict[str, str]:
        """Key to digest for everything stored under ``model``. This is what makes
        a rebuild incremental."""

    @abstractmethod
    def entries(self, model: str, *,
                kinds: Iterable[str] | None = None) -> Iterator[VectorEntry]: ...

    @abstractmethod
    def get(self, model: str, key: str) -> VectorEntry | None: ...


class SQLiteVectorIndex(VectorIndex):
    """Vectors in a BLOB column, searched by scanning.

    A scan is O(n) per query and is the right amount of machinery for a prototype
    whose largest corpus is a few thousand documents. It is also the reason the
    interface above exists: replacing this with pgvector or an ANN index is a new
    ``VectorIndex``, not a change to the projection or to retrieval.

    Vectors are stored as float32. The rounding is below the resolution of any
    ranking decision and halves the database.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @classmethod
    def for_store(cls, store: Store) -> "SQLiteVectorIndex":
        connection = getattr(store, "connection", None)
        if connection is None:
            raise TypeError(f"{type(store).__name__} exposes no SQLite connection; "
                            "pass a VectorIndex implementation for this backend")
        return cls(connection)

    def upsert(self, entries: Iterable[VectorEntry]) -> None:
        now = utcnow().isoformat()
        self._conn.executemany(
            """INSERT INTO vector_index
               (key, model, kind, source_id, digest, dimension, vector, built_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(key, model) DO UPDATE SET
                   kind=excluded.kind, source_id=excluded.source_id,
                   digest=excluded.digest, dimension=excluded.dimension,
                   vector=excluded.vector, built_at=excluded.built_at""",
            [(e.key, e.model, e.kind, e.source_id, e.digest, len(e.vector),
              _pack(e.vector), now) for e in entries],
        )
        self._conn.commit()

    def delete(self, model: str, keys: Iterable[str]) -> None:
        self._conn.executemany("DELETE FROM vector_index WHERE model = ? AND key = ?",
                               [(model, key) for key in keys])
        self._conn.commit()

    def digests(self, model: str) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT key, digest FROM vector_index WHERE model = ?", (model,))
        return {row["key"]: row["digest"] for row in rows}

    def entries(self, model: str, *,
                kinds: Iterable[str] | None = None) -> Iterator[VectorEntry]:
        sql = "SELECT * FROM vector_index WHERE model = ?"
        params: list = [model]
        if kinds is not None:
            kinds = list(kinds)
            sql += " AND kind IN (%s)" % ",".join("?" * len(kinds))
            params.extend(kinds)
        for row in self._conn.execute(sql, params):
            yield _row_to_entry(row)

    def get(self, model: str, key: str) -> VectorEntry | None:
        row = self._conn.execute(
            "SELECT * FROM vector_index WHERE model = ? AND key = ?",
            (model, key)).fetchone()
        return _row_to_entry(row) if row else None


@dataclass(frozen=True)
class _Dense:
    """Every vector for one model, stacked so a query costs one dot product.

    The scan this replaces was 63% of a benchmark run on an 83-file repository:
    a Python-level ``cosine`` per entry, over lists of boxed floats, running its
    arithmetic at about four megaflops on hardware that does thousands of times
    that. Nothing about the ranking changes - the accumulation is still float64
    and the defensive renormalisation below reproduces ``cosine`` branch for
    branch - only the loop moves out of the interpreter.

    Rows are ordered by key so that a *stable* sort on similarity alone
    reproduces the ``(-similarity, key)`` ordering the scan produced, without
    carrying the keys through the sort.
    """

    keys: list[str]
    kinds: list[str]
    source_ids: list[str]
    matrix: "_np.ndarray"       # (rows, dimension), float64
    norms: "_np.ndarray"        # (rows,), each row's length


def _stack(entries: Iterable[VectorEntry]) -> _Dense:
    """Build the dense form. Costs one transient copy of the index in lists."""
    rows = sorted(entries, key=lambda entry: entry.key)
    if not rows:
        empty = _np.zeros((0, 0), dtype=_np.float64)
        return _Dense([], [], [], empty, _np.zeros(0, dtype=_np.float64))
    matrix = _np.empty((len(rows), len(rows[0].vector)), dtype=_np.float64)
    for position, entry in enumerate(rows):
        matrix[position] = entry.vector
    return _Dense(
        keys=[entry.key for entry in rows],
        kinds=[entry.kind for entry in rows],
        source_ids=[entry.source_id for entry in rows],
        matrix=matrix,
        norms=_np.sqrt(_np.einsum("ij,ij->i", matrix, matrix)),
    )


class VectorProjection:
    """The section 22 projection: build embeddings from the core, search them."""

    def __init__(self, store: Store, index: VectorIndex | None = None,
                 provider: EmbeddingProvider | None = None) -> None:
        self.store = store
        self.provider = provider or get_provider()
        self.index = index if index is not None else SQLiteVectorIndex.for_store(store)
        #: Stacked vectors per ``kinds`` filter, built on first search and dropped
        #: by ``rebuild``. A caller that writes through ``self.index`` directly is
        #: mutating behind this cache and must call ``invalidate``.
        self._dense: dict[tuple[str, ...] | None, _Dense] = {}

    @property
    def model(self) -> str:
        return self.provider.name

    def rebuild(self, *, kinds: tuple[str, ...] = KINDS,
                as_of: datetime | None = None,
                batch_size: int = 128) -> VectorBuildReport:
        """Bring the index in line with the core.

        Embeds documents that are new or whose text changed, and deletes entries
        whose source is gone. Both directions matter: an object deleted from the
        repository must not stay retrievable, and the temporal model (spec section
        18) means a relation can stop being valid without being deleted, which
        drops it out of ``documents()`` and so out of the index.

        ``kinds`` scopes the rebuild. Scoping it also scopes the pruning, so
        rebuilding only objects never deletes relation vectors.
        """
        report = VectorBuildReport(model=self.model, dimension=self.provider.dimension)
        known = self.index.digests(self.model)
        scoped = {key for key in known if key.split("::", 1)[0] in kinds}
        seen: set[str] = set()
        pending: list[RetrievalDocument] = []

        for document in documents(self.store, kinds=kinds, as_of=as_of):
            seen.add(document.key)
            if known.get(document.key) == document.digest:
                report.unchanged += 1
                continue
            pending.append(document)
            if len(pending) >= batch_size:
                self._embed_batch(pending, report)
                pending = []
        if pending:
            self._embed_batch(pending, report)

        stale = scoped - seen
        if stale:
            self.index.delete(self.model, stale)
            report.removed = len(stale)
        self.invalidate()
        return report

    def invalidate(self) -> None:
        """Drop the stacked form. The next search rebuilds it from the index."""
        self._dense.clear()

    def search(self, query: str, *, limit: int = 10,
               kinds: Iterable[str] | None = None,
               threshold: float = 0.0) -> list[VectorMatch]:
        """Nearest documents to a free-text query, most similar first.

        ``threshold`` drops weak matches. It defaults to 0, which keeps everything
        with any positive similarity: this returns a ranking, and the decision
        about what is good enough belongs to the caller that knows the question.
        """
        vector = self.provider.embed_one(query)
        if _np is not None:
            return self._dense_search(vector, limit, kinds, threshold)
        matches = [
            VectorMatch(entry.key, entry.kind, entry.source_id,
                        cosine(vector, entry.vector))
            for entry in self.index.entries(self.model, kinds=kinds)
        ]
        matches = [m for m in matches if m.similarity > threshold]
        matches.sort(key=lambda m: (-m.similarity, m.key))
        return matches[:limit]

    def _dense_search(self, vector: Vector, limit: int,
                      kinds: Iterable[str] | None,
                      threshold: float) -> list[VectorMatch]:
        """The scan above, as one matrix-vector product.

        Every step mirrors the list comprehension it replaces. ``cosine`` returns
        a plain dot product when both vectors are unit to within 1e-9 and
        ``dot / (na * nb)`` otherwise, so both branches are reproduced rather
        than assumed away: stored vectors are normalised *then rounded to
        float32*, which leaves their norms off unity by more than that tolerance,
        so the dividing branch is the one that usually runs.
        """
        dense = self._dense_for(kinds)
        if not dense.keys:
            return []

        query = _np.asarray(vector, dtype=_np.float64)
        if query.shape[0] != dense.matrix.shape[1]:
            raise ValueError(
                f"dimension mismatch: {query.shape[0]} vs {dense.matrix.shape[1]}")
        query_norm = float(_np.sqrt(query @ query))
        if query_norm == 0.0:
            return []

        dots = dense.matrix @ query
        scale = dense.norms * query_norm
        scores = _np.divide(dots, scale, out=_np.zeros_like(dots), where=scale > 0.0)
        if abs(query_norm - 1.0) < 1e-9:
            scores = _np.where(_np.abs(dense.norms - 1.0) < 1e-9, dots, scores)

        keep = _np.flatnonzero(scores > threshold)
        if keep.size == 0:
            return []
        # Rows are key-ordered, so a stable sort on similarity alone leaves ties
        # in key order - the tie-break the scan got from sorting on (-sim, key).
        order = keep[_np.argsort(-scores[keep], kind="stable")][:limit]
        return [VectorMatch(dense.keys[i], dense.kinds[i], dense.source_ids[i],
                            float(scores[i])) for i in order]

    def _dense_for(self, kinds: Iterable[str] | None) -> _Dense:
        scope = None if kinds is None else tuple(sorted(kinds))
        dense = self._dense.get(scope)
        if dense is None:
            dense = _stack(self.index.entries(self.model, kinds=scope))
            self._dense[scope] = dense
        return dense

    def similarity(self, query_vector: Vector, key: str) -> float:
        """Similarity between an already-embedded query and one indexed document.

        Hybrid retrieval needs this: a candidate found by the lexical or graph
        channel still has to be scored on the vector channel, and re-running the
        whole search to find out is wasteful.
        """
        entry = self.index.get(self.model, key)
        return 0.0 if entry is None else cosine(query_vector, entry.vector)

    def _embed_batch(self, batch: list[RetrievalDocument],
                     report: VectorBuildReport) -> None:
        vectors = self.provider.embed([d.text for d in batch])
        self.index.upsert(
            VectorEntry(key=d.key, model=self.model, kind=d.kind,
                        source_id=d.source_id, digest=d.digest, vector=v)
            for d, v in zip(batch, vectors)
        )
        report.embedded += len(batch)
        for d in batch:
            report.by_kind[d.kind] = report.by_kind.get(d.kind, 0) + 1


def _pack(vector: Vector) -> bytes:
    return array("f", vector).tobytes()


def _unpack(blob: bytes) -> Vector:
    out = array("f")
    out.frombytes(blob)
    return list(out)


def _row_to_entry(row: sqlite3.Row) -> VectorEntry:
    return VectorEntry(key=row["key"], model=row["model"], kind=row["kind"],
                       source_id=row["source_id"], digest=row["digest"],
                       vector=_unpack(row["vector"]))
