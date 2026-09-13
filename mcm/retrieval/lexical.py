"""Lexical projection (spec section 23, development step 15).

Full-text search over the same documents the vector projection embeds. Spec
section 23 names what this is for: symbol lookup, exact names, error messages,
function names, commit messages, configuration keys. Every one of those is a case
where a near-miss is a wrong answer, which is why they are served by an inverted
index rather than by similarity.

Two things are indexed per document: the rendered text as written, and its token
expansion. ``validate_token`` therefore matches a search for ``validate_token``,
for ``validate`` and for ``token``, without the tokenizer having to guess which
convention the repository uses. The expansion comes from ``embedding.tokenize``,
so the lexical and vector channels agree about what a token is.

SQLite FTS5 here, PostgreSQL full-text search later (spec section 5). The
``LexicalIndex`` interface is the seam.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from ..storage.database import Store
from .documents import KINDS, RetrievalDocument, documents
from .embedding import tokenize

_SCHEMA = """
-- Derived index (spec section 23). Rebuildable from objects, relations and
-- evidence; never a source of truth.
CREATE VIRTUAL TABLE IF NOT EXISTS lexical_index USING fts5(
    key UNINDEXED,
    kind UNINDEXED,
    source_id UNINDEXED,
    digest UNINDEXED,
    text,
    terms
);
"""


@dataclass(frozen=True)
class LexicalMatch:
    key: str
    kind: str
    source_id: str
    #: Raw BM25 from the index. More negative is a better match, and the scale
    #: depends on the corpus, so this is for ordering and debugging only.
    score: float
    #: ``score`` mapped onto [0, 1] *within one result set*, best match at 1.0.
    #: Not comparable between queries. Spec section 26 needs every channel on a
    #: common scale to be weighted against the others; this is how lexical gets
    #: there, and the limitation is the honest cost of using BM25 for it.
    relevance: float


@dataclass
class LexicalBuildReport:
    indexed: int = 0
    unchanged: int = 0
    removed: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (f"{self.indexed} indexed, {self.unchanged} unchanged, "
                f"{self.removed} removed")


class LexicalIndex(ABC):
    """Storage and matching for full text."""

    @abstractmethod
    def upsert(self, documents_: Iterable[RetrievalDocument]) -> None: ...

    @abstractmethod
    def delete(self, keys: Iterable[str]) -> None: ...

    @abstractmethod
    def digests(self) -> dict[str, str]: ...

    @abstractmethod
    def match(self, terms: list[str], *, limit: int,
              kinds: Iterable[str] | None = None) -> list[tuple[str, str, str, float]]:
        """Documents matching any of ``terms``, as ``(key, kind, source_id, score)``
        with the implementation's native relevance score."""


#: Host parameters per statement. SQLite's compiled-in ceiling is 999 on builds
#: older than 3.32; staying under it costs nothing and asks nobody.
_MAX_PARAMS = 500


class SQLiteLexicalIndex(LexicalIndex):
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @classmethod
    def for_store(cls, store: Store) -> "SQLiteLexicalIndex":
        connection = getattr(store, "connection", None)
        if connection is None:
            raise TypeError(f"{type(store).__name__} exposes no SQLite connection; "
                            "pass a LexicalIndex implementation for this backend")
        return cls(connection)

    def upsert(self, documents_: Iterable[RetrievalDocument]) -> None:
        # FTS5 has no upsert and no unique constraint, so a replace is a delete
        # followed by an insert. Batched into one transaction to keep it atomic.
        rows = [(d.key, d.kind, d.source_id, d.digest, d.text,
                 " ".join(tokenize(d.text))) for d in documents_]
        if not rows:
            return
        self._delete_keys([row[0] for row in rows])
        self._conn.executemany(
            """INSERT INTO lexical_index (key, kind, source_id, digest, text, terms)
               VALUES (?, ?, ?, ?, ?, ?)""", rows)
        self._conn.commit()

    def delete(self, keys: Iterable[str]) -> None:
        self._delete_keys(list(keys))
        self._conn.commit()

    def _delete_keys(self, keys: list[str]) -> None:
        """Remove rows by key in as few statements as the parameter limit allows.

        ``key`` is an FTS5 ``UNINDEXED`` column, so there is no index to find it
        by and every ``DELETE ... WHERE key = ?`` reads the whole table. Issuing
        one per key - which is what ``executemany`` does - makes building the
        index quadratic in the document count: 15 seconds on a repository with
        eight thousand documents, and hours on one with several hundred thousand.
        Collapsing a batch into one ``IN`` clause does not remove the scan, it
        removes the *repetition* of it, which is the part that grew.
        """
        for start in range(0, len(keys), _MAX_PARAMS):
            chunk = keys[start:start + _MAX_PARAMS]
            self._conn.execute(
                "DELETE FROM lexical_index WHERE key IN (%s)"
                % ",".join("?" * len(chunk)), chunk)

    def digests(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, digest FROM lexical_index")
        return {row["key"]: row["digest"] for row in rows}

    def match(self, terms: list[str], *, limit: int,
              kinds: Iterable[str] | None = None) -> list[tuple[str, str, str, float]]:
        if not terms:
            return []
        # Every term is quoted, which makes it an FTS5 string literal rather than
        # syntax. A user searching for "NOT" or "user*" gets those characters,
        # not an operator.
        expression = " OR ".join('"%s"' % term.replace('"', '""') for term in terms)
        sql = ("SELECT key, kind, source_id, bm25(lexical_index) AS score "
               "FROM lexical_index WHERE lexical_index MATCH ?")
        params: list = [expression]
        if kinds is not None:
            kinds = list(kinds)
            sql += " AND kind IN (%s)" % ",".join("?" * len(kinds))
            params.extend(kinds)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        return [(r["key"], r["kind"], r["source_id"], r["score"])
                for r in self._conn.execute(sql, params)]


class LexicalProjection:
    """The section 23 projection: index the core as text, search it as text."""

    def __init__(self, store: Store, index: LexicalIndex | None = None) -> None:
        self.store = store
        self.index = index if index is not None else SQLiteLexicalIndex.for_store(store)

    def rebuild(self, *, kinds: tuple[str, ...] = KINDS,
                as_of: datetime | None = None,
                batch_size: int = 256) -> LexicalBuildReport:
        """Bring the index in line with the core, on the same digest comparison
        the vector projection uses."""
        report = LexicalBuildReport()
        known = self.index.digests()
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
                self._flush(pending, report)
                pending = []
        if pending:
            self._flush(pending, report)

        stale = scoped - seen
        if stale:
            self.index.delete(stale)
            report.removed = len(stale)
        return report

    def search(self, query: str, *, limit: int = 10,
               kinds: Iterable[str] | None = None) -> list[LexicalMatch]:
        """Documents matching any token of the query, best first."""
        rows = self.index.match(_query_terms(query), limit=limit, kinds=kinds)
        return [LexicalMatch(key, kind, source_id, score, relevance)
                for (key, kind, source_id, score), relevance
                in zip(rows, _normalise([row[3] for row in rows]))]

    def _flush(self, batch: list[RetrievalDocument],
               report: LexicalBuildReport) -> None:
        self.index.upsert(batch)
        report.indexed += len(batch)
        for d in batch:
            report.by_kind[d.kind] = report.by_kind.get(d.kind, 0) + 1


def _query_terms(query: str) -> list[str]:
    """Tokens to match, deduplicated but kept in query order."""
    seen: dict[str, None] = {}
    for token in tokenize(query):
        seen.setdefault(token, None)
    return list(seen)


def _normalise(scores: list[float]) -> list[float]:
    """Map BM25 scores onto [0, 1] with the best match at 1.0.

    BM25 in SQLite is negative and unbounded below, so there is no absolute scale
    to normalise against. Within one result set the ranking is what carries the
    information, and this preserves it. A single result scores 1.0: it is the best
    match of those that matched, which is all BM25 ever claims.
    """
    if not scores:
        return []
    best, worst = min(scores), max(scores)
    if best == worst:
        return [1.0] * len(scores)
    return [(worst - score) / (worst - best) for score in scores]
