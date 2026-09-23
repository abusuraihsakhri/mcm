"""SQLite implementation of the canonical Store."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from ..core.evidence import Evidence, EvidenceType
from ..core.objects import MCMObject, ObjectType
from ..core.provenance import ExtractionMethod, Provenance
from ..core.relations import Inference, MCMRelation, RelationType
from .database import Direction, Store

_SCHEMA = Path(__file__).with_name("schema.sql")


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


class SQLiteStore(Store):
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA.read_text(encoding="utf-8"))
        self._conn.commit()
        self._bulk_depth = 0

    def _settle(self) -> None:
        """Commit, unless a ``bulk()`` block is batching writes.

        Every write commits on its own by default, so a crash mid-ingest leaves a
        valid partial database. That durability costs one fsync per write, which
        dominates ingestion: 14401 commits for a 94-file repository, 62% of wall
        time. ``bulk()`` trades it away for the duration of a block.
        """
        if self._bulk_depth == 0:
            self._conn.commit()

    @contextmanager
    def bulk(self) -> Iterator["SQLiteStore"]:
        """Batch every write in this block into one transaction.

        Ingestion rebuilds the whole graph from source, so a failed run is
        discarded rather than resumed; there is nothing to salvage from a partial
        write and no reason to pay per-row durability for it. On exit the block
        commits once, or rolls back entirely if the body raised.

        Nesting is counted, so an inner block defers to the outer one and a single
        commit still ends the whole thing.
        """
        self._bulk_depth += 1
        try:
            yield self
        except BaseException:
            self._bulk_depth -= 1
            if self._bulk_depth == 0:
                self._conn.rollback()
            raise
        else:
            self._bulk_depth -= 1
            if self._bulk_depth == 0:
                self._conn.commit()

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying connection, for derived indexes that live in the same file.

        Not part of the ``Store`` interface. The vector and lexical projections
        (spec sections 22, 23) keep their tables beside the canonical ones so that
        a prototype database is still one file, and so an in-memory store can be
        indexed at all. Nothing reached through here is canonical: every table a
        projection owns is rebuildable from objects, relations and evidence, and
        the projections own their own DDL rather than adding it to ``schema.sql``.
        """
        return self._conn

    # --- objects ----------------------------------------------------------
    def put_object(self, obj: MCMObject) -> None:
        self._conn.execute(
            """INSERT INTO objects (id, type, name, properties, state, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   type=excluded.type, name=excluded.name,
                   properties=excluded.properties, state=excluded.state,
                   updated_at=excluded.updated_at""",
            (obj.id, obj.type.value, obj.name, json.dumps(obj.properties),
             json.dumps(obj.state), _iso(obj.created_at), _iso(obj.updated_at)),
        )
        self._settle()

    def get_object(self, object_id: str) -> MCMObject | None:
        row = self._conn.execute("SELECT * FROM objects WHERE id = ?", (object_id,)).fetchone()
        return _row_to_object(row) if row else None

    def find_objects(self, *, type: ObjectType | None = None,
                     name: str | None = None) -> list[MCMObject]:
        clauses, params = [], []
        if type is not None:
            clauses.append("type = ?")
            params.append(type.value)
        if name is not None:
            clauses.append("name = ?")
            params.append(name)
        sql = "SELECT * FROM objects"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return [_row_to_object(r) for r in self._conn.execute(sql, params)]

    def all_objects(self) -> Iterable[MCMObject]:
        return [_row_to_object(r) for r in self._conn.execute("SELECT * FROM objects")]

    # --- relations --------------------------------------------------------
    def put_relation(self, relation: MCMRelation) -> None:
        inference = (
            json.dumps({
                "rule": relation.inference.rule,
                "path": relation.inference.path,
                "premise_relation_ids": relation.inference.premise_relation_ids,
            }) if relation.inference else None
        )
        self._conn.execute(
            """INSERT INTO relations (id, relation_type, arguments, properties, confidence,
                                      evidence_ids, valid_from, valid_until, provenance_id,
                                      inference, is_derived)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   properties=excluded.properties, confidence=excluded.confidence,
                   evidence_ids=excluded.evidence_ids, valid_until=excluded.valid_until,
                   inference=excluded.inference""",
            (relation.id, relation.relation_type.value, json.dumps(relation.arguments),
             json.dumps(relation.properties), relation.confidence,
             json.dumps(relation.evidence_ids), _iso(relation.valid_from),
             _iso(relation.valid_until), relation.provenance_id, inference,
             int(relation.is_derived)),
        )
        # Refresh the adjacency projection for this relation.
        self._conn.execute("DELETE FROM relation_arguments WHERE relation_id = ?", (relation.id,))
        self._conn.executemany(
            "INSERT INTO relation_arguments (relation_id, position, object_id) VALUES (?, ?, ?)",
            [(relation.id, i, arg) for i, arg in enumerate(relation.arguments)],
        )
        self._settle()

    def get_relation(self, relation_id: str) -> MCMRelation | None:
        row = self._conn.execute("SELECT * FROM relations WHERE id = ?", (relation_id,)).fetchone()
        return _row_to_relation(row) if row else None

    def relations_for(self, object_id: str, *, direction: Direction = "any",
                      types: Iterable[RelationType] | None = None,
                      include_derived: bool = False,
                      as_of: datetime | None = None,
                      include_historical: bool = False) -> list[MCMRelation]:
        if direction not in {"out", "in", "any"}:
            raise ValueError(f"invalid relation direction: {direction!r}")

        sql = ("SELECT r.* FROM relations r "
               "JOIN relation_arguments a ON a.relation_id = r.id "
               "WHERE a.object_id = ?")
        params: list = [object_id]
        if direction == "out":
            sql += " AND a.position = 0"
        elif direction == "in":
            sql += " AND a.position > 0"
        if types is not None:
            type_list = [t.value for t in types]
            if not type_list:
                return []
            placeholders = ", ".join(["?"] * len(type_list))
            sql += " AND r.relation_type IN (" + placeholders + ")"
            params.extend(type_list)
        if not include_derived:
            sql += " AND r.is_derived = 0"
        clause, extra = _validity_clause(as_of, include_historical, prefix="r.")
        sql += clause
        params.extend(extra)
        return [_row_to_relation(r) for r in self._conn.execute(sql, params)]

    def all_relations(self, *, include_derived: bool = False,
                      as_of: datetime | None = None,
                      include_historical: bool = False) -> Iterable[MCMRelation]:
        sql = "SELECT * FROM relations WHERE 1 = 1"
        if not include_derived:
            sql += " AND is_derived = 0"
        clause, params = _validity_clause(as_of, include_historical)
        sql += clause
        return [_row_to_relation(r) for r in self._conn.execute(sql, params)]

    def close_relation(self, relation_id: str, valid_until: datetime) -> None:
        """End a relation's validity interval without deleting it.

        Spec section 18: never simply delete historical knowledge. A dependency
        that disappears in a refactor stops being returned by ordinary reads while
        remaining answerable by an ``as_of`` query against an earlier moment.
        """
        self._conn.execute("UPDATE relations SET valid_until = ? WHERE id = ?",
                           (_iso(valid_until), relation_id))
        self._settle()

    # A closed relation that is observed again is reopened by ``put_relation``,
    # whose upsert writes valid_until back to NULL. V1 stores one validity
    # interval per relation record, so the gap between the two observations is
    # not represented. Modelling disjoint intervals would need a row per interval
    # and an identity scheme that tolerates more than one row per claim.

    # --- evidence and provenance -----------------------------------------
    def put_evidence(self, evidence: Evidence) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO evidence
               (id, source_type, source_ref, content, extraction_method, confidence, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (evidence.id, evidence.source_type.value, evidence.source_ref, evidence.content,
             evidence.extraction_method, evidence.confidence, _iso(evidence.timestamp)),
        )
        self._settle()

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        row = self._conn.execute("SELECT * FROM evidence WHERE id = ?", (evidence_id,)).fetchone()
        return _row_to_evidence(row) if row else None

    def all_evidence(self) -> Iterable[Evidence]:
        return [_row_to_evidence(r) for r in self._conn.execute("SELECT * FROM evidence")]

    def put_provenance(self, provenance: Provenance) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO provenance
               (id, method, agent, source_ref, source_reliability, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (provenance.id, provenance.method.value, provenance.agent, provenance.source_ref,
             provenance.source_reliability, _iso(provenance.created_at)),
        )
        self._settle()

    def get_provenance(self, provenance_id: str) -> Provenance | None:
        row = self._conn.execute("SELECT * FROM provenance WHERE id = ?",
                                 (provenance_id,)).fetchone()
        if not row:
            return None
        return Provenance(
            id=row["id"], method=ExtractionMethod(row["method"]), agent=row["agent"],
            source_ref=row["source_ref"], source_reliability=row["source_reliability"],
            created_at=_dt(row["created_at"]),
        )

    def close(self) -> None:
        self._conn.close()


def _validity_clause(as_of: datetime | None, include_historical: bool,
                     prefix: str = "") -> tuple[str, list]:
    """SQL restricting rows to those valid at a moment (spec section 18)."""
    if include_historical:
        return "", []
    moment = _iso(as_of or datetime.now(timezone.utc))
    return (f" AND ({prefix}valid_from IS NULL OR {prefix}valid_from <= ?)"
            f" AND ({prefix}valid_until IS NULL OR {prefix}valid_until > ?)",
            [moment, moment])


def _row_to_evidence(row: sqlite3.Row) -> Evidence:
    return Evidence(
        id=row["id"], source_type=EvidenceType(row["source_type"]),
        source_ref=row["source_ref"], content=row["content"],
        extraction_method=row["extraction_method"], confidence=row["confidence"],
        timestamp=_dt(row["timestamp"]),
    )


def _row_to_object(row: sqlite3.Row) -> MCMObject:
    return MCMObject(
        id=row["id"], type=ObjectType(row["type"]), name=row["name"],
        properties=json.loads(row["properties"]), state=json.loads(row["state"]),
        created_at=_dt(row["created_at"]), updated_at=_dt(row["updated_at"]),
    )


def _row_to_relation(row: sqlite3.Row) -> MCMRelation:
    raw = json.loads(row["inference"]) if row["inference"] else None
    inference = Inference(**raw) if raw else None
    return MCMRelation(
        id=row["id"], relation_type=RelationType(row["relation_type"]),
        arguments=json.loads(row["arguments"]), properties=json.loads(row["properties"]),
        confidence=row["confidence"], evidence_ids=json.loads(row["evidence_ids"]),
        valid_from=_dt(row["valid_from"]), valid_until=_dt(row["valid_until"]),
        provenance_id=row["provenance_id"], inference=inference,
    )
