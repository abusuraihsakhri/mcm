-- Canonical semantic core. Everything else in MCM is a projection of these tables.

CREATE TABLE IF NOT EXISTS objects (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    name        TEXT NOT NULL,
    properties  TEXT NOT NULL DEFAULT '{}',
    state       TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_objects_type ON objects(type);
CREATE INDEX IF NOT EXISTS idx_objects_name ON objects(name);

CREATE TABLE IF NOT EXISTS relations (
    id             TEXT PRIMARY KEY,
    relation_type  TEXT NOT NULL,
    arguments      TEXT NOT NULL,
    properties     TEXT NOT NULL DEFAULT '{}',
    confidence     REAL NOT NULL DEFAULT 1.0,
    evidence_ids   TEXT NOT NULL DEFAULT '[]',
    valid_from     TEXT,
    valid_until    TEXT,
    provenance_id  TEXT,
    inference      TEXT,          -- NULL for asserted facts, JSON for derivations
    is_derived     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);
CREATE INDEX IF NOT EXISTS idx_relations_derived ON relations(is_derived);

-- Symbolic projection (spec section 24): the adjacency index that makes exact
-- relation matching and dependency traversal cheap. Rebuildable from relations.
CREATE TABLE IF NOT EXISTS relation_arguments (
    relation_id TEXT NOT NULL,
    position    INTEGER NOT NULL,
    object_id   TEXT NOT NULL,
    PRIMARY KEY (relation_id, position),
    FOREIGN KEY (relation_id) REFERENCES relations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_relargs_object ON relation_arguments(object_id, position);

CREATE TABLE IF NOT EXISTS evidence (
    id                TEXT PRIMARY KEY,
    source_type       TEXT NOT NULL,
    source_ref        TEXT NOT NULL,
    content           TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    confidence        REAL NOT NULL DEFAULT 1.0,
    timestamp         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provenance (
    id                 TEXT PRIMARY KEY,
    method             TEXT NOT NULL,
    agent              TEXT NOT NULL,
    source_ref         TEXT NOT NULL,
    source_reliability REAL NOT NULL DEFAULT 1.0,
    created_at         TEXT NOT NULL
);
