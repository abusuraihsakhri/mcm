"""Repository ingestion (spec section 19).

Runs the nine ingestion steps that V1 covers: walk the tree, create Repository /
Directory / File objects, parse each Python file, create symbol objects, and emit
relations. Git history (steps 7-9) is not part of V1.

Ingestion is a two-pass process. The first pass parses every file and builds the
repository-wide symbol table; the second resolves names against it. A single pass
cannot resolve a forward reference to a module it has not read yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..core.evidence import EvidenceType
from ..core.ids import path_id, repository_id, symbol_id
from ..core.objects import MCMObject, ObjectType, utcnow
from ..core.provenance import ExtractionMethod
from ..core.relations import MCMRelation, RelationType as RT
from ..storage.database import Store
from ..equivalence.engine import fingerprint_of
from .dependencies import (RelationBatch, assert_relation, extract_calls,
                           extract_imports, extract_inheritance)
from .parser import ParsedModule, parse_python
from .sources import SourceFile, SourceProvider, WorkingTreeProvider, directories_for
from .symbols import KIND_TO_TYPE, SymbolTable, symbol_kind


@dataclass
class IngestionReport:
    repository_id: str
    files: int = 0
    objects: int = 0
    relations: int = 0
    evidence: int = 0
    unresolved: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    #: Temporal diff against the previous ingest (spec section 18).
    opened: int = 0
    closed: int = 0
    unchanged: int = 0
    closed_relations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.files} files, {self.objects} objects, {self.relations} relations, "
                f"{self.evidence} evidence records, {len(self.unresolved)} unresolved names")

    def temporal_summary(self) -> str:
        return f"{self.unchanged} unchanged, {self.opened} opened, {self.closed} closed"


class RepositoryIngestor:
    def __init__(self, store: Store) -> None:
        self.store = store

    def ingest(self, root: str | Path, name: str | None = None) -> IngestionReport:
        """Ingest the working tree at ``root``."""
        root = Path(root).resolve()
        repo = name or root.name
        return self.ingest_source(WorkingTreeProvider(root), repo=repo, root=root)

    def ingest_source(self, provider: SourceProvider, *, repo: str,
                      root: Path | str = "", moment: datetime | None = None,
                      ) -> IngestionReport:
        """Ingest one revision from a source provider.

        ``moment`` is when this revision is taken to have become true. It defaults
        to now for a working-tree ingest, and is the commit date when replaying
        history, which is what puts real dates on relation validity instead of
        ingest times.
        """
        moment = moment or utcnow()
        report = IngestionReport(repository_id=repository_id(repo))

        # Read the current state before writing anything, so the diff compares
        # this revision against what was valid immediately before it.
        previously_valid = {relation.id for relation in
                            self.store.all_relations(as_of=moment)
                            if _belongs_to(relation, repo)}

        source_files = list(provider.files())
        batch = RelationBatch()

        # One revision is one transaction. A half-ingested revision is not a
        # useful thing to keep -- the next run rebuilds it from source anyway --
        # and committing per write costs an fsync each, which dominates ingestion.
        with self.store.bulk():
            self._create_repository_object(repo, root)
            self._create_path_objects(repo, source_files, report, batch)

            # Pass 1: parse everything, build the symbol table.
            table = SymbolTable(repo=repo)
            parsed_modules: list[tuple[ParsedModule, str, bytes]] = []
            for source in source_files:
                try:
                    parsed = parse_python(source.content, source.relpath)
                except Exception as exc:  # noqa: BLE001 - report, do not abort ingestion
                    report.parse_errors.append(f"{source.relpath}: {exc}")
                    continue
                table.register_module(parsed)
                parsed_modules.append((parsed, source.relpath, source.content))

            for parsed, _, _ in parsed_modules:
                table.bind_imports(parsed)

            # Pass 2: emit symbol objects and resolved relations.
            for parsed, relpath, content in parsed_modules:
                file_id = path_id(repo, relpath, "file")
                self._create_symbol_objects(repo, parsed, file_id, batch, report,
                                            content)
                batch.extend(extract_imports(parsed, table, file_id))
                batch.extend(extract_inheritance(parsed, table))
                batch.extend(extract_calls(parsed, table))

            self._create_library_objects(batch, report)
            self._apply_temporal_diff(batch, previously_valid, report, moment)
            self._persist(batch, report)

        report.unresolved = batch.unresolved
        return report

    def _apply_temporal_diff(self, batch: RelationBatch, previously_valid: set[str],
                             report: IngestionReport, moment: datetime) -> None:
        """Open new relations and close ones that stopped being observed.

        Spec section 18: a dependency removed by a refactor is closed rather than
        deleted, so a query against an earlier moment still answers. A relation
        that was closed and is observed again is reopened by ``put_relation``,
        whose upsert clears ``valid_until``.
        """
        extracted = {relation.id for relation in batch.relations}
        for relation in batch.relations:
            if relation.id not in previously_valid:
                relation.valid_from = moment

        closed = previously_valid - extracted
        for relation_id in sorted(closed):
            self.store.close_relation(relation_id, moment)

        report.opened = len(extracted - previously_valid)
        report.closed = len(closed)
        report.unchanged = len(extracted & previously_valid)
        report.closed_relations = sorted(closed)

    # --- object creation --------------------------------------------------

    def _create_repository_object(self, repo: str, root: Path | str) -> None:
        self.store.put_object(MCMObject(
            id=repository_id(repo), type=ObjectType.REPOSITORY, name=repo,
            properties={"root": str(root)},
        ))

    def _create_path_objects(self, repo: str, source_files: list[SourceFile],
                             report: IngestionReport, batch: RelationBatch) -> None:
        """Create Directory and File objects and their containment relations.

        Directories are derived from the file paths rather than read from disk,
        because a Git revision has no directory entries to walk.
        """
        for relpath in directories_for(f.relpath for f in source_files):
            self.store.put_object(MCMObject(
                id=path_id(repo, relpath, "directory"), type=ObjectType.DIRECTORY,
                name=relpath.rsplit("/", 1)[-1], properties={"relpath": relpath},
            ))
            report.objects += 1
            batch.add(*assert_relation(
                RT.CONTAINS, [self._parent_id(repo, relpath),
                              path_id(repo, relpath, "directory")],
                method=ExtractionMethod.FILESYSTEM, source_ref=relpath,
                evidence_type=EvidenceType.SOURCE_CODE,
                content="filesystem containment of " + relpath,
            ))

        for source in source_files:
            self.store.put_object(MCMObject(
                id=path_id(repo, source.relpath, "file"), type=ObjectType.FILE,
                name=source.relpath.rsplit("/", 1)[-1],
                properties={"relpath": source.relpath, "language": "python",
                            "size_bytes": source.size_bytes},
            ))
            report.objects += 1
            report.files += 1
            batch.add(*assert_relation(
                RT.CONTAINS, [self._parent_id(repo, source.relpath),
                              path_id(repo, source.relpath, "file")],
                method=ExtractionMethod.FILESYSTEM, source_ref=source.relpath,
                evidence_type=EvidenceType.SOURCE_CODE,
                content="filesystem containment of " + source.relpath,
            ))

    @staticmethod
    def _parent_id(repo: str, relpath: str) -> str:
        parent = relpath.rsplit("/", 1)[0] if "/" in relpath else ""
        return path_id(repo, parent, "directory") if parent else repository_id(repo)

    def _create_symbol_objects(self, repo: str, parsed: ParsedModule, file_id: str,
                               batch: RelationBatch, report: IngestionReport,
                               content: bytes = b"") -> None:
        for sym in parsed.symbols:
            kind = symbol_kind(parsed.relpath, sym.kind, sym.name)
            object_id = symbol_id(repo, parsed.relpath, kind, sym.qualname)
            self.store.put_object(MCMObject(
                id=object_id, type=KIND_TO_TYPE[kind], name=sym.name,
                properties={
                    "qualname": sym.qualname,
                    "relpath": parsed.relpath,
                    # Line numbers are properties, never identity (spec section 20).
                    "start_line": sym.start_line,
                    "end_line": sym.end_line,
                    "parameters": sym.parameters,
                    "docstring": sym.docstring,
                    # Equivalence digests (spec section 38). Recorded at ingestion
                    # so equivalence classes are a grouping over the store rather
                    # than a re-read of a repository that may no longer be on disk.
                    **_equivalence_digests(content, sym),
                },
            ))
            report.objects += 1

            # A method is contained by its class; everything else by its file.
            if "." in sym.qualname:
                owner_qualname = sym.qualname.rsplit(".", 1)[0]
                container_id = symbol_id(repo, parsed.relpath, "class", owner_qualname)
            else:
                container_id = file_id
            source_ref = parsed.relpath + ":" + str(sym.start_line)
            batch.add(*assert_relation(
                RT.CONTAINS, [container_id, object_id],
                method=ExtractionMethod.AST, source_ref=source_ref,
                evidence_type=EvidenceType.AST,
                content="definition of " + sym.qualname,
                properties={"start_line": sym.start_line, "end_line": sym.end_line},
            ))

    def _create_library_objects(self, batch: RelationBatch, report: IngestionReport) -> None:
        """Create Module objects for external libraries referenced by relations."""
        seen: set[str] = set()
        for relation in batch.relations:
            for arg in relation.arguments:
                if arg.startswith("lib://") and arg not in seen:
                    seen.add(arg)
                    self.store.put_object(MCMObject(
                        id=arg, type=ObjectType.MODULE, name=arg[len("lib://"):],
                        properties={"external": True},
                    ))
                    report.objects += 1

    # --- persistence ------------------------------------------------------

    def _persist(self, batch: RelationBatch, report: IngestionReport) -> None:
        for evidence in batch.evidence:
            self.store.put_evidence(evidence)
        for provenance in batch.provenance:
            self.store.put_provenance(provenance)
        for relation in batch.relations:
            self.store.put_relation(relation)
        report.relations = len({r.id for r in batch.relations})
        report.evidence = len({e.id for e in batch.evidence})


def _belongs_to(relation: MCMRelation, repo: str) -> bool:
    """True when a relation was extracted from this repository.

    Matched on the first argument, which for every extracted relation is an
    object inside the repository. The boundary check stops repository ``app``
    from claiming relations belonging to ``app_extras``.
    """
    prefix = "repo://" + repo
    head = relation.arguments[0]
    if not head.startswith(prefix):
        return False
    rest = head[len(prefix):]
    return rest == "" or rest[0] in "/#"


def _equivalence_digests(content: bytes, sym) -> dict:
    """Syntactic, AST and normal-form digests for one definition (spec section 38).

    Computed here because the source bytes are already in hand. A definition that
    does not parse standalone contributes no AST or normal form rather than a
    digest of broken text, so two unparseable definitions never land in the same
    equivalence class for the wrong reason.
    """
    if not content or sym.end_byte <= sym.start_byte:
        return {}
    source = content[sym.start_byte:sym.end_byte].decode("utf-8", errors="replace")
    fingerprint = fingerprint_of(source)
    digests = {"text_digest": fingerprint.text}
    if fingerprint.ast is not None:
        digests["ast_digest"] = fingerprint.ast
        digests["normal_form_digest"] = fingerprint.normal_form
    return digests
