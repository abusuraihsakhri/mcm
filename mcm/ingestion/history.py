"""Historical ingestion (spec section 19, steps 7 to 9).

Replays a repository forwards, one commit at a time. Each revision runs the
ordinary extraction pipeline against a ``GitRevisionProvider``, so relation
validity ends up carrying real commit dates instead of ingest times: a call
introduced in March is valid from March, and one removed in June is closed in
June.

Step 9, associating changes with source objects, is done by comparing the *text
of each definition* between a commit and its parent rather than by mapping diff
hunks onto line numbers. Two reasons:

* Hunk line numbers refer to the file at that revision, so mapping them onto the
  line spans held in the store would attribute changes to whatever symbol now
  occupies those lines.
* Object identity is already line-independent by design (spec section 20). A
  definition that moved but did not change is the same object, and comparing its
  text says so directly.

Nested definitions are attributed to both the member and its container, because
editing a method really does change the text of its class.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from ..core.change import ADDED, COSMETIC, ChangeKind
from ..core.evidence import EvidenceType
from ..core.ids import commit_id, path_id, repository_id, symbol_id
from ..core.objects import MCMObject, ObjectType
from ..core.provenance import ExtractionMethod
from ..core.relations import RelationType as RT
from ..storage.database import Store
from .dependencies import RelationBatch, assert_relation
from .git import GitCommit, GitError, GitReader
from ..equivalence.normalise import forms_of
from .parser import parse_python
from .repository import RepositoryIngestor
from .sources import GitRevisionProvider, skipped
from .symbols import symbol_kind


@dataclass
class CommitReport:
    commit: GitCommit
    opened: int = 0
    closed: int = 0
    changed_symbols: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.commit.short} {self.commit.subject[:44]:44} "
                f"+{self.opened:<3} -{self.closed:<3} "
                f"{len(self.changed_symbols)} symbols")


@dataclass
class HistoryReport:
    repository_id: str
    commits: list[CommitReport] = field(default_factory=list)

    @property
    def total_opened(self) -> int:
        return sum(c.opened for c in self.commits)

    @property
    def total_closed(self) -> int:
        return sum(c.closed for c in self.commits)

    def summary(self) -> str:
        return (f"{len(self.commits)} commits, {self.total_opened} relations opened, "
                f"{self.total_closed} closed")


class HistoryIngestor:
    """Ingest a repository by replaying its commits in order."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.ingestor = RepositoryIngestor(store)

    def ingest(self, root: Path | str, name: str | None = None, *,
               max_commits: int | None = None) -> HistoryReport:
        root = Path(root).resolve()
        repo = name or root.name
        reader = GitReader(root)
        if not reader.is_repository():
            raise GitError(f"{root} is not a Git repository")

        report = HistoryReport(repository_id=repository_id(repo))
        commits = reader.commits(max_count=max_commits)
        known_shas = {commit.sha for commit in commits}

        for commit in commits:
            report.commits.append(
                self._ingest_commit(reader, repo, root, commit, known_shas))
        return report

    def _ingest_commit(self, reader: GitReader, repo: str, root: Path,
                       commit: GitCommit, known_shas: set[str]) -> CommitReport:
        entry = CommitReport(commit=commit)
        self._create_commit_object(repo, commit)

        batch = RelationBatch()
        self._link_to_parents(repo, commit, known_shas, batch)

        # Extract this revision through the ordinary pipeline. The commit date is
        # the moment, so opened and closed relations carry historical validity.
        revision = self.ingestor.ingest_source(
            GitRevisionProvider(reader, commit.sha),
            repo=repo, root=root, moment=commit.date,
        )
        entry.opened = revision.opened
        entry.closed = revision.closed
        entry.parse_errors = revision.parse_errors

        self._attribute_changes(reader, repo, commit, entry, batch)
        self._persist(batch, commit)
        return entry

    # --- commit objects ---------------------------------------------------

    def _create_commit_object(self, repo: str, commit: GitCommit) -> None:
        self.store.put_object(MCMObject(
            id=commit_id(repo, commit.sha),
            type=ObjectType.COMMIT,
            name=commit.short,
            properties={
                "sha": commit.sha,
                "author": commit.author,
                "date": commit.date.isoformat(),
                "subject": commit.subject,
                "body": commit.body,
                "parents": list(commit.parents),
            },
        ))

    def _link_to_parents(self, repo: str, commit: GitCommit, known_shas: set[str],
                         batch: RelationBatch) -> None:
        """PRECEDES(parent, child), for parents inside the ingested range.

        A parent outside the range is skipped rather than invented, so a
        --max-commits window does not produce edges to objects that do not exist.
        """
        for parent in commit.parents:
            if parent not in known_shas:
                continue
            batch.add(*assert_relation(
                RT.PRECEDES, [commit_id(repo, parent), commit_id(repo, commit.sha)],
                method=ExtractionMethod.GIT, source_ref=commit.short,
                evidence_type=EvidenceType.COMMIT,
                content=f"{parent[:8]} is a parent of {commit.short}",
            ))

    # --- step 9: associate changes with source objects --------------------

    def _attribute_changes(self, reader: GitReader, repo: str, commit: GitCommit,
                           entry: CommitReport, batch: RelationBatch) -> None:
        changed = reader.changed_paths(commit)
        parent = commit.parents[0] if commit.parents else None

        for relpath, status in sorted(changed.items()):
            if not relpath.endswith(".py") or skipped(relpath):
                continue
            entry.changed_files.append(relpath)
            batch.add(*assert_relation(
                RT.TRANSFORMS, [commit_id(repo, commit.sha), path_id(repo, relpath, "file")],
                method=ExtractionMethod.GIT, source_ref=f"{commit.short}:{relpath}",
                evidence_type=EvidenceType.COMMIT,
                content=f"{commit.short} {_status_word(status)} {relpath}",
                properties={"status": status},
            ))

            after = _definitions(reader.blob(commit.sha, relpath), relpath)
            before = _definitions(reader.blob(parent, relpath) if parent else None, relpath)
            for qualname, kind, change_kind in _changed_definitions(before, after):
                object_id = symbol_id(repo, relpath, symbol_kind(relpath, kind, qualname), qualname)
                entry.changed_symbols.append(object_id)
                batch.add(*assert_relation(
                    RT.TRANSFORMS, [commit_id(repo, commit.sha), object_id],
                    method=ExtractionMethod.GIT,
                    source_ref=f"{commit.short}:{relpath}",
                    evidence_type=EvidenceType.COMMIT,
                    content=(f"{commit.short} changed the definition of {qualname} "
                             f"in {relpath}"),
                    properties={"subject": commit.subject,
                                "change_kind": change_kind},
                ))

    def _persist(self, batch: RelationBatch, commit: GitCommit) -> None:
        """Write the commit-derived relations.

        These open at the commit date and never close. A commit changing a
        definition is a permanent historical fact once it happens, unlike a call
        edge, which stops being true when the call is removed. Leaving valid_from
        unset would make "this commit changed decode_claims" answer true for
        moments before the commit existed.
        """
        for evidence in batch.evidence:
            self.store.put_evidence(evidence)
        for provenance in batch.provenance:
            self.store.put_provenance(provenance)
        for relation in batch.relations:
            relation.valid_from = commit.date
            self.store.put_relation(relation)


# --- definition comparison ------------------------------------------------

@dataclass(frozen=True)
class _Definition:
    """One definition as it stood at one revision."""

    kind: str
    digest: str
    parameters: tuple[str, ...]
    #: Parse-tree digest, or None when the definition does not parse standalone.
    ast_digest: str | None = None


def _definitions(content: bytes | None, relpath: str) -> dict[str, _Definition]:
    """qualname -> the definition as it stands at this revision."""
    if not content:
        return {}
    try:
        parsed = parse_python(content, relpath)
    except Exception:  # noqa: BLE001 - a revision may not parse; report nothing
        return {}
    return {
        sym.qualname: _Definition(
            kind=sym.kind,
            digest=hashlib.sha256(content[sym.start_byte:sym.end_byte]).hexdigest(),
            parameters=tuple(sym.parameters),
            ast_digest=_ast_digest(content, sym),
        )
        for sym in parsed.symbols
    }


def _changed_definitions(before: dict[str, _Definition],
                         after: dict[str, _Definition]) -> list[tuple[str, str, str]]:
    """Definitions added, removed, or whose text differs, and *how* each changed.

    The observed change kind matters beyond bookkeeping: spec section 61 compares a
    predicted change against an observed one, and a comparison needs both sides to
    be the same kind of thing. Both revisions are already parsed here, so
    classifying the edit costs one comparison of parameter lists.

    ``RENAME`` is absent. A rename appears as one definition disappearing and
    another arriving, because object identity is built from the qualname (spec
    section 20), and telling that pair apart from an unrelated delete and add needs
    body matching this does not attempt. Renames are therefore observed as a REMOVE
    plus an ADDED, which is what the identity model says actually happened.
    """
    changed: list[tuple[str, str, str]] = []
    for qualname, definition in after.items():
        previous = before.get(qualname)
        if previous is None:
            changed.append((qualname, definition.kind, ADDED))
        elif previous.digest != definition.digest:
            changed.append((qualname, definition.kind,
                            _classify(previous, definition)))
    for qualname, definition in before.items():
        if qualname not in after:
            changed.append((qualname, definition.kind, ChangeKind.REMOVE.value))
    return sorted(changed)


def _status_word(status: str) -> str:
    return {"A": "added", "M": "modified", "D": "deleted", "R": "renamed"}.get(
        status, "touched")


def _ast_digest(content: bytes, sym) -> str | None:
    source = content[sym.start_byte:sym.end_byte].decode("utf-8", errors="replace")
    forms = forms_of(source)
    return forms.ast if forms is not None else None


def _classify(before: _Definition, after: _Definition) -> str:
    """How a definition changed between two revisions.

    The text differs by the time this is called. What it means depends on what
    else moved with it:

    * a different parameter list is a SIGNATURE change, and every call site has to
      follow it
    * the same parse tree is a COSMETIC edit - a reformat, a comment, a quote style
      - and propagates nowhere
    * anything else is a BEHAVIOUR change

    Without the cosmetic case, a repository-wide reformat would enter spec section
    61's evaluation as hundreds of behaviour changes whose predicted consequences
    no commit ever confirms, and the accuracy report would be measuring the
    formatter.
    """
    if before.parameters != after.parameters:
        return ChangeKind.SIGNATURE.value
    if (before.ast_digest is not None
            and before.ast_digest == after.ast_digest):
        return COSMETIC
    return ChangeKind.BEHAVIOUR.value
