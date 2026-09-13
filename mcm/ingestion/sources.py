"""Where source files come from.

Ingestion originally walked the filesystem directly. Historical ingestion needs
the same pipeline pointed at a Git revision instead, so the file supply is behind
an interface. Nothing above this module knows whether it is reading a working
tree or a commit.

This is also what makes ``ingest`` reusable per revision: replaying a repository's
history is running the existing pipeline once per commit against a different
provider, rather than a second implementation of extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache",
             ".pytest_cache", ".ruff_cache", "build", "dist", ".eggs"}


@dataclass(frozen=True)
class SourceFile:
    relpath: str
    content: bytes

    @property
    def size_bytes(self) -> int:
        return len(self.content)


class SourceProvider(Protocol):
    """A set of Python files at one point in a repository's history."""

    #: Human-readable description of the revision, used in evidence source refs.
    label: str

    def files(self) -> Iterable[SourceFile]: ...


def skipped(relpath: str) -> bool:
    return any(part in SKIP_DIRS for part in relpath.split("/"))


def directories_for(relpaths: Iterable[str]) -> list[str]:
    """Every directory implied by a set of file paths, shallowest first.

    Derived from the paths rather than read from disk, because a Git revision has
    no directory entries to walk.
    """
    found: set[str] = set()
    for relpath in relpaths:
        parts = relpath.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            found.add("/".join(parts[:index]))
    return sorted(found, key=lambda d: (d.count("/"), d))


class WorkingTreeProvider:
    """Python files as they currently sit on disk."""

    label = "working tree"

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    def files(self) -> Iterable[SourceFile]:
        for relpath in sorted(self._walk()):
            yield SourceFile(relpath=relpath,
                             content=(self.root / relpath).read_bytes())

    def _walk(self) -> Iterable[str]:
        """Python files under the root, pruning skipped directories as it goes.

        ``rglob`` would descend into ``.venv`` and ``.git`` in full and leave
        ``skipped`` to discard the results afterwards. On a repository with a
        virtualenv beside the source that is most of the walk: 7622 directory
        scans to find 94 files. Pruning at the directory level visits only the
        tree that can contain a match.
        """
        stack = [self.root]
        while stack:
            directory = stack.pop()
            try:
                entries = list(directory.iterdir())
            except OSError:  # unreadable directory: report nothing, keep walking
                continue
            for entry in entries:
                if entry.is_dir():
                    # Symlinked directories are not descended into. A link
                    # pointing at an ancestor is a cycle, and this walk has no
                    # visited set to notice one.
                    if entry.name not in SKIP_DIRS and not entry.is_symlink():
                        stack.append(entry)
                elif entry.suffix == ".py":
                    yield entry.relative_to(self.root).as_posix()


class GitRevisionProvider:
    """Python files as they stood at one commit."""

    def __init__(self, reader, sha: str) -> None:
        self.reader = reader
        self.sha = sha
        self.label = f"commit {sha[:8]}"

    def files(self) -> Iterable[SourceFile]:
        wanted = [relpath for relpath in self.reader.files_at(self.sha)
                  if relpath.endswith(".py") and not skipped(relpath)]
        # One git invocation for the whole revision. Reading these one at a time
        # costs a process per file, which is what made large repositories
        # impractical to replay.
        contents = self.reader.blobs(self.sha, wanted)
        for relpath in wanted:
            content = contents.get(relpath)
            if content is not None:
                yield SourceFile(relpath=relpath, content=content)
