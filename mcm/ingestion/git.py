"""Read Git history through the git CLI (spec section 19, steps 7 to 9).

Uses subprocess rather than a library binding. The prototype needs four
questions answered - list the commits, list the files at a commit, read a file at
a commit, list what a commit touched - and the CLI answers all four without
adding a dependency that would have to be replaced later.

Commit dates are read as author dates in ISO-8601 with offset, so they compare
correctly against the ISO strings the store keeps for relation validity.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

#: Record and field separators. ASCII control characters, so they cannot appear
#: in a commit subject or an author name.
_RECORD = "\x1e"
_FIELD = "\x1f"
_LOG_FORMAT = _FIELD.join(["%H", "%P", "%an", "%aI", "%s", "%b"]) + _RECORD


class GitError(RuntimeError):
    """A git command failed, or the path is not a repository."""


@dataclass(frozen=True)
class GitCommit:
    sha: str
    parents: tuple[str, ...]
    author: str
    date: datetime
    subject: str
    body: str = ""

    @property
    def short(self) -> str:
        return self.sha[:8]

    @property
    def is_root(self) -> bool:
        return not self.parents


@dataclass
class GitReader:
    root: Path

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    # --- plumbing ---------------------------------------------------------

    def _run(self, *args: str, binary: bool = False):
        try:
            completed = subprocess.run(
                ["git", *args], cwd=self.root, capture_output=True, check=True,
            )
        except FileNotFoundError as exc:
            raise GitError("git executable not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            raise GitError(f"git {' '.join(args)} failed: {detail}") from exc
        if binary:
            return completed.stdout
        return completed.stdout.decode("utf-8", errors="replace")

    def is_repository(self) -> bool:
        try:
            return self._run("rev-parse", "--is-inside-work-tree").strip() == "true"
        except GitError:
            return False

    # --- history ----------------------------------------------------------

    def commits(self, *, max_count: int | None = None,
                paths: list[str] | None = None) -> list[GitCommit]:
        """Commits in oldest-first order.

        Oldest first because history is replayed forwards: each revision is
        ingested against the state left by its predecessor.

        ``max_count`` selects the most *recent* N commits, which are then
        reversed. So a window starts partway through history, and the first
        revision in it opens every relation that existed by then rather than at
        the commit that introduced each one. Validity dates under a window are
        therefore upper bounds: no later than, not exactly when.
        """
        args = ["log", "--reverse", f"--format={_LOG_FORMAT}"]
        if max_count is not None:
            args.append(f"--max-count={max_count}")
        if paths:
            args.extend(["--", *paths])
        return [c for c in (_parse_commit(r) for r in self._run(*args).split(_RECORD))
                if c is not None]

    def files_at(self, sha: str) -> list[str]:
        """Every tracked path at a revision."""
        output = self._run("ls-tree", "-r", "--name-only", sha)
        return [line for line in output.splitlines() if line]

    def blob(self, sha: str, relpath: str) -> bytes | None:
        """File content at a revision, or None if it did not exist."""
        try:
            return self._run("show", f"{sha}:{relpath}", binary=True)
        except GitError:
            return None

    def blobs(self, sha: str, relpaths: list[str]) -> dict[str, bytes]:
        """File contents at a revision, in one git invocation.

        ``blob`` spawns a process per file. That is fine for a handful and ruinous
        for a revision: Django has 2932 Python files, and process creation alone
        makes reading them one at a time take minutes. ``git cat-file --batch``
        answers the whole list over a single pipe.

        Paths that do not exist at the revision are absent from the result rather
        than mapped to None, so the caller iterates what it got.
        """
        if not relpaths:
            return {}

        request = "".join(f"{sha}:{path}\n" for path in relpaths).encode("utf-8")
        try:
            completed = subprocess.run(
                ["git", "cat-file", "--batch"], cwd=self.root,
                input=request, capture_output=True, check=True,
            )
        except FileNotFoundError as exc:
            raise GitError("git executable not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            raise GitError(f"git cat-file --batch failed: {detail}") from exc

        # Responses come back in request order, each either
        #   "<oid> <type> <size>\n" followed by <size> bytes and a newline, or
        #   "<spec> missing\n"
        # Sizes are authoritative: content is binary and may contain newlines, so
        # the stream cannot be split on them.
        out = completed.stdout
        found: dict[str, bytes] = {}
        cursor = 0
        for path in relpaths:
            newline = out.find(b"\n", cursor)
            if newline == -1:
                break
            header = out[cursor:newline].decode("utf-8", errors="replace")
            cursor = newline + 1
            fields = header.rsplit(" ", 2)
            if len(fields) != 3 or not fields[2].isdigit():
                continue  # "missing", "ambiguous": no body follows
            size = int(fields[2])
            found[path] = out[cursor:cursor + size]
            cursor += size + 1  # trailing newline after the body
        return found

    def changed_paths(self, commit: GitCommit) -> dict[str, str]:
        """Paths a commit touched, mapped to their status letter.

        ``A`` added, ``M`` modified, ``D`` deleted, ``R`` renamed. A root commit
        is diffed against the empty tree so its files appear as additions rather
        than as nothing.
        """
        if commit.is_root:
            args = ["show", "--name-status", "--format=", "--root", commit.sha]
        else:
            args = ["diff", "--name-status", f"{commit.parents[0]}..{commit.sha}"]
        changed: dict[str, str] = {}
        for line in self._run(*args).splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                # A rename reports both paths; the destination is what now exists.
                changed[parts[-1]] = parts[0][0]
        return changed


def _parse_commit(record: str) -> GitCommit | None:
    record = record.strip("\n")
    if not record.strip():
        return None
    fields = record.split(_FIELD)
    if len(fields) < 5:
        return None
    sha, parents, author, date, subject = fields[:5]
    body = fields[5] if len(fields) > 5 else ""
    return GitCommit(
        sha=sha.strip(),
        parents=tuple(p for p in parents.split() if p),
        author=author,
        date=datetime.fromisoformat(date),
        subject=subject,
        body=body.strip(),
    )
