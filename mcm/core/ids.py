"""Deterministic identifiers for MCM objects and derived records.

Spec sections 7 and 20: source-code entities get stable, human-readable IDs that
survive refactoring. Line numbers are recorded as *properties*, never as identity.

Object ID grammar::

    repo://<repo>                                  Repository
    repo://<repo>/<relpath>#directory              Directory
    repo://<repo>/<relpath>#file                   File
    repo://<repo>/<relpath>#function:<qualname>    Function
    repo://<repo>/<relpath>#class:<qualname>       Class
    repo://<repo>/<relpath>#method:<Class.name>    Method
    repo://<repo>/<relpath>#test:<qualname>        Test
    lib://<name>                                   External module
    commit://<repo>/<sha>                          Commit

Derived records (relations, evidence, provenance) are content-addressed so that
re-ingesting an unchanged repository is idempotent.
"""

from __future__ import annotations

import hashlib

_HASH_LEN = 16


def _digest(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:_HASH_LEN]


def repository_id(repo: str) -> str:
    return f"repo://{repo}"


def path_id(repo: str, relpath: str, kind: str) -> str:
    """ID for a filesystem entity. ``kind`` is ``file`` or ``directory``."""
    return f"repo://{repo}/{_norm(relpath)}#{kind}"


def symbol_id(repo: str, relpath: str, kind: str, qualname: str) -> str:
    """ID for a code symbol. ``kind`` is function/class/method/test."""
    return f"repo://{repo}/{_norm(relpath)}#{kind}:{qualname}"


def library_id(name: str) -> str:
    """ID for a module that is not resolvable inside the repository."""
    return f"lib://{name}"


def commit_id(repo: str, sha: str) -> str:
    """ID for a commit. The full SHA is already a stable content address."""
    return f"commit://{repo}/{sha}"


def relation_id(relation_type: str, arguments: list[str], method: str) -> str:
    """Content-addressed relation ID.

    Includes the extraction method so that the same claim reached by AST analysis
    and by LLM inference stays two distinct records with separate evidence
    (spec section 54).
    """
    return "rel:" + _digest(relation_type, method, *arguments)


def evidence_id(source_type: str, source_ref: str, content: str) -> str:
    return "ev:" + _digest(source_type, source_ref, content)


def provenance_id(method: str, agent: str, source_ref: str) -> str:
    return "prov:" + _digest(method, agent, source_ref)


def _norm(relpath: str) -> str:
    return relpath.replace("\\", "/").strip("/")
