"""MCP server exposing MCM to coding agents over stdio.

Claude Code, Cursor, Codex, OpenCode and Antigravity all reach external tools
through the Model Context Protocol, so this is the surface that makes MCM usable
from an agent rather than from a shell. The REST API in ``mcm.api`` stays; it
answers a different question, which is how a service calls MCM over a network.

The transport is line-delimited JSON-RPC 2.0 on stdin and stdout. That is the
whole of MCP stdio framing, so it is implemented here rather than pulled in: the
package's only runtime dependencies are the parser and PyYAML, and an agent
integration is a bad reason to add a third.

**stdout carries protocol, stderr carries everything else.** A stray ``print``
into stdout corrupts the JSON-RPC stream and the client drops the connection with
no useful error, which is the single most common way a hand-written MCP server
fails. Every diagnostic here goes to stderr.

Run it directly for a smoke test::

    echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python -m mcm.mcp_server
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from .agent.context import build_context, package_json
from .agent.minimisation import minimise
from .core.change import Change, ChangeKind
from .ingestion.repository import RepositoryIngestor
from .reasoning.change_propagation import propagate, propagation_json
from .reasoning.constraint_checker import check_constraints, load_constraints
from .retrieval.embedding import get_provider
from .retrieval.hybrid import (HybridRetriever, RetrievalWeights, build_indexes,
                               result_json)
from .retrieval.symbolic import resolve_one
from .retrieval.vector import VectorProjection
from .storage.sqlite_store import SQLiteStore

#: MCP revisions this server implements. The framing and the four methods used
#: here are unchanged across them, so the client's choice is honoured when it
#: names one of these and the newest is offered otherwise.
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")

SERVER_INFO = {"name": "mcm", "version": "0.1.0"}


def log(message: str) -> None:
    """Diagnostics to stderr, never stdout."""
    print(f"[mcm-mcp] {message}", file=sys.stderr, flush=True)


# --- tool implementations ---------------------------------------------------
#
# Each takes the parsed arguments and returns a JSON-serialisable result. They
# open the store per call rather than holding one open: an agent session can run
# for hours, the database is a file another process may be rebuilding, and the
# cost of opening SQLite is far below the cost of the query that follows.


class ToolError(Exception):
    """A tool failed in a way the agent should see and can act on."""


def _store(db: str) -> SQLiteStore:
    if db != ":memory:" and not Path(db).exists():
        raise ToolError(
            f"No MCM index at {db!r}. Build one first with "
            f"`mcm ingest <repo-path> --db {db}`, or point MCM_DB at an existing index."
        )
    return SQLiteStore(db)


def tool_search(db: str, args: dict) -> dict:
    query = args.get("query")
    if not query:
        raise ToolError("'query' is required")
    limit = int(args.get("limit", 10))
    store = _store(db)
    try:
        weights = RetrievalWeights.parse(args["weights"]) if args.get("weights") else None
        retriever = HybridRetriever(
            store, weights=weights,
            vector=VectorProjection(store, provider=get_provider(args.get("provider"))),
        )
        payload = result_json(retriever.retrieve(query, limit=limit))
        # An unbuilt index and a query that genuinely matches nothing both come
        # back empty. An agent cannot tell those apart, and will usually conclude
        # the code does not exist. Say which one this is.
        if not payload.get("pool_size"):
            raise ToolError(
                "This index has no searchable content, so the query could not be "
                "answered (that is different from finding no match). Run mcm_ingest "
                "on the repository first."
            )
        return payload
    finally:
        store.close()


def tool_impact(db: str, args: dict) -> dict:
    target = args.get("target")
    if not target:
        raise ToolError("'target' is required")
    kind = str(args.get("change_kind", "SIGNATURE")).upper()
    try:
        change_kind = ChangeKind(kind)
    except ValueError:
        allowed = ", ".join(k.value for k in ChangeKind)
        raise ToolError(f"Unknown change_kind {kind!r}. Expected one of: {allowed}") from None
    store = _store(db)
    try:
        obj = resolve_one(store, target)
        change = Change(target_id=obj.id, kind=change_kind,
                        details=args.get("details") or {})
        result = propagate(store, change, max_depth=int(args.get("depth", 6)))
        return propagation_json(result)
    except KeyError as exc:
        raise ToolError(str(exc).strip('"')) from None
    finally:
        store.close()


def tool_context(db: str, args: dict) -> dict:
    task = args.get("task")
    if not task:
        raise ToolError("'task' is required")
    store = _store(db)
    try:
        package = build_context(store, task, focus=args.get("focus"),
                                max_depth=int(args.get("depth", 6)))
        if args.get("minimise", True):
            package = minimise(store, package, max_depth=int(args.get("depth", 6)))
        return package_json(store, package)
    except (KeyError, ValueError) as exc:
        raise ToolError(str(exc).strip('"')) from None
    finally:
        store.close()


def tool_constraints(db: str, args: dict) -> dict:
    store = _store(db)
    try:
        constraints = load_constraints(args.get("path"))
        results = check_constraints(store, constraints)
        return {
            "checked": len(results),
            "violations": [
                {"constraint": r.constraint.name, "severity": r.constraint.severity,
                 "satisfied": r.satisfied, "detail": r.explain()}
                for r in results if not r.satisfied
            ],
        }
    finally:
        store.close()


def tool_ingest(db: str, args: dict) -> dict:
    path = args.get("path")
    if not path:
        raise ToolError("'path' is required")
    if not Path(path).exists():
        raise ToolError(f"No such path: {path}")
    store = SQLiteStore(db)
    try:
        report = RepositoryIngestor(store).ingest(path, name=args.get("name"))
        # Build the retrieval projections in the same call. They are derived, so
        # this is not extra knowledge, but mcm_search is unusable without them and
        # an agent has no reason to guess that indexing is a second step.
        vector = VectorProjection(store, provider=get_provider(args.get("provider")))
        vector_report, lexical_report = build_indexes(store, vector=vector)
        return {
            "repository_id": report.repository_id,
            "summary": report.summary(),
            "vector_index": vector_report.summary(),
            "lexical_index": lexical_report.summary(),
            "parse_errors": report.parse_errors[:20],
        }
    finally:
        store.close()


#: name -> (handler, description, JSON Schema for arguments).
#:
#: Descriptions are written for the agent that reads them, not for a human
#: browsing docs: an agent picks a tool from this string alone, so each says what
#: the tool answers and when to reach for it.
TOOLS: dict[str, tuple[Callable[[str, dict], Any], str, dict]] = {
    "mcm_search": (
        tool_search,
        "Find the definitions in this codebase most relevant to a natural-language "
        "description, ranked by a hybrid of graph structure, vector similarity and "
        "lexical match. Use before editing unfamiliar code, instead of guessing file "
        "paths or grepping for names you are not sure exist.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "What you are looking for, in plain language."},
                "limit": {"type": "integer", "default": 10,
                          "description": "Maximum results to return."},
            },
            "required": ["query"],
        },
    ),
    "mcm_impact": (
        tool_impact,
        "Predict what a proposed change to one definition would break, BEFORE making "
        "it. Returns transitive callers, the tests that cover them, and a confidence "
        "per hop. Use when changing a function signature, renaming, or deleting "
        "anything that other code might call.",
        {
            "type": "object",
            "properties": {
                "target": {"type": "string",
                           "description": "The definition to change, by name or qualname "
                                          "(e.g. 'Signer.sign')."},
                "change_kind": {
                    "type": "string",
                    "enum": [k.value for k in ChangeKind],
                    "default": "SIGNATURE",
                    "description": "The kind of edit being considered.",
                },
                "depth": {"type": "integer", "default": 6,
                          "description": "Maximum hops to propagate."},
            },
            "required": ["target"],
        },
    ),
    "mcm_context": (
        tool_context,
        "Assemble the smallest set of facts sufficient to carry out a described task, "
        "each one traced to the source that supports it. Use at the start of a task to "
        "load context deliberately instead of reading whole files into the window.",
        {
            "type": "object",
            "properties": {
                "task": {"type": "string",
                         "description": "The task, in plain language."},
                "focus": {"type": "string",
                          "description": "Optional definition to centre the context on."},
                "minimise": {"type": "boolean", "default": True,
                             "description": "Drop facts the task does not entail."},
            },
            "required": ["task"],
        },
    ),
    "mcm_constraints": (
        tool_constraints,
        "Check the codebase against its declared architectural rules and report "
        "violations. Use after a change to confirm it did not breach a layering rule "
        "or an invariant the project declares.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "Constraint file. Defaults to the built-in set."},
            },
        },
    ),
    "mcm_ingest": (
        tool_ingest,
        "Index a repository into the MCM graph, or refresh an existing index after "
        "code has changed. Every other tool reads what this produces, so run it once "
        "per repository and again when the index is stale.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repository root to index."},
                "name": {"type": "string", "description": "Label for the repository."},
            },
            "required": ["path"],
        },
    ),
}


# --- JSON-RPC plumbing ------------------------------------------------------

def _tool_descriptors() -> list[dict]:
    return [{"name": name, "description": description, "inputSchema": schema}
            for name, (_, description, schema) in TOOLS.items()]


def handle(message: Any, db: str) -> dict | None:
    """Answer one JSON-RPC message, or return None when none is owed.

    A notification (a message with no ``id``) is acknowledged by silence; the
    spec forbids replying to one, and replying anyway is another way to desync a
    client that is strict about it.
    """
    if not isinstance(message, dict):
        return _err(None, -32600, "Request must be a JSON object")
    if message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
        return _err(None, -32600, "Expected JSON-RPC 2.0 and a method name")
    method = message.get("method")
    request_id = message.get("id")
    is_notification = request_id is None

    if is_notification:
        return None
    if not isinstance(message.get("params", {}), dict):
        return _err(request_id, -32602, "params must be an object")

    if method == "initialize":
        asked = (message.get("params") or {}).get("protocolVersion")
        version = asked if asked in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
        return _ok(request_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        })

    if is_notification:
        return None

    if method == "tools/list":
        return _ok(request_id, {"tools": _tool_descriptors()})

    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        if not isinstance(name, str) or not isinstance(params.get("arguments", {}), dict):
            return _err(request_id, -32602, "Expected a tool name and an arguments object")
        entry = TOOLS.get(name)
        if entry is None:
            return _err(request_id, -32602, f"Unknown tool: {name}")
        handler = entry[0]
        try:
            result = handler(db, params.get("arguments") or {})
        except ToolError as exc:
            return _ok(request_id, _content(str(exc), is_error=True))
        except Exception as exc:  # noqa: BLE001 - surface to the agent, keep serving
            log(f"tool {name} raised:\n{traceback.format_exc()}")
            return _ok(request_id, _content(f"{type(exc).__name__}: {exc}", is_error=True))
        return _ok(request_id, _content(json.dumps(result, indent=2, default=str)))

    if method == "ping":
        return _ok(request_id, {})

    return _err(request_id, -32601, f"Method not found: {method}")


def _content(text: str, *, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _ok(request_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _err(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def serve(db: str, stdin=None, stdout=None) -> None:
    """Read requests until stdin closes."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    log(f"serving {len(TOOLS)} tools against {db}")

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            _write(stdout, _err(None, -32700, f"Parse error: {exc}"))
            continue
        try:
            response = handle(message, db)
        except Exception as exc:  # noqa: BLE001 - a crash here kills the session
            log(f"dispatch failed:\n{traceback.format_exc()}")
            response = _err(message.get("id") if isinstance(message, dict) else None,
                            -32603, "Internal server error")
        if response is not None:
            _write(stdout, response)


def _write(stdout, payload: dict) -> None:
    stdout.write(json.dumps(payload) + "\n")
    stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mcm-mcp", description="Expose MCM to coding agents over MCP stdio")
    parser.add_argument("--db", default=os.environ.get("MCM_DB", "mcm.db"),
                        help="MCM index to serve (default: $MCM_DB, else mcm.db)")
    args = parser.parse_args(argv)
    try:
        serve(args.db)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
