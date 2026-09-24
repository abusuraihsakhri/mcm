"""Tests for the MCP stdio server (mcm/mcp_server.py).

Two things are being protected here. The first is the JSON-RPC contract: a client
that gets malformed framing drops the connection with no diagnosis, so the shape
of every response matters as much as its content. The second is that a failing
tool reports the failure *to the agent* as an error result, rather than raising
and killing the session.
"""

from __future__ import annotations

import io
import json

import pytest

from mcm.ingestion.repository import RepositoryIngestor
from mcm.mcp_server import (SUPPORTED_PROTOCOLS, TOOLS, ToolError, handle,
                            serve, tool_context, tool_impact, tool_search)
from mcm.retrieval.embedding import get_provider
from mcm.retrieval.hybrid import build_indexes
from mcm.retrieval.vector import VectorProjection
from mcm.storage.sqlite_store import SQLiteStore

from conftest import DEMO_REPO


@pytest.fixture(scope="module")
def indexed_db(tmp_path_factory):
    """A real on-disk index, because the tools open the database by path."""
    path = tmp_path_factory.mktemp("mcp") / "index.db"
    store = SQLiteStore(path)
    RepositoryIngestor(store).ingest(DEMO_REPO, name="app")
    build_indexes(store, vector=VectorProjection(store, provider=get_provider(None)))
    store.close()
    return str(path)


def call(name: str, arguments: dict, db: str) -> dict:
    """Drive one tool through the dispatcher, as a client would."""
    return handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": name, "arguments": arguments}}, db)


class TestProtocol:
    def test_initialize_echoes_a_supported_protocol_version(self):
        response = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                           "params": {"protocolVersion": "2025-03-26"}}, ":memory:")
        assert response["result"]["protocolVersion"] == "2025-03-26"
        assert response["result"]["serverInfo"]["name"] == "mcm"

    def test_initialize_falls_back_when_the_client_asks_for_an_unknown_version(self):
        response = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                           "params": {"protocolVersion": "1999-01-01"}}, ":memory:")
        assert response["result"]["protocolVersion"] in SUPPORTED_PROTOCOLS

    def test_notifications_are_answered_with_silence(self):
        """Replying to a notification desyncs strict clients."""
        assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"},
                      ":memory:") is None

    def test_tools_list_advertises_every_tool_with_a_schema(self):
        response = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, ":memory:")
        listed = response["result"]["tools"]
        assert {t["name"] for t in listed} == set(TOOLS)
        for tool in listed:
            assert tool["description"].strip()
            assert tool["inputSchema"]["type"] == "object"

    def test_unknown_method_is_a_jsonrpc_error(self):
        response = handle({"jsonrpc": "2.0", "id": 3, "method": "no/such"}, ":memory:")
        assert response["error"]["code"] == -32601

    def test_unknown_tool_is_a_jsonrpc_error(self):
        response = call("mcm_nonexistent", {}, ":memory:")
        assert response["error"]["code"] == -32602

    def test_every_response_carries_the_request_id(self):
        for method in ("initialize", "tools/list", "ping"):
            response = handle({"jsonrpc": "2.0", "id": 77, "method": method}, ":memory:")
            assert response["id"] == 77
            assert response["jsonrpc"] == "2.0"


class TestServeLoop:
    @pytest.mark.parametrize("payload", [None, [], 42, "text", True,
        {"jsonrpc": "1.0", "id": 2, "method": "ping"},
        {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": []},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": []}}])
    def test_invalid_request_does_not_drop_stream(self, payload):
        out = io.StringIO()
        incoming = json.dumps(payload) + '\n' + json.dumps(
            {"jsonrpc": "2.0", "id": 3, "method": "ping"}) + '\n'
        serve(":memory:", stdin=io.StringIO(incoming), stdout=out)
        first, second = map(json.loads, out.getvalue().splitlines())
        assert first["error"]["code"] in {-32600, -32602}
        assert second == {"jsonrpc": "2.0", "id": 3, "result": {}}

    def test_initialize_notification_is_silent(self):
        assert handle({"jsonrpc": "2.0", "method": "initialize"}, ":memory:") is None

    def test_malformed_json_is_reported_without_dropping_the_stream(self):
        out = io.StringIO()
        serve(":memory:", stdin=io.StringIO('not json\n{"jsonrpc":"2.0","id":1,"method":"ping"}\n'),
              stdout=out)
        first, second = [json.loads(line) for line in out.getvalue().splitlines()]
        assert first["error"]["code"] == -32700
        assert second["id"] == 1  # the loop kept serving

    def test_blank_lines_are_skipped(self):
        out = io.StringIO()
        serve(":memory:", stdin=io.StringIO('\n\n{"jsonrpc":"2.0","id":5,"method":"ping"}\n'),
              stdout=out)
        assert len(out.getvalue().splitlines()) == 1

    def test_every_line_written_is_exactly_one_json_object(self):
        """Framing is line-delimited; an embedded newline would split a message."""
        out = io.StringIO()
        serve(":memory:", stdin=io.StringIO('{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n'),
              stdout=out)
        for line in out.getvalue().splitlines():
            json.loads(line)


class TestTools:
    def test_search_finds_the_definition_a_description_refers_to(self, indexed_db):
        result = tool_search(indexed_db, {"query": "verify an authentication token",
                                          "limit": 5})
        ids = [r["object_id"] for r in result["results"]]
        assert any("validate_token" in i for i in ids)

    def test_impact_reports_callers_of_a_changed_function(self, indexed_db):
        result = tool_impact(indexed_db, {"target": "validate_token",
                                          "change_kind": "SIGNATURE"})
        touched = json.dumps(result)
        assert "authenticate" in touched

    def test_context_centres_on_the_definition_the_task_names(self, indexed_db):
        result = tool_context(indexed_db, {"task": "add an expiry check to token validation"})
        assert "validate_token" in result["focus"]["id"]

    def test_an_unresolvable_target_is_an_error_result_not_a_crash(self, indexed_db):
        response = call("mcm_impact", {"target": "does_not_exist_anywhere"}, indexed_db)
        assert response["result"]["isError"] is True
        assert "does_not_exist_anywhere" in response["result"]["content"][0]["text"]

    def test_a_missing_index_explains_how_to_build_one(self, tmp_path):
        with pytest.raises(ToolError, match="mcm ingest"):
            tool_search(str(tmp_path / "absent.db"), {"query": "anything"})

    def test_search_on_an_empty_index_says_so_rather_than_returning_nothing(self, tmp_path):
        """Nothing-indexed and nothing-matched are different answers."""
        path = tmp_path / "empty.db"
        SQLiteStore(path).close()
        with pytest.raises(ToolError, match="no searchable content"):
            tool_search(str(path), {"query": "anything"})

    def test_a_missing_required_argument_is_reported(self, indexed_db):
        response = call("mcm_search", {}, indexed_db)
        assert response["result"]["isError"] is True

    def test_an_unknown_change_kind_lists_the_valid_ones(self, indexed_db):
        with pytest.raises(ToolError, match="SIGNATURE"):
            tool_impact(indexed_db, {"target": "validate_token", "change_kind": "WOBBLE"})

    def test_ingest_builds_the_indexes_search_needs(self, tmp_path):
        db = str(tmp_path / "fresh.db")
        response = call("mcm_ingest", {"path": str(DEMO_REPO), "name": "app"}, db)
        assert response["result"]["isError"] is False
        # Search works immediately afterwards, without a separate index step.
        assert tool_search(db, {"query": "authentication"})["pool_size"] > 0

    def test_tool_results_are_json_serialisable_text(self, indexed_db):
        response = call("mcm_search", {"query": "token", "limit": 2}, indexed_db)
        json.loads(response["result"]["content"][0]["text"])
