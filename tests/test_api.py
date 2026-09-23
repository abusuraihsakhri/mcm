"""Tests for FastAPI serving layer (spec sections 41 and 42)."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from mcm.api.app import create_app
from mcm.ingestion.repository import RepositoryIngestor
from mcm.storage.sqlite_store import SQLiteStore

DEMO_REPO = Path(__file__).resolve().parent.parent / "examples" / "app"


@pytest.fixture
def client():
    store = SQLiteStore()
    RepositoryIngestor(store).ingest(DEMO_REPO, name="app")
    app = create_app(store)
    client = TestClient(app)
    yield client
    store.close()


class TestAPIEndpoints:
    def test_health(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json() == {"status": "ok", "system": "MCM"}

    def test_get_object_by_id_and_symbolic_resolution(self, client):
        from urllib.parse import quote
        # Full ID URL-encoded so '#' is not treated as fragment
        full_id = "repo://app/auth.py#function:authenticate"
        res = client.get(f"/objects/{quote(full_id, safe='')}")
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == "authenticate"
        assert data["type"] == "Function"

        # Via query parameter
        res_param = client.get("/objects", params={"id": full_id})
        assert res_param.status_code == 200
        assert res_param.json()["id"] == data["id"]

        # Symbolic reference resolution
        res2 = client.get("/objects/authenticate")
        assert res2.status_code == 200
        assert res2.json()["id"] == data["id"]

    def test_get_object_relations(self, client):
        res = client.get("/objects/authenticate/relations")
        assert res.status_code == 200
        data = res.json()
        assert "incoming" in data
        assert "outgoing" in data
        assert len(data["incoming"]) + len(data["outgoing"]) > 0

        object_id = data["object_id"]
        assert all(object_id in rel["arguments"][1:] for rel in data["incoming"])
        assert all(rel["arguments"][0] == object_id for rel in data["outgoing"])

    def test_post_impact_section_42_format(self, client):
        payload = {"target": "validate_token", "depth": 6}
        res = client.post("/impact", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "target" in data
        assert "direct_dependencies" in data
        assert "affected_objects" in data
        assert "affected_tests" in data
        assert "confidence" in data
        assert any("authenticate" in a["name"] for a in data["affected_objects"])

    def test_post_query(self, client):
        res = client.post("/query", json={"reference": "authenticate", "mode": "deps"})
        assert res.status_code == 200
        data = res.json()
        assert "depends_on" in data

    def test_constraints_check(self, client):
        res = client.post("/constraints/check")
        assert res.status_code == 200
        results = res.json()
        assert isinstance(results, list)
        assert len(results) > 0
        assert "verdict" in results[0]

    def test_contradictions_detection(self, client):
        res = client.post("/contradictions")
        assert res.status_code == 200
        assert isinstance(res.json(), list)

    def test_memory_update_endpoint(self, client):
        payload = {
            "agent_name": "test-agent",
            "items": [
                {
                    "category": "DECISION",
                    "status": "CONFIRMED",
                    "subject": "Cache auth tokens",
                    "content": "Store valid tokens in Redis for 10 minutes.",
                    "confidence": 0.95,
                    "method": "HUMAN",
                }
            ],
            "rebuild": True,
        }
        res = client.post("/memory", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["items_processed"] == 1
        assert data["objects_created"] >= 1
        assert data["contradictions_count"] == 0

    def test_not_found_errors(self, client):
        res = client.get("/objects/nonexistent_function_xyz")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

    def test_observation_tests_endpoint(self, client):
        payload = {
            "predicted_affected_tests": ["test_auth", "test_token"],
            "before_outcomes": {"test_auth": True, "test_token": True, "test_db": True},
            "after_outcomes": {"test_auth": False, "test_token": True, "test_db": False},
        }
        res = client.post("/observations/tests", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["true_positives"] == ["test_auth"]
        assert data["false_positives"] == ["test_token"]
        assert data["false_negatives"] == ["test_db"]
        assert data["precision"] == 0.5
        assert data["recall"] == 0.5

    def test_observation_trace_endpoint(self, client):
        payload = {
            "trace_id": "api-unit-trace",
            "calls": [
                {"caller": "repo://app/auth.py#function:authenticate",
                 "callee": "repo://app/auth.py#function:validate_token",
                 "count": 2},
                {"caller": "repo://app/auth.py#function:authenticate",
                 "callee": "repo://app/audit.py#function:log_access",
                 "count": 1},
            ],
            "update_confidence": True,
        }
        res = client.post("/observations/trace", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["confirmed_count"] >= 1
        assert data["discovered_count"] >= 1
        assert data["confidence_updated_count"] >= 1

    def test_path_validation_blocks_traversal_and_non_directory(self, client):
        # File path instead of directory
        res = client.post("/repositories/ingest", json={"path": str(DEMO_REPO / "auth.py")})
        assert res.status_code == 400
        assert "not a directory" in res.json()["detail"].lower()

        # Prohibited system root directory. Path("/") resolves to the host
        # filesystem root on POSIX and to the current drive root on Windows.
        res2 = client.post("/repositories/ingest", json={"path": str(Path("/"))})
        assert res2.status_code == 403
        assert "prohibited" in res2.json()["detail"].lower()
