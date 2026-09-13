"""End-to-End AI Agent Pair-Programming Simulation with MCM.

Simulates an autonomous AI agent using MCM's semantic core, REST API,
context minimisation, impact simulation, constraint validation, dynamic
runtime execution tracing, and contradiction detection.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from pprint import pprint

# Ensure utf-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient

from mcm.api.app import create_app
from mcm.core.objects import ObjectType
from mcm.reasoning.observation import RuntimeTraceObservationChannel
from mcm.retrieval.embedding import SentenceTransformerProvider, get_provider
from mcm.storage.sqlite_store import SQLiteStore

try:
    import torch
except ImportError:
    torch = None


def print_banner(title: str) -> None:
    print(f"\n{'=' * 75}")
    print(f"[*] AGENT STEP: {title}")
    print(f"{'=' * 75}")


def run_agent_simulation() -> None:
    start_total = time.perf_counter()
    app_dir = Path(__file__).resolve().parent.parent / "examples" / "app"

    # Step 0: Initialize MCM Semantic Core & FastAPI Server
    print_banner("1. Ingesting Target Repository into Semantic Core")
    store = SQLiteStore()
    app = create_app(store)
    client = TestClient(app)

    t0 = time.perf_counter()
    res = client.post("/repositories/ingest", json={"path": str(app_dir), "name": "auth-service"})
    ingest_time = (time.perf_counter() - t0) * 1000
    assert res.status_code == 200, res.text
    data = res.json()
    print(f"✓ Ingestion completed in {ingest_time:.1f}ms")
    print(f"  Summary: {data['summary']}")
    print(f"  Temporal Status: {data['temporal_summary']}")

    # Step 1: Agent receives prompt: "How does token validation work and what relies on it?"
    print_banner("2. Agent Queries Semantic Core for Context Minimisation")
    t0 = time.perf_counter()
    res = client.post("/query", json={"reference": "validate_token", "mode": "impact", "depth": 6})
    query_time = (time.perf_counter() - t0) * 1000
    assert res.status_code == 200, res.text
    q_data = res.json()
    print(f"✓ Query resolved in {query_time:.1f}ms")
    print(f"  Target: {q_data['target']['name']} ({q_data['target']['type']})")
    print(f"  Direct Dependents: {len(q_data['direct_dependents'])} direct callers")
    print(f"  Indirect Dependents: {len(q_data['indirect_dependents'])} downstream components")
    for item in q_data['direct_dependents'] + q_data['indirect_dependents']:
        obj = item['object']
        print(f"    - {obj['name']} ({obj['type']}) | Confidence: {item['confidence']:.2f} [{item['band']}] (depth {item['depth']})")

    # Step 2: Agent Pre-Action Simulation (Spec Section 42 Format)
    print_banner("3. Pre-Action Impact Simulation (Before Modifying validate_token)")
    t0 = time.perf_counter()
    res = client.post("/impact", json={"target": "validate_token", "depth": 6})
    impact_time = (time.perf_counter() - t0) * 1000
    assert res.status_code == 200, res.text
    impact = res.json()
    print(f"✓ Pre-action impact analyzed in {impact_time:.1f}ms")
    print(f"  Direct Dependencies: {[d['name'] for d in impact['direct_dependencies']]}")
    print(f"  All Affected Code Units: {[a['name'] for a in impact['affected_objects']]}")
    print(f"  Targeted Tests to Run: {[t['name'] for t in impact['affected_tests']]}")
    print(f"  Aggregated Confidence: {impact['confidence']:.2f}")

    # Step 3: Agent Verifies Architectural Constraints
    print_banner("4. Checking Architectural Invariants & Constraints")
    t0 = time.perf_counter()
    res = client.post("/constraints/check")
    const_time = (time.perf_counter() - t0) * 1000
    assert res.status_code == 200, res.text
    constraints = res.json()
    print(f"✓ Constraints checked in {const_time:.1f}ms")
    for c in constraints:
        print(f"  [{c['verdict']}] {c['name']} ({c['type']})")
        if c['violations']:
            for v in c['violations']:
                print(f"      Violation: {v['message']}")

    # Step 4: Observation Channel 1 - Dynamic Runtime Execution Tracing (Section 61)
    print_banner("5. Dynamic Runtime Execution Tracing & Confidence Learning (Spec 61 & 62)")
    tracer = RuntimeTraceObservationChannel(root_dir=app_dir)
    print(f"  Attaching runtime tracer to {app_dir}...")

    # Execute code in examples/app to trace calls
    import sys
    import types
    if "jwt" not in sys.modules:
        dummy_jwt = types.ModuleType("jwt")
        dummy_jwt.encode = lambda payload, *args, **kwargs: "mock.jwt.token"
        dummy_jwt.decode = lambda token, *args, **kwargs: {"sub": "alice"}
        sys.modules["jwt"] = dummy_jwt

    sys.path.insert(0, str(app_dir))
    try:
        import auth
        import user
        with tracer.trace("agent-runtime-run") as trace_obs:
            test_user = user.User("alice", active=True)
            auth.login("alice")
            auth.validate_token("sample-token")
        trace_summary = tracer.stop_trace()
    finally:
        if str(app_dir) in sys.path:
            sys.path.remove(str(app_dir))

    print(f"✓ Dynamic execution trace captured: {len(trace_summary.calls)} distinct call edges")
    call_payload = [
        {"caller": caller, "callee": callee, "count": count}
        for (caller, callee), count in trace_summary.calls.items()
    ]
    res = client.post("/observations/trace", json={
        "trace_id": "agent-dynamic-run",
        "calls": call_payload,
        "update_confidence": True,
        "alpha": 4.0
    })
    assert res.status_code == 200, res.text
    trace_res = res.json()
    print(f"  Trace Reconciliation: {trace_res['summary']}")
    print(f"  Confirmed Edges: {trace_res['confirmed_count']}")
    print(f"  Empirical Confidence Updates: {trace_res['confidence_updated_count']}")

    # Step 5: Observation Channel 2 - Test Execution Verification
    print_banner("6. Post-Action Test Verification Channel (Spec Section 61)")
    test_req = {
        "predicted_affected_tests": [t['name'] for t in impact['affected_tests']],
        "before_outcomes": {
            "test_authenticate_returns_username": True,
            "test_login_rejects_inactive_user": True,
            "test_db_unrelated": True,
        },
        "after_outcomes": {
            "test_authenticate_returns_username": False,  # Broke as predicted!
            "test_login_rejects_inactive_user": True,     # Passed
            "test_db_unrelated": True,                    # Passed
        },
    }
    res = client.post("/observations/tests", json=test_req)
    assert res.status_code == 200, res.text
    test_eval = res.json()
    print(f"✓ Test result comparison:")
    print(f"  Summary: {test_eval['summary']}")
    print(f"  True Positives (Accurately Predicted Breaks): {test_eval['true_positives']}")
    print(f"  Precision: {test_eval['precision']:.2f}, Recall: {test_eval['recall']:.2f}")

    # Step 6: Agent Updates Long-Term Epistemic Memory (Spec Section 32)
    print_banner("7. Ingesting Structured Agent Decisions & Memory Items")
    memory_req = {
        "agent_name": "pair-programming-agent",
        "items": [
            {
                "category": "DECISION",
                "status": "CONFIRMED",
                "subject": "Add expiration validation",
                "content": "Updated validate_token to check JWT exp claim and reject expired tokens.",
                "confidence": 1.0,
                "source_ref": "agent-plan.md",
                "method": "HUMAN",
            },
            {
                "category": "OBSERVATION",
                "status": "OBSERVED",
                "subject": "authenticate depends on validate_token",
                "content": "Runtime tracing verified authenticate invokes validate_token on every login.",
                "confidence": 1.0,
                "source_ref": "runtime-trace",
                "method": "EXECUTION_TRACE",
            }
        ],
        "rebuild": True
    }
    res = client.post("/memory", json=memory_req)
    assert res.status_code == 200, res.text
    mem_res = res.json()
    print(f"✓ Memory items ingested: {mem_res['items_processed']}")
    print(f"  Objects Created: {mem_res['objects_created']}")
    print(f"  Summary: {mem_res['summary']}")

    # Step 7: Contradiction Audit (Spec Section 31)
    print_banner("8. Contradiction Audit Across Knowledge Base")
    res = client.post("/contradictions")
    assert res.status_code == 200, res.text
    contra_list = res.json()
    print(f"✓ Contradictions detected: {len(contra_list)} (Epistemic consistency verified)")

    # Step 8: GPU Hardware Acceleration Status
    print_banner("9. Hardware GPU Engine Status & Acceleration")
    if torch is not None and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"✓ GPU Active: {gpu_name} ({vram:.2f} GB VRAM)")
        provider = get_provider("sentence-transformers")
        vec = provider.embed("authentication token security")
        print(f"✓ GPU Tensor Embeddings: generated {len(vec)}-dimensional vector on CUDA cores")
    else:
        print("ℹ GPU not available; fallback CPU engine active")

    total_time = time.perf_counter() - start_total
    print(f"\n{'=' * 75}")
    print(f"🎉 ALL AI AGENT TESTS COMPLETED SUCCESSFULLY IN {total_time:.2f}s")
    print(f"{'=' * 75}\n")
    store.close()


if __name__ == "__main__":
    run_agent_simulation()
