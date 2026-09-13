"""Tests for observation channels and post-action updates (spec sections 61 and 62)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mcm.core.evidence import Evidence
from mcm.core.objects import MCMObject, ObjectType, utcnow
from mcm.core.provenance import ExtractionMethod, Provenance
from mcm.core.relations import MCMRelation, RelationType as RT
from mcm.reasoning.observation import (
    RuntimeTraceObservationChannel,
    TestRunnerObservationChannel,
    reconcile_trace,
    update_confidence_from_trace,
)
from mcm.storage.sqlite_store import SQLiteStore


def test_test_runner_observation_channel_calculates_metrics():
    channel = TestRunnerObservationChannel()
    predicted = {"test_auth_success", "test_auth_failure", "test_profile"}
    before = {
        "test_auth_success": True,
        "test_auth_failure": True,
        "test_profile": True,
        "test_db_ping": True,
    }
    after = {
        "test_auth_success": False,  # Broke (TP)
        "test_auth_failure": True,   # Survived (FP)
        "test_profile": False,       # Broke (TP)
        "test_db_ping": False,       # Broke unexpectedly (FN)
    }

    comp = channel.compare(predicted, before, after)
    assert comp.true_positives == {"test_auth_success", "test_profile"}
    assert comp.false_positives == {"test_auth_failure"}
    assert comp.false_negatives == {"test_db_ping"}
    assert comp.precision == pytest.approx(2 / 3)
    assert comp.recall == pytest.approx(2 / 3)
    assert comp.f1 == pytest.approx(2 / 3)
    assert "Precision=" in comp.summary()


def test_runtime_trace_observation_channel_captures_function_calls(tmp_path):
    channel = RuntimeTraceObservationChannel(root_dir=tmp_path)

    # Define test functions whose code objects live inside tmp_path
    mod_file = tmp_path / "service.py"
    code = """
def callee():
    return 42

def caller():
    return callee() + 1
"""
    mod_file.write_text(code, encoding="utf-8")

    # Execute and compile with filename inside tmp_path
    namespace: dict = {}
    compiled = compile(code, str(mod_file), "exec")
    exec(compiled, namespace)

    with channel.trace(trace_id="unit_run") as trace:
        result = namespace["caller"]()

    assert result == 43
    caller_id = f"repo://service.py#function:caller"
    callee_id = f"repo://service.py#function:callee"

    assert (caller_id, callee_id) in trace.calls
    assert trace.calls[(caller_id, callee_id)] >= 1


def test_reconcile_and_update_confidence_from_trace(tmp_path):
    store = SQLiteStore()
    now = utcnow()

    caller_id = "repo://app.py#function:login"
    callee_static = "repo://auth.py#function:verify"
    callee_dynamic = "repo://audit.py#function:log_event"

    store.put_object(MCMObject(id=caller_id, type=ObjectType.FUNCTION, name="login"))
    store.put_object(MCMObject(id=callee_static, type=ObjectType.FUNCTION, name="verify"))
    store.put_object(MCMObject(id=callee_dynamic, type=ObjectType.FUNCTION, name="log_event"))

    # Add initial static CALLS relation with confidence 0.70
    static_rel = MCMRelation(
        id="rel_calls_1",
        relation_type=RT.CALLS,
        arguments=[caller_id, callee_static],
        confidence=0.70,
        valid_from=now,
    )
    store.put_relation(static_rel)

    # Simulate runtime trace
    channel = RuntimeTraceObservationChannel(root_dir=tmp_path)
    trace = channel.start_trace("simulated")
    trace.add_call(caller_id, callee_static)
    trace.add_call(caller_id, callee_dynamic)
    channel.stop_trace()

    reconciliation = reconcile_trace(store, trace)
    assert len(reconciliation.confirmed) == 1
    assert reconciliation.confirmed[0].id == "rel_calls_1"
    assert len(reconciliation.discovered) == 1
    assert reconciliation.discovered[0][0] == caller_id
    assert reconciliation.discovered[0][1] == callee_dynamic

    # Update confidence from trace
    updated = update_confidence_from_trace(store, reconciliation, alpha=4.0)
    assert updated == 2

    # Verify static relation confidence was boosted
    updated_static = store.get_relation("rel_calls_1")
    assert updated_static is not None
    # (0.70 * 4 + 1.0) / 5 = 3.8 / 5 = 0.76
    assert updated_static.confidence == pytest.approx(0.76)

    # Verify discovered relation was created
    new_rels = [r for r in store.all_relations() if r.relation_type == RT.CALLS]
    assert len(new_rels) == 2
    dynamic_rel = [r for r in new_rels if r.arguments == [caller_id, callee_dynamic]][0]
    assert dynamic_rel.confidence == 0.95

    # Check evidence attached to relation
    assert len(updated_static.evidence_ids) >= 1
    ev = store.get_evidence(updated_static.evidence_ids[0])
    assert ev is not None
    assert "Dynamic runtime execution trace" in ev.content

    store.close()
