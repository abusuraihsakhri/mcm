"""Observation channels and post-action learning (spec sections 61 and 62).

Spec section 61 specifies three observation channels after execution:
1. Actual diff (Git co-change, modeled in prediction.py)
2. Actual test result (TestRunnerObservationChannel)
3. Actual runtime behaviour (RuntimeTraceObservationChannel)

Spec section 62 specifies learning from prediction error:
- If MCM predicts A -> B and changing A affects B (or runtime confirms the call),
  increase confidence.
- If MCM predicts A -> B but changing A repeatedly leaves B unaffected,
  or if runtime tracing disproves the edge, attenuate confidence / decay.
- Confidence_{t+1} = Update(Confidence_t, Evidence_t).
"""

from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..algebra.confidence import DEFAULT_EDGE_DECAY, EDGE_DECAY
from ..core.evidence import Evidence, EvidenceType
from ..core.ids import evidence_id, relation_id
from ..core.objects import MCMObject, ObjectType, utcnow
from ..core.provenance import DEFAULT_SOURCE_RELIABILITY, ExtractionMethod, Provenance
from ..core.relations import MCMRelation, RelationType as RT
from ..storage.database import Store


# --- 1. Test Result Observation Channel (Section 61) ------------------------

@dataclass(frozen=True)
class TestOutcome:
    """The result of executing one test definition."""
    __test__ = False

    test_id: str
    passed: bool
    duration_ms: float = 0.0
    error_message: str | None = None
    tested_definitions: frozenset[str] = field(default_factory=frozenset)


@dataclass
class TestObservation:
    """Summary of actual test execution across a suite."""
    __test__ = False

    run_id: str
    timestamp: datetime
    outcomes: dict[str, TestOutcome] = field(default_factory=dict)

    @property
    def passed(self) -> set[str]:
        return {tid for tid, o in self.outcomes.items() if o.passed}

    @property
    def failed(self) -> set[str]:
        return {tid for tid, o in self.outcomes.items() if not o.passed}


@dataclass
class TestComparison:
    """Compares Predicted(Δ) affected tests against Observed(Δ) test failures."""
    __test__ = False

    predicted_affected: set[str]
    actual_failed: set[str]
    all_executed: set[str]

    @property
    def true_positives(self) -> set[str]:
        """Tests predicted to be affected that actually failed."""
        return self.predicted_affected & self.actual_failed

    @property
    def false_positives(self) -> set[str]:
        """Tests predicted to be affected that passed (e.g. change was compatible)."""
        return self.predicted_affected & (self.all_executed - self.actual_failed)

    @property
    def false_negatives(self) -> set[str]:
        """Tests that failed but were NOT predicted by MCM (unmodeled dependencies)."""
        return self.actual_failed - self.predicted_affected

    @property
    def precision(self) -> float:
        if not self.predicted_affected:
            return 1.0 if not self.actual_failed else 0.0
        return len(self.true_positives) / len(self.predicted_affected)

    @property
    def recall(self) -> float:
        if not self.actual_failed:
            return 1.0
        return len(self.true_positives) / len(self.actual_failed)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0.0:
            return 0.0
        return 2.0 * (p * r) / (p + r)

    def summary(self) -> str:
        return (f"Test Comparison: Precision={self.precision:.3f}, Recall={self.recall:.3f}, F1={self.f1:.3f} | "
                f"TP={len(self.true_positives)} FP={len(self.false_positives)} FN={len(self.false_negatives)}")


class TestRunnerObservationChannel:
    """Observes test suite execution and compares against predicted impact."""
    __test__ = False

    def __init__(self, store: Store | None = None) -> None:
        self.store = store

    def compare(
        self,
        predicted_affected_tests: set[str],
        before_outcomes: dict[str, bool],
        after_outcomes: dict[str, bool],
    ) -> TestComparison:
        """Compare tests that passed before but broke after a change."""
        broken: set[str] = set()
        executed: set[str] = set(after_outcomes.keys())
        for test_id, passed_after in after_outcomes.items():
            passed_before = before_outcomes.get(test_id, True)
            if passed_before and not passed_after:
                broken.add(test_id)

        return TestComparison(
            predicted_affected=set(predicted_affected_tests),
            actual_failed=broken,
            all_executed=executed,
        )


# --- 2. Runtime Execution Trace Observation Channel (Section 61) -------------

@dataclass(frozen=True)
class RuntimeCallEdge:
    """An observed dynamic call from caller to callee."""

    caller_id: str
    callee_id: str
    count: int = 1


@dataclass
class RuntimeTraceObservation:
    """The dynamic call graph observed during execution of a workload."""

    trace_id: str
    timestamp: datetime
    calls: dict[tuple[str, str], int] = field(default_factory=dict)
    invoked_functions: set[str] = field(default_factory=set)

    def add_call(self, caller: str, callee: str) -> None:
        self.invoked_functions.add(caller)
        self.invoked_functions.add(callee)
        key = (caller, callee)
        self.calls[key] = self.calls.get(key, 0) + 1

    @property
    def edges(self) -> list[RuntimeCallEdge]:
        return [RuntimeCallEdge(caller, callee, count) for (caller, callee), count in self.calls.items()]


class RuntimeTraceObservationChannel:
    """Dynamic call-graph tracer that captures execution paths using sys.settrace.

    Filters paths by root_dir to ignore third-party libraries and Python builtins.
    """

    def __init__(self, root_dir: Path | str, repo_prefix: str = "repo://") -> None:
        self.root_dir = Path(root_dir).resolve()
        self.repo_prefix = repo_prefix
        self._active_trace: RuntimeTraceObservation | None = None
        self._last_trace: RuntimeTraceObservation | None = None
        self._prev_trace_func: Any = None

    def _normalize_id(self, filename: str, func_name: str) -> str | None:
        try:
            file_path = Path(filename).resolve()
            rel = file_path.relative_to(self.root_dir)
            posix_rel = rel.as_posix()
            return f"{self.repo_prefix}{posix_rel}#function:{func_name}"
        except (ValueError, RuntimeError):
            return None

    def trace(self, trace_id: str = "runtime_trace") -> "RuntimeTraceContext":
        return RuntimeTraceContext(self, trace_id)

    def start_trace(self, trace_id: str = "runtime_trace") -> RuntimeTraceObservation:
        self._active_trace = RuntimeTraceObservation(trace_id=trace_id, timestamp=utcnow())
        self._prev_trace_func = sys.gettrace()
        norm_root = os.path.normcase(str(self.root_dir))

        def tracer(frame: Any, event: str, arg: Any) -> Any:
            if event == "call":
                code = frame.f_code
                filename = code.co_filename
                norm_file = os.path.normcase(os.path.abspath(filename))
                if not norm_file.startswith(norm_root):
                    return None
                callee_id = self._normalize_id(filename, code.co_name)
                if callee_id:
                    caller_frame = frame.f_back
                    if caller_frame:
                        caller_code = caller_frame.f_code
                        caller_id = self._normalize_id(caller_code.co_filename, caller_code.co_name)
                    else:
                        caller_id = None
                    if caller_id and callee_id and self._active_trace:
                        self._active_trace.add_call(caller_id, callee_id)
            return tracer

        sys.settrace(tracer)
        return self._active_trace

    def stop_trace(self) -> RuntimeTraceObservation:
        sys.settrace(self._prev_trace_func)
        res = self._active_trace or self._last_trace or RuntimeTraceObservation("empty", utcnow())
        self._last_trace = res
        self._active_trace = None
        return res


class RuntimeTraceContext:
    def __init__(self, channel: RuntimeTraceObservationChannel, trace_id: str) -> None:
        self.channel = channel
        self.trace_id = trace_id
        self.observation: RuntimeTraceObservation | None = None

    def __enter__(self) -> RuntimeTraceObservation:
        self.observation = self.channel.start_trace(self.trace_id)
        return self.observation

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.channel.stop_trace()


# --- 3. Dynamic Reconciliation & Confidence Learning (Section 62) -----------

@dataclass
class TraceReconciliation:
    """Results of comparing observed dynamic calls against stored static relations."""

    confirmed: list[MCMRelation]
    discovered: list[tuple[str, str, int]]
    unexercised: list[MCMRelation]

    def summary(self) -> str:
        return (f"Trace Reconciliation: {len(self.confirmed)} confirmed, "
                f"{len(self.discovered)} dynamic-only discovered, "
                f"{len(self.unexercised)} static unexercised")


def reconcile_trace(
    store: Store,
    trace: RuntimeTraceObservation,
) -> TraceReconciliation:
    """Compare observed dynamic calls against relations stored in MCM."""
    static_calls = [r for r in store.all_relations() if r.relation_type == RT.CALLS]
    static_map: dict[tuple[str, str], MCMRelation] = {}
    for r in static_calls:
        if len(r.arguments) >= 2:
            static_map[(r.arguments[0], r.arguments[1])] = r

    confirmed: list[MCMRelation] = []
    discovered: list[tuple[str, str, int]] = []
    unexercised_keys = set(static_map.keys())

    for (caller, callee), count in trace.calls.items():
        key = (caller, callee)
        if key in static_map:
            confirmed.append(static_map[key])
            unexercised_keys.discard(key)
        else:
            discovered.append((caller, callee, count))

    unexercised = [static_map[k] for k in unexercised_keys]
    return TraceReconciliation(confirmed=confirmed, discovered=discovered, unexercised=unexercised)


def update_confidence_from_trace(
    store: Store,
    reconciliation: TraceReconciliation,
    *,
    alpha: float = 4.0,
    agent_name: str = "runtime-tracer",
) -> int:
    """Spec Section 62: Update confidence based on dynamic execution evidence.

    Confirmed static edges receive empirical evidence boosting confidence toward 1.0.
    Discovered dynamic edges (e.g. dynamic dispatch, reflection) are created as new
    relations in the store with ExtractionMethod.EXECUTION_TRACE provenance.
    """
    updated_count = 0
    now = utcnow()

    # 1. Boost confirmed relations
    prov = Provenance.create(
        method=ExtractionMethod.EXECUTION_TRACE,
        agent=agent_name,
        source_ref=f"trace-run@{now.isoformat()}",
        source_reliability=1.0,
    )
    store.put_provenance(prov)

    for rel in reconciliation.confirmed:
        # Confidence update: Confidence_{t+1} = (C_t * alpha + 1.0) / (alpha + 1)
        new_conf = round((rel.confidence * alpha + 1.0) / (alpha + 1.0), 4)
        if new_conf != rel.confidence:
            ev = Evidence.create(
                source_type=EvidenceType.RUNTIME_TRACE,
                source_ref=prov.source_ref,
                content=f"Dynamic runtime execution trace confirmed {rel.arguments[0]} -> {rel.arguments[1]}",
                extraction_method=ExtractionMethod.EXECUTION_TRACE.value,
                confidence=1.0,
            )
            store.put_evidence(ev)
            updated_rel = MCMRelation(
                id=rel.id,
                relation_type=rel.relation_type,
                arguments=rel.arguments,
                confidence=new_conf,
                evidence_ids=list(rel.evidence_ids) + [ev.id],
                valid_from=rel.valid_from,
                valid_until=rel.valid_until,
                provenance_id=prov.id,
            )
            store.put_relation(updated_rel)
            updated_count += 1

    # 2. Ingest newly discovered dynamic relations
    for caller, callee, count in reconciliation.discovered:
        r_id = relation_id(RT.CALLS.value, caller, callee)
        ev = Evidence.create(
            source_type=EvidenceType.RUNTIME_TRACE,
            source_ref=prov.source_ref,
            content=f"Observed {count} dynamic runtime calls from {caller} to {callee}",
            extraction_method=ExtractionMethod.EXECUTION_TRACE.value,
            confidence=0.95,
        )
        store.put_evidence(ev)
        rel = MCMRelation(
            id=r_id,
            relation_type=RT.CALLS,
            arguments=[caller, callee],
            confidence=0.95,
            evidence_ids=[ev.id],
            valid_from=now,
            provenance_id=prov.id,
        )
        store.put_relation(rel)
        updated_count += 1

    return updated_count
