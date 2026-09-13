# Observation Channels & Confidence Learning

Spec sections 61 and 62.

## The Three Post-Action Observation Channels

Spec section 61 specifies three channels to observe after an action or change $\Delta$ executes:
1. **Actual diff**: Ingested commit-by-commit via Git history (`mcm/reasoning/prediction.py`).
2. **Actual test result**: Observed before and after change execution via `TestRunnerObservationChannel`.
3. **Actual runtime behaviour**: Dynamically captured call graphs and execution paths via `RuntimeTraceObservationChannel`.

## Test Runner Observation Channel

`TestRunnerObservationChannel` compares predicted affected tests from MCM's impact analysis (`mcm.reasoning.dependency_propagation.analyse_impact`) against the actual test failures observed after applying a change:

```python
from mcm.reasoning.observation import TestRunnerObservationChannel

channel = TestRunnerObservationChannel()
comparison = channel.compare(
    predicted_affected_tests={"test_auth_success", "test_auth_failure"},
    before_outcomes={"test_auth_success": True, "test_auth_failure": True, "test_db": True},
    after_outcomes={"test_auth_success": False, "test_auth_failure": True, "test_db": False},
)

print(comparison.summary())
# Test Comparison: Precision=0.500, Recall=0.500, F1=0.500 | TP=1 FP=1 FN=1
```

- **True Positives (TP)**: Tests predicted to be affected that actually failed.
- **False Positives (FP)**: Tests predicted to be affected that still passed (change was backward-compatible or didn't trigger code paths).
- **False Negatives (FN)**: Tests that broke but were *not* predicted by MCM (reveals missing dependencies or emergent side effects).

## Runtime Execution Trace Observation Channel

`RuntimeTraceObservationChannel` uses dynamic instrumentation via `sys.settrace` to monitor execution within a repository's root directory, filtering out third-party libraries and standard library internals:

```python
from mcm.reasoning.observation import RuntimeTraceObservationChannel, reconcile_trace, update_confidence_from_trace

channel = RuntimeTraceObservationChannel(root_dir="path/to/repo")

with channel.trace(trace_id="workload-1") as trace:
    run_workload_or_tests()

reconciliation = reconcile_trace(store, trace)
print(reconciliation.summary())
# Trace Reconciliation: 12 confirmed, 3 dynamic-only discovered, 4 static unexercised
```

### Trace Reconciliation Categories

1. **Confirmed**: Static `CALLS` relations present in MCM's semantic core that were actively exercised during runtime execution.
2. **Discovered**: Dynamic function calls observed at runtime that static AST parsing could not discover (dynamic dispatch, reflection, plugin hooks, callback tables).
3. **Unexercised**: Static `CALLS` relations in the model that were not traversed during this workload.

## Learning from Prediction Error (Spec Section 62)

Section 62 specifies updating edge confidence based on empirical evidence:
- Confirmed relations receive empirical evidence boosting confidence toward 1.0 using the update formula:
  $$\text{Confidence}_{t+1} = \frac{C_t \cdot \alpha + 1.0}{\alpha + 1.0}$$
- Discovered relations are inserted into the store as newly learned relations with `ExtractionMethod.EXECUTION_TRACE` provenance and initial empirical confidence.
- Both operations record full immutable `Evidence` objects with `EvidenceType.RUNTIME_TRACE`.

```python
updated_count = update_confidence_from_trace(store, reconciliation, alpha=4.0)
```

## REST API Serving

Observation channels are exposed via FastAPI endpoints:
- `POST /observations/tests`: Computes precision, recall, and TP/FP/FN on test suites.
- `POST /observations/trace`: Ingests runtime dynamic execution traces, reconciles against stored static relations, and optionally updates edge confidences.
