"""FastAPI application for MCM (spec sections 41 and 42).

REST API endpoints for semantic core querying, impact analysis,
rule derivation, constraint checks, contradiction detection, and memory update.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ..agent.memory_update import EpistemicStatus, ItemCategory, MemoryItem, MemoryUpdater
from ..agent.query import QueryType, run_query
from ..core.objects import MCMObject, utcnow
from ..core.provenance import ExtractionMethod
from ..core.relations import RelationType as RT
from ..ingestion.repository import RepositoryIngestor
from ..reasoning.causal_reasoning import regression_candidates
from ..reasoning.change_propagation import propagate, propagation_json
from ..reasoning.constraint_checker import check_constraints, load_constraints
from ..reasoning.contradiction import contradiction_json, detect_contradictions
from ..reasoning.dependency_propagation import analyse_impact
from ..reasoning.engine import derive
from ..reasoning.observation import (
    RuntimeTraceObservation,
    TestRunnerObservationChannel,
    reconcile_trace,
    update_confidence_from_trace,
)
from ..reasoning.rules import load_rules
from ..retrieval.symbolic import resolve_one
from ..storage.database import Store
from ..storage.sqlite_store import SQLiteStore


# --- Pydantic Request/Response Models ---------------------------------------

class IngestRequest(BaseModel):
    path: str
    name: str | None = None


class ImpactRequest(BaseModel):
    target: str
    depth: int = Field(default=6, ge=1, le=20)
    as_of: datetime | None = None


class QueryRequest(BaseModel):
    reference: str
    mode: str = "impact"
    depth: int = Field(default=6, ge=1, le=20)
    as_of: datetime | None = None


class MemoryItemModel(BaseModel):
    category: str
    status: str
    subject: str
    content: str
    target: str | None = None
    relation_type: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_text: str | None = None
    source_ref: str = "api"
    method: str = "LLM_INFERENCE"


class MemoryBatchRequest(BaseModel):
    agent_name: str = "api-agent"
    items: list[MemoryItemModel]
    rebuild: bool = True


class TestObservationRequest(BaseModel):
    predicted_affected_tests: list[str]
    before_outcomes: dict[str, bool]
    after_outcomes: dict[str, bool]


class RuntimeTraceCallModel(BaseModel):
    caller: str
    callee: str
    count: int = 1


class RuntimeTraceReconcileRequest(BaseModel):
    trace_id: str = "api-trace"
    calls: list[RuntimeTraceCallModel]
    update_confidence: bool = False
    alpha: float = Field(default=4.0, ge=1.0, le=20.0)


def validate_repository_path(raw_path: str, allowed_roots: list[Path] | None = None) -> Path:
    """Validate that the given path is a safe, existing directory.

    Guards against arbitrary filesystem ingestion, system root directory ingestion,
    and path traversal.
    """
    try:
        resolved = Path(raw_path).resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path syntax: {raw_path}") from exc

    if not resolved.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {raw_path}")
    if not resolved.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {raw_path}")

    # Guard against ingesting root directories or dangerous OS system directories
    root_ancestors = {Path(p).resolve() for p in [
        "C:\\", "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
        "/", "/etc", "/var", "/usr", "/root", "/bin", "/sbin"
    ]}
    if resolved in root_ancestors:
        raise HTTPException(status_code=403, detail="Ingesting system root directories is prohibited")

    # If allowed roots are configured (or via MCM_ALLOWED_INGEST_ROOTS), enforce boundary
    env_roots = os.environ.get("MCM_ALLOWED_INGEST_ROOTS", "")
    roots = list(allowed_roots or [])
    if env_roots:
        roots.extend(Path(p.strip()).resolve() for p in env_roots.split(";") if p.strip())

    if roots:
        if not any(resolved == r or r in resolved.parents for r in roots):
            raise HTTPException(
                status_code=403,
                detail=f"Path {raw_path} is outside configured allowed ingestion roots",
            )

    return resolved


# --- Application Factory ---------------------------------------------------

def create_app(store: Store | None = None, allowed_ingest_roots: list[Path] | None = None) -> FastAPI:
    """Create a FastAPI app instance backed by the given or default store."""
    app = FastAPI(
        title="Mathematical Context Model (MCM) API",
        description="REST API for the MCM semantic memory and reasoning engine",
        version="0.1.0",
    )
    current_store = store or SQLiteStore()

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok", "system": "MCM"}

    # 1. Ingestion
    @app.post("/repositories/ingest")
    def ingest_repository(req: IngestRequest) -> dict[str, Any]:
        target_path = validate_repository_path(req.path, allowed_roots=allowed_ingest_roots)
        report = RepositoryIngestor(current_store).ingest(target_path, name=req.name)
        return {
            "repository_id": report.repository_id,
            "summary": report.summary(),
            "temporal_summary": report.temporal_summary(),
            "unresolved_count": len(report.unresolved),
        }

    # 2. Objects
    @app.get("/objects")
    def get_object_by_param(id: str = Query(...)) -> dict[str, Any]:
        object_id = unquote(id)
        obj = current_store.get_object(object_id)
        if obj is None:
            try:
                obj = resolve_one(current_store, object_id)
            except (KeyError, ValueError):
                raise HTTPException(status_code=404, detail=f"Object not found: {object_id}")
        return {
            "id": obj.id,
            "type": obj.type.value,
            "name": obj.name,
            "properties": obj.properties,
            "state": obj.state,
            "created_at": obj.created_at.isoformat() if obj.created_at else None,
            "updated_at": obj.updated_at.isoformat() if obj.updated_at else None,
        }

    @app.get("/objects/{object_id:path}/relations")
    def get_object_relations(object_id: str) -> dict[str, Any]:
        object_id = unquote(object_id)
        obj = current_store.get_object(object_id)
        if obj is None:
            try:
                obj = resolve_one(current_store, object_id)
            except (KeyError, ValueError):
                raise HTTPException(status_code=404, detail=f"Object not found: {object_id}")
        incoming = current_store.relations_for(obj.id, direction="incoming")
        outgoing = current_store.relations_for(obj.id, direction="outgoing")
        return {
            "object_id": obj.id,
            "incoming": [
                {"id": r.id, "type": r.relation_type.value, "arguments": r.arguments,
                 "confidence": r.confidence} for r in incoming
            ],
            "outgoing": [
                {"id": r.id, "type": r.relation_type.value, "arguments": r.arguments,
                 "confidence": r.confidence} for r in outgoing
            ],
        }

    @app.get("/objects/{object_id:path}")
    def get_object(object_id: str) -> dict[str, Any]:
        object_id = unquote(object_id)
        obj = current_store.get_object(object_id)
        if obj is None:
            # Fall back to symbolic resolution
            try:
                obj = resolve_one(current_store, object_id)
            except (KeyError, ValueError):
                raise HTTPException(status_code=404, detail=f"Object not found: {object_id}")
        return {
            "id": obj.id,
            "type": obj.type.value,
            "name": obj.name,
            "properties": obj.properties,
            "state": obj.state,
            "created_at": obj.created_at.isoformat() if obj.created_at else None,
            "updated_at": obj.updated_at.isoformat() if obj.updated_at else None,
        }

    # 3. Query
    @app.post("/query")
    def post_query(req: QueryRequest) -> dict[str, Any]:
        mode_map = {
            "impact": QueryType.IMPACT,
            "deps": QueryType.DEPENDENCY,
            "lookup": QueryType.LOOKUP,
            "equivalence": QueryType.EQUIVALENCE,
        }
        if req.mode.lower() not in mode_map:
            raise HTTPException(status_code=400, detail=f"Unsupported query mode: {req.mode}")
        try:
            return run_query(current_store, mode_map[req.mode.lower()], req.reference,
                             max_depth=req.depth, as_of=req.as_of)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # 4. Reason (Derive rules to fixpoint)
    @app.post("/reason")
    def post_reason(as_of: datetime | None = None) -> dict[str, Any]:
        rules = load_rules()
        result = derive(current_store, rules, as_of=as_of)
        return {
            "iterations": result.iterations,
            "complete": result.complete,
            "rule_counts": result.rule_counts,
            "derived_relations_count": len(result.derived),
        }

    # 5. Impact (Spec section 42)
    @app.post("/impact")
    def post_impact(req: ImpactRequest) -> dict[str, Any]:
        try:
            obj = resolve_one(current_store, req.target)
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail=f"Target object not found: {req.target}")

        res = analyse_impact(current_store, obj.id, max_depth=req.depth, as_of=req.as_of)
        return {
            "target": res.target.id,
            "direct_dependencies": [
                {"id": a.object.id, "name": a.object.name, "confidence": a.confidence}
                for a in res.direct
            ],
            "affected_objects": [
                {"id": a.object.id, "name": a.object.name, "confidence": a.confidence, "band": a.band}
                for a in res.all_affected
            ],
            "affected_tests": [
                {"id": t.object.id, "name": t.object.name, "confidence": t.confidence}
                for t in res.affected_tests
            ],
            "confidence": 1.0 if not res.all_affected else min(a.confidence for a in res.all_affected),
        }

    # 6. Constraints
    @app.post("/constraints/check")
    def post_constraints_check(as_of: datetime | None = None) -> list[dict[str, Any]]:
        constraints = load_constraints()
        results = check_constraints(current_store, constraints, as_of=as_of)
        return [
            {
                "name": r.constraint.name,
                "type": r.constraint.type.value,
                "verdict": r.verdict.value,
                "reason": r.reason,
                "checked": r.checked,
                "violations": [
                    {"message": v.message, "objects": v.objects, "evidence_ids": v.evidence_ids}
                    for v in r.violations
                ],
            }
            for r in results
        ]

    # 7. Contradictions (Spec section 31)
    @app.post("/contradictions")
    def post_contradictions(as_of: datetime | None = None) -> list[dict[str, Any]]:
        contradictions = detect_contradictions(current_store, as_of=as_of)
        return contradiction_json(contradictions)

    # 8. Memory (Spec section 32)
    @app.post("/memory")
    def post_memory(req: MemoryBatchRequest) -> dict[str, Any]:
        updater = MemoryUpdater(current_store, agent_name=req.agent_name)
        items: list[MemoryItem] = []
        for raw in req.items:
            try:
                rel_t = RT(raw.relation_type) if raw.relation_type else None
                items.append(MemoryItem(
                    category=ItemCategory(raw.category),
                    status=EpistemicStatus(raw.status),
                    subject=raw.subject,
                    content=raw.content,
                    target=raw.target,
                    relation_type=rel_t,
                    confidence=raw.confidence,
                    evidence_text=raw.evidence_text,
                    source_ref=raw.source_ref,
                    method=ExtractionMethod(raw.method),
                ))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid memory item: {exc}")

        report = updater.apply(items, rebuild=req.rebuild)
        return {
            "summary": report.summary(),
            "items_processed": report.items_processed,
            "objects_created": report.objects_created,
            "relations_created": report.relations_created,
            "evidence_created": report.evidence_created,
            "contradictions_count": len(report.contradictions),
        }

    # 9. History
    @app.get("/history/{object_id:path}")
    def get_history(object_id: str, depth: int = Query(default=6, ge=1, le=20)) -> dict[str, Any]:
        obj = current_store.get_object(object_id)
        if obj is None:
            try:
                obj = resolve_one(current_store, object_id)
            except (KeyError, ValueError):
                raise HTTPException(status_code=404, detail=f"Object not found: {object_id}")
        report = regression_candidates(current_store, obj.id, max_depth=depth)
        return {
            "target": report.target.id,
            "candidates_count": len(report.candidates),
            "candidates": [
                {
                    "commit_sha": c.commit.properties.get("sha", ""),
                    "commit_date": c.commit.properties.get("date", ""),
                    "changed_definitions": c.changed_definitions,
                }
                for c in report.candidates
            ],
        }

    # 10. Provenance
    @app.get("/provenance/{relation_id:path}")
    def get_provenance(relation_id: str) -> dict[str, Any]:
        rel = current_store.get_relation(relation_id)
        if rel is None:
            raise HTTPException(status_code=404, detail=f"Relation not found: {relation_id}")
        if not rel.provenance_id:
            raise HTTPException(status_code=404, detail="Relation carries no provenance record")
        prov = current_store.get_provenance(rel.provenance_id)
        if prov is None:
            raise HTTPException(status_code=404, detail="Provenance record missing")
        return {
            "id": prov.id,
            "method": prov.method.value,
            "agent": prov.agent,
            "source_ref": prov.source_ref,
            "source_reliability": prov.source_reliability,
            "created_at": prov.created_at.isoformat() if prov.created_at else None,
        }

    # 11. Observation Channels (Spec Sections 61 and 62)
    @app.post("/observations/tests")
    def post_test_observation(req: TestObservationRequest) -> dict[str, Any]:
        channel = TestRunnerObservationChannel(current_store)
        comp = channel.compare(
            predicted_affected_tests=set(req.predicted_affected_tests),
            before_outcomes=req.before_outcomes,
            after_outcomes=req.after_outcomes,
        )
        return {
            "precision": round(comp.precision, 4),
            "recall": round(comp.recall, 4),
            "f1": round(comp.f1, 4),
            "true_positives": sorted(comp.true_positives),
            "false_positives": sorted(comp.false_positives),
            "false_negatives": sorted(comp.false_negatives),
            "summary": comp.summary(),
        }

    @app.post("/observations/trace")
    def post_trace_observation(req: RuntimeTraceReconcileRequest) -> dict[str, Any]:
        trace = RuntimeTraceObservation(trace_id=req.trace_id, timestamp=utcnow())
        for c in req.calls:
            for _ in range(c.count):
                trace.add_call(c.caller, c.callee)

        reconciliation = reconcile_trace(current_store, trace)
        updated_count = 0
        if req.update_confidence:
            updated_count = update_confidence_from_trace(
                current_store, reconciliation, alpha=req.alpha
            )

        return {
            "summary": reconciliation.summary(),
            "confirmed_count": len(reconciliation.confirmed),
            "discovered_count": len(reconciliation.discovered),
            "unexercised_count": len(reconciliation.unexercised),
            "confirmed": [
                {"id": r.id, "arguments": r.arguments, "confidence": r.confidence}
                for r in reconciliation.confirmed
            ],
            "discovered": [
                {"caller": caller, "callee": callee, "count": count}
                for caller, callee, count in reconciliation.discovered
            ],
            "confidence_updated_count": updated_count,
        }

    return app
