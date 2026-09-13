"""Agent memory update and reconciliation (spec section 32).

After every important agent interaction:
1. Extract: facts, decisions, constraints, observations, changes, errors, solutions
2. Classify: observed, inferred, hypothesized, confirmed, rejected
3. Attach provenance and evidence.
4. Update the canonical semantic state.
5. Rebuild affected derived views.

Do NOT store raw conversational tokens: Compression + Structure + Recoverability.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..core.evidence import Evidence, EvidenceType
from ..core.ids import evidence_id, provenance_id, relation_id
from ..core.objects import MCMObject, ObjectType, utcnow
from ..core.provenance import DEFAULT_SOURCE_RELIABILITY, ExtractionMethod, Provenance
from ..core.relations import MCMRelation, RelationType as RT
from ..reasoning.contradiction import Contradiction, detect_contradictions
from ..retrieval.hybrid import build_indexes
from ..storage.database import Store


class ItemCategory(str, Enum):
    FACT = "FACT"
    DECISION = "DECISION"
    CONSTRAINT = "CONSTRAINT"
    OBSERVATION = "OBSERVATION"
    CHANGE = "CHANGE"
    ERROR = "ERROR"
    SOLUTION = "SOLUTION"


class EpistemicStatus(str, Enum):
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    HYPOTHESIZED = "HYPOTHESIZED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


_CATEGORY_OBJECT_TYPES: dict[ItemCategory, ObjectType] = {
    ItemCategory.DECISION: ObjectType.DECISION,
    ItemCategory.OBSERVATION: ObjectType.OBSERVATION,
    ItemCategory.ERROR: ObjectType.BUG,
    ItemCategory.CONSTRAINT: ObjectType.REQUIREMENT,
    ItemCategory.FACT: ObjectType.CONCEPT,
    ItemCategory.CHANGE: ObjectType.CONCEPT,
    ItemCategory.SOLUTION: ObjectType.CONCEPT,
}

_STATUS_RELATIONS: dict[EpistemicStatus, RT] = {
    EpistemicStatus.OBSERVED: RT.OBSERVED,
    EpistemicStatus.INFERRED: RT.INFERRED,
    EpistemicStatus.HYPOTHESIZED: RT.ASSUMED,
    EpistemicStatus.CONFIRMED: RT.CONFIRMED,
    EpistemicStatus.REJECTED: RT.REJECTED,
}


@dataclass
class MemoryItem:
    """One extracted, structured knowledge unit from an agent interaction."""

    category: ItemCategory
    status: EpistemicStatus
    subject: str
    content: str
    target: str | None = None
    relation_type: RT | None = None
    confidence: float = 1.0
    evidence_text: str | None = None
    source_ref: str = "interaction"
    method: ExtractionMethod = ExtractionMethod.LLM_INFERENCE

    def __post_init__(self) -> None:
        if not isinstance(self.category, ItemCategory):
            self.category = ItemCategory(self.category)
        if not isinstance(self.status, EpistemicStatus):
            self.status = EpistemicStatus(self.status)
        if not isinstance(self.method, ExtractionMethod):
            self.method = ExtractionMethod(self.method)


@dataclass
class MemoryUpdateReport:
    items_processed: int = 0
    objects_created: int = 0
    relations_created: int = 0
    evidence_created: int = 0
    contradictions: list[Contradiction] = field(default_factory=list)
    projections_rebuilt: bool = False

    def summary(self) -> str:
        lines = [
            f"Memory Update Complete: {self.items_processed} items processed",
            f"  Objects created:   {self.objects_created}",
            f"  Relations created: {self.relations_created}",
            f"  Evidence attached: {self.evidence_created}",
            f"  Projections rebuilt: {self.projections_rebuilt}",
        ]
        if self.contradictions:
            lines.append(f"  WARNING: {len(self.contradictions)} contradiction(s) detected!")
        else:
            lines.append("  Semantic store state: Coherent (0 contradictions)")
        return "\n".join(lines)


class MemoryUpdater:
    """Updates the canonical semantic store from agent interactions."""

    def __init__(self, store: Store, agent_name: str = "coding-agent") -> None:
        self.store = store
        self.agent_name = agent_name

    def apply(
        self,
        items: list[MemoryItem],
        *,
        rebuild: bool = True,
        timestamp: datetime | None = None,
    ) -> MemoryUpdateReport:
        """Commit memory items to the semantic store."""
        report = MemoryUpdateReport(items_processed=len(items))
        now = timestamp or utcnow()

        for item in items:
            self._process_item(item, report, now)

        # Contradiction check: ensures newly added claims do not conflict silently
        report.contradictions = detect_contradictions(self.store, as_of=now)

        # Rebuild derived views (vector and lexical projections)
        if rebuild:
            build_indexes(self.store, as_of=now)
            report.projections_rebuilt = True

        return report

    def _process_item(self, item: MemoryItem, report: MemoryUpdateReport, now: datetime) -> None:
        # 1. Attach Provenance
        reliability = DEFAULT_SOURCE_RELIABILITY.get(item.method, 0.8)
        prov = Provenance.create(
            method=item.method,
            agent=self.agent_name,
            source_ref=item.source_ref,
            source_reliability=reliability,
        )
        prov.created_at = now
        self.store.put_provenance(prov)

        # 2. Attach Evidence if provided
        ev_id = None
        if item.evidence_text:
            ev = Evidence.create(
                source_type=EvidenceType.LLM_INFERENCE if item.method == ExtractionMethod.LLM_INFERENCE else EvidenceType.USER,
                source_ref=item.source_ref,
                content=item.evidence_text,
                extraction_method=item.method.value,
                confidence=item.confidence,
            )
            ev.timestamp = now
            self.store.put_evidence(ev)
            ev_id = ev.id
            report.evidence_created += 1

        # 3. Create or resolve the subject object
        subject_id = self._ensure_object(item.subject, item.category, item.content, report, now)

        # 4. If a target is specified, link them with a typed relation
        if item.target:
            target_id = self._ensure_object(item.target, ItemCategory.FACT, "", report, now)
            rel_type = item.relation_type or _STATUS_RELATIONS[item.status]
            rel_id = f"rel://mem/{hashlib.blake2b(f'{subject_id}:{target_id}:{rel_type.value}:{now.isoformat()}'.encode(), digest_size=8).hexdigest()}"
            rel = MCMRelation(
                id=rel_id,
                relation_type=rel_type,
                arguments=[subject_id, target_id],
                confidence=item.confidence,
                properties={"category": item.category.value, "status": item.status.value},
                evidence_ids=[ev_id] if ev_id else [],
                provenance_id=prov.id,
                valid_from=now,
            )
            self.store.put_relation(rel)
            report.relations_created += 1
        else:
            # Self-status relation (e.g. subject is CONFIRMED or DECIDED)
            rel_type = _STATUS_RELATIONS[item.status]
            rel_id = f"rel://mem/{hashlib.blake2b(f'{subject_id}:{rel_type.value}:{now.isoformat()}'.encode(), digest_size=8).hexdigest()}"
            rel = MCMRelation(
                id=rel_id,
                relation_type=rel_type,
                arguments=[subject_id],
                confidence=item.confidence,
                properties={"content": item.content, "category": item.category.value},
                evidence_ids=[ev_id] if ev_id else [],
                provenance_id=prov.id,
                valid_from=now,
            )
            self.store.put_relation(rel)
            report.relations_created += 1

    def _ensure_object(
        self,
        ref_or_name: str,
        category: ItemCategory,
        details: str,
        report: MemoryUpdateReport,
        now: datetime,
    ) -> str:
        # Check if already a full URI or exists
        existing = self.store.get_object(ref_or_name)
        if existing is not None:
            return existing.id

        # Generate stable object URI
        obj_type = _CATEGORY_OBJECT_TYPES.get(category, ObjectType.CONCEPT)
        name_slug = ref_or_name.replace(" ", "_").lower()
        obj_id = f"{obj_type.value.lower()}://agent/{name_slug}"
        obj = self.store.get_object(obj_id)
        if obj is None:
            obj = MCMObject(
                id=obj_id,
                type=obj_type,
                name=ref_or_name,
                properties={"details": details, "created_by": self.agent_name},
                created_at=now,
                updated_at=now,
            )
            self.store.put_object(obj)
            report.objects_created += 1
        return obj_id
