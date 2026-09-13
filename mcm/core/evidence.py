"""Evidence records (spec section 15).

Every nontrivial fact points at the artifact that supports it. This is what makes
``Fact != Inference`` checkable rather than merely asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .ids import evidence_id
from .objects import utcnow


class EvidenceType(str, Enum):
    SOURCE_CODE = "SOURCE_CODE"
    AST = "AST"
    TEST = "TEST"
    COMMIT = "COMMIT"
    DOCUMENTATION = "DOCUMENTATION"
    USER = "USER"
    LLM_INFERENCE = "LLM_INFERENCE"
    STATIC_ANALYSIS = "STATIC_ANALYSIS"
    RUNTIME_TRACE = "RUNTIME_TRACE"
    HUMAN_CONFIRMATION = "HUMAN_CONFIRMATION"


@dataclass
class Evidence:
    id: str
    source_type: EvidenceType
    source_ref: str
    content: str
    extraction_method: str
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if not isinstance(self.source_type, EvidenceType):
            self.source_type = EvidenceType(self.source_type)

    @classmethod
    def create(
        cls,
        source_type: EvidenceType,
        source_ref: str,
        content: str,
        extraction_method: str,
        confidence: float = 1.0,
    ) -> "Evidence":
        return cls(
            id=evidence_id(source_type.value, source_ref, content),
            source_type=source_type,
            source_ref=source_ref,
            content=content,
            extraction_method=extraction_method,
            confidence=confidence,
        )
