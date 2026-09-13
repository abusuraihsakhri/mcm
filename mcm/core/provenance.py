"""Provenance records (spec section 16).

Answers "where did this come from?" for every relation. Kept separate from
Evidence because the two carry different uncertainty dimensions (spec section 17):
Evidence carries *evidence strength*, Provenance carries *source reliability*.
They are deliberately not multiplied together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .ids import provenance_id
from .objects import utcnow


class ExtractionMethod(str, Enum):
    AST = "AST"
    STATIC_ANALYSIS = "STATIC_ANALYSIS"
    FILESYSTEM = "FILESYSTEM"
    GIT = "GIT"
    LLM_INFERENCE = "LLM_INFERENCE"
    SYMBOLIC_INFERENCE = "SYMBOLIC_INFERENCE"
    HUMAN = "HUMAN"
    EXECUTION_TRACE = "EXECUTION_TRACE"
    TEST_RUNNER = "TEST_RUNNER"


#: Reliability of a source, independent of any particular claim it supports.
#: Deterministic parsers and runtime dynamic traces are treated as near-perfect; LLM inference is not.
DEFAULT_SOURCE_RELIABILITY: dict[ExtractionMethod, float] = {
    ExtractionMethod.AST: 1.0,
    ExtractionMethod.STATIC_ANALYSIS: 0.95,
    ExtractionMethod.FILESYSTEM: 1.0,
    ExtractionMethod.GIT: 1.0,
    ExtractionMethod.SYMBOLIC_INFERENCE: 0.9,
    ExtractionMethod.HUMAN: 0.9,
    ExtractionMethod.LLM_INFERENCE: 0.6,
    ExtractionMethod.EXECUTION_TRACE: 1.0,
    ExtractionMethod.TEST_RUNNER: 1.0,
}


@dataclass
class Provenance:
    id: str
    method: ExtractionMethod
    agent: str
    source_ref: str
    source_reliability: float = 1.0
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if not isinstance(self.method, ExtractionMethod):
            self.method = ExtractionMethod(self.method)

    @classmethod
    def create(
        cls,
        method: ExtractionMethod,
        agent: str,
        source_ref: str,
        source_reliability: float | None = None,
    ) -> "Provenance":
        if source_reliability is None:
            source_reliability = DEFAULT_SOURCE_RELIABILITY[method]
        return cls(
            id=provenance_id(method.value, agent, source_ref),
            method=method,
            agent=agent,
            source_ref=source_ref,
            source_reliability=source_reliability,
        )
