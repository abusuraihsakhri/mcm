"""Typed semantic objects - the canonical knowledge unit (spec section 7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class ObjectType(str, Enum):
    """The object vocabulary from spec section 7.

    The full vocabulary is declared here, but V1 ingestion only emits the subset
    that deterministic Python analysis can justify. Spec Rule 8: do not build a
    large ontology before testing the basic hypothesis.
    """

    REPOSITORY = "Repository"
    DIRECTORY = "Directory"
    FILE = "File"
    MODULE = "Module"
    CLASS = "Class"
    FUNCTION = "Function"
    METHOD = "Method"
    VARIABLE = "Variable"
    PARAMETER = "Parameter"
    TYPE = "Type"
    API = "API"
    DATABASE_TABLE = "DatabaseTable"
    CONFIGURATION = "Configuration"
    TEST = "Test"
    REQUIREMENT = "Requirement"
    BUG = "Bug"
    COMMIT = "Commit"
    DECISION = "Decision"
    OBSERVATION = "Observation"
    DOCUMENTATION = "Documentation"
    CONCEPT = "Concept"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class MCMObject:
    """A typed semantic object.

    ``properties`` holds descriptive facts that do not participate in identity
    (source line span, signature, docstring). ``state`` holds mutable status that
    later phases update - it is kept separate so that a state change never looks
    like a redefinition of the object.
    """

    id: str
    type: ObjectType
    name: str
    properties: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("MCMObject requires a non-empty id")
        if not isinstance(self.type, ObjectType):
            self.type = ObjectType(self.type)
