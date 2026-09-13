"""First-class constraints (spec sections 13 and 14).

A constraint is a knowledge object, not a validation callback. It has an
identity, a scope, provenance, and a verdict that can be recomputed at any time
and explained.

Section 13 requires constraints to be evaluatable. Not all of them are. The eight
types in section 14 split cleanly:

* **Decidable against the semantic model.** ARCHITECTURAL, DEPENDENCY, SECURITY
  and TEST constraints are claims about the relation graph, and the store already
  holds everything needed to settle them.
* **Claims about runtime values.** PRECONDITION, POSTCONDITION, INVARIANT and
  TYPE constraints talk about ``token.expiry`` and ``user.valid``. Deciding those
  statically needs symbolic execution, which V2 does not have.

Both are stored. The second kind returns UNEVALUATABLE with the names it would
need, because a verdict of SATISFIED on a constraint that was never checked is
worse than no verdict at all.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch

from .ids import _digest
from .objects import MCMObject, ObjectType


class ConstraintType(str, Enum):
    PRECONDITION = "PRECONDITION"
    POSTCONDITION = "POSTCONDITION"
    INVARIANT = "INVARIANT"
    TYPE_CONSTRAINT = "TYPE_CONSTRAINT"
    DEPENDENCY_CONSTRAINT = "DEPENDENCY_CONSTRAINT"
    SECURITY_CONSTRAINT = "SECURITY_CONSTRAINT"
    ARCHITECTURAL_CONSTRAINT = "ARCHITECTURAL_CONSTRAINT"
    TEST_CONSTRAINT = "TEST_CONSTRAINT"


class Verdict(str, Enum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    #: The constraint is well-formed but cannot be decided from what MCM knows.
    UNEVALUATABLE = "UNEVALUATABLE"


@dataclass(frozen=True)
class Selector:
    """Picks a set of objects. Every field is an additional restriction."""

    type: ObjectType | None = None
    id_glob: str | None = None
    name_glob: str | None = None

    def matches(self, obj: MCMObject) -> bool:
        if self.type is not None and obj.type is not self.type:
            return False
        if self.id_glob is not None and not fnmatch(obj.id, self.id_glob):
            return False
        if self.name_glob is not None and not fnmatch(obj.name, self.name_glob):
            return False
        return True

    def describe(self) -> str:
        parts = []
        if self.type is not None:
            parts.append(self.type.value)
        if self.id_glob:
            parts.append(self.id_glob)
        if self.name_glob:
            parts.append(f"name={self.name_glob}")
        return " ".join(parts) if parts else "anything"

    @classmethod
    def from_dict(cls, raw: dict | str | None) -> "Selector":
        if raw is None:
            return cls()
        if isinstance(raw, str):
            return cls(id_glob=raw)
        object_type = raw.get("type")
        return cls(
            type=ObjectType(object_type) if object_type else None,
            id_glob=raw.get("id_glob"),
            name_glob=raw.get("name_glob"),
        )


@dataclass
class Constraint:
    """A checkable requirement over the semantic model.

    Exactly one of ``predicate`` and ``expression`` carries the content.
    ``predicate`` names a check the constraint checker implements against the
    relation graph. ``expression`` holds a value-level claim, kept verbatim so it
    survives until something can decide it.
    """

    id: str
    name: str
    type: ConstraintType
    scope: Selector = field(default_factory=Selector)
    predicate: str | None = None
    parameters: dict = field(default_factory=dict)
    expression: str | None = None
    description: str = ""
    provenance_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.type, ConstraintType):
            self.type = ConstraintType(self.type)
        if bool(self.predicate) == bool(self.expression):
            raise ValueError(
                f"constraint {self.name}: give exactly one of predicate or expression"
            )

    @property
    def is_decidable(self) -> bool:
        """True when this constraint is a claim about the relation graph."""
        return self.predicate is not None

    def free_names(self) -> list[str]:
        """Names a value-level expression would need bound to be decided.

        Parsed rather than pattern-matched, so the UNEVALUATABLE reason names
        exactly what is missing instead of echoing the expression back.
        """
        if not self.expression:
            return []
        try:
            tree = ast.parse(self.expression, mode="eval")
        except SyntaxError:
            return []
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                root = node
                parts = []
                while isinstance(root, ast.Attribute):
                    parts.append(root.attr)
                    root = root.value
                if isinstance(root, ast.Name):
                    names.add(".".join([root.id, *reversed(parts)]))
            elif isinstance(node, ast.Name):
                names.add(node.id)
        # Drop bases already covered by a longer attribute path.
        return sorted(n for n in names
                      if not any(other != n and other.startswith(n + ".") for other in names))


def constraint_id(name: str, constraint_type: str) -> str:
    return "con:" + _digest(constraint_type, name)
