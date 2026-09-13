"""Graph projection over the semantic core (spec section 21).

    Node = MCMObject
    Edge = MCMRelation

This is a *view*. It holds no state of its own and answers every question by
reading the canonical store, which is what stops the graph from quietly becoming
the source of truth (spec section 21). Swapping in a real graph database later
means reimplementing this class, not migrating any data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from ..core.relations import MCMRelation, RelationType
from .database import Direction, Store


@dataclass(frozen=True)
class Neighbour:
    """An adjacent object and the relation that reaches it."""

    object_id: str
    relation: MCMRelation


class GraphProjection:
    def __init__(self, store: Store) -> None:
        self.store = store

    def neighbours(self, object_id: str, *, direction: Direction = "out",
                   types: Iterable[RelationType] | None = None,
                   include_derived: bool = False,
                   as_of: datetime | None = None) -> list[Neighbour]:
        """Objects adjacent to ``object_id``.

        ``direction="out"`` follows relations where the object is arguments[0] and
        returns the later arguments. ``direction="in"`` finds relations that point
        *at* this object and returns their arguments[0] - which is how impact
        analysis walks backwards from a change site to everything that depends on
        it.
        """
        out: list[Neighbour] = []
        for relation in self.store.relations_for(
            object_id, direction=direction, types=types,
            include_derived=include_derived, as_of=as_of,
        ):
            if direction == "out":
                out.extend(Neighbour(arg, relation) for arg in relation.arguments[1:])
            elif direction == "in":
                if relation.arguments[0] != object_id:
                    out.append(Neighbour(relation.arguments[0], relation))
            else:
                out.extend(Neighbour(arg, relation)
                           for arg in relation.arguments if arg != object_id)
        return out
