"""Dependency algebra (spec sections 10, 29, 30).

``A > B`` means ``state(B) -> state(A)``: A's behaviour is a function of B's.
Impact is the *reverse* closure of that relation:

    Impact(A) = { x | x >* A }

that is, everything whose behaviour is a function of A's, directly or through a
chain. The walk crosses any edge type the RelationSpec table subsumes under
DEPENDS_ON, and no others - traversal is licensed by the algebra, not by whatever
edges happen to exist (spec section 11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..core.relations import MCMRelation
from ..storage.database import Direction, Store
from ..storage.projections import GraphProjection
from .confidence import path_confidence
from .specs import dependency_edge_types


@dataclass(frozen=True)
class DependencyPath:
    """One object reached by the closure, with the route that reached it.

    ``objects`` runs start-first: ``[start, ..., reached]``. ``relations`` has one
    entry per hop, so ``len(relations) == len(objects) - 1 == depth``.
    """

    object_id: str
    depth: int
    objects: tuple[str, ...]
    relations: tuple[MCMRelation, ...]

    @property
    def confidence(self) -> float:
        return path_confidence(list(self.relations))

    @property
    def edge_types(self) -> tuple[str, ...]:
        return tuple(r.relation_type.value for r in self.relations)


@dataclass
class Closure:
    target_id: str
    paths: dict[str, DependencyPath] = field(default_factory=dict)
    truncated: bool = False
    #: The moment this closure was computed at. None means "now" (spec section 18).
    as_of: datetime | None = None

    def at_depth(self, depth: int) -> list[DependencyPath]:
        return [p for p in self.paths.values() if p.depth == depth]

    def ordered(self) -> list[DependencyPath]:
        """Shallowest first, then by descending confidence, then by id."""
        return sorted(self.paths.values(), key=lambda p: (p.depth, -p.confidence, p.object_id))


def dependents_of(store: Store, target_id: str, *, max_depth: int = 6,
                  as_of: datetime | None = None) -> Closure:
    """Everything that depends on ``target_id``, directly or transitively.

    This is the impact question: what could break if the target changes.
    """
    return _closure(store, target_id, direction="in", max_depth=max_depth, as_of=as_of)


def dependencies_of(store: Store, source_id: str, *, max_depth: int = 6,
                    as_of: datetime | None = None) -> Closure:
    """Everything ``source_id`` depends on (spec section 27)."""
    return _closure(store, source_id, direction="out", max_depth=max_depth, as_of=as_of)


def _closure(store: Store, start_id: str, *, direction: Direction, max_depth: int,
             as_of: datetime | None = None) -> Closure:
    """Level-by-level closure over dependency edges.

    Breadth-first, so each object is recorded at its shortest depth. Among routes
    of equal depth the highest-confidence one is kept, with the lexicographically
    smaller path as a deterministic tie-break. One route is reported per object;
    alternatives are not enumerated.

    ``as_of`` runs the walk against the repository as it stood at that moment, so
    a closure over a past state crosses only the relations that were valid then.
    """
    graph = GraphProjection(store)
    edge_types = dependency_edge_types()
    closure = Closure(target_id=start_id, as_of=as_of)
    seen = {start_id}
    frontier = [DependencyPath(object_id=start_id, depth=0, objects=(start_id,), relations=())]
    depth = 0

    while frontier and depth < max_depth:
        candidates: dict[str, DependencyPath] = {}
        for current in frontier:
            for neighbour in graph.neighbours(
                current.object_id, direction=direction, types=edge_types, as_of=as_of
            ):
                if neighbour.object_id in seen:
                    continue
                candidate = DependencyPath(
                    object_id=neighbour.object_id,
                    depth=current.depth + 1,
                    objects=(*current.objects, neighbour.object_id),
                    relations=(*current.relations, neighbour.relation),
                )
                incumbent = candidates.get(neighbour.object_id)
                if incumbent is None or _prefer(candidate, incumbent):
                    candidates[neighbour.object_id] = candidate

        seen.update(candidates)
        closure.paths.update(candidates)
        frontier = list(candidates.values())
        depth += 1

    closure.truncated = bool(frontier) and any(
        n.object_id not in seen
        for path in frontier
        for n in graph.neighbours(path.object_id, direction=direction,
                                  types=edge_types, as_of=as_of)
    )
    return closure


def _prefer(candidate: DependencyPath, incumbent: DependencyPath) -> bool:
    """Higher confidence wins; ties break on path order so results are stable."""
    if candidate.confidence != incumbent.confidence:
        return candidate.confidence > incumbent.confidence
    return candidate.objects < incumbent.objects
