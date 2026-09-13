"""Hybrid retrieval and the scoring model (spec sections 25 and 26, step 15).

Spec section 25 requires that every query potentially use several retrieval
mechanisms, pooled and reranked before anything reasons over the result:

    query -> lexical | vector | graph | symbolic -> candidate pool -> rerank

Spec section 26 gives the reranker:

    Score(x|q) = w_v*V + w_l*L + w_g*G + w_s*S + w_p*P

with every weight configurable and no assumption that graph retrieval wins. The
defaults here sum to 1.0 so a score reads as a fraction, and they are *guesses*.
Nothing has been tuned, because tuning without the benchmark in spec sections 45
to 50 would be choosing weights to flatter whatever example was at hand.

**What is retrieved.** Objects. Relation and evidence documents are indexed
(spec section 22 asks for all three) but they are *routes*, not results: a hit on
``CALLS(authenticate, validate_token)`` is evidence that both endpoints are
relevant, and a hit on an error string in an evidence record is a way to reach the
code it came from. Spec section 43 wants evidence shown in support of an answer,
which is the opposite of returning it instead of one. Each route is reported, so
every score can be traced to the document that produced it.

**What this is not.** No learned reranker. There is no training data and no
relevance judgements, so a cross-encoder here would be an untested guess wearing a
model's reputation. The weighted sum is the "initial scoring model" spec section 26
asks for, and it is the thing the ablation studies are meant to attack.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from ..core.objects import MCMObject
from ..core.provenance import Provenance
from ..storage.database import Store
from ..storage.projections import GraphProjection
from .documents import EVIDENCE, OBJECT, RELATION
from .embedding import tokenize
from .lexical import LexicalProjection
from .symbolic import resolve
from .vector import VectorProjection

#: Channel identifiers, used in ``Candidate.channels`` and in the route records.
VECTOR = "vector"
LEXICAL = "lexical"
GRAPH = "graph"
SYMBOLIC = "symbolic"
PROVENANCE = "provenance"


@dataclass(frozen=True)
class RetrievalWeights:
    """The section 26 scoring model. Every term is configurable.

    Defaults sum to 1.0, so a default-weighted score lies in [0, 1]. Change them
    and the scale is yours; the ranking is unaffected by scale either way.

    ``route_relation`` and ``route_evidence`` attenuate a score that arrived
    through a relation or an evidence record rather than through the object's own
    document. They are weights in the same sense as the rest, and are exposed for
    the same reason: they encode a belief about indirectness that deserves to be
    tested rather than buried.
    """

    vector: float = 0.25
    lexical: float = 0.25
    graph: float = 0.25
    symbolic: float = 0.15
    provenance: float = 0.10

    route_relation: float = 0.7
    route_evidence: float = 0.6

    #: How much graph relevance survives one hop from an anchor, and how far to
    #: walk. Spec section 26 warns against assuming graph retrieval is superior;
    #: a decay of 0.5 means a second-hop neighbour contributes a quarter of what
    #: an exact match does on its own channel.
    graph_decay: float = 0.5
    graph_depth: int = 2

    @property
    def total(self) -> float:
        return self.vector + self.lexical + self.graph + self.symbolic + self.provenance

    def route_weight(self, kind: str) -> float:
        if kind == OBJECT:
            return 1.0
        if kind == RELATION:
            return self.route_relation
        return self.route_evidence

    @classmethod
    def parse(cls, spec: str) -> "RetrievalWeights":
        """Parse ``v=0.4,l=0.3,g=0.2,s=0.1,p=0.0``.

        Single letters for the five scoring terms; full names for the rest.
        Unnamed terms keep their default, so an ablation names only what it
        changes.
        """
        aliases = {"v": "vector", "l": "lexical", "g": "graph",
                   "s": "symbolic", "p": "provenance"}
        values: dict[str, float] = {}
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            name, _, raw = item.partition("=")
            name = aliases.get(name.strip(), name.strip())
            if name not in cls.__dataclass_fields__:
                known = ", ".join(sorted(cls.__dataclass_fields__))
                raise ValueError(f"unknown weight {name!r}; known weights: {known}")
            values[name] = int(raw) if name == "graph_depth" else float(raw)
        return cls(**values)


@dataclass(frozen=True)
class Route:
    """How a candidate entered the pool: which document, found by which channel."""

    channel: str
    kind: str
    key: str
    raw: float
    contribution: float


@dataclass
class Candidate:
    object_id: str
    object: MCMObject | None
    vector: float = 0.0
    lexical: float = 0.0
    graph: float = 0.0
    symbolic: float = 0.0
    provenance: float = 0.0
    score: float = 0.0
    routes: list[Route] = field(default_factory=list)

    @property
    def channels(self) -> list[str]:
        """Channels that contributed anything. A candidate found by several
        channels is a different kind of result from one found by exactly one, and
        spec section 25 exists because those two should not be conflated."""
        found = [name for name, value in (
            (VECTOR, self.vector), (LEXICAL, self.lexical),
            (GRAPH, self.graph), (SYMBOLIC, self.symbolic)) if value > 0.0]
        return found


@dataclass
class RetrievalResult:
    query: str
    weights: RetrievalWeights
    candidates: list[Candidate]
    pool_size: int
    indexed: bool = True

    def top(self, n: int = 5) -> list[Candidate]:
        return self.candidates[:n]


class HybridRetriever:
    """Pools four retrieval channels and reranks them (spec sections 25, 26)."""

    def __init__(self, store: Store, *,
                 vector: VectorProjection | None = None,
                 lexical: LexicalProjection | None = None,
                 weights: RetrievalWeights | None = None) -> None:
        self.store = store
        self.vector = vector if vector is not None else VectorProjection(store)
        self.lexical = lexical if lexical is not None else LexicalProjection(store)
        self.weights = weights or RetrievalWeights()
        self.graph = GraphProjection(store)

    def retrieve(self, query: str, *, limit: int = 10, pool_size: int = 50,
                 as_of: datetime | None = None) -> RetrievalResult:
        routes = _RouteMap(self.store, as_of)
        candidates: dict[str, Candidate] = {}

        self._retrieve_lexical(query, pool_size, routes, candidates)
        self._retrieve_vector(query, pool_size, routes, candidates)
        self._retrieve_symbolic(query, candidates)
        self._retrieve_graph(candidates, as_of)
        self._score_provenance(candidates, as_of)

        for candidate in candidates.values():
            candidate.object = self.store.get_object(candidate.object_id)
            w = self.weights
            candidate.score = (w.vector * candidate.vector
                               + w.lexical * candidate.lexical
                               + w.graph * candidate.graph
                               + w.symbolic * candidate.symbolic
                               + w.provenance * candidate.provenance)

        ranked = sorted(candidates.values(), key=lambda c: (-c.score, c.object_id))
        return RetrievalResult(
            query=query, weights=self.weights, candidates=ranked[:limit],
            pool_size=len(candidates),
            indexed=bool(self.lexical.index.digests()),
        )

    # --- channels ---------------------------------------------------------

    def _retrieve_lexical(self, query: str, pool_size: int, routes: "_RouteMap",
                          candidates: dict[str, Candidate]) -> None:
        for match in self.lexical.search(query, limit=pool_size):
            self._spread(candidates, routes, LEXICAL, match.kind, match.key,
                         match.source_id, match.relevance)

    def _retrieve_vector(self, query: str, pool_size: int, routes: "_RouteMap",
                         candidates: dict[str, Candidate]) -> None:
        for match in self.vector.search(query, limit=pool_size):
            self._spread(candidates, routes, VECTOR, match.kind, match.key,
                         match.source_id, match.similarity)

    def _retrieve_symbolic(self, query: str,
                           candidates: dict[str, Candidate]) -> None:
        """Exact resolution of the query and of each of its tokens.

        This is the channel that spec section 22 is protecting when it says the
        vector store must not own exact matching. If the query names something
        that exists, that object is in the pool with a perfect symbolic score
        regardless of what similarity thought.
        """
        for term in _symbolic_terms(query):
            for obj in resolve(self.store, term):
                candidate = _get(candidates, obj.id)
                candidate.symbolic = 1.0
                candidate.routes.append(
                    Route(SYMBOLIC, OBJECT, obj.id, 1.0, 1.0))

    def _retrieve_graph(self, candidates: dict[str, Candidate],
                        as_of: datetime | None) -> None:
        """Relevance by proximity to what the other channels already found.

        Anchors are the objects the text channels are most confident about. From
        each, relevance decays by ``graph_decay`` per hop, and a candidate keeps
        the best route it has. This is the channel that finds the function nobody
        named: the one the anchor calls.

        Traversal is over asserted relations only. Derived relations are not
        persisted (spec Rule 2), and retrieving through an inference would quietly
        let the inference engine decide what the agent gets to see.
        """
        anchors = self._anchors(candidates)
        if not anchors:
            return
        # Anchors seed the walk but score nothing on this channel. They are here
        # because the text or symbolic channels found them, and crediting them for
        # their own graph position would count the same evidence twice under two
        # weights. Graph relevance is relevance an object has *only* because of
        # where it sits relative to something else.
        best: dict[str, tuple[float, int]] = {}
        queue: deque[tuple[str, float, int, str]] = deque()
        for object_id, weight in anchors:
            queue.append((object_id, weight, 0, object_id))

        while queue:
            object_id, weight, depth, origin = queue.popleft()
            if depth >= self.weights.graph_depth:
                continue
            reach = weight * self.weights.graph_decay
            if reach <= 0.0:
                continue
            for neighbour in self.graph.neighbours(object_id, direction="any",
                                                   as_of=as_of):
                # A path that returns to the anchor it started from says nothing
                # about the anchor. Without this, any function scores on the graph
                # channel by way of the file that contains it.
                if neighbour.object_id == origin:
                    continue
                current = best.get(neighbour.object_id)
                if current is not None and current[0] >= reach:
                    continue
                best[neighbour.object_id] = (reach, depth + 1)
                queue.append((neighbour.object_id, reach, depth + 1, origin))

        for object_id, (weight, depth) in best.items():
            candidate = _get(candidates, object_id)
            candidate.graph = max(candidate.graph, weight)
            candidate.routes.append(
                Route(GRAPH, OBJECT, object_id, float(depth), weight))

    def _anchors(self, candidates: dict[str, Candidate]) -> list[tuple[str, float]]:
        """Objects the graph walk starts from: exact symbolic matches if there are
        any, otherwise the strongest text hits.

        Deferring to symbolic matches matters. When the query names a real symbol,
        walking from a vaguely similar function instead would drag in a
        neighbourhood that has nothing to do with the question.
        """
        exact = [(c.object_id, 1.0) for c in candidates.values() if c.symbolic > 0.0]
        if exact:
            return exact
        scored = [(c.object_id, max(c.vector, c.lexical)) for c in candidates.values()]
        scored = [pair for pair in scored if pair[1] > 0.0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:3]

    def _score_provenance(self, candidates: dict[str, Candidate],
                          as_of: datetime | None) -> None:
        """P: how much the extractors behind this object are trusted.

        The mean ``source_reliability`` of the provenance records on the asserted
        relations that touch it. Evidence strength is deliberately *not* folded in:
        spec section 17 forbids multiplying the uncertainty dimensions together
        without a justified model, and there is no such model here. An object with
        no provenance at all scores 0 rather than a flattering default - an
        unsourced object is exactly what this term exists to rank down.
        """
        cache: dict[str, Provenance | None] = {}
        for candidate in candidates.values():
            relations = self.store.relations_for(candidate.object_id, direction="any",
                                                 as_of=as_of)
            values = []
            for relation in relations:
                if relation.provenance_id is None:
                    continue
                if relation.provenance_id not in cache:
                    cache[relation.provenance_id] = self.store.get_provenance(
                        relation.provenance_id)
                record = cache[relation.provenance_id]
                if record is not None:
                    values.append(record.source_reliability)
            candidate.provenance = sum(values) / len(values) if values else 0.0

    def _spread(self, candidates: dict[str, Candidate], routes: "_RouteMap",
                channel: str, kind: str, key: str, source_id: str,
                raw: float) -> None:
        """Attribute a document hit to the objects it is about."""
        if raw <= 0.0:
            return
        contribution = raw * self.weights.route_weight(kind)
        for object_id in routes.objects_for(kind, source_id):
            candidate = _get(candidates, object_id)
            if channel == LEXICAL:
                candidate.lexical = max(candidate.lexical, contribution)
            else:
                candidate.vector = max(candidate.vector, contribution)
            candidate.routes.append(Route(channel, kind, key, raw, contribution))


class _RouteMap:
    """Document to object attribution, built once per query.

    One pass over the relations gives both directions that matter: a relation
    document reaches its arguments, and an evidence document reaches the arguments
    of every relation that cites it. The pass is O(relations) and happens once; an
    index would be a fourth projection to keep in step with the core, which is not
    worth it until a corpus makes the scan measurable.
    """

    def __init__(self, store: Store, as_of: datetime | None) -> None:
        self._relations: dict[str, list[str]] = {}
        self._evidence: dict[str, set[str]] = {}
        for relation in store.all_relations(include_derived=False, as_of=as_of):
            self._relations[relation.id] = list(relation.arguments)
            for evidence_id in relation.evidence_ids:
                self._evidence.setdefault(evidence_id, set()).update(relation.arguments)

    def objects_for(self, kind: str, source_id: str) -> list[str]:
        if kind == OBJECT:
            return [source_id]
        if kind == RELATION:
            return self._relations.get(source_id, [])
        if kind == EVIDENCE:
            return sorted(self._evidence.get(source_id, ()))
        return []


def _symbolic_terms(query: str) -> list[str]:
    """Candidate references to resolve exactly: the query itself, and its tokens.

    The whole query first, because ``app/auth.py:authenticate`` is a reference and
    tokenising it would destroy it.
    """
    terms = [query.strip()]
    for token in tokenize(query):
        if token not in terms and len(token) > 1:
            terms.append(token)
    return terms


def _get(candidates: dict[str, Candidate], object_id: str) -> Candidate:
    candidate = candidates.get(object_id)
    if candidate is None:
        candidate = Candidate(object_id=object_id, object=None)
        candidates[object_id] = candidate
    return candidate


def build_indexes(store: Store, *, vector: VectorProjection | None = None,
                  lexical: LexicalProjection | None = None,
                  as_of: datetime | None = None) -> tuple:
    """Rebuild both derived text indexes. Returns the two build reports."""
    vector = vector if vector is not None else VectorProjection(store)
    lexical = lexical if lexical is not None else LexicalProjection(store)
    return vector.rebuild(as_of=as_of), lexical.rebuild(as_of=as_of)


def explain(result: RetrievalResult, *, limit: int = 5) -> str:
    """Human-readable ranking with the per-channel breakdown (spec section 43).

    The breakdown is the point. A result that scored on one channel and a result
    that scored on four are different claims, and a reader who cannot see which is
    which has no way to distrust the ranking.
    """
    lines = [f'Query: "{result.query}"']
    if not result.indexed:
        lines.append("  the text indexes are empty - run `mcm index` first")
    lines.append(f"  {result.pool_size} candidates pooled, showing {min(limit, len(result.candidates))}")
    lines.append(f"  weights: v={result.weights.vector} l={result.weights.lexical} "
                 f"g={result.weights.graph} s={result.weights.symbolic} "
                 f"p={result.weights.provenance}")
    lines.append("")
    for position, candidate in enumerate(result.top(limit), start=1):
        obj = candidate.object
        name = obj.name if obj else candidate.object_id
        where = (obj.properties.get("relpath") if obj else None) or ""
        kind = obj.type.value if obj else "?"
        lines.append(f"{position}. {name}  [{kind}] {where}")
        lines.append(f"     score {candidate.score:.4f}  "
                     f"v={candidate.vector:.3f} l={candidate.lexical:.3f} "
                     f"g={candidate.graph:.3f} s={candidate.symbolic:.3f} "
                     f"p={candidate.provenance:.3f}")
        found = ", ".join(candidate.channels) or "none"
        lines.append(f"     found by: {found}")
        for route in _best_routes(candidate):
            lines.append(f"       via {route.channel} {route.kind} "
                         f"{_short(route.key)} ({route.contribution:.3f})")
    return "\n".join(lines)


def _best_routes(candidate: Candidate, limit: int = 3) -> list[Route]:
    ordered = sorted(candidate.routes, key=lambda r: -r.contribution)
    seen: set[tuple[str, str]] = set()
    out: list[Route] = []
    for route in ordered:
        signature = (route.channel, route.kind)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(route)
        if len(out) == limit:
            break
    return out


def _short(key: str) -> str:
    return key if len(key) <= 56 else key[:53] + "..."


def result_json(result: RetrievalResult) -> dict:
    """Serialise in the style of the spec section 42 response."""
    return {
        "mode": "retrieve",
        "query": result.query,
        "indexed": result.indexed,
        "pool_size": result.pool_size,
        "weights": {
            "vector": result.weights.vector, "lexical": result.weights.lexical,
            "graph": result.weights.graph, "symbolic": result.weights.symbolic,
            "provenance": result.weights.provenance,
            "route_relation": result.weights.route_relation,
            "route_evidence": result.weights.route_evidence,
            "graph_decay": result.weights.graph_decay,
            "graph_depth": result.weights.graph_depth,
        },
        "results": [
            {
                "object_id": c.object_id,
                "name": c.object.name if c.object else None,
                "type": c.object.type.value if c.object else None,
                "relpath": c.object.properties.get("relpath") if c.object else None,
                "score": round(c.score, 6),
                "channels": {
                    "vector": round(c.vector, 6), "lexical": round(c.lexical, 6),
                    "graph": round(c.graph, 6), "symbolic": round(c.symbolic, 6),
                    "provenance": round(c.provenance, 6),
                },
                "found_by": c.channels,
                "routes": [
                    {"channel": r.channel, "kind": r.kind, "key": r.key,
                     "raw": round(r.raw, 6), "contribution": round(r.contribution, 6)}
                    for r in _best_routes(c)
                ],
            }
            for c in result.candidates
        ],
    }


def retrieve(store: Store, query: str, *, limit: int = 10,
             weights: RetrievalWeights | None = None,
             as_of: datetime | None = None) -> RetrievalResult:
    """Convenience entry point for callers that hold nothing but a store."""
    return HybridRetriever(store, weights=weights).retrieve(
        query, limit=limit, as_of=as_of)


def channel_names() -> Iterable[str]:
    return (VECTOR, LEXICAL, GRAPH, SYMBOLIC, PROVENANCE)
