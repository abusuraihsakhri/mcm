"""The systems under test (spec section 45).

Spec section 45 asks for three baselines and an optional fourth::

    A. Vector RAG      B. Graph memory      C. MCM      D. Hybrid graph + vector

They are built here as **independent implementations**, not as MCM with channels
switched off. The distinction decides whether the experiment can fail. A baseline
assembled from ``RetrievalWeights(vector=1, ...)`` would inherit MCM's
symbol-aware document construction, its route map from relations and evidence back
to objects, and its exact-resolution channel. It would then lose to MCM by
construction, and spec section 3's requirement that the result be able to come out
``MCM < Graph`` would be unmeetable. Component ablations of MCM are a separate
question, asked in ``ablation.py`` under spec section 50 and labelled as such.

What the baselines *do* share with MCM is the corpus and the embedding provider.
Both are deliberate. A different corpus would make the comparison meaningless, and
a weaker embedding model for Baseline A would rig the headline comparison in the
crudest possible way. Baseline A gets the same vectors MCM's vector channel gets;
what differs is that it chunks raw text on line windows the way a RAG pipeline
does, instead of embedding symbol documents.

**The budget.** Every system fills the same token budget and is free to spend it
however its architecture prefers: few large chunks, many small definitions, or a
context package. A budget in tokens rather than a count of items is what an agent
actually has, and it is the only framing under which "30 explanatory facts" and
"1000 relevant-looking chunks" from spec section 48 can be compared at all.

**Systems do not grade themselves.** ``answer`` returns what the system would put
in front of an agent, in rank order, with the token cost of each unit and the
definitions that unit contains. Whether any of it was useful is decided by the
harness against ground truth the system never sees.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

from ..ingestion.git import GitReader
from ..ingestion.parser import parse_python
from ..ingestion.repository import RepositoryIngestor
from ..ingestion.sources import GitRevisionProvider
from ..reasoning.dependency_propagation import analyse_impact
from ..retrieval.embedding import EmbeddingProvider, cosine, get_provider, tokenize
from ..retrieval.hybrid import HybridRetriever, RetrievalWeights
from ..retrieval.lexical import LexicalProjection
from ..retrieval.vector import VectorProjection
from ..storage.sqlite_store import SQLiteStore
from .metrics import token_estimate
from .tasks import IMPACT, BenchmarkTask, RepoSnapshot, query_for, unit_id

#: Context an agent is willing to spend on retrieval for one task.
DEFAULT_BUDGET = 4000


@dataclass(frozen=True)
class RetrievedUnit:
    """One thing a system puts in front of the agent.

    ``tokens`` is what reading it costs. ``covers`` maps each definition the unit
    contains to that definition's own token cost, which is how a 900-token file
    and a 40-token function are compared without pretending they are the same
    size.
    """

    unit_id: str
    relpath: str
    tokens: int
    covers: dict[str, int] = field(default_factory=dict)
    detail: str = ""


@dataclass
class SystemAnswer:
    """What a system returned, before anyone decided whether it was right."""

    system: str
    units: list[RetrievedUnit] = field(default_factory=list)
    latency_ms: float = 0.0
    note: str = ""

    @property
    def total_tokens(self) -> int:
        return sum(unit.tokens for unit in self.units)

    def ranked_definitions(self, exclude: frozenset[str] = frozenset()
                           ) -> list[str]:
        """Definitions in rank order, deduplicated, first occurrence winning.

        A unit covering several definitions contributes all of them at its own
        rank. That lets a chunk retriever deliver more answers per slot than a
        definition retriever, which is exactly the trade it makes in reality: it
        buys recall with tokens, and the efficiency metric charges it for them.

        ``exclude`` drops the symbol an impact question names. The harness applies
        it to every system rather than trusting each to remember, because a
        chunk retriever cannot drop the seed on its own - the seed arrives inside
        a window with its neighbours - and letting it occupy a rank slot would
        penalise that system for a rule the others follow silently. The tokens
        the seed cost are still charged: spending context to deliver the symbol
        you were already given is a real property of chunk retrieval, not a
        harness artefact.
        """
        seen: set[str] = set()
        ordered: list[str] = []
        for unit in self.units:
            for definition in unit.covers:
                if definition not in seen and definition not in exclude:
                    seen.add(definition)
                    ordered.append(definition)
        return ordered

    def ranked_files(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for unit in self.units:
            if unit.relpath and unit.relpath not in seen:
                seen.add(unit.relpath)
                ordered.append(unit.relpath)
        return ordered

    def useful_tokens(self, targets: frozenset[str], granularity: str) -> int:
        """Token cost of the parts of the answer that were actually on target.

        Counted once per target, at the size of the target itself rather than the
        size of whatever container delivered it. See ``metrics`` for why.
        """
        if granularity == "file":
            counted: set[str] = set()
            total = 0
            for unit in self.units:
                if unit.relpath in targets and unit.relpath not in counted:
                    counted.add(unit.relpath)
                    total += unit.tokens
            return total
        counted = set()
        total = 0
        for unit in self.units:
            for definition, cost in unit.covers.items():
                if definition in targets and definition not in counted:
                    counted.add(definition)
                    total += cost
        return total


class BenchmarkSystem(Protocol):
    """One architecture, indexed once per repository and asked many questions."""

    name: str

    def prepare(self, snapshot: RepoSnapshot) -> None: ...

    def answer(self, task: BenchmarkTask, *, budget: int) -> SystemAnswer: ...

    def release(self) -> None:
        """Drop the index, once every question has been asked of it.

        Optional. The runner calls it so that a repository too large for four
        indexes at once can still be scored: MCM's in-memory store alone reaches
        1.8GB on django, and holding all four was what made the large tier fail.
        A system that is cheap to hold can leave this as the no-op default.
        """


# --- Baseline A: vector RAG ------------------------------------------------

@dataclass(frozen=True)
class _Chunk:
    relpath: str
    start_line: int
    end_line: int
    text: str
    tokens: int


class BaselineVectorRAG:
    """Spec section 45 Baseline A: chunked embeddings and cosine similarity.

    The pipeline a RAG system actually runs. Files are cut into overlapping fixed
    line windows with no regard for where definitions begin or end, each window is
    embedded whole, and retrieval is nearest-neighbour against the query. There is
    no symbol table, no graph, no notion that two chunks are related.

    Overlapping windows are not a handicap invented for this study; they are
    standard practice, because a window boundary through the middle of a function
    is otherwise unrecoverable. The overlap costs tokens, which the efficiency
    metric charges for, and that trade is a real property of the architecture
    rather than an artefact of the harness.
    """

    def __init__(self, *, chunk_lines: int = 40, stride: int = 30,
                 provider: EmbeddingProvider | None = None) -> None:
        self.name = "vector-rag"
        self.chunk_lines = chunk_lines
        self.stride = stride
        self.provider = provider or get_provider()
        self._chunks: list[_Chunk] = []
        self._vectors: list[list[float]] = []
        self._snapshot: RepoSnapshot | None = None

    def prepare(self, snapshot: RepoSnapshot) -> None:
        if self._snapshot is snapshot and self._chunks:
            return      # already indexed; Baseline D shares this instance
        self._snapshot = snapshot
        self._chunks = []
        for relpath in snapshot.files:
            text = snapshot.sources[relpath].decode("utf-8", errors="replace")
            lines = text.splitlines()
            if not lines:
                continue
            for start in range(0, len(lines), self.stride):
                window = lines[start:start + self.chunk_lines]
                if not window:
                    break
                body = "\n".join(window)
                if not body.strip():
                    continue
                self._chunks.append(_Chunk(
                    relpath=relpath,
                    start_line=start + 1,
                    end_line=start + len(window),
                    text=body,
                    tokens=token_estimate(body),
                ))
                if start + self.chunk_lines >= len(lines):
                    break
        self._vectors = self.provider.embed([c.text for c in self._chunks])

    def release(self) -> None:
        self._chunks, self._vectors = [], []

    def answer(self, task: BenchmarkTask, *, budget: int) -> SystemAnswer:
        started = time.perf_counter()
        snapshot = self._snapshot
        assert snapshot is not None, "prepare() first"

        query = self.provider.embed_one(query_for(task))
        scored = sorted(
            ((cosine(query, vector), index)
             for index, vector in enumerate(self._vectors)),
            key=lambda pair: (-pair[0], pair[1]))

        units: list[RetrievedUnit] = []
        spent = 0
        for score, index in scored:
            chunk = self._chunks[index]
            if spent + chunk.tokens > budget:
                continue
            units.append(RetrievedUnit(
                unit_id=f"{chunk.relpath}#{chunk.start_line}-{chunk.end_line}",
                relpath=chunk.relpath,
                tokens=chunk.tokens,
                covers=snapshot.covering(chunk.relpath, chunk.start_line,
                                         chunk.end_line),
                detail=f"cosine {score:.3f}",
            ))
            spent += chunk.tokens
            if spent >= budget:
                break
        return SystemAnswer(system=self.name, units=units,
                            latency_ms=(time.perf_counter() - started) * 1000)


# --- Baseline B: graph memory ----------------------------------------------

@dataclass
class _Node:
    unit: str
    relpath: str
    qualname: str
    name: str
    tokens: int


class BaselineGraph:
    """Spec section 45 Baseline B: a symbol graph with name-match seeding.

    Graph memory as the term is normally used: nodes are definitions, edges are
    calls and imports, a query is matched against node names, and relevance
    spreads outward by traversal. It holds structure that Baseline A cannot
    represent, and that is the point of including it.

    What it does not hold is everything MCM adds on top of a graph: evidence,
    provenance, validity intervals, constraints, relation algebra, confidence. If
    MCM beats this, the margin is attributable to those; if it does not, the
    structure was doing the work all along and spec section 50's ablations should
    show the same thing from the other direction.
    """

    def __init__(self, *, decay: float = 0.5, depth: int = 2) -> None:
        self.name = "graph"
        self.decay = decay
        self.depth = depth
        self._nodes: dict[str, _Node] = {}
        self._edges: dict[str, set[str]] = {}
        self._by_name: dict[str, list[str]] = {}
        self._snapshot: RepoSnapshot | None = None

    def prepare(self, snapshot: RepoSnapshot) -> None:
        if self._snapshot is snapshot and self._nodes:
            return      # already indexed; Baseline D shares this instance
        self._snapshot = snapshot
        self._nodes, self._edges, self._by_name = {}, {}, {}
        # Nodes.
        for relpath, symbols in snapshot.definitions.items():
            for symbol in symbols:
                key = unit_id(relpath, symbol.qualname)
                self._nodes[key] = _Node(
                    unit=key, relpath=relpath, qualname=symbol.qualname,
                    name=symbol.name,
                    tokens=snapshot.unit_tokens.get(key, 0))
                self._by_name.setdefault(symbol.name.lower(), []).append(key)
                self._edges.setdefault(key, set())

        # Edges. Call sites resolved by plain name match, which is what a graph
        # memory without type inference can honestly do. Ambiguous names link to
        # every candidate; that is noise the architecture really has.
        for relpath in snapshot.files:
            try:
                parsed = parse_python(snapshot.sources[relpath], relpath)
            except Exception:  # noqa: BLE001 - unparsable file contributes no edges
                continue
            owners = sorted(snapshot.definitions.get(relpath, ()),
                            key=lambda s: s.start_line)
            for call in parsed.calls:
                if call.caller_qualname is None:
                    continue          # module-level call: no definition to link
                caller = unit_id(relpath, call.caller_qualname)
                if caller not in self._edges:
                    continue
                # ``jwt.encode`` resolves on "encode", ``validate_token`` on
                # itself. Name matching only - a graph memory without type
                # inference cannot do better, and pretending otherwise would be
                # giving the baseline an analysis it does not have.
                target = (call.attribute or call.base or "").lower()
                for candidate in self._by_name.get(target, ()):
                    if candidate != caller:
                        self._edges[caller].add(candidate)
                        self._edges.setdefault(candidate, set()).add(caller)
            # Same-file containment: a graph memory knows co-location.
            keys = [unit_id(relpath, s.qualname) for s in owners]
            for key in keys:
                for other in keys:
                    if key != other:
                        self._edges[key].add(other)

    def nodes(self) -> dict[str, _Node]:
        """The definition table, shared with Baseline D rather than rebuilt."""
        return self._nodes

    def answer(self, task: BenchmarkTask, *, budget: int) -> SystemAnswer:
        started = time.perf_counter()
        scores: dict[str, float] = {}

        if task.family == IMPACT and task.seed in self._nodes:
            seeds = {task.seed: 1.0}
        else:
            seeds = self._seed(query_for(task))

        frontier: deque[tuple[str, float, int]] = deque(
            (key, weight, 0) for key, weight in seeds.items())
        for key, weight in seeds.items():
            scores[key] = max(scores.get(key, 0.0), weight)
        while frontier:
            key, weight, hop = frontier.popleft()
            if hop >= self.depth:
                continue
            spread = weight * self.decay
            if spread <= 0.01:
                continue
            for neighbour in self._edges.get(key, ()):
                if scores.get(neighbour, 0.0) >= spread:
                    continue
                scores[neighbour] = spread
                frontier.append((neighbour, spread, hop + 1))

        if task.family == IMPACT and task.seed:
            scores.pop(task.seed, None)

        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return SystemAnswer(system=self.name,
                            units=_fill(ranked, self._nodes, budget),
                            latency_ms=(time.perf_counter() - started) * 1000)

    def _seed(self, query: str) -> dict[str, float]:
        """Name matching, exact then substring. No embeddings by design.

        Adding a vector channel here would make Baseline B a hybrid, which spec
        section 45 lists separately as Baseline D.
        """
        terms = [t.lower() for t in tokenize(query) if len(t) > 2]
        found: dict[str, float] = {}
        for term in terms:
            for key in self._by_name.get(term, ()):
                found[key] = 1.0
        if not found:
            for term in terms:
                for name, keys in self._by_name.items():
                    if term in name or name in term:
                        for key in keys:
                            found.setdefault(key, 0.6)
        return found


# --- Baseline D: hybrid graph + vector --------------------------------------

class BaselineHybrid:
    """Spec section 45 Baseline D: reciprocal rank fusion of A and B.

    RRF rather than a weighted score sum, because the two rankings are not on a
    common scale and normalising them would be one more tuned parameter with
    nothing to tune it against. RRF is the standard answer to that problem and
    needs no calibration, which keeps this an honest off-the-shelf hybrid rather
    than a strawman.
    """

    def __init__(self, *, k: int = 60,
                 vector: BaselineVectorRAG | None = None,
                 graph: BaselineGraph | None = None) -> None:
        self.name = "hybrid"
        self.k = k
        self.vector = vector or BaselineVectorRAG()
        self.graph = graph or BaselineGraph()
        self._nodes: dict[str, _Node] = {}

    def prepare(self, snapshot: RepoSnapshot) -> None:
        self.vector.prepare(snapshot)
        self.graph.prepare(snapshot)
        self._nodes = self.graph.nodes()

    def answer(self, task: BenchmarkTask, *, budget: int) -> SystemAnswer:
        started = time.perf_counter()
        # Ask both for a generous slice, then fuse. Fusing truncated lists would
        # measure the truncation rather than the fusion.
        wide = budget * 3
        left = self.vector.answer(task, budget=wide).ranked_definitions()
        right = self.graph.answer(task, budget=wide).ranked_definitions()

        fused: dict[str, float] = {}
        for ranking in (left, right):
            for position, definition in enumerate(ranking, start=1):
                fused[definition] = fused.get(definition, 0.0) + 1.0 / (self.k + position)

        if task.family == IMPACT and task.seed:
            fused.pop(task.seed, None)

        ranked = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
        return SystemAnswer(system=self.name,
                            units=_fill(ranked, self._nodes, budget),
                            latency_ms=(time.perf_counter() - started) * 1000)


# --- Baseline C: MCM --------------------------------------------------------

class MCMSystem:
    """Spec section 45 Baseline C: the system this study is about.

    Ingests the horizon revision through the real pipeline, builds both text
    projections, and answers through ``HybridRetriever``. The impact family is
    answered by dependency propagation seeded at the named symbol rather than by
    text retrieval, because that is the analysis MCM exists to perform and asking
    it to do keyword search for a structural question would be testing the wrong
    component.

    The store is built from ``GitRevisionProvider`` at the horizon SHA, so it sees
    exactly the corpus the baselines see. No working-tree checkout, and no path by
    which a later revision could leak in.
    """

    def __init__(self, *, weights: RetrievalWeights | None = None,
                 name: str = "mcm", max_depth: int = 6,
                 use_reasoning: bool = True,
                 provider: EmbeddingProvider | None = None,
                 on_disk: bool = False) -> None:
        self.name = name
        #: Keep the graph in a temporary file rather than in RAM. The in-memory
        #: store is faster and is the default, but it is also the largest single
        #: allocation in a run: roughly 1.8GB on django, on top of a GPU
        #: embedding model. On a large repository that is the difference between
        #: scoring and being killed.
        self.on_disk = on_disk
        self._store_dir: tempfile.TemporaryDirectory | None = None
        self.weights = weights
        self.max_depth = max_depth
        #: Shared with the baselines so the comparison stays about architecture
        #: rather than about who got the better embedding model.
        self.provider = provider or get_provider()
        #: Spec section 50 ablates symbolic reasoning. With this off, the impact
        #: family is answered by text retrieval like every other family, which is
        #: what MCM is without the analysis built on top of its graph.
        self.use_reasoning = use_reasoning
        self.store: SQLiteStore | None = None
        self._repo: str = ""
        self._retriever: HybridRetriever | None = None
        self._units: dict[str, _Node] = {}
        self._by_unit: dict[str, str] = {}     # unit id -> object id
        self._by_object: dict[str, str] = {}   # object id -> unit id

    def release(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None
        if self._store_dir is not None:
            self._store_dir.cleanup()
            self._store_dir = None
        self._retriever = None
        self._units, self._by_unit, self._by_object = {}, {}, {}

    def prepare(self, snapshot: RepoSnapshot) -> None:
        horizon = snapshot.horizon
        self._repo = horizon.repo
        reader = GitReader(horizon.root)
        if self.store is not None:
            # Six ablations each prepare their own store; without this they are
            # all held open at once for the length of a run.
            self.store.close()
        if self.on_disk:
            self._store_dir = tempfile.TemporaryDirectory(prefix="mcm-bench-")
            self.store = SQLiteStore(Path(self._store_dir.name) / "index.db")
        else:
            self.store = SQLiteStore(":memory:")
        RepositoryIngestor(self.store).ingest_source(
            GitRevisionProvider(reader, horizon.sha),
            repo=horizon.repo, root=horizon.root, moment=horizon.when)

        vector = VectorProjection(self.store, provider=self.provider)
        lexical = LexicalProjection(self.store)
        vector.rebuild()
        lexical.rebuild()
        self._retriever = HybridRetriever(self.store, vector=vector,
                                          lexical=lexical, weights=self.weights)

        self._units, self._by_unit, self._by_object = {}, {}, {}
        for relpath, symbols in snapshot.definitions.items():
            for symbol in symbols:
                key = unit_id(relpath, symbol.qualname)
                self._units[key] = _Node(
                    unit=key, relpath=relpath, qualname=symbol.qualname,
                    name=symbol.name, tokens=snapshot.unit_tokens.get(key, 0))
        for obj in self.store.all_objects():
            key = _unit_of(obj.id)
            if key is not None and key in self._units:
                self._by_unit[key] = obj.id
                self._by_object[obj.id] = key

    def answer(self, task: BenchmarkTask, *, budget: int) -> SystemAnswer:
        started = time.perf_counter()
        assert self._retriever is not None, "prepare() first"

        if task.family == IMPACT and task.seed and self.use_reasoning:
            ranked = self._impact(task)
        else:
            ranked = self._retrieve(task)

        if task.family == IMPACT and task.seed:
            ranked = [pair for pair in ranked if pair[0] != task.seed]

        return SystemAnswer(system=self.name,
                            units=_fill(ranked, self._units, budget),
                            latency_ms=(time.perf_counter() - started) * 1000)

    def set_weights(self, weights: RetrievalWeights | None) -> None:
        """Swap the section 26 scoring weights without rebuilding the index.

        Weights are applied at query time, so the store and both projections
        survive the change. Tuning would be unaffordable otherwise: re-ingesting
        per candidate weight vector costs about twenty seconds a trial.
        """
        self.weights = weights
        if self._retriever is not None:
            self._retriever.weights = weights or RetrievalWeights()

    def channel_scores(self, task: BenchmarkTask
                       ) -> list[tuple[str, tuple[float, float, float, float, float]]]:
        """Per-channel scores for every pooled candidate, before weighting.

        The separation that makes weight tuning cheap. Spec section 26's model is
        ``Score = w_v*V + w_l*L + w_g*G + w_s*S + w_p*P``, so the channel values
        are independent of the weights and one retrieval serves any number of
        weight vectors.

        The whole pool is returned, not a top-N slice. Truncating under the
        current weights would hide candidates that a different weighting would
        have promoted, and the tuner would then be searching a space shaped by the
        weights it started from.
        """
        assert self._retriever is not None, "prepare() first"
        result = self._retriever.retrieve(query_for(task), limit=1_000_000,
                                          pool_size=200)
        out: list[tuple[str, tuple[float, float, float, float, float]]] = []
        for candidate in result.candidates:
            key = self._by_object.get(candidate.object_id)
            if key is None:
                key = _unit_of(candidate.object_id)
            if key is not None and key in self._units:
                out.append((key, (candidate.vector, candidate.lexical,
                                  candidate.graph, candidate.symbolic,
                                  candidate.provenance)))
        return out

    def units(self) -> dict[str, _Node]:
        """The definition table, so a tuner can fill a budget without retrieving."""
        return self._units

    def _retrieve(self, task: BenchmarkTask) -> list[tuple[str, float]]:
        result = self._retriever.retrieve(query_for(task), limit=200, pool_size=200)
        ranked: list[tuple[str, float]] = []
        for candidate in result.candidates:
            key = self._by_object.get(candidate.object_id)
            if key is None:
                key = _unit_of(candidate.object_id)
            if key is not None and key in self._units:
                ranked.append((key, candidate.score))
        return ranked

    def _impact(self, task: BenchmarkTask) -> list[tuple[str, float]]:
        """Dependency propagation from the named symbol (spec section 29).

        Falls back to retrieval when the seed has no object, which happens when
        ingestion skipped the file. Reported as a fallback rather than silently,
        because an impact score produced by keyword search is not an impact
        analysis.
        """
        object_id = self._by_unit.get(task.seed or "")
        if object_id is None or self.store is None:
            return self._retrieve(task)
        try:
            impact = analyse_impact(self.store, object_id, max_depth=self.max_depth)
        except KeyError:
            return self._retrieve(task)
        ranked: list[tuple[str, float]] = []
        for affected in impact.all_affected:
            key = self._by_object.get(affected.object.id)
            if key is not None and key in self._units:
                ranked.append((key, affected.confidence))
        if not ranked:
            return self._retrieve(task)
        return sorted(ranked, key=lambda pair: (-pair[1], pair[0]))


def _unit_of(object_id: str) -> str | None:
    """MCM object ID -> benchmark unit ID, or None for non-definition objects.

    ``repo://name/pkg/mod.py#function:outer.inner`` -> ``pkg/mod.py:outer.inner``
    """
    if "#" not in object_id or not object_id.startswith("repo://"):
        return None
    path_part, _, tail = object_id.partition("#")
    kind, _, qualname = tail.partition(":")
    if kind not in ("function", "class", "method", "test") or not qualname:
        return None
    relpath = path_part.split("/", 3)[-1] if path_part.count("/") >= 3 else ""
    return f"{relpath}:{qualname}" if relpath else None


def _fill(ranked: list[tuple[str, float]], nodes: dict[str, _Node],
          budget: int) -> list[RetrievedUnit]:
    """Take definitions off a ranking until the token budget is gone.

    A definition too large for the remaining budget is skipped rather than
    truncated, and the walk continues to smaller ones. Truncating would deliver a
    fragment and charge full price for it.
    """
    units: list[RetrievedUnit] = []
    spent = 0
    for key, score in ranked:
        node = nodes.get(key)
        if node is None:
            continue
        cost = max(node.tokens, 1)
        if spent + cost > budget:
            continue
        units.append(RetrievedUnit(
            unit_id=key, relpath=node.relpath, tokens=cost,
            covers={key: node.tokens}, detail=f"score {score:.3f}"))
        spent += cost
        if spent >= budget:
            break
    return units


def fill_budget(ranked: list[tuple[str, float]], nodes: dict[str, _Node],
                budget: int) -> list[RetrievedUnit]:
    """Public name for the budget filler, so a tuner charges the same prices."""
    return _fill(ranked, nodes, budget)


def default_systems(provider: EmbeddingProvider | None = None, *,
                    on_disk: bool = False) -> list[BenchmarkSystem]:
    """The spec section 45 line-up, in the order the hypothesis names them.

    Every system that embeds anything is handed the *same* provider. Section 45
    is a comparison of architectures, and giving MCM a trained model while the
    vector baseline keeps feature hashing would compare embeddings instead.
    """
    provider = provider or get_provider()
    vector = BaselineVectorRAG(provider=provider)
    graph = BaselineGraph()
    return [vector, graph, BaselineHybrid(vector=vector, graph=graph),
            MCMSystem(provider=provider, on_disk=on_disk)]
