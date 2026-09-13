"""Hyperparameter tuning on a held-out repository (spec sections 26, 49).

Spec section 26 leaves every weight in the scoring model configurable and declines
to choose values. Until now they were unfitted guesses, and the section 50
ablations could not distinguish "this component does not help" from "the default
weights spend score on it badly". This module settles that by fitting on a
repository that is not one of the ones being reported.

**Every system is tuned, not just MCM.** Fitting MCM's weights while leaving the
baselines at whatever constants were typed first would turn the evaluation into
tuned-versus-untuned, which is the same class of error as building the baselines
out of MCM's own retriever. Baseline A gets its chunk geometry fitted, Baseline B
its traversal, Baseline D its fusion constant, on the same tasks, against the same
objective, by the same search.

**The tuning repository is never evaluated.** Fitting on httpx and reporting on
itsdangerous and flask keeps every number in ``docs/evaluation.md`` a held-out
score. The tuning-set gain is reported next to the held-out gain precisely so that
overfitting is visible: a large gain on the tuning repository that does not
transfer is the expected failure here, and it should be legible rather than
flattering.

**Why the search is affordable.** Spec section 26's model is linear in the
channels::

    Score(x|q) = w_v*V + w_l*L + w_g*G + w_s*S + w_p*P

The channel values do not depend on the weights, so retrieval runs once per task
and any number of weight vectors are scored against the captured values for free.
Only ``graph_decay`` and ``graph_depth`` change the channel values themselves, so
those form a small outer loop that does pay for re-retrieval. One outer step costs
a pass over the tasks; the inner sweep over weight vectors costs a re-sort.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from ..retrieval.hybrid import RetrievalWeights
from .metrics import mean
from .runner import score_answer
from .systems import (BaselineGraph, BaselineHybrid, BaselineVectorRAG,
                      MCMSystem, SystemAnswer, fill_budget)
from .tasks import IMPACT, TaskSuite

#: Graph traversal settings searched in the outer loop, where re-retrieval is
#: unavoidable because they change the channel's values rather than its weight.
DECAYS = (0.3, 0.5, 0.7)
DEPTHS = (1, 2, 3)

#: Chunk geometries for Baseline A. Window and stride are both index-time, so each
#: pair costs a re-embed of the corpus.
CHUNK_GEOMETRIES = ((20, 15), (30, 20), (40, 30), (40, 40), (60, 45), (80, 60))

#: Fusion constants for Baseline D. The literature default is 60.
RRF_KS = (10, 20, 40, 60, 100)


@dataclass(frozen=True)
class TunedConfiguration:
    """What the search chose, for every system."""

    weights: RetrievalWeights
    chunk_lines: int
    stride: int
    graph_decay: float
    graph_depth: int
    rrf_k: int

    def to_json(self) -> dict:
        w = self.weights
        return {
            "mcm_weights": {
                "vector": round(w.vector, 4), "lexical": round(w.lexical, 4),
                "graph": round(w.graph, 4), "symbolic": round(w.symbolic, 4),
                "provenance": round(w.provenance, 4),
                "graph_decay": w.graph_decay, "graph_depth": w.graph_depth,
            },
            "vector_rag": {"chunk_lines": self.chunk_lines, "stride": self.stride},
            "graph": {"decay": self.graph_decay, "depth": self.graph_depth},
            "hybrid": {"rrf_k": self.rrf_k},
        }

    @property
    def weight_spec(self) -> str:
        """The ``--weights`` string that reproduces this MCM configuration."""
        w = self.weights
        return (f"v={w.vector:.4f},l={w.lexical:.4f},g={w.graph:.4f},"
                f"s={w.symbolic:.4f},p={w.provenance:.4f},"
                f"graph_decay={w.graph_decay},graph_depth={w.graph_depth}")


@dataclass
class Trial:
    label: str
    score: float
    detail: dict = field(default_factory=dict)


@dataclass
class TuningReport:
    repo: str
    horizon: str
    objective: str
    tasks: int
    configuration: TunedConfiguration
    #: system -> (default score on the tuning repo, tuned score on it)
    gains: dict[str, tuple[float, float]] = field(default_factory=dict)
    trials: dict[str, list[Trial]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Tuned on {self.repo} @ {self.horizon} "
                 f"({self.tasks} tasks, objective {self.objective})"]
        for system, (before, after) in self.gains.items():
            delta = after - before
            lines.append(f"  {system:<14} {before:.4f} -> {after:.4f} "
                         f"({delta:+.4f})")
        lines.append(f"  mcm weights: {self.configuration.weight_spec}")
        cfg = self.configuration
        lines.append(f"  vector-rag: chunk={cfg.chunk_lines} stride={cfg.stride}; "
                     f"graph: decay={cfg.graph_decay} depth={cfg.graph_depth}; "
                     f"hybrid: k={cfg.rrf_k}")
        return "\n".join(lines)


def _objective(suite: TaskSuite, answers: dict[str, SystemAnswer], *,
               metric: str, k: int) -> float:
    """Mean metric over the tuning tasks, scored by the ordinary harness.

    Tuning against a different scorer than the evaluation uses would fit the
    wrong thing, so ``score_answer`` is reused rather than reimplemented.
    """
    values = []
    for task in suite.tasks:
        answer = answers.get(task.task_id)
        if answer is not None:
            values.append(score_answer(task, answer, k=k).value(metric))
    return mean(values)


# --- MCM --------------------------------------------------------------------

def _sample_weights(rng: random.Random, samples: int) -> list[tuple[float, ...]]:
    """Weight vectors on the 5-simplex, plus the current defaults.

    Random search rather than a grid: five continuous dimensions make a grid of
    any useful resolution unaffordable, and random search covers a simplex better
    than a coarse grid at equal cost. The defaults are included explicitly so the
    report can say whether tuning beat them or merely matched them.
    """
    default = RetrievalWeights()
    vectors = [(default.vector, default.lexical, default.graph,
                default.symbolic, default.provenance)]
    # Corners: each channel alone. These are the configurations an ablation would
    # reach, included so the search cannot miss a degenerate optimum.
    for index in range(5):
        vectors.append(tuple(1.0 if i == index else 0.0 for i in range(5)))
    for _ in range(samples):
        raw = [rng.random() for _ in range(5)]
        total = sum(raw) or 1.0
        vectors.append(tuple(value / total for value in raw))
    return vectors


def compute_top_basin_centroid(
    evaluated: list[tuple[float, RetrievalWeights]],
    top_k: int = 5,
) -> RetrievalWeights:
    """Compute the centroid of weights in the top-performing basin.

    Finds the (decay, depth) basin with the highest peak score, gathers its
    top-k scoring weight vectors, and returns their mean vector normalized
    to sum to 1.0 on the 5-simplex. This regularizes against selection bias
    over the 1,404-configuration search space on small or noisy tuning suites.
    """
    if not evaluated:
        return RetrievalWeights()
    if top_k <= 1:
        return max(evaluated, key=lambda pair: pair[0])[1]

    basins: dict[tuple[float, int], list[tuple[float, RetrievalWeights]]] = {}
    for score, weights in evaluated:
        key = (weights.graph_decay, weights.graph_depth)
        basins.setdefault(key, []).append((score, weights))

    best_basin_key = max(basins.keys(), key=lambda k: max(score for score, _ in basins[k]))
    best_candidates = sorted(basins[best_basin_key], key=lambda pair: -pair[0])[:top_k]

    count = len(best_candidates)
    v_mean = sum(w.vector for _, w in best_candidates) / count
    l_mean = sum(w.lexical for _, w in best_candidates) / count
    g_mean = sum(w.graph for _, w in best_candidates) / count
    s_mean = sum(w.symbolic for _, w in best_candidates) / count
    p_mean = sum(w.provenance for _, w in best_candidates) / count

    total = v_mean + l_mean + g_mean + s_mean + p_mean or 1.0
    decay, depth = best_basin_key

    return RetrievalWeights(
        vector=v_mean / total,
        lexical=l_mean / total,
        graph=g_mean / total,
        symbolic=s_mean / total,
        provenance=p_mean / total,
        graph_decay=decay,
        graph_depth=depth,
    )


def tune_mcm(suite: TaskSuite, system: MCMSystem, *, budget: int, k: int,
             metric: str, samples: int = 150, seed: int = 20250911,
             regularize: bool = True, top_basin_k: int = 5,
             progress: bool = False) -> tuple[RetrievalWeights, float, float,
                                              list[Trial]]:
    """Fit the section 26 weights and the graph traversal settings.

    When regularize is True, computes the centroid of top candidates in the
    highest-scoring traversal basin to prevent overfitting the 1,404-configuration
    search space. Returns the chosen weights, default score, tuned score, and trials.
    """
    rng = random.Random(seed)
    candidates = _sample_weights(rng, samples)
    nodes = system.units()
    trials: list[Trial] = []
    all_evaluated: list[tuple[float, RetrievalWeights]] = []

    best: tuple[float, RetrievalWeights] | None = None
    default_score: float | None = None
    default_weights = RetrievalWeights()

    for decay in DECAYS:
        for depth in DEPTHS:
            # One retrieval pass per traversal setting; channel values captured.
            system.set_weights(RetrievalWeights(graph_decay=decay,
                                                graph_depth=depth))
            captured: dict[str, list[tuple[str, tuple[float, ...]]]] = {}
            impact_answers: dict[str, SystemAnswer] = {}
            for task in suite.tasks:
                if task.family == IMPACT and task.seed and system.use_reasoning:
                    # Impact is answered by propagation, which no weight touches.
                    # Computed once here so the objective still covers it.
                    impact_answers[task.task_id] = system.answer(task, budget=budget)
                else:
                    captured[task.task_id] = system.channel_scores(task)
            if progress:
                print(f"    captured decay={decay} depth={depth}", flush=True)

            use_gpu = torch is not None and torch.cuda.is_available()
            task_precomputed = {}
            if use_gpu and captured:
                device = torch.device("cuda:0")
                w_mat = torch.tensor(candidates, dtype=torch.float32, device=device)
                for task_id, channels in captured.items():
                    if channels:
                        keys = [k for k, _ in channels]
                        c_mat = torch.tensor([c for _, c in channels], dtype=torch.float32, device=device)
                        scores_tensor = torch.matmul(w_mat, c_mat.T)
                        task_precomputed[task_id] = (keys, scores_tensor.cpu().tolist())

            for cand_idx, weights_vector in enumerate(candidates):
                weights = RetrievalWeights(
                    vector=weights_vector[0], lexical=weights_vector[1],
                    graph=weights_vector[2], symbolic=weights_vector[3],
                    provenance=weights_vector[4],
                    graph_decay=decay, graph_depth=depth)
                answers = dict(impact_answers)
                for task_id, channels in captured.items():
                    if task_id in task_precomputed:
                        keys, all_cand_scores = task_precomputed[task_id]
                        cand_scores = all_cand_scores[cand_idx]
                        ranked = sorted(zip(keys, cand_scores),
                                        key=lambda pair: (-pair[1], pair[0]))
                    else:
                        ranked = sorted(
                            ((key, weights_vector[0] * c[0] + weights_vector[1] * c[1]
                              + weights_vector[2] * c[2] + weights_vector[3] * c[3]
                              + weights_vector[4] * c[4])
                             for key, c in channels),
                            key=lambda pair: (-pair[1], pair[0]))
                    answers[task_id] = SystemAnswer(
                        system=system.name,
                        units=fill_budget(ranked, nodes, budget))
                score = _objective(suite, answers, metric=metric, k=k)
                label = (f"v={weights.vector:.2f} l={weights.lexical:.2f} "
                         f"g={weights.graph:.2f} s={weights.symbolic:.2f} "
                         f"p={weights.provenance:.2f} d={decay} h={depth}")
                trials.append(Trial(label=label, score=score))
                all_evaluated.append((score, weights))
                if (decay == default_weights.graph_decay
                        and depth == default_weights.graph_depth
                        and weights_vector is candidates[0]):
                    default_score = score
                if best is None or score > best[0]:
                    best = (score, weights)

    if regularize and all_evaluated:
        centroid_weights = compute_top_basin_centroid(all_evaluated, top_k=top_basin_k)
        system.set_weights(centroid_weights)
        centroid_answers = {t.task_id: system.answer(t, budget=budget) for t in suite.tasks}
        centroid_score = _objective(suite, centroid_answers, metric=metric, k=k)
        trials.append(Trial(
            label=f"CENTROID v={centroid_weights.vector:.2f} l={centroid_weights.lexical:.2f} "
                  f"g={centroid_weights.graph:.2f} s={centroid_weights.symbolic:.2f} "
                  f"p={centroid_weights.provenance:.2f} d={centroid_weights.graph_decay} "
                  f"h={centroid_weights.graph_depth}",
            score=centroid_score,
            detail={"regularized": True, "top_basin_k": top_basin_k},
        ))
        chosen_weights = centroid_weights
        chosen_score = centroid_score
    else:
        chosen_weights = best[1] if best else default_weights
        chosen_score = best[0] if best else 0.0

    system.set_weights(chosen_weights)
    return (chosen_weights,
            default_score if default_score is not None else 0.0,
            chosen_score,
            trials)


# --- the baselines ----------------------------------------------------------

def tune_vector_rag(suite: TaskSuite, *, budget: int, k: int, metric: str,
                    progress: bool = False
                    ) -> tuple[tuple[int, int], float, float, list[Trial]]:
    """Fit Baseline A's chunk window and stride."""
    trials: list[Trial] = []
    best: tuple[float, tuple[int, int]] | None = None
    default_score = 0.0
    for chunk_lines, stride in CHUNK_GEOMETRIES:
        system = BaselineVectorRAG(chunk_lines=chunk_lines, stride=stride)
        system.prepare(suite.snapshot)
        answers = {t.task_id: system.answer(t, budget=budget) for t in suite.tasks}
        score = _objective(suite, answers, metric=metric, k=k)
        trials.append(Trial(label=f"chunk={chunk_lines} stride={stride}",
                            score=score))
        if (chunk_lines, stride) == (40, 30):
            default_score = score
        if best is None or score > best[0]:
            best = (score, (chunk_lines, stride))
        if progress:
            print(f"    vector-rag chunk={chunk_lines} stride={stride}: "
                  f"{score:.4f}", flush=True)
    return best[1], default_score, best[0], trials


def tune_graph(suite: TaskSuite, *, budget: int, k: int, metric: str,
               progress: bool = False
               ) -> tuple[tuple[float, int], float, float, list[Trial]]:
    """Fit Baseline B's decay and depth.

    Cheap: the node and edge tables are built once and only traversal changes.
    """
    system = BaselineGraph()
    system.prepare(suite.snapshot)
    trials: list[Trial] = []
    best: tuple[float, tuple[float, int]] | None = None
    default_score = 0.0
    for decay in DECAYS:
        for depth in DEPTHS:
            system.decay, system.depth = decay, depth
            answers = {t.task_id: system.answer(t, budget=budget)
                       for t in suite.tasks}
            score = _objective(suite, answers, metric=metric, k=k)
            trials.append(Trial(label=f"decay={decay} depth={depth}", score=score))
            if (decay, depth) == (0.5, 2):
                default_score = score
            if best is None or score > best[0]:
                best = (score, (decay, depth))
    if progress:
        print(f"    graph best {best[1]}: {best[0]:.4f}", flush=True)
    return best[1], default_score, best[0], trials


def tune_hybrid(suite: TaskSuite, geometry: tuple[int, int],
                traversal: tuple[float, int], *, budget: int, k: int,
                metric: str, progress: bool = False
                ) -> tuple[int, float, float, list[Trial]]:
    """Fit Baseline D's fusion constant, over its already-tuned components.

    The hybrid is given the tuned vector and graph baselines rather than the
    defaults, because the fair version of "graph plus vector" is the best graph
    plus the best vector.
    """
    vector = BaselineVectorRAG(chunk_lines=geometry[0], stride=geometry[1])
    graph = BaselineGraph(decay=traversal[0], depth=traversal[1])
    trials: list[Trial] = []
    best: tuple[float, int] | None = None
    default_score = 0.0
    for rrf_k in RRF_KS:
        system = BaselineHybrid(k=rrf_k, vector=vector, graph=graph)
        system.prepare(suite.snapshot)
        answers = {t.task_id: system.answer(t, budget=budget) for t in suite.tasks}
        score = _objective(suite, answers, metric=metric, k=k)
        trials.append(Trial(label=f"rrf_k={rrf_k}", score=score))
        if rrf_k == 60:
            default_score = score
        if best is None or score > best[0]:
            best = (score, rrf_k)
    if progress:
        print(f"    hybrid best k={best[1]}: {best[0]:.4f}", flush=True)
    return best[1], default_score, best[0], trials


# --- the whole search -------------------------------------------------------

def tune(suite: TaskSuite, *, budget: int = 4000, k: int = 10,
         metric: str = "efficiency", samples: int = 150, seed: int = 20250911,
         regularize: bool = True, top_basin_k: int = 5,
         progress: bool = False) -> TuningReport:
    """Fit every system on this suite. The suite must not be an evaluation repo."""
    report = TuningReport(repo=suite.horizon.repo, horizon=suite.horizon.short,
                          objective=metric, tasks=len(suite.tasks),
                          configuration=TunedConfiguration(
                              RetrievalWeights(), 40, 30, 0.5, 2, 60))

    if progress:
        print("  tuning vector-rag", flush=True)
    geometry, va, vb, v_trials = tune_vector_rag(
        suite, budget=budget, k=k, metric=metric, progress=progress)

    if progress:
        print("  tuning graph", flush=True)
    traversal, ga, gb, g_trials = tune_graph(
        suite, budget=budget, k=k, metric=metric, progress=progress)

    if progress:
        print("  tuning hybrid", flush=True)
    rrf_k, ha, hb, h_trials = tune_hybrid(
        suite, geometry, traversal, budget=budget, k=k, metric=metric,
        progress=progress)

    if progress:
        print("  tuning mcm (this is the slow one)", flush=True)
    mcm = MCMSystem(name="mcm")
    mcm.prepare(suite.snapshot)
    weights, ma, mb, m_trials = tune_mcm(
        suite, mcm, budget=budget, k=k, metric=metric, samples=samples,
        seed=seed, regularize=regularize, top_basin_k=top_basin_k,
        progress=progress)
    if mcm.store is not None:
        mcm.store.close()

    report.configuration = TunedConfiguration(
        weights=weights, chunk_lines=geometry[0], stride=geometry[1],
        graph_decay=traversal[0], graph_depth=traversal[1], rrf_k=rrf_k)
    report.gains = {"vector-rag": (va, vb), "graph": (ga, gb),
                    "hybrid": (ha, hb), "mcm": (ma, mb)}
    report.trials = {"vector-rag": v_trials, "graph": g_trials,
                     "hybrid": h_trials, "mcm": m_trials}
    return report


def tuned_systems(configuration: TunedConfiguration) -> list:
    """The spec section 45 line-up, each system carrying its fitted settings."""
    vector = BaselineVectorRAG(chunk_lines=configuration.chunk_lines,
                               stride=configuration.stride)
    graph = BaselineGraph(decay=configuration.graph_decay,
                          depth=configuration.graph_depth)
    return [vector, graph,
            BaselineHybrid(k=configuration.rrf_k, vector=vector, graph=graph),
            MCMSystem(weights=configuration.weights)]


def write_configuration(report: TuningReport, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tuned_on": report.repo,
        "horizon": report.horizon,
        "objective": report.objective,
        "tasks": report.tasks,
        "configuration": report.configuration.to_json(),
        "weight_spec": report.configuration.weight_spec,
        "tuning_set_gains": {
            system: {"default": round(before, 6), "tuned": round(after, 6),
                     "gain": round(after - before, 6)}
            for system, (before, after) in report.gains.items()
        },
        "top_trials": {
            system: [{"label": t.label, "score": round(t.score, 6)}
                     for t in sorted(trials, key=lambda t: -t.score)[:10]]
            for system, trials in report.trials.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_configuration(path: Path | str) -> TunedConfiguration:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = payload["configuration"]
    w = cfg["mcm_weights"]
    return TunedConfiguration(
        weights=RetrievalWeights(
            vector=w["vector"], lexical=w["lexical"], graph=w["graph"],
            symbolic=w["symbolic"], provenance=w["provenance"],
            graph_decay=w["graph_decay"], graph_depth=w["graph_depth"]),
        chunk_lines=cfg["vector_rag"]["chunk_lines"],
        stride=cfg["vector_rag"]["stride"],
        graph_decay=cfg["graph"]["decay"], graph_depth=cfg["graph"]["depth"],
        rrf_k=cfg["hybrid"]["rrf_k"])
