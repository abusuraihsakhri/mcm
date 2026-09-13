"""Evaluation framework (spec sections 45 to 50, development step 20).

This is the only part of the system that can test the spec section 3 hypothesis,
and section 3 is explicit that the experiment must be able to come out the other
way::

    H1: MCM + hybrid retrieval + inference > Graph memory > Vector RAG

    Do NOT assume this hierarchy is true.

Three commitments follow from that, and they shape every module here.

**Baselines are independent implementations, not ablations.** Baseline A chunks
raw file text on fixed line windows and retrieves by cosine similarity, the way a
vector RAG pipeline actually works. Baseline B builds its own symbol graph with
name-match seeding and breadth-first expansion. Neither borrows MCM's document
construction or its route map. Had they been built as ``RetrievalWeights`` with
channels zeroed, they would inherit MCM's symbol-aware chunking and the ranking
would be settled before a single task ran. Ablations of that kind belong to spec
section 50 and live in ``ablation.py``, where they are labelled as ablations.

**Ground truth is held-out Git history.** A repository's history is split at a
horizon commit. Every system indexes the repository as it stood at the horizon;
tasks are drawn from commits *after* it. The answer to "what should change to do
X" is what the commit that did X actually changed. No system can see it, nobody
hand-picked it, and re-running on a different horizon is one argument.

**Systems do not grade themselves.** A system returns what it would put in front
of an agent and how many tokens that costs. The harness decides what was useful.

Not covered, and named rather than quietly dropped: spec section 46's families B
(dependency), E (constraint), F (causal) and G (equivalence) have no ground truth
derivable from commit history. Deriving dependency truth from an AST pass would
score MCM near-perfectly by construction, because MCM's graph is built from that
same analysis. See ``docs/evaluation.md``.
"""

from .metrics import (ContextEfficiency, RetrievalMetrics, TaskScore,
                      paired_bootstrap, token_estimate)
from .systems import (BaselineGraph, BaselineHybrid, BaselineVectorRAG,
                      BenchmarkSystem, MCMSystem, RetrievedUnit, SystemAnswer)
from .tasks import (BenchmarkTask, Horizon, RepoSnapshot, TaskSuite,
                    generate_tasks)

__all__ = [
    "BenchmarkTask", "Horizon", "RepoSnapshot", "TaskSuite", "generate_tasks",
    "BenchmarkSystem", "SystemAnswer", "RetrievedUnit",
    "BaselineVectorRAG", "BaselineGraph", "BaselineHybrid", "MCMSystem",
    "RetrievalMetrics", "ContextEfficiency", "TaskScore",
    "token_estimate", "paired_bootstrap",
]
