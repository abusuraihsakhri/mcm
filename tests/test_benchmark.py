"""Tests for the evaluation framework (spec sections 45 to 50, step 20).

The benchmark is the one component whose output is a claim about the other
components, so its own failure mode is different from theirs: a retrieval bug
returns bad answers, a benchmark bug returns a bad *conclusion*, and a bad
conclusion looks exactly like a good one. Most of what follows therefore tests
properties of the experiment rather than the arithmetic.

The properties that matter:

* ground truth is unreachable from the indexed revision (the no-leak property)
* the history split runs in the direction it claims to
* a system cannot influence its own score
* the comparison can return all three spec section 3 outcomes, including the two
  that contradict the hypothesis
* baselines are independent implementations, not MCM with channels switched off
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from mcm.benchmark.ablation import NOT_EXERCISED, ablations, contributions
from mcm.benchmark.metrics import (ContextEfficiency, paired_bootstrap,
                                   score_ranking, token_estimate)
from mcm.benchmark.runner import (GREATER, LESS, SIMILAR, chain_verdict, report,
                                  run_suite, run_json, score_answer, verdict)
from mcm.benchmark.systems import (BaselineGraph, BaselineVectorRAG, MCMSystem,
                                   RetrievedUnit, SystemAnswer, default_systems)
from mcm.benchmark.tasks import (COVERAGE_FLOOR, FAMILIES, IMPACT, PLANNING,
                                 generate_tasks, query_for)
from mcm.ingestion.git import GitReader


# --- a repository with enough history to split -----------------------------

def _run(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True,
                   capture_output=True, text=True)


@pytest.fixture(scope="module")
def repo(tmp_path_factory) -> Path:
    """A repository whose later commits change definitions its earlier ones had.

    Built rather than borrowed because the split has to be exercised at a known
    place: the tests below need to know which definitions existed at the horizon
    and which arrived afterwards, and a real repository would make that a moving
    target.
    """
    root = tmp_path_factory.mktemp("bench-repo")
    _run(root, "init", "-q")
    _run(root, "config", "user.email", "t@example.com")
    _run(root, "config", "user.name", "Test")

    (root / "core.py").write_text(
        "def alpha(x):\n    return x + 1\n\n\n"
        "def beta(x):\n    return alpha(x) * 2\n", encoding="utf-8")
    (root / "helper.py").write_text(
        "def gamma(y):\n    return y - 1\n", encoding="utf-8")
    _run(root, "add", "-A")
    _run(root, "commit", "-q", "-m", "initial core and helper definitions")

    # Padding, so the horizon can sit well before the tasks.
    for index in range(6):
        (root / f"pad{index}.py").write_text(
            f"def pad_{index}():\n    return {index}\n", encoding="utf-8")
        _run(root, "add", "-A")
        _run(root, "commit", "-q", "-m", f"add padding module number {index}")

    (root / "core.py").write_text(
        "def alpha(x):\n    return x + 100\n\n\n"
        "def beta(x):\n    return alpha(x) * 3\n", encoding="utf-8")
    _run(root, "add", "-A")
    _run(root, "commit", "-q", "-m", "change alpha offset and beta multiplier")

    (root / "helper.py").write_text(
        "def gamma(y):\n    return y - 2\n\n\n"
        "def delta(y):\n    return gamma(y)\n", encoding="utf-8")
    _run(root, "add", "-A")
    _run(root, "commit", "-q", "-m", "fix gamma decrement and add delta wrapper")
    return root


@pytest.fixture(scope="module")
def suite(repo):
    return generate_tasks(repo, repo="bench", horizon_back=2, max_tasks=10)


# --- the no-leak property ---------------------------------------------------

class TestNoLeak:
    """The reason this benchmark can be believed at all."""

    def test_tasks_come_from_after_the_horizon(self, repo, suite):
        reader = GitReader(repo)
        history = [c.sha for c in reader.commits()]
        horizon_index = history.index(suite.horizon.sha)
        for task in suite.tasks:
            assert history.index(task.commit_sha) > horizon_index

    def test_the_snapshot_is_the_horizon_revision_not_the_working_tree(
            self, repo, suite):
        """A working-tree snapshot would contain every answer.

        ``delta`` is added by the final commit. If the snapshot were read from
        disk it would be present, and every system would be able to retrieve a
        definition that did not exist when it indexed.
        """
        assert "helper.py:delta" not in suite.snapshot.unit_ids
        assert "helper.py:gamma" in suite.snapshot.unit_ids

    def test_unreachable_truth_is_excluded_from_targets(self, suite):
        for task in suite.tasks:
            assert task.reachable_definitions <= suite.snapshot.unit_ids
            assert task.targets <= (task.reachable_definitions
                                    | task.reachable_files)

    def test_ceiling_reports_how_much_truth_was_unreachable(self, suite):
        """A task whose answer is partly new code says so rather than averaging
        the impossible part into everyone's recall."""
        for task in suite.tasks:
            assert 0.0 < task.ceiling <= 1.0
            if task.truth_definitions != task.reachable_definitions:
                assert task.ceiling < 1.0


class TestHistorySplit:
    """The horizon counts back from HEAD, and reading it the other way is silent."""

    def test_horizon_is_near_head_not_near_the_root(self, repo, suite):
        reader = GitReader(repo)
        history = [c.sha for c in reader.commits()]   # oldest first
        position = history.index(suite.horizon.sha)
        assert position > len(history) // 2, (
            "horizon landed in early history: commits() is oldest-first and the "
            "split must count back from the end")

    def test_a_shallow_history_is_refused_rather_than_truncated(self, repo):
        with pytest.raises(ValueError, match="need at least"):
            generate_tasks(repo, repo="bench", horizon_back=10_000)

    def test_rejections_are_counted_not_discarded(self, suite):
        assert suite.considered >= len(
            {t.commit_sha for t in suite.tasks})
        assert all(count > 0 for count in suite.rejected.values())


class TestTaskGeneration:
    def test_families_are_drawn_from_the_supported_set(self, suite):
        assert suite.tasks
        assert {t.family for t in suite.tasks} <= set(FAMILIES)

    def test_impact_never_asks_for_the_symbol_it_names(self, suite):
        for task in suite.of_family(IMPACT):
            assert task.seed is not None
            assert task.seed not in task.targets

    def test_planning_is_scored_on_files(self, suite):
        for task in suite.of_family(PLANNING):
            assert task.granularity == "file"
            assert all("/" in t or t.endswith(".py") for t in task.targets)

    def test_impact_query_names_the_symbol(self, suite):
        for task in suite.of_family(IMPACT):
            assert task.seed.split(":")[-1] in query_for(task)

    def test_chore_commits_are_rejected(self, tmp_path):
        """Filtering is on the query text, decided before any system ran."""
        root = tmp_path / "chore"
        root.mkdir()
        _run(root, "init", "-q")
        _run(root, "config", "user.email", "t@example.com")
        _run(root, "config", "user.name", "Test")
        (root / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        _run(root, "add", "-A")
        _run(root, "commit", "-q", "-m", "initial module with one function")
        for index in range(4):
            (root / f"p{index}.py").write_text(f"def p{index}():\n    pass\n",
                                               encoding="utf-8")
            _run(root, "add", "-A")
            _run(root, "commit", "-q", "-m", f"add padding module {index}")
        (root / "m.py").write_text("def f():\n    return 2\n", encoding="utf-8")
        _run(root, "add", "-A")
        _run(root, "commit", "-q", "-m", "fix mypy findings in the module")

        made = generate_tasks(root, repo="chore", horizon_back=1)
        assert not made.tasks
        assert made.rejected.get("tooling or dependency chore") == 1


# --- scoring ----------------------------------------------------------------

class TestRetrievalMetrics:
    def test_recall_precision_and_mrr(self):
        m = score_ranking(["a", "b", "c", "d"], frozenset({"b", "d"}), k=4)
        assert m.recall == 1.0
        assert m.precision == 0.5
        assert m.mrr == pytest.approx(0.5)     # "b" is second

    def test_recall_is_capped_by_k_not_inflated(self):
        m = score_ranking(["a", "b", "c"], frozenset({"a", "b", "c"}), k=1)
        assert m.recall == pytest.approx(1 / 3)

    def test_no_targets_scores_zero_rather_than_dividing_by_zero(self):
        assert score_ranking(["a"], frozenset()).recall == 0.0

    def test_mrr_uses_the_full_ranking(self):
        """MRR is a property of where the first hit is, not of the top-K window."""
        assert score_ranking(["x"] * 20 + ["hit"], frozenset({"hit"}),
                             k=5).mrr == pytest.approx(1 / 21)


class TestContextEfficiency:
    def test_ratio_is_useful_over_total(self):
        assert ContextEfficiency(30, 120, 3).ratio == 0.25

    def test_returning_nothing_scores_zero_not_one(self):
        """Spec section 48 measures information delivered.

        An empty answer has retrieved no useless information, which a naive
        reading would call perfect efficiency.
        """
        assert ContextEfficiency(0, 0, 0).ratio == 0.0

    def test_useful_is_the_target_size_not_the_container_size(self):
        """The whole point of the metric. A big chunk delivering a small answer
        is credited with the small answer."""
        answer = SystemAnswer(system="s", units=[
            RetrievedUnit(unit_id="f.py#1-90", relpath="f.py", tokens=900,
                          covers={"f.py:wanted": 40, "f.py:other": 60}),
        ])
        assert answer.useful_tokens(frozenset({"f.py:wanted"}), "definition") == 40
        assert answer.total_tokens == 900

    def test_a_target_delivered_twice_is_counted_once(self):
        answer = SystemAnswer(system="s", units=[
            RetrievedUnit("a", "f.py", 100, {"f.py:x": 40}),
            RetrievedUnit("b", "f.py", 100, {"f.py:x": 40}),
        ])
        assert answer.useful_tokens(frozenset({"f.py:x"}), "definition") == 40


class TestTokenEstimate:
    def test_counts_identifiers_and_punctuation(self):
        assert token_estimate("def f(x):") == 6      # def f ( x ) :

    def test_is_stable_and_shared(self):
        """Every system is measured with this, so its bias is common-mode."""
        assert token_estimate("a b") == token_estimate("a  b") == 2


class TestScoringIsNotSelfReported:
    def test_a_system_cannot_change_its_own_score(self, suite):
        """``score_answer`` reads ground truth and the answer, nothing else."""
        task = suite.tasks[0]
        honest = SystemAnswer(system="x", units=[
            RetrievedUnit(unit_id=t, relpath=t.split(":")[0], tokens=10,
                          covers={t: 10})
            for t in sorted(task.targets)])
        # Same units, but the system claims a flattering note and zero latency.
        boastful = SystemAnswer(system="x", units=list(honest.units),
                                note="best answer ever", latency_ms=0.0)
        assert (score_answer(task, honest).retrieval.recall
                == score_answer(task, boastful).retrieval.recall)


# --- statistics -------------------------------------------------------------

class TestPairedBootstrap:
    def test_is_deterministic(self):
        left, right = [0.4, 0.6, 0.8, 0.2], [0.1, 0.2, 0.3, 0.1]
        assert (paired_bootstrap(left, right).low
                == paired_bootstrap(left, right).low)

    def test_pairs_must_line_up(self):
        with pytest.raises(ValueError, match="unpaired"):
            paired_bootstrap([1.0, 2.0], [1.0])

    def test_empty_input_is_not_a_result(self):
        assert paired_bootstrap([], []).n == 0

    def test_a_clear_win_separates(self):
        c = paired_bootstrap([0.9] * 20, [0.1] * 20)
        assert c.separated and c.difference > 0
        assert verdict(c) == GREATER

    def test_a_clear_loss_reports_less_than(self):
        """Spec section 3 requires MCM < Graph to be reachable."""
        c = paired_bootstrap([0.1] * 20, [0.9] * 20)
        assert c.separated and verdict(c) == LESS

    def test_noise_reports_approximately_equal(self):
        values = [0.5, 0.4, 0.6, 0.55, 0.45, 0.5, 0.52, 0.48]
        c = paired_bootstrap(values, list(reversed(values)))
        assert not c.separated and verdict(c) == SIMILAR


# --- the systems ------------------------------------------------------------

class TestBaselinesAreIndependent:
    """Spec section 45 baselines must be able to beat MCM.

    If they were ``RetrievalWeights`` with channels zeroed they would share MCM's
    document construction and the result would be settled before any task ran.
    """

    def test_baselines_hold_no_mcm_store_or_retriever(self, suite):
        for system in (BaselineVectorRAG(), BaselineGraph()):
            system.prepare(suite.snapshot)
            assert not hasattr(system, "store")
            assert not hasattr(system, "_retriever")

    def test_vector_rag_chunks_by_lines_not_by_symbol(self, suite):
        system = BaselineVectorRAG(chunk_lines=2, stride=2)
        system.prepare(suite.snapshot)
        assert system._chunks
        assert all(c.end_line - c.start_line + 1 <= 2 for c in system._chunks)

    def test_graph_uses_no_embeddings(self, suite):
        system = BaselineGraph()
        system.prepare(suite.snapshot)
        assert not hasattr(system, "provider")
        assert system._edges

    def test_vector_rag_and_mcm_share_an_embedding_provider(self):
        """Fairness invariant, and the easiest one to break by accident.

        Giving Baseline A a weaker embedder than MCM's vector channel would rig
        the headline comparison in the crudest available way. Both resolve the
        provider through ``get_provider()`` with no spec, so both follow
        ``MCM_EMBEDDING_PROVIDER`` together or not at all.
        """
        from mcm.retrieval.embedding import get_provider
        from mcm.retrieval.vector import VectorProjection
        from mcm.storage.sqlite_store import SQLiteStore

        store = SQLiteStore()
        try:
            mcm_side = VectorProjection(store).provider
            baseline_side = BaselineVectorRAG().provider
            assert type(mcm_side) is type(baseline_side)
            assert mcm_side.name == baseline_side.name == get_provider().name
            assert mcm_side.dimension == baseline_side.dimension
        finally:
            store.close()


@pytest.fixture(scope="module")
def prepared(suite):
    systems = default_systems()
    for system in systems:
        system.prepare(suite.snapshot)
    return systems


@pytest.fixture(scope="module")
def run(suite):
    return run_suite(suite, default_systems(), budget=1500)


class TestSystemsAnswer:
    def test_every_system_respects_the_token_budget(self, suite, prepared):
        for system in prepared:
            for task in suite.tasks:
                answer = system.answer(task, budget=200)
                assert answer.total_tokens <= 200, system.name

    def test_answers_only_name_definitions_that_existed(self, suite, prepared):
        known = suite.snapshot.unit_ids
        for system in prepared:
            for task in suite.tasks:
                for definition in system.answer(
                        task, budget=1000).ranked_definitions():
                    assert definition in known, system.name

    def test_impact_scoring_excludes_the_seed_for_every_system(self, suite,
                                                              prepared):
        """Uniformly, by the harness.

        A chunk retriever cannot drop the seed itself: it arrives inside a window
        with its neighbours. Leaving it in that system's ranking alone would
        penalise it for a rule the symbol-level systems follow for free.
        """
        for system in prepared:
            for task in suite.of_family(IMPACT):
                answer = system.answer(task, budget=2000)
                ranked = answer.ranked_definitions(
                    exclude=frozenset({task.seed}))
                assert task.seed not in ranked, system.name
                assert score_answer(task, answer).retrieval.hits <= len(
                    task.targets)

    def test_rankings_are_deduplicated(self, suite, prepared):
        for system in prepared:
            for task in suite.tasks:
                ranked = system.answer(task, budget=2000).ranked_definitions()
                assert len(ranked) == len(set(ranked)), system.name


class TestCoverageFloor:
    """A window must show most of a definition before it counts as delivering it.

    Built on a twenty-line function so the arithmetic is unambiguous. A two-line
    definition makes a single line exactly half of it, which sits on the boundary
    and tests nothing.
    """

    @pytest.fixture
    def snapshot(self, tmp_path_factory):
        from mcm.benchmark.tasks import Horizon, RepoSnapshot
        from mcm.ingestion.parser import parse_python

        body = "\n".join(f"    x = {n}" for n in range(20))
        source = f"def big(a):\n{body}\n    return x\n".encode("utf-8")
        horizon = Horizon(repo="t", root=tmp_path_factory.mktemp("cf"),
                          sha="0" * 40, when=None)
        snap = RepoSnapshot(horizon=horizon)
        snap.sources["m.py"] = source
        snap.definitions["m.py"] = list(parse_python(source, "m.py").symbols)
        snap.unit_tokens["m.py:big"] = token_estimate(source.decode())
        return snap

    def test_a_clipped_definition_is_not_delivered(self, snapshot):
        symbol = snapshot.definitions["m.py"][0]
        assert symbol.end_line - symbol.start_line + 1 >= 20
        covered = snapshot.covering("m.py", symbol.end_line, symbol.end_line)
        assert "m.py:big" not in covered

    def test_a_fully_contained_definition_is_delivered(self, snapshot):
        symbol = snapshot.definitions["m.py"][0]
        covered = snapshot.covering("m.py", symbol.start_line, symbol.end_line)
        assert "m.py:big" in covered

    def test_the_floor_is_adjustable_so_sensitivity_can_be_measured(self, snapshot):
        """The headline metric moved when this rule changed, so it is exposed."""
        symbol = snapshot.definitions["m.py"][0]
        window = (symbol.end_line, symbol.end_line)
        assert "m.py:big" not in snapshot.covering("m.py", *window)
        snapshot.coverage_floor = 0.0
        assert "m.py:big" in snapshot.covering("m.py", *window)
        assert COVERAGE_FLOOR == 0.5


# --- the run ----------------------------------------------------------------

class TestRun:
    def test_every_system_answers_every_task(self, suite, run):
        assert len(run.scores) == len(suite.tasks) * len(run.systems)

    def test_the_chain_is_judged_link_by_link(self, run):
        links = chain_verdict(run, "efficiency")
        assert [(left, right) for left, right, _, _ in links] == [
            ("mcm", "graph"), ("graph", "vector-rag")]
        assert all(outcome in (GREATER, SIMILAR, LESS)
                   for _, _, outcome, _ in links)

    def test_the_report_states_whether_the_hypothesis_held(self, run):
        text = report(run)
        assert ("Hypothesis H1 supported" in text
                or "Hypothesis H1 NOT supported" in text)

    def test_the_report_carries_the_caveats_that_qualify_it(self, run):
        text = report(run)
        assert "One repository is one data point" in text
        assert "ceiling" in text

    def test_json_round_trips_the_per_task_scores(self, run):
        payload = run_json(run)
        assert len(payload["scores"]) == len(run.scores)
        assert payload["tasks"]["rejected"] == run.suite.rejected
        assert set(payload["overall"]) == set(run.systems)


class TestAblations:
    def test_every_ablation_is_an_mcm_configuration(self):
        for item in ablations():
            assert isinstance(item.system(), MCMSystem)

    def test_ablation_names_are_distinct(self):
        names = [a.name for a in ablations()]
        assert len(names) == len(set(names))

    def test_reasoning_ablation_changes_the_impact_path(self, suite):
        full = MCMSystem(name="full")
        without = MCMSystem(name="without", use_reasoning=False)
        assert full.use_reasoning and not without.use_reasoning

    def test_components_the_benchmark_cannot_test_are_named(self):
        """Spec section 50 lists seven; three are not exercised by these tasks.

        Reporting a null result for them would look like a finding and be an
        artefact of what this harness measures.
        """
        assert "temporal" in NOT_EXERCISED
        assert "constraints" in NOT_EXERCISED
        assert all(reason for reason in NOT_EXERCISED.values())

    def test_contributions_can_report_a_component_that_costs(self, suite):
        """A component that hurts must be findable, not assumed useful."""
        run = run_suite(suite, [MCMSystem(name="mcm"),
                                MCMSystem(name="mcm-no-vector",
                                          weights=None)], budget=1000)
        found = contributions(run, metric="efficiency")
        assert all(-1.0 <= c.difference <= 1.0 for c in found)


class TestNegligibleDifferences:
    """Separation and size are different questions.

    With enough paired tasks a difference of 0.00005 separates cleanly. Calling
    that a win is true and useless, and it happened: the flask ablation reported
    the provenance channel as `costs, separated` on a difference that rounded to
    -0.000.
    """

    def test_a_tiny_but_separated_difference_is_not_a_win(self):
        left = [0.4000 + (i % 3) * 1e-6 for i in range(200)]
        right = [value - 1e-6 for value in left]
        c = paired_bootstrap(left, right)
        assert c.separated, "precondition: this difference does separate"
        assert c.negligible
        assert verdict(c) == SIMILAR

    def test_a_real_difference_is_still_a_win(self):
        c = paired_bootstrap([0.40] * 50, [0.30] * 50)
        assert c.separated and not c.negligible
        assert verdict(c) == GREATER

    def test_the_threshold_scales_with_the_metric(self):
        """One percent of the larger mean, so it suits tokens as well as ratios.

        The comparison is strict, so a difference of exactly one percent is not
        negligible. These cases sit either side of that boundary rather than on it.
        """
        assert paired_bootstrap([0.010] * 50, [0.0099] * 50).negligible
        assert paired_bootstrap([4000.0] * 50, [3980.0] * 50).negligible   # 0.5%
        assert not paired_bootstrap([4000.0] * 50, [3000.0] * 50).negligible
        assert not paired_bootstrap([4000.0] * 50, [3960.0] * 50).negligible  # 1.0%

    def test_summary_says_negligible_rather_than_separated(self):
        left = [0.5 + (i % 2) * 1e-7 for i in range(100)]
        c = paired_bootstrap(left, [v - 1e-7 for v in left])
        assert "negligible" in c.summary()


# --- tuning -----------------------------------------------------------------

class TestTuning:
    """Fitting hyperparameters on a repository that is not being reported.

    The correctness core is the re-weighting shortcut: the tuner captures
    per-channel scores once and re-scores thousands of weight vectors offline. If
    that shortcut does not reproduce what the real system does, the search
    optimises something the evaluation never runs.
    """

    def test_reweighting_captured_channels_matches_the_real_system(self, suite):
        """The shortcut and the system must agree, or tuning fits a phantom."""
        from mcm.benchmark.systems import fill_budget
        from mcm.retrieval.hybrid import RetrievalWeights

        weights = RetrievalWeights(vector=0.4, lexical=0.3, graph=0.2,
                                   symbolic=0.1, provenance=0.0)
        system = MCMSystem(name="mcm")
        system.prepare(suite.snapshot)
        try:
            system.set_weights(weights)
            nodes = system.units()
            for task in suite.tasks:
                if task.family == IMPACT:
                    continue        # answered by propagation, not by weights
                direct = system.answer(task, budget=2000)
                channels = system.channel_scores(task)
                ranked = sorted(
                    ((key, weights.vector * c[0] + weights.lexical * c[1]
                      + weights.graph * c[2] + weights.symbolic * c[3]
                      + weights.provenance * c[4])
                     for key, c in channels),
                    key=lambda pair: (-pair[1], pair[0]))
                replayed = fill_budget(ranked, nodes, 2000)
                assert ([u.unit_id for u in replayed]
                        == [u.unit_id for u in direct.units]), task.task_id
        finally:
            if system.store is not None:
                system.store.close()

    def test_setting_weights_does_not_rebuild_the_index(self, suite):
        from mcm.retrieval.hybrid import RetrievalWeights

        system = MCMSystem(name="mcm")
        system.prepare(suite.snapshot)
        try:
            store_before = system.store
            units_before = dict(system.units())
            system.set_weights(RetrievalWeights(vector=1.0))
            assert system.store is store_before
            assert system.units() == units_before
        finally:
            if system.store is not None:
                system.store.close()

    def test_channel_capture_returns_the_whole_pool(self, suite):
        """Truncating under the current weights would hide what a different
        weighting would have promoted, shaping the search space by its start."""
        from mcm.retrieval.hybrid import RetrievalWeights

        system = MCMSystem(name="mcm")
        system.prepare(suite.snapshot)
        try:
            task = next(t for t in suite.tasks if t.family != IMPACT)
            system.set_weights(RetrievalWeights(vector=1.0, lexical=0.0,
                                                graph=0.0, symbolic=0.0,
                                                provenance=0.0))
            wide = system.channel_scores(task)
            system.set_weights(RetrievalWeights(graph=1.0, vector=0.0,
                                                lexical=0.0, symbolic=0.0,
                                                provenance=0.0))
            other = system.channel_scores(task)
            assert {k for k, _ in wide} == {k for k, _ in other}
        finally:
            if system.store is not None:
                system.store.close()

    def test_every_system_is_tuned_not_only_mcm(self, suite):
        """Fairness. Tuning MCM against unfitted baselines is tuned-vs-untuned."""
        from mcm.benchmark.tuning import tune

        report = tune(suite, budget=1200, samples=2)
        assert set(report.gains) == {"vector-rag", "graph", "hybrid", "mcm"}
        assert set(report.trials) == {"vector-rag", "graph", "hybrid", "mcm"}
        assert all(trials for trials in report.trials.values())

    def test_a_tuned_configuration_round_trips(self, suite, tmp_path):
        from mcm.benchmark.tuning import (load_configuration, tune,
                                          write_configuration)

        report = tune(suite, budget=1200, samples=2)
        path = write_configuration(report, tmp_path / "cfg.json")
        loaded = load_configuration(path)
        assert loaded.chunk_lines == report.configuration.chunk_lines
        assert loaded.rrf_k == report.configuration.rrf_k
        assert loaded.weights.vector == pytest.approx(
            report.configuration.weights.vector)
        assert loaded.weights.graph_depth == report.configuration.weights.graph_depth

    def test_tuned_systems_carry_the_fitted_settings(self):
        from mcm.benchmark.tuning import TunedConfiguration, tuned_systems
        from mcm.retrieval.hybrid import RetrievalWeights

        cfg = TunedConfiguration(
            weights=RetrievalWeights(vector=0.7, lexical=0.1, graph=0.1,
                                     symbolic=0.1, provenance=0.0),
            chunk_lines=60, stride=45, graph_decay=0.3, graph_depth=1, rrf_k=20)
        vector, graph, hybrid, mcm = tuned_systems(cfg)
        assert vector.chunk_lines == 60 and vector.stride == 45
        assert graph.decay == 0.3 and graph.depth == 1
        assert hybrid.k == 20
        assert mcm.weights.vector == 0.7

    def test_the_weight_spec_is_parseable_back(self):
        """The fitted weights have to survive a trip through the CLI format."""
        from mcm.benchmark.tuning import TunedConfiguration
        from mcm.retrieval.hybrid import RetrievalWeights

        original = RetrievalWeights(vector=0.31, lexical=0.19, graph=0.27,
                                    symbolic=0.13, provenance=0.10,
                                    graph_decay=0.3, graph_depth=3)
        cfg = TunedConfiguration(original, 40, 30, 0.3, 3, 60)
        parsed = RetrievalWeights.parse(cfg.weight_spec)
        assert parsed.vector == pytest.approx(original.vector)
        assert parsed.graph_depth == original.graph_depth
        assert parsed.graph_decay == pytest.approx(original.graph_decay)

    def test_compute_top_basin_centroid_computes_normalized_mean(self):
        from mcm.benchmark.tuning import compute_top_basin_centroid
        from mcm.retrieval.hybrid import RetrievalWeights

        evals = [
            (0.80, RetrievalWeights(vector=0.4, lexical=0.2, graph=0.2, symbolic=0.1, provenance=0.1,
                                    graph_decay=0.5, graph_depth=2)),
            (0.75, RetrievalWeights(vector=0.2, lexical=0.4, graph=0.2, symbolic=0.1, provenance=0.1,
                                    graph_decay=0.5, graph_depth=2)),
            (0.60, RetrievalWeights(vector=0.1, lexical=0.1, graph=0.6, symbolic=0.1, provenance=0.1,
                                    graph_decay=0.3, graph_depth=1)),
        ]
        centroid = compute_top_basin_centroid(evals, top_k=2)
        assert centroid.graph_decay == 0.5
        assert centroid.graph_depth == 2
        assert centroid.vector == pytest.approx(0.3)
        assert centroid.lexical == pytest.approx(0.3)
        assert pytest.approx(centroid.vector + centroid.lexical + centroid.graph + centroid.symbolic + centroid.provenance) == 1.0

    def test_regularized_tune_returns_valid_report(self, suite):
        from mcm.benchmark.tuning import tune

        report = tune(suite, budget=1200, samples=2, regularize=True, top_basin_k=2)
        assert report.configuration.weights.vector >= 0.0
        centroid_trials = [t for t in report.trials["mcm"] if "CENTROID" in t.label]
        assert len(centroid_trials) == 1
