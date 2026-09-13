"""Command-line entry point for the prototype.

    mcm ingest <path> [--db mcm.db] [--name app]
    mcm impact <reference> [--db] [--json] [--depth 6] [--as-of ISO8601]
    mcm deps   <reference> [--db] [--json] [--as-of ISO8601]
    mcm lookup <reference> [--db]
    mcm derive [--db] [--rules FILE] [--json] [--as-of ISO8601]
    mcm constraints [--db] [--file FILE] [--json] [--as-of ISO8601]
    mcm index  [--db] [--provider SPEC] [--as-of ISO8601]
    mcm search <query> [--db] [--limit N] [--weights SPEC] [--json] [--as-of]
    mcm context <task> [--db] [--focus REF] [--floor F] [--full] [--json] [--as-of]
    mcm propagate <reference> --kind KIND [--details TEXT] [--db] [--json] [--as-of]
    mcm simulate  <reference> --kind KIND [--details TEXT] [--db] [--json] [--as-of]
    mcm accuracy [--db] [--json]
    mcm equivalent <reference> [--domain D] [--db] [--json]
    mcm equivalence [--domain D] [--min N] [--db] [--json]
    mcm history <path> [--db] [--name] [--max-commits N]
    mcm why <reference> [--db] [--since SHA-or-date]
    mcm benchmark <repo> [--horizon N] [--max-tasks N] [--budget N]
                         [--k N] [--metric M] [--coverage-floor F]
                         [--ablate | --tune [--samples N] | --config PATH]
                         [--json PATH]
    mcm demo
    mcm history-demo

``mcm demo`` runs the spec section 65 proof-of-concept end to end against
examples/app in an in-memory store.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .agent.context import build_context, package_json
from .agent.context import explain as context_report
from .agent.minimisation import minimise
from .agent.query import QueryType, run_query
from .ingestion.git import GitError
from .ingestion.history import HistoryIngestor
from .ingestion.repository import RepositoryIngestor
from .core.change import Change, ChangeKind
from .equivalence.engine import Domain, equivalence_classes
from .equivalence.engine import explain as equivalence_report
from .reasoning.change_propagation import propagate, propagation_json
from .reasoning.change_propagation import explain as propagation_report
from .reasoning.constraint_checker import check_constraints, load_constraints, report
from .reasoning.prediction import accuracy_json, evaluate
from .reasoning.prediction import explain as accuracy_report
from .reasoning.simulation import simulate, simulation_json
from .reasoning.simulation import explain as simulation_report
from .reasoning.causal_reasoning import explain as explain_regression
from .reasoning.causal_reasoning import regression_candidates
from .reasoning.dependency_propagation import analyse_impact, explain
from .reasoning.engine import derive
from .reasoning.rules import load_rules
from .retrieval.embedding import get_provider
from .retrieval.hybrid import HybridRetriever, RetrievalWeights, build_indexes
from .retrieval.hybrid import explain as retrieval_report
from .retrieval.hybrid import result_json
from .retrieval.symbolic import resolve_one
from .retrieval.vector import VectorProjection
from .core.objects import ObjectType
from .storage.sqlite_store import SQLiteStore

#: Default per-task context budget for ``mcm benchmark`` (spec section 48).
BENCH_BUDGET = 4000

DEMO_REPO = Path(__file__).resolve().parent.parent / "examples" / "app"
DEMO_TARGET = "lib://jwt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mcm", description="Mathematical Context Model")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest a repository")
    p_ingest.add_argument("path")
    p_ingest.add_argument("--db", default="mcm.db")
    p_ingest.add_argument("--name", default=None)

    for name, help_text in [("impact", "what could break if this changes"),
                            ("deps", "what this depends on"),
                            ("lookup", "find objects by name")]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("reference")
        p.add_argument("--db", default="mcm.db")
        p.add_argument("--depth", type=int, default=6)
        p.add_argument("--as-of", dest="as_of", default=None,
                       help="answer against the repository as it stood at this time")
        p.add_argument("--json", action="store_true")

    p_derive = sub.add_parser("derive", help="run the inference rules to a fixpoint")
    p_derive.add_argument("--db", default="mcm.db")
    p_derive.add_argument("--rules", default=None)
    p_derive.add_argument("--as-of", dest="as_of", default=None)
    p_derive.add_argument("--json", action="store_true")

    p_constraints = sub.add_parser("constraints", help="check constraints")
    p_constraints.add_argument("--db", default="mcm.db")
    p_constraints.add_argument("--file", default=None)
    p_constraints.add_argument("--as-of", dest="as_of", default=None)
    p_constraints.add_argument("--json", action="store_true")

    p_contra = sub.add_parser("contradictions", help="detect opposing relations and conflicting claims (spec section 31)")
    p_contra.add_argument("--db", default="mcm.db")
    p_contra.add_argument("--as-of", dest="as_of", default=None)
    p_contra.add_argument("--json", action="store_true")

    p_index = sub.add_parser("index", help="build the vector and lexical projections")
    p_index.add_argument("--db", default="mcm.db")
    p_index.add_argument("--provider", default=None,
                         help="embedding provider, e.g. hashed:512 (default: $MCM_EMBEDDING_PROVIDER)")
    p_index.add_argument("--as-of", dest="as_of", default=None)

    p_search = sub.add_parser("search", help="hybrid retrieval across all four channels")
    p_search.add_argument("query")
    p_search.add_argument("--db", default="mcm.db")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.add_argument("--provider", default=None)
    p_search.add_argument("--weights", default=None,
                          help="override scoring weights, e.g. v=0.4,l=0.3,g=0.2,s=0.1,p=0.0")
    p_search.add_argument("--as-of", dest="as_of", default=None)
    p_search.add_argument("--json", action="store_true")

    p_context = sub.add_parser("context", help="build a minimal context package for a task")
    p_context.add_argument("task")
    p_context.add_argument("--db", default="mcm.db")
    p_context.add_argument("--focus", default=None,
                           help="the object the task is about; default: top retrieval hit")
    p_context.add_argument("--floor", type=float, default=0.0,
                           help="drop claims below this confidence (spec section 34)")
    p_context.add_argument("--depth", type=int, default=6)
    p_context.add_argument("--full", action="store_true",
                           help="skip minimisation and show everything gathered")
    p_context.add_argument("--as-of", dest="as_of", default=None)
    p_context.add_argument("--json", action="store_true")

    p_propagate = sub.add_parser(
        "propagate", help="estimate what a specific change forces elsewhere")
    p_propagate.add_argument("reference")
    p_propagate.add_argument("--kind", required=True,
                             choices=[k.value.lower() for k in ChangeKind],
                             help="what is being changed about it")
    p_propagate.add_argument("--details", default="",
                             help="free text recorded with the change")
    p_propagate.add_argument("--db", default="mcm.db")
    p_propagate.add_argument("--depth", type=int, default=6)
    p_propagate.add_argument("--as-of", dest="as_of", default=None)
    p_propagate.add_argument("--json", action="store_true")

    p_simulate = sub.add_parser(
        "simulate", help="check a change against the state it would produce")
    p_simulate.add_argument("reference")
    p_simulate.add_argument("--kind", required=True,
                            choices=[k.value.lower() for k in ChangeKind])
    p_simulate.add_argument("--details", default="")
    p_simulate.add_argument("--db", default="mcm.db")
    p_simulate.add_argument("--depth", type=int, default=6)
    p_simulate.add_argument("--as-of", dest="as_of", default=None)
    p_simulate.add_argument("--json", action="store_true")

    p_accuracy = sub.add_parser(
        "accuracy", help="compare past predictions against what commits actually did")
    p_accuracy.add_argument("--db", default="mcm.db")
    p_accuracy.add_argument("--depth", type=int, default=6)
    p_accuracy.add_argument("--json", action="store_true")

    p_equivalent = sub.add_parser(
        "equivalent", help="what is equivalent to this, under a stated domain")
    p_equivalent.add_argument("reference")
    p_equivalent.add_argument("--domain", default=Domain.NORMAL_FORM.value,
                              choices=[d.value for d in Domain])
    p_equivalent.add_argument("--db", default="mcm.db")
    p_equivalent.add_argument("--json", action="store_true")

    p_equivalence = sub.add_parser(
        "equivalence", help="definitions that share a form")
    p_equivalence.add_argument("--domain", default=Domain.NORMAL_FORM.value,
                               choices=[d.value for d in Domain])
    p_equivalence.add_argument("--min", dest="minimum", type=int, default=2)
    p_equivalence.add_argument("--db", default="mcm.db")
    p_equivalence.add_argument("--json", action="store_true")

    p_history = sub.add_parser("history", help="ingest Git history commit by commit")
    p_history.add_argument("path")
    p_history.add_argument("--db", default="mcm.db")
    p_history.add_argument("--name", default=None)
    p_history.add_argument("--max-commits", dest="max_commits", type=int, default=None)

    p_why = sub.add_parser("why", help="commits that changed what this depends on")
    p_why.add_argument("reference")
    p_why.add_argument("--db", default="mcm.db")
    p_why.add_argument("--since", default=None,
                       help="a commit SHA prefix or an ISO date; the last known-good point")
    p_why.add_argument("--depth", type=int, default=6)

    p_bench = sub.add_parser(
        "benchmark", help="evaluate MCM against the spec section 45 baselines")
    p_bench.add_argument("path", help="a Git repository to draw tasks from")
    p_bench.add_argument("--name", default=None, help="repository label")
    p_bench.add_argument("--horizon", type=int, default=200,
                         help="commits back from HEAD to split history at")
    p_bench.add_argument("--max-tasks", type=int, default=40)
    p_bench.add_argument("--budget", type=int, default=BENCH_BUDGET,
                         help="tokens of context each system may spend per task")
    p_bench.add_argument("--k", type=int, default=10, help="K for Recall@K")
    p_bench.add_argument("--coverage-floor", dest="coverage_floor", type=float,
                         default=0.5,
                         help="fraction of a definition a retrieved window must "
                              "contain to count as delivering it")
    p_bench.add_argument("--metric", default="efficiency",
                         help="metric the section 3 chain is judged on")
    p_bench.add_argument("--ablate", action="store_true",
                         help="run the section 50 ablations instead of the baselines")
    p_bench.add_argument("--tune", action="store_true",
                         help="fit every system's hyperparameters on this repository "
                              "instead of evaluating; write them with --json")
    p_bench.add_argument("--config", default=None,
                         help="evaluate using hyperparameters fitted elsewhere "
                              "(a file written by --tune)")
    p_bench.add_argument("--samples", type=int, default=150,
                         help="weight vectors sampled per traversal setting when tuning")
    p_bench.add_argument("--regularize", action=argparse.BooleanOptionalAction, default=True,
                         help="regularize MCM tuning by taking the centroid of the top basin")
    p_bench.add_argument("--top-k-basin", dest="top_basin_k", type=int, default=5,
                         help="number of top candidates to average in top basin")
    p_bench.add_argument("--provider", default=None,
                         help="embedding provider every system shares, e.g. "
                              "'sentence-transformers:all-MiniLM-L6-v2@cuda'. "
                              "Defaults to offline feature hashing, which needs "
                              "nothing installed but is not a trained model.")
    p_bench.add_argument("--on-disk", action="store_true",
                         help="keep MCM's graph in a temporary file instead of "
                              "RAM. Slower, and roughly 1-2GB lighter, which is "
                              "what makes a large repository scoreable at all.")
    p_bench.add_argument("--json", dest="as_json", default=None,
                         help="write full results to this path")

    sub.add_parser("demo", help="run the spec section 65 proof-of-concept")
    sub.add_parser("history-demo", help="build a repo with a regression and find it")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        return _ingest(args)
    if args.command == "demo":
        return _demo()
    if args.command == "derive":
        return _derive(args)
    if args.command == "constraints":
        return _constraints(args)
    if args.command == "contradictions":
        return _contradictions(args)
    if args.command == "index":
        return _index(args)
    if args.command == "search":
        return _search(args)
    if args.command == "context":
        return _context(args)
    if args.command == "propagate":
        return _propagate(args)
    if args.command == "simulate":
        return _simulate(args)
    if args.command == "accuracy":
        return _accuracy(args)
    if args.command == "equivalent":
        return _equivalent(args)
    if args.command == "equivalence":
        return _equivalence(args)
    if args.command == "history":
        return _history(args)
    if args.command == "why":
        return _why(args)
    if args.command == "history-demo":
        return _history_demo()
    if args.command == "benchmark":
        return _benchmark(args)
    return _query(args)


def _benchmark(args) -> int:
    """Spec sections 45 to 50: build a task suite and run every system on it."""
    from .benchmark.ablation import ablation_systems
    from .benchmark.ablation import report as ablation_report
    from .benchmark.runner import report as benchmark_report
    from .benchmark.runner import run_suite, write_json
    from .benchmark.systems import default_systems
    from .retrieval.embedding import get_provider
    from .benchmark.tasks import generate_tasks
    from .benchmark.tuning import (load_configuration, tune, tuned_systems,
                                   write_configuration)

    try:
        suite = generate_tasks(args.path, repo=args.name,
                               horizon_back=args.horizon,
                               max_tasks=args.max_tasks,
                               coverage_floor=args.coverage_floor)
    except (ValueError, GitError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not suite.tasks:
        print(suite.summary(), file=sys.stderr)
        print("No tasks survived the filters. Try a larger --horizon.",
              file=sys.stderr)
        return 1

    print(suite.summary(), flush=True)

    if args.tune:
        # Fitting, not evaluating. Whatever is reported from this repository
        # afterwards is a training score, which is why the two are separate runs.
        report = tune(suite, budget=args.budget, k=args.k, metric=args.metric,
                      samples=args.samples, regularize=args.regularize,
                      top_basin_k=args.top_basin_k, progress=True)
        print()
        print(report.summary())
        print()
        print("  These are training scores. Evaluate the fitted settings on a")
        print("  repository that was not tuned on, with --config.")
        if args.as_json:
            print()
            print(f"Configuration written to {write_configuration(report, args.as_json)}")
        return 0

    if args.config:
        configuration = load_configuration(args.config)
        if args.ablate:
            print("--config and --ablate cannot be combined: the ablations vary "
                  "the weights that --config fixes.", file=sys.stderr)
            return 2
        systems = tuned_systems(configuration)
        print(f"  using hyperparameters from {args.config}")
        print(f"  mcm weights: {configuration.weight_spec}")
    else:
        systems = (ablation_systems() if args.ablate
                   else default_systems(get_provider(args.provider),
                                        on_disk=args.on_disk))
    run = run_suite(suite, systems, budget=args.budget, k=args.k, progress=True)
    print()
    print(ablation_report(run, metric=args.metric) if args.ablate
          else benchmark_report(run, metric=args.metric))
    if args.as_json:
        path = write_json(run, args.as_json, metric=args.metric)
        print()
        print(f"Full results written to {path}")
    return 0


def _index(args) -> int:
    store = SQLiteStore(args.db)
    try:
        provider = get_provider(args.provider)
        vector = VectorProjection(store, provider=provider)
        vector_report, lexical_report = build_indexes(store, vector=vector,
                                                      as_of=_moment(args.as_of))
        print("Vector projection:  " + vector_report.summary())
        print("Lexical projection: " + lexical_report.summary())
        print()
        print("Both indexes are derived from objects, relations and evidence.")
        print("Deleting them loses no knowledge; `mcm index` rebuilds them.")
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        store.close()
    return 0


def _search(args) -> int:
    store = SQLiteStore(args.db)
    try:
        weights = RetrievalWeights.parse(args.weights) if args.weights else None
        retriever = HybridRetriever(
            store, weights=weights,
            vector=VectorProjection(store, provider=get_provider(args.provider)),
        )
        result = retriever.retrieve(args.query, limit=args.limit,
                                    as_of=_moment(args.as_of))
        print(json.dumps(result_json(result), indent=2) if args.json
              else retrieval_report(result, limit=args.limit))
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        store.close()
    return 0


def _propagate(args) -> int:
    store = SQLiteStore(args.db)
    try:
        obj = resolve_one(store, args.reference)
        change = Change(target_id=obj.id, kind=ChangeKind(args.kind.upper()),
                        details=args.details)
        result = propagate(store, change, max_depth=args.depth,
                           as_of=_moment(args.as_of))
        print(json.dumps(propagation_json(result), indent=2) if args.json
              else propagation_report(result))
    except (KeyError, ValueError) as exc:
        print(str(exc).strip('"'), file=sys.stderr)
        return 1
    finally:
        store.close()
    return 0


def _simulate(args) -> int:
    store = SQLiteStore(args.db)
    try:
        obj = resolve_one(store, args.reference)
        change = Change(target_id=obj.id, kind=ChangeKind(args.kind.upper()),
                        details=args.details)
        result = simulate(store, change, max_depth=args.depth,
                          as_of=_moment(args.as_of))
        print(json.dumps(simulation_json(result), indent=2) if args.json
              else simulation_report(result))
    except (KeyError, ValueError) as exc:
        print(str(exc).strip('"'), file=sys.stderr)
        return 1
    finally:
        store.close()
    return 0


def _accuracy(args) -> int:
    store = SQLiteStore(args.db)
    try:
        report_ = evaluate(store, max_depth=args.depth)
        print(json.dumps(accuracy_json(report_), indent=2) if args.json
              else accuracy_report(report_))
    finally:
        store.close()
    return 0


def _equivalent(args) -> int:
    store = SQLiteStore(args.db)
    try:
        payload = run_query(store, QueryType.EQUIVALENCE, args.reference,
                            domain=Domain(args.domain))
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            target = payload["target"]
            print(f'{target["name"]} under the {payload["domain"]} domain')
            for match in payload["equivalent_to"]:
                obj = match["object"]
                print(f'  {obj["name"]}  [{obj["type"]}] {obj["relpath"] or ""}'.rstrip())
                print(f'    {match["reason"]}')
            if not payload["equivalent_to"]:
                print("  nothing shares this form")
                if args.domain == Domain.SEMANTIC.value:
                    print("  note: semantic equivalence is not decided here, so "
                          "this is not a claim that nothing is equivalent")
    except (KeyError, ValueError) as exc:
        print(str(exc).strip('"'), file=sys.stderr)
        return 1
    finally:
        store.close()
    return 0


def _equivalence(args) -> int:
    store = SQLiteStore(args.db)
    try:
        classes = equivalence_classes(store, domain=Domain(args.domain),
                                      minimum=args.minimum)
        if args.json:
            print(json.dumps([
                {"domain": c.domain.value, "digest": c.digest, "size": c.size,
                 "members": [{"id": m.id, "name": m.name,
                              "relpath": m.properties.get("relpath")}
                             for m in c.members]}
                for c in classes
            ], indent=2))
        else:
            print(equivalence_report(classes))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        store.close()
    return 0


def _context(args) -> int:
    store = SQLiteStore(args.db)
    try:
        as_of = _moment(args.as_of)
        package = build_context(store, args.task, focus=args.focus,
                                max_depth=args.depth, floor=args.floor, as_of=as_of)
        if not args.full:
            package = minimise(store, package, max_depth=args.depth, as_of=as_of)
        print(json.dumps(package_json(store, package), indent=2) if args.json
              else context_report(store, package))
    except (KeyError, ValueError) as exc:
        print(str(exc).strip('"'), file=sys.stderr)
        return 1
    finally:
        store.close()
    return 0


def _history(args) -> int:
    store = SQLiteStore(args.db)
    try:
        report_ = HistoryIngestor(store).ingest(args.path, name=args.name,
                                                max_commits=args.max_commits)
        print(f"Replayed {report_.repository_id}")
        print("  " + report_.summary())
        for entry in report_.commits:
            print("  " + entry.summary())
    except GitError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        store.close()
    return 0


def _resolve_since(store, raw: str | None):
    """Accept a commit SHA prefix or an ISO date as the last known-good point."""
    if raw is None:
        return None
    try:
        return _moment(raw)
    except ValueError:
        pass
    matches = [obj for obj in store.find_objects(type=ObjectType.COMMIT)
               if obj.properties.get("sha", "").startswith(raw)]
    if not matches:
        raise ValueError(f"no commit matches {raw!r}, and it is not an ISO date")
    if len(matches) > 1:
        raise ValueError(f"{raw!r} matches {len(matches)} commits")
    return datetime.fromisoformat(matches[0].properties["date"])


def _why(args) -> int:
    store = SQLiteStore(args.db)
    try:
        obj = resolve_one(store, args.reference)
        since = _resolve_since(store, args.since)
        report_ = regression_candidates(store, obj.id, since=since, max_depth=args.depth)
        print(explain_regression(store, report_))
    except (KeyError, ValueError) as exc:
        print(str(exc).strip(chr(34)), file=sys.stderr)
        return 1
    finally:
        store.close()
    return 0


def _history_demo() -> int:
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
    from history_fixture import build  # noqa: PLC0415 - demo-only import

    with tempfile.TemporaryDirectory() as tmp:
        root = build(Path(tmp) / "repo")
        store = SQLiteStore()
        report_ = HistoryIngestor(store).ingest(root, name="app")
        print("Replayed a repository with a regression in its history")
        print("  " + report_.summary())
        for entry in report_.commits:
            print("  " + entry.summary())
        print()
        print('Query: "Why did authentication start failing after the tests landed?"')
        print("=" * 72)
        target = "repo://app/auth.py#function:authenticate"
        since = datetime.fromisoformat(report_.commits[2].commit.date.isoformat())
        print(explain_regression(store, regression_candidates(store, target, since=since)))
        store.close()
    return 0


def _moment(raw: str | None) -> datetime | None:
    """Parse an --as-of value. A bare date means midnight UTC."""
    if raw is None:
        return None
    parsed = datetime.fromisoformat(raw)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _ingest(args) -> int:
    store = SQLiteStore(args.db)
    report_ = RepositoryIngestor(store).ingest(args.path, name=args.name)
    print(f"Ingested {report_.repository_id}")
    print("  " + report_.summary())
    print("  " + report_.temporal_summary())
    for relation_id in report_.closed_relations[:10]:
        relation = store.get_relation(relation_id)
        if relation is not None:
            print(f"    closed: {relation.relation_type.value} "
                  f"{' -> '.join(relation.arguments)}")
    for error in report_.parse_errors:
        print(f"  parse error: {error}", file=sys.stderr)
    if report_.unresolved:
        print(f"  {len(report_.unresolved)} unresolved names (not guessed):")
        for item in report_.unresolved[:10]:
            print(f"    {item}")
        if len(report_.unresolved) > 10:
            print(f"    ... and {len(report_.unresolved) - 10} more")
    store.close()
    return 0


def _query(args) -> int:
    store = SQLiteStore(args.db)
    try:
        as_of = _moment(args.as_of)
        mode = {"impact": QueryType.IMPACT, "deps": QueryType.DEPENDENCY,
                "lookup": QueryType.LOOKUP}[args.command]
        if args.json or mode is not QueryType.IMPACT:
            payload = run_query(store, mode, args.reference, max_depth=args.depth,
                                as_of=as_of)
            print(json.dumps(payload, indent=2))
        else:
            obj = resolve_one(store, args.reference)
            print(explain(store, analyse_impact(store, obj.id, max_depth=args.depth,
                                                as_of=as_of)))
    except (KeyError, ValueError) as exc:
        print(str(exc).strip('"'), file=sys.stderr)
        return 1
    finally:
        store.close()
    return 0


def _derive(args) -> int:
    store = SQLiteStore(args.db)
    try:
        rules = load_rules(args.rules)
        result = derive(store, rules, as_of=_moment(args.as_of))
        if args.json:
            print(json.dumps({
                "iterations": result.iterations,
                "complete": result.complete,
                "rule_counts": result.rule_counts,
                "derived": [
                    {"type": r.relation_type.value, "arguments": r.arguments,
                     "confidence": round(r.confidence, 4), "rule": r.properties.get("rule"),
                     "depth": r.properties.get("depth"), "path": r.inference.path}
                    for r in result.relations
                ],
            }, indent=2))
        else:
            print(f"{len(result.derived)} relations derived in "
                  f"{result.iterations} iterations "
                  f"({'fixpoint reached' if result.complete else 'BOUND HIT'})")
            for name, count in sorted(result.rule_counts.items()):
                print(f"  {name:40} {count}")
            print()
            print("All output is derived. Nothing was written to the store.")
    finally:
        store.close()
    return 0


def _constraints(args) -> int:
    store = SQLiteStore(args.db)
    try:
        constraints = load_constraints(args.file)
        results = check_constraints(store, constraints, as_of=_moment(args.as_of))
        if args.json:
            print(json.dumps([
                {"name": r.constraint.name, "type": r.constraint.type.value,
                 "verdict": r.verdict.value, "reason": r.reason, "checked": r.checked,
                 "violations": [{"message": v.message, "objects": v.objects,
                                 "evidence_ids": v.evidence_ids} for v in r.violations]}
                for r in results
            ], indent=2))
        else:
            print(report(results))
    finally:
        store.close()
    return 0


def _contradictions(args) -> int:
    from .reasoning.contradiction import contradiction_json, detect_contradictions, explain as explain_contradictions
    store = SQLiteStore(args.db)
    try:
        results = detect_contradictions(store, as_of=_moment(args.as_of))
        if args.json:
            print(json.dumps(contradiction_json(results), indent=2))
        else:
            print(explain_contradictions(results))
    finally:
        store.close()
    return 0


def _demo() -> int:
    store = SQLiteStore()
    report_ = RepositoryIngestor(store).ingest(DEMO_REPO, name="app")
    print("Ingested examples/app:", report_.summary())
    print()
    print('Query: "What could break if I replace the JWT implementation?"')
    print("=" * 72)
    print(explain(store, analyse_impact(store, DEMO_TARGET)))
    print()
    print("Constraints")
    print("=" * 72)
    print(report(check_constraints(store, load_constraints())))
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
