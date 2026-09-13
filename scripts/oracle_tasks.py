"""Build agent tasks whose ground truth is verified by running the tests.

The first version of this experiment scored an agent against the definitions a
commit happened to touch. That is the wrong target and the ceiling proved it: the
definitions `mcm_impact` names covered a mean of 17% of a commit's edits, because
commits also touch things that are not causally downstream of anything. Scoring
against all of them penalises propagation for being right.

Here the ground truth is what actually breaks.

    1. Check out the parent of a commit. The tests pass.
    2. Apply only the part of the commit that changed the target's own file.
       The tests now fail, because callers elsewhere are stale.
    3. For every other file the commit touched, apply it alone and re-run.
       A file that moves the suite from failing to passing, or that is needed
       alongside others to do so, is genuinely required.
    4. The required set is the answer. An agent is right when it names those,
       not when it names everything the commit happened to include.

A task is only kept when step 2 actually breaks something. If reverting the
callers leaves the suite green, there was nothing to find and the task would
reward guessing.

    python scripts/oracle_tasks.py --repo repos/click --out oracle/ --max-tasks 20
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field


@dataclass
class OracleTask:
    task_id: str
    repository: str
    commit: str
    parent: str
    #: File holding the definition whose signature changed.
    target_file: str
    target: str
    #: Files that must be updated for the suite to pass again, verified.
    required_files: list[str]
    #: Files the commit touched that turned out not to be required.
    incidental_files: list[str] = field(default_factory=list)
    #: Definitions inside the target file that the commit rewrote. The task
    #: names these, because an agent asked to repair a repository is told what
    #: it changed; it is the *consequences* that are hidden.
    changed_defs: list[str] = field(default_factory=list)
    baseline_failures: int = 0
    #: Failures after applying only the target file: the damage the agent is
    #: asked to repair. -1 means the package stopped importing entirely, which
    #: is a real break and a more severe one than a failing assertion.
    broken_failures: int = 0
    import_error: bool = False


class Repo:
    """A git worktree plus the command that runs its tests."""

    def __init__(self, root: pathlib.Path, src: str, tests: str,
                 python: pathlib.Path) -> None:
        self.root = root
        self.src = src
        self.tests = tests
        self.python = python

    def git(self, *args: str, cwd: pathlib.Path | None = None) -> str:
        done = subprocess.run(["git", *args], cwd=cwd or self.root,
                              capture_output=True, encoding="utf-8", errors="replace")
        return done.stdout

    def worktree(self, sha: str, at: pathlib.Path) -> None:
        subprocess.run(["git", "worktree", "add", "--detach", "--force",
                        str(at), sha],
                       cwd=self.root, capture_output=True, encoding="utf-8", errors="replace", check=True)

    def drop_worktree(self, at: pathlib.Path) -> None:
        subprocess.run(["git", "worktree", "remove", "--force", str(at)],
                       cwd=self.root, capture_output=True, encoding="utf-8", errors="replace")

    def run_tests(self, at: pathlib.Path, timeout: int = 600) -> tuple[int, str]:
        """How many tests are unhappy, and a short tail for diagnosis.

        Failures and collection errors are counted together, because what the
        oracle needs is one number that goes up when the revision breaks and down
        when it is repaired. Keeping them apart, and treating any error as fatal,
        threw away 39 of flask's 40 candidate commits: old revisions of its
        `test_cli.py` import a name that pytest 9 removed, so every parent looked
        unusable for a reason that has nothing to do with the commit under test.
        `--continue-on-collection-errors` lets the rest of the suite run, and the
        constant error is subtracted out by comparing against the same revision's
        own baseline.

        PYTHONPATH points at the worktree's own source, so the suite exercises
        the checked-out revision rather than whatever is installed in the
        environment. Getting that wrong means every run silently tests the same
        code and every task looks identical.
        """
        env = dict(os.environ, PYTHONPATH=str(at / self.src))
        try:
            done = subprocess.run(
                [str(self.python), "-m", "pytest", self.tests, "-q", "--no-header",
                 "-p", "no:cacheprovider", "--continue-on-collection-errors",
                 "--timeout", "60"],
                cwd=at, capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            return (-1, "suite timed out")
        tail = (done.stdout.strip().splitlines() or [""])[-1]
        counts = re.findall(r"(\d+) (failed|errors?)", tail)
        if not counts and " passed" not in tail:
            return (-1, tail or "no summary line")
        return (sum(int(n) for n, _ in counts), tail)

    def apply_file_from(self, at: pathlib.Path, sha: str, relpath: str) -> bool:
        done = subprocess.run(["git", "checkout", sha, "--", relpath],
                              cwd=at, capture_output=True, encoding="utf-8", errors="replace")
        return done.returncode == 0


def changed_line_numbers(repo: Repo, sha: str, relpath: str) -> set[int]:
    """Lines of the post-change file that the commit added or altered."""
    diff = repo.git("diff", "-U0", f"{sha}~1..{sha}", "--", relpath)
    lines: set[int] = set()
    for header in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", diff, re.M):
        start = int(header.group(1))
        count = int(header.group(2) or 1)
        lines.update(range(start, start + count))
    return lines


def defs_touching(source: str, lines: set[int]) -> list[str]:
    """Qualified names of the definitions those lines fall inside.

    Nested definitions are reported by their dotted path, so a method reads as
    `Class.method` rather than as a bare name that could be any of several.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found: list[str] = []

    def walk(node, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
                continue
            name = f"{prefix}{child.name}"
            end = getattr(child, "end_lineno", child.lineno)
            if isinstance(child, ast.ClassDef):
                walk(child, f"{name}.")
            elif any(child.lineno <= n <= end for n in lines):
                found.append(name)
        return None

    walk(tree, "")
    return found


def reset_worktree(repo: Repo, at: pathlib.Path, sha: str,
                   files: list[str]) -> None:
    """Put the worktree back to the parent revision for every file the commit
    touched, so the next target starts from the same clean state."""
    for relpath in files:
        done = subprocess.run(["git", "checkout", f"{sha}~1", "--", relpath],
                              cwd=at, capture_output=True, encoding="utf-8", errors="replace")
        if done.returncode != 0:               # the file is new in this commit
            (at / relpath).unlink(missing_ok=True)


def python_files_changed(repo: Repo, sha: str) -> list[str]:
    out = repo.git("diff", "--name-only", f"{sha}~1..{sha}")
    return [p for p in out.splitlines() if p.endswith(".py")]


def build(repo: Repo, name: str, max_tasks: int, max_files: int,
          scan: int, max_targets: int = 3) -> list[OracleTask]:
    tasks: list[OracleTask] = []
    # Why candidates are discarded, printed at the end. Without it a run that
    # yields nothing is silent about whether the commits were wrong, the
    # environment was broken, or the breakage simply never happened.
    dropped = {"not 2+ source files": 0, "parent unusable": 0,
               "nothing broke": 0, "only tests required": 0}
    shas = repo.git("log", "--format=%H", f"-{scan}").split()
    print(f"scanning {len(shas)} commits for ones the tests can adjudicate")

    for sha in shas:
        if len(tasks) >= max_tasks:
            break
        changed = python_files_changed(repo, sha)
        source_files = [f for f in changed if "/test" not in f and
                        not f.rsplit("/", 1)[-1].startswith("test_")]
        # Two or more *source* files, not merely two files. A commit that
        # changes one module and adds tests for it is additive: applying the
        # module alone leaves the suite green, so there is nothing for an agent
        # to find. The tasks worth having are the ones where changing one module
        # obliged another module to follow.
        if not (2 <= len(source_files) <= max_files):
            dropped["not 2+ source files"] += 1
            continue

        work = pathlib.Path(tempfile.mkdtemp(prefix="oracle-"))
        at = work / "wt"
        try:
            repo.worktree(f"{sha}~1", at)
            baseline, line = repo.run_tests(at)
            # A suite with pre-existing failures is still usable: what matters is
            # the change from this revision's own baseline, not an absolute zero.
            # Requiring green would discard flask entirely over two unrelated
            # async tests, and every repository has a couple of those.
            if baseline < 0:
                dropped["parent unusable"] += 1
                continue                       # errored or timed out; unusable

            # Every source file gets a turn as the target, not just the first.
            # Which file a commit "changed first" is an artefact of path sorting,
            # and taking it as the target threw away commits where a different
            # one was the cause: flask yielded nothing from 40 candidates that
            # way. The parent baseline above is measured once and reused.
            task = None
            for target_file in source_files[:max_targets]:
                reset_worktree(repo, at, sha, changed)
                if not repo.apply_file_from(at, sha, target_file):
                    continue
                broken, line = repo.run_tests(at)
                if broken <= baseline:
                    dropped["nothing broke"] += 1
                    continue                   # nothing newly broke; nothing to find
                # `broken` is mutated by the loop below as files are applied, so
                # the count at this moment has to be kept separately or the
                # record ends up describing the repaired state rather than the
                # damage.
                after_target_only = broken
                import_error = "error" in line.lower()
                others = [f for f in changed if f != target_file]

                # Which of the other files are actually needed to get back?
                required, incidental = [], []
                for candidate in others:
                    repo.apply_file_from(at, sha, candidate)
                    after, _ = repo.run_tests(at)
                    if after < broken:          # fixed something that was failing
                        required.append(candidate)
                        broken = after
                    else:
                        incidental.append(candidate)
                        subprocess.run(["git", "checkout", f"{sha}~1", "--", candidate],
                                       cwd=at, capture_output=True, encoding="utf-8", errors="replace")

                # A commit whose only requirement is a test file updated its own
                # expectations; no caller had to follow, so there is nothing here
                # that impact analysis could have told an agent.
                required_source = [f for f in required
                                   if "/test" not in f
                                   and not f.rsplit("/", 1)[-1].startswith("test_")]
                if not required_source:
                    dropped["only tests required"] += 1
                    continue

                changed_defs = defs_touching(
                    repo.git("show", f"{sha}:{target_file}"),
                    changed_line_numbers(repo, sha, target_file))[:5]
                task = OracleTask(
                    task_id=f"{name}:{sha[:8]}", repository=name, commit=sha,
                    parent=f"{sha}~1", target_file=target_file,
                    target=target_file, required_files=required,
                    changed_defs=changed_defs,
                    incidental_files=incidental, baseline_failures=baseline,
                    broken_failures=after_target_only,
                    import_error=import_error)
                break                          # one task per commit

            if task is None:
                continue
            tasks.append(task)
            print(f"  {sha[:8]}: {task.target_file} -> "
                  f"{len(task.required_files)} required, "
                  f"{len(task.incidental_files)} incidental", flush=True)
        finally:
            repo.drop_worktree(at)
            shutil.rmtree(work, ignore_errors=True)
    print("  discarded: " + ", ".join(f"{v} {k}" for k, v in dropped.items() if v))
    return tasks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--repo", type=pathlib.Path, required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--src", default="src")
    p.add_argument("--tests", default="tests")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("oracle"))
    p.add_argument("--max-tasks", type=int, default=20)
    p.add_argument("--max-files", type=int, default=6)
    p.add_argument("--scan", type=int, default=120)
    p.add_argument("--max-targets", type=int, default=3,
                   help="how many of a commit's source files to try as the target")
    p.add_argument("--python", type=pathlib.Path,
                   default=pathlib.Path(sys.executable))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    repo = Repo(args.repo.resolve(), args.src, args.tests, args.python.resolve())
    name = args.name or args.repo.name
    tasks = build(repo, name, args.max_tasks, args.max_files, args.scan,
                  args.max_targets)
    (args.out / f"{name}-tasks.json").write_text(
        json.dumps([asdict(t) for t in tasks], indent=2), encoding="utf-8")
    print(f"\n{len(tasks)} tasks with test-verified ground truth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
