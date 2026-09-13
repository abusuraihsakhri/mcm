"""Build a Git repository whose history can test an edit prediction.

``history_fixture.py`` cannot. Its only change to an existing definition is a
behaviour change with no co-edits, and a diff can neither confirm nor deny a
behaviour change - so the accuracy report correctly refuses to score it.

This repository contains both cases an evaluation needs:

    1  initial: geometry.area, and report.summarise which calls it
    2  add a test that calls summarise
    3  SIGNATURE change to area, and the call site in summarise updated with it
       -> a MUST_UPDATE prediction that the commit confirms
    4  BEHAVIOUR change to area alone, no caller touched
       -> a MAY_DIFFER prediction that no diff can settle either way

Commit 3 is the one that matters: an edit was predicted and an edit was made, in
the same commit, because a signature change that does not fix its call sites ships
broken code.

Run directly to build one::

    python examples/prediction_fixture.py /tmp/prediction-repo
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

GEOMETRY_V1 = '''"""Area arithmetic."""


def area(width, height):
    return width * height
'''

# Commit 3: a new parameter. Every call site has to change with it.
GEOMETRY_V2 = '''"""Area arithmetic."""


def area(width, height, unit="m"):
    return "%s %s2" % (width * height, unit)
'''

# Commit 4: the body changes, the signature does not. Callers keep working.
GEOMETRY_V3 = '''"""Area arithmetic."""


def area(width, height, unit="m"):
    return "%s %s2" % (round(width * height, 2), unit)
'''

REPORT_V1 = '''"""Reporting over the geometry helpers."""

from geometry import area


def summarise(width, height):
    return "area: " + str(area(width, height))
'''

# Commit 3 updates the call site in the same commit as the signature change.
REPORT_V2 = '''"""Reporting over the geometry helpers."""

from geometry import area


def summarise(width, height):
    return "area: " + str(area(width, height, unit="cm"))
'''

TEST_REPORT = '''from report import summarise


def test_summarise_mentions_area():
    assert "area" in summarise(2, 3)
'''

COMMITS = [
    ("initial geometry and reporting",
     {"geometry.py": GEOMETRY_V1, "report.py": REPORT_V1}),
    ("add a reporting test",
     {"tests/__init__.py": "", "tests/test_report.py": TEST_REPORT}),
    ("give area a unit parameter and update its caller",
     {"geometry.py": GEOMETRY_V2, "report.py": REPORT_V2}),
    ("round the area result",
     {"geometry.py": GEOMETRY_V3}),
]


def build(dest: Path | str) -> Path:
    """Create the repository at ``dest`` and return its path."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _git(dest, "init", "-q")
    _git(dest, "config", "user.email", "fixture@example.invalid")
    _git(dest, "config", "user.name", "Fixture Author")
    _git(dest, "config", "commit.gpgsign", "false")

    for index, (subject, files) in enumerate(COMMITS):
        for relpath, content in files.items():
            path = dest / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        _git(dest, "add", "-A")
        stamp = f"2026-04-{index + 1:02d}T10:00:00+00:00"
        _git(dest, "commit", "-q", "-m", subject, "--date", stamp, env_date=stamp)
    return dest


def _git(cwd: Path, *args: str, env_date: str | None = None) -> None:
    import os

    env = os.environ.copy()
    if env_date:
        env["GIT_AUTHOR_DATE"] = env_date
        env["GIT_COMMITTER_DATE"] = env_date
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, env=env)


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "prediction-repo")
    print(build(target))
