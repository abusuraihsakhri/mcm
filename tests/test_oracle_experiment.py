"""The measuring instrument for the agent experiment, checked.

Scoring is where a silent bug becomes a confident wrong number: an agent that
answered correctly can be recorded as having said nothing, and no amount of
re-running the trials would show it. One such bug did occur -- every answer was
discarded as naming the target file -- so these pin the comparison rules down.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from oracle_experiment import normalise, portable, same_file, score  # noqa: E402
from oracle_tasks import defs_touching  # noqa: E402


@pytest.mark.parametrize("written, truth", [
    ("src/click/core.py", "src/click/core.py"),
    ("./src/click/core.py", "src/click/core.py"),
    ("src\\click\\core.py", "src/click/core.py"),
    ("`src/click/core.py`", "src/click/core.py"),
    ("src/click/core.py::Group.invoke", "src/click/core.py"),
    ("click/core.py", "src/click/core.py"),          # rooted differently
])
def test_the_same_file_written_differently_still_matches(written, truth):
    assert same_file(normalise(written), normalise(truth))


@pytest.mark.parametrize("written, truth", [
    ("core.py", "src/click/mycore.py"),              # suffix, not a path segment
    ("src/click/decorators.py", "src/click/core.py"),
    ("", "src/click/core.py"),
])
def test_different_files_do_not_match(written, truth):
    assert not same_file(normalise(written), normalise(truth))


def test_naming_exactly_the_required_file_scores_one():
    assert score(["src/click/core.py"], ["src/click/core.py"]) == (1.0, 1.0)


def test_naming_everything_buys_recall_at_the_cost_of_precision():
    recall, precision = score(
        ["src/click/core.py", "src/click/types.py", "src/click/utils.py"],
        ["src/click/core.py"])
    assert recall == 1.0
    assert precision == pytest.approx(1 / 3)


def test_saying_nothing_scores_nothing():
    assert score([], ["src/click/core.py"]) == (0.0, 0.0)


def test_half_the_required_files_is_half_the_recall():
    recall, precision = score(["src/click/core.py"],
                              ["src/click/core.py", "src/click/types.py"])
    assert recall == 0.5
    assert precision == 1.0


SOURCE = '''
def alone():
    return 1


class Holder:
    def method(self):
        return 2

    def untouched(self):
        return 3
'''


def test_a_changed_line_names_the_definition_it_falls_inside():
    assert defs_touching(SOURCE, {3}) == ["alone"]


def test_a_method_is_named_by_its_class():
    assert defs_touching(SOURCE, {8}) == ["Holder.method"]


def test_lines_outside_every_definition_name_nothing():
    assert defs_touching(SOURCE, {1}) == []


def test_a_file_that_does_not_parse_is_not_an_error():
    assert defs_touching("def broken(:", {1}) == []


def test_a_path_inside_the_working_directory_is_recorded_relative():
    """`tasks.json` is committed, so it must not carry anyone's home directory."""
    inside = pathlib.Path.cwd() / "oracle-experiment" / "index" / "x.db"
    assert portable(inside) == "oracle-experiment/index/x.db"


def test_a_path_outside_the_working_directory_is_left_alone():
    outside = pathlib.Path(tempfile.gettempdir()).resolve() / "elsewhere" / "x.db"
    assert portable(outside) == outside.as_posix()
