"""Temporal model tests (spec section 18).

The central property: a dependency removed by a refactor stops answering ordinary
queries without being deleted, so the same question asked at two moments gives two
answers and neither is lost.
"""

import shutil
import time
from datetime import datetime, timedelta, timezone

import pytest

from mcm.algebra.dependency import dependencies_of, dependents_of
from mcm.core.objects import MCMObject, ObjectType
from mcm.core.relations import MCMRelation, RelationType as RT
from mcm.ingestion.repository import RepositoryIngestor
from mcm.storage.sqlite_store import SQLiteStore

from conftest import AUTHENTICATE, CREATE_TOKEN, DEMO_REPO, JWT

REFACTOR_BEFORE = "    token = create_token(user)\n    return validate_token(token)"
REFACTOR_AFTER = "    return validate_token(user.token)"


@pytest.fixture(scope="module")
def refactored(tmp_path_factory):
    """Ingest the fixture, refactor away one call, ingest again.

    Returns the store plus the moment between the two ingests.
    """
    work = tmp_path_factory.mktemp("repo") / "app"
    shutil.copytree(DEMO_REPO, work)

    store = SQLiteStore()
    ingestor = RepositoryIngestor(store)
    first = ingestor.ingest(work, name="app")
    between = datetime.now(timezone.utc)
    time.sleep(0.02)

    auth = work / "auth.py"
    auth.write_text(auth.read_text(encoding="utf-8").replace(REFACTOR_BEFORE, REFACTOR_AFTER),
                    encoding="utf-8")
    second = ingestor.ingest(work, name="app")

    yield store, between, first, second
    store.close()


class TestRelationValidity:
    def _at(self, days: int) -> datetime:
        return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=days)

    def test_closed_relation_is_invalid_after_its_end(self, empty_store):
        empty_store.put_object(MCMObject(id="a", type=ObjectType.FILE, name="a"))
        empty_store.put_object(MCMObject(id="b", type=ObjectType.FILE, name="b"))
        empty_store.put_relation(MCMRelation(
            id="r", relation_type=RT.IMPORTS, arguments=["a", "b"],
            valid_from=self._at(0), valid_until=self._at(10)))

        assert empty_store.relations_for("a", as_of=self._at(5))
        assert not empty_store.relations_for("a", as_of=self._at(11))

    def test_include_historical_ignores_validity(self, empty_store):
        empty_store.put_relation(MCMRelation(
            id="r", relation_type=RT.IMPORTS, arguments=["a", "b"],
            valid_from=self._at(0), valid_until=self._at(10)))
        assert not list(empty_store.all_relations(as_of=self._at(11)))
        assert len(list(empty_store.all_relations(include_historical=True))) == 1

    def test_close_relation_does_not_delete(self, empty_store):
        empty_store.put_relation(MCMRelation(id="r", relation_type=RT.CALLS,
                                             arguments=["a", "b"]))
        empty_store.close_relation("r", self._at(5))
        assert empty_store.get_relation("r") is not None
        assert empty_store.get_relation("r").valid_until == self._at(5)
        assert not empty_store.relations_for("a", as_of=self._at(6))


class TestReingestionDiff:
    def test_first_ingest_opens_everything(self, refactored):
        _, _, first, _ = refactored
        assert first.opened > 0
        assert first.closed == 0
        assert first.unchanged == 0

    def test_refactor_closes_exactly_the_removed_relation(self, refactored):
        store, _, _, second = refactored
        assert second.closed == 1
        assert second.opened == 0
        closed = store.get_relation(second.closed_relations[0])
        assert closed.relation_type is RT.CALLS
        assert closed.arguments == [AUTHENTICATE, CREATE_TOKEN]

    def test_unchanged_relations_are_not_reopened(self, refactored):
        _, _, first, second = refactored
        assert second.unchanged == first.opened - 1

    def test_closed_relation_is_retained_not_deleted(self, refactored):
        store, _, _, _ = refactored
        historical = list(store.all_relations(include_historical=True))
        current = list(store.all_relations())
        assert len(historical) == len(current) + 1


class TestPointInTimeQueries:
    def test_the_same_question_gives_two_answers(self, refactored):
        store, between, _, _ = refactored
        assert CREATE_TOKEN not in dependencies_of(store, AUTHENTICATE).paths
        assert CREATE_TOKEN in dependencies_of(store, AUTHENTICATE, as_of=between).paths

    def test_impact_closure_respects_the_moment(self, refactored):
        store, between, _, _ = refactored
        now_depth = dependents_of(store, JWT).paths[AUTHENTICATE].depth
        then_depth = dependents_of(store, JWT, as_of=between).paths[AUTHENTICATE].depth
        assert now_depth == then_depth == 3

    def test_a_moment_before_the_first_ingest_sees_nothing(self, refactored):
        store, _, _, _ = refactored
        ancient = datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert dependents_of(store, JWT, as_of=ancient).paths == {}

    def test_default_reads_exclude_closed_relations(self, refactored):
        store, _, _, second = refactored
        closed_id = second.closed_relations[0]
        assert closed_id not in {r.id for r in store.all_relations()}
        assert closed_id in {r.id for r in store.all_relations(include_historical=True)}


class TestIdempotenceUnderTemporalDiff:
    def test_reingesting_unchanged_source_closes_nothing(self, tmp_path):
        work = tmp_path / "app"
        shutil.copytree(DEMO_REPO, work)
        store = SQLiteStore()
        ingestor = RepositoryIngestor(store)
        ingestor.ingest(work, name="app")
        second = ingestor.ingest(work, name="app")
        assert second.closed == 0
        assert second.opened == 0
        store.close()
