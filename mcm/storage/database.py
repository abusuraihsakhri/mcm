"""Storage interface for the canonical semantic core.

Spec section 5 names PostgreSQL + JSONB. V1 ships a SQLite backend so the
prototype runs with no infrastructure, behind this interface so a Postgres
backend is a drop-in (spec Rule 9: every component must be replaceable).

Nothing here is graph-shaped or vector-shaped. Graph adjacency, full-text and
embedding indexes are *projections* built from this store, never the other way
round (spec sections 21 to 24).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Iterator, Literal

from ..core.evidence import Evidence
from ..core.objects import MCMObject, ObjectType
from ..core.provenance import Provenance
from ..core.relations import MCMRelation, RelationType

Direction = Literal["out", "in", "any"]


class Store(ABC):
    """Canonical read/write access to objects, relations, evidence, provenance."""

    @contextmanager
    def bulk(self) -> Iterator["Store"]:
        """Batch the writes in this block, where the backend can.

        An optimisation, not a guarantee: the default does nothing, so a store
        that writes straight through stays correct without implementing it.
        Callers use it to say "these writes belong together and none of them
        needs to survive on its own", which is true of a whole-revision ingest.
        """
        yield self

    # --- objects ----------------------------------------------------------
    @abstractmethod
    def put_object(self, obj: MCMObject) -> None: ...

    @abstractmethod
    def get_object(self, object_id: str) -> MCMObject | None: ...

    @abstractmethod
    def find_objects(self, *, type: ObjectType | None = None,
                     name: str | None = None) -> list[MCMObject]: ...

    @abstractmethod
    def all_objects(self) -> Iterable[MCMObject]: ...

    # --- relations --------------------------------------------------------
    @abstractmethod
    def put_relation(self, relation: MCMRelation) -> None: ...

    @abstractmethod
    def get_relation(self, relation_id: str) -> MCMRelation | None: ...

    @abstractmethod
    def relations_for(self, object_id: str, *, direction: Direction = "any",
                      types: Iterable[RelationType] | None = None,
                      include_derived: bool = False,
                      as_of: datetime | None = None,
                      include_historical: bool = False) -> list[MCMRelation]:
        """Relations touching ``object_id``.

        ``direction`` is relative to argument position: ``out`` means the object
        is arguments[0], ``in`` means it appears at any later position.

        ``include_derived`` defaults to False. Reads return asserted facts unless
        a caller explicitly asks for inferences (spec Rule 2).

        Every read is a read at a point in time (spec section 18). ``as_of``
        defaults to now, so a relation closed by a later refactor drops out of
        ordinary queries without being deleted. ``include_historical=True``
        ignores validity entirely and returns every interval on record.
        """

    @abstractmethod
    def all_relations(self, *, include_derived: bool = False,
                      as_of: datetime | None = None,
                      include_historical: bool = False) -> Iterable[MCMRelation]: ...

    @abstractmethod
    def close_relation(self, relation_id: str, valid_until: datetime) -> None:
        """End a relation's validity interval without deleting it (spec section 18)."""

    # --- evidence and provenance -----------------------------------------
    @abstractmethod
    def put_evidence(self, evidence: Evidence) -> None: ...

    @abstractmethod
    def get_evidence(self, evidence_id: str) -> Evidence | None: ...

    @abstractmethod
    def all_evidence(self) -> Iterable[Evidence]:
        """Every evidence record. The lexical and vector projections index these
        (spec sections 22, 23): evidence content is source text, test output and
        commit messages, which is what a search for an error string has to hit."""

    @abstractmethod
    def put_provenance(self, provenance: Provenance) -> None: ...

    @abstractmethod
    def get_provenance(self, provenance_id: str) -> Provenance | None: ...

    # --- lifecycle --------------------------------------------------------
    @abstractmethod
    def close(self) -> None: ...
