"""Symbolic retrieval: exact lookup (spec section 24).

Exact-match retrieval over names and identifiers. No similarity, no ranking by
embedding. Spec section 22 is explicit that the vector projection "must not be
responsible for exact dependency reasoning" - this module is what it must not
replace.
"""

from __future__ import annotations

from ..core.objects import MCMObject
from ..storage.database import Store


def resolve(store: Store, reference: str) -> list[MCMObject]:
    """Find objects matching a reference, most specific interpretation first.

    Accepts a full object ID, a ``relpath:qualname`` pair, or a bare name. A bare
    name can legitimately match several objects (a function and the test that
    shares its stem), so this returns a list and lets the caller decide. Guessing
    here would put an arbitrary choice underneath every downstream inference.
    """
    exact = store.get_object(reference)
    if exact is not None:
        return [exact]

    if ":" in reference and "#" not in reference:
        relpath, _, qualname = reference.partition(":")
        matches = [
            obj for obj in store.all_objects()
            if obj.properties.get("relpath") == relpath
            and obj.properties.get("qualname") == qualname
        ]
        if matches:
            return matches

    by_name = store.find_objects(name=reference)
    if by_name:
        return sorted(by_name, key=lambda o: o.id)

    by_qualname = [obj for obj in store.all_objects()
                   if obj.properties.get("qualname") == reference]
    return sorted(by_qualname, key=lambda o: o.id)


def resolve_one(store: Store, reference: str) -> MCMObject:
    """Resolve to exactly one object, or raise with the ambiguity spelled out."""
    matches = resolve(store, reference)
    if not matches:
        raise KeyError(f"no object matches {reference!r}")
    if len(matches) > 1:
        options = "\n  ".join(obj.id for obj in matches)
        raise KeyError(f"{reference!r} is ambiguous; matches:\n  {options}")
    return matches[0]
