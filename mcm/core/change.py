"""Proposed changes (spec section 30).

Spec section 30 defines change propagation as

    X_{t+1} = F(X_t, Δ)

and asks the system to estimate ``ΔA ⇒ {ΔB, ΔC, ΔD}`` *before* the agent executes
anything. Impact analysis (spec section 29) answers the weaker question "what could
break if A changes", treating Δ as opaque. This module is the Δ.

**Why four kinds and not twenty.** Spec Rule 8 forbids building a large ontology
before the basic hypothesis is tested, and there is a sharper constraint than that:
spec section 61 will compare a *predicted* change against an *observed* one, so a
change kind that cannot be read off a diff by the parser this system already has is
a kind that can never be checked. Each of these four can be:

    REMOVE      the definition is no longer in the file
    RENAME      the definition's name changed, its body did not
    SIGNATURE   the parameter list changed
    BEHAVIOUR   the body changed, the interface did not

"Performance regression", "semantics-preserving refactor" and "stricter validation"
are all things an agent might want to say and none of them is decidable from an
AST, so none of them is here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChangeKind(str, Enum):
    REMOVE = "REMOVE"
    RENAME = "RENAME"
    SIGNATURE = "SIGNATURE"
    BEHAVIOUR = "BEHAVIOUR"


#: What each kind does to the *interface* an object presents to its dependents.
#: This is the property the propagation table reads, and keeping it here means a
#: fifth change kind declares its own semantics instead of being special-cased.
BREAKS_REFERENCE: frozenset[ChangeKind] = frozenset({
    ChangeKind.REMOVE, ChangeKind.RENAME,
})

#: Kinds that leave the name intact but change how it must be called.
BREAKS_CALL: frozenset[ChangeKind] = frozenset({
    ChangeKind.REMOVE, ChangeKind.RENAME, ChangeKind.SIGNATURE,
})

#: An *observed* edit that changed a definition's text without changing its parse
#: tree: a reformat, a comment, a quote style. Not a ``ChangeKind`` because there is
#: nothing to propagate - the code means exactly what it meant before - and spec
#: section 61 would otherwise count every reformat as a behaviour change whose
#: predicted consequences no commit confirms. Detected by the equivalence engine
#: (spec section 38).
COSMETIC = "COSMETIC"

#: An *observed* edit that is not a change to an existing object: the definition
#: was not there at the previous revision. Deliberately not a ``ChangeKind``,
#: because nothing can be proposed about an object that does not exist yet and
#: nothing can depend on it, so it propagates nowhere and is excluded from the
#: prediction comparison in spec section 61.
ADDED = "ADDED"


@dataclass(frozen=True)
class Change:
    """A proposed or predicted change to one object.

    The same type is used for the change an agent proposes and for the changes
    this system predicts will follow from it, because spec section 30 writes both
    sides of ``ΔA ⇒ {ΔB, ΔC, ΔD}`` as changes, and because spec section 61 has to
    compare a prediction with an observation of the same shape.
    """

    target_id: str
    kind: ChangeKind
    details: str = ""

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("a Change requires a target object")
        if not isinstance(self.kind, ChangeKind):
            object.__setattr__(self, "kind", ChangeKind(self.kind))

    @property
    def breaks_reference(self) -> bool:
        """True when dependents can no longer name the target as they do now."""
        return self.kind in BREAKS_REFERENCE

    @property
    def breaks_call(self) -> bool:
        """True when existing call sites stop being correct."""
        return self.kind in BREAKS_CALL

    def describe(self) -> str:
        return f"{self.kind.value} {self.target_id}" + (
            f" ({self.details})" if self.details else "")
