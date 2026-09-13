"""Equivalence engine (spec section 38, development step 19).

Spec section 38 asks for an abstraction supporting syntactic, AST, normal-form and
semantic equivalence, and says: *for code transformations, investigate integration
with e-graphs / equality saturation rather than building a new implementation from
scratch.*

Taking that seriously settles the design. Three of the four domains need no e-graph
at all - they are digest comparisons over a parse tree this system already
produces. The fourth is undecidable, and an e-graph is where it *would* be decided.
So the saturation backend is a **seam with an honest UNKNOWN behind it**, not a
reimplementation and not a stub. Nothing here tries to be egg.

## The hierarchy is the load-bearing part

    syntactic  ⟹  AST  ⟹  normal-form  ⟹  semantic

Equality at any domain implies equality at every weaker-discriminating domain to
its right. The consequence that matters is the one running the other way:

**A negative result must never propagate rightwards.** Two functions with different
normal forms may compute exactly the same thing - that is the ordinary case, not a
corner case. So ``SEMANTIC`` can return ``EQUIVALENT`` on the strength of matching
normal forms, and must never return ``NOT_EQUIVALENT`` without a backend able to
*prove* inequivalence. Getting this backwards is the one genuinely damaging thing
this module could do: it would let a refactoring agent conclude that two
implementations differ because they were spelled differently.

Verdicts are three-way for the same reason the constraint engine's are (spec
section 13): a question that was not decided must not answer as though it were.

## What the domain is for

Spec section 10 requires an equivalence relation to name its domain, and
``RELATION_SPECS`` says where it goes: "the domain belongs in relation properties,
not in the type". ``equivalence_relation`` builds a relation that carries it.

Nothing is persisted. An equivalence found here is decidable and could legitimately
be asserted, but writing it is a separate explicit decision, the same stance
``derive`` takes with the rule engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Protocol

from ..core.ids import relation_id
from ..core.objects import MCMObject
from ..core.provenance import ExtractionMethod
from ..core.relations import MCMRelation, RelationType as RT
from ..storage.database import Store
from .normalise import Forms, forms_of


class Domain(str, Enum):
    """The equivalence domains from spec section 38."""

    SYNTACTIC = "syntactic"
    AST = "ast"
    NORMAL_FORM = "normal_form"
    SEMANTIC = "semantic"


class Verdict(str, Enum):
    EQUIVALENT = "EQUIVALENT"
    NOT_EQUIVALENT = "NOT_EQUIVALENT"
    #: Undecided. Never a synonym for NOT_EQUIVALENT.
    UNKNOWN = "UNKNOWN"


#: Weakest to strongest discrimination. Equality at one implies equality at the
#: next; inequality never implies inequality at the next.
ORDER: tuple[Domain, ...] = (Domain.SYNTACTIC, Domain.AST, Domain.NORMAL_FORM,
                             Domain.SEMANTIC)

#: Domains answerable from a stored digest, and the property each one reads.
DIGEST_PROPERTY: dict[Domain, str] = {
    Domain.SYNTACTIC: "text_digest",
    Domain.AST: "ast_digest",
    Domain.NORMAL_FORM: "normal_form_digest",
}

#: What deciding semantic equivalence would take. Reported verbatim when the
#: question is refused, the way the constraint engine names the runtime values it
#: would need (spec section 13).
SEMANTIC_REQUIREMENT = (
    "equality saturation over a rewrite system, a solver, or differential testing; "
    "no saturation backend is configured"
)


@dataclass(frozen=True)
class Fingerprint:
    """The digests a definition can be compared by.

    Any field may be ``None``: a definition that did not parse has no normal form,
    and the engine reports that as UNKNOWN rather than treating absence as
    difference.
    """

    text: str | None = None
    ast: str | None = None
    normal_form: str | None = None

    def digest(self, domain: Domain) -> str | None:
        if domain is Domain.SYNTACTIC:
            return self.text
        if domain is Domain.AST:
            return self.ast
        return self.normal_form


@dataclass(frozen=True)
class EquivalenceResult:
    domain: Domain
    verdict: Verdict
    reason: str
    #: The domain that actually settled it. For a semantic EQUIVALENT this is
    #: NORMAL_FORM, because that is what was checked and what a reader should be
    #: able to audit.
    witness: Domain | None = None

    @property
    def equivalent(self) -> bool:
        return self.verdict is Verdict.EQUIVALENT


class SaturationBackend(Protocol):
    """Adapter for an e-graph or equality-saturation engine (spec section 5).

    An ``egglog`` or ``egg`` adapter is this protocol over that library's API. None
    is bundled: it cannot be exercised offline, and an untested backend in the
    default install would be a worse claim than no backend. The engine works
    without one and says what it is missing.
    """

    @property
    def name(self) -> str: ...

    def compare(self, left: str, right: str) -> tuple[Verdict, str]:
        """Decide two definitions' semantic equivalence from their source."""


class NoBackend:
    """The default. Refuses the question and names what would answer it."""

    @property
    def name(self) -> str:
        return "none"

    def compare(self, left: str, right: str) -> tuple[Verdict, str]:
        return Verdict.UNKNOWN, SEMANTIC_REQUIREMENT


class EquivalenceEngine:
    """Compares definitions under a stated domain."""

    def __init__(self, backend: SaturationBackend | None = None) -> None:
        self.backend = backend if backend is not None else NoBackend()

    def compare(self, left: Fingerprint, right: Fingerprint, *,
                domain: Domain = Domain.NORMAL_FORM,
                sources: tuple[str, str] | None = None) -> EquivalenceResult:
        """Compare two fingerprints under ``domain``.

        ``sources`` is needed only for ``SEMANTIC`` with a backend configured, and
        only when the normal forms already differ.
        """
        if domain is Domain.SEMANTIC:
            return self._semantic(left, right, sources)
        return self._by_digest(left, right, domain)

    def compare_sources(self, left: str, right: str, *,
                        domain: Domain = Domain.NORMAL_FORM) -> EquivalenceResult:
        return self.compare(fingerprint_of(left), fingerprint_of(right),
                            domain=domain, sources=(left, right))

    def compare_objects(self, store: Store, left_id: str, right_id: str, *,
                        domain: Domain = Domain.NORMAL_FORM) -> EquivalenceResult:
        """Compare two ingested objects by their stored digests.

        No source access. Ingestion records the digests, so this answers over any
        database including one whose repository is no longer on disk.
        """
        left = store.get_object(left_id)
        right = store.get_object(right_id)
        for object_id, obj in ((left_id, left), (right_id, right)):
            if obj is None:
                raise KeyError(f"unknown object: {object_id}")
        return self.compare(fingerprint_from(left), fingerprint_from(right),
                            domain=domain)

    # --- domains ----------------------------------------------------------

    def _by_digest(self, left: Fingerprint, right: Fingerprint,
                   domain: Domain) -> EquivalenceResult:
        a, b = left.digest(domain), right.digest(domain)
        if a is None or b is None:
            return EquivalenceResult(
                domain, Verdict.UNKNOWN,
                f"no {domain.value} form recorded for one of the definitions; "
                "it did not parse, or predates this property being stored")
        if a == b:
            return EquivalenceResult(domain, Verdict.EQUIVALENT,
                                     f"identical {domain.value} form", domain)
        return EquivalenceResult(domain, Verdict.NOT_EQUIVALENT,
                                 f"{domain.value} forms differ", domain)

    def _semantic(self, left: Fingerprint, right: Fingerprint,
                  sources: tuple[str, str] | None) -> EquivalenceResult:
        """Semantic equivalence, answered only where it can be.

        The positive direction comes free from the hierarchy: two definitions with
        the same normal form differ only in the names of their own parameters, so
        their semantics coincide.

        The negative direction does not come free and is never inferred. Different
        normal forms are the normal state of two implementations of the same thing.
        """
        normal = self._by_digest(left, right, Domain.NORMAL_FORM)
        if normal.verdict is Verdict.EQUIVALENT:
            return EquivalenceResult(
                Domain.SEMANTIC, Verdict.EQUIVALENT,
                "normal forms coincide, so the definitions differ only in the "
                "names they bind; semantics follow",
                Domain.NORMAL_FORM)

        if sources is not None and not isinstance(self.backend, NoBackend):
            verdict, reason = self.backend.compare(*sources)
            return EquivalenceResult(Domain.SEMANTIC, verdict,
                                     f"{self.backend.name}: {reason}",
                                     Domain.SEMANTIC)

        return EquivalenceResult(
            Domain.SEMANTIC, Verdict.UNKNOWN,
            "normal forms differ, which says nothing about semantics; deciding "
            f"this needs {SEMANTIC_REQUIREMENT}")


# --- fingerprints ---------------------------------------------------------

def fingerprint_of(source: str) -> Fingerprint:
    """Fingerprint a definition from its source text."""
    import hashlib  # noqa: PLC0415 - local to keep the module's imports about equivalence

    forms: Forms | None = forms_of(source)
    text = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if forms is None:
        return Fingerprint(text=text)
    return Fingerprint(text=text, ast=forms.ast, normal_form=forms.normal_form)


def fingerprint_from(obj: MCMObject) -> Fingerprint:
    """Fingerprint an ingested object from the digests stored on it."""
    return Fingerprint(
        text=obj.properties.get(DIGEST_PROPERTY[Domain.SYNTACTIC]),
        ast=obj.properties.get(DIGEST_PROPERTY[Domain.AST]),
        normal_form=obj.properties.get(DIGEST_PROPERTY[Domain.NORMAL_FORM]),
    )


# --- equivalence classes ---------------------------------------------------

@dataclass
class EquivalenceClass:
    domain: Domain
    digest: str
    members: list[MCMObject]

    @property
    def size(self) -> int:
        return len(self.members)


def equivalence_classes(store: Store, *, domain: Domain = Domain.NORMAL_FORM,
                        minimum: int = 2) -> list[EquivalenceClass]:
    """Group ingested definitions that share a form.

    A grouping by stored digest, so it costs one pass rather than comparing every
    pair. Classes smaller than ``minimum`` are dropped, which by default means only
    genuine duplicates are returned.

    ``SEMANTIC`` is refused rather than silently answered at ``NORMAL_FORM``:
    semantic classes are exactly what cannot be computed by grouping, since two
    members of one can have nothing syntactic in common.
    """
    if domain is Domain.SEMANTIC:
        raise ValueError(
            "semantic equivalence classes cannot be computed by grouping: two "
            f"members may share no form at all. Deciding them needs "
            f"{SEMANTIC_REQUIREMENT}")

    prop = DIGEST_PROPERTY[domain]
    buckets: dict[str, list[MCMObject]] = {}
    for obj in store.all_objects():
        digest = obj.properties.get(prop)
        if digest:
            buckets.setdefault(digest, []).append(obj)

    classes = [
        EquivalenceClass(domain=domain, digest=digest,
                         members=sorted(members, key=lambda o: o.id))
        for digest, members in buckets.items() if len(members) >= minimum
    ]
    classes.sort(key=lambda c: (-c.size, c.members[0].id))
    return classes


def equivalence_relation(left_id: str, right_id: str,
                         result: EquivalenceResult) -> MCMRelation:
    """Build the relation an equivalence would assert. Not persisted.

    Spec section 10 requires an equivalence to name its domain, and
    ``RELATION_SPECS`` says the domain belongs in relation properties rather than
    in the type. So both semantic and syntactic equivalence use ``EQUIVALENT_TO``
    and differ in a property, except that a result witnessed semantically uses
    ``SEMANTICALLY_EQUIVALENT_TO`` because spec section 9 provides it.
    """
    if result.verdict is not Verdict.EQUIVALENT:
        raise ValueError("only an EQUIVALENT result can be asserted as a relation")
    relation_type = (RT.SEMANTICALLY_EQUIVALENT_TO
                     if result.domain is Domain.SEMANTIC else RT.EQUIVALENT_TO)
    arguments = sorted([left_id, right_id])
    return MCMRelation(
        id=relation_id(relation_type.value, arguments,
                       ExtractionMethod.STATIC_ANALYSIS.value),
        relation_type=relation_type,
        arguments=arguments,
        properties={"domain": result.domain.value,
                    "witness": result.witness.value if result.witness else None,
                    "reason": result.reason},
    )


def explain(classes: Iterable[EquivalenceClass]) -> str:
    classes = list(classes)
    if not classes:
        return ("No two definitions share a form. Note that this rules out "
                "duplication, not similarity: semantic equivalence is not "
                "decided here.")
    lines = [f"{len(classes)} equivalence classes "
             f"({classes[0].domain.value} domain)", ""]
    for klass in classes:
        lines.append(f"{klass.size} definitions share a {klass.domain.value} form:")
        for member in klass.members:
            where = member.properties.get("relpath") or ""
            lines.append(f"  {member.name}  [{member.type.value}] {where}".rstrip())
        lines.append("")
    lines.append("Nothing was written. Asserting these as EQUIVALENT_TO relations "
                 "is a separate decision.")
    return "\n".join(lines)
