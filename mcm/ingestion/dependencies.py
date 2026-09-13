"""Turn resolved names into typed relations with evidence and provenance.

Two extractors run here, and they are deliberately distinguished (spec section 54):

* AST extraction, confidence 1.0. ``CALLS``, ``IMPORTS``, ``USES``, ``INHERITS``
  are read straight off the syntax tree.
* A static-analysis heuristic, confidence below 1.0. ``TESTS`` is asserted when a
  pytest-style test function calls a repository symbol. Calling is not the same as
  testing - a test may call a helper - so this claim is weaker than the ``CALLS``
  fact it sits alongside, and carries its own evidence saying so.

Both relations are asserted (not derived): each is read from an artifact by a
named extractor, rather than composed from other relations. Relation IDs include
the extraction method, so the two coexist without collision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.evidence import Evidence, EvidenceType
from ..core.ids import library_id, path_id, relation_id
from ..core.provenance import ExtractionMethod, Provenance
from ..core.relations import MCMRelation, RelationType as RT
from .parser import ParsedModule
from .symbols import SymbolTable, is_test_file

#: Confidence for the "a test that calls X tests X" heuristic.
TESTS_HEURISTIC_CONFIDENCE = 0.9

AGENT = "mcm.ingestion"


@dataclass
class RelationBatch:
    """Relations plus the evidence and provenance records they point at."""

    relations: list[MCMRelation] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    provenance: list[Provenance] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    def add(self, relation: MCMRelation, evidence: Evidence, provenance: Provenance) -> None:
        self.relations.append(relation)
        self.evidence.append(evidence)
        self.provenance.append(provenance)

    def extend(self, other: "RelationBatch") -> None:
        self.relations.extend(other.relations)
        self.evidence.extend(other.evidence)
        self.provenance.extend(other.provenance)
        self.unresolved.extend(other.unresolved)


def assert_relation(relation_type: RT, arguments: list[str], *,
                    method: ExtractionMethod, source_ref: str, evidence_type: EvidenceType,
                    content: str, confidence: float = 1.0,
                    properties: dict | None = None) -> tuple[MCMRelation, Evidence, Provenance]:
    """Build an asserted relation together with its evidence and provenance.

    Spec Rule 4: every important relation needs provenance. Making this the only
    convenient way to create a relation is how that rule is enforced.
    """
    evidence = Evidence.create(
        source_type=evidence_type, source_ref=source_ref, content=content,
        extraction_method=method.value, confidence=confidence,
    )
    provenance = Provenance.create(method=method, agent=AGENT, source_ref=source_ref)
    relation = MCMRelation(
        id=relation_id(relation_type.value, arguments, method.value),
        relation_type=relation_type,
        arguments=arguments,
        properties=properties or {},
        confidence=confidence,
        evidence_ids=[evidence.id],
        provenance_id=provenance.id,
    )
    return relation, evidence, provenance


def extract_imports(parsed: ParsedModule, table: SymbolTable, file_id: str) -> RelationBatch:
    batch = RelationBatch()
    for imp in parsed.imports:
        target_relpath = table.resolve_module(imp.module)
        if target_relpath:
            target_id = path_id(table.repo, target_relpath, "file")
        else:
            target_id = library_id(imp.module)
        source_ref = parsed.relpath + ":" + str(imp.line)
        if imp.names:
            statement = "from " + imp.module + " import " + ", ".join(imp.names)
        else:
            statement = "import " + imp.module
        batch.add(*assert_relation(
            RT.IMPORTS, [file_id, target_id],
            method=ExtractionMethod.AST, source_ref=source_ref,
            evidence_type=EvidenceType.AST, content=statement,
            properties={"line": imp.line},
        ))
    return batch


def extract_inheritance(parsed: ParsedModule, table: SymbolTable) -> RelationBatch:
    batch = RelationBatch()
    bindings = table.bindings.get(parsed.relpath, {})
    for sym in parsed.symbols:
        if sym.kind != "class":
            continue
        child_id = table.lookup(parsed.relpath, sym.qualname)
        if child_id is None:
            continue
        for base in sym.bases:
            source_ref = parsed.relpath + ":" + str(sym.start_line)
            parent_id = bindings.get(base) or table.lookup(parsed.relpath, base)
            if parent_id is None:
                batch.unresolved.append(source_ref + " base " + base)
                continue
            batch.add(*assert_relation(
                RT.INHERITS, [child_id, parent_id],
                method=ExtractionMethod.AST,
                source_ref=source_ref,
                evidence_type=EvidenceType.AST,
                content="class " + sym.qualname + " inherits " + base,
            ))
    return batch


def extract_calls(parsed: ParsedModule, table: SymbolTable) -> RelationBatch:
    batch = RelationBatch()
    bindings = table.bindings.get(parsed.relpath, {})
    test_file = is_test_file(parsed.relpath)

    for call in parsed.calls:
        if call.caller_qualname is None:
            continue  # module-level call: no enclosing symbol to attribute it to
        caller_id = table.lookup(parsed.relpath, call.caller_qualname)
        if caller_id is None:
            continue
        source_ref = parsed.relpath + ":" + str(call.line)

        if call.attribute is None:
            target_id = bindings.get(call.base) or table.lookup(parsed.relpath, call.base)
            if target_id is None:
                batch.unresolved.append(source_ref + " call " + call.callee_raw)
                continue
            batch.add(*assert_relation(
                RT.CALLS, [caller_id, target_id],
                method=ExtractionMethod.AST, source_ref=source_ref,
                evidence_type=EvidenceType.AST,
                content=call.caller_qualname + " calls " + call.callee_raw,
                properties={"line": call.line},
            ))
            if test_file and call.caller_qualname.startswith("test_"):
                batch.add(*assert_relation(
                    RT.TESTS, [caller_id, target_id],
                    method=ExtractionMethod.STATIC_ANALYSIS, source_ref=source_ref,
                    evidence_type=EvidenceType.TEST,
                    content=("test function " + call.caller_qualname + " invokes "
                             + call.callee_raw + "; treated as test coverage"),
                    confidence=TESTS_HEURISTIC_CONFIDENCE,
                    properties={"line": call.line, "heuristic": "test_calls_subject"},
                ))
            continue

        # Attribute call. Resolvable only when the base is an imported module or
        # library; anything else needs type inference, which V1 does not do.
        base_target = bindings.get(call.base)
        if base_target is None:
            batch.unresolved.append(
                source_ref + " attribute call " + call.callee_raw
                + " (base " + call.base + " needs type inference)"
            )
            continue
        batch.add(*assert_relation(
            RT.USES, [caller_id, base_target],
            method=ExtractionMethod.AST, source_ref=source_ref,
            evidence_type=EvidenceType.AST,
            content=call.caller_qualname + " uses " + call.callee_raw,
            properties={"line": call.line, "member": call.attribute},
        ))
    return batch
