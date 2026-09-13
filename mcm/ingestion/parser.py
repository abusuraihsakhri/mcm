"""Tree-sitter Python parser (spec sections 19 and 5).

Produces a purely syntactic description of one file. No inference happens here:
every fact this module emits is readable directly off the concrete syntax tree,
which is what lets ingestion mark it AST evidence at confidence 1.0
(spec Rule 5: keep deterministic code analysis separate from LLM inference).

Name resolution is deliberately *not* done here. This module reports the callee
exactly as written; resolving it to an object ID needs whole-repository context
and lives in ``resolve.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tree_sitter_python
from tree_sitter import Language, Node, Parser

_LANGUAGE = Language(tree_sitter_python.language())


@dataclass
class SymbolDef:
    """A function, method or class definition."""

    qualname: str
    name: str
    kind: str  # function | method | class
    start_line: int
    end_line: int
    #: Byte span of the definition in the source. Used to compare a definition
    #: across revisions without depending on where it sits in the file.
    start_byte: int = 0
    end_byte: int = 0
    parameters: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    docstring: str | None = None


@dataclass
class ImportDef:
    """An import statement. ``module`` is the dotted module path as written."""

    module: str
    names: list[str] = field(default_factory=list)
    alias: str | None = None
    line: int = 0


@dataclass
class CallSite:
    """A call expression, attributed to the symbol that lexically encloses it."""

    caller_qualname: str | None  # None for module-level calls
    callee_raw: str              # e.g. "validate_token" or "jwt.encode"
    base: str                    # leftmost identifier: "validate_token" / "jwt"
    attribute: str | None        # "encode" when the callee is an attribute access
    line: int


@dataclass
class ParsedModule:
    relpath: str
    symbols: list[SymbolDef] = field(default_factory=list)
    imports: list[ImportDef] = field(default_factory=list)
    calls: list[CallSite] = field(default_factory=list)


def parse_python(source: bytes, relpath: str) -> ParsedModule:
    tree = Parser(_LANGUAGE).parse(source)
    module = ParsedModule(relpath=relpath)
    _walk(tree.root_node, source, module, scope=[], in_class=False)
    return module


# --- traversal ------------------------------------------------------------

def _walk(node: Node, src: bytes, mod: ParsedModule, scope: list[str], in_class: bool) -> None:
    for child in node.children:
        if child.type == "decorated_definition":
            target = child.child_by_field_name("definition")
            if target is not None:
                _walk_definition(target, src, mod, scope, in_class)
            continue
        if child.type in ("function_definition", "class_definition"):
            _walk_definition(child, src, mod, scope, in_class)
            continue
        if child.type in ("import_statement", "import_from_statement"):
            _read_import(child, src, mod)
            continue
        if child.type == "call":
            _read_call(child, src, mod, scope)
        _walk(child, src, mod, scope, in_class)


def _walk_definition(node: Node, src: bytes, mod: ParsedModule,
                     scope: list[str], in_class: bool) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = _text(name_node, src)
    qualname = ".".join([*scope, name])
    is_class = node.type == "class_definition"

    mod.symbols.append(SymbolDef(
        qualname=qualname,
        name=name,
        kind="class" if is_class else ("method" if in_class else "function"),
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        start_byte=node.start_byte,
        end_byte=node.end_byte,
        parameters=[] if is_class else _read_parameters(node, src),
        bases=_read_bases(node, src) if is_class else [],
        docstring=_read_docstring(node, src),
    ))
    _walk(node, src, mod, [*scope, name], in_class=is_class)


def _read_parameters(node: Node, src: bytes) -> list[str]:
    params = node.child_by_field_name("parameters")
    if params is None:
        return []
    out = []
    for child in params.named_children:
        if child.type == "identifier":
            out.append(_text(child, src))
        else:
            ident = child.child_by_field_name("name")
            if ident is not None:
                out.append(_text(ident, src))
            elif child.named_children:
                out.append(_text(child.named_children[0], src))
    return out


def _read_bases(node: Node, src: bytes) -> list[str]:
    args = node.child_by_field_name("superclasses")
    if args is None:
        return []
    return [_text(c, src) for c in args.named_children if c.type in ("identifier", "attribute")]


def _read_docstring(node: Node, src: bytes) -> str | None:
    body = node.child_by_field_name("body")
    if body is None or not body.named_children:
        return None
    first = body.named_children[0]
    if first.type == "expression_statement" and first.named_children:
        literal = first.named_children[0]
        if literal.type == "string":
            return _text(literal, src).strip("\"'")
    return None


def _read_import(node: Node, src: bytes, mod: ParsedModule) -> None:
    line = node.start_point[0] + 1
    if node.type == "import_statement":
        for child in node.named_children:
            if child.type == "dotted_name":
                mod.imports.append(ImportDef(module=_text(child, src), line=line))
            elif child.type == "aliased_import":
                name = child.child_by_field_name("name")
                alias = child.child_by_field_name("alias")
                mod.imports.append(ImportDef(
                    module=_text(name, src) if name else "",
                    alias=_text(alias, src) if alias else None,
                    line=line,
                ))
        return

    module_node = node.child_by_field_name("module_name")
    module_name = _text(module_node, src) if module_node else ""
    # Node wrappers are not identity-stable across child_by_field_name calls,
    # so the module node is excluded by byte range rather than by identity.
    module_span = (module_node.start_byte, module_node.end_byte) if module_node else None
    names = [
        _text(c, src) for c in node.named_children
        if c.type in ("dotted_name", "identifier")
        and (c.start_byte, c.end_byte) != module_span
    ]
    mod.imports.append(ImportDef(module=module_name, names=names, line=line))


def _read_call(node: Node, src: bytes, mod: ParsedModule, scope: list[str]) -> None:
    fn = node.child_by_field_name("function")
    if fn is None:
        return
    raw = _text(fn, src)
    if fn.type == "identifier":
        base, attribute = raw, None
    elif fn.type == "attribute":
        obj = fn.child_by_field_name("object")
        attr = fn.child_by_field_name("attribute")
        base = _text(obj, src) if obj else raw
        attribute = _text(attr, src) if attr else None
    else:
        return  # subscripts, calls on call results: not resolvable syntactically

    mod.calls.append(CallSite(
        caller_qualname=".".join(scope) if scope else None,
        callee_raw=raw,
        base=base,
        attribute=attribute,
        line=node.start_point[0] + 1,
    ))


def _text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
