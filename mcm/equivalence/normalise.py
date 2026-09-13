"""Normal forms for a definition (spec section 38).

Two digests per definition, both computed from its source text:

``ast``
    The parse tree with comments removed. Two definitions sharing this digest
    differ only in comments and whitespace.
``normal_form``
    The same tree with the definition's own name and its parameters renamed to
    positional placeholders. Two definitions sharing this digest differ only in
    what they call themselves and their arguments.

**Every node is included, named and anonymous.** Skipping anonymous tokens is the
usual shortcut for AST comparison and it is wrong here: tree-sitter records the
operator in ``a + b`` as an anonymous child, so dropping those would make ``a + b``
and ``a - b`` identical. Keeping them means a purely cosmetic difference that
tree-sitter tokenises - an added trailing comma, for instance - reads as a
difference too.

That error is one-directional, which is the point. These digests can say two texts
differ when they are cosmetically the same; they can never say two texts match when
they do not. Given the domain hierarchy in ``engine.py`` - where a negative result
is never allowed to propagate to a stronger domain - that is the safe direction.

Docstrings are kept in the digest. Dropping them would let a docstring rewrite be
reported as a cosmetic edit, and a docstring is a string literal the module really
does carry.
"""

from __future__ import annotations

import hashlib
import textwrap
from dataclasses import dataclass

from tree_sitter import Language, Node, Parser
import tree_sitter_python

_LANGUAGE = Language(tree_sitter_python.language())

#: Node types whose content is not part of the program.
_IGNORED = frozenset({"comment"})

#: Definition nodes a rename map can be built from.
_DEFINITIONS = frozenset({"function_definition", "class_definition"})


@dataclass(frozen=True)
class Forms:
    """The normal forms of one definition."""

    ast: str
    normal_form: str


def forms_of(source: str) -> Forms | None:
    """Digest a definition's source, or ``None`` if it does not parse.

    ``None`` rather than a digest of the broken text: a definition that cannot be
    parsed has no normal form, and inventing one would put two unparseable
    definitions in the same equivalence class for the wrong reason.

    The source is dedented first. A method's text arrives indented, and an indented
    ``def`` is not a parseable unit on its own.
    """
    text = textwrap.dedent(source).strip()
    if not text:
        return None
    tree = Parser(_LANGUAGE).parse(text.encode("utf-8"))
    root = tree.root_node
    if root.has_error:
        return None

    definition = _definition_node(root)
    renames = _rename_map(definition) if definition is not None else {}
    return Forms(
        ast=_digest(_tokens(root, {})),
        normal_form=_digest(_tokens(root, renames)),
    )


def _definition_node(root: Node) -> Node | None:
    """The function or class this source defines, if it defines exactly one."""
    for child in root.children:
        if child.type == "decorated_definition":
            target = child.child_by_field_name("definition")
            if target is not None:
                return target
        if child.type in _DEFINITIONS:
            return child
    return None


def _rename_map(definition: Node) -> dict[str, str]:
    """The definition's own name and its parameters, mapped to placeholders.

    Only these. Every other identifier in the body is a *free* name - a call to
    another function, an imported symbol, an attribute - and renaming those would
    erase the references that the rest of the semantic model is built on. Two
    functions that call different things are not equivalent under any domain this
    module implements.

    A local variable that shadows a parameter is therefore renamed with it, which
    is correct, and a local that does not shadow anything is left alone, which
    makes this a bounded alpha-equivalence rather than a complete one. Two
    implementations differing only in the name of a temporary are *not* reported
    as normal-form equivalent, and that limit is deliberate: deciding it needs
    scope analysis this phase does not have.
    """
    names: list[str] = []
    name_node = definition.child_by_field_name("name")
    if name_node is not None:
        names.append(_text(name_node))
    parameters = definition.child_by_field_name("parameters")
    if parameters is not None:
        names.extend(_parameter_names(parameters))

    mapping: dict[str, str] = {}
    for name in names:
        if name not in mapping:
            mapping[name] = f"v{len(mapping)}"
    return mapping


def _parameter_names(parameters: Node) -> list[str]:
    """Parameter names, through typed and defaulted forms.

    ``x``, ``x: int``, ``x=1``, ``*args`` and ``**kwargs`` all bind a name, and the
    first identifier under each parameter node is it.
    """
    out: list[str] = []
    for child in parameters.named_children:
        if child.type == "identifier":
            out.append(_text(child))
            continue
        found = _first_identifier(child)
        if found is not None:
            out.append(found)
    return out


def _first_identifier(node: Node) -> str | None:
    if node.type == "identifier":
        return _text(node)
    for child in node.named_children:
        found = _first_identifier(child)
        if found is not None:
            return found
    return None


def _tokens(node: Node, renames: dict[str, str]) -> list[str]:
    """Flatten the tree to a token stream, substituting renamed identifiers."""
    out: list[str] = []
    _collect(node, renames, out)
    return out


def _collect(node: Node, renames: dict[str, str], out: list[str]) -> None:
    if node.type in _IGNORED:
        return
    if node.child_count == 0:
        text = _text(node)
        if node.type == "identifier" and text in renames:
            out.append(f"identifier:{renames[text]}")
        else:
            out.append(f"{node.type}:{text}")
        return
    out.append(node.type)
    for child in node.children:
        _collect(child, renames, out)
    out.append("/" + node.type)


def _text(node: Node) -> str:
    return node.text.decode("utf-8", errors="replace")


def _digest(tokens: list[str]) -> str:
    """Length-prefixed, so no token's own text can forge a separator and make two
    different streams hash alike."""
    digest = hashlib.sha256()
    for token in tokens:
        encoded = token.encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii") + b":" + encoded)
    return digest.hexdigest()
