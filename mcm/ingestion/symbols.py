"""Repository-wide symbol table used to resolve names to object IDs.

Resolution is intentionally conservative. Anything that would need type inference
(``user.is_active()`` where ``user`` is a local) is reported as unresolved rather
than guessed. An unresolved call is visible in the ingestion report; a guessed one
would silently become a false dependency edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.ids import library_id, path_id, symbol_id
from ..core.objects import ObjectType
from .parser import ParsedModule

#: Symbol kinds that get an object ID, mapped to their ObjectType.
KIND_TO_TYPE = {
    "function": ObjectType.FUNCTION,
    "method": ObjectType.METHOD,
    "class": ObjectType.CLASS,
    "test": ObjectType.TEST,
}


def module_name_for(relpath: str) -> str:
    """Dotted module name implied by a path: ``tests/test_auth.py`` -> ``tests.test_auth``."""
    stem = relpath[:-3] if relpath.endswith(".py") else relpath
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.replace("/", ".")


def is_test_file(relpath: str) -> bool:
    base = relpath.rsplit("/", 1)[-1]
    return base.startswith("test_") or base.endswith("_test.py") or "/tests/" in f"/{relpath}"


def symbol_kind(relpath: str, kind: str, name: str) -> str:
    """Promote test functions to the ``test`` kind so they get ObjectType.TEST."""
    if kind == "function" and name.startswith("test_") and is_test_file(relpath):
        return "test"
    return kind


@dataclass
class SymbolTable:
    repo: str
    #: relpath -> {qualname -> object_id}
    by_module: dict[str, dict[str, str]] = field(default_factory=dict)
    #: dotted module name -> relpath, plus unambiguous bare basenames
    modules: dict[str, str] = field(default_factory=dict)
    #: relpath -> {local_name -> target_object_id}
    bindings: dict[str, dict[str, str]] = field(default_factory=dict)

    def register_module(self, parsed: ParsedModule) -> None:
        relpath = parsed.relpath
        self.by_module.setdefault(relpath, {})
        dotted = module_name_for(relpath)
        self.modules[dotted] = relpath
        bare = dotted.rsplit(".", 1)[-1]
        # A bare basename resolves only while it stays unambiguous.
        if bare not in self.modules:
            self.modules[bare] = relpath
        elif self.modules[bare] != relpath:
            self.modules[bare] = ""  # ambiguous: refuse to resolve

        for sym in parsed.symbols:
            kind = symbol_kind(relpath, sym.kind, sym.name)
            self.by_module[relpath][sym.qualname] = symbol_id(
                self.repo, relpath, kind, sym.qualname
            )

    def resolve_module(self, dotted: str) -> str | None:
        """Return the relpath of an internal module, or None if external/ambiguous."""
        relpath = self.modules.get(dotted)
        if relpath:
            return relpath
        # ``from a.b import c`` where a/b.py exists but a/ is not a package root
        tail = dotted.rsplit(".", 1)[-1]
        relpath = self.modules.get(tail)
        return relpath or None

    def lookup(self, relpath: str, qualname: str) -> str | None:
        return self.by_module.get(relpath, {}).get(qualname)

    def bind_imports(self, parsed: ParsedModule) -> None:
        """Map each name a module imports onto the object it refers to."""
        relpath = parsed.relpath
        local = self.bindings.setdefault(relpath, {})
        for imp in parsed.imports:
            target_relpath = self.resolve_module(imp.module)
            if imp.names:
                for name in imp.names:
                    if target_relpath:
                        object_id = self.lookup(target_relpath, name)
                        if object_id is None:
                            # Imported name is not a symbol we parsed (a constant,
                            # a re-export). Bind to the module file instead.
                            object_id = path_id(self.repo, target_relpath, "file")
                    else:
                        object_id = library_id(f"{imp.module}.{name}")
                    local[name] = object_id
            else:
                bound_as = imp.alias or imp.module.split(".")[0]
                local[bound_as] = (
                    path_id(self.repo, target_relpath, "file") if target_relpath
                    else library_id(imp.module)
                )
