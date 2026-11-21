"""``sft_wick/_certify.py`` shares no code with the rest of the package.

Its counts are compared with those of the contraction engine, so it must
not reach the engine by any import.  An AST walk checks that every import
in the file names a standard-library module, with no relative import and no
``sft_wick`` module, and that ``importlib.import_module`` and ``__import__``
appear only in calls with a literal module name.  A subprocess then loads the file by its path, outside
the package, runs a count, and checks that no ``sft_wick`` module and no
module outside the standard library was loaded on the way.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERTIFY = ROOT / "src" / "sft_wick" / "_certify.py"
RELATIVE = "<relative import>"
DYNAMIC = "<dynamic import>"
LOADERS = ("import_module", "__import__")


def _literal_call(node: ast.AST) -> str | None:
    """The module named by ``importlib.import_module('…')`` or
    ``__import__('…')``."""
    if not (isinstance(node, ast.Call) and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)):
        return None
    func = node.func
    if ((isinstance(func, ast.Attribute) and func.attr == "import_module")
            or (isinstance(func, ast.Name) and func.id == "__import__")):
        return node.args[0].value
    return None


def _loader_name(node: ast.AST) -> str | None:
    """``import_module`` or ``__import__`` when ``node`` names one."""
    if isinstance(node, ast.Attribute) and node.attr in LOADERS:
        return node.attr
    if isinstance(node, ast.Name) and node.id in LOADERS:
        return node.id
    return None


def _dynamic(tree: ast.AST) -> list[int]:
    """Lines that reach a loader other than by a call with a literal name:
    an import of ``import_module`` by name, a call with a computed name, or
    a reference to the loader that is not a call."""
    literal = {id(node.func) for node in ast.walk(tree)
               if _literal_call(node) is not None}
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name in LOADERS for alias in node.names):
                lines.append(node.lineno)
        elif _loader_name(node) is not None and id(node) not in literal:
            lines.append(node.lineno)
    return lines


def _imports(path: Path) -> list[tuple[int, str]]:
    """``(line, dotted module)`` of every import in ``path``, with
    ``DYNAMIC`` for each line found by :func:`_dynamic`."""
    tree = ast.parse(path.read_text(), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            out.append((node.lineno,
                        RELATIVE if node.level else node.module))
        else:
            name = _literal_call(node)
            if name is not None:
                out.append((node.lineno, name))
    return out + [(line, DYNAMIC) for line in _dynamic(tree)]


def _forbidden(module: str) -> bool:
    return module.split(".")[0] not in sys.stdlib_module_names


def test_CI1_certify_imports_only_the_standard_library():
    found = _imports(CERTIFY)
    assert found, "the walk found no import at all"
    bad = [f"{CERTIFY.relative_to(ROOT)}:{line} imports {module}"
           for line, module in found if _forbidden(module)]
    assert not bad, "\n".join(bad)


def test_CI2_the_ast_walk_sees_each_spelling_of_an_import(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("import math\n"
                     "from collections.abc import Sequence\n"
                     "import sft_wick.wick\n"
                     "from sft_wick import wick\n"
                     "from . import wick\n"
                     "from .wick import r_classes\n"
                     "from .. import outside\n"
                     "import importlib\n"
                     "importlib.import_module('sft_wick.simplify')\n"
                     "__import__('sft_wick')\n"
                     "import numpy\n"
                     "def late():\n"
                     "    import sft_wick.perturbation\n"
                     "from importlib import import_module as load\n"
                     "load('sft_wick')\n"
                     "name = 'sft_wick'\n"
                     "importlib.import_module(name)\n"
                     "__import__(name)\n"
                     "loader = __import__\n")
    found = [module for _, module in _imports(probe)]
    assert sorted(m for m in found if _forbidden(m)) == sorted([
        "sft_wick.wick", "sft_wick", RELATIVE, RELATIVE, RELATIVE,
        "sft_wick.simplify", "sft_wick", "numpy", "sft_wick.perturbation",
        DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC])
    assert sorted(m for m in found if not _forbidden(m)) == [
        "collections.abc", "importlib", "importlib", "math"]


def test_CI3_loading_the_file_by_path_loads_no_package_module(tmp_path):
    code = f"""
import importlib.util, sys
before = set(sys.modules)
spec = importlib.util.spec_from_file_location("certify_alone", {str(CERTIFY)!r})
module = importlib.util.module_from_spec(spec)
sys.modules["certify_alone"] = module
spec.loader.exec_module(module)
moment = module.count_valid_pairings(
    ["x", "y", "z0", "z1"], [1, 1, 2, 2], [2, 3])
connected = module.count_valid_pairings(
    ["x", "y", "z0", "z1"], [1, 1, 2, 2], [2, 3], target="connected",
    method="dp")
loaded = {{name.split(".")[0] for name in set(sys.modules) - before}}
loaded.discard("certify_alone")
print(moment, connected,
      sorted(m for m in sys.modules if m.split(".")[0] == "sft_wick"),
      sorted(loaded - set(sys.stdlib_module_names)))
"""
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "30 28 [] []"
