"""The oracle in ``validation/`` and the reference engines share no code with
sft-wick.

``validation/hierarchy.py`` is compared against the package, so it must not
reach the package's code by any import.  The AST walk checks each file's own
imports, with relative imports resolved: the oracle may import the standard
library, numpy, scipy, yaml, hypothesis, the two engines in
``examples/reference`` and its own modules other than the lowering.  Only
``validation/lower.py`` and ``validation/lower_v042.py`` (the lowering onto a
release) may import sft_wick, and no other oracle module may import them.  A
subprocess then imports every other oracle module, runs the oracle on a
case of every demo and on a model with every noise block, and checks that
no sft_wick module was loaded on the way, which catches an import made by a
dependency or built at run time.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINES = ("ito_moments", "hormander_moments")
LOWERING = ("lower.py", "lower_v042.py")
ALLOWED = {"numpy", "scipy", "yaml", "hypothesis", "validation", *ENGINES}


def _oracle_files() -> list:
    files = [f for f in sorted((ROOT / "validation").glob("*.py"))
             if f.name not in LOWERING]
    return files + [ROOT / "examples" / "reference" / f"{name}.py"
                    for name in ENGINES]


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


def _imports(path: Path, package: str = "validation") -> list:
    """``(line, dotted module)`` of every import in ``path``; a relative
    import is resolved against ``package``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                out.append((node.lineno, node.module))
            elif node.level > 1:
                out.append((node.lineno, "<beyond the package>"))
            elif node.module:
                out.append((node.lineno, f"{package}.{node.module}"))
            else:
                out += [(node.lineno, f"{package}.{a.name}")
                        for a in node.names]
        else:
            name = _literal_call(node)
            if name is not None:
                out.append((node.lineno, name))
    return out


def _forbidden(module: str) -> bool:
    top = module.split(".")[0]
    if top not in ALLOWED and top not in sys.stdlib_module_names:
        return True
    lowering = {f"validation.{name[:-3]}" for name in LOWERING}
    return any(module == m or module.startswith(m + ".") for m in lowering)


def test_VI1_the_oracle_files_import_only_allowed_modules():
    files = _oracle_files()
    assert {f.name for f in files} >= {"modelspec.py", "hierarchy.py",
                                       "demo_cases.py", "ito_moments.py"}
    bad = [f"{path.relative_to(ROOT)}:{line} imports {module}"
           for path in files for line, module in _imports(path)
           if _forbidden(module)]
    assert not bad, "\n".join(bad)


def test_VI2_the_ast_walk_sees_each_spelling_of_an_import(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("import sft_wick.evaluate\n"
                     "from sft_wick import System\n"
                     "import importlib\n"
                     "importlib.import_module('sft_wick.workflow')\n"
                     "__import__('sft_wick')\n"
                     "from . import modelspec\n"
                     "from .lower import to_system\n"
                     "from validation.lower_v042 import to_system\n"
                     "from . import lower\n"
                     "from .. import outside\n")
    found = [module for _, module in _imports(probe)]
    assert [m for m in found if m.startswith("sft_wick")] == [
        "sft_wick.evaluate", "sft_wick", "sft_wick.workflow", "sft_wick"]
    assert sum(_forbidden(m) for m in found) == 8
    assert not _forbidden("validation.modelspec")
    assert not _forbidden("validation.lowering_notes")


def test_VI3_running_the_oracle_loads_no_sft_wick_module():
    modules = [f"validation.{f.stem}" for f in _oracle_files()
               if f.parent.name == "validation" and f.stem != "__init__"]
    code = f"""
import importlib, sys
for name in {modules!r}:
    importlib.import_module(name)
from validation import demo_cases as dc
from validation.hierarchy import moments, tags_up_to
from validation.modelspec import *
seen = set()
for case in dc.all_cases():
    if case.group not in seen:
        seen.add(case.group)
        moments(case.model, case.observable, case.tags)
model = ModelSpec(
    gamma=(0.9, 1.2), source=(0.1, 0.2), quadratic=[[[0.1] * 2] * 2] * 2,
    cubic=[[[[0.1] * 2] * 2] * 2] * 2,
    ou=(OUBlock(0.4, 0.6, Kernel("gaussian", (0.8,)), (1.0, 0.5)),),
    multiplicative=Multiplicative([[0.3], [0.2]], [[[0.1, 0.0]], [[0.0, 0.1]]]),
    poisson=Poisson(1.1, (0.5, -0.4), (0.6, 0.9), "exponential", (0.4, 1.5)),
    cumulants=(Cumulant(1, (0.1, 0.2)), Cumulant(3, [[[0.1] * 2] * 2] * 2,
                                                  "white")))
obs = Observable([(0, "x"), (1, "y")], {{"x": 0.0, "y": 0.5}}, 1.0,
                 times={{"x": 0.6}}, target="connected")
moments(model, obs, tags_up_to(model, 1))
print(len(seen), sorted(m for m in sys.modules if m.startswith("sft_wick")))
"""
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "8 []"


@pytest.mark.parametrize("name", ENGINES)
def test_VI4_the_engines_load_no_sft_wick_module(name):
    code = (f"import sys\nsys.path.insert(0, 'examples/reference')\n"
            f"import {name}\n"
            "print(sorted(m for m in sys.modules if m.startswith('sft_wick')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
