"""GD1-GD3: a committed markdown file still matches the script that writes it.

Several documentation files under ``examples/`` are written by a script:
``budget.md`` by ``demo2_kappa4/make_figures.py``, ``order1_diagrams.md`` by
``table1/generate_table1.py``, the diagram README by ``demo3/make_figures.py``.
Editing such a file by hand looks like a normal documentation change and
survives review, then the next run of the generator silently writes the old
text back.  That happened during the 0.6.0 prose pass: a rewrite of
``order1_diagrams.md`` landed while ``generate_table1.py`` still held the
previous sentence.

Re-running a generator here is not an option (minutes to an hour, and some
need caches that are not in the repository).  Instead each generator is
parsed, and every literal prose run it writes must appear verbatim in one of
its output files.  A run is the text between two interpolations, so an
f-string contributes its fixed parts and the values it formats are not
checked.  Strings that never reach the output are skipped, and the rule for
skipping them is structural rather than a list of exceptions: docstrings,
strings inside a ``raise``, and strings handed to a plotting call.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# (generator, files it writes).  A generator's runs may appear in any of them.
GENERATORS = [
    pytest.param(
        "examples/paper_assets/demo2_kappa4/make_figures.py",
        ["examples/paper_assets/demo2_kappa4/budget.md",
         "examples/paper_assets/demo2_kappa4/fk_diagrams.md"],
        id="GD1_demo2_budget",
    ),
    pytest.param(
        "examples/paper_assets/table1/generate_table1.py",
        ["examples/paper_assets/table1/table1.md",
         "examples/paper_assets/table1/order1_diagrams.md",
         "examples/paper_assets/table1/tab_scaling.tex"],
        id="GD2_table1",
    ),
    pytest.param(
        "examples/demo3/make_figures.py",
        ["examples/demo3/diagrams/README.md", "examples/demo3/README.md"],
        id="GD3_demo3_diagrams",
    ),
]

MIN_WORDS = 8          # shorter runs are table cells, axis labels, punctuation
PLOT_CALL_PREFIXES = ("set_", "plot", "legend", "suptitle", "title", "text",
                      "annotate", "bar", "axhline", "axvline", "errorbar",
                      "fill_between", "savefig", "label")


def _skipped_nodes(tree: ast.AST) -> set[int]:
    """Ids of string nodes that cannot reach the generated file."""
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                skip.add(id(body[0].value))          # docstring
        elif isinstance(node, ast.Raise):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    skip.add(id(sub))                # exception message
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", "")
            is_plot = isinstance(name, str) and name.startswith(PLOT_CALL_PREFIXES)
            for kw in node.keywords:                 # label=…, title=…
                if kw.arg in ("label", "title", "xlabel", "ylabel"):
                    for sub in ast.walk(kw.value):
                        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                            skip.add(id(sub))
            if is_plot:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        skip.add(id(sub))
    return skip


def _prose_runs(tree: ast.AST) -> list[str]:
    """Literal text runs that look like prose, f-string fragments included."""
    skip = _skipped_nodes(tree)
    runs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in skip:
                runs.append(node.value)
    return runs


def _is_prose(text: str) -> bool:
    if "$" in text or "\\" in text or "|" in text:
        return False                                  # LaTeX, table rows
    return len(text.split()) >= MIN_WORDS


@pytest.mark.parametrize("generator, outputs", GENERATORS)
def test_generated_file_matches_its_generator(generator: str,
                                              outputs: list[str]) -> None:
    gen_path = REPO / generator
    tree = ast.parse(gen_path.read_text(encoding="utf-8"))
    haystack = "\n".join((REPO / p).read_text(encoding="utf-8") for p in outputs)

    checked, drifted = 0, []
    for run in _prose_runs(tree):
        text = run.strip()
        if not _is_prose(text):
            continue
        checked += 1
        if text not in haystack:
            drifted.append(text)

    assert checked >= 3, (
        f"{generator}: only {checked} prose runs found, so this test would "
        f"pass without checking anything"
    )
    assert not drifted, (
        f"{generator} writes text that is not in {outputs}.  Either the "
        f"generated file was hand-edited (re-running the generator would "
        f"revert it) or the generator was changed without regenerating.\n"
        + "\n".join(f"  - {t[:160]!r}" for t in drifted)
    )
