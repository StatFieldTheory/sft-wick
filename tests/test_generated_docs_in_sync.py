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
         "examples/paper_assets/demo2_kappa4/fk_diagrams.md",
         "examples/demo2/INTERPRETATION.md"],
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


def test_demo2_interpretation_quotes_the_current_numerical_budget():
    """Fixed-prose checks cannot detect stale interpolated error estimates.

    Recompute the key error sizes and residual statistics from the saved
    arrays, independently of the figure generator's formatted prose.
    """
    import numpy as np

    text = (REPO / "examples/demo2/INTERPRETATION.md").read_text()
    with np.load(REPO / "examples/paper_assets/demo2_kappa4/budget.npz",
                 allow_pickle=False) as b:
        for channel, tolerance in (("ffk4", 1e-4), ("fffk", 1e-4), ("ffff", 1e-3)):
            for pair in ("00", "01", "11"):
                ref = b[f"ref_{channel}_{pair}"]
                error = np.abs(b[f"{channel}_{pair}"] - ref)
                nonzero = ref != 0
                relative = np.max(error[nonzero] / np.abs(ref[nonzero])) if nonzero.any() else 0
                assert (f"| {channel.upper()} | {pair} | {error.max():.2e} | "
                        f"{relative:.2e} |") in text
                assert np.all(error <= tolerance*np.abs(ref) + 1e-20)
        assert f"{b['ref_fffk_01'][-1, 0]:.9e}" in text
        assert f"{b['ref_fk5_01'][-1, 0]:.6e}" in text

        # r=0 is a node of both theory grids and the simulation grid.
        assert b["r"][0] == b["r_sub"][0] == b["r_sim"][0] == 0.0
        total = sum(b[f"{channel}_01"][:, 0] for channel in
                    ("o0_exact", "ff_exact", "fk", "ffk4", "fffk", "ffff"))
        residual = b["sim_extrap_xi"][1, :, 0] - total
        pull = residual / b["sim_extrap_err"][1, :, 0]
        leading_pull = ((b["sim_extrap_xi"][1, :, 0] - b["fk_01"][:, 0])
                        / b["sim_extrap_err"][1, :, 0])
        assert (f"χ² from {np.sum(leading_pull**2):.1f} to "
                f"{np.sum(pull**2):.1f}") in text
        assert f"{np.mean(pull):+.2f}" in text

        i = int(np.argmin(np.abs(b["t"] - 15.0)))
        value = b["sim_extrap_xi"][1, i, 0]
        error = b["sim_extrap_err"][1, i, 0]
        assert f"{value:.3e} ± {error:.2e}" in text
        assert f"{total[i]:.3e}" in text
        assert f"{residual[i]:+.2e} ({pull[i]:+.1f}σ)" in text


def test_demo2_diagonal_figure_includes_every_listed_channel(monkeypatch):
    """FFFK_00 is nonzero: the plotted total must agree with the budget."""
    import importlib.util
    import numpy as np
    from matplotlib.figure import Figure

    path = REPO / "examples/paper_assets/demo2_kappa4/make_figures.py"
    spec = importlib.util.spec_from_file_location("demo2_budget_figures_test", path)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)
    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *args, **kwargs: saved.append(fig))
    figures.fig_xi00_vs_time()
    expected = sum(figures.B[f"{channel}_00"][:, 0] for channel in
                   ("o0_exact", "ff_exact", "fk", "ffk4", "fffk", "ffff"))
    simulation = figures.B["sim_extrap_xi"][0, :, 0]
    for axis, want in zip(saved[0].axes, (expected, simulation - expected)):
        assert any(np.shape(line.get_ydata()) == want.shape
                   and np.allclose(line.get_ydata(), want, rtol=1e-13, atol=1e-20)
                   for line in axis.lines), "figure omits a nonzero channel from its total"
