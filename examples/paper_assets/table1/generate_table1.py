#!/usr/bin/env python
"""Table 1 of the paper (scaling of the topology engine) and the
order-1 diagrams of its first row.

Corrects the header of the submitted table: for the cubic vertex
``F_abc phi_a phi_b psi_c`` the operator count is ``3n + m``, not
``2n + m``, and the observable alternates with the parity of ``n`` --
``<phi_a(x) phi_b(y) phi_c(z)>`` (m = 3) for odd ``n`` and
``<phi_a(x) phi_b(y)>`` (m = 2) for even ``n`` -- so the table now
carries an observable column.  Wall-clock is reported to three
significant digits (the submitted table printed ``0.000`` for order 1).

Outputs (next to this script):

* ``tab_scaling.tex``  -- the LaTeX table
* ``table1.md``        -- the same numbers as Markdown, plus the machine
* ``order1_diagrams.tex`` / ``order1_diagram_{k}.tex`` -- TikZ, same style
  as ``fig_example_diagrams_order2.pdf`` (blue solid C, red dashed R,
  circles for external points, squares for vertices);
  ``order1_diagram_{k}_standalone.tex`` compile on their own
* ``order1_diagrams.pdf`` -- the matplotlib rendering, with multiplicities
* ``order1_diagrams.md``  -- each DiagramTerm's LaTeX (coupling sum,
  propagators, prefactor)

Run::

    python examples/paper_assets/table1/generate_table1.py
"""
from __future__ import annotations

import platform
import time
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from sft_wick import (
    Action, DiagramRenderer, Field, Vertex, compute_moment, reset_uid_counter,
)
from sft_wick.drawing_tikz import TikzRenderer
from sft_wick.expressions import Sum

HERE = Path(__file__).resolve().parent
ORDERS = [1, 2, 3]
N_COMPONENTS = 3


def make_setup(n_components=N_COMPONENTS):
    phi = Field("phi", "physical", n_components=n_components)
    psi = Field("psi", "response", n_components=n_components)
    action = Action(vertices=[Vertex(fields=[phi, phi, psi], coupling="F")])
    return phi, psi, action


def observable(order, phi):
    """The observable for a given order, with a DISTINCT spatial label per
    external operator.

    Sharing a label (the ``phi_a(x) phi_b(x)`` spelling used before 0.4.0)
    is refused at interacting orders since 0.4.0: the spatial contraction is
    keyed by label, so same-label externals collapse and the enumeration
    loses the sum over assignments of externals to legs.  That collapse is
    what produced the submitted table's 4 and 75; the correct counts are
    6 and 80.  Coincident *points* remain fully supported -- set them
    through ``positions`` at evaluation time, which the topology count
    here does not reach.
    """
    if order % 2 == 1:
        return ([phi("a", "x"), phi("b", "y"), phi("c", "z")],
                r"\langle\varphi_a(x)\varphi_b(y)\varphi_c(z)\rangle", 3)
    return [phi("a", "x"), phi("b", "y")], r"\langle\varphi_a(x)\varphi_b(y)\rangle", 2


def double_factorial(n):
    out = 1
    while n > 1:
        out *= n
        n -= 2
    return out


def fmt_big(x):
    if x < 1e4:
        return f"{int(x)}"
    exp = int(np.floor(np.log10(x)))
    return rf"{x / 10 ** exp:.1f}\!\times\!10^{{{exp}}}"


def sig3(x):
    return f"{x:.3g}" if x >= 1e-3 else f"{x:.2e}"


def benchmark(n_repeat=3):
    rows = []
    for order in ORDERS:
        best_op = best_sp = np.inf
        for _ in range(n_repeat):
            reset_uid_counter()
            phi, psi, action = make_setup()
            obs, obs_tex, m = observable(order, phi)
            t0 = time.perf_counter()
            res_op = compute_moment(obs, action, order=order, collect_topology=False,
                                    response_phase=False)
            best_op = min(best_op, time.perf_counter() - t0)
            expr = res_op.order(order)
            n_op = len(expr.terms) if isinstance(expr, Sum) else (0 if expr == 0 else 1)

            reset_uid_counter()
            phi, psi, action = make_setup()
            obs, obs_tex, m = observable(order, phi)
            t0 = time.perf_counter()
            res_sp = compute_moment(obs, action, order=order, collect_topology=True,
                                    response_phase=False)
            best_sp = min(best_sp, time.perf_counter() - t0)
            n_sp = len(res_sp.diagram_terms(order))
        n_ops = 3 * order + m
        raw = double_factorial(n_ops - 1)
        rows.append(dict(order=order, m=m, obs_tex=obs_tex, n_ops=n_ops, raw=raw,
                         n_op=n_op, n_sp=n_sp, t_op=best_op, t_sp=best_sp,
                         reduction=raw / max(n_sp, 1)))
        print(f"order {order}: {n_ops} operators, raw {raw}, op-level terms {n_op}, "
              f"diagrams {n_sp}, t_op={best_op:.3g}s, t_sp={best_sp:.3g}s")
    return rows


def write_table(rows):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Scaling of the spatial-topology engine against direct Wick enumeration,",
        r"for the cubic vertex $F_{abc}\,\varphi_a\,\varphi_b\,\psi_c$ with $N=3$ components.",
        r"The observable alternates with the parity of $n$ so that the operator count $3n+m$",
        r"is even; the raw pairing count $(3n{+}m{-}1)!!$ is the number of operator pairings",
        r"produced by Wick's theorem on $\Sint^n\,\mathcal{O}$; \sftwick{} enumerates spatial",
        r"topologies instead, then collapses topologically equivalent ones into distinct",
        r"\texttt{DiagramTerm} objects.  Wall-clock: best of three, single core.}",
        r"\label{tab:scaling}",
        r"\begin{tabular}{@{}clccccc@{}}",
        r"\toprule",
        r"Order $n$ & Observable & Operators & Raw pairings & Distinct diagrams & Wall-clock (s) & Reduction \\",
        r" & & $3n{+}m$ & $(3n{+}m{-}1)!!$ & \sftwick{} & \sftwick{} & factor \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"  {r['order']} & ${r['obs_tex']}$ & {r['n_ops']} & ${fmt_big(r['raw'])}$ & "
            f"{r['n_sp']} & {sig3(r['t_sp'])} & ${fmt_big(int(round(r['reduction'])))}$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (HERE / "tab_scaling.tex").write_text("\n".join(lines) + "\n")

    md = ["| Order n | Observable | Operators 3n+m | Raw pairings (3n+m−1)!! | "
          "Distinct diagrams | Wall-clock (s) | Reduction |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['order']} | ${r['obs_tex']}$ | {r['n_ops']} | {r['raw']} | "
                  f"{r['n_sp']} | {sig3(r['t_sp'])} | {r['reduction']:.0f} |")
    md.append("")
    md.append(f"Machine: {platform.machine()} {platform.system()}, "
              f"Python {platform.python_version()}; best of three runs, single core.")
    md.append("")
    md.append("The diagram counts and reduction factors are exact and machine "
              "independent.  The wall-clock column is not: it is sensitive to "
              "load on the measuring machine (a factor ~2.7 has been observed "
              "between an idle and a busy run of this same script), so treat it "
              "as an order of magnitude unless it was measured on an idle "
              "machine.")
    (HERE / "table1.md").write_text("\n".join(md) + "\n")


def order1_diagrams():
    reset_uid_counter()
    phi, psi, action = make_setup()
    obs, obs_tex, _ = observable(1, phi)
    result = compute_moment(obs, action, order=1)
    dts = result.diagram_terms(1)
    n_ops = 3 * 1 + len(obs)
    raw = double_factorial(n_ops - 1)

    # How many of the raw pairings the Ito prescription removes is
    # MEASURED, not asserted: re-run with ito=False and count.  At order 1
    # there is a single psi, so no psi-psi pairing exists and <psi psi> = 0
    # prunes nothing -- the whole reduction is R(x, x) = 0.  Deriving it
    # keeps the prose honest if that ever stops being true.
    reset_uid_counter()
    phi_n, _psi_n, action_n = make_setup()
    obs_n, _tex_n, _m_n = observable(1, phi_n)
    n_no_ito = len(compute_moment(obs_n, action_n, order=1,
                                  ito=False).diagrams_by_order[1])
    # ``diagrams_by_order`` holds one record per Wick pairing; grouping by
    # canonical form gives the distinct diagrams, in the same order as the
    # DiagramTerms, whose prefactors already contain the pairing
    # multiplicities.  Counts are derived, never hardcoded -- a literal
    # ``== 4`` here is what turned the 0.4.0 label change into a traceback
    # instead of a visible recount.
    infos = result.diagrams_by_order[1]
    fds = [info.to_feynman_diagram() for info in infos]
    groups: "OrderedDict[tuple, list]" = OrderedDict()
    for fd in fds:
        groups.setdefault(fd.canonical_form(), []).append(fd)
    unique = [g[0] for g in groups.values()]
    mults = [len(g) for g in groups.values()]
    assert len(unique) == len(dts), (len(unique), len(dts))
    for fd, dt in zip(unique, dts):
        kinds_fd = sorted(d.get("kind") for _u, _v, d in fd.graph.edges(data=True))
        kinds_dt = sorted(p.kind for p in dt.propagators)
        assert kinds_fd == kinds_dt, (kinds_fd, kinds_dt)

    tikz = TikzRenderer(standalone=False)
    parts = []
    for k, fd in enumerate(unique, start=1):
        src = tikz.to_string(fd)
        (HERE / f"order1_diagram_{k}.tex").write_text(src)
        (HERE / f"order1_diagram_{k}_standalone.tex").write_text(
            tikz.to_string(fd, standalone=True))
        parts.append(f"% diagram {k}\n" + src)
    (HERE / "order1_diagrams.tex").write_text("\n".join(parts))

    renderer = DiagramRenderer(figsize=(4.2, 3.8))
    fig = renderer.draw_all(unique, ncols=4, multiplicities=mults, shared_legend=True)
    fig.savefig(HERE / "order1_diagrams.pdf", bbox_inches="tight")

    md = [f"# Order-1 diagrams of ${obs_tex}$ (cubic vertex, N = 3)", "",
          f"`compute_moment(obs, action, order=1)` returns {len(dts)} `DiagramTerm`s. "
          f"Wick's theorem pairs the {n_ops} operators in {raw} ways, of which "
          f"{n_no_ito - len(fds)} pair the vertex's psi with one of its own phi's "
          f"and vanish under the Ito prescription (R(x, x) = 0), leaving {len(fds)}; "
          f"the topology engine groups those into the {len(unique)} distinct "
          f"topologies here, with pairing multiplicities {mults} summing to "
          f"{sum(mults)}, already folded into each term's rational prefactor.  "
          "(<psi psi> = 0 removes nothing at order 1 -- there is only one psi "
          "in the expression; it starts pruning at order 2.)  Each line is the "
          "term's full LaTeX: coefficient (rational prefactor times the MSR "
          "phase), coupling sum, propagators, integrals and index sums.", ""]
    for k, (dt, mult) in enumerate(zip(dts, mults), start=1):
        props = " ".join(p.to_latex() for p in dt.propagators)
        md.append(f"{k}. multiplicity {mult}, propagators `{props}`  ")
        md.append(f"   $$ {dt.to_latex()} $$")
        md.append("")
    (HERE / "order1_diagrams.md").write_text("\n".join(md) + "\n")
    for k, dt in enumerate(dts, start=1):
        print(k, " ".join(p.to_latex() for p in dt.propagators), "|", dt.to_latex())


if __name__ == "__main__":
    rows = benchmark()
    write_table(rows)
    order1_diagrams()
    print("wrote", sorted(p.name for p in HERE.iterdir() if p.suffix in (".tex", ".md", ".pdf")))
