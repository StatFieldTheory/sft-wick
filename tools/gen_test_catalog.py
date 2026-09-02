#!/usr/bin/env python
"""Generate ``docs/verification/catalog.rst`` -- what the test suite checks.

For every ``tests/test_*.py`` the catalogue records the subject area,
what is checked, the independent reference it is checked against, the
tolerance, and the number of collected tests (parametrised cases
included, taken from ``pytest --collect-only``).

The descriptive columns come from :data:`FILE_META` below -- a curated
table, because "what reference does this test use" is not something a
script can read off reliably -- while the counts are always measured.
The test ``tests/test_catalog_current.py`` regenerates the page and
fails when it differs from the committed one, or when a test file has
no entry here, so the catalogue cannot silently go stale.

Run::

    python tools/gen_test_catalog.py          # rewrite the .rst
    python tools/gen_test_catalog.py --check  # exit 1 if it is stale
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "verification" / "catalog.rst"

# Subject areas, in the order they appear in the catalogue.
SUBJECTS = [
    "Symbolic engine",
    "Multiplicities and canonical forms",
    "Propagator numerics",
    "Integrators",
    "Workflow and YAML",
    "Drawing and LaTeX",
    "Spectral propagators",
    "Self-consistency",
]

# file -> (subject, what is checked, reference, tolerance)
FILE_META: dict[str, tuple[str, str, str, str]] = {
    "tests/test_fields.py": (
        "Symbolic engine",
        "Field / FieldOperator construction, unique UIDs, component indices",
        "specification (exact structural assertions)", "exact"),
    "tests/test_propagators.py": (
        "Symbolic engine",
        "pairing rules: φφ→C, φψ→R, ψψ→0; Itô R(x,x)=0",
        "MSR contraction rules", "exact"),
    "tests/test_wick.py": (
        "Symbolic engine",
        "operator-level and spatial-level Wick contraction of small products",
        "hand-enumerated pairings; the two engines against each other",
        "exact"),
    "tests/test_expressions.py": (
        "Symbolic engine",
        "expression tree: hashing, equality, rational arithmetic, LaTeX",
        "specification", "exact"),
    "tests/test_simplify.py": (
        "Symbolic engine",
        "simplification passes, diagonal / isotropic index collapse, "
        "collection by diagram",
        "hand-simplified forms; idempotence", "exact"),
    "tests/test_perturbation.py": (
        "Symbolic engine",
        "compute_moment orders, DiagramTerm structure, coupling evaluation, "
        "response phase",
        "hand-derived expansions", "exact"),
    "tests/test_deductive_expansion.py": (
        "Multiplicities and canonical forms",
        "every pairing at order ≤ 2 classified by vanishing reason; "
        "topology multiplicities; canonical-form deduplication",
        "brute-force Wick reference (tests/brute_wick.py, no sft-wick imports)",
        "exact"),
    "tests/test_diagrams.py": (
        "Multiplicities and canonical forms",
        "FeynmanDiagram canonical form and isomorphism deduplication",
        "isomorphic relabellings hash-compare equal", "exact"),
    "tests/test_eval_symbolic_batched.py": (
        "Symbolic engine",
        "batched symbolic evaluator of coupling sums",
        "scalar evaluator on the same expressions", "exact / 1e-12"),
    "tests/test_latex_local_coupling.py": (
        "Drawing and LaTeX",
        "local couplings render without a spurious spacetime argument",
        "regression (referee-reported rendering)", "exact"),
    "tests/test_deductive_numerics.py": (
        "Propagator numerics",
        "C from ∫∫RκR: closed form vs dblquad, spline tables, spatial "
        "homogeneity modes, white noise, QMC vs nquad, parallel vs serial",
        "closed-form OU C; domain-split dblquad at 1e-12; alternative "
        "backends", "1e-8 (quadrature) / 1e-2 (spline, QMC)"),
    "tests/test_c_propagator_gauss_legendre.py": (
        "Propagator numerics",
        "Gauss-Legendre C quadrature with the diagonal split, "
        "translation and rotation kernels, table builds, speed",
        "dblquad", "1e-4"),
    "tests/test_closed_form_dispatch_boundaries.py": (
        "Propagator numerics",
        "built-in closed form vs Gauss-Legendre vs dblquad at the "
        "dispatch boundary cells (ridge, t→t_min, t_max, γ=1/σ_t, r=0, r_max, "
        "per-component γ, t_min≠0, white noise); dispatcher choices",
        "each method against the others; analytic stationary limit",
        "1e-10 (GL) / 1e-8 (dblquad)"),
    "tests/test_propagator_dispatch.py": (
        "Propagator numerics",
        "separable-kernel shared temporal table, time-symmetric build, "
        "auto node-count selection, progress reporting",
        "per-r full build; dblquad", "1e-12 / 1e-6"),
    "tests/test_evaluate_interpolation_accuracy.py": (
        "Propagator numerics",
        "linear vs cubic C-table interpolation on steep tails",
        "closed form", "recorded bounds"),
    "tests/test_dt_discretization.py": (
        "Propagator numerics",
        "the propagators.dt knob converges the spline table",
        "finer grid", "convergence order"),
    "tests/test_diagonal_A_time_dependent.py": (
        "Propagator numerics",
        "time-dependent γ(t) via cumulative-Γ spline",
        "constant-γ closed form; explicit R", "1e-6"),
    "tests/test_d_dim_spatial.py": (
        "Propagator numerics",
        "vector positions through the L1 evaluate path",
        "scalar-separation equivalent", "1e-10"),
    "tests/test_diag_C_offdiagonal.py": (
        "Propagator numerics",
        "off-diagonal C entries with diag_C=false",
        "closed-form cross-correlation", "1e-8"),
    "tests/test_spectral.py": (
        "Spectral propagators",
        "disorder-averaged R*, C* from a spectral density; averaging; "
        "delta-density reduction to OU",
        "OU closed form at a delta density; Marchenko-Pastur convergence",
        "1e-12 (delta) / recorded"),
    "tests/test_gauss_legendre_integrator.py": (
        "Integrators",
        "tensor-product Gauss-Legendre time integration on the causal simplex",
        "hand-derived quadrature; QMC", "1e-5"),
    "tests/test_evaluate_pipeline.py": (
        "Integrators",
        "spatial analysis, causal orderings, integrand assembly",
        "specification", "exact"),
    "tests/test_msr_numerics_regressions.py": (
        "Integrators",
        "causal lower bounds from external response legs, two-time "
        "observables, C-table diagonal ridge, reality projection, "
        "external_times through every backend",
        "closed forms; all five backends against each other", "1e-6 - 1e-10"),
    "tests/test_dynamic_coupling.py": (
        "Integrators",
        "spacetime-dependent (callable) κ^(m) couplings, per-sample and "
        "vectorised contracts, propagator-indexed contraction",
        "static tensor at the same point; two contracts against each other",
        "1e-12"),
    "tests/test_equal_time_nonlocal.py": (
        "Integrators",
        "equal_time non-local vertices (single time integral)",
        "explicit δ-function reduction", "1e-8"),
    "tests/test_R_contracted_vertex.py": (
        "Integrators",
        "already_R_contracted non-local vertices across L0/L1/L2",
        "raw-κ³ evaluation of the same diagram", "1e-12"),
    "tests/test_demo2_kernels.py": (
        "Integrators",
        "demo2's hand-written R-contracted κ³ / κ⁴ kernels, the raw-vs-"
        "R-contracted route on a NON-constant kernel, the "
        "already_R_contracted contract, pinned FK and order-0 values, "
        "the single-site cumulant ladder",
        "cusp-aware adaptive quadrature and randomised-Sobol QMC of the "
        "raw leg integrals; the cumulant generating function",
        "1e-6 - 2e-2 (measured per configuration)"),
    "tests/test_coincident_external_labels.py": (
        "Integrators",
        "external operators sharing a spatial label are refused at L1 "
        "and L0 rather than silently mis-counted",
        "the distinct-label spelling of the same observable", "exact"),
    "tests/test_higher_cumulants.py": (
        "Integrators",
        "non-Gaussian driving at cumulant order m ≥ 4 (κ⁴, κ⁵)",
        "brute-force Wick counting; MSR prefactors", "exact / 1e-10"),
    "tests/test_matrix_r_evaluation.py": (
        "Integrators",
        "matrix-valued R with callable couplings in the scalar loop",
        "qmc_vectorized where legal; closed-form 1-D integrals", "1e-6"),
    "tests/test_demo3_shot_noise.py": (
        "Propagator numerics",
        "demo 3 filtered-Poisson cumulants, the R-contracted kernel K_R, and "
        "the t_tilde branch dispatch",
        "Campbell's theorem vs Monte Carlo of the event process; m-dimensional "
        "quadrature of the raw leg integral; the package's own ClosedFormC; "
        "60-digit mpmath at the branch boundary",
        "exact / 1e-10"),
    "tests/test_demo3_levels.py": (
        "Workflow and YAML",
        "demo 3 level A: the free-field m-point function through the package, "
        "and the R-contracted vertex against the raw one",
        "closed form (level A is a single diagram, hence exact); QMC on the "
        "raw-vertex path",
        "exact / 2e-4"),
    "tests/test_workflow.py": (
        "Workflow and YAML",
        "System / Expansion / Propagators / SweepResult surface; end-to-end "
        "vs the raw API; two-time sweeps",
        "raw L0 pipeline (validate_phase5); closed forms", "1e-6"),
    "tests/test_workflow_config.py": (
        "Workflow and YAML",
        "YAML → System lowering, run_workflow, overrides, dt, parallel "
        "layers, closed-form-only path, explicit R, callable modules",
        "L1 flow on the same physics", "1e-6 / 2e-2 (spline vs exact C)"),
    "tests/test_selfconsistency.py": (
        "Self-consistency",
        "solve_self_consistency: convergence, divergence, oscillation, "
        "max_iter reporting, mixing, state containers",
        "analytic fixed points; contrived failure modes", "tol as configured"),
    "tests/test_drawing_style.py": (
        "Drawing and LaTeX",
        "matplotlib renderer, style abstractions, labels, layout",
        "specification", "exact"),
    "tests/test_drawing_tikz.py": (
        "Drawing and LaTeX",
        "TikZ backend output structure and styles",
        "specification", "exact"),
    "tests/test_catalog_current.py": (
        "Workflow and YAML",
        "this catalogue matches the collected suite",
        "tools/gen_test_catalog.py", "exact"),
}


def collect_counts() -> Counter:
    """``{test file: number of collected tests}`` from pytest."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", "tests/"],
        cwd=ROOT, capture_output=True, text=True,
    )
    counts: Counter = Counter()
    for line in proc.stdout.splitlines():
        m = re.match(r"^(tests/test_[^:]+\.py)::", line)
        if m:
            counts[m.group(1)] += 1
    if not counts:
        raise RuntimeError(f"pytest collected nothing:\n{proc.stdout}\n{proc.stderr}")
    return counts


def test_files() -> list[str]:
    return sorted(
        str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_*.py")
    )


def render(counts: Counter) -> str:
    files = test_files()
    missing = [f for f in files if f not in FILE_META]
    if missing:
        raise SystemExit(
            "tools/gen_test_catalog.py: add FILE_META entries for "
            + ", ".join(missing)
        )
    by_subject: "OrderedDict[str, list]" = OrderedDict((s, []) for s in SUBJECTS)
    for f in files:
        subject, what, ref, tol = FILE_META[f]
        by_subject[subject].append((f, what, ref, tol, counts.get(f, 0)))

    total = sum(counts.get(f, 0) for f in files)
    lines = [
        ".. This file is GENERATED by tools/gen_test_catalog.py -- do not edit.",
        "",
        "Validation catalogue",
        "====================",
        "",
        f"The suite has **{total} tests** in {len(files)} files (parametrised",
        "cases counted individually).  Each row names what is checked, the",
        "independent reference it is checked against, and the tolerance.",
        "Regenerate with ``python tools/gen_test_catalog.py`` (also run by",
        "``make -C docs html``); ``tests/test_catalog_current.py`` fails when",
        "this page is out of date.",
        "",
        ".. caution::",
        "",
        "   The tolerances below are the ones the tests are *written* to",
        "   assert.  For assertions using ``pytest.approx(x, rel=...)``",
        "   without an explicit ``abs=``, the comparison is against",
        "   ``max(rel * expected, 1e-12)`` — so wherever the compared",
        "   quantity is small, the 1e-12 default floor is what is actually",
        "   enforced, and the enforced tolerance is looser than the one",
        "   quoted here.  61 such sites remain; see",
        "   `issue #5 <https://github.com/StatFieldTheory/sft-wick/issues/5>`_",
        "   for the audit and the per-file counts.  The agreements these",
        "   rows claim have been measured directly; what is at issue is",
        "   whether every run re-checks them at the stated tolerance.",
        "",
    ]
    for subject, rows in by_subject.items():
        if not rows:
            continue
        n_sub = sum(r[4] for r in rows)
        lines += [
            subject,
            "-" * len(subject),
            "",
            f"*{n_sub} tests in {len(rows)} files.*",
            "",
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 22 34 26 10 8",
            "",
            "   * - File",
            "     - What is checked",
            "     - Reference",
            "     - Tolerance",
            "     - Tests",
        ]
        for f, what, ref, tol, n in rows:
            lines += [
                f"   * - ``{f.removeprefix('tests/')}``",
                f"     - {what}",
                f"     - {ref}",
                f"     - {tol}",
                f"     - {n}",
            ]
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed catalogue is stale")
    args = ap.parse_args(argv)
    text = render(collect_counts())
    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current != text:
            print(f"{OUT} is stale; run python tools/gen_test_catalog.py")
            return 1
        print(f"{OUT} is current")
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
