"""Phase-5 acceptance: reimplement demo1's ⟨φ_a(0) φ_b(r)⟩(t_f) using
only the new spatial-coordinate API, cross-check against the κ²-ratio
reference (what demo1's ``integrate_gl`` does internally).

Two methods compared at identical (a, b, r, t_f, order) points:

- **Method A (new Phase-5 API)**:
  ``PropagatorCache(model, homogeneity='translation')`` + lazy
  ``precompute_C_table_translation`` + ``integrate_moment(ig, positions=
  {'x': 0, 'y': r}, method='qmc_vectorized')``.  Each C propagator's
  spatial factor enters naturally via a per-r 2-D time-spline built
  on-demand.

- **Method B (demo1's κ²-ratio reference)**:
  Legacy cache (``precompute_C_table`` only, no x-awareness) →
  integrand evaluated at r = 0 → multiply the diagram's contribution by
  ``exp(-r/σ_x) ** n_cross_C`` where ``n_cross_C`` is the number of
  cross-group C propagators in that diagram.

Under translation invariance (the assumption both rely on), these are
mathematically equivalent; bit-identical Sobol paths (same seed, same
vectorised integrator) → any mismatch comes from how the spatial
factor is produced.  Expected max rel-err ≲ 1e-4 (2-D time-spline
precision).

Run::

    python examples/demo1/validate_phase5.py

Exit code 0 iff all 48 rows pass < 5e-3 relative.
"""

from __future__ import annotations

import sys
import time

import numpy as np

# Make package importable when run from the source tree.
_here = __import__("pathlib").Path(__file__).resolve().parent
_src = _here.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from sft_wick import (  # noqa: E402
    Action, Field, Vertex, compute_moment, reset_uid_counter,
)
from sft_wick.evaluate import (  # noqa: E402
    PropagatorCache, PropagatorModel, integrate_moment,
)


# =========================================================================
# Physical parameters — match examples/demo1/analysis_combined.ipynb
# =========================================================================

LAM = 0.05
SIGMA_T = 0.3
SIGMA_X = 1.0
GAMMA = 1.0
N_COMP = 2

# Grid for cross-check — corresponds to Figure-2 reduced set.
R_GRID = [0.0, 0.5, 1.0, 2.5]
T_GRID = [1.0, 15.0]
AB_GRID = [(0, 0), (1, 1)]
ORDER_GRID = [0, 2, 4]

# QMC config.  Identical seed between methods → identical Sobol samples
# → only the C-factor math differs.
N_SAMPLES = 2**13
SEED = 42

# Tolerance for the per-row assertion.  Expected error is bounded by
# 2-D time-spline precision on the n_grid_t=60 grid over [0, T_MAX]
# (O(h^4) ~ 1e-4) plus any QMC inconsistencies from integrand ordering.
REL_TOL = 5e-3


# =========================================================================
# Closed-form C_t(t1, t2) for the separable OU kernel
# =========================================================================

def C_t_closed_form(t1: float, t2: float) -> float:
    """Analytic ``C(t1, t2; 0, 0)`` for OU κ² — the same formula
    demo1's ``AnalyticalCache._C_scalar`` uses (notebook cell 2)."""
    g, a = GAMMA, 1.0 / SIGMA_T
    t_lo = min(t1, t2)
    t_hi = max(t1, t2)
    if t_lo <= 0:
        return 0.0
    gpa, gma = g + a, g - a
    E1 = np.expm1(2 * g * t_lo) / (2 * g)
    E2 = t_lo if abs(gma) < 1e-14 else np.expm1(gma * t_lo) / gma
    E3 = np.expm1(gpa * t_lo) / gpa
    E4 = np.exp(gma * t_hi)
    I = E1 / gpa - E2 / gpa + E4 * E3 / gma - E1 / gma
    return LAM * np.exp(-g * (t1 + t2)) * I


# =========================================================================
# Model + closed-form-backed fast cache (same pattern as Phase-5 tests)
# =========================================================================

def _make_model() -> PropagatorModel:
    def R_time(t1, t2):
        return float(np.exp(-GAMMA * (t1 - t2))) if t1 >= t2 else 0.0

    def kappa2(n1, t1, n2, t2):
        r = abs(
            float(np.asarray(n1).sum()) - float(np.asarray(n2).sum())
        )
        kt = LAM * np.exp(-abs(t1 - t2) / SIGMA_T)
        kx = np.exp(-r / SIGMA_X)
        return kt * kx * np.eye(N_COMP)

    return PropagatorModel(
        R_time=R_time, kappa2=kappa2, n_components=N_COMP,
        iso_R=True, diag_C=True, t_min=0.0,
    )


class FastCache(PropagatorCache):
    """``PropagatorCache`` subclass whose ``_C_value_direct`` uses the
    closed-form OU kernel.  Avoids a ~30-minute ``dblquad`` build of
    the spline table — same trick as
    :class:`tests.test_deductive_numerics.TestSpatialAwareCache`'s
    ``_FastCache``.
    """

    def _C_value_direct(self, n1, t1, n2, t2):
        r = abs(
            float(np.asarray(n1).sum()) - float(np.asarray(n2).sum())
        )
        C_t = C_t_closed_form(t1, t2)
        C_x = np.exp(-r / SIGMA_X)
        C_mat = np.zeros((N_COMP, N_COMP))
        for a in range(N_COMP):
            C_mat[a, a] = C_t * C_x
        return C_mat


# =========================================================================
# Diagram bookkeeping helper
# =========================================================================

def count_cross_group_c(dt) -> int:
    """Number of C propagators whose endpoints land in different
    direction groups (via R-connectivity, from
    :meth:`DiagramTerm.analyze_spatial`).  Same helper used by
    Phase-5's S5 deductive test."""
    spatial = dt.analyze_spatial()
    n_cross = 0
    for p in dt.propagators:
        if p.kind != "C":
            continue
        d_l = spatial.direction_map[p.spatial_left]
        d_r = spatial.direction_map[p.spatial_right]
        if d_l != d_r:
            n_cross += 1
    return n_cross


# =========================================================================
# Main validation
# =========================================================================

def main() -> int:
    print("=" * 92)
    print("Phase-5 acceptance: reimplement demo1's ξ(r, t) via new API")
    print("=" * 92)

    # 1. compute_moment(order=4) once — same flags as demo1 notebook.
    print("\n[1/4] Generating diagram terms via compute_moment(order=4)…")
    t0 = time.perf_counter()
    reset_uid_counter()
    phi = Field("phi", "physical", n_components=N_COMP)
    psi = Field("psi", "response", n_components=N_COMP)
    F = np.zeros((N_COMP, N_COMP, N_COMP))
    F[0, 1, 1] = 1.0          # φ₁² drives φ₀
    F[1, 0, 1] = 0.5          # φ₀ φ₁ drives φ₁ (symmetrised)
    F[1, 1, 0] = 0.5
    F_MSR = -1j * F
    action = Action(vertices=[Vertex([psi, phi, phi], coupling="F")])
    obs = [phi("a", "x"), phi("b", "y")]
    result = compute_moment(
        obs, action, order=4,
        ito=True, response_phase=True, collect_topology=True,
        diag_R=True, diag_C=True, iso_R=True,
    )
    print(f"      compute_moment(order=4): {time.perf_counter() - t0:.1f}s")
    for ord_ in ORDER_GRID:
        dts = result.diagram_terms(ord_)
        ncross_hist = {}
        for dt in dts:
            n = count_cross_group_c(dt)
            ncross_hist[n] = ncross_hist.get(n, 0) + 1
        hist = ", ".join(
            f"n_cross={k}:{v}" for k, v in sorted(ncross_hist.items())
        )
        print(f"      order {ord_}: {len(dts)} diagrams ({hist})")

    # 2. Build two caches.
    print("\n[2/4] Building caches…")
    t0 = time.perf_counter()
    t_max = max(T_GRID)

    # Method A: translation-homogeneity cache, lazy mode.  No r-grid
    # precomputed — per-r 2-D spline built on-demand during
    # integrate_moment calls.  Needs t_max ≥ max(T_GRID).
    cache_A = FastCache(model=_make_model())  # default homogeneity='translation'
    cache_A.precompute_C_table_translation(
        t_max=t_max, n_grid_t=60,  # lazy: r_max/n_grid_r left None
    )
    print(f"      Method-A cache (translation, lazy): "
          f"{time.perf_counter() - t0:.2f}s")

    # Method B: legacy time-only cache at reference x=0.  No spatial
    # awareness — integrate_moment(positions=None) ignores x (same as
    # pre-Phase-5 behaviour).  We will manually multiply by
    # exp(-r/σ_x)**n_cross after integration.
    t0 = time.perf_counter()
    cache_B = FastCache(model=_make_model())
    cache_B.precompute_C_table(
        t_max=t_max, n_grid=60, direction=0.0,
    )
    print(f"      Method-B cache (legacy, r=0 reference): "
          f"{time.perf_counter() - t0:.2f}s")

    # 3. Sweep.
    print(f"\n[3/4] Cross-checking {len(AB_GRID)}×{len(R_GRID)}×"
          f"{len(T_GRID)}×{len(ORDER_GRID)} = "
          f"{len(AB_GRID)*len(R_GRID)*len(T_GRID)*len(ORDER_GRID)} "
          f"points…")
    t0 = time.perf_counter()

    rows: list[tuple] = []
    for a, b in AB_GRID:
        fi = {"a": a, "b": b}
        for order in ORDER_GRID:
            dts = result.diagram_terms(order)
            for r in R_GRID:
                for t_f in T_GRID:
                    # --- Method A: new spatial-aware API ---
                    A_val = 0.0
                    for dt in dts:
                        ig = dt.build_integrand(
                            {"F": F_MSR}, fixed_indices=fi,
                        )
                        val, _ = integrate_moment(
                            ig, lambda_f=t_f, cache=cache_A,
                            method="qmc_vectorized",
                            n_samples=N_SAMPLES, seed=SEED,
                            positions={"x": 0.0, "y": r},
                        )
                        A_val += val

                    # --- Method B: κ²-ratio reference ---
                    B_val = 0.0
                    factor_unit = np.exp(-r / SIGMA_X)
                    for dt in dts:
                        ig = dt.build_integrand(
                            {"F": F_MSR}, fixed_indices=fi,
                        )
                        val0, _ = integrate_moment(
                            ig, lambda_f=t_f, cache=cache_B,
                            method="qmc_vectorized",
                            n_samples=N_SAMPLES, seed=SEED,
                            # no positions — legacy cache ignores them
                        )
                        n_cross = count_cross_group_c(dt)
                        B_val += val0 * (factor_unit ** n_cross)

                    denom = max(abs(B_val), 1e-15)
                    rel = abs(A_val - B_val) / denom
                    passed = rel < REL_TOL
                    rows.append((a, b, r, t_f, order, A_val, B_val,
                                 rel, passed))

    elapsed = time.perf_counter() - t0
    print(f"      done: {elapsed:.1f}s "
          f"({elapsed / len(rows) * 1000:.0f} ms/point)")

    # 4. Report.
    print("\n[4/4] Results")
    print("-" * 92)
    print(f"{'(a,b)':<7} {'r':<5} {'t_f':<6} {'order':<6} "
          f"{'A (new API)':<16} {'B (ref)':<16} "
          f"{'rel_err':<10} {'pass'}")
    print("-" * 92)
    for a, b, r, t_f, ord_, A_val, B_val, rel, ok in rows:
        mark = "OK " if ok else "!! "
        print(f"({a},{b})   {r:<5.2f} {t_f:<6.2f} {ord_:<6d} "
              f"{A_val: .6e}   {B_val: .6e}   "
              f"{rel:.2e}   {mark}")
    print("-" * 92)

    n_pass = sum(1 for *_, ok in rows if ok)
    max_rel = max(rel for *_, rel, _ in rows)
    median_rel = float(np.median([rel for *_, rel, _ in rows]))
    print(f"\n{n_pass}/{len(rows)} passed, max rel_err = {max_rel:.2e}, "
          f"median rel_err = {median_rel:.2e}")

    if n_pass != len(rows):
        print("\nFAILED: some comparisons exceeded the tolerance.")
        return 1

    print("\nAll comparisons passed.  The new Phase-5 spatial-aware API")
    print("reproduces demo1's κ²-ratio reference to within "
          f"{max_rel:.2e} relative error — no manual spatial-factor")
    print("post-processing needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
