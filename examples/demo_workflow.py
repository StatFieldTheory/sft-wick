"""Minimal demo of the high-level :mod:`sft_wick.workflow` API.

This is **the single-file introduction** to using sft-wick.  Everything
up to the first plot command is the user-facing API — no monkey
patching, no subclassing, no direct calls into
:class:`PropagatorCache` / :func:`compute_moment` / :func:`integrate_moment`.

Compare to:

- ``examples/demo1/analysis_combined.ipynb``: ~300 lines of user code
  using the raw API plus a custom ``AnalyticalCache`` subclass and
  a custom ``integrate_gl`` integrator.
- ``examples/demo1/validate_phase5.py``: ~200 lines using the raw
  Phase-5 API.
- ``examples/demo1/validate_wrapper.py``: ~100 lines demonstrating
  numerical equivalence between wrapper and raw API.
- **This file (``demo_workflow.py``)**: <50 physics lines, no raw
  API, produces the same numbers.

Run::

    python examples/demo_workflow.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Make package importable from the source tree.
_here = Path(__file__).resolve().parent
_src = _here.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import sft_wick as sw


# =====================================================================
# Step 1 — Describe the physical system (≈ 15 lines).
# =====================================================================

# Cubic F^(3) interaction tensor: F_{a,b,c} ψ_a φ_b φ_c.
# (demo1's setup — asymmetric, irreducible under index permutation.)
F = np.zeros((2, 2, 2))
F[0, 1, 1] = 1.0
F[1, 0, 1] = 0.5
F[1, 1, 0] = 0.5

system = sw.System(
    # Two-component physical field φ_a, a ∈ {0, 1}.  The response
    # field ψ_a is introduced automatically — the user never needs
    # to reference it.
    field=sw.FieldSpec("phi", n_components=2),

    # Linear operator A_{ab} = −γ_a δ_{ab}: each component decays
    # independently at rate γ_a.  Sets R(t, t') = Θ(t−t') exp(−γ Δt).
    linear=sw.DiagonalA(gamma=[1.0, 1.0]),

    # Local cubic vertex.  Pass the **bare** F as it appears in the
    # equation of motion — the wrapper handles the MSR ``-i`` factor
    # automatically (see :attr:`LocalVertex.msr_coupling`).
    vertices=[sw.LocalVertex("F", coupling=F)],

    # Gaussian driving with separable OU two-point cumulant:
    # κ² = λ exp(−|Δt|/σ_t) · exp(−|Δx|/σ_x) · I_N.
    noise=sw.GaussianNoise(
        kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
            spatial=sw.ExponentialSpatial(sigma_x=1.0),
        ),
        # If the user wanted an additional white-noise contribution:
        # sigma2=sw.ConstantImpulse(amplitude=0.1),
    ),
)


# =====================================================================
# Step 2 — Closed-form C (optional fast path).
#
# The demo's OU κ² admits an analytic C = C_t(t1,t2) · exp(−|Δx|/σ_x).
# Passing this via ``c_closed_form=`` skips the dblquad spline build.
# For a user whose κ² lacks a closed-form C, simply omit the kwarg;
# the wrapper will fall back to dblquad (slower but general).
# =====================================================================


def C_closed_form(n1, t1, n2, t2):
    """C for the separable OU kernel."""
    lam, sigma_t, sigma_x, gamma, N = 0.05, 0.3, 1.0, 1.0, 2
    a = 1.0 / sigma_t
    tl, th = (t1, t2) if t1 <= t2 else (t2, t1)
    if tl <= 0:
        return np.zeros((N, N))
    gpa, gma = gamma + a, gamma - a
    E1 = np.expm1(2 * gamma * tl) / (2 * gamma)
    E2 = tl if abs(gma) < 1e-14 else np.expm1(gma * tl) / gma
    E3 = np.expm1(gpa * tl) / gpa
    E4 = np.exp(gma * th)
    I = E1 / gpa - E2 / gpa + E4 * E3 / gma - E1 / gma
    C_t = lam * np.exp(-gamma * (t1 + t2)) * I
    r = abs(float(np.asarray(n1).sum()) - float(np.asarray(n2).sum()))
    return C_t * np.exp(-r / sigma_x) * np.eye(N)


# =====================================================================
# Step 3 — The three-line workflow.
# =====================================================================

print("=== 1) Perturbative expansion ===")
t0 = time.perf_counter()
expansion = system.expand(
    ("phi_a(x)", "phi_b(y)"),
    orders=[0, 2, 4],
)
print(f"    {time.perf_counter() - t0:.1f}s")
for o, info in expansion.summary().items():
    print(f"    order {o}: {info['n_diagrams']} diagrams "
          f"(by vertex type: {info['by_vertex_type']}, "
          f"by n_cross_C: {info['by_n_cross_C']})")

print("\n=== 2) Build propagator cache ===")
t0 = time.perf_counter()
propagators = system.propagators(
    t_max=15.0, n_grid_t=60,
    c_closed_form=C_closed_form,
)
print(f"    {time.perf_counter() - t0:.2f}s "
      f"(homogeneity={propagators.homogeneity}, "
      f"lazy={propagators.is_lazy})")

print("\n=== 3) Sweep ξ_{ab}(r, t_f) across a 4 × 2 × 2 grid ===")
t0 = time.perf_counter()
sweep = expansion.sweep(
    propagators,
    positions_grid={"x": [0.0], "y": [0.0, 0.5, 1.0, 2.5]},
    t_final_grid=[1.0, 15.0],
    component_pairs=[(0, 0), (1, 1)],
    orders=[0, 2, 4],
    n_samples=2 ** 13,
    seed=42,
)
print(f"    {time.perf_counter() - t0:.1f}s — "
      f"{len(sweep.rows)} diagram-level rows produced")


# =====================================================================
# Step 4 — Structured results (pandas-native).
# =====================================================================

totals = sweep.totals()
print("\n=== Tidy totals (per positions, t_final, (a,b), order) ===")
print(totals.head(10).to_string(index=False))

print("\n=== Convergence check: per-order magnitudes at r=0.5, t_f=15, (a,b)=(1,1) ===")
mask = (
    (abs(totals["y"] - 0.5) < 1e-12)
    & (abs(totals["t_final"] - 15.0) < 1e-12)
    & (totals["a"] == 1) & (totals["b"] == 1)
)
conv = totals.loc[mask, ["order", "value"]].set_index("order")
for order, val in conv["value"].items():
    print(f"    order {order}: {val: .6e}")

cumulative = conv["value"].cumsum()
print(f"    cumulative sum   : {cumulative.iloc[-1]: .6e}")


# =====================================================================
# Step 5 — Channel decomposition (useful for FF/FK-style analyses).
# =====================================================================

print("\n=== Channel totals (per vertex type) ===")
vt = sweep.by_vertex_type_totals()
print(vt.head(10).to_string(index=False))


# =====================================================================
# (Optional) Step 6 — Plot ξ(r) at each t_final.
# =====================================================================

try:
    import matplotlib.pyplot as plt  # noqa: F401
    fig = sweep.plot(
        x="y", y="value",
        hue="order", facet_col="t_final",
    )
    out = _here / "demo_workflow_plot.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n[plot saved to {out.name}]")
except Exception as e:
    print(f"\n[skipping plot: {e}]")
