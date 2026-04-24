"""Verify that the wrapper's "derive R and C from A and κ²" path
produces the same physics as "user provides the closed-form C".

The demo1 system has diagonal A = −γ·I (so R is the trivial
scalar exponential ``R(t, t') = Θ(t−t') exp(−γ Δt)``) and a separable
translation-invariant κ², so the wrapper can build C entirely by
itself — one ``dblquad`` per (t₁, t₂, r) grid point — without the
user supplying any closed form.

This script runs both:

- **Method A**: ``system.propagators(..., c_closed_form=C_analytic)``
  — the fast path used in ``demo_workflow.py`` / ``validate_wrapper.py``.
- **Method B**: ``system.propagators(...)`` with **no** ``c_closed_form``
  — the wrapper falls back to ``dblquad`` for every C evaluation.

Under the same physical spec they should produce the same moment
values up to spline-interpolation noise (~1e-4 relative) and
``dblquad`` precision (~1e-8).  If this check fails, there is a
bug in the ``R + κ² → C`` derivation path.

Run::

    python examples/demo1/validate_R_C_derivation.py

Takes ~1 min on M-series (dominated by the serial lazy dblquad
builds on Method B).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
_src = _here.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import sft_wick as sw  # noqa: E402


# --- Shared system spec ------------------------------------------------ #

F = np.zeros((2, 2, 2))
F[0, 1, 1] = 1.0
F[1, 0, 1] = 0.5
F[1, 1, 0] = 0.5


def _make_system() -> sw.System:
    return sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            ),
        ),
    )


# --- Analytic C for Method A ------------------------------------------ #

def _C_analytic(n1, t1, n2, t2):
    """Closed form for separable OU κ²: `C_t · exp(-|Δx|/σ_x)`."""
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
# Build the two propagator caches
# =====================================================================

print("=" * 88)
print("R + κ² → C derivation check: closed-form C vs dblquad-from-scratch")
print("=" * 88)

# Small grid to keep Method B (dblquad per grid point) tractable.
# With lazy mode each unique r (encountered during the sweep) triggers
# a (n_grid_t × n_grid_t × n_components) = 25×25×2 = 1250 dblquad calls.
N_GRID_T = 25
T_MAX = 15.0

system_A = _make_system()
system_B = _make_system()
expansion = system_A.expand(
    ("phi_a(x)", "phi_b(y)"), orders=[0, 2, 4],
)

print("\n[A] Building propagators with c_closed_form (closed-form C)…")
t0 = time.perf_counter()
props_A = system_A.propagators(
    t_max=T_MAX, n_grid_t=N_GRID_T,
    c_closed_form=_C_analytic,  # fast: bypasses dblquad
)
print(f"    built in {time.perf_counter() - t0:.2f}s")

print("\n[B] Building propagators with NO c_closed_form (dblquad + κ²)…")
t0 = time.perf_counter()
props_B = system_B.propagators(
    t_max=T_MAX, n_grid_t=N_GRID_T,
    # No c_closed_form → wrapper calls scipy.integrate.dblquad on
    # self.model.kappa2 for each (t1, t2) grid point.
    # n_jobs=-1 fans the grid points out across all CPU cores via
    # joblib (each new r value in lazy mode triggers one parallel
    # build).
    n_jobs=-1,
)
print(f"    constructor: {time.perf_counter() - t0:.2f}s "
      f"(lazy + n_jobs=-1; the parallel dblquad builds happen on "
      f"first evaluation of each new r)")


# =====================================================================
# Evaluate at a few cross-check points
# =====================================================================

print("\nComparing per-order moment values at representative points…")
print()
print(f"{'(a,b)':<7} {'r':<5} {'t_f':<6} {'order':<6} "
      f"{'A (closed-form)':<17} {'B (dblquad)':<17} "
      f"{'rel_err':<10} {'pass'}")
print("-" * 88)

TEST_POINTS = [
    # (a, b, r, t_f)
    (0, 0, 0.5, 15.0),
    (0, 0, 1.0, 15.0),
    (1, 1, 0.5, 15.0),
    (1, 1, 1.0, 15.0),
]

max_rel = 0.0
n_fail = 0

total_B_time = 0.0

for (a, b, r, t_f) in TEST_POINTS:
    for order in [0, 2, 4]:
        # Method A
        res_A = expansion.evaluate(
            props_A,
            positions={"x": 0.0, "y": r},
            t_final=t_f,
            component_pair=(a, b),
            orders=[order],
            n_samples=2 ** 13, seed=42,
        )
        val_A = res_A.total

        # Method B — first time a new r is queried, this triggers
        # a lazy dblquad spline build.
        t0 = time.perf_counter()
        res_B = expansion.evaluate(
            props_B,
            positions={"x": 0.0, "y": r},
            t_final=t_f,
            component_pair=(a, b),
            orders=[order],
            n_samples=2 ** 13, seed=42,
        )
        total_B_time += time.perf_counter() - t0
        val_B = res_B.total

        denom = max(abs(val_A), 1e-30)
        rel = abs(val_A - val_B) / denom
        max_rel = max(max_rel, rel)
        # 3e-4: combined 2-D time-spline error on an n_grid_t=25 grid
        # (~1e-4) plus dblquad precision (~1e-8).  Real-world
        # expectation: max ~1e-4.
        ok = rel < 3e-4
        if not ok:
            n_fail += 1
        mark = "OK " if ok else "!! "
        print(f"({a},{b})   {r:<5.2f} {t_f:<6.2f} {order:<6d} "
              f"{val_A: .6e}   {val_B: .6e}   "
              f"{rel:.2e}   {mark}")

print("-" * 88)
print(
    f"Method B dblquad-backed evaluations: total {total_B_time:.1f}s "
    f"(≈ lazy builds on the fly)."
)
print(
    f"max rel_err = {max_rel:.2e}  —  "
    f"{'PASS' if n_fail == 0 else 'FAIL'}  "
    f"({len(TEST_POINTS) * 3 - n_fail}/{len(TEST_POINTS) * 3} checks)"
)

if n_fail > 0:
    print("\nDivergence detected — the 'R + κ² → C' derivation path is "
          "producing different numbers from the closed-form-C path.  "
          "Likely bugs: kappa2 callable signature, dblquad bounds, or "
          "spline-table sampling.")
    sys.exit(1)

print("\nThe wrapper's R + κ² → C path is consistent with the closed-form")
print("C path.  Users who don't know a closed form for C can rely on the")
print("pure physical inputs (A, κ²) and get the same answer.")
