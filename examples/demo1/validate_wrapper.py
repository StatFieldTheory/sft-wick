"""Numerical regression test for the high-level workflow API.

Runs demo 1's 48-point sweep through the :class:`System` /
:class:`Expansion` / :class:`Propagators` / :class:`SweepResult`
interface in <15 lines of physics-level code, and compares the
``(a, b) = (0, 0)`` rows with a stored table.

The table was first copied from ``validate_phase5.py``, which then
reported time-integrated moments.  That script now reports fixed-time
correlators, the L1 default, so the two no longer compute the same
quantity; the equivalence of the L1 and raw-API routes is tested by
``tests/test_workflow.py::test_WF4_end_to_end_matches_validate_phase5``.
The table below is this script's own output (time-integrated,
``qmc_vectorized``, 2^13 samples, seed 42).  Refreshed twice: on
2026-09-11, when the order-2 and order-4 rows no longer matched the copied
values (7.6e-4 to 5.4e-3 relative, from the sampling fixes of 0.3.0-0.4.x;
order 0 matched to 2.4e-7), and on 2026-09-12, when 0.6.0 gave the Sobol
samplers 64-bit points (a different scrambled sequence, so every QMC draw
is redrawn) and split the Gauss-Legendre domain at the cusp of the OU
kernel.  The second refresh moved 16 of the 24 rows, by 5.6e-5 to 1.2e-1:
order 4 at 2^13 samples has a seed-to-seed spread of tens of percent here,
which is why demo 1's paper sweep uses ``gauss_legendre``
(``examples/demo1/L2/INTEGRATION_ERROR.md``).

On 2026-10-13 the 0.6.1 code (commit
``5198fb9845334982d103059ac7b84f7afb2228ca``) reproduced 23/24 stored-value
failures. Its full unrounded 48 totals and 1136 diagram rows were exactly
equal to the 0.7.0 code under the unchanged settings. The 24 values
below were refreshed from that independently loaded main baseline.
They are a numerical regression snapshot for these QMC and spline
settings, not exact continuum values or a comparison with Phase-5's
fixed-time observable. Independent time-integrated OU order-0 checks
measure finite discretization errors separately.

Run::

    python examples/demo1/validate_wrapper.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

_here = __import__("pathlib").Path(__file__).resolve().parent
_src = _here.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import sft_wick as sw  # noqa: E402


# =========================================================================
# 1. Define the system.  This is the whole "spec", ~10 lines of physics.
# =========================================================================

F = np.zeros((2, 2, 2))
F[0, 1, 1] = 1.0
F[1, 0, 1] = 0.5
F[1, 1, 0] = 0.5

system = sw.System(
    field=sw.FieldSpec("phi", n_components=2),
    linear=sw.DiagonalA(gamma=[1.0, 1.0]),
    # Bare F (as in the equation of motion).  The wrapper applies
    # the MSR ``-i`` prefactor automatically via
    # :attr:`LocalVertex.msr_coupling`.
    vertices=[sw.LocalVertex("F", coupling=F)],
    noise=sw.GaussianNoise(
        kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
            spatial=sw.ExponentialSpatial(sigma_x=1.0),
        ),
    ),
)


# =========================================================================
# 2. Closed-form C via the ``c_closed_form`` kwarg: lazy spline builds
#    drop from minutes (dblquad) to sub-second.
# =========================================================================

sys.path.insert(0, str(_here))
from validate_phase5 import C_t_closed_form  # noqa: E402


def _C_demo1(n1, t1, n2, t2):
    """Closed form for demo1's separable OU κ²."""
    sigma_x = 1.0
    N = 2
    r = abs(float(np.asarray(n1).sum()) - float(np.asarray(n2).sum()))
    return C_t_closed_form(t1, t2) * np.exp(-r / sigma_x) * np.eye(N)


# =========================================================================
# 3. Expansion + sweep, one line each.
# =========================================================================

print("=" * 90)
print("Workflow-API regression: reproduce the stored numerical snapshot")
print("=" * 90)

t0 = time.perf_counter()
expansion = system.expand(
    ("phi_a(x)", "phi_b(y)"),
    orders=[0, 2, 4],
)
print(f"\n[1/3] system.expand(...) : {time.perf_counter() - t0:.1f}s")
for o, info in expansion.summary().items():
    print(f"      order {o}: {info['n_diagrams']} diagrams, "
          f"vertex_types={info['by_vertex_type']}")

t0 = time.perf_counter()
propagators = system.propagators(
    t_max=15.0, n_grid_t=60,
    c_closed_form=_C_demo1,    # bypass dblquad → sub-second builds
)
print(f"\n[2/3] system.propagators(fast)  : "
      f"{time.perf_counter() - t0:.2f}s  "
      f"(homogeneity={propagators.homogeneity}, "
      f"lazy={propagators.is_lazy})")

t0 = time.perf_counter()
sweep = expansion.sweep(
    propagators,
    positions_grid={"x": [0.0], "y": [0.0, 0.5, 1.0, 2.5]},
    t_final_grid=[1.0, 15.0],
    component_pairs=[(0, 0), (1, 1)],
    orders=[0, 2, 4],
    # This script's reference values were produced by Phase-5's raw
    # ``integrate_moment`` which time-integrates all externals; use
    # the matching convention via ``integrate_over='all'`` so the
    # numerical cross-check is meaningful.
    integrate_over="all",
    n_samples=2 ** 13,
    seed=42,
)
print(f"\n[3/3] expansion.sweep(...)      : "
      f"{time.perf_counter() - t0:.1f}s  "
      f"({len(sweep.rows)} diagram-level rows)")


# =========================================================================
# 4. Compare the identical-settings stored numerical snapshot.
# =========================================================================

print("\n" + "-" * 90)
print(f"{'(a,b)':<7} {'r':<5} {'t_f':<6} {'order':<6} "
      f"{'new wrapper':<16} {'match?'}")
print("-" * 90)

totals = sweep.totals()  # DataFrame with summed-per-diagram values
mismatches = 0
for _, row in totals.iterrows():
    a = int(row["a"]); b = int(row["b"])
    r = float(row["y"]); t_f = float(row["t_final"])
    order = int(row["order"])
    val = float(row["value"])
    print(f"({a},{b})   {r:<5.2f} {t_f:<6.2f} {order:<6d} "
          f"{val: .6e}  -")

print("-" * 90)

# Expected totals for a=0, b=0 at each (r, t_f, order): unrounded output
# from main 5198fb9845334982d103059ac7b84f7afb2228ca at the settings above.
# This is a numerical regression snapshot, not an analytic reference.
EXPECTED = {
    (0, 0, 0.0, 1.0, 0): 0.0032715061398110685,
    (0, 0, 0.0, 15.0, 0): 0.39935689358025317,
    (0, 0, 0.5, 1.0, 0): 0.001984268777233538,
    (0, 0, 0.5, 15.0, 0): 0.2422222001240189,
    (0, 0, 1.0, 1.0, 0): 0.0012035198505026385,
    (0, 0, 1.0, 15.0, 0): 0.1469151908382667,
    (0, 0, 2.5, 1.0, 0): 0.00026854157698446803,
    (0, 0, 2.5, 15.0, 0): 0.03278121005997958,
    (0, 0, 0.0, 1.0, 2): 7.018868500417241e-06,
    (0, 0, 0.0, 15.0, 2): 0.038002513820243805,
    (0, 0, 0.5, 1.0, 2): 4.289909252294353e-06,
    (0, 0, 0.5, 15.0, 2): 0.031158074918936098,
    (0, 0, 1.0, 1.0, 2): 2.9554029427699385e-06,
    (0, 0, 1.0, 15.0, 2): 0.027743315669267694,
    (0, 0, 2.5, 1.0, 2): 1.6759013108806938e-06,
    (0, 0, 2.5, 15.0, 2): 0.024385857336698315,
    (0, 0, 0.0, 1.0, 4): 3.44894092697764e-08,
    (0, 0, 0.0, 15.0, 4): 0.003274225009355242,
    (0, 0, 0.5, 1.0, 4): 1.8927091915741514e-08,
    (0, 0, 0.5, 15.0, 4): 0.0022965027215034396,
    (0, 0, 1.0, 1.0, 4): 1.1990636603338727e-08,
    (0, 0, 1.0, 15.0, 4): 0.0018544592357331903,
    (0, 0, 2.5, 1.0, 4): 5.9073418134316464e-09,
    (0, 0, 2.5, 15.0, 4): 0.0014634651017417953,
}

print("\nCross-check vs stored numerical snapshot (a=0, b=0):")
max_rel = 0.0
for (a, b, r, t_f, ord_), expected in EXPECTED.items():
    mask = (
        (totals["a"] == a)
        & (totals["b"] == b)
        & (abs(totals["y"] - r) < 1e-12)
        & (abs(totals["t_final"] - t_f) < 1e-12)
        & (totals["order"] == ord_)
    )
    got = float(totals.loc[mask, "value"].iloc[0])
    rel = abs(got - expected) / (abs(expected) + 1e-30)
    max_rel = max(max_rel, rel)
    mark = "OK " if rel < 1e-4 else "!! "
    print(f"  (a={a},b={b},r={r},t_f={t_f},ord={ord_}): "
          f"{got: .6e} vs {expected: .6e}  rel={rel:.2e}  {mark}")
    if rel >= 1e-4:
        mismatches += 1

print(f"\n{'-' * 90}")
print(f"max rel_err = {max_rel:.2e}, "
      f"mismatches = {mismatches} / {len(EXPECTED)}")
if mismatches == 0:
    print("PASS: the sweep reproduces the stored numerical snapshot.")
    sys.exit(0)
else:
    print("FAIL: wrapper diverged from the stored numerical snapshot.")
    sys.exit(1)
