"""Acceptance test for the high-level workflow API.

Reproduce ``validate_phase5.py``'s 48-point comparison but through
the new :class:`System` / :class:`Expansion` / :class:`Propagators` /
:class:`SweepResult` interface — the goal is that <15 lines of
physics-level code replace the earlier ~60-line raw-API flow and
produce **bit-identical** numbers.

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
# 1. Define the system — this is the whole "spec", ~10 lines of physics.
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
# 2. Closed-form C via ``c_closed_form`` kwarg — lazy spline builds
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
# 3. Expansion + sweep — one-liner each.
# =========================================================================

print("=" * 90)
print("Workflow-API acceptance: reproduce validate_phase5.py output")
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
# 4. Cross-check against validate_phase5.py's Method-A output.
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

# Expected totals — extracted from validate_phase5.py's printed output
# for a=0,b=0 at each (r, t_f, order).  Bit-match is the target.
EXPECTED = {
    (0, 0, 0.0, 1.0, 0):  3.274077e-03,
    (0, 0, 0.0, 15.0, 0): 3.996863e-01,
    (0, 0, 0.5, 1.0, 0):  1.985828e-03,
    (0, 0, 0.5, 15.0, 0): 2.424220e-01,
    (0, 0, 1.0, 1.0, 0):  1.204466e-03,
    (0, 0, 1.0, 15.0, 0): 1.470364e-01,
    (0, 0, 2.5, 1.0, 0):  2.687526e-04,
    (0, 0, 2.5, 15.0, 0): 3.280825e-02,
    (0, 0, 0.0, 1.0, 2):  7.033309e-06,
    (0, 0, 0.0, 15.0, 2): 3.982281e-02,
    (0, 0, 0.5, 1.0, 2):  4.292876e-06,
    (0, 0, 0.5, 15.0, 2): 3.212865e-02,
    (0, 0, 1.0, 1.0, 2):  2.952800e-06,
    (0, 0, 1.0, 15.0, 2): 2.824665e-02,
    (0, 0, 2.5, 1.0, 2):  1.668009e-06,
    (0, 0, 2.5, 15.0, 2): 2.437748e-02,
    (0, 0, 0.0, 1.0, 4):  3.358017e-08,
    (0, 0, 0.0, 15.0, 4): 2.930656e-03,
    (0, 0, 0.5, 1.0, 4):  1.843353e-08,
    (0, 0, 0.5, 15.0, 4): 2.231587e-03,
    (0, 0, 1.0, 1.0, 4):  1.167776e-08,
    (0, 0, 1.0, 15.0, 4): 1.899270e-03,
    (0, 0, 2.5, 1.0, 4):  5.744135e-09,
    (0, 0, 2.5, 15.0, 4): 1.584070e-03,
}

print("\nCross-check vs validate_phase5.py (a=0, b=0) rows:")
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
    print("PASS — wrapper reproduces validate_phase5.py to numerical precision.")
    sys.exit(0)
else:
    print("FAIL — wrapper diverged from validate_phase5.py.")
    sys.exit(1)
