"""FK channel via the new dynamic-coupling path — the Plan-A
follow-up to ``validate_wrapper_demo2.py``.

Proves the workflow can compute the FK channel of demo2 natively:
no hand-coded ``_fk_spatial_integral``, no bespoke 4-D
Gauss-Legendre — user writes a single ``K_fn(n_list, t_list)``
callable that computes the third cumulant at the three ψ-leg
spacetime points, passes it as ``sw.NonLocalVertex("K", 3,
coupling=K_fn)``, and the workflow picks up the per-sample
evaluation automatically.

Demo2's third cumulant (from the quadratic-deformation non-Gaussian
noise ``η̃ = η + α(η² − λ)``):

    κ^{(3)}_{abc}(1,2,3) = δ_{ab} δ_{bc}
        · { 2α λ² [κ(1,3) κ(2,3) + κ(1,2) κ(2,3) + κ(1,2) κ(1,3)]
            + 8α³λ³ κ(1,2) κ(2,3) κ(1,3) }          (α³ term added 2026-09)

    κ(i, j) = exp(−|t_i − t_j| / σ_t) · exp(−|x_i − x_j| / σ_x)

Run::

    python examples/demo2/validate_FK_dynamic.py

Cross-check target (from demo2/analysis.ipynb cell 12, r=0.5, t_f=3,
without the α³ term and with the un-converged 4-D rule of the time):
    ξ^{FK}_{01} = +1.884322e-04
The converged value from the R-contracted kernel (``k3_R_coupling.py``)
at r = 0.5, t = 3.48 is 1.816e-04.
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


# =====================================================================
# Physical constants (match demo2)
# =====================================================================

LAM = 0.05
SIGMA_T = 0.3
SIGMA_X = 1.0
GAMMA = 1.0
ALPHA = 0.6
N_COMP = 2

# Cubic F tensor (bare — wrapper applies MSR -i).
F = np.zeros((N_COMP, N_COMP, N_COMP))
F[0, 1, 1] = 1.0
F[1, 0, 1] = 0.5
F[1, 1, 0] = 0.5


# =====================================================================
# κ^{(3)} as a spacetime-dependent callable
# =====================================================================

def K_fn(n_list, t_list):
    """Third cumulant at 3 spacetime points.

    Args:
        n_list: length-3 array of ψ-leg spatial coordinates.
        t_list: length-3 array of ψ-leg times.

    Returns:
        ``(N, N, N)`` array — the tensor
        ``κ^{(3)}_{abc}(1,2,3) = 2αλ² δ_{ab}δ_{bc} · […]``.

    The three ``κ(i, j) κ(j, k)`` terms in the bracket are the
    Wick pairings of ``⟨η² · η · η⟩ − ⟨η²⟩⟨η⟩⟨η⟩`` for Gaussian η.
    """
    n = np.asarray(n_list, dtype=float)
    t = np.asarray(t_list, dtype=float)

    def kappa(i, j):
        return (
            np.exp(-abs(t[i] - t[j]) / SIGMA_T)
            * np.exp(-abs(n[i] - n[j]) / SIGMA_X)
        )

    bracket = kappa(0, 2) * kappa(1, 2) \
        + kappa(0, 1) * kappa(1, 2) \
        + kappa(0, 1) * kappa(0, 2)

    amplitude = (2.0 * ALPHA * LAM ** 2 * bracket
                 + 8.0 * ALPHA ** 3 * LAM ** 3 * kappa(0, 1) * kappa(1, 2) * kappa(0, 2))

    # Component structure: δ_{ab}δ_{bc} → non-zero only on the
    # diagonal (a=b=c).  Build explicitly.
    K = np.zeros((N_COMP, N_COMP, N_COMP), dtype=float)
    for a in range(N_COMP):
        K[a, a, a] = amplitude
    return K


# =====================================================================
# Closed-form C for FF (same as before)
# =====================================================================

sys.path.insert(0, str(_here))
from validate_wrapper_demo2 import _C_demo2_eff  # noqa: E402


# =====================================================================
# Build the system — K is a callable now
# =====================================================================

LAM_EFF = LAM * (1.0 + 2.0 * ALPHA ** 2 * LAM)

system = sw.System(
    field=sw.FieldSpec("phi", n_components=N_COMP),
    linear=sw.DiagonalA(gamma=[GAMMA] * N_COMP),
    vertices=[sw.LocalVertex("F", coupling=F)],
    # K as a callable — wrapper multiplies by the MSR
    # -(i^3)/3! = +i/6 factor internally.
    nonlocal_vertices=[
        sw.NonLocalVertex("K", order=3, coupling=K_fn),
    ],
    noise=sw.GaussianNoise(
        kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=LAM_EFF, sigma_t=SIGMA_T),
            spatial=sw.ExponentialSpatial(sigma_x=SIGMA_X),
        ),
    ),
)

print("=" * 90)
print("Demo2 FK via dynamic coupling (Plan A)")
print("=" * 90)

expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2])
groups = expansion.by_vertex_type(2)
print(f"order 2: {len(groups.get('F', []))} F + "
      f"{len(groups.get('FK', []))} FK + "
      f"{len(groups.get('K', []))} K")

props = system.propagators(
    t_max=15.0, n_grid_t=60,
    c_closed_form=_C_demo2_eff,
)


# =====================================================================
# Evaluate FK at (a=0, b=1, r=0.5, t_f=3.0)
# =====================================================================

print()
print(f"Evaluating FK at r=0.5, t_f=3.0 via "
      f"expansion.evaluate(vertex_types={{'FK'}}) …")
t0 = time.perf_counter()
res_01 = expansion.evaluate(
    props,
    positions={"x": 0.0, "y": 0.5},
    t_final=3.0,
    component_pair=(0, 1),
    orders=[2],
    vertex_types={"FK"},
    n_samples=2 ** 13, seed=42,
)
elapsed = time.perf_counter() - t0
print(f"  FK (0,1, r=0.5, t=3):  got {res_01.total: .6e} in {elapsed:.1f}s")
print(f"  demo2 reference      : +1.884322e-04")

ref = 1.884322e-04
rel = abs(res_01.total - ref) / abs(ref)
print(f"  relative error       : {rel:.2e}")


# Selection rule: FK=0 for (0,0) and (1,1)
res_00 = expansion.evaluate(
    props, positions={"x": 0.0, "y": 0.5},
    t_final=3.0, component_pair=(0, 0),
    orders=[2], vertex_types={"FK"},
    n_samples=2 ** 13, seed=42,
)
res_11 = expansion.evaluate(
    props, positions={"x": 0.0, "y": 0.5},
    t_final=3.0, component_pair=(1, 1),
    orders=[2], vertex_types={"FK"},
    n_samples=2 ** 13, seed=42,
)
print(f"  FK (0,0) [should be 0]: {res_00.total: .6e}")
print(f"  FK (1,1) [should be 0]: {res_11.total: .6e}")

# Verdict
if rel < 5e-2 and abs(res_00.total) < 1e-8 and abs(res_11.total) < 1e-8:
    print("\nPASS — FK channel reproduced via the workflow's dynamic")
    print("coupling path.  No bespoke integrator needed.")
    sys.exit(0)
else:
    print("\nFAIL")
    sys.exit(1)
