"""Tests for the Gauss-Legendre tensor-product integrator
(``method='gauss_legendre'``).

The motivating use-case is demo2's FK channel: a 4D smooth
integrand on a causal simplex.  At large external time t_f, the
integrand peak is ``σ_t / γ ≈ 0.3`` sq units in a simplex of area
``t_f^2 / 2``; Sobol QMC severely under-resolves the peak unless
``n_samples`` is enormous, while a tensor-product Gauss-Legendre
rule with even modest ``n_gauss`` (4-8 nodes per dimension) gives
exponential convergence on the smooth integrand.

These tests lock the contract by comparing the L2 sweep result to
the notebook's hand-derived ``_fk_spatial_integral`` (verbatim from
``examples/demo2/analysis.ipynb``) at matched ``n_gauss``.  The two
should agree to floating-point because they evaluate the same
4D integral with the same nodes -- only the orchestration
differs (the L2 path enumerates diagrams symbolically).
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

# ---------- Notebook reference (verbatim from analysis.ipynb) ----------

_LAM = 0.05
_SIGMA_T = 0.3
_SIGMA_X = 1.0
_GAMMA = 1.0
_ALPHA = 0.6


def _gl_nodes_unit(n):
    x, w = leggauss(n)
    return (x + 1) / 2, w / 2


def _T(dt):
    return np.exp(-np.abs(dt) / _SIGMA_T)


def _fk_spatial_integral_reference(r, t_f, n_gauss):
    """Verbatim copy of analysis.ipynb's ``_fk_spatial_integral``."""
    u, w = _gl_nodes_unit(n_gauss)

    tau_0 = u * t_f
    jac_0 = t_f * w
    tau_1 = u * t_f
    jac_1 = t_f * w
    tau_2 = u[:, None] * tau_0[None, :]
    tau_3 = u[:, None] * tau_0[None, :]
    jac_2 = tau_0[None, :] * w[:, None]
    jac_3 = tau_0[None, :] * w[:, None]

    R_ext_0 = np.exp(-_GAMMA * (t_f - tau_0))
    R_ext_1 = np.exp(-_GAMMA * (t_f - tau_1))
    R_02 = np.exp(-_GAMMA * (tau_0[None, :] - tau_2))
    R_03 = np.exp(-_GAMMA * (tau_0[None, :] - tau_3))

    T12 = _T(tau_1[:, None, None] - tau_2[None, :, :])
    T13 = _T(tau_1[:, None, None] - tau_3[None, :, :])
    T23 = _T(tau_2[:, None, :] - tau_3[None, :, :])

    er1 = np.exp(-r / _SIGMA_X)
    er2 = np.exp(-2 * r / _SIGMA_X)

    K1 = er1 * T13[:, None, :, :] * T23[None, :, :, :]
    K2 = er1 * T12[:, :, None, :] * T23[None, :, :, :]
    K3 = er2 * T12[:, :, None, :] * T13[:, None, :, :]
    K_total = 2.0 * _ALPHA * _LAM**2 * (K1 + K2 + K3) / 6.0

    R_part = (R_ext_0[None, None, None, :]
              * R_ext_1[:, None, None, None]
              * R_02[None, :, None, :]
              * R_03[None, None, :, :])

    W = (jac_0[None, None, None, :]
         * jac_1[:, None, None, None]
         * jac_2[None, :, None, :]
         * jac_3[None, None, :, :])

    return 6.0 * float(np.sum(R_part * K_total * W))


def xi_FK_01_reference(r, t_f, n_gauss):
    """Notebook xi_{01}^{FK}.  The F tensor for demo2 makes
    F[0, 1, 1] + F[1, 0, 0] = 1 + 0 = 1, so ``xi_FK_01 = base``."""
    return 1.0 * _fk_spatial_integral_reference(r, t_f, n_gauss)


# ---------- L2 path under test ----------


def _build_demo2_FK_system():
    """Build the demo2 FK system (1 F vertex + 1 K vertex, alpha=0.6,
    lam=0.05) at the L1 layer.  Mirrors ``config_FK.yaml``."""
    import sft_wick.workflow.specs as sp
    from sft_wick.workflow import System

    F = np.array(
        [[[0.0, 0.0], [0.0, 1.0]],
         [[0.0, 0.5], [0.5, 0.0]]]
    )

    def _kappa3(n_2d, t_2d):
        n = np.asarray(n_2d, dtype=float)
        t = np.asarray(t_2d, dtype=float)

        def kappa(i, j):
            return (
                np.exp(-np.abs(t[i] - t[j]) / _SIGMA_T)
                * np.exp(-np.abs(n[i] - n[j]) / _SIGMA_X)
            )

        bracket = (
            kappa(0, 2) * kappa(1, 2)
            + kappa(0, 1) * kappa(1, 2)
            + kappa(0, 1) * kappa(0, 2)
        )
        amplitude = 2.0 * _ALPHA * _LAM**2 * bracket
        K = np.zeros((amplitude.shape[0], 2, 2, 2), dtype=float)
        for a in range(2):
            K[:, a, a, a] = amplitude
        return K

    return System(
        field=sp.FieldSpec(name="phi", n_components=2),
        linear=sp.DiagonalA(gamma=[1.0, 1.0]),
        noise=sp.GaussianNoise(
            kappa2=sp.SeparableTranslation(
                temporal=sp.ExponentialTemporal(lam=_LAM, sigma_t=_SIGMA_T),
                spatial=sp.ExponentialSpatial(sigma_x=_SIGMA_X),
            ),
        ),
        vertices=(sp.LocalVertex(name="F", coupling=F),),
        nonlocal_vertices=(
            sp.NonLocalVertex(
                name="K", order=3,
                coupling=_kappa3,
                coupling_vectorized=True,
            ),
        ),
    )


def _C_fn_bare(n1, t1, n2, t2):
    """Closed-form C with bare lam (FK channel uses this)."""
    t1 = np.atleast_1d(np.asarray(t1, dtype=float))
    t2 = np.atleast_1d(np.asarray(t2, dtype=float))
    n1_a = np.broadcast_to(np.asarray(n1, dtype=float), t1.shape)
    n2_a = np.broadcast_to(np.asarray(n2, dtype=float), t1.shape)
    g = _GAMMA
    a = 1.0 / _SIGMA_T
    t_lo = np.minimum(t1, t2)
    t_hi = np.maximum(t1, t2)
    gpa = g + a
    gma = g - a
    pos = t_lo > 0
    safe_lo = np.where(pos, t_lo, 1.0)
    E1 = np.expm1(2 * g * safe_lo) / (2 * g)
    E2 = np.expm1(gma * safe_lo) / gma if abs(gma) > 1e-14 else safe_lo
    E3 = np.expm1(gpa * safe_lo) / gpa
    E4 = np.exp(gma * t_hi)
    I = E1 / gpa - E2 / gpa + E4 * E3 / gma - E1 / gma
    val = _LAM * np.exp(-g * (t1 + t2)) * I
    diag = np.where(pos, val, 0.0) * np.exp(-np.abs(n1_a - n2_a) / _SIGMA_X)
    return diag[:, None, None] * np.eye(2)[None, :, :]


@pytest.mark.parametrize("t_f", [1.0, 5.0, 15.0, 50.0])
@pytest.mark.parametrize("n_gauss", [6, 8])
def test_FK_gauss_legendre_matches_notebook(t_f, n_gauss):
    """L2 sweep with method='gauss_legendre' at fixed n_gauss must
    match the notebook's hand-derived 4D GL quadrature at the same
    n_gauss to floating-point precision (rtol=1e-9 is conservative;
    a properly-implemented GL on the same nodes evaluating the same
    integrand should agree to ~1e-13).
    """
    system = _build_demo2_FK_system()

    expansion = system.expand(
        observable=("phi_a(x)", "phi_b(y)"),
        orders=(2,),
    )
    propagators = system.propagators(
        t_max=t_f,
        n_grid_t=10,  # ignored under c_closed_form_only
        c_closed_form=_C_fn_bare,
        c_closed_form_only=True,
        c_closed_form_vectorized=True,
    )

    # Cross pair only -- diagonal pairs are zero by selection rule.
    result = expansion.evaluate(
        propagators,
        positions={"x": 0.0, "y": 0.0},
        t_final=t_f,
        component_pair=(0, 1),
        orders=(2,),
        vertex_types={"FK"},
        method="gauss_legendre",
        n_gauss=n_gauss,
    )

    reference = xi_FK_01_reference(r=0.0, t_f=t_f, n_gauss=n_gauss)
    actual = result.total

    np.testing.assert_allclose(
        actual, reference, rtol=1e-9, atol=1e-15,
        err_msg=f"L2 GL vs notebook GL mismatch at t_f={t_f}, "
                f"n_gauss={n_gauss}: actual={actual!r}, ref={reference!r}",
    )


def test_nquad_with_dynamic_coupling_raises():
    """Regression: ``method='nquad'`` previously silently returned 0
    on diagrams with a callable (spacetime-dependent) coupling
    because ``integrate_moment_nquad`` multiplied by the placeholder
    ``coupling_array=zeros`` instead of materialising the dynamic
    coupling per-call.

    Rather than fix the silent-zero bug by routing per-call through
    the dynamic-coupling promise (which works but is dramatically
    slower than tensor-product GL on the same 4D smooth integrand),
    we surface the limitation explicitly and direct users to
    ``method='gauss_legendre'``.  This test locks the contract.
    """
    system = _build_demo2_FK_system()
    expansion = system.expand(
        observable=("phi_a(x)", "phi_b(y)"),
        orders=(2,),
    )
    propagators = system.propagators(
        t_max=1.0,
        n_grid_t=10,
        c_closed_form=_C_fn_bare,
        c_closed_form_only=True,
        c_closed_form_vectorized=True,
    )

    with pytest.raises(NotImplementedError, match="gauss_legendre"):
        expansion.evaluate(
            propagators,
            positions={"x": 0.0, "y": 0.0},
            t_final=1.0,
            component_pair=(0, 1),
            orders=(2,),
            vertex_types={"FK"},
            method="nquad",
        )


def test_gauss_legendre_n_gauss_convergence():
    """Verify GL is asymptotically stable as n_gauss grows.

    At t_f=15, r=0, the integrand is sharply peaked in a band of
    width ~σ_t/γ ≈ 0.3 inside a 4-simplex of side 15.  Low-n_gauss
    rules can miss or alias the peak (e.g. n=4 underestimates by
    50%); convergence is non-monotone but stabilises by n≈12.
    """
    refs = {n: xi_FK_01_reference(0.0, 15.0, n) for n in (8, 12, 16, 20)}
    # Successive differences shrink (asymptotic stabilisation).
    diffs = [
        abs(refs[12] - refs[8]),
        abs(refs[16] - refs[12]),
        abs(refs[20] - refs[16]),
    ]
    assert diffs[0] > diffs[1] > diffs[2], (
        f"expected asymptotic stabilisation; got refs={refs}, "
        f"diffs={diffs}"
    )
    # Final value within 25% of the simulation truth (~4.11e-4).
    sim_truth = 4.11e-4
    assert 0.75 * sim_truth < refs[20] < 1.25 * sim_truth, (
        f"GL n=20 value {refs[20]:.4e} should bracket sim {sim_truth:.4e}"
    )
