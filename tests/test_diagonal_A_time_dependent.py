"""Tests for the time-dependent extension of
:class:`sft_wick.DiagonalA` — ``γ(t) -> array(N)``.
"""

from __future__ import annotations

import numpy as np

import sft_wick as sw


# =====================================================================
# T1 — static list vs constant callable are numerically equivalent
# =====================================================================


def test_T1_callable_constant_matches_static() -> None:
    """A callable ``γ(t) = [g_0, g_1]`` (constant) builds the same R
    as passing ``[g_0, g_1]`` directly."""
    static = sw.DiagonalA(gamma=[1.0, 0.5])
    callable_A = sw.DiagonalA(
        gamma=lambda t: np.array([1.0, 0.5]),
        t_max_cache=5.0, n_grid_cache=200,
    )
    R_s = static.build_R_callable()
    R_c = callable_A.build_R_callable()

    # Static non-iso → returns (2, 2) diagonal matrix
    # Callable non-iso → same
    for t1, t2 in [(1.0, 0.5), (3.0, 1.0), (2.0, 0.0), (0.5, 1.0)]:
        v_s = R_s(t1, t2)
        v_c = R_c(t1, t2)
        if isinstance(v_s, np.ndarray):
            np.testing.assert_allclose(v_s, v_c, atol=1e-8)
        else:
            assert abs(v_s - v_c) < 1e-8


def test_T1b_callable_iso_constant_matches_static_iso() -> None:
    """Iso constant γ (all components equal) both ways → scalar R."""
    static = sw.DiagonalA(gamma=[1.0, 1.0])
    callable_A = sw.DiagonalA(
        gamma=lambda t: np.array([1.0, 1.0]),
        t_max_cache=5.0, n_grid_cache=200,
    )
    R_s = static.build_R_callable()
    R_c = callable_A.build_R_callable()

    assert static.is_iso_R is True
    assert callable_A.is_iso_R is True

    for t1, t2 in [(1.0, 0.5), (3.0, 1.0)]:
        np.testing.assert_allclose(R_s(t1, t2), R_c(t1, t2), atol=1e-8)


# =====================================================================
# T2 — linear γ(t) closed form
# =====================================================================


def test_T2_linear_gamma_matches_closed_form() -> None:
    """For ``γ(t) = g_0 + g_1 · t`` the primitive is
    ``Γ(t) = g_0 t + g_1 t²/2`` and
    ``R(t_1, t_2) = exp(-(Γ(t_1) - Γ(t_2)))``.
    """
    g0, g1 = 1.0, 0.3

    def gamma_fn(t: float) -> np.ndarray:
        return np.array([g0 + g1 * t])

    A = sw.DiagonalA(gamma=gamma_fn, t_max_cache=5.0, n_grid_cache=400)
    R = A.build_R_callable()

    for t1, t2 in [(1.0, 0.5), (3.0, 1.0), (4.0, 0.2)]:
        Gamma_t1 = g0 * t1 + g1 * t1 ** 2 / 2
        Gamma_t2 = g0 * t2 + g1 * t2 ** 2 / 2
        expected = float(np.exp(-(Gamma_t1 - Gamma_t2)))
        got = R(t1, t2)
        rel = abs(got - expected) / expected
        # 400-pt trapezoidal + cubic spline on a linear γ should hit
        # ~1e-5 easily (trapezoid is exact for linear, spline exact
        # for cubic Γ).
        assert rel < 1e-4, (
            f"(t_1, t_2) = ({t1}, {t2}): got {got:.6e} vs "
            f"closed form {expected:.6e}, rel={rel:.2e}"
        )


# =====================================================================
# T3 — causality: R(t_1, t_2) = 0 when t_1 < t_2
# =====================================================================


def test_T3_causality() -> None:
    A = sw.DiagonalA(
        gamma=lambda t: np.array([1.0 + 0.1 * np.sin(t)]),
        t_max_cache=5.0, n_grid_cache=200,
    )
    R = A.build_R_callable()
    assert R(0.5, 1.0) == 0.0, "R must vanish for t_1 < t_2 (Heaviside)"
    assert R(2.0, 2.0) >= 0.0  # equal-time: 1.0 for iso path


# =====================================================================
# T4 — end-to-end workflow with time-dependent γ (smoke)
# =====================================================================


def test_T4_time_dependent_gamma_end_to_end(tmp_path) -> None:
    """Quick end-to-end: a time-dependent γ gives a System that can
    be expanded and integrated without error; order-0 value at
    equal time reduces to C(t_f, t_f; r) from the bare static γ
    when γ is actually constant."""
    N = 1
    # Use a constant-via-callable γ — order-0 result should match the
    # static analogue.
    system_callable = sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(
            gamma=lambda t: np.array([1.0]),
            t_max_cache=2.0, n_grid_cache=40,
        ),
        vertices=[],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
            spatial=sw.ExponentialSpatial(sigma_x=1.0),
        )),
    )
    system_static = sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(gamma=[1.0]),
        vertices=[],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
            spatial=sw.ExponentialSpatial(sigma_x=1.0),
        )),
    )

    exp_c = system_callable.expand(("phi(x)", "phi(y)"), orders=[0])
    exp_s = system_static.expand(("phi(x)", "phi(y)"), orders=[0])
    props_c = system_callable.propagators(t_max=1.5, n_grid_t=10)
    props_s = system_static.propagators(t_max=1.5, n_grid_t=10)

    res_c = exp_c.evaluate(
        props_c, positions={"x": 0.0, "y": 0.5},
        t_final=1.0, component_pair=(0, 0), orders=[0],
        n_samples=2 ** 10, seed=0,
    )
    res_s = exp_s.evaluate(
        props_s, positions={"x": 0.0, "y": 0.5},
        t_final=1.0, component_pair=(0, 0), orders=[0],
        n_samples=2 ** 10, seed=0,
    )

    # Constant-via-callable should match static to dblquad + spline
    # precision.  γ(t)=1 is linear-constant so trapezoidal on Γ is
    # exact, and cubic spline on a linear Γ is exact — the R agrees
    # to machine precision, and hence C matches via the normal
    # dblquad build.
    rel = abs(res_c.total - res_s.total) / abs(res_s.total)
    assert rel < 1e-3, (
        f"callable-constant γ vs static γ: "
        f"{res_c.total:.6e} vs {res_s.total:.6e} (rel {rel:.2e})"
    )
