"""L1 lowering: structure the default flags assume, now checked.

Three L1 inputs returned wrong numbers without an error (found 2026-09-11):

1. ``DiagonalA(gamma=callable)`` chose between a scalar and a matrix R by
   evaluating ``gamma`` at t = 0 and t = 1 only.  A rate whose components
   agree at those two times was lowered to a scalar R that gives every
   component the rate of component 0.
2. Its cumulative-rate spline started at t = 0 whatever ``System.t_min``
   was, so every R with a time below 0 came from extrapolation.
3. ``System.expand`` and ``System.propagators`` default to ``diag_R=True``
   and ``diag_C=True``, which keep only the diagonal of R and C.  A dense R
   (``ExplicitR(iso_R=False)``), a component-mixing κ² (``GeneralKappa2``)
   or a component-mixing white noise (a ``ConstantImpulse`` or
   ``CustomImpulse`` matrix) has off-diagonal entries, and they were
   dropped.

References: ``scipy.integrate.quad`` of the rate for (1)-(2); for (3) the
Lyapunov equation of the Markov embedding of the exponential noise, and for
the kernels of vector positions ``scipy.integrate.dblquad`` of ``R κ² R``.
Neither shares code with sft-wick.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from scipy.integrate import dblquad, quad, solve_ivp
from scipy.linalg import expm

import sft_wick as sw
from sft_wick.workflow.specs import (ConstantImpulse, CustomImpulse,
                                     GeneralKappa2)

N = 2


# ---------------------------------------------------------------------------
# 1-2. Callable decay rate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GammaAgreesAtZeroAndOne:
    """Components equal at t = 0 and t = 1, different in between."""

    def __call__(self, t):
        return np.array([1.0, 1.0 + 0.8 * np.sin(np.pi * t)])


@dataclass(frozen=True)
class GammaEqual:
    def __call__(self, t):
        v = 1.0 + 0.3 * np.sin(t)
        return np.array([v, v])


@dataclass(frozen=True)
class GammaScalar:
    def __call__(self, t):
        return np.array([1.0 + 0.8 * np.sin(2.0 * t)])


def _exact_R(gamma, a, t1, t2):
    return np.exp(-quad(lambda s: gamma(s)[a], t2, t1,
                        epsabs=1e-13, epsrel=1e-12)[0])


def _noise():
    return sw.GaussianNoise(kappa2=sw.SeparableTranslation(
        temporal=sw.ExponentialTemporal(lam=1.0, sigma_t=0.5),
        spatial=sw.ExponentialSpatial(sigma_x=1.0)))


def test_callable_gamma_iso_detection_sees_the_whole_grid():
    g = GammaAgreesAtZeroAndOne()
    lin = sw.DiagonalA(gamma=g, t_max_cache=5.0, n_grid_cache=2000)
    assert lin.is_iso_R is False
    R = lin.build_R_callable()
    for t1, t2 in [(1.5, 0.5), (2.5, 0.2), (4.0, 1.3)]:
        got = np.diag(np.asarray(R(t1, t2)))
        exact = [_exact_R(g, a, t1, t2) for a in range(N)]
        assert got == pytest.approx(exact, rel=1e-5, abs=0.0)
    system = sw.System(field=sw.FieldSpec("phi", N), linear=lin, noise=_noise())
    assert system.iso_R is False


def test_callable_gamma_with_equal_components_stays_scalar():
    g = GammaEqual()
    lin = sw.DiagonalA(gamma=g, t_max_cache=5.0, n_grid_cache=2000)
    assert lin.is_iso_R is True
    R = lin.build_R_callable()
    assert float(R(2.5, 0.2)) == pytest.approx(_exact_R(g, 0, 2.5, 0.2),
                                               rel=1e-5, abs=0.0)


def test_callable_gamma_spline_reaches_a_negative_t_min():
    g = GammaScalar()
    system = sw.System(
        field=sw.FieldSpec("phi", 1),
        linear=sw.DiagonalA(gamma=g, t_max_cache=5.0, n_grid_cache=4000),
        noise=_noise(), t_min=-3.0)
    R = system.build_propagator_model().R_time
    for t1, t2 in [(1.0, 0.2), (1.0, -0.5), (0.5, -2.0), (-0.5, -3.0)]:
        assert float(R(t1, t2)) == pytest.approx(_exact_R(g, 0, t1, t2),
                                                 rel=1e-5, abs=0.0)


def test_diagonal_a_t_min_cache_sets_the_spline_start():
    g = GammaScalar()
    lin = sw.DiagonalA(gamma=g, t_max_cache=5.0, n_grid_cache=4000,
                       t_min_cache=-3.0)
    R = lin.build_R_callable()
    assert float(R(0.5, -2.0)) == pytest.approx(_exact_R(g, 0, 0.5, -2.0),
                                                rel=1e-5, abs=0.0)


def test_propagators_beyond_t_max_cache_are_refused():
    system = sw.System(
        field=sw.FieldSpec("phi", 1),
        linear=sw.DiagonalA(gamma=GammaScalar(), t_max_cache=2.0),
        noise=_noise())
    with pytest.raises(ValueError, match="t_max_cache"):
        system.propagators(t_max=3.0, progress=False)


# ---------------------------------------------------------------------------
# 3. Off-diagonal R and C under the default flags
# ---------------------------------------------------------------------------

A_DENSE = ((-1.0, 2.5), (0.0, -1.7))          # non-normal drift
LAM, TAU = 0.7, 0.4
T_OBS = 1.5


@dataclass(frozen=True)
class DenseR:
    A: tuple = A_DENSE

    def __call__(self, t, s):
        if t < s:
            return np.zeros((N, N))
        return expm(np.asarray(self.A) * (t - s))


def _lyapunov_blocks():
    """``z = (φ, η)`` with ``dη = −η/τ dt + √(2λ/τ) dW`` and φ(0) = 0."""
    Z, I = np.zeros((N, N)), np.eye(N)
    M = np.block([[np.asarray(A_DENSE), I], [Z, -I / TAU]])
    BB = np.block([[Z, Z], [Z, (2 * LAM / TAU) * I]])
    P0 = np.block([[Z, Z], [Z, LAM * I]])
    return M, BB, P0


def _P(t):
    M, BB, P0 = _lyapunov_blocks()
    sol = solve_ivp(lambda _s, y: (M @ y.reshape(4, 4)
                                   + y.reshape(4, 4) @ M.T + BB).ravel(),
                    (0.0, t), P0.ravel(), rtol=1e-12, atol=1e-14)
    return sol.y[:, -1].reshape(4, 4)


@dataclass(frozen=True)
class DenseC:
    """Exact C of the dense system (Markov embedding), ``(N, N)``."""

    def __call__(self, n1, t1, n2, t2):
        M, _, _ = _lyapunov_blocks()
        if t1 >= t2:
            return (expm(M * (t1 - t2)) @ _P(t2))[:N, :N]
        return (_P(t1) @ expm(M * (t2 - t1)).T)[:N, :N]


def _dense_system(vertices=()):
    return sw.System(
        field=sw.FieldSpec("phi", N),
        linear=sw.ExplicitR(R_time=DenseR(), iso_R=False),
        vertices=list(vertices), noise=_noise_for_dense())


def _noise_for_dense():
    return sw.GaussianNoise(kappa2=sw.SeparableTranslation(
        temporal=sw.ExponentialTemporal(lam=LAM, sigma_t=TAU),
        spatial=sw.ExponentialSpatial(sigma_x=1.0)))


OBS = ("phi_a(x)", "phi_b(y)")


def test_dense_R_refuses_diag_R():
    with pytest.raises(ValueError, match="diag_R=False"):
        _dense_system().expand(OBS, orders=[0])


def test_dense_R_refuses_diag_C_in_expand():
    with pytest.raises(ValueError, match="diag_C=False"):
        _dense_system().expand(OBS, orders=[0], diag_R=False)


def test_dense_R_refuses_diagonal_propagator_tables():
    with pytest.raises(ValueError, match="diag_C=False"):
        _dense_system().propagators(t_max=1.6, n_grid_t=11, progress=False)


def test_dense_R_with_full_flags_matches_the_lyapunov_reference():
    """The route the refusals point to gives the right numbers."""
    FB = np.array([[0.3, -0.6], [0.9, -0.4]])
    system = _dense_system([sw.LocalVertex("F", coupling=FB)])
    exp = system.expand(OBS, orders=[0], diag_R=False, diag_C=False)
    props = system.propagators(t_max=1.6, c_closed_form=DenseC(),
                               c_closed_form_only=True, diag_C=False,
                               progress=False)
    P = _P(T_OBS)
    for ab in [(0, 1), (1, 1)]:
        r = exp.evaluate(props, positions={"x": 0.0, "y": 0.0},
                         t_final=T_OBS, component_pair=ab, orders=[0],
                         method="qmc", n_samples=16)
        assert r.total == pytest.approx(P[ab], rel=1e-10, abs=0.0)


@dataclass(frozen=True)
class MixingKappa2:
    def __call__(self, n1, t1, n2, t2):
        return np.exp(-abs(t1 - t2)) * np.array([[1.0, 0.3], [0.3, 1.0]])


@dataclass(frozen=True)
class DiagonalKappa2:
    def __call__(self, n1, t1, n2, t2):
        return np.exp(-abs(t1 - t2)) * np.diag([1.0, 0.6])


@dataclass(frozen=True)
class AngleKappa2:
    """``e^{-|t1 - t2|} (1 + cos θ) M``, θ the angle between the two
    positions.  It raises at a zero position and is zero at two positions
    of opposite sign, so of the guard's position pairs only the one with
    two positions of the same sign shows ``M``."""

    matrix: tuple

    def __call__(self, n1, t1, n2, t2):
        a = np.atleast_1d(np.asarray(n1, dtype=float))
        b = np.atleast_1d(np.asarray(n2, dtype=float))
        norms = float(np.linalg.norm(a) * np.linalg.norm(b))
        if norms == 0.0:
            raise ValueError("an angle needs two non-zero positions")
        cos = float(a @ b) / norms
        return np.exp(-abs(t1 - t2)) * (1.0 + cos) * np.asarray(self.matrix)


@dataclass(frozen=True)
class MixesApartKappa2:
    """Diagonal at two equal positions, component-mixing between distinct
    ones.  It raises when one position is zero and the other is not, and at
    a negative position, so the guard's position pairs give a diagonal
    matrix, two raises, then the mixing at the pair of two distinct
    positive positions."""

    def __call__(self, n1, t1, n2, t2):
        if (n1 == 0.0) != (n2 == 0.0) or min(n1, n2) < 0.0:
            raise ValueError("one zero position, or a negative one")
        mix = 0.3 if n1 != n2 else 0.0
        return np.exp(-abs(t1 - t2)) * np.array([[1.0, mix], [mix, 0.8]])


@dataclass(frozen=True)
class MixingWhite:
    def __call__(self, n1, t, n2):
        return np.array([[1.0, 0.6], [0.6, 0.5]])


@dataclass(frozen=True)
class DiagonalWhite:
    def __call__(self, n1, t, n2):
        return np.diag([1.0, 0.5])


def _system(kappa2=None, sigma2=None, gamma=(1.0, 1.0)):
    kappa2 = kappa2 if kappa2 is not None else sw.SeparableTranslation(
        temporal=sw.ExponentialTemporal(lam=0.5, sigma_t=0.5),
        spatial=sw.ExponentialSpatial(sigma_x=1.0))
    return sw.System(field=sw.FieldSpec("phi", N),
                     linear=sw.DiagonalA(gamma=list(gamma)),
                     noise=sw.GaussianNoise(kappa2=kappa2, sigma2=sigma2))


MIXING = {
    "GeneralKappa2": dict(kappa2=GeneralKappa2(fn=MixingKappa2())),
    "ConstantImpulse matrix": dict(
        sigma2=ConstantImpulse(np.array([[1.0, 0.6], [0.6, 0.5]]))),
    "CustomImpulse": dict(sigma2=CustomImpulse(fn=MixingWhite())),
    "GeneralKappa2 that raises at a zero position": dict(
        kappa2=GeneralKappa2(fn=AngleKappa2(((1.0, 0.3), (0.3, 0.8))))),
    "GeneralKappa2 that mixes after a diagonal point and a raise": dict(
        kappa2=GeneralKappa2(fn=MixesApartKappa2())),
}

DIAGONAL = {
    "separable": dict(),
    "distinct decay rates": dict(gamma=(1.0, 1.7)),
    "GeneralKappa2 diagonal": dict(kappa2=GeneralKappa2(fn=DiagonalKappa2())),
    "ConstantImpulse scalar": dict(sigma2=ConstantImpulse(0.8)),
    "ConstantImpulse diagonal": dict(sigma2=ConstantImpulse(np.diag([1.0, 0.5]))),
    "CustomImpulse diagonal": dict(sigma2=CustomImpulse(fn=DiagonalWhite())),
    "GeneralKappa2 that raises at a zero position": dict(
        kappa2=GeneralKappa2(fn=AngleKappa2(((1.0, 0.0), (0.0, 0.8))))),
}


@pytest.mark.parametrize("case", sorted(MIXING))
def test_component_mixing_noise_refuses_diag_C(case):
    system = _system(**MIXING[case])
    with pytest.raises(ValueError, match="diag_C=False"):
        system.expand(OBS, orders=[0])
    with pytest.raises(ValueError, match="diag_C=False"):
        system.propagators(t_max=1.0, n_grid_t=5, progress=False)
    system.expand(OBS, orders=[0], diag_C=False)   # the stated way out


@pytest.mark.parametrize("case", sorted(DIAGONAL))
def test_diagonal_structure_is_not_refused(case):
    system = _system(**DIAGONAL[case])
    system.expand(OBS, orders=[0])
    system.propagators(t_max=1.0, n_grid_t=5, progress=False)


def test_a_probe_that_cannot_evaluate_is_not_a_refusal():
    """A kernel that rejects the probe's scalar positions says nothing
    about its structure; the guard stays out of the way."""

    @dataclass(frozen=True)
    class VectorOnly:
        def __call__(self, n1, t1, n2, t2):
            if np.ndim(n1) != 1:
                raise TypeError("vector positions only")
            return np.exp(-abs(t1 - t2)) * np.eye(N)

    _system(kappa2=GeneralKappa2(fn=VectorOnly())).expand(OBS, orders=[0])


X3, Y3 = np.array([0.3, -0.2, 0.9]), np.array([-0.5, 0.4, 0.1])
MIX3 = ((1.0, 0.3), (0.3, 0.8))


def _everywhere(n1, t1, n2, t2):
    return True


def _from_six_tenths(n1, t1, n2, t2):
    return min(t1, t2) >= 0.6


def _between_fifteen_and_twenty_five_hundredths(n1, t1, n2, t2):
    return 0.15 <= min(t1, t2) and max(t1, t2) <= 0.25


def _from_y_to_x(n1, t1, n2, t2):
    return np.array_equal(n1, Y3) and np.array_equal(n2, X3)


def _at_y_with_itself(n1, t1, n2, t2):
    return np.array_equal(n1, Y3) and np.array_equal(n2, Y3)


@dataclass(frozen=True)
class VectorKappa2:
    """``e^{-|t1 - t2|} e^{-|n1 - n2|} M`` for positions in R^3, with the
    off-diagonal entries of ``M`` only where ``mixes(n1, t1, n2, t2)``.  It
    subtracts its positions as arrays and raises for a position of another
    shape, so no probe of ``System`` reads it.  With ``apart_only`` it also
    raises at two equal positions.  The second term is 0.0 and sets NumPy's
    underflow flag at every call."""

    matrix: tuple = MIX3
    mixes: Any = _everywhere
    apart_only: bool = False

    def __call__(self, n1, t1, n2, t2):
        if np.shape(n1) != (3,) or np.shape(n2) != (3,):
            raise TypeError("positions are vectors of three coordinates")
        if self.apart_only and np.array_equal(n1, n2):
            raise ValueError("two equal positions")
        m = np.asarray(self.matrix)
        if not self.mixes(n1, t1, n2, t2):
            m = np.diag(np.diag(m))
        apart = np.linalg.norm(n1 - n2)
        return (np.exp(-abs(t1 - t2) - apart) + np.exp(-800.0 - apart)) * m


@dataclass(frozen=True)
class VectorSigma2:
    """A white noise with the matrix of :class:`VectorKappa2`."""

    kernel: VectorKappa2 = VectorKappa2()

    def __call__(self, n1, t, n2):
        return self.kernel(n1, t, n2, t)


@dataclass(frozen=True)
class AnyClosedForm:
    """Stands in for a closed form of C: the guard does not read it, and a
    call that is not refused returns at once."""

    def __call__(self, n1, t1, n2, t2):
        return 0.1 * np.exp(-abs(t1 - t2)) * np.eye(N)


#: name -> (noise of the system, diag_C in expand, diag_C on the tables[,
#: noise of the system that is expanded, when it is another])
READ_AT_THE_CALL = {
    "flag in expand only": (
        dict(kappa2=GeneralKappa2(fn=VectorKappa2())), True, False),
    "flag on the tables only": (
        dict(kappa2=GeneralKappa2(fn=VectorKappa2())), False, True,
        dict(kappa2=GeneralKappa2(fn=VectorKappa2(((1.0, 0.0), (0.0, 0.8)))))),
    "sigma2": (dict(sigma2=CustomImpulse(fn=VectorSigma2())), True, True),
    "sigma2 beside a diagonal callable kappa2": (
        dict(kappa2=GeneralKappa2(fn=VectorKappa2(((1.0, 0.0), (0.0, 0.8)))),
             sigma2=CustomImpulse(fn=VectorSigma2())), True, True),
    "mixing from 0.6 of the window on": (
        dict(kappa2=GeneralKappa2(fn=VectorKappa2(mixes=_from_six_tenths))),
        True, True),
    "mixing between 0.15 and 0.25 of the window": (
        dict(kappa2=GeneralKappa2(fn=VectorKappa2(
            mixes=_between_fifteen_and_twenty_five_hundredths))), True, True),
    "mixing from y to x only": (
        dict(kappa2=GeneralKappa2(fn=VectorKappa2(mixes=_from_y_to_x))),
        True, True),
    "mixing at y with itself only": (
        dict(kappa2=GeneralKappa2(fn=VectorKappa2(mixes=_at_y_with_itself))),
        True, True),
}


@pytest.mark.parametrize("case", READ_AT_THE_CALL)
def test_diag_C_is_refused_at_the_positions_of_the_call(case):
    """The probes of ``System`` read scalar positions, so a κ² or σ² of
    vectors passes ``expand`` and ``propagators``.  ``evaluate`` and
    ``sweep`` read it at each pair of positions that two of their points
    hold together, at each position with itself, and over their own time
    window, with the flag on either side.  The kernels read are those of
    the tables: one case expands another system than the one they come
    from.

    The positions of ``evaluate`` are lists, which the kernel is handed as
    arrays, and the caller's ``np.errstate(all="raise")`` does not reach the
    reading, in which the kernel underflows and the window starts at 0.
    The latest time read is ``t_final`` with the points left free, and the
    pinned time with both pinned.  Each position axis of the sweep starts
    with a value that is not a real position, a complex number on ``x`` and
    a label on ``y``, which the reading leaves out.  The vectors and the
    latest time come second on their axes, and the refusal shows them as
    lists."""
    noise, on_expansion, on_tables, *own = READ_AT_THE_CALL[case]
    system = _system(**noise, gamma=(1.0, 1.7))
    expanded = _system(**(own[0] if own else noise), gamma=(1.0, 1.7))
    exp = expanded.expand(OBS, orders=[0], diag_C=on_expansion)
    props = system.propagators(t_max=1.0, c_closed_form=AnyClosedForm(),
                               c_closed_form_only=True, diag_C=on_tables,
                               progress=False)
    for times in (dict(t_final=1.0),
                  dict(t_final=0.3, external_times={"x": 1.0, "y": 1.0})):
        with np.errstate(all="raise"), pytest.raises(
                ValueError, match=r"largest entry at the times? .*diag_C=False"):
            exp.evaluate(props,
                         positions={"x": X3.tolist(), "y": Y3.tolist()},
                         **times, component_pair=(0, 1), orders=[0],
                         method="gauss_legendre")
    with pytest.raises(ValueError,
                       match=r"positions \[.*\] and \[.*diag_C=False"):
        exp.sweep(props, positions_grid={"x": [2j, X3], "y": ["a site", Y3]},
                  t_final_grid=[0.3, 1.0], component_pairs=[(0, 1)],
                  orders=[0], method="gauss_legendre", progress=False)


def _mixes_from_one(n1, t1, n2, t2):
    return min(t1, t2) >= 1.0


def _mixes_up_to_four_tenths(n1, t1, n2, t2):
    return min(t1, t2) <= 0.4


#: name -> (kernel of the tables, t_min, t_final[, κ² of the system that is
#: expanded, when it is another]): C is diagonal on the window of the call
NOT_READ_BY_THE_CALL = {
    "diagonal to 5e-13, raising at two equal positions": (
        VectorKappa2(((1.0, 5e-13), (5e-13, 0.8)), apart_only=True), 0.0, 1.0),
    "mixing from the latest time on": (
        VectorKappa2(mixes=_mixes_from_one), 0.0, 1.0, VectorKappa2()),
    "mixing up to t_min only": (
        VectorKappa2(mixes=_mixes_up_to_four_tenths), 0.4, 2.0),
}


@pytest.mark.parametrize("case", NOT_READ_BY_THE_CALL)
def test_a_kernel_diagonal_on_the_window_is_evaluated_under_diag_C(case):
    """The other side.  C is the integral of ``R κ² R`` over the window from
    ``t_min`` to the latest time of the call, so a mixing that starts at
    that time, or ends at ``t_min``, adds nothing to it.  A position pair
    at which the kernel raises is left out, and an off-diagonal entry of
    5e-13 of the largest is below the 1e-12 that is read as mixing.  The
    kernels read are those of the tables: one case expands a system whose
    own κ² mixes everywhere.  Each runs with both flags on and gives the
    double integral of the diagonal entry, with the points left free and
    with both pinned at the latest time."""
    kernel, t_min, T, *own = NOT_READ_BY_THE_CALL[case]
    d, g = 0.8, 1.7

    def system_of(fn):
        return sw.System(field=sw.FieldSpec("phi", N),
                         linear=sw.DiagonalA(gamma=[1.0, g]),
                         noise=sw.GaussianNoise(kappa2=GeneralKappa2(fn=fn)),
                         t_min=t_min)

    system = system_of(kernel)
    exp = system_of(own[0] if own else kernel).expand(OBS, orders=[0])
    props = system.propagators(t_max=T, n_grid_t=5, progress=False)
    half = dblquad(lambda s2, s1: np.exp(-g * (2 * T - s1 - s2) - (s1 - s2)),
                   t_min, T, t_min, lambda s1: s1,
                   epsabs=1e-14, epsrel=1e-13)[0]
    apart = np.linalg.norm(X3 - Y3)
    # The latest time read is t_final with the points left free, and the
    # pinned time, below a t_final that is then not read, with both pinned.
    for times in (dict(t_final=T),
                  dict(t_final=T + 0.5, external_times={"x": T, "y": T})):
        got = exp.evaluate(props, positions={"x": X3, "y": Y3}, **times,
                           component_pair=(1, 1), orders=[0],
                           method="gauss_legendre").total
        assert got == pytest.approx(2.0 * d * np.exp(-apart) * half,
                                    rel=1e-9, abs=0.0), times


@pytest.mark.parametrize("gamma", [[1e-9, 5e-9], [5e-9, 1e-9], [1e-9, 0.0]])
def test_small_rates_are_told_apart_with_no_absolute_floor(gamma):
    """``np.allclose`` called these equal, so R and C of the second listed
    rate took the first.  At ``t = 1e9`` that was a factor 55 in R and 4.3 in
    C.  The reference is the OU solution written out here."""
    t_final = 1e9
    noise = sw.GaussianNoise(
        kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=1.0),
            spatial=sw.ExponentialSpatial(sigma_x=1.0),
        ),
        sigma2=sw.ConstantImpulse(0.7),
    )
    lin = sw.DiagonalA(gamma=gamma)
    assert lin.is_iso_R is False
    system = sw.System(field=sw.FieldSpec("phi", 2), linear=lin, noise=noise)
    props = system.propagators(
        t_max=2 * t_final, n_grid_t=6, diag_C=False, c_closed_form_only=True
    )
    C = np.asarray(props.cache.C_value(0.0, t_final, 0.0, t_final))
    for a, rate in enumerate(gamma):
        var = 0.7 * (
            t_final
            if rate == 0.0
            else -np.expm1(-2 * rate * t_final) / (2 * rate)
        )
        assert C[a, a] == pytest.approx(var, rel=1e-10, abs=0.0)
    R = lin.build_R_callable()
    assert np.diag(R(t_final, 0.0)) == pytest.approx(
        np.exp(-np.array(gamma) * t_final), rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("gap, equal", [(1e-10, True), (1e-8, False)])
def test_rates_equal_to_the_tolerance_lower_to_one_rate(gap, equal):
    """Each side of the relative tolerance of an index-free C, at a size the
    old absolute floor could not see and at a size it could."""
    for scale in (1e-12, 1.0, 1e8):
        lin = sw.DiagonalA(gamma=[scale, scale * (1 + gap)])
        assert lin.is_iso_R is equal
