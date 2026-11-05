"""Dispatcher-level tests for the C-propagator build path.

* separability: the lazy translation cache builds ONE temporal table and
  rescales it by ``κ_x(r)`` -- numbers identical to the per-``r`` build;
* time symmetry: the upper-triangle build reproduces the full one;
* ``c_method='auto'``: the Gauss-Legendre node count is refined until the
  rule is converged for the requested ``t_max``, every table uses that
  count, ``c_source`` and the dry run report it, user callables and direct
  cells take the same rule per pair of positions, and dblquad is the last
  step, with a warning for that pair;
* progress reporting never changes a number and is silent by default.
"""

from __future__ import annotations

import io
import math
import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pytest
import yaml
from scipy.integrate import quad

import sft_wick as sw
from sft_wick import evaluate as ev
from sft_wick import progress as prog
from sft_wick.evaluate import PropagatorCache, PropagatorModel, select_gl_node_count


def _demo1_like(sigma_t=0.5, gamma=(1.0, 1.0), temporal="exponential"):
    F = np.zeros((2, 2, 2))
    F[0, 1, 1] = 1.0
    if temporal == "exponential":
        kt = sw.ExponentialTemporal(lam=0.05, sigma_t=sigma_t)
    else:
        kt = sw.GaussianTemporal(lam=0.05, sigma_t=sigma_t)
    return sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=list(gamma)),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=kt, spatial=sw.ExponentialSpatial(sigma_x=1.0))),
    )


# --------------------------------------------------------------------- #
# Separability and symmetry in the lazy translation cache               #
# --------------------------------------------------------------------- #


class _NoShortcuts(PropagatorCache):
    """Reference: the per-r, full-square build the lazy cache used to do."""

    def _lazy_spatial_factor(self):
        return None

    def _c_time_symmetric(self):
        return False


@pytest.mark.parametrize("c_method", ["gauss_legendre", "dblquad"])
def test_lazy_translation_shared_temporal_table_matches_per_r_build(c_method):
    system = _demo1_like()
    model = system.build_propagator_model()
    kw = dict(homogeneity="translation", c_method=c_method, n_gauss=16)
    fast = PropagatorCache(model, **kw)
    slow = _NoShortcuts(model, **kw)
    # Cost note (2026-09): n_grid_t was 12.  ``fast`` and ``slow`` build
    # the SAME grid by two routes, so the resolution is not what is
    # asserted -- the agreement is.  Measured: worst relative gap over
    # the four r's is 3.436e-16 (GL) / 1.718e-16 (dblquad) at n_grid_t
    # = 12, and bit-identically the same at 8 and at 6, while the build
    # cost falls as n_grid_t**2 (dblquad: 6.4 s -> 2.6 s).  Tolerances
    # below are unchanged.
    n_grid_t, t_max = 8, 4.0
    fast.precompute_C_table_translation(t_max=t_max, n_grid_t=n_grid_t)
    slow.precompute_C_table_translation(t_max=t_max, n_grid_t=n_grid_t)

    rs = [0.0, 0.5, 1.0, 2.5]
    t1 = np.array([0.3, 1.0, 2.5, 4.0, 4.0, 3.1])
    t2 = np.array([0.3, 2.0, 2.5, 4.0, 0.2, 0.9])
    for r in rs:
        a = fast.C_at_batch(t1, t2, np.zeros(6), np.full(6, r))
        b = slow.C_at_batch(t1, t2, np.zeros(6), np.full(6, r))
        scale = np.max(np.abs(b))
        # GL is a fixed rule, so the two builds differ only by rounding;
        # dblquad re-adapts per cell (its epsabs floor moves with the
        # magnitude), so agreement there is bounded by ITS tolerance.
        tol = 1e-12 if c_method == "gauss_legendre" else 1e-6
        assert np.max(np.abs(a - b)) <= tol * scale, (r, np.max(np.abs(a - b)) / scale)

    # The shortcut ran the quadrature grid exactly once for the four r's;
    # the reference ran it four times.
    assert fast._lazy_translation.n_grid_builds == 1
    assert slow._lazy_translation.n_grid_builds == len(rs)


def test_separability_shortcut_is_off_with_white_noise_or_user_c():
    system = _demo1_like()
    model = system.build_propagator_model()
    with_white = sw.System(
        field=system.field, linear=system.linear, vertices=system.vertices,
        noise=sw.GaussianNoise(kappa2=system.noise.kappa2,
                               sigma2=sw.ConstantImpulse(0.2)),
    ).build_propagator_model()
    assert PropagatorCache(model)._lazy_spatial_factor() is not None
    assert PropagatorCache(with_white)._lazy_spatial_factor() is None
    user = PropagatorCache(model, c_value_fn=lambda n1, t1, n2, t2: np.eye(2))
    assert user._lazy_spatial_factor() is None
    assert user._c_time_symmetric() is False


def test_builtin_closed_form_cache_uses_the_shortcut_too():
    system = _demo1_like()
    props = system.propagators(t_max=4.0, n_grid_t=10)
    lazy = props.cache._lazy_translation
    t = np.array([1.0, 2.0, 4.0])
    for r in (0.0, 0.7, 1.9):
        props.cache.C_at_batch(t, t, np.zeros(3), np.full(3, r))
    assert lazy.n_grid_builds == 1


# --------------------------------------------------------------------- #
# c_method='auto': node-count selection                                  #
# --------------------------------------------------------------------- #


def test_select_gl_node_count_escalates_with_the_horizon():
    """The demo1 kernel (σ_t = 0.3) is converged at n=20 for t_max=10 but
    not for t_max=100, where a fixed n=20 is ~2% off (documented in
    PropagatorCache).  The selector must notice."""
    model = _demo1_like(sigma_t=0.3).build_propagator_model()
    assert select_gl_node_count(model, 10.0) == 20
    n_long = select_gl_node_count(model, 100.0)
    assert n_long is None or n_long > 20


def test_auto_resolves_per_system_family_at_l1():
    gaussian = _demo1_like(temporal="gaussian", sigma_t=0.5)
    props = gaussian.propagators(t_max=6.0, n_grid_t=10)
    assert props.c_source.startswith("quadrature:gauss_legendre(n="), props.c_source
    n = props.cache.n_gauss_resolved
    assert n >= 20

    # Its table must agree with dblquad at the deep corner: the whole
    # point of the convergence check.
    ref = PropagatorCache(
        gaussian.build_propagator_model(), c_method="dblquad",
        quad_opts={"epsabs": 1e-13, "epsrel": 1e-10},
    )
    got = props.cache._C_value_direct(np.asarray(0.0), 6.0, np.asarray(0.0), 6.0)
    exp = ref._C_value_direct(np.asarray(0.0), 6.0, np.asarray(0.0), 6.0)
    assert np.max(np.abs(got - exp)) <= 1e-6 * np.max(np.abs(exp))


def test_a_table_resolves_auto():
    model = _demo1_like().build_propagator_model()
    cache = PropagatorCache(model)          # c_method='auto'
    assert cache.c_method_resolved == "auto"
    cache.precompute_C_table_translation(t_max=3.0, n_grid_t=6)
    assert cache.c_method_resolved == "gauss_legendre"
    assert cache.n_gauss_resolved == 20


@pytest.mark.parametrize("sigma_t, gamma", [
    (0.002, (0.5, 0.1)),
    (0.3, (300.0, 1e-3)),
])
def test_a_table_no_node_count_converges_on_takes_the_graded_rule(sigma_t, gamma):
    """A narrow kernel and a fast rate, ``auto`` on the defaults.  The first
    row sent ``dblquad`` past the ridge: 5.8e-10 against 5.0e-3.  The second
    gave 2.5e-11 against 1.11e-5."""
    t_max = 32.0
    props = _slow_system(sigma_t, gamma).propagators(
        t_max=t_max, n_grid_t=4, c_closed_form=None)
    assert props.c_source == "quadrature:graded(levels=12)"
    want = [_equal_time_c(0.05, sigma_t, g, t_max - 2.0) for g in gamma]
    table = np.diag(props.cache.C_value(0.0, t_max, 0.0, t_max))
    direct = np.diag(props.cache._C_value_direct(0.0, t_max, 0.0, t_max))
    assert table == pytest.approx(want, rel=1e-8, abs=0.0)
    assert direct == pytest.approx(want, rel=1e-8, abs=0.0)


class _NarrowGaussian:
    """A user kernel equal to ``GaussianTemporal(lam, sigma)``."""

    def __init__(self, lam, sigma):
        self.lam, self.sigma = lam, sigma

    def __call__(self, dt):
        return self.lam * float(np.exp(-dt * dt / (2.0 * self.sigma**2)))


def _callable_system(kernel, gamma=0.3, t_min=2.0):
    return sw.System(
        field=sw.FieldSpec("phi", n_components=1),
        linear=sw.DiagonalA(gamma=[gamma]),
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.CustomKernel(fn=kernel),
            spatial=sw.ExponentialSpatial(sigma_x=1.0))),
        t_min=t_min)


@pytest.mark.parametrize("sigma_t, source", [
    (1000.0, "quadrature:gauss_legendre(n=20)"),
    (0.3, "quadrature:gauss_legendre(n=68)"),
    (0.01, "quadrature:graded(levels=12)"),
])
def test_auto_checks_a_user_kernel(sigma_t, source):
    """``dblquad``, unchecked, gave 1.71e-4 against 6.25e-3 at ``sigma_t =
    0.01`` and was off by 0.33 at 0.05."""
    props = _callable_system(_NarrowGaussian(0.05, sigma_t)).propagators(
        t_max=32.0, n_grid_t=4, c_closed_form=None)
    assert props.c_source == source
    assert float(props.cache.C_value(0.0, 32.0, 0.0, 32.0)[0, 0]) == pytest.approx(
        _equal_time_c(0.05, sigma_t, 0.3, 30.0), rel=1e-8, abs=0.0)


def test_a_cell_before_any_table_is_checked_and_needs_no_history():
    """It was ``dblquad`` here too, off by 0.97 at this cell."""
    def fresh():
        return PropagatorCache(
            _callable_system(_NarrowGaussian(0.05, 0.01), t_min=0.0)
            .build_propagator_model())

    want = _equal_time_c(0.05, 0.01, 0.3, 30.0)
    cache = fresh()
    first = cache._C_value_direct(0.0, 30.0, 0.0, 30.0)
    assert first[0, 0] == pytest.approx(want, rel=1e-8, abs=0.0)
    assert cache.c_method_resolved == "auto"

    other = fresh()
    other._C_value_direct(0.0, 3.0, 0.0, 3.0)
    assert np.array_equal(other._C_value_direct(0.0, 30.0, 0.0, 30.0), first)
    assert other._C_value_direct(0.0, 0.0, 0.0, 30.0)[0, 0] == 0.0


def test_a_kink_inside_the_interval_reaches_the_warning():
    """Depths of the graded rule share their interior panels.  Comparing only
    those passed this kernel at 12 levels, off by 1.5e-5, with no warning."""
    def triangle(dt):
        return max(0.0, 1.0 - abs(dt) / 2.0)

    cache = PropagatorCache(_callable_system(triangle).build_propagator_model())
    with pytest.warns(UserWarning, match="falls back to dblquad"):
        assert cache.resolve_c_method(22.0) == ("dblquad", None)


class _WidthByPosition:
    """A kernel of time width 0.3 at the origin and 0.01 at ``x = 1``."""

    def __call__(self, n1, t1, n2, t2):
        sigma = 0.3 if abs(float(n1)) < 0.5 else 0.01
        return 0.05 * np.exp(-((t1 - t2) ** 2) / (2.0 * sigma**2)) * np.eye(1)


def _general_system(kernel, gamma=0.3):
    return sw.System(
        field=sw.FieldSpec("phi", n_components=1),
        linear=sw.DiagonalA(gamma=[gamma]),
        noise=sw.GaussianNoise(kappa2=sw.GeneralKappa2(fn=kernel)))


def test_a_lazy_table_finds_its_rule_at_each_position():
    """Off by 0.72 from an unchecked ``dblquad``."""
    props = _general_system(_WidthByPosition()).propagators(
        t_max=12.0, n_grid_t=4, c_closed_form=None)
    assert float(props.cache.C_value(1.0, 12.0, 1.0, 12.0)[0, 0]) == pytest.approx(
        _equal_time_c(0.05, 0.01, 0.3, 12.0), rel=1e-8, abs=0.0)


def test_a_full_grid_is_checked_at_its_positions_whatever_was_resolved():
    """Off by 6.9e-2 at ``x = 1`` from the node count of the origin."""
    cache = PropagatorCache(
        _general_system(_WidthByPosition()).build_propagator_model(),
        homogeneity="general")
    assert cache.resolve_c_method(12.0)[0] == "gauss_legendre"
    pairs = [(np.asarray(x), np.asarray(x)) for x in (0.0, 1.0)]
    assert cache.resolve_c_method(12.0, pairs) == ("graded", 12)


class _NarrowOrOscillating:
    """Time width 0.01 where both positions are above 0, and elsewhere a
    kernel of many periods over the interval, which a count of 30 resolves
    and 8 nodes on a quarter of the interval do not."""

    def __call__(self, n1, t1, n2, t2):
        if float(n1) > 0 and float(n2) > 0:
            return np.exp(-((t1 - t2) ** 2) / (2.0 * 0.01**2)) * np.eye(1)
        return _oscillating(t1, t2) * np.eye(1)


def _oscillating(t1, t2):
    lag = t1 - t2
    return (np.exp(-(lag**2) / 50.0 - 0.2 * (t1 + t2)) * np.cos(7.0 * lag)
            * np.cos(4.9 * t1) * np.cos(4.9 * t2))


def test_a_position_keeps_the_count_it_converged_at_beside_a_narrow_one():
    """The narrow position sets the graded rule, and that rule was taken to
    resolve whatever a count of 30 does: the oscillating positions passed at
    30 and were served by 12 levels, 3.3e-5 off with no warning."""
    gamma, span = 0.3, 5.0
    x, w = np.polynomial.legendre.leggauss(500)
    s, w = 0.5 * span * (x + 1.0), 0.5 * span * w * np.exp(-gamma * (span - 0.5 * span * (x + 1.0)))
    want = w @ _oscillating(s[:, None], s[None, :]) @ w

    cache = PropagatorCache(
        _general_system(_NarrowOrOscillating(), gamma).build_propagator_model(),
        homogeneity="general")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cache.precompute_C_table_general(
            t_max=span, n_grid_t=4, x_max=1.0, n_grid_x=2)
    assert cache.c_method_resolved == "graded"

    def table(x1, x2):
        return float(np.asarray(cache.C_at_batch(
            np.array([span]), np.array([span]), np.array([x1]), np.array([x2])
        )).ravel()[0])

    assert table(-1.0, -1.0) == pytest.approx(want, rel=1e-8, abs=0.0)
    assert table(1.0, -1.0) == pytest.approx(want, rel=1e-8, abs=0.0)
    assert table(1.0, 1.0) == pytest.approx(
        _equal_time_c(1.0, 0.01, gamma, span), rel=1e-8, abs=0.0)
    # A direct cell inside the table is the cell the table holds: it took the
    # table's depth for a pair that stays on its count, 3.3e-5 off.
    assert _c(cache, -1.0, span, -1.0, span) == pytest.approx(want, rel=1e-8, abs=0.0)
    for x in (-1.0, 1.0):
        asked = cache._table_kwargs(np.asarray(x), np.asarray(x), span)
        assert _c(cache, x, span, x, span) == float(
            cache._C_value_direct(x, span, x, span, **asked)[0, 0])


def test_the_graded_method_with_no_depth_is_graded():
    """Named with no depth and no table, it was ``n=20`` on the tensor rule,
    off by 0.49 on a ridge of width 0.01."""
    cache = PropagatorCache(
        _callable_system(_NarrowGaussian(0.05, 0.01)).build_propagator_model())
    assert cache._C_value_direct(0.0, 30.0, 0.0, 30.0, method="graded")[0, 0] == (
        pytest.approx(_equal_time_c(0.05, 0.01, 0.3, 28.0), rel=1e-8, abs=0.0))


def _narrow_at_a_node(key):
    """Time width 0.3 everywhere but near ``key = 0.5``, 0.03 there."""
    return 0.3 - 0.27 * np.exp(-((key - 0.5) ** 2) / 0.01)


def _table_narrow_inside(kind):
    """A table and the positions of a grid node where the kernel narrows.

    The node is neither an end of the grid nor the origin.  The ``general``
    kernel is a Paciorek-Schervish covariance, so it is a valid one.
    """
    def key(n1, n2):
        if kind == "translation":
            return abs(float(n2) - float(n1))
        return float(np.dot(n1, n2))

    def kernel(n1, t1, n2, t2):
        if kind == "general":
            a, b = _narrow_at_a_node(float(n1)), _narrow_at_a_node(float(n2))
            size = a * a + b * b
            weight = np.sqrt(2.0 * a * b / size)
        else:
            size = 2.0 * _narrow_at_a_node(key(n1, n2)) ** 2
            weight = 1.0
        return 0.05 * weight * np.exp(-((t1 - t2) ** 2) / size) * np.eye(1)

    model = ev.PropagatorModel(
        n_components=1,
        R_time=lambda t, s: np.exp(-0.4 * (t - s)) * float(t > s),
        kappa2=kernel, t_min=0.0, iso_R=True, diag_C=True)
    cache = PropagatorCache(model, homogeneity=kind)
    if kind == "translation":
        cache.precompute_C_table_translation(2.0, 2, r_max=1.0, n_grid_r=3)
        return cache, np.asarray(0.0), np.asarray(0.5)
    if kind == "rotation":
        cache.precompute_C_table_rotation(2.0, 2, n_grid_cos=5)
        return cache, np.array([1.0, 0.0]), np.array([0.5, np.sqrt(0.75)])
    cache.precompute_C_table_general(2.0, 2, x_max=1.0, n_grid_x=5)
    return cache, np.asarray(0.5), np.asarray(0.5)


@pytest.mark.parametrize("kind", ["translation", "rotation", "general"])
def test_a_full_table_finds_its_rule_at_every_grid_position(kind):
    """The rule was tried at the ends of the grid only: ``n=20`` at the node
    where the kernel narrows was off by 2.8e-4, with no warning."""
    cache, n1, n2 = _table_narrow_inside(kind)
    assert float(cache.C_value(n1, 2.0, n2, 2.0)[0, 0]) == pytest.approx(
        _equal_time_c(0.05, 0.03, 0.4, 2.0), rel=1e-7, abs=0.0)


class _RaisesOnAScalar:
    """Width 0.01 at any position, reading components of a vector one."""

    def __call__(self, n1, t1, n2, t2):
        return (0.05 + 0.0 * n1[0] * n2[1]) * np.exp(
            -((t1 - t2) ** 2) / (2.0 * 0.01**2)) * np.eye(1)


class _VanishesAtTheOrigin:
    def __call__(self, n1, t1, n2, t2):
        return 0.05 * float(n1) * float(n2) * np.exp(
            -((t1 - t2) ** 2) / (2.0 * 0.01**2)) * np.eye(1)


@pytest.mark.parametrize("kernel, x", [
    (_RaisesOnAScalar(), np.array([0.3, 0.4])),
    (_VanishesAtTheOrigin(), 1.0),
])
def test_a_lazy_table_decides_nothing_at_an_origin_that_says_nothing(kernel, x):
    """Both ended in ``dblquad``, off by 8.0e-4 and 0.72, under a warning that
    no rule converged although none was tried where C is read."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        props = _general_system(kernel).propagators(
            t_max=12.0, n_grid_t=4, c_closed_form=None)
        value = float(props.cache.C_value(x, 12.0, x, 12.0)[0, 0])
    assert props.c_source == "quadrature:per_position"
    assert value == pytest.approx(
        _equal_time_c(0.05, 0.01, 0.3, 12.0), rel=1e-8, abs=0.0)


def test_a_separable_kernel_that_vanishes_at_the_origin_finds_its_rule_at_the_pair():
    """The origin holds no C, so the rule of the origin decides nothing.  At 20
    nodes the width 0.06 is not resolved: 1.2e-4 off, with no warning."""
    gamma, lam, sigma_t, t = 0.4, 0.05, 0.06, 3.0
    system = sw.System(
        field=sw.FieldSpec("phi", 1),
        linear=sw.DiagonalA(gamma=[gamma]),
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.GaussianTemporal(lam=lam, sigma_t=sigma_t),
            spatial=sw.CustomKernel(fn=lambda r: r * r))),
    )
    props = system.propagators(t_max=t, n_grid_t=4, c_closed_form=None)
    value = float(props.cache.C_value(0.0, t, 1.0, t)[0, 0])
    assert value == pytest.approx(
        _equal_time_c(lam, sigma_t, gamma, t), rel=1e-8, abs=0.0
    )


class _SmoothOrStepped:
    """Smooth at the origin; a step at a lag of 2 at ``x = 1``, which no rule
    converges on."""

    def __call__(self, n1, t1, n2, t2):
        lag = abs(t1 - t2)
        step = 0.0 if abs(float(n1)) < 0.5 else float(lag > 2.0)
        return (np.exp(-lag * lag) + step) * np.eye(1)


def test_no_rule_at_a_position_means_dblquad_there():
    """The warning said ``dblquad`` and the origin's 20 nodes built the table,
    1.3e-2 off at the cell below with 0 ``dblquad`` calls."""
    gamma, t = 0.4, 8.0
    cache = PropagatorCache(
        _general_system(_SmoothOrStepped(), gamma).build_propagator_model(),
        homogeneity="general")
    assert cache._resolve_build_method(None, None, t)[0] == "gauss_legendre"
    x = np.asarray(1.0)
    with pytest.warns(UserWarning, match="falls back to dblquad") as caught:
        kwargs = cache._table_kwargs(x, x, t)
        cache._table_kwargs(x, x, t)
    assert len(caught) == 1 and "positions (1., 1.)" in str(caught[0].message)
    assert kwargs == {"method": "dblquad"}
    # The other pairs keep their rule, and the label says the table is mixed.
    assert cache._table_kwargs(0.0, 0.0, t)["method"] == "gauss_legendre"
    assert cache.c_method_resolved == "auto"

    def integrand(u):
        return ((np.exp(-u * u) + float(u > 2.0)) * np.exp(-gamma * u)
                * -np.expm1(-2 * gamma * (t - u)) / (2 * gamma))

    want = 2 * (quad(integrand, 0.0, 2.0, epsabs=0.0, epsrel=1e-11)[0]
                + quad(integrand, 2.0, t, epsabs=0.0, epsrel=1e-11)[0])
    got = cache._C_value_direct(x, t, x, t, **kwargs)[0, 0]
    assert got == pytest.approx(want, rel=1e-5, abs=0.0)


def test_an_explicit_build_method_keeps_its_rule_at_every_position():
    cache = PropagatorCache(
        _general_system(_SmoothOrStepped()).build_propagator_model(),
        homogeneity="general")
    assert cache._resolve_build_method("gauss_legendre", 12, 4.0) == (
        "gauss_legendre", 12)
    kwargs = {"method": "gauss_legendre", "n_gauss": 12}
    assert cache._table_kwargs(1.0, 1.0, 4.0) == kwargs


class _StepOrNarrow:
    """A step at a lag of 0.37 at the origin, which no rule converges on, and
    a Gaussian of width 0.01 at ``x = 1``."""

    def __call__(self, n1, t1, n2, t2):
        lag = abs(t1 - t2)
        if float(n1) == 0.0:
            return float(lag > 0.37) * np.eye(1)
        return float(np.exp(-((lag / 0.01) ** 2))) * np.eye(1)


class _NarrowInsideABox:
    """A Gaussian of width 0.002, and exactly 0 at a separation of 1 or more."""

    def __call__(self, n1, t1, n2, t2):
        inside = abs(float(n1) - float(n2)) < 1.0
        return float(inside) * np.exp(
            -((t1 - t2) ** 2) / (2.0 * 0.002**2)) * np.eye(1)


class _RidgeAndLateStep:
    """A ridge of width 0.002, and a step at a lag of 25."""

    def __call__(self, n1, t1, n2, t2):
        lag = abs(t1 - t2)
        return (np.exp(-((lag / 0.002) ** 2) / 2.0) + 0.3 * float(lag > 25.0)
                ) * np.eye(1)


def _c(cache, x1, t1, x2, t2):
    return float(cache._C_value_direct(x1, t1, x2, t2)[0, 0])


@pytest.mark.parametrize("kernel, gamma, first, cell", [
    (_StepOrNarrow(), 0.1, (0.0, 30.0, 0.0, 30.0), (1.0, 30.0, 1.0, 30.0)),
    (_NarrowInsideABox(), 0.5, (0.0, 20.0, 5.0, 20.0), (0.0, 20.0, 0.0, 20.0)),
    (_RidgeAndLateStep(), 0.05, (0.0, 30.0, 0.0, 29.0), (0.0, 25.0, 0.0, 25.0)),
])
def test_the_rule_of_a_cell_does_not_depend_on_the_calls_before(
        kernel, gamma, first, cell):
    """A rule is a function of its cell, not of the cells read before."""
    model = _general_system(kernel, gamma).build_propagator_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = _c(PropagatorCache(model, homogeneity="general"), *cell)
        cache = PropagatorCache(model, homogeneity="general")
        _c(cache, *first)
        assert _c(cache, *cell) == fresh
    assert fresh > 1e-3


@pytest.mark.parametrize("kernel, x2", [
    (lambda n1, t1, n2, t2: 0.0 * np.eye(1), 0.0),
    (_NarrowInsideABox(), 3.0),
])
def test_a_c_that_is_zero_is_no_failed_rule(kernel, x2):
    """C is zero at the rule's probes, so no rule is searched for and nothing
    is warned about."""
    cache = PropagatorCache(
        _general_system(kernel).build_propagator_model(), homogeneity="general")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert _c(cache, 0.0, 5.0, x2, 5.0) == 0.0


def _two_rates(gamma, sigma_t):
    rates = np.asarray(gamma)
    return sw.evaluate.PropagatorModel(
        n_components=2,
        R_time=lambda t, s: np.diag(np.exp(-rates * (t - s))) * float(t > s),
        kappa2=lambda n1, t1, n2, t2: float(
            np.exp(-((t1 - t2) ** 2) / (2.0 * sigma_t**2))) * np.eye(2),
        t_min=2.0, iso_R=False, diag_C=True)


def test_a_component_is_not_judged_against_a_stronger_one():
    """``C_11`` came back 0 under ``n=20`` with no warning, 1.0 off, because
    the scale of the check was that of ``C_00``."""
    gamma, sigma_t = (0.05, 3e5), 2.0
    model = _two_rates(gamma, sigma_t)
    assert select_gl_node_count(model, 30.0) is None
    cache = PropagatorCache(model, homogeneity="general")
    got = cache._C_value_direct(0.0, 30.0, 0.0, 30.0, method="graded", n_gauss=20)
    assert got[1, 1] == pytest.approx(
        _equal_time_c(1.0, sigma_t, gamma[1], 28.0), rel=1e-7, abs=0.0)
    assert got[0, 0] == pytest.approx(
        _equal_time_c(1.0, sigma_t, gamma[0], 28.0), rel=1e-7, abs=0.0)
    # Rates of one size stay on the tensor rule.
    assert select_gl_node_count(_two_rates((0.05, 0.1), sigma_t), 30.0) == 20


def test_a_position_is_not_judged_against_a_stronger_one():
    """A far position 1e-9 of the origin, with a kernel only the finer rule
    resolves, passed at the scale of the origin."""
    def kernel(n1, t1, n2, t2):
        far = abs(float(n1)) > 0.5
        size, width = (1e-9, 0.05) if far else (1.0, 3.0)
        return size * np.exp(-((t1 - t2) ** 2) / (2.0 * width**2)) * np.eye(1)

    model = _general_system(kernel).build_propagator_model()
    near, far = np.asarray(0.0), np.asarray(1.0)
    cache = PropagatorCache(model, homogeneity="general")
    count = select_gl_node_count(model, 8.0, positions=(far, far))
    assert count > 20
    assert cache.resolve_c_method(8.0, [(near, near), (far, far)]) == (
        "gauss_legendre", count)


class _OneWayR:
    """R of ``A = [[-0.2, 1], [0, -30]]``: row 1 has no component 0.  ``off``
    puts rounding of an eigenbasis formula where the exact entry is 0."""

    def __init__(self, off):
        self.off = off

    def __call__(self, t, s):
        u = t - s
        slow, fast = np.exp(-0.2 * u), np.exp(-30.0 * u)
        return np.array(
            [[slow, (slow - fast) / 29.8], [self.off * slow, fast]]
        ) * float(u >= 0)


class _OneStrongComponent:
    def __init__(self, big):
        self.big = big

    def __call__(self, n1, t1, n2, t2):
        return np.diag([self.big, 1.0]) * np.exp(-((t1 - t2) ** 2) / 8.0)


@pytest.mark.parametrize("big, off", [(1e2, 0.0), (1e8, 0.0), (1e14, 0.0), (1e14, 1e-17)])
def test_an_entry_r_keeps_from_the_strong_component_has_no_rounding_of_it(big, off):
    """R mixes components here, but ``C_11`` holds no term of the strong
    one.  ``n=20`` passed on a difference of 1e-13 of ``C_00``, ``C_11`` 65 %
    off at ``big = 1e14`` and 1.6e-3 at ``1e8``, with no warning."""
    cache = PropagatorCache(
        sw.evaluate.PropagatorModel(
            n_components=2,
            R_time=_OneWayR(off),
            kappa2=_OneStrongComponent(big),
            iso_R=False,
            diag_C=False,
        ),
        homogeneity="general",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rule = cache.resolve_c_method(30.0)
    got = cache._C_value_direct(0.0, 30.0, 0.0, 30.0, **cache._direct_kwargs(*rule))
    assert got[1, 1] == pytest.approx(
        _equal_time_c(1.0, 2.0, 30.0, 30.0), rel=1e-8, abs=0.0), rule
    assert got[1, 0] == pytest.approx(got[0, 1], rel=1e-7, abs=0.0), rule


def test_an_entry_that_cancels_a_strong_component_keeps_its_rounding():
    """The other side: ``C_22`` is about 1 and its terms about 1e10, which
    cancel, so two rules differ there by rounding.  With no allowance the
    build took 300 s and ended in ``per_position``."""
    big, sigma_t, span = 1e10, 2.0, 30.0
    rates = np.array([0.2, 1.0, 3.0])
    to_x = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    to_y = np.array([[0.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 1.0]])
    sizes = np.array([big, 1.0, 1.0])

    def response(t, s):
        return (to_x * np.exp(-rates * (t - s))) @ to_y * float(t >= s)

    def kernel(n1, t1, n2, t2):
        return (to_x * sizes) @ to_x.T * np.exp(-((t1 - t2) ** 2) / (2 * sigma_t**2))

    system = sw.System(
        field=sw.FieldSpec("phi", n_components=3),
        linear=sw.ExplicitR(R_time=response, iso_R=False),
        noise=sw.GaussianNoise(kappa2=sw.GeneralKappa2(fn=kernel)))
    with pytest.warns(UserWarning, match="cannot be checked") as caught:
        props = system.propagators(t_max=span, n_grid_t=5, diag_C=False)
    assert len(caught) == 1 and "(2, 2)" in str(caught[0].message)
    assert props.c_source == "quadrature:gauss_legendre(n=20)"
    modes = [_equal_time_c(k, sigma_t, g, span) for k, g in zip(sizes, rates)]
    want = (to_x * modes) @ to_x.T
    # Rounding of the terms sets what is reachable: 4e-8 here.
    assert props.cache.C_value(0.0, span, 0.0, span)[2, 2] == pytest.approx(
        want[2, 2], rel=1e-7, abs=0.0)


@pytest.mark.parametrize("start", [73, 74, 120])
def test_a_large_starting_count_is_tested(start):
    """Above 73 the refinement passed the cap before any test, and a smooth
    kernel went to the graded rule, about 500 times slower."""
    kernel = _NarrowGaussian(0.05, 1.0)
    cache = PropagatorCache(
        _callable_system(kernel, gamma=0.5, t_min=0.0).build_propagator_model(),
        n_gauss=start)
    assert cache.resolve_c_method(5.0) == ("gauss_legendre", start)


def test_a_count_given_to_a_cell_before_any_rule_is_used():
    model = _callable_system(_NarrowGaussian(0.05, 1.0)).build_propagator_model()
    cache = PropagatorCache(model)
    gl = cache._C_value_direct(0.0, 3.0, 0.0, 3.0, method="gauss_legendre", n_gauss=12)
    assert np.array_equal(cache._C_value_direct(0.0, 3.0, 0.0, 3.0, n_gauss=12), gl)
    with pytest.raises(ValueError, match="'graded'"):
        cache._C_value_direct(0.0, 3.0, 0.0, 3.0, method="simpson")


def test_a_cell_past_the_table_takes_its_own_rule():
    """Off by 1.1e-1 from the 30 nodes of a table reaching ``t = 3``."""
    model = _callable_system(_NarrowGaussian(0.05, 0.05)).build_propagator_model()
    want = _equal_time_c(0.05, 0.05, 0.3, 30.0)
    cache = PropagatorCache(model)
    cache.precompute_C_table(t_max=3.0, n_grid=4)
    assert float(cache.C_value(0.0, 32.0, 0.0, 32.0)[0, 0]) == pytest.approx(
        want, rel=1e-8, abs=0.0)
    assert cache.resolve_c_method(32.0)[0] == "graded"


def test_dblquad_is_the_last_resort_and_says_so(monkeypatch):
    monkeypatch.setattr(ev, "select_gl_node_count", lambda *a, **k: None)
    monkeypatch.setattr(ev, "select_graded_levels", lambda *a, **k: None)
    cache = PropagatorCache(_slow_system().build_propagator_model())
    with pytest.warns(UserWarning, match="falls back to dblquad"):
        assert cache.resolve_c_method(32.0) == ("dblquad", None)


def test_l0_rejects_unknown_c_method():
    model = _demo1_like().build_propagator_model()
    with pytest.raises(ValueError):
        PropagatorCache(model, c_method="simpson")


def _slow_system(sigma_t=0.3, gamma=(0.1, 0.3)):
    """A kernel with a converged count far above the starting 20 when the
    horizon is 30 and the rates are slow."""
    return sw.System(
        field=sw.FieldSpec("phi", n_components=len(gamma)),
        linear=sw.DiagonalA(gamma=list(gamma)),
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.GaussianTemporal(lam=0.05, sigma_t=sigma_t),
            spatial=sw.ExponentialSpatial(sigma_x=1.0))),
        t_min=2.0,
    )


def _equal_time_c(lam, sigma_t, gamma, span):
    """``C_aa(t, t)`` from a one-dimensional integral over the lag ``u``,
    sharing no code with the package::

        2 λ ∫_0^L e^{-u²/(2σ²)} e^{-γu} (1 - e^{-2γ(L-u)}) / (2γ) du
    """
    def integrand(u):
        return (np.exp(-u * u / (2 * sigma_t**2) - gamma * u)
                * -np.expm1(-2 * gamma * (span - u)) / (2 * gamma))

    edges = sorted({
        0.0, span,
        *(min(k * sigma_t, span) for k in (1, 4, 10)),
        *(min(k / gamma, span) for k in (1, 4, 20, 60)),
    })
    return 2 * lam * sum(
        quad(integrand, a, b, epsabs=0.0, epsrel=1e-12, limit=200)[0]
        for a, b in zip(edges, edges[1:]))


@pytest.mark.parametrize("sigma_t, gamma, t_max, tol", [
    (0.3, (0.1, 0.3), 32.0, 1e-9),
    (0.01, (0.1, 0.3), 3.0, 1e-9),
    (1.0, (1e-3, 2e-3), 100.0, 1e-9),
    (1000.0, (0.1, 3.0), 32.0, 1e-9),
])
def test_auto_table_agrees_with_an_independent_integral(sigma_t, gamma, t_max, tol):
    """The table reports a converged count, so it must hold that accuracy.
    It used to be built with ``c_n_gauss`` nodes, off by 2.1e-3 and 5.7e-3
    at the first row's deep corner."""
    props = _slow_system(sigma_t, gamma).propagators(
        t_max=t_max, n_grid_t=4, c_closed_form=None)
    got = np.diag(props.cache.C_value(0.0, t_max, 0.0, t_max))
    want = [_equal_time_c(0.05, sigma_t, g, t_max - 2.0) for g in gamma]
    assert props.cache.c_method_resolved == "gauss_legendre"
    assert got == pytest.approx(want, rel=tol, abs=0.0), props.c_source
    direct = np.diag(props.cache._C_value_direct(0.0, t_max, 0.0, t_max))
    assert direct == pytest.approx(want, rel=tol, abs=0.0), props.c_source


@pytest.mark.parametrize("homogeneity, table", [
    ("translation", {}),
    ("translation", {"r_max": 1.0, "n_grid_r": 2}),
    ("rotation", {"n_grid_cos": 2}),
    ("general", {"x_max": 1.0, "n_grid_x": 2}),
])
def test_every_table_path_uses_the_resolved_count(monkeypatch, homogeneity, table):
    system = _slow_system()
    if homogeneity == "rotation":
        system = sw.System(
            field=system.field, linear=system.linear, t_min=system.t_min,
            noise=sw.GaussianNoise(kappa2=sw.SeparableRotation(
                temporal=sw.GaussianTemporal(lam=0.05, sigma_t=0.3),
                angular=sw.LegendreAngular(coeffs=[1.0, 0.3]))))
    cache = PropagatorCache(system.build_propagator_model(), homogeneity=homogeneity)
    method, n = cache.resolve_c_method(32.0)
    assert method == "gauss_legendre" and n > cache.n_gauss

    used = []
    real = ev._C_value_direct_gl

    def spy(*args, n_gauss):
        used.append(n_gauss)
        return real(*args, n_gauss=n_gauss)

    monkeypatch.setattr(ev, "_C_value_direct_gl", spy)
    build = getattr(cache, f"precompute_C_table_{homogeneity}")
    # An ``n_gauss`` here is a node count only for an explicit method.
    build(t_max=32.0, n_grid_t=2 if table else 4, n_gauss=cache.n_gauss, **table)
    if not table:
        cache.C_value(0.0, 32.0, 0.5, 32.0)
    assert set(used) == {n}
    assert cache.n_gauss_resolved == n


def test_explicit_gauss_legendre_keeps_its_count_without_a_check():
    system = _slow_system()
    props = system.propagators(t_max=32.0, n_grid_t=4, c_closed_form=None,
                               c_method="gauss_legendre", c_n_gauss=20)
    assert props.c_source == "quadrature:gauss_legendre(n=20)"
    got = np.diag(props.cache.C_value(0.0, 32.0, 0.0, 32.0))
    want = np.array([_equal_time_c(0.05, 0.3, g, 30.0) for g in (0.1, 0.3)])
    assert np.abs(got / want - 1.0) == pytest.approx([2.1e-3, 5.7e-3], rel=0.1, abs=0.0)

    # A per-build count is what the table holds, so it is what is reported.
    cache = PropagatorCache(system.build_propagator_model(),
                            c_method="gauss_legendre", n_gauss=20)
    cache.precompute_C_table_translation(t_max=32.0, n_grid_t=2, n_gauss=30)
    assert cache.n_gauss_resolved == 30


def _dry_run_doc():
    return {
        "system": {
            "field": {"name": "phi", "n_components": 2},
            "t_min": 2.0,
            "linear": {"type": "diagonal", "gamma": [0.1, 0.3]},
            "noise": {"kappa2": {
                "type": "separable_translation",
                "temporal": {"type": "gaussian", "lam": 0.05, "sigma_t": 0.3},
                "spatial": {"type": "exponential", "sigma_x": 1.0}}},
        },
        "expand": {"observable": ["phi_a(x)", "phi_b(y)"], "orders": [0]},
        "propagators": {"t_max": 32.0, "n_grid_t": 4, "c_closed_form": None},
        "sweep": {"positions_grid": {"x": [0.0], "y": [0.7]},
                  "t_final_grid": [31.0], "component_tuples": [[0, 0]],
                  "method": "gauss_legendre", "n_gauss": 2},
    }


def test_dry_run_times_a_position_when_the_origin_says_nothing(tmp_path):
    """The kernel raises on the scalar origin, where the timing used to read."""
    from sft_wick.workflow.config import load_workflow_config
    from sft_wick.workflow.estimate import estimate_cost

    (tmp_path / "vector_kernel.py").write_text(
        "import numpy as np\n\n\n"
        "def kappa(n1, t1, n2, t2):\n"
        "    return (1.0 + 0.0 * n1[0]) * np.exp(-(t1 - t2) ** 2) * np.eye(2)\n")
    doc = _dry_run_doc()
    doc["system"]["noise"]["kappa2"] = {
        "type": "callable_module", "module": "./vector_kernel.py", "attr": "kappa"}
    doc["sweep"]["positions_grid"] = {"x": [[0.0, 0.0]], "y": [[0.0, 0.7]]}
    doc["propagators"].pop("c_closed_form")
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(doc))
    est = estimate_cost(load_workflow_config(path), bench_seconds=0.01)
    assert est.c_source == "quadrature:per_position"


def test_dry_run_warns_once_where_no_rule_converges(tmp_path, monkeypatch):
    """The benchmark table ran the selection a second time, about 25 s, and
    the warning came twice."""
    from sft_wick.workflow.config import load_workflow_config
    from sft_wick.workflow.estimate import estimate_cost

    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(_dry_run_doc()))
    monkeypatch.setattr(ev, "select_gl_node_count", lambda *a, **k: None)
    monkeypatch.setattr(ev, "select_graded_levels", lambda *a, **k: None)
    with pytest.warns(UserWarning, match="falls back to dblquad") as caught:
        estimate_cost(load_workflow_config(path), bench_seconds=0.01)
    assert sum("falls back" in str(w.message) for w in caught) == 1


def test_dry_run_counts_both_time_orders_of_a_general_table(tmp_path, monkeypatch):
    """A general table stores the earlier-first half for any kernel, and the
    estimate counted one half: 324 cells of 702 at n_grid_t = 8, n_grid_x = 3."""
    from sft_wick.workflow.config import build_system, load_workflow_config
    from sft_wick.workflow.estimate import estimate_cost

    doc = _dry_run_doc()
    doc["propagators"].update(
        n_grid_t=8, homogeneity="general", n_grid_x=3, x_max=1.0)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(doc))
    cfg = load_workflow_config(path)
    estimated = estimate_cost(cfg, bench_seconds=0.01).n_c_calls

    cells = []
    tabulate = PropagatorCache._tabulate
    monkeypatch.setattr(
        PropagatorCache, "_tabulate",
        lambda self, shape, found, *args: cells.append(len(found))
        or tabulate(self, shape, found, *args))
    build_system(cfg.system).propagators(
        t_max=32.0, n_grid_t=8, n_grid_x=3, x_max=1.0, homogeneity="general",
        c_closed_form=None, c_n_gauss=4, c_method="gauss_legendre")
    assert len(cells) == 2
    assert estimated == pytest.approx(sum(cells), rel=0.2, abs=0.0)


def test_dry_run_reports_the_count_the_run_uses(tmp_path, monkeypatch):
    from sft_wick.workflow.config import load_workflow_config
    from sft_wick.workflow.estimate import estimate_cost

    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(_dry_run_doc()))
    runs = []
    select = ev.select_gl_node_count

    def counted(*args, **kwargs):
        runs.append(1)
        return select(*args, **kwargs)

    monkeypatch.setattr(ev, "select_gl_node_count", counted)
    est = estimate_cost(load_workflow_config(path), bench_seconds=0.01)
    assert len(runs) == 1  # the benchmark table takes an explicit rule
    monkeypatch.undo()
    run = _slow_system().propagators(t_max=32.0, n_grid_t=4, c_closed_form=None)
    assert est.c_source == run.c_source
    assert run.c_source == f"quadrature:gauss_legendre(n={run.cache.n_gauss_resolved})"


# --------------------------------------------------------------------- #
# Progress reporting                                                     #
# --------------------------------------------------------------------- #


def test_progress_callback_receives_every_stage_and_numbers_are_unchanged():
    system = _demo1_like()
    events: list = []

    def cb(desc, done, total):
        events.append((desc, done, total))

    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2])
    props = system.propagators(t_max=3.0, n_grid_t=8, c_closed_form=None)
    quiet = exp.sweep(
        props, positions_grid={"x": [0.0], "y": [0.0, 0.5]},
        t_final_grid=[1.0, 3.0], component_pairs=[(0, 0)],
        n_samples=256, progress=False,
    ).totals()
    loud = exp.sweep(
        props, positions_grid={"x": [0.0], "y": [0.0, 0.5]},
        t_final_grid=[1.0, 3.0], component_pairs=[(0, 0)],
        n_samples=256, progress=cb,
    ).totals()
    assert loud.equals(quiet)
    descs = {d for d, _, _ in events}
    assert any(d.startswith("sweep") for d in descs), descs
    finished = [(d, n, t) for d, n, t in events if n == t]
    assert finished, "callback never saw the bar complete"
    # 4 grid points x 7 diagrams (1 at order 0, 6 at order 2).
    assert max(t for _, _, t in events) == 4 * 7


def test_progress_is_silent_when_disabled_and_prints_when_forced(monkeypatch):
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setenv("SFT_WICK_PROGRESS", "0")
    with prog.progress_bar(3, "thing") as tick:
        tick(); tick(); tick()
    assert err.getvalue() == ""

    monkeypatch.setenv("SFT_WICK_PROGRESS", "1")
    monkeypatch.setattr(prog, "_DELAY", 0.0)   # bars hide sub-second loops
    err2 = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err2)
    with prog.progress_bar(3, "thing", unit="cell") as tick:
        tick(); tick(); tick()
    out = err2.getvalue()
    assert "thing" in out and "3/3" in out

    # A loop that finishes inside the delay prints nothing at all.
    monkeypatch.setattr(prog, "_DELAY", 60.0)
    err4 = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err4)
    with prog.progress_bar(3, "fast", unit="cell") as tick:
        tick(); tick(); tick()
    assert err4.getvalue() == ""

    # An explicit scope beats the environment.
    err3 = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err3)
    with prog.progress(False):
        with prog.progress_bar(2, "quiet") as tick:
            tick(); tick()
    assert err3.getvalue() == ""


def test_progress_map_parallel_matches_serial_and_ticks():
    tasks = list(range(12))
    ticks: list = []
    with prog.progress(lambda d, n, t: ticks.append((n, t))):
        serial = prog.progress_map(lambda x: x * x, tasks, "sq", n_jobs=1)
        parallel = prog.progress_map(lambda x: x * x, tasks, "sq", n_jobs=2)
    assert serial == parallel == [x * x for x in tasks]
    assert (12, 12) in ticks


def test_the_graded_rule_has_no_panel_of_zero_width():
    """Both halves of the rule held the edge 0.5, so a kernel that is NaN at
    the middle of an interval made the cell NaN through nodes of weight 0."""
    nodes, weights = ev._graded_rule(0.0, 1.0, 12, 8)
    assert nodes.size == 2 * 12 * 8
    assert np.all(weights > 0.0)
    assert weights.sum() == pytest.approx(1.0, abs=1e-14)

    def nan_at_the_middle(n1, t1, n2, t2):
        if t1 == 8.0 or t2 == 8.0:
            return np.full((1, 1), np.nan)
        return 0.05 * np.exp(-((t1 - t2) ** 2) / (2.0 * 0.01**2)) * np.eye(1)

    model = _general_system(nan_at_the_middle).build_propagator_model()
    got = ev._C_value_direct_gl(model, 0.0, 16.0, 0.0, 16.0, graded_levels=16)
    assert got[0, 0] == pytest.approx(
        _equal_time_c(0.05, 0.01, 0.3, 16.0), rel=1e-7, abs=0.0)


def _one_rate_model():
    return ev.PropagatorModel(
        n_components=1,
        R_time=lambda t, s: np.exp(-(t - s)) * float(t > s),
        kappa2=lambda n1, t1, n2, t2: np.exp(-abs(t1 - t2) / 0.7) * np.eye(1),
        t_min=0.0, iso_R=True, diag_C=True)


@pytest.mark.parametrize("build", [
    {"r_max": 1.0, "n_grid_r": 3},
    {},
], ids=["full", "lazy"])
def test_a_subclass_keeps_the_one_argument_resolve_c_method(build):
    """An override from before ``positions`` raised ``TypeError`` on a full
    table and was skipped with no signal on a lazy one."""
    calls = []

    class Pinned(PropagatorCache):
        def resolve_c_method(self, t_max):
            calls.append(t_max)
            return "gauss_legendre", 30

    cache = Pinned(_one_rate_model(), homogeneity="translation")
    cache.precompute_C_table_translation(4.0, 6, **build)
    assert calls == [4.0]
    assert cache.c_method_resolved == "gauss_legendre"
    assert cache.n_gauss_resolved == 30


def test_complex_closed_form_is_refused():
    """C is real.  One route used to keep the real part with a ComplexWarning
    and the other the complex value."""

    def c_fn(_n1, _t1, _n2, _t2):
        return np.array([[1.0 + 2.0j]])

    model = PropagatorModel(
        R_time=lambda t, s: float(np.exp(-(t - s))),
        kappa2=lambda _n1, _t1, _n2, _t2: np.eye(1),
        n_components=1,
    )
    cache = PropagatorCache(model, c_value_fn=c_fn)
    with pytest.raises(ValueError, match="must be real"):
        cache.C_value(0.0, 1.0, 0.0, 1.0)


def test_a_cache_pickled_by_an_earlier_release_is_refused_on_load():
    """The first read raised a bare ``AttributeError``."""
    cache = PropagatorCache(_general_system(_WidthByPosition()).build_propagator_model(),
                            homogeneity="general")
    state = cache.__getstate__()
    del state["_table_horizon"]
    with pytest.raises(ValueError, match="earlier release"):
        PropagatorCache.__new__(PropagatorCache).__setstate__(state)


def test_the_rules_of_a_full_table_are_found_by_its_workers(monkeypatch):
    """Serial in the parent, they were 90 percent of a build on two workers."""
    def kernel(n1, t1, n2, t2):
        width = 0.3 + 0.1 * abs(float(n1) + float(n2))
        return np.array([[np.exp(-((t1 - t2) / width) ** 2)]])

    def table(n_jobs):
        cache = PropagatorCache(_general_system(kernel).build_propagator_model(),
                                homogeneity="general")
        cache.precompute_C_table_general(
            t_max=2.0, n_grid_t=4, x_max=1.0, n_grid_x=3, n_jobs=n_jobs)
        return cache._table_rules

    serial = table(1)
    in_parent = []
    find = PropagatorCache._find_c_rule

    def spy(self, *args):
        in_parent.append(args)
        return find(self, *args)

    monkeypatch.setattr(PropagatorCache, "_find_c_rule", spy)
    assert table(2) == serial
    assert not in_parent


@dataclass(frozen=True)
class _WidthOfPosition:
    """``exp(-|dt| / w)`` with ``w = 1`` at the origin and ``0.05`` at 1, in a
    frozen dataclass whose one field happens to be named ``temporal``."""

    temporal: float = 1.0

    def __call__(self, n1, t1, n2, t2):
        width = 1.0 if abs(float(n1)) + abs(float(n2)) < 0.5 else 0.05
        return np.eye(1) * np.exp(-abs(t1 - t2) / width)


def test_a_field_named_temporal_does_not_declare_a_separable_kernel():
    """The time shape of this kernel changes with position, so the origin's
    rule says nothing about ``x = 1``: it was 4.4 percent off here.  Reference:
    the closed form of ``C(T, T)`` written out here."""
    gamma, lam, t_end = 0.5, 20.0, 20.0
    exact = (2.0 / (lam - gamma)) * (
        -np.expm1(-2.0 * gamma * t_end) / (2.0 * gamma)
        + np.expm1(-(gamma + lam) * t_end) / (gamma + lam)
    )
    system = sw.System(
        field=sw.FieldSpec(n_components=1),
        linear=sw.DiagonalA(gamma=[gamma]),
        noise=sw.GaussianNoise(kappa2=sw.GeneralKappa2(fn=_WidthOfPosition())),
    )
    cache = PropagatorCache(system.build_propagator_model(), homogeneity="general")
    got = float(cache._C_value_direct(1.0, t_end, 1.0, t_end)[0, 0])
    assert got == pytest.approx(exact, rel=1e-10, abs=0.0)


@dataclass(frozen=True)
class _Yukawa:
    """``e^{-r} / r``, which is not finite at the origin."""

    def __call__(self, r):
        with np.errstate(divide="ignore"):
            return np.exp(-r) / np.float64(r)


@dataclass(frozen=True)
class _RaisingYukawa:
    """The same kernel through ``math``, which raises at the origin."""

    def __call__(self, r):
        return math.exp(-r) / r


def _narrow_time_integral(gamma, sigma_t, t_end):
    """``C(T, T) / κ_x`` for a Gaussian temporal kernel, as a 1-D integral."""
    value, _ = quad(
        lambda v: np.exp(-v * v / (2.0 * sigma_t**2))
        * (np.exp(-gamma * v) - np.exp(-gamma * (2.0 * t_end - v)))
        / gamma,
        0.0,
        min(t_end, 12.0 * sigma_t),
        epsabs=0.0,
        epsrel=1e-12,
        limit=500,
    )
    return value


def _singular_system(spatial, sigma_t):
    return sw.System(
        field=sw.FieldSpec(n_components=1),
        linear=sw.DiagonalA(gamma=[1.0]),
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.GaussianTemporal(lam=1.0, sigma_t=sigma_t),
            spatial=sw.CustomKernel(fn=spatial),
        )),
    )


def test_a_kernel_singular_at_the_origin_finds_its_rule_at_the_pair():
    """The origin says nothing about ``r = 1`` when C is not finite there, so
    no ``dblquad`` follows from it: that value was 1e12 too small here.
    Reference: a 1-D quadrature that shares no code with the package."""
    sigma_t, t_end = 1e-3, 30.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        props = _singular_system(_Yukawa(), sigma_t).propagators(
            t_max=t_end, n_grid_t=7, progress=False
        )
        got = props.cache._C_value_direct(0.0, t_end, 1.0, t_end)
    assert props.c_source == "quadrature:per_position"
    exact = np.exp(-1.0) * _narrow_time_integral(1.0, sigma_t, t_end)
    assert float(got[0, 0]) == pytest.approx(exact, rel=1e-9, abs=0.0)


def test_a_kernel_that_raises_at_the_origin_is_read_through_its_table():
    """The grids shared across separations need the origin, and fall back
    when it raises."""
    sigma_t, t_end = 0.3, 10.0
    props = _singular_system(_RaisingYukawa(), sigma_t).propagators(
        t_max=t_end, n_grid_t=7, progress=False
    )
    exact = np.exp(-1.0) * _narrow_time_integral(1.0, sigma_t, t_end)
    got = float(props.cache.C_value(0.0, t_end, 1.0, t_end)[0, 0])
    assert got == pytest.approx(exact, rel=1e-7, abs=0.0)


def test_a_separable_kernel_with_no_rule_at_the_origin_stays_on_dblquad(monkeypatch):
    """Where C is finite at the origin and no rule converges there, no other
    pair of a separable kernel can do better, so none repeats the search."""
    system = _demo1_like()
    monkeypatch.setattr(ev, "select_gl_node_count", lambda *a, **k: None)
    monkeypatch.setattr(ev, "select_graded_levels", lambda *a, **k: None)
    cache = PropagatorCache(system.build_propagator_model())
    with pytest.warns(UserWarning, match="dblquad"):
        assert cache._c_rule(0.0, 1.0, 5.0) == ("dblquad", None)
    assert set(cache._c_rules) == {(ev._pair_key(0.0, 0.0), 5.0)}


@dataclass(frozen=True)
class _WeakCusp:
    """A strong smooth component beside a cusped one of weight ``eps``."""

    eps: float

    def __call__(self, n1, t1, n2, t2):
        d = t1 - t2
        return np.diag([np.exp(-d * d / 0.18), self.eps * np.exp(-abs(d) / 0.3)])


@pytest.mark.parametrize("eps", [1.0, 1e-2, 1e-6, 1e-12])
def test_a_weak_cusp_is_judged_on_its_own_entry(eps):
    """A change of scale in one component does not change whether it has a
    cusp.  The answer drives the ``dblquad`` split and the integrators' split."""
    model = sw.System(
        field=sw.FieldSpec(n_components=2),
        linear=sw.DiagonalA(gamma=[1.0, 1.5]),
        noise=sw.GaussianNoise(kappa2=sw.GeneralKappa2(fn=_WeakCusp(eps))),
    ).build_propagator_model()
    assert ev._probe_diagonal_cusp(model, (0.0, 0.0))


def test_a_smooth_entry_beside_a_zero_one_is_not_a_cusp():
    """The other side of the boundary: entries that are smooth or zero stay
    unsplit, whatever their sizes."""
    kernel = lambda n1, t1, n2, t2: np.diag(  # noqa: E731
        [np.exp(-((t1 - t2) ** 2) / 0.18), 1e-9 * np.exp(-((t1 - t2) ** 2)), 0.0]
    )
    model = PropagatorModel(
        R_time=lambda a, b: 1.0, kappa2=kernel, n_components=3, iso_R=True
    )
    assert not ev._probe_diagonal_cusp(model, (0.0, 0.0))


@pytest.mark.parametrize("setting", ["auto", "gauss_legendre", "dblquad"])
def test_a_failed_build_leaves_no_rule_for_a_direct_cell(setting):
    """A per-build count stands for that build.  A build refused before it had
    a table used to leave it for every direct cell, at 44 percent off, on any
    setting of the instance.  Reference: nested scipy ``quad``, which shares no
    code with the package."""
    gamma, width, t_end = 0.3, 0.15, 12.0

    def kernel(n1, t1, n2, t2):
        return np.array([[np.exp(-abs(t1 - t2) / width)]])

    exact, _ = quad(
        lambda s: np.exp(-gamma * (t_end - s))
        * quad(
            lambda u: np.exp(-gamma * (t_end - u)) * np.exp(-abs(s - u) / width),
            0.0,
            t_end,
            points=[s],
            limit=400,
        )[0],
        0.0,
        t_end,
        limit=400,
    )
    model = PropagatorModel(
        R_time=lambda a, b: np.exp(-gamma * (a - b)),
        kappa2=kernel,
        n_components=1,
        iso_R=True,
    )
    cache = PropagatorCache(model, c_method=setting, n_gauss=60)
    with pytest.raises(ValueError, match="at least two"):
        cache.precompute_C_table_translation(
            t_end, n_grid_t=1, r_max=1.0, n_grid_r=3,
            c_method="gauss_legendre", n_gauss=4,
        )
    assert float(cache.C_value(0.0, t_end, 0.0, t_end)[0, 0]) == pytest.approx(
        exact, rel=1e-7, abs=0.0
    )


def _rule_value_or_warning(gamma, kernel, t_end, cells):
    """The rule ``'auto'`` finds up to ``t_end`` and C at ``cells`` under it,
    or ``None`` once a warning has said C falls back to ``dblquad``."""
    model = PropagatorModel(
        R_time=lambda a, b: np.exp(-gamma * (a - b)),
        kappa2=kernel,
        n_components=1,
        iso_R=True,
    )
    cache = PropagatorCache(model, c_method="auto")
    x = np.asarray(0.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        method, count = cache._c_rule(x, x, t_end)
    if method == "dblquad":
        assert any("dblquad" in str(w.message) for w in caught)
        return None
    return [
        cache._C_value_direct(x, t1, x, t2, method=method, n_gauss=count)[0, 0]
        for t1, t2 in cells
    ]


@pytest.mark.parametrize("width, amplitude", [(0.02, 1e4), (0.2, 1.0)])
def test_the_rule_of_c_reads_interior_times(width, amplitude):
    """A floor with a burst that the corner sees below tolerance and the middle
    of the table lies before.  The rule from the extreme cells alone was 1.94
    off at the first width and 4e-5 at the second, with no warning.  Reference:
    the floor in closed form plus a 1-D ``quad`` of the separable burst."""
    gamma, s0 = 5.0, 14.0

    def burst(s):
        return np.exp(-0.5 * ((s - s0) / width) ** 2)

    def kernel(n1, t1, n2, t2):
        return np.array([[1.0 + amplitude * burst(t1) * burst(t2)]])

    def exact(t):
        leg, _ = quad(
            lambda s: np.exp(-gamma * (t - s)) * burst(s),
            s0 - 12 * width,
            min(t, s0 + 12 * width),
            epsabs=0.0,
            epsrel=1e-12,
            limit=400,
        )
        return (1.0 - np.exp(-gamma * t)) ** 2 / gamma**2 + amplitude * leg**2

    times = (14.0, 14.3)
    values = _rule_value_or_warning(gamma, kernel, 16.0, [(t, t) for t in times])
    if values is not None:
        assert values == pytest.approx([exact(t) for t in times], rel=1e-7, abs=0.0)


def test_the_rule_of_c_reads_cells_with_the_later_second_time():
    """A ridge on one side of the lag.  The probes held ``t1 >= t2``, so the
    rule came from the side without it: 5e-2 and 1e-1 off at ``(8, 16)`` and
    ``(6, 14.5)``, with no warning.  Reference: the lag integral with an
    analytic inner integral, in ``quad``."""
    gamma, lag, width, amplitude = 5.0, 8.0, 0.05, 100.0

    def ridge(u):
        return np.exp(-0.5 * ((u - lag) / width) ** 2)

    def kernel(n1, t1, n2, t2):
        return np.array([[np.exp(-abs(t1 - t2)) + amplitude * ridge(t2 - t1)]])

    def over_lags(t1, t2, f, low, high):
        """``int du f(u) int ds exp(-gamma (t1 - s + t2 - s - u))``."""

        def inner(u):
            first, last = max(0.0, -u), min(t1, t2 - u)
            ends = np.exp(2 * gamma * last) - np.exp(2 * gamma * first)
            return np.exp(-gamma * (t1 + t2 - u)) * ends / (2 * gamma)

        return quad(
            lambda u: f(u) * inner(u),
            low,
            high,
            epsabs=0.0,
            epsrel=1e-11,
            limit=400,
        )[0]

    def exact(t1, t2):
        floor = lambda u: np.exp(-abs(u))  # noqa: E731
        return (
            over_lags(t1, t2, floor, -t1, 0.0)
            + over_lags(t1, t2, floor, 0.0, t2)
            + amplitude
            * over_lags(t1, t2, ridge, lag - 12 * width, lag + 12 * width)
        )

    cells = [(8.0, 16.0), (6.0, 14.5)]
    values = _rule_value_or_warning(gamma, kernel, 16.0, cells)
    if values is not None:
        assert values == pytest.approx([exact(*c) for c in cells], rel=1e-7, abs=0.0)


def _read_in_both_orders(kernel, near, far):
    """``C(far) / C(near)`` at ``t = 1.5`` from a lazy table, with each pair of
    positions read first in turn."""
    ratios = []
    for pairs in ((near, far), (far, near)):
        system = sw.System(
            field=sw.FieldSpec("phi", 1),
            linear=sw.DiagonalA(gamma=[0.7]),
            noise=sw.GaussianNoise(kappa2=kernel),
        )
        cache = system.propagators(t_max=2.0, n_grid_t=11, progress=False).cache
        for x, y in pairs:
            cache.C_value(x, 1.5, y, 1.5)
        at_far, at_near = (cache.C_value(x, 1.5, y, 1.5)[0, 0] for x, y in (far, near))
        ratios.append(at_far / at_near)
    return ratios


class _MovingPattern:
    """``κ² = exp(-(x - y - v (t - t'))² / 2)``: translation invariant, and even
    in ``x - y`` only at ``v = 0``."""

    def __init__(self, speed):
        self.speed = speed

    def __call__(self, n1, t1, n2, t2):
        lag = float(np.ravel(n1)[0]) - float(np.ravel(n2)[0]) - self.speed * (t1 - t2)
        return math.exp(-0.5 * lag ** 2) * np.eye(1)


def _moving_pattern(speed):
    return sw.System(
        field=sw.FieldSpec("phi", n_components=1), linear=sw.DiagonalA(gamma=[0.6]),
        noise=sw.GaussianNoise(kappa2=sw.GeneralKappa2(fn=_MovingPattern(speed))))


@pytest.mark.parametrize("grid", [{}, {"r_max": 2.0, "n_grid_r": 9}], ids=["lazy", "full"])
def test_a_translation_table_refuses_noise_not_even_in_the_separation(grid):
    """The table is keyed on ``|x - y|``.  It returned 0.889096 at ``(1, 0)``
    against 0.437489 from dblquad, a factor 2.03, and order 2 of ``<phi phi>``
    came out 19 % high, with no error."""
    with pytest.raises(ValueError, match="homogeneity='general'"):
        props = _moving_pattern(1.5).propagators(
            4.0, 9, homogeneity="translation", progress=False, **grid)
        props.cache.C_at_batch(np.array([2.0]), np.array([3.0]), 1.0, 0.0)


@pytest.mark.parametrize("x, y", [(1.0, 0.0), (0.0, 1.0)])
def test_a_translation_table_takes_noise_even_in_the_separation(x, y):
    from scipy.integrate import dblquad

    ref = dblquad(
        lambda s2, s1: math.exp(-0.6 * (5.0 - s1 - s2) - 0.5 * (x - y) ** 2),
        0.0, 2.0, 0.0, 3.0, epsabs=1e-10, epsrel=1e-10)[0]
    props = _moving_pattern(0.0).propagators(
        4.0, 41, homogeneity="translation", progress=False)
    got = np.ravel(props.cache.C_at_batch(np.array([2.0]), np.array([3.0]), x, y))[0]
    assert got == pytest.approx(ref, rel=1e-5, abs=0.0)


@pytest.mark.parametrize(
    "near, far, sigma",
    [
        ((0.0, 0.0), (0.0, 7e-14), 9e-14),
        ((0.0, 1.02e-9), (0.0, 1.04e-9), 1.5e-10),
        ((1.0, 1.0), (1.0, 1.0 + 1e-13), 1e-13),
    ],
    ids=["below-the-old-key", "one-old-bin", "at-coordinate-1"],
)
def test_lazy_tables_tell_small_separations_apart(near, far, sigma):
    """A key of 10 decimals put these separations on one table, and the value
    of both was that of the one read first: ratio 1.0 against 0.739, 0.400,
    0.607.  Reference: the Gaussian envelope, a factor of a separable C."""
    kernel = sw.SeparableTranslation(
        temporal=sw.GaussianTemporal(lam=0.8, sigma_t=0.4),
        spatial=sw.GaussianSpatial(sigma_x=sigma),
    )
    r_near, r_far = abs(near[1] - near[0]), abs(far[1] - far[0])
    expected = np.exp(-0.5 * ((r_far / sigma) ** 2 - (r_near / sigma) ** 2))
    assert _read_in_both_orders(kernel, near, far) == pytest.approx(
        [expected, expected], rel=1e-9, abs=0.0
    )


@dataclass(frozen=True)
class _RidgeAtSeparation:
    """A ridge of width ``width`` at a separation of 1."""

    width: float

    def __call__(self, r):
        return float(np.exp(-0.5 * ((r - 1.0) / self.width) ** 2))


def test_a_lazy_key_holds_13_digits_of_a_separation():
    """Separations of 1 and ``1 + 1.5e-10`` differ in the tenth digit.  The
    Gaussian envelope gives the ratio."""
    width = 1e-10
    kernel = sw.SeparableTranslation(
        temporal=sw.GaussianTemporal(lam=0.8, sigma_t=0.4),
        spatial=sw.CustomKernel(fn=_RidgeAtSeparation(width)),
    )
    near, far = (0.0, 1.0), (0.0, 1.0 + 1.5 * width)
    expected = np.exp(-0.5 * ((far[1] - 1.0) / width) ** 2)
    assert _read_in_both_orders(kernel, near, far) == pytest.approx(
        [expected, expected], rel=1e-3, abs=0.0
    )


@dataclass(frozen=True)
class _NarrowInAngle:
    """``exp(-(1 - pole cos) / width**2)``, narrow about ``cos = pole``."""

    pole: float
    width: float

    def __call__(self, cos_theta):
        return float(np.exp(-(1.0 - self.pole * cos_theta) / self.width**2))


@pytest.mark.parametrize("pole, angle", [(1.0, 3e-6), (-1.0, 3e-6), (-1.0, 1e-6)])
def test_lazy_tables_tell_small_angles_apart(pole, angle):
    """A cosine rounded to 10 decimals cannot see an angle below 1e-5, in any
    units, at either pole: ratio 1.0 against 0.755.  Reference: the envelope at
    ``1 - pole cos = angle**2 / 2``, a factor of a separable C.  At pole -1 and
    an angle below 1.4e-6 the distance from the pole rounds to 2 at 13 digits,
    and only the distance from the antipode tells the pairs apart."""
    width = 4 / 3 * angle
    kernel = sw.SeparableRotation(
        temporal=sw.GaussianTemporal(lam=0.8, sigma_t=0.4),
        angular=sw.CustomKernel(fn=_NarrowInAngle(pole, width)),
    )
    north = np.array([0.0, 0.0, 1.0])
    near = (north, pole * north)
    far = (north, pole * np.array([np.sin(angle), 0.0, np.cos(angle)]))
    expected = np.exp(-0.5 * (angle / width) ** 2)
    assert _read_in_both_orders(kernel, near, far) == pytest.approx(
        [expected, expected], rel=1e-3, abs=0.0
    )


@pytest.mark.parametrize("a", [1.0, 1e-9, 1e-30], ids=str)
@pytest.mark.parametrize(
    "gamma, t, kink, width",
    [(1.0, 20.0, None, 0.3), (5.0, 60.0, 1.3, 1.0)],
    ids=["gaussian", "kinked"],
)
def test_dblquad_C_is_linear_in_the_amplitude_of_the_kernel(gamma, t, kink, width, a):
    """scipy's default ``epsabs`` stops the outer integral on its first samples
    when C is small: 1.7e-2 off at a = 1e-9 on the Gaussian and 4.8e-4 on the
    kinked kernel, while every other rule is scale-free.  The reference is
    ``(1/g) int_0^T kappa(d) (exp(-g d) - exp(-g (2T - d))) dd`` by panels of
    Gauss-Legendre, which imports nothing from sft-wick."""

    def shape(d):
        ramp = 0.0 if kink is None else 0.5 * np.maximum(d - kink, 0.0) * np.exp(-d)
        return np.exp(-((d / width) ** 2)) + ramp

    nodes, weights = np.polynomial.legendre.leggauss(40)
    edges = sorted({0.0, *(k * width for k in (1, 3, 6)), (kink or 0.0), t})
    edges = np.concatenate([np.linspace(lo, hi, 9)[:-1] for lo, hi in zip(edges, edges[1:])] + [[t]])
    reference = 0.0
    for lo, hi in zip(edges, edges[1:]):
        d = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
        decay = np.exp(-gamma * d) - np.exp(-gamma * (2.0 * t - d))
        reference += 0.5 * (hi - lo) * np.sum(weights * shape(d) * decay) / gamma

    model = PropagatorModel(
        R_time=lambda t1, s: np.exp(-gamma * (t1 - s)),
        kappa2=lambda n1, t1, n2, t2: a * shape(abs(t1 - t2)) * np.eye(1),
        n_components=1,
        diag_C=True,
    )
    cache = PropagatorCache(model, c_method="dblquad")
    assert cache._C_value_direct(0.0, t, 0.0, t)[0, 0] / a == pytest.approx(
        reference, rel=1e-6, abs=0.0
    )


@pytest.mark.parametrize("t_min", [0.0, 1e9, 1e12], ids=str)
def test_a_time_only_table_reads_a_lag_off_the_diagonal_at_any_origin(t_min):
    """The tolerance was ``1e-9 max(1, |t|)``, so at t = 1e9 a lag of 0.9 read
    the diagonal line at the mean time: 0.33 against 0.22.  A lag of 0 stays on
    the line."""
    from sft_wick.workflow.closed_forms import ou_exponential_phi

    gamma, sigma = 1.0, 0.5

    def closed_form(n1, t1, n2, t2):
        return np.array(
            [[float(ou_exponential_phi(gamma, 1 / sigma, t1 - t_min, t2 - t_min))]]
        )

    model = PropagatorModel(
        R_time=lambda t, s: np.exp(-gamma * (t - s)),
        kappa2=lambda n1, t1, n2, t2: np.exp(-abs(t1 - t2) / sigma) * np.eye(1),
        n_components=1,
        diag_C=True,
        t_min=t_min,
    )
    cache = PropagatorCache(model, c_value_fn=closed_form)
    cache.precompute_C_table(t_min + 10.0, n_grid=60)
    for lag in (0.0, 0.9):
        t1, t2 = np.array([t_min + 6.0 + lag]), np.array([t_min + 6.0])
        assert cache.C_diagonal_batch(t1, t2)[0, 0] == pytest.approx(
            closed_form(0.0, t1[0], 0.0, t2[0])[0, 0], rel=1e-3, abs=0.0
        )


def test_equal_times_that_differ_by_rounding_stay_on_the_diagonal():
    """At ``t_min = 1e12`` one ulp of a time is above 1e-9 of the span, so
    only rounding keeps a lag of one ulp on the diagonal line.  The 2-D spline
    rounds the cusp there: 5.7e-4 off."""
    from sft_wick.workflow.closed_forms import ou_exponential_phi

    gamma, sigma, t_min = 1.0, 0.5, 1e12

    def closed_form(n1, t1, n2, t2):
        return np.array(
            [[float(ou_exponential_phi(gamma, 1 / sigma, t1 - t_min, t2 - t_min))]]
        )

    model = PropagatorModel(
        R_time=lambda t, s: np.exp(-gamma * (t - s)),
        kappa2=lambda n1, t1, n2, t2: np.exp(-abs(t1 - t2) / sigma) * np.eye(1),
        n_components=1,
        diag_C=True,
        t_min=t_min,
    )
    cache = PropagatorCache(model, c_value_fn=closed_form)
    cache.precompute_C_table(t_min + 10.0, n_grid=60)
    t = np.linspace(t_min + 1.0, t_min + 9.0, 25)
    got = np.ravel(cache.C_diagonal_batch(t + np.spacing(t_min), t))
    want = [closed_form(0.0, s, 0.0, s)[0, 0] for s in t]
    assert got == pytest.approx(want, rel=1e-4, abs=0.0)


def test_an_explicit_method_after_auto_is_the_method_of_that_build():
    """The horizon of the first build stayed, so the second took the rule of
    ``'auto'``: 0.478 where four nodes give 0.268."""

    def kernel(n1, t1, n2, t2):
        return np.array([[np.exp(-abs(t1 - t2) / 0.15)]])

    model = PropagatorModel(
        R_time=lambda a, b: np.exp(-0.3 * (a - b)),
        kappa2=kernel,
        n_components=1,
        iso_R=True,
    )
    cache = PropagatorCache(model, homogeneity="translation")
    build = dict(n_grid_t=5, r_max=1.0, n_grid_r=3)
    cache.precompute_C_table_translation(12.0, **build)
    cache.precompute_C_table_translation(
        12.0, **build, c_method="gauss_legendre", n_gauss=4
    )
    assert cache._table_kwargs(np.asarray(0.0), np.asarray(1.0), 12.0) == {
        "method": "gauss_legendre",
        "n_gauss": 4,
    }


def test_white_noise_makes_the_time_shape_of_c_depend_on_position():
    """A separable κ² has one time shape at every position, and a σ² beside it
    does not.  The origin's 20 nodes read at a pulse of σ² of width 0.2, where
    κ² has vanished, gave 4e-10 against 2.5e-4, with no warning."""

    def pulse(t):
        return np.exp(-0.5 * ((t - 17.3) / 0.2) ** 2) * np.eye(1)

    kernel = sw.SeparableTranslation(
        temporal=sw.GaussianTemporal(lam=1.0, sigma_t=1.0),
        spatial=sw.GaussianSpatial(sigma_x=0.1),
    )
    for sigma2, depends in ((None, False), (sw.CustomImpulse(fn=pulse), True)):
        system = sw.System(
            field=sw.FieldSpec("phi", 1),
            linear=sw.DiagonalA(gamma=[0.3]),
            noise=sw.GaussianNoise(kappa2=kernel, sigma2=sigma2),
        )
        cache = PropagatorCache(system.build_propagator_model(), c_method="auto")
        assert cache._c_time_shape_depends_on_position() is depends


@pytest.mark.parametrize("origin", [0.0, 1e10], ids=str)
@pytest.mark.parametrize("homogeneity", ["translation", "general"])
def test_a_lazy_table_tells_ulps_of_large_positions_apart(origin, homogeneity):
    """A floor of 8 eps max(|x|, |y|) on the key put separations up to 9 ulps on
    the r = 0 table, so the value depended on the origin of the positions and
    on which pair was read first."""
    gamma, sigma, width = 1.0, 0.5, 2e-6

    def kernel(n1, t1, n2, t2):
        r = abs(float(np.ravel(n1)[0]) - float(np.ravel(n2)[0]))
        return np.exp(-abs(t1 - t2) / sigma - r * r / (2 * width**2)) * np.eye(1)

    model = PropagatorModel(
        R_time=lambda t, s: np.exp(-gamma * (t - s)),
        kappa2=kernel,
        n_components=1,
        diag_C=True,
    )
    cache = PropagatorCache(
        model, homogeneity=homogeneity, c_method="gauss_legendre", n_gauss=20
    )
    build = getattr(cache, f"precompute_C_table_{homogeneity}")
    build(t_max=2.0, n_grid_t=8)
    x, y = origin, origin + 4 * np.spacing(1e10)
    t = np.array([2.0])
    separated = cache.C_at_batch(t, t, x, y)[0, 0]
    coincident = cache.C_at_batch(t, t, x, x)[0, 0]
    assert separated / coincident == pytest.approx(
        np.exp(-((y - x) ** 2) / (2 * width**2)), rel=1e-9, abs=0.0
    )
