"""Gauss-Legendre: the domain is split at C's diagonal kink.

White noise makes ``C(t1, t2) = ∫ R σ² R`` kinked on ``t1 = t2``, and a
κ² with a ``|Δt|`` cusp makes ``C = ∫∫ R κ² R`` kinked there too, in its
third derivative (``∂²C/∂t1∂t2 = R κ² R``).  A C propagator between two
internal times that the diagram leaves unordered puts that kink inside the
integration domain, and tensor-product Gauss-Legendre then converges
algebraically (``n^-2`` under white noise: a 2-D order-2 channel was
7.9e-5 off at 64 nodes).  ``integrate_moment_gauss_legendre`` integrates
each consistent order of such pairs as its own causal sub-simplex.

Reference: the exact Itô moment hierarchy of the same system
(``examples/reference/ito_moments.py``), which shares no code with the
package.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.evaluate import (PropagatorCache, PropagatorModel, _c_has_diagonal_kink,
                               _kink_orientations, _kink_pairs,
                               _probe_c_diagonal_kink, _probe_diagonal_cusp)


def _c_kink_pairs(spatial):
    return _kink_pairs(spatial, c_kink=True)
from sft_wick.workflow.specs import ConstantImpulse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "examples" / "reference"))
import ito_moments as im  # noqa: E402


def _spatial(orderings, c_pairs):
    return SimpleNamespace(
        equal_time_aliases=(), time_integration_vars=("a", "b", "c"),
        time_orderings=list(orderings),
        c_propagators=[(l, r, None, None) for l, r in c_pairs])


def test_three_unordered_times_give_the_six_orders():
    sp = _spatial([], [("a", "b"), ("b", "c"), ("a", "c")])
    pairs = _c_kink_pairs(sp)
    assert len(pairs) == 3
    assert len(_kink_orientations(sp, pairs)) == 6


def test_an_existing_order_is_respected():
    """With a < b already imposed, the consistent total orders of three
    times are the 3 linear extensions; pairs already ordered through the
    causal structure are not split."""
    sp = _spatial([("a", "b")], [("a", "b"), ("b", "c"), ("a", "c")])
    pairs = _c_kink_pairs(sp)
    assert ("a", "b") not in pairs and len(pairs) == 2
    assert len(_kink_orientations(sp, pairs)) == 3


def test_a_tadpole_is_not_a_kink_pair():
    sp = _spatial([], [("a", "a"), ("b", "b")])
    assert _c_kink_pairs(sp) == []


def test_two_unordered_parents_are_a_kink_pair_without_white_noise():
    """A vertex with several ψ legs at one time has several parents, and its
    upper bound ``min(parents)`` is kinked where two unordered ones cross.
    That does not depend on the noise."""
    sp = _spatial([("c", "a"), ("c", "b")], [])
    assert _kink_pairs(sp, c_kink=False) == [("a", "b")]
    assert len(_kink_orientations(sp, [("a", "b")])) == 2
    ordered = _spatial([("c", "a"), ("c", "b"), ("b", "a")], [])
    assert _kink_pairs(ordered, c_kink=False) == []


N = 2
GAMMA, T_FINAL = 1.0, 1.8
S2 = np.array([[0.8, 0.3], [0.3, 0.5]])
F = np.array([[[0.25, -0.40], [0.15, 0.30]],
              [[-0.35, 0.20], [0.45, -0.10]]])


def _system(sigma2):
    return sw.System(
        field=sw.FieldSpec("phi", N), linear=sw.DiagonalA(gamma=[GAMMA] * N),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=1.0),
                spatial=sw.ExponentialSpatial(sigma_x=1.0)),
            sigma2=sigma2))


@dataclass(frozen=True)
class _Exp:
    """``λ e^{−|Δt|/σ_t}`` as a user callable: cusped, and declaring
    nothing."""

    sigma_t: float

    def __call__(self, dt) -> float:
        return float(np.exp(-abs(dt) / self.sigma_t))


@dataclass(frozen=True)
class _Gauss:
    """``e^{−Δt²/2σ_t²}`` as a user callable: smooth at ``Δt = 0``."""

    sigma_t: float

    def __call__(self, dt) -> float:
        return float(np.exp(-0.5 * (dt / self.sigma_t) ** 2))


def _colored(temporal):
    return sw.System(
        field=sw.FieldSpec("phi", N), linear=sw.DiagonalA(gamma=[GAMMA] * N),
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=temporal,
            spatial=sw.ExponentialSpatial(sigma_x=1.0))))


def test_kink_detection_follows_white_noise_and_the_kernel_cusp():
    """White noise jumps C's first derivative, a ``|Δt|`` cusp in κ² its
    third (``∂²C/∂t1∂t2 = R κ² R``); a Gaussian κ² leaves C smooth."""
    white = _system(ConstantImpulse(S2)).propagators(
        t_max=2.0, c_closed_form="auto", c_closed_form_only=True,
        diag_C=False, progress=False)
    exponential = _colored(sw.ExponentialTemporal(lam=0.5, sigma_t=0.5)
                           ).propagators(t_max=2.0, n_grid_t=8,
                                         c_closed_form=None, progress=False)
    gaussian = _colored(sw.GaussianTemporal(lam=0.5, sigma_t=0.5)
                        ).propagators(t_max=2.0, n_grid_t=8,
                                      c_closed_form=None, progress=False)
    assert _c_has_diagonal_kink(white.cache)
    assert _c_has_diagonal_kink(exponential.cache)
    assert not _c_has_diagonal_kink(gaussian.cache)


def test_the_builtin_closed_form_declares_the_kink():
    """``builtin_closed_form_for`` exists only for the exponential temporal
    kernel, so the closed form it returns carries the cusp."""
    colored = _colored(sw.ExponentialTemporal(lam=0.5, sigma_t=0.5)
                       ).propagators(t_max=2.0, c_closed_form="auto",
                                     c_closed_form_only=True, progress=False)
    assert _c_has_diagonal_kink(colored.cache)


def test_a_custom_kernel_is_probed():
    """A callable κ² declares nothing, so the cusp probe decides: the same
    two kernels, written as ``CustomKernel``."""
    cusped = _colored(sw.CustomKernel(fn=_Exp(0.5))).propagators(
        t_max=2.0, n_grid_t=8, c_closed_form=None, progress=False)
    smooth = _colored(sw.CustomKernel(fn=_Gauss(0.5))).propagators(
        t_max=2.0, n_grid_t=8, c_closed_form=None, progress=False)
    assert _c_has_diagonal_kink(cusped.cache)
    assert not _c_has_diagonal_kink(smooth.cache)


def _hierarchy_ff(a, b, gamma=GAMMA, t_final=T_FINAL):
    """``⟨φ_a φ_b⟩`` at O(F²) for white noise ``S2`` shared by the two
    observation points (``ConstantImpulse`` is position independent, so
    the noise is the same at x and y)."""
    D = 2 * N                                   # (component, point)
    idx = {(c, i): c * 2 + i for c in range(N) for i in range(2)}
    sde = im.PolySDE(D=D, n_tags=1)
    sde.add_linear_drift(-gamma * np.eye(D))
    S = np.zeros((D, D))
    Q = np.zeros((D, D, D))
    for (c, i), k in idx.items():
        for (d, j), l in idx.items():
            S[k, l] = S2[c, d]
        for d in range(N):
            for e in range(N):
                Q[k, idx[(d, i)], idx[(e, i)]] += F[c, d, e]
    sde.add_constant_diffusion(S)
    sde.add_quadratic_drift(Q, tag=(1,))
    mono = im.unit(D, idx[(a, 0)], idx[(b, 1)])
    return im.solve(sde, [(mono, (2,))], [t_final])[(mono, (2,))][0]


@pytest.mark.parametrize("ab", [(0, 1), (1, 1)])
def test_white_noise_order_2_converges_exponentially(ab):
    system = _system(ConstantImpulse(S2))
    props = system.propagators(t_max=2.0, c_closed_form="auto",
                               c_closed_form_only=True, diag_C=False,
                               progress=False)
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2], diag_C=False)
    ref = _hierarchy_ff(*ab)
    got = exp.evaluate(props, positions={"x": 0.0, "y": 0.7},
                       t_final=T_FINAL, component_pair=ab, orders=[2],
                       method="gauss_legendre", n_gauss=16).total
    assert got == pytest.approx(ref, rel=1e-11, abs=0.0)


class _WhiteC:
    """C of an OU process under white noise ``S2``, ``t_min = 0``, as a user
    closed form: a ``|t1 - t2|`` cusp, and scalar or batched."""

    def __init__(self, gamma, kink=None):
        self.gamma = gamma
        if kink is not None:
            self.has_diagonal_kink = kink

    def __call__(self, x1, t1, x2, t2):
        g = self.gamma
        decay = np.exp(-g * np.abs(t1 - t2)) - np.exp(-g * (t1 + t2))
        return np.multiply.outer(decay / (2.0 * g), S2)


def _closed_form_run(c_fn, gamma, t_final, method, vectorised=False, **kw):
    """Order 2 from ``c_fn`` on a model whose κ² is smooth and unread."""
    system = sw.System(
        field=sw.FieldSpec("phi", N), linear=sw.DiagonalA(gamma=[gamma] * N),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.GaussianTemporal(lam=1.0, sigma_t=0.5),
            spatial=sw.ExponentialSpatial(sigma_x=1.0))))
    props = system.propagators(
        t_max=t_final, c_closed_form=c_fn, c_closed_form_only=True,
        c_closed_form_vectorized=vectorised, diag_C=False, progress=False)
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2], diag_C=False)
    return exp.evaluate(props, positions={"x": 0.0, "y": 0.7},
                        t_final=t_final, component_pair=(0, 1), orders=[2],
                        method=method, **kw).total


@pytest.mark.parametrize("gamma, t_final, vectorised", [
    (0.01, 1.8, False), (1.0, 0.05, True), (1.0, 1.8, True), (1.0, 30.0, False),
])
def test_an_undeclared_closed_form_is_split_by_gauss_legendre(
        gamma, t_final, vectorised):
    """The ``|t1 - t2|`` cusp is found with nothing declared.  Unsplit, the
    order-2 value is 1.8e-4 to 9.5e-3 off at 24 nodes."""
    ref = _hierarchy_ff(0, 1, gamma, t_final)
    got = _closed_form_run(_WhiteC(gamma), gamma, t_final, "gauss_legendre",
                           vectorised, n_gauss=24)
    assert got == pytest.approx(ref, rel=1e-9, abs=0.0)
    unsplit = _closed_form_run(_WhiteC(gamma, kink=False), gamma, t_final,
                               "gauss_legendre", vectorised, n_gauss=24)
    assert abs(unsplit / ref - 1.0) > 1e-4


def test_an_undeclared_closed_form_is_split_by_nquad():
    """1.3e-7 off unsplit, at the default tolerance."""
    ref = _hierarchy_ff(0, 1)
    got = _closed_form_run(_WhiteC(GAMMA), GAMMA, T_FINAL, "nquad")
    assert got == pytest.approx(ref, rel=1e-10, abs=0.0)


def test_a_closed_form_does_not_read_the_kernel_it_replaces():
    """A smooth closed form on a cusped κ², and a closed form that answers
    for itself, whatever the model says."""
    def smooth(n1, t1, n2, t2):
        return np.exp(-0.5 * (t1 - t2) ** 2) * np.eye(N)

    class Declared:
        has_diagonal_kink = False

        def __call__(self, n1, t1, n2, t2):
            return np.exp(-abs(t1 - t2)) * np.eye(N)

    system = _colored(sw.ExponentialTemporal(lam=0.5, sigma_t=0.5))
    for fn in (smooth, Declared()):
        props = system.propagators(t_max=2.0, c_closed_form=fn,
                                   c_closed_form_only=True, progress=False)
        assert not _c_has_diagonal_kink(props.cache), fn


def _time_limited_c(tau_noise):
    """C of ``φ' = -φ + η``, ``⟨ηη⟩ = δ(s - s') exp(-(s / τ)²)``: its cusp on
    the diagonal dies out at about ``τ``."""
    from scipy.special import erf

    def fn(n1, t1, n2, t2):
        t1, t2 = np.asarray(t1, dtype=float), np.asarray(t2, dtype=float)
        b = tau_noise
        limit = erf(np.minimum(t1, t2) / tau_noise - b) + erf(b)
        c = np.exp(b * b - t1 - t2) * tau_noise * np.sqrt(np.pi) / 2 * limit
        return c[..., None, None]
    return fn


def _picard_order_2(c, t_final, tau_noise, f):
    """O(F²) of ``⟨φ(T)²⟩`` for ``φ' = -φ + F φ² + η``, from the second
    Picard iterate, panelled by hand.  No package code."""
    from numpy.polynomial.legendre import leggauss

    x, w = leggauss(20)

    def rule(cuts):
        a, b = np.asarray(cuts[:-1]), np.asarray(cuts[1:])
        return ((0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * x).ravel(),
                (0.5 * (b - a)[:, None] * w).ravel())

    def inner(s):
        graded = s - s * np.geomspace(1e-7, 1.0, 40)
        sp, wp = rule(np.unique(np.r_[0.0, graded, s]))
        return np.sum(wp * (
            2 * np.exp(s + sp - 2 * t_final) * (
                c(s, s) * c(sp, sp) + 2 * c(s, sp) ** 2)
            + 4 * np.exp(sp - t_final) * (
                c(t_final, s) * c(sp, sp) + 2 * c(t_final, sp) * c(s, sp))))

    cuts = np.linspace(0.0, t_final, 41)
    fine = np.geomspace(1e-7, 1.0, 30) * min(tau_noise, t_final)
    outer, wo = rule(np.unique(np.r_[cuts, fine, t_final - fine]))
    return f ** 2 * sum(ws * inner(s) for s, ws in zip(outer, wo))


@pytest.mark.parametrize("sigma2", [False, True])
@pytest.mark.parametrize("t_final, tau_noise", [(4.0, 0.6), (30.0, 3.0)])
def test_a_cusp_that_dies_out_is_split(t_final, tau_noise, sigma2):
    """Noise limited to the start of the range, undeclared, with or without
    the ``σ²`` on the model.  It was 5.1e-4 and 1.8e-2 off at 24 nodes."""
    fn = _time_limited_c(tau_noise)
    f = 0.6
    system = sw.System(
        field=sw.FieldSpec("phi", 1), linear=sw.DiagonalA(gamma=[1.0]),
        vertices=[sw.LocalVertex("F", coupling=np.full((1, 1, 1), f))],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.GaussianTemporal(lam=0.0, sigma_t=0.5),
                spatial=sw.ExponentialSpatial(sigma_x=1.0)),
            sigma2=sw.CustomImpulse(
                lambda t: np.array([[np.exp(-(t / tau_noise) ** 2)]]))
            if sigma2 else None))
    props = system.propagators(
        t_max=t_final, c_closed_form=fn, c_closed_form_only=True,
        c_closed_form_vectorized=True, diag_C=False, progress=False)
    got = system.expand(("phi(x)", "phi(y)"), orders=[2], diag_C=False).evaluate(
        props, positions={"x": 0.0, "y": 0.0}, t_final=t_final, orders=[2],
        method="gauss_legendre", n_gauss=48).total

    def scalar(t1, t2):
        return fn(0.0, t1, 0.0, t2)[..., 0, 0]

    ref = _picard_order_2(scalar, t_final, tau_noise, f)
    assert got == pytest.approx(ref, rel=1e-9, abs=0.0)


def test_a_closed_form_on_a_white_noise_model_is_split_unless_it_declares():
    """A pulse of noise can sit between the probe's times, so the ``σ²`` of
    the model decides for an undeclared closed form."""
    class Quiet:
        has_diagonal_kink = False

        def __call__(self, n1, t1, n2, t2):
            return np.exp(-(t1 - t2) ** 2) * np.eye(N)

    system = _system(ConstantImpulse(S2))
    for fn, split in ((lambda *a: Quiet()(*a), True), (Quiet(), False)):
        props = system.propagators(t_max=2.0, c_closed_form=fn,
                                   c_closed_form_only=True, diag_C=False,
                                   progress=False)
        assert _c_has_diagonal_kink(props.cache) is split


class _CuspApart:
    """A cusp on the time diagonal at ``x1 != x2``, none at ``x1 = x2``."""

    def __call__(self, n1, t1, n2, t2):
        cusp = 0.0 if float(n1) == float(n2) else abs(t1 - t2)
        return float(np.exp(-(t1 - t2) ** 2 - cusp)) * np.eye(1)


def _order_2_term():
    system = sw.System(
        field=sw.FieldSpec("phi", 1), linear=sw.DiagonalA(gamma=[1.0]),
        vertices=[sw.LocalVertex("F", coupling=np.full((1, 1, 1), 0.5))],
        noise=_colored(sw.GaussianTemporal(lam=0.5, sigma_t=0.5)).noise)

    def joins_x_and_y(term):
        spatial = term.analyze_spatial()
        at = spatial.direction_map
        return any(at[left] != at[right]
                   for left, right, *_ in spatial.c_propagators)

    expansion = system.expand(("phi(x)", "phi(y)"), orders=[2], diag_C=False)
    return system, next(filter(joins_x_and_y, expansion.diagrams(2)))


@pytest.mark.parametrize("table, x, y, split", [
    ("spatial", 0.0, 0.5, True), ("spatial", 0.5, 0.5, False),
    ("no table", 0.0, 0.5, True), ("no table", 0.5, 0.5, False),
    ("time only", 0.0, 0.5, False),
])
def test_the_probe_reads_c_at_the_positions_the_integrators_read(
        table, x, y, split):
    """A time-only table holds C at its own position, so ``x`` and ``y`` do
    not reach it.  A cache with no table reads C at the points of the diagram
    (``C_value``), where the probe used to read the ambient ``direction``."""
    system, term = _order_2_term()
    if table == "spatial":
        cache = system.propagators(
            t_max=2.0, c_closed_form=_CuspApart(), c_closed_form_only=True,
            progress=False).cache
    else:
        cache = PropagatorCache(system.build_propagator_model(),
                                c_value_fn=_CuspApart())
    if table == "time only":
        cache.precompute_C_table(2.0, n_grid=6)
    ig = term.build_integrand(system.build_coupling_values())
    assert ig._c_kink_at(cache, 0.0, 2.0, {"x": x, "y": y}, 0.0) is split


def test_the_kink_answer_does_not_depend_on_the_ambient_direction():
    """``C ∝ x1 x2`` vanishes at ``direction = 0``, where the probe used to
    read it, so ``nquad`` ran unsplit at ``x = y = 1``: 9.4e-07 off here, under
    an error estimate of 1.5e-08.  The reference is the closed form of the
    double integral, whose weight 2 is that of the diagram."""
    gamma, decay, rate, t_final = 1.3, 0.2, 1.0, 10.0

    def closed_form(n1, t1, n2, t2):
        scale = float(np.ravel(n1)[0]) * float(np.ravel(n2)[0])
        return np.array(
            [[scale * np.exp(-rate * abs(t1 - t2) - decay * (t1 + t2))]])

    model = PropagatorModel(
        R_time=lambda t, s: np.exp(-gamma * (t - s)),
        kappa2=lambda *_: np.zeros((1, 1)), n_components=1, diag_C=True)
    phi, psi = sw.Field("phi", "physical"), sw.Field("psi", "response")
    terms = sw.compute_moment(
        [phi("x"), phi("y")], sw.Action([sw.Vertex(fields=[psi, phi, phi], coupling="g")]),
        order=2, response_phase=True, collect_topology=True, diag_R=True, diag_C=True,
        iso_R=True, iso_C=True).diagram_terms(2)
    coupling = {"g": np.array(-1j)}
    ig = next(ig for ig in (t.build_integrand(coupling) for t in terms)
              if [c[:2] for c in ig.spatial.c_propagators] == [("y_0", "y_1")] * 2)
    got, _ = ig.integrate_moment_nquad(
        t_final, PropagatorCache(model, c_value_fn=closed_form),
        positions={"x": 1.0, "y": 1.0})

    sum_rate = 2.0 * (gamma - 2.0 * decay)
    above, below = sum_rate / 2.0 + 2.0 * rate, sum_rate / 2.0 - 2.0 * rate
    ref = 2.0 * 2.0 * np.exp(-2.0 * gamma * t_final) / above * (
        np.expm1(sum_rate * t_final) / sum_rate
        - np.expm1(below * t_final) / below)
    assert got == pytest.approx(ref, rel=1e-12, abs=0.0)


class _NarrowC:
    """``e^{-0.3 (t1 + t2)}`` times a cusp or a ridge of ``width`` on the
    diagonal of a 2 by 2 C, declaring nothing."""

    def __init__(self, shape, width):
        self.shape, self.width = shape, width

    def __call__(self, n1, t1, n2, t2):
        t1, t2 = np.asarray(t1, dtype=float), np.asarray(t2, dtype=float)
        lag = (t1 - t2) / self.width
        feature = np.exp(-np.abs(lag)) if self.shape == "cusp" else np.exp(-0.5 * lag ** 2)
        return np.asarray(np.exp(-0.3 * (t1 + t2)) * feature)[..., None, None] * np.eye(2)


@pytest.mark.parametrize("shape", ["cusp", "ridge"])
@pytest.mark.parametrize("width", [1e-5, 1e-7, 1e-9])
def test_a_feature_on_the_diagonal_is_split_at_any_width(shape, width):
    """An unsplit tensor rule has nodes on ``t1 = t2`` and reads the feature
    at full height: 15 % at 16 nodes and halving with each doubling, with no
    error, below a width of 1e-6.  As the width falls only ``C(s, s) C(s', s')``
    survives, which gives the reference."""
    gamma, t_final = 0.8, 3.0
    F = np.zeros((2, 2, 2))
    F[0, 0, 0] = 0.5
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[gamma, gamma]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=_colored(sw.ExponentialTemporal(lam=1.0, sigma_t=1.0)).noise)
    props = system.propagators(
        t_max=t_final, c_closed_form=_NarrowC(shape, width), c_closed_form_only=True,
        c_closed_form_vectorized=True, progress=False)
    got = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2]).evaluate(
        props, positions={"x": 0.0, "y": 0.0}, t_final=t_final, component_pair=(0, 0),
        orders=[2], method="gauss_legendre", n_gauss=16).total
    source = np.exp(-gamma * t_final) * np.expm1((gamma - 0.6) * t_final) / (gamma - 0.6)
    assert got == pytest.approx((0.5 * source) ** 2, rel=1e-9, abs=0.0)


class _CuspLate:
    """A cusp on the diagonal from ``t = 8``, and at equal positions only."""

    def __call__(self, n1, t1, n2, t2):
        cusp = 0.0 if float(n1) != float(n2) or min(t1, t2) < 8.0 else abs(t1 - t2)
        return float(np.exp(-(((t1 - t2) / 10.0) ** 2) - cusp)) * np.eye(1)


def test_the_probe_is_asked_of_each_range_and_each_position():
    """The answer is kept by range and by position, and reads every C of the
    diagram: with the cusp at equal positions and from ``t = 8``, a diagram
    splits on the long range when one of its C has both ends at one point."""
    system = sw.System(
        field=sw.FieldSpec("phi", 1), linear=sw.DiagonalA(gamma=[1.0]),
        vertices=[sw.LocalVertex("F", coupling=np.full((1, 1, 1), 0.5))],
        noise=_colored(sw.GaussianTemporal(lam=0.5, sigma_t=0.5)).noise)
    cache = system.propagators(
        t_max=20.0, c_closed_form=_CuspLate(), c_closed_form_only=True,
        progress=False).cache
    couplings = system.build_coupling_values()
    apart = {"x": 0.0, "y": 0.5}
    verdicts = []
    for term in system.expand(("phi(x)", "phi(y)"), orders=[2], diag_C=False).diagrams(2):
        ig = term.build_integrand(couplings)
        at = ig.spatial.direction_map
        pairs = [(at[left], at[right]) for left, right, *_ in ig.spatial.c_propagators]
        verdicts.append(any(left == right for left, right in pairs))
        assert ig._c_kink_at(cache, 0.0, 3.0, apart, 0.0) is False
        assert ig._c_kink_at(cache, 0.0, 20.0, apart, 0.0) is verdicts[-1]
        assert ig._c_kink_at(cache, 0.0, 20.0, {"x": 0.5, "y": 0.5}, 0.0) is True
    assert any(verdicts) and not all(verdicts)


@pytest.mark.parametrize("bad", [
    pytest.param(lambda tau: 1 / 0, id="raises"),
    pytest.param(lambda tau: np.full(np.shape(tau) + (2, 2), np.nan), id="not finite"),
])
def test_a_c_that_cannot_be_read_counts_as_kinked(bad):
    assert _probe(bad, 3.0)


def test_noise_at_one_time_is_no_jump():
    """Noise gives level ratios by chance, and this seed at one time, so an
    entry must show the jump at two of the times."""
    rng = np.random.default_rng(4)

    def noisy(tau):
        return np.exp(-tau ** 2) * (1.0 + 1e-12 * rng.standard_normal(np.shape(tau)))

    assert not _probe(_entries(noisy), 3.0)


@pytest.mark.parametrize("t_final", [5.0, 30.0, 100.0, 1000.0])
def test_a_c_that_only_varies_fast_is_not_split(t_final):
    """``0.1 e^{-(t1+t2)} cosh((t1-t2)/2) + 0.01``, with nothing on the
    diagonal.  The probe split it from ``T = 30``, and the split lost
    accuracy: 2.55e-7 against 5e-12 at 16 nodes.  The late-time value is the
    fixed point of ``phi0 = z0 + g phi1^2``, ``phi1 = z1 + g phi0 phi1 / 2``,
    ``g^2`` coefficient ``1 / 2000``."""
    def c_fn(n1, t1, n2, t2):
        return (0.1 * np.exp(-(t1 + t2)) * np.cosh(0.5 * (t1 - t2)) + 0.01) * np.eye(2)

    F2 = np.zeros((2, 2, 2))
    F2[0, 1, 1] = 1.0
    F2[1, 0, 1] = 0.5
    system = sw.System(
        field=sw.FieldSpec("phi", 2), linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        vertices=[sw.LocalVertex("F", coupling=F2)],
        noise=_colored(sw.GaussianTemporal(lam=0.05, sigma_t=0.5)).noise)
    props = system.propagators(
        t_max=t_final, c_closed_form=c_fn, c_closed_form_only=True, progress=False)
    assert _c_has_diagonal_kink(props.cache, (0.0, t_final), ((0.0, 0.3),)) is False
    if t_final == 30.0:
        expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2], diag_C=True)
        total = expansion.evaluate(
            props, t_final=t_final, positions={"x": 0.0, "y": 0.3},
            component_pair=(0, 0), orders=[2], method="gauss_legendre", n_gauss=16,
        ).total
        assert total == pytest.approx(1.0 / 2000.0, rel=1e-10, abs=0.0)


class _CuspOffTheOrigin:
    """``n1 n2 e^{-|t1-t2|/0.5}``, which reads 0 at the origin."""

    def __call__(self, n1, t1, n2, t2):
        return float(n1) * float(n2) * float(np.exp(-abs(t1 - t2) / 0.5)) * np.eye(1)


@pytest.mark.parametrize("window, span", [
    (lambda s: np.clip(1.0 - s / 0.6, 0.0, None) ** 2, 0.6),
    (lambda s: np.clip(s - 2.4, 0.0, None) ** 2, 3.0),
    (lambda s: np.clip(s - 3.5, 0.0, None) ** 2, 10.0),
], ids=["over_before_0.7", "starts_after_2.3", "starts_past_a_window_of_3"])
def test_the_cusp_probe_reads_the_range_it_is_asked_about(window, span):
    """It read t_min + 0.7 and t_min + 2.3 whatever the range, so a kernel off
    at both reported no cusp."""
    def kernel(n1, t1, n2, t2):
        return np.eye(2) * window(t1) * window(t2) * np.exp(-abs(t1 - t2) / 0.4)

    model = sw.evaluate.PropagatorModel(
        R_time=lambda t, s: 1.0, kappa2=kernel, n_components=2)
    assert _probe_diagonal_cusp(model, (0.0, 0.0), span)


@pytest.mark.heavy
def test_a_kernel_that_vanishes_at_the_origin_is_probed_where_it_is_read():
    """Order 2 at ``x = y = 1`` against the exact moment hierarchy of the
    Markov embedding, which imports nothing from the package.  The verdict
    came from the origin, and the error at 16 nodes was 780 times larger."""
    system = sw.System(
        field=sw.FieldSpec("phi", 1), linear=sw.DiagonalA(gamma=[0.7]),
        vertices=[sw.LocalVertex("F", coupling=np.full((1, 1, 1), 0.6))],
        noise=sw.GaussianNoise(kappa2=sw.GeneralKappa2(fn=_CuspOffTheOrigin())))
    cache = PropagatorCache(system.build_propagator_model(), homogeneity="general")
    assert not cache._kappa2_has_diagonal_cusp()
    assert cache._kappa2_has_diagonal_cusp((1.0, 1.0))
    props = system.propagators(
        t_max=6.0, n_grid_t=121, c_method="gauss_legendre", c_closed_form=None,
        progress=False)
    total = system.expand(("phi(x)", "phi(y)"), orders=[2]).evaluate(
        props, t_final=6.0, positions={"x": 1.0, "y": 1.0}, orders=[2],
        method="gauss_legendre", n_gauss=16).total
    assert total == pytest.approx(EXACT_ORDER_2, rel=1e-6, abs=0.0)


def _parallel_closed_form_run(n_jobs):
    system = sw.System(
        field=sw.FieldSpec("phi", N), linear=sw.DiagonalA(gamma=[1.0] * N),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=_colored(sw.GaussianTemporal(lam=1.0, sigma_t=0.5)).noise)
    props = system.propagators(
        t_max=2.0, c_closed_form=_WhiteC(1.0), c_closed_form_only=True,
        c_closed_form_vectorized=True, diag_C=False, progress=False)
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2], diag_C=False)
    total = expansion.evaluate(
        props, positions={"x": 0.0, "y": 0.7}, t_final=2.0, component_pair=(0, 1),
        orders=[2], method="gauss_legendre", n_gauss=4, n_jobs=n_jobs).total
    return total, props.cache


def test_the_probe_answer_reaches_the_workers():
    """Each worker repeated the probe on its copy of the cache, and the parent
    never saw the answer."""
    serial, _ = _parallel_closed_form_run(1)
    parallel, cache = _parallel_closed_form_run(2)
    assert parallel == serial
    assert cache._c_kink_memo


def _early_or_late_cusp(end, floor):
    """A cusp at one end of ``[0, 10]`` under a smooth part ``floor`` times
    smaller or larger, and nothing on the diagonal at the other end."""
    def direct(x1, t1, x2, t2):
        t1, t2 = np.asarray(t1), np.asarray(t2)
        a, b = (t1, t2) if end == "early" else (10.0 - t1, 10.0 - t2)
        out = floor * np.exp(-((t1 - t2) ** 2) / 50.0) + np.exp(
            -(a ** 2 + b ** 2) / (2.0 * 0.5 ** 2) - np.abs(t1 - t2) / 0.5)
        return out[..., None, None] * np.eye(2)

    return SimpleNamespace(_C_value_direct=direct, _closed_form_vectorized=True,
                           model=SimpleNamespace(diag_C=False))


@pytest.mark.parametrize("end, floor", [("early", 1e-8), ("late", 0.01)])
def test_a_cusp_at_one_end_of_the_range_is_found_by_its_probe_times(end, floor):
    """With three probe times, or with none of the late ones, these were
    missed by the jump test and the cut test alike, at order-2 errors of
    3.8e-2 and 1.2e-2 at 24 nodes."""
    assert _probe_c_diagonal_kink(_early_or_late_cusp(end, floor), (0.0, 10.0),
                                  0.0, 0.0)


#: Order-2 ``<phi phi>`` at ``T = 6`` for ``gamma = 0.7``, ``F = 0.6`` and
#: stationary OU noise of variance 1 and time 0.5, from
#: ``examples/reference/ito_moments.py``.
EXACT_ORDER_2 = 1.31157451561


def _probe(fn, span, offset=0.0, vectorised=True, diag_C=False):
    """The probe on ``C = fn(τ)``, ``τ = (t1 - t2) / span``, an ``(n, 2, 2)``
    array under the batched contract."""
    def direct(x1, t1, x2, t2):
        return fn((np.asarray(t1) - np.asarray(t2)) / span)

    cache = SimpleNamespace(_C_value_direct=direct,
                            _closed_form_vectorized=vectorised,
                            model=SimpleNamespace(diag_C=diag_C))
    return _probe_c_diagonal_kink(cache, (offset, offset + span), 0.0, 0.0)


def _entries(c00, c11=None):
    """A diagonal ``(.., 2, 2)`` C from its two entries."""
    def fn(tau):
        c1 = c00(tau)
        c2 = c1 if c11 is None else c11(tau)
        return np.stack([c1, 0.0 * c1, 0.0 * c1, c2], axis=-1).reshape(
            c1.shape + (2, 2))
    return fn


def _gauss(width):
    return lambda tau: np.exp(-0.5 * (tau / width) ** 2)


SMOOTH_C = {
    "unit width": _gauss(1.0),
    # Falls back to smooth at the finest rungs, after looking like a jump.
    "rounded far below the range": lambda tau: 1.0 + 1e-8 * np.sqrt(tau ** 2 + 1e-10),
}
KINKED_C = {
    "unit cusp": lambda tau: np.exp(-np.abs(tau)),
    "weak cusp": lambda tau: np.exp(-tau ** 2) + 1e-5 * np.abs(tau),
    "third derivative": lambda tau: np.exp(-np.abs(tau)) * (1.0 + np.abs(tau)),
    "fast cusp": lambda tau: np.exp(-1e4 * np.abs(tau)),
    "rounded cusp": lambda tau: np.sqrt(tau ** 2 + 1e-12),
    "narrow": _gauss(1e-3),
    # Moves no integral, and an unsplit tensor rule reads them at full height.
    "below the nodes nearest the cut": _gauss(1e-6),
    "a value at equal times alone": lambda tau: _gauss(1.0)(tau) + 0.1 * (tau == 0.0),
    # Moves no integral: only the jump test sees it.
    "weak second-derivative jump":
        lambda tau: np.exp(-tau ** 2) + 1e-6 * tau * np.abs(tau),
    # What a hand-written C of an exponential kernel has.  Only the stencil
    # of order 3 sees it, on a C flat enough to leave it above the curvature.
    "weak third-derivative jump": lambda tau: 1.0 + 1e-4 * np.abs(tau) ** 3,
    # Moves an integral by 1e-8: only a tolerance below that sees it.
    "faint and narrow": lambda tau: 1.0 + 1e-6 * _gauss(1e-3)(tau),
}
RANGES = [(3.0, 0.0), (1e-6, 0.0), (1e6, 0.0), (30.0, 1e6)]


@pytest.mark.parametrize("span, offset", RANGES)
@pytest.mark.parametrize("vectorised", [True, False])
def test_the_closed_form_probe_separates_a_jump_from_a_smooth_c(
        span, offset, vectorised):
    for name, fn in SMOOTH_C.items():
        assert not _probe(_entries(fn), span, offset, vectorised), name
    for name, fn in KINKED_C.items():
        assert _probe(_entries(fn), span, offset, vectorised), name
    only_small = _entries(lambda tau: 1e6 * np.exp(-tau ** 2),
                          lambda tau: 1e-8 * np.exp(-np.abs(tau)))
    assert _probe(only_small, span, offset, vectorised)
