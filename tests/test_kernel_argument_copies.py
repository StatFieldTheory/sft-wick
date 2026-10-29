r"""User kernels get copies of the arrays the package reads again.

KC1-KC3: a ``GeneralKappa2`` or ``CustomImpulse`` callable is evaluated at
every quadrature node of one C value with the same two position objects,
and the transposition probe handed it the caller's positions.  A callable
that forms ``x1 - x2`` in place in its first argument (``n1 -= n2``)
therefore changed every later node: at 3-D positions C_00 came out as
9.0e-10 on ``gauss_legendre`` and 0.0 on ``dblquad`` against 5.85e-2, with
no error.  Every κ² and σ² call now receives copies of array and list
positions, as the closed-form C hooks do since the copies of
``PropagatorCache._C_value_direct_on_copies``.

KC4: a vectorised closed-form C that fills and returns one preallocated
buffer was returned as is, so the integrators, which keep one lookup per C
factor alive, saw the last factor in place of every earlier one: +78 %
under ``diag_C`` and +27 % without it at order 2.

KC5: the same for an L0 coupling callable.  The evaluator keeps the tensors
of every vertex copy until it contracts them, so a callable returning one
buffer gave both κ³ copies of a six-point function the last one's values
(-0.004268 against -0.003554, +20 %) on ``gauss_legendre`` and
``qmc_vectorized`` for a vectorised callable, and on the per-sample ``qmc``
loop for either contract.  L1 wraps every callable in its MSR factor, which
allocates, so L1 was not affected.

KC6: a κ² returning one buffer gave the cusp probe three equal values (no
cusp found, so ``dblquad`` integrated a ``|Δt|`` cusp without the split:
7.9e-8 relative instead of 1.9e-16, 38 times slower), and the transposition
probe compared the buffer with its own transpose.

KC7-KC9: the same for values kept across a second call to the same
callable.  ``integrate_two_point_qmc`` divided κ² at a separation by κ² at
zero separation, so a κ² returning one buffer made the ratio 1 (2.23 times
too large at separation 0.8).  The Gauss-Legendre C quadrature kept
R(t1, λ1) while it called R(t2, λ2), so a dense ``ExplicitR`` returning one
buffer gave C_00 48 % high and C_10 of the wrong sign.  The YAML R probes
compared R(t1, t2) after R(t2, t1) had overwritten it, and accepted an R
without the Heaviside factor.

References share no code with the package: for κ², the inner time
integral of the Gaussian temporal kernel in closed form (``erf``) and the
outer one by scipy ``quad``, plus a 1-D ``quad`` for the σ² term; for the
buffer, the same hook returning a fresh array and the per-sample contract.
Inputs are generic: two components with distinct rates, a mixing
amplitude matrix with entries of both signs, ``t_min != 0``, distinct
points and distinct times.
"""
from __future__ import annotations

import os
from math import erf, exp, pi, sqrt

import numpy as np
import pytest
from scipy.integrate import quad

import sft_wick as sw
from sft_wick.evaluate import PropagatorCache

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

T0 = 0.4
GAMMA = (0.6, 1.7)
G = np.array([[0.7, -0.25], [-0.25, 1.3]])
S = np.array([[0.45, -0.1], [-0.1, 0.3]])
SIGMA_T, ELL = 0.45, 0.9
T1, T2 = 1.9, 1.15
POSITIONS = {
    "3d": (np.array([0.31, -0.77, 0.12]), np.array([-0.42, 0.15, 0.98])),
    "1-vector": (np.array([0.55]), np.array([-0.35])),
    "float": (0.55, -0.35),
}


def _envelope(x1, x2) -> float:
    d = np.asarray(x1, dtype=float) - np.asarray(x2, dtype=float)
    return exp(-float(np.sum(d * d)) / (2 * ELL**2))


class _Kappa:
    """``G e^{-|x1-x2|²/2ℓ²} e^{-(s1-s2)²/2σ²}``; ``write`` forms the
    separation in place in ``n1``."""

    def __init__(self, write):
        self.write = write

    def __call__(self, n1, l1, n2, l2):
        if self.write:
            n1 -= n2
            env = exp(-float(np.sum(np.asarray(n1, float) ** 2)) / (2 * ELL**2))
        else:
            env = _envelope(n1, n2)
        return G * env * np.exp(-((l1 - l2) ** 2) / (2 * SIGMA_T**2))


class _Sigma:
    """``S e^{-|x1-x2|²/2ℓ²} (1 + cos(t) / 4)``."""

    def __init__(self, write):
        self.write = write

    def __call__(self, n1, t, n2):
        if self.write:
            n1 -= n2
            env = exp(-float(np.sum(np.asarray(n1, float) ** 2)) / (2 * ELL**2))
        else:
            env = _envelope(n1, n2)
        return S * env * (1.0 + 0.25 * np.cos(t))


def _reference(x1, t1, x2, t2, white):
    out = np.zeros((2, 2))
    env = _envelope(x1, x2)
    root = SIGMA_T * sqrt(2)
    for a in range(2):
        for b in range(2):
            ga, gb = GAMMA[a], GAMMA[b]

            def inner(l1):
                # ∫_{T0}^{t2} e^{-gb (t2 - l2)} e^{-(l1 - l2)²/2σ²} dl2
                m = l1 + gb * SIGMA_T**2
                pref = exp(-gb * t2 + gb * l1 + 0.5 * gb**2 * SIGMA_T**2)
                return pref * SIGMA_T * sqrt(pi / 2) * (
                    erf((t2 - m) / root) - erf((T0 - m) / root))

            val = quad(lambda l1: exp(-ga * (t1 - l1)) * inner(l1), T0, t1,
                       epsabs=0, epsrel=1e-13, limit=200)[0]
            out[a, b] = G[a, b] * env * val
            if white:
                out[a, b] += quad(
                    lambda s: exp(-ga * (t1 - s) - gb * (t2 - s)) * S[a, b]
                    * env * (1 + 0.25 * np.cos(s)),
                    T0, min(t1, t2), epsabs=0, epsrel=1e-13)[0]
    return out


def _system(kappa_write, sigma_write=None):
    sigma2 = None if sigma_write is None else sw.CustomImpulse(fn=_Sigma(sigma_write))
    return sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=list(GAMMA)),
        noise=sw.GaussianNoise(kappa2=sw.GeneralKappa2(fn=_Kappa(kappa_write)),
                               sigma2=sigma2),
        t_min=T0,
    )


def _copy(x):
    return x.copy() if isinstance(x, np.ndarray) else x


# ---------------------------------------------------------------------------
# KC1-KC3: κ² and σ² callables that write into their positions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["gauss_legendre", "dblquad"])
@pytest.mark.parametrize("shape", sorted(POSITIONS))
@pytest.mark.parametrize("writes", [
    (False, None), (True, None), (False, False), (True, True), (False, True),
], ids=["read", "write", "read+sigma", "write+sigma", "read+sigma_write"])
def test_KC1_writing_kernels_give_the_exact_C(method, shape, writes):
    kappa_write, sigma_write = writes
    model = _system(kappa_write, sigma_write).build_propagator_model(diag_C=False)
    cache = PropagatorCache(model, c_method=method, n_gauss=28)
    x1, x2 = POSITIONS[shape]
    caller_x1, caller_x2 = _copy(x1), _copy(x2)
    got = np.asarray(cache.C_value(caller_x1, T1, caller_x2, T2), dtype=float)
    want = _reference(x1, T1, x2, T2, white=sigma_write is not None)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(caller_x1, x1)
    np.testing.assert_array_equal(caller_x2, x2)


@pytest.mark.parametrize("c_method", ["gauss_legendre", "dblquad", "auto"])
def test_KC2_L1_order_zero_with_a_writing_kappa(c_method):
    """System -> propagators -> Expansion.evaluate at order 0 and d = 3:
    the writing kernel gives the reading kernel's values bit for bit, both
    match the reference to the table's interpolation error, and the
    caller's positions are unchanged (the probes run on c_method='auto')."""
    x, y = POSITIONS["3d"]
    totals = {}
    for write in (False, True):
        system = _system(write)
        props = system.propagators(t_max=2.4, n_grid_t=10, c_closed_form=None,
                                   c_method=c_method, diag_C=False,
                                   progress=False)
        exp_ = system.expand(("phi_a(p)", "phi_b(q)"), orders=0, diag_C=False,
                             progress=False)
        px, py = x.copy(), y.copy()
        totals[write] = [
            exp_.evaluate(props, positions={"p": px, "q": py}, t_final=T1,
                          external_times={"p": T1, "q": T2},
                          component_pair=pair, method="gauss_legendre",
                          n_gauss=4).total
            for pair in ((0, 0), (0, 1), (1, 1))
        ]
        np.testing.assert_array_equal(px, x)
        np.testing.assert_array_equal(py, y)
    assert totals[True] == totals[False]
    # The (t1, t2) table of 10 x 10 nodes interpolates to about 1.3e-4; the
    # writing kernel gave 9e-10 for C_00 before.
    want = _reference(x, T1, y, T2, white=False)
    np.testing.assert_allclose(totals[True], [want[0, 0], want[0, 1], want[1, 1]],
                               rtol=5e-4)


def test_KC3_transposition_probe_hands_the_kernel_copies():
    from sft_wick.evaluate import _probe_c_transpose

    x1, x2 = POSITIONS["3d"]
    for write in (False, True):
        for sigma in (None, write):
            model = _system(write, sigma).build_propagator_model(diag_C=False)
            px, py = x1.copy(), x2.copy()
            for swap in (False, True):
                # The kernel is symmetric under both transpositions.
                assert _probe_c_transpose(model, px, py, 2.4, swap=swap)
            np.testing.assert_array_equal(px, x1)
            np.testing.assert_array_equal(py, x2)


# ---------------------------------------------------------------------------
# KC4: a vectorised closed-form C that reuses one output buffer
# ---------------------------------------------------------------------------

T_MIN_C = 0.45
GAMMA_C = [0.7, 1.6]
A_C = np.array([[0.9, 0.25], [0.25, 1.3]])
LAM_C = [0.35, 0.8]


def _c_formula(x1, t1, x2, t2):
    """A smooth ``C(x1, t1; x2, t2)`` of shape ``(n, 2, 2)``."""
    t1 = np.atleast_1d(np.asarray(t1, dtype=float))
    t2 = np.atleast_1d(np.asarray(t2, dtype=float))
    r = np.abs(np.asarray(x1, dtype=float) - np.asarray(x2, dtype=float))
    out = np.empty((t1.shape[0], 2, 2))
    for a in range(2):
        for b in range(2):
            out[:, a, b] = (
                A_C[a, b] * (1 - np.exp(-GAMMA_C[a] * (t1 - T_MIN_C)))
                * (1 - np.exp(-GAMMA_C[b] * (t2 - T_MIN_C)))
                * np.exp(-0.5 * (LAM_C[a] + LAM_C[b]) * np.abs(t1 - t2))
                * np.exp(-0.7 * r))
    return out


class _Fresh:
    vectorized = True

    def __call__(self, x1, t1, x2, t2):
        return _c_formula(x1, t1, x2, t2)


class _Reuse:
    """Fills and returns one buffer per batch length."""

    vectorized = True

    def __init__(self):
        self.buffers = {}

    def __call__(self, x1, t1, x2, t2):
        values = _c_formula(x1, t1, x2, t2)
        out = self.buffers.setdefault(values.shape[0], np.empty_like(values))
        out[...] = values
        return out


class _PerSample:
    def __call__(self, x1, t1, x2, t2):
        return _c_formula(x1, t1, x2, t2)[0]


def _buffer_system():
    return sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=GAMMA_C),
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.GaussianTemporal(lam=0.3, sigma_t=0.5),
            spatial=sw.ExponentialSpatial(sigma_x=1.0))),
        vertices=[sw.LocalVertex("F", coupling=np.array(
            [[[0.4, -0.2], [-0.2, 0.6]], [[-0.5, 0.1], [0.1, 0.3]]]))],
        t_min=T_MIN_C,
    )


def _closed_form_props(system, hook, vectorized, diag_C):
    return system.propagators(t_max=3.0, c_closed_form=hook,
                              c_closed_form_only=True,
                              c_closed_form_vectorized=vectorized,
                              diag_C=diag_C, progress=False)


@pytest.mark.parametrize("diag_C", [True, False])
def test_KC4_batch_lookups_are_owned_arrays(diag_C):
    cache = _closed_form_props(_buffer_system(), _Reuse(), True, diag_C).cache
    ta1, ta2 = np.array([0.6, 1.1, 1.9]), np.array([0.9, 0.5, 1.7])
    tb1, tb2 = np.array([2.2, 0.8, 1.3]), np.array([1.4, 2.5, 0.7])
    first = cache.C_at_batch(ta1, ta2, 0.2, -0.6)
    second = cache.C_at_batch(tb1, tb2, -0.3, 0.9)
    assert not np.shares_memory(first, second)
    want = _c_formula(0.2, ta1, -0.6, ta2)
    if diag_C:
        want = np.einsum("iaa->ia", want)
    np.testing.assert_array_equal(first, want)


@pytest.mark.parametrize("method, kw", [
    ("qmc_vectorized", dict(n_samples=2048, seed=11)),
    ("gauss_legendre", dict(n_gauss=8)),
], ids=["qmc", "gl"])
@pytest.mark.parametrize("diag_C, pair", [(True, (1, 1)), (False, (1, 0))])
def test_KC4_a_reused_buffer_gives_the_fresh_value(method, kw, diag_C, pair):
    system = _buffer_system()
    exp_ = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2], diag_C=diag_C,
                         progress=False)
    totals = {}
    for name, hook, vectorized in (("fresh", _Fresh(), True),
                                   ("reuse", _Reuse(), True),
                                   ("per_sample", _PerSample(), False)):
        props = _closed_form_props(system, hook, vectorized, diag_C)
        totals[name] = exp_.evaluate(
            props, positions={"x": 0.25, "y": -0.55}, t_final=T_MIN_C + 1.6,
            component_pair=pair, orders=[2], method=method, **kw).total
    assert totals["fresh"] != 0.0
    assert totals["reuse"] == totals["fresh"] == totals["per_sample"]


# ---------------------------------------------------------------------------
# KC5: an L0 coupling callable that fills and returns one buffer
# ---------------------------------------------------------------------------

_K0 = np.array([[[0.31, -0.57], [0.12, 0.44]], [[-0.26, 0.08], [0.73, -0.19]]])
_SIX = ("u", "v", "w", "x", "y", "z")


def _kappa3(n, t):
    """A κ³ with no index symmetry and a weight that differs per leg, so the
    two vertex copies of the six-point function get different tensors."""
    n = np.asarray(n, dtype=float)
    t = np.asarray(t, dtype=float)
    w = np.exp(-0.3 * t[0] - 0.5 * t[1] - 0.7 * t[2]) * (1 + 0.2 * n[0] - 0.1 * n[2])
    return (1j / 6) * w[..., None, None, None] * _K0


class _FreshCoupling:
    def __init__(self, vectorized):
        self.vectorized = vectorized

    def __call__(self, n, t):
        return _kappa3(n, t)


class _ReusedCoupling(_FreshCoupling):
    def __init__(self, vectorized):
        super().__init__(vectorized)
        self.buffer = None

    def __call__(self, n, t):
        values = _kappa3(n, t)
        if self.buffer is None or self.buffer.shape != values.shape:
            self.buffer = np.empty_like(values)
        self.buffer[...] = values
        return self.buffer


@pytest.fixture(scope="module")
def six_point():
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[0.8, 1.3]),
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.4, sigma_t=0.7),
            spatial=sw.GaussianSpatial(sigma_x=1.3))),
        nonlocal_vertices=[sw.NonLocalVertex(
            "K", order=3, coupling=_FreshCoupling(True), coupling_vectorized=True)],
        t_min=0.3,
    )
    exp_ = system.expand([f"phi_{c}({p})" for c, p in zip("abcdef", _SIX)],
                         orders=[2], progress=False)
    props = system.propagators(t_max=2.0, n_grid_t=12, c_closed_form_only=True,
                               progress=False)
    return exp_, props


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized", "qmc"])
@pytest.mark.parametrize("vectorized", [True, False], ids=["vectorized", "per_sample"])
def test_KC5_a_reused_coupling_buffer_gives_the_fresh_value(six_point, method,
                                                           vectorized):
    """Two copies of one κ³ vertex: the evaluator keeps each copy's tensors
    while it contracts, so a callable returning one buffer made both copies
    read the last one (-0.004268 against -0.003554 on the batched paths, and
    on the per-sample loop for either contract)."""
    from sft_wick.evaluate import integrate_diagrams

    exp_, props = six_point
    totals = {}
    for name, cls in (("fresh", _FreshCoupling), ("reuse", _ReusedCoupling)):
        totals[name], _ = integrate_diagrams(
            exp_.dts_by_order[2], {"K": cls(vectorized)}, 1.7, props.cache,
            t_min=0.3, method=method, n_gauss=3, n_samples=256, seed=1,
            fixed_indices=dict(zip("abcdef", (0, 1, 1, 0, 1, 0))),
            positions=dict(zip(_SIX, (0.1, -0.4, 0.7, 0.25, -0.15, 0.55))),
            external_times=dict(zip(_SIX, (1.7, 1.2, 1.5, 0.9, 1.6, 1.3))))
    assert totals["fresh"] != 0.0
    assert totals["reuse"] == totals["fresh"]


# ---------------------------------------------------------------------------
# KC6: a κ² that fills and returns one buffer, in the two probes
# ---------------------------------------------------------------------------

_G_ADV = np.array([[0.9, -0.41], [-0.41, 1.3]])


def _advected(x1, l1, x2, l2):
    """A frozen-flow kernel: no symmetry under ``l1 <-> l2`` at ``x1 != x2``,
    and smooth at ``l1 = l2``."""
    dx = float(np.sum(np.asarray(x1, float) - np.asarray(x2, float)))
    dt = l1 - l2
    return _G_ADV * exp(-((dx - 1.7 * dt) ** 2) / 0.72 - dt * dt / 1.28)


def _cusped(x1, l1, x2, l2):
    d = np.asarray(x1, float) - np.asarray(x2, float)
    return np.diag([0.9, 1.3]) * exp(-0.7 * float(np.sum(d * d)) - abs(l1 - l2) / 0.45)


class _ReusedKappa:
    def __init__(self, fn):
        self.fn = fn
        self.buffer = np.empty((2, 2))

    def __call__(self, x1, l1, x2, l2):
        self.buffer[...] = self.fn(x1, l1, x2, l2)
        return self.buffer


def _kappa_model(fn):
    system = sw.System(field=sw.FieldSpec("phi", n_components=2),
                       linear=sw.DiagonalA(gamma=[0.73, 1.61]),
                       noise=sw.GaussianNoise(kappa2=sw.GeneralKappa2(fn=fn)),
                       t_min=0.37)
    return system.build_propagator_model(diag_C=False)


@pytest.mark.parametrize("kernel, cusp", [(_cusped, True), (_advected, False)],
                         ids=["cusped", "smooth"])
def test_KC6_a_reused_kappa_buffer_gets_the_fresh_probe_verdicts(kernel, cusp):
    """The cusp probe compared three aliased values and saw no cusp (so
    dblquad lost the split: 7.9e-8 relative, 38x slower), and the
    transposition probe compared the buffer with its own transpose."""
    from sft_wick.evaluate import _probe_c_transpose, _probe_diagonal_cusp

    x1, x2 = np.array([0.4]), np.array([-0.5])
    fresh, reused = _kappa_model(kernel), _kappa_model(_ReusedKappa(kernel))
    assert _probe_diagonal_cusp(fresh, (x1, x2)) is cusp
    assert _probe_diagonal_cusp(reused, (x1, x2)) is cusp
    for swap in (False, True):
        want = _probe_c_transpose(fresh, x1, x2, 2.2, swap=swap)
        assert _probe_c_transpose(reused, x1, x2, 2.2, swap=swap) is want
    assert _probe_c_transpose(fresh, x1, x2, 2.2, swap=False) is (kernel is _cusped)


# ---------------------------------------------------------------------------
# KC7-KC9: R and κ² values kept across a later call to the same callable
# ---------------------------------------------------------------------------

class _Reused:
    """Fills one buffer with ``fn``'s value and returns it."""

    def __init__(self, fn, shape):
        self.fn = fn
        self.buffer = np.empty(shape)

    def __call__(self, *args):
        self.buffer[...] = self.fn(*args)
        return self.buffer


def _kappa_two_point(n1, t1, n2, t2):
    r = float(abs(np.atleast_1d(n1)[0] - np.atleast_1d(n2)[0]))
    return np.diag([1.0, 1.4]) * exp(-r)


@pytest.mark.parametrize("order", [0, 2])
def test_KC7_two_point_qmc_keeps_its_own_kappa_at_zero_separation(order):
    """``integrate_two_point_qmc`` keeps κ²(0, 0) and divides κ²(n_l, n_r)
    by it; a κ² returning one buffer made the ratio 1, so the value at
    separation 0.8 came out e^0.8 = 2.23 times too large."""
    from sft_wick import Action, Field, Vertex, compute_moment
    from sft_wick.evaluate import PropagatorModel, integrate_two_point_qmc

    phi = Field("phi", "physical", n_components=2)
    psi = Field("psi", "response", n_components=2)
    res = compute_moment([phi("a", "x"), phi("b", "y")],
                         Action([Vertex(fields=[psi, phi, phi], coupling="F")]),
                         order=order, diag_R=True, diag_C=True, iso_R=True)
    F = np.zeros((2, 2, 2))
    F[0, 0, 0], F[1, 1, 1], F[0, 1, 1] = -0.5, 0.3, 0.2
    totals = []
    for kappa in (_kappa_two_point, _Reused(_kappa_two_point, (2, 2))):
        model = PropagatorModel(R_time=lambda t, tp: np.exp(-0.9 * (t - tp)),
                                kappa2=kappa, n_components=2, iso_R=True,
                                diag_C=True, t_min=0.0)
        cache = PropagatorCache(model)
        cache.precompute_C_table(t_max=3.0, n_grid=30)
        integrands = [dt.build_integrand({"F": -1j * F},
                                         fixed_indices={"a": 1, "b": 1})
                      for dt in res.diagram_terms(order)]
        totals.append(integrate_two_point_qmc(
            integrands, 3.0, {"x": 0.0, "y": 0.8}, cache, n_samples=1024,
            seed=3)[0])
    assert totals[0] != 0.0
    assert totals[1] == totals[0]


_A_DENSE = np.array([[0.9, 0.35], [-0.2, 1.4]])


def _R_dense(t, tp):
    from scipy.linalg import expm
    return expm(-_A_DENSE * (t - tp)) if t > tp else np.zeros((2, 2))


@pytest.mark.parametrize("sigma2", [False, True], ids=["kappa2", "white"])
def test_KC8_gl_c_quadrature_keeps_its_own_first_response(sigma2):
    """The Gauss-Legendre C quadrature keeps R(t1, λ1) while it calls
    R(t2, λ2); a dense ``ExplicitR`` returning one buffer gave C_00 48 %
    high and C_10 of the wrong sign."""
    values = []
    for R in (_R_dense, _Reused(_R_dense, (2, 2))):
        noise = sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.4, sigma_t=0.7),
                spatial=sw.GaussianSpatial(sigma_x=1.3)),
            sigma2=sw.ConstantImpulse(np.diag([0.45, 0.3])) if sigma2 else None)
        system = sw.System(field=sw.FieldSpec("phi", n_components=2),
                           linear=sw.ExplicitR(R_time=R, iso_R=False),
                           noise=noise, t_min=0.2)
        props = system.propagators(t_max=2.0, n_grid_t=8, diag_C=False,
                                   c_method="gauss_legendre", progress=False)
        values.append(np.asarray(props.cache.C_value(
            np.array(0.0), 1.7, np.array(0.5), 1.1)))
    assert np.all(values[0] != 0.0)
    np.testing.assert_array_equal(values[1], values[0])


def _R_without_heaviside(t1, t2):
    return np.full((2, 2), np.nan) if t1 > t2 else np.zeros((2, 2))


def test_KC9_yaml_r_probes_keep_their_first_value():
    """The R probe compared R(t1, t2) after R(t2, t1) had overwritten it, so
    a buffer-reusing R that is NaN at causal times was accepted while the
    same R returning fresh arrays was refused."""
    from sft_wick.workflow.config import _probe_R_time

    for R in (_R_without_heaviside, _Reused(_R_without_heaviside, (2, 2))):
        with pytest.raises(ValueError):
            _probe_R_time(R, False, 2, 0.0, 3.0)
