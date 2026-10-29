r"""Custom caches get owned outputs and read-only arguments.

CA1: ``select_gl_node_count`` built its probe cells from the caller's
``t_max`` object and handed it to ``R_time`` at every quadrature node, and
the three spatial table builders kept that object as the table's range.  A
0-d array ``t_max`` and an ``R_time`` that writes into its time argument
(``t -= 1e-5`` after reading it) moved the caller's ``t_max`` from 2.1 to
1.580, the method resolved to ``dblquad`` instead of Gauss-Legendre
``n = 20``, and C at the table nodes changed by up to 23 % relative, with
no error.  ``t_max`` is now converted to a float on entry.

CA2: with object-dtype positions ``PropagatorCache.C_value`` skips its
memo and returned the C hook's own array, so a hook that fills and returns
one buffer changed the first result at the second lookup (C_01 from
-0.0235 to -0.0075), and a ``C_diagonal`` taken before it (C_11 from 0.078
to 0.016).  The result is now an owned copy on that branch as well.

CA3: the batched integrators keep one batch per C factor and per matrix R
factor while they ask the cache for the next one.  A user-written cache
(a ``PropagatorCache`` subclass that overrides a batch method, or a cache
that is not a ``PropagatorCache``) whose ``C_diagonal_batch``,
``C_matrix_batch``, ``C_at_batch`` or ``R_matrix_batch`` fills and returns
one buffer therefore gave every kept factor the last factor's values.  At
order 2 on ``gauss_legendre`` and ``qmc_vectorized``: ``C_at_batch`` 2.8
times the value, ``R_matrix_batch`` +16.5 % (+2.8 % and +8.7 % with the
propagator lookup bank), ``C_diagonal_batch`` -1.2 %, ``C_matrix_batch``
-0.07 %; the same in ``integrate_two_point_qmc``; -5.5 % and -12.5 % in
the zero-dimensional branch of a 7-point function.  The batch helpers now
copy the output of any batch method that is not the package's own; the
package's caches return fresh arrays and pay nothing.

CA4: the same integrators hand the cache views of the time batches they
read again (``times_arr`` columns, the shared fixed-time arrays, the
two-point ``t_arr``).  A cache that forms the lag in place (``t1 -= t2``)
changed every later factor of the sample: -72 % through ``R_matrix_batch``,
-59 % through the scalar ``R_time_batch``, -3 % to -11 % through the C
batch methods, -63 % in ``integrate_two_point_qmc``.  The filled time
arrays are now read-only, so such a write raises ``ValueError`` instead.

CA5: ``_C_at_fixed_positions`` passed the caller's position arrays to
``C_at_batch``, the same object for both ends of a C inside one group, and
the spatial path of ``C_value`` passed views of them.  A cache that forms
the separation in place (``x1 -= x2``) zeroed the caller's positions, and
the later lookups read the zeros: 3.2 times the order-2 value at 3-D
positions on both batched integrators and in ``integrate_two_point_qmc``.
Through ``C_value`` (order 0) the caller's ``x`` became ``x - y``.  The
cache now receives owned float copies.

References share no code with the package: C of the generic system in
closed form in its inner time integral (``erf``) and by a 48-node
Gauss-Legendre rule in the outer one, itself checked against scipy
``dblquad``; for the buffers and the writes, the same cache returning fresh
arrays and reading its arguments only.  Inputs are generic: two components
with distinct rates (equal rates where the scalar-R branch is the subject),
a mixing matrix and couplings with entries of both signs, a dense
non-normal R, ``t_min != 0``, distinct points and distinct times.
"""
from __future__ import annotations

import os
from math import exp

import numpy as np
import pytest
from scipy.integrate import dblquad
from scipy.special import erf

import sft_wick as sw
from sft_wick.evaluate import (
    PropagatorCache,
    PropagatorModel,
    integrate_diagrams,
    integrate_two_point_qmc,
    select_gl_node_count,
)

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

T0 = 0.4
T_MAX = 2.1
GAMMA = (0.6, 1.7)
SIGMA_T, ELL = 0.45, 0.9
G_MIX = np.array([[0.7, -0.25], [-0.25, 1.3]])
G_DIAG = np.diag([0.7, 1.3])
F = np.array([[[0.35, -0.6], [0.2, 0.15]], [[-0.45, 0.25], [0.5, -0.3]]])
X3, Y3 = np.array([0.31, -0.77, 0.12]), np.array([-0.42, 0.15, 0.98])
TX, TY = 1.9, 1.45
PAIR = (1, 0)

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(48)


def _time_integral(ga, gb, t1, t2):
    """``∫_{T0}^{t1} dl1 ∫_{T0}^{t2} dl2 e^{-ga(t1-l1)} e^{-gb(t2-l2)}
    e^{-(l1-l2)²/2σ²}``: the inner integral by ``erf``, the outer by a
    48-node Gauss-Legendre rule (the outer integrand is smooth)."""
    if t1 <= T0 or t2 <= T0:
        return 0.0
    l1 = T0 + 0.5 * (t1 - T0) * (_NODES + 1.0)
    w = 0.5 * (t1 - T0) * _WEIGHTS
    m = l1 + gb * SIGMA_T**2
    root = SIGMA_T * np.sqrt(2.0)
    inner = (np.exp(-gb * t2 + gb * l1 + 0.5 * gb**2 * SIGMA_T**2)
             * SIGMA_T * np.sqrt(np.pi / 2)
             * (erf((t2 - m) / root) - erf((T0 - m) / root)))
    return float(np.sum(w * np.exp(-ga * (t1 - l1)) * inner))


def _envelope(x1, x2):
    d = np.asarray(x1, dtype=float) - np.asarray(x2, dtype=float)
    return exp(-float(np.sum(d * d)) / (2 * ELL**2))


def _c_reference(x1, t1, x2, t2, g, rates=GAMMA, v=None):
    """C of ``R(t, s) = V e^{-diag(rates)(t-s)} V⁻¹`` and
    ``κ² = g env(x1, x2) e^{-(l1-l2)²/2σ²}``:
    ``C = V (P ∘ W M Wᵀ) Vᵀ`` with ``P_km`` the time integral at rates
    ``k, m``, ``M = g env`` and ``W = V⁻¹``."""
    v = np.eye(2) if v is None else v
    w = np.linalg.inv(v)
    m = w @ (g * _envelope(x1, x2)) @ w.T
    p = np.array([[_time_integral(rk, rm, t1, t2) for rm in rates] for rk in rates])
    return v @ (p * m) @ v.T


class _Kappa:
    """``g env(x1, x2) e^{-(l1-l2)²/2σ²}``."""

    def __init__(self, g):
        self.g = g

    def __call__(self, n1, l1, n2, l2):
        return self.g * _envelope(n1, n2) * exp(-((l1 - l2) ** 2) / (2 * SIGMA_T**2))


class _ExactC:
    """The reference as a ``c_value_fn`` hook: it fills the tables fast and
    leaves the integrators as the only package code under test."""

    def __init__(self, g, rates=GAMMA):
        self.g, self.rates = g, rates

    def __call__(self, n1, t1, n2, t2):
        return _c_reference(n1, float(t1), n2, float(t2), self.g, self.rates)


def test_CA0_reference_matches_scipy_dblquad():
    for ga, gb, t1, t2 in ((0.6, 1.7, 1.9, 1.45), (1.7, 0.6, 1.1, 2.05)):
        want = dblquad(
            lambda l2, l1: exp(-ga * (t1 - l1) - gb * (t2 - l2)
                               - (l1 - l2) ** 2 / (2 * SIGMA_T**2)),
            T0, t1, T0, t2, epsabs=0, epsrel=1e-13)[0]
        assert _time_integral(ga, gb, t1, t2) == pytest.approx(want, rel=1e-12, abs=0.0)


# ---------------------------------------------------------------------------
# CA1: a 0-d array t_max and an R_time that writes into its time argument
# ---------------------------------------------------------------------------

V_DENSE = np.array([[1.0, 0.4], [-0.3, 1.0]])
W_DENSE = np.linalg.inv(V_DENSE)


class _DenseR:
    """A dense non-normal R; ``eps`` writes into the first time argument
    after reading it (a rebind for a float, in place for a 0-d array)."""

    def __init__(self, eps=0.0):
        self.eps = eps

    def __call__(self, t, s):
        value = V_DENSE @ np.diag(np.exp(-np.array(GAMMA) * (float(t) - float(s)))) @ W_DENSE
        if self.eps:
            t -= self.eps
        return value


def _dense_model(eps):
    return PropagatorModel(R_time=_DenseR(eps), kappa2=_Kappa(G_MIX), n_components=2,
                           iso_R=False, diag_C=False, t_min=T0)


def test_CA1_gl_node_count_takes_a_float_t_max():
    want = select_gl_node_count(_dense_model(0.0), T_MAX)
    t_max = np.array(T_MAX)
    assert select_gl_node_count(_dense_model(1e-5), t_max) == want == 20
    assert float(t_max) == T_MAX


_CA1_POINTS = ((0.3, 1, -0.5, 8), (0.3, 5, -0.5, 10), (0.3, 11, -0.5, 11))


@pytest.mark.parametrize("mode", ["general", "translation"])
def test_CA1_lazy_tables_take_a_float_t_max(mode):
    """Writer plus 0-d ``t_max`` against the plain R plus a float, read at
    table nodes; both against the reference."""
    n_grid = 12
    ts = np.linspace(T0, T_MAX, n_grid)
    values, ranges, methods = {}, {}, {}
    for eps, t_max in ((0.0, T_MAX), (1e-5, np.array(T_MAX))):
        cache = PropagatorCache(_dense_model(eps), homogeneity=mode, c_method="auto")
        getattr(cache, f"precompute_C_table_{mode}")(t_max=t_max, n_grid_t=n_grid)
        values[eps] = np.array([cache.C_value(x1, ts[i], x2, ts[j])
                                for x1, i, x2, j in _CA1_POINTS])
        ranges[eps] = cache._c_table_range
        methods[eps] = (cache.c_method_resolved, cache.n_gauss_resolved)
        assert float(t_max) == T_MAX
    assert methods[1e-5] == methods[0.0] == ("gauss_legendre", 20)
    assert ranges[1e-5] == (T0, T_MAX) and type(ranges[1e-5][1]) is float
    np.testing.assert_array_equal(values[1e-5], values[0.0])
    want = np.array([_c_reference(x1, ts[i], x2, ts[j], G_MIX, v=V_DENSE)
                     for x1, i, x2, j in _CA1_POINTS])
    np.testing.assert_allclose(values[0.0], want, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# CA2: C_value with object-dtype positions and a hook that reuses a buffer
# ---------------------------------------------------------------------------

class _ReusingHook:
    def __init__(self, g):
        self.fn = _ExactC(g)
        self.buffer = np.empty((2, 2))

    def __call__(self, n1, t1, n2, t2):
        self.buffer[...] = self.fn(n1, t1, n2, t2)
        return self.buffer


def _gaussian_system(g, rates=GAMMA, vertices=(), nonlocal_vertices=()):
    return sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=list(rates)),
        noise=sw.GaussianNoise(kappa2=sw.GeneralKappa2(fn=_Kappa(g))),
        vertices=list(vertices),
        nonlocal_vertices=list(nonlocal_vertices),
        t_min=T0,
    )


@pytest.mark.parametrize("layer", ["L0", "L1"])
def test_CA2_C_value_returns_an_owned_array_with_object_positions(layer):
    hook = _ReusingHook(G_MIX)
    if layer == "L0":
        model = _gaussian_system(G_MIX).build_propagator_model(diag_C=False)
        cache = PropagatorCache(model, c_value_fn=hook)
    else:
        cache = _gaussian_system(G_MIX).propagators(
            t_max=T_MAX, c_closed_form=hook, c_closed_form_only=True,
            diag_C=False, progress=False).cache
    x, y = np.array([0.35], dtype=object), np.array([-0.6], dtype=object)
    first = cache.C_value(x, 1.25, y, 0.85)
    diag = cache.C_diagonal(x, 1.25, y, 0.85)
    want = _c_reference(0.35, 1.25, -0.6, 0.85, G_MIX)
    second = cache.C_value(np.array([1.1], dtype=object), 1.7,
                           np.array([0.2], dtype=object), 0.55)
    assert not np.shares_memory(first, hook.buffer)
    np.testing.assert_allclose(first, want, rtol=1e-13)
    np.testing.assert_allclose(diag, np.diag(want), rtol=1e-13)
    np.testing.assert_allclose(second, _c_reference(1.1, 1.7, 0.2, 0.55, G_MIX),
                               rtol=1e-13)


# ---------------------------------------------------------------------------
# CA3: batch methods of a user-written cache that return one buffer
# ---------------------------------------------------------------------------

def _subclass(name, body):
    def method(self, *args):
        return body(self, getattr(PropagatorCache, name).__get__(self), *args)

    return type(f"_Custom_{name}", (PropagatorCache,), {name: method})


def _reuse(cache, parent, *args):
    value = parent(*args)
    buffers = cache.__dict__.setdefault("_buffers", {})
    out = buffers.setdefault(value.shape, np.empty_like(value))
    out[...] = value
    return out


class _Forwarding:
    """Not a ``PropagatorCache``: forwards every attribute to ``inner`` and
    wraps one batch method in ``body``."""

    def __init__(self, inner, name, body):
        self._inner, self._name, self._body = inner, name, body

    def __getattr__(self, attr):
        value = getattr(self._inner, attr)
        if attr == self._name:
            return lambda *args: self._body(self, value, *args)
        return value


_TABLES = {
    "legacy": lambda c: c.precompute_C_table(t_max=T_MAX, n_grid=20),
    "spatial": lambda c: c.precompute_C_table_general(t_max=T_MAX, n_grid_t=16),
}
_SYSTEMS = {"diag": (G_DIAG, True), "mix": (G_MIX, False)}
_METHODS = {
    "gl": ("gauss_legendre", dict(n_gauss=6)),
    "qmc": ("qmc_vectorized", dict(n_samples=1024, seed=5)),
}


@pytest.fixture(scope="module")
def expansions():
    out = {}
    for kind, (g, diag_C) in _SYSTEMS.items():
        system = _gaussian_system(g, vertices=[sw.LocalVertex("F", coupling=F)])
        out[kind] = (system, system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2],
                                           diag_C=diag_C, progress=False))
    return out


def _build(cls, kind, table):
    g, diag_C = _SYSTEMS[kind]
    model = _gaussian_system(g).build_propagator_model(diag_C=diag_C)
    cache = cls(model, homogeneity="general", c_value_fn=_ExactC(g))
    _TABLES[table](cache)
    return cache


def _evaluate(expansions, kind, cache, method, orders=(2,), pair=PAIR, **extra):
    name, kw = _METHODS[method]
    return expansions[kind][1].evaluate(
        sw.propagators_from_cache(cache), positions={"x": X3, "y": Y3},
        t_final=TX, external_times={"x": TX, "y": TY}, component_pair=pair,
        orders=list(orders), method=name, **kw, **extra).total


_PLAIN: dict = {}


def _plain_total(expansions, kind, table, method, **extra):
    key = (kind, table, method, tuple(sorted(extra.items())))
    if key not in _PLAIN:
        _PLAIN[key] = _evaluate(expansions, kind, _build(PropagatorCache, kind, table),
                                method, **extra)
    return _PLAIN[key]


_BUFFER_CASES = [
    ("diag", "legacy", "C_diagonal_batch"),
    ("mix", "legacy", "C_matrix_batch"),
    ("diag", "spatial", "C_at_batch"),
    ("mix", "spatial", "C_at_batch"),
    ("mix", "legacy", "R_matrix_batch"),
    ("diag", "spatial", "R_matrix_batch"),
]
_BANK = dict(propagator_cache_bytes=1 << 22, propagator_cache_contract="fixed_model")


@pytest.mark.parametrize("method", sorted(_METHODS))
@pytest.mark.parametrize("kind, table, name", _BUFFER_CASES,
                         ids=[f"{k}-{t}-{n}" for k, t, n in _BUFFER_CASES])
@pytest.mark.parametrize("variant", ["subclass", "subclass+bank", "forwarding"])
def test_CA3_a_reused_batch_buffer_gives_the_fresh_value(expansions, method, kind,
                                                         table, name, variant):
    extra = _BANK if variant.endswith("bank") else {}
    want = _plain_total(expansions, kind, table, method, **extra)
    if variant == "forwarding":
        cache = _Forwarding(_build(PropagatorCache, kind, table), name, _reuse)
    else:
        cache = _build(_subclass(name, _reuse), kind, table)
    assert want != 0.0
    assert _evaluate(expansions, kind, cache, method, **extra) == want


@pytest.mark.parametrize("pair", [(0, 0), (0, 1), (1, 1)])
def test_CA3_order_zero_matches_the_reference(expansions, pair):
    """The fresh value itself: order 0 through the spatial table at 3-D
    points against the closed form, to the 16 x 16 table's interpolation
    error."""
    cache = _build(_subclass("C_at_batch", _reuse), "mix", "spatial")
    got = _evaluate(expansions, "mix", cache, "gl", orders=(0,), pair=pair)
    want = _c_reference(X3, TX, Y3, TY, G_MIX)[pair]
    assert got == pytest.approx(want, rel=2e-4, abs=0.0)


def _two_point_integrands(expansions, kind):
    system, exp_ = expansions[kind]
    cv = system.build_coupling_values()
    return [dt.build_integrand(cv, dict(zip("ab", PAIR))) for dt in exp_.dts_by_order[2]]


def _two_point(integrands, cache, positions):
    return integrate_two_point_qmc(integrands, TX, positions, cache, t_min=T0,
                                   n_samples=1024, seed=7,
                                   external_times={"x": TX, "y": TY})[0]


@pytest.mark.parametrize("kind, table, name", [
    ("mix", "spatial", "C_at_batch"),
    ("mix", "legacy", "C_matrix_batch"),
    ("diag", "legacy", "R_matrix_batch"),
])
def test_CA3_two_point_qmc_with_a_reused_buffer(expansions, kind, table, name):
    # The time-only table cannot see positions; coincident points keep
    # its C_matrix_batch branch reachable.
    positions = {"x": X3, "y": Y3} if table == "spatial" else {"x": 0.3, "y": 0.3}
    integrands = _two_point_integrands(expansions, kind)
    want = _two_point(integrands, _build(PropagatorCache, kind, table), positions)
    got = _two_point(integrands, _build(_subclass(name, _reuse), kind, table), positions)
    assert want != 0.0
    assert got == want


_K0 = np.array([[[0.31, -0.57], [0.12, 0.44]], [[-0.26, 0.08], [0.73, -0.19]]])
_SEVEN = ("u", "v", "w", "x", "y", "z", "q")


class _KappaR3:
    """An R-contracted κ³ (L0, MSR factor included) with no index symmetry
    and a weight that differs per leg."""

    def __call__(self, n, t):
        n, t = np.asarray(n, dtype=float), np.asarray(t, dtype=float)
        w = np.exp(-0.3 * t[0] - 0.5 * t[1] - 0.7 * t[2]) * (1 + 0.2 * n[0] - 0.1 * n[2])
        return (1j / 6) * w * _K0


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
@pytest.mark.parametrize("table", ["legacy", "spatial"])
def test_CA3_zero_dimensional_branch_with_a_reused_buffer(method, table):
    """Seven points, one absorbed κ³: no integration variable is left and
    each diagram keeps two C batches.  The batched zero-dimensional branch
    needs a scalar R, so the two rates are equal here."""
    rates = (0.9, 0.9)
    system = _gaussian_system(G_DIAG, rates=rates, nonlocal_vertices=[sw.NonLocalVertex(
        "K", order=3, coupling=_KappaR3(), already_R_contracted=True)])
    exp_ = system.expand([f"phi_{c}({p})" for c, p in zip("abcdefg", _SEVEN)],
                         orders=[1], progress=False)
    model = system.build_propagator_model()
    assert model.iso_R
    totals = {}
    for name, cls in (("fresh", PropagatorCache),
                      ("reuse", _subclass("C_at_batch" if table == "spatial"
                                          else "C_diagonal_batch", _reuse))):
        cache = cls(model, homogeneity="general", c_value_fn=_ExactC(G_DIAG, rates))
        _TABLES[table](cache)
        totals[name], _ = integrate_diagrams(
            exp_.dts_by_order[1], {"K": _KappaR3()}, TX, cache, t_min=T0,
            method=method, n_gauss=3, n_samples=64, seed=1,
            fixed_indices=dict(zip("abcdefg", (0, 1, 1, 0, 1, 0, 1))),
            positions=dict(zip(_SEVEN, (0.1, -0.4, 0.7, 0.25, -0.15, 0.55, -0.3))),
            external_times=dict(zip(_SEVEN, (1.9, 1.2, 1.5, 0.9, 1.6, 1.3, 1.75))))
    assert totals["fresh"] != 0.0
    assert totals["reuse"] == totals["fresh"]


# ---------------------------------------------------------------------------
# CA4: batch methods that write into their time arguments
# ---------------------------------------------------------------------------

def _lag_in_place(cache, parent, t1, t2, *rest):
    """Reads its arguments, then forms the lag in place in ``t1``."""
    value = parent(t1, t2, *rest)
    t1 -= t2
    return value


_WRITE_CASES = [
    ("diag", "legacy", "R_matrix_batch"),
    ("diag", "legacy", "C_diagonal_batch"),
    ("mix", "legacy", "C_matrix_batch"),
    ("mix", "spatial", "C_at_batch"),
]


@pytest.mark.parametrize("method", sorted(_METHODS))
@pytest.mark.parametrize("kind, table, name", _WRITE_CASES,
                         ids=[f"{k}-{t}-{n}" for k, t, n in _WRITE_CASES])
def test_CA4_a_cache_writing_into_its_times_raises(expansions, method, kind, table,
                                                   name):
    cache = _build(_subclass(name, _lag_in_place), kind, table)
    with pytest.raises(ValueError, match="read-only"):
        _evaluate(expansions, kind, cache, method)


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized", "two_point"])
def test_CA4_a_scalar_R_batch_writing_into_its_times_raises(method):
    rates = (0.9, 0.9)
    system = _gaussian_system(G_DIAG, rates=rates,
                              vertices=[sw.LocalVertex("F", coupling=F)])
    exp_ = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2], progress=False)
    model = system.build_propagator_model()
    assert model.iso_R
    cache = _subclass("R_time_batch", _lag_in_place)(
        model, homogeneity="general", c_value_fn=_ExactC(G_DIAG, rates))
    _TABLES["spatial"](cache)
    with pytest.raises(ValueError, match="read-only"):
        if method == "two_point":
            cv = system.build_coupling_values()
            _two_point([dt.build_integrand(cv, dict(zip("ab", PAIR)))
                        for dt in exp_.dts_by_order[2]], cache, {"x": X3, "y": Y3})
        else:
            exp_.evaluate(sw.propagators_from_cache(cache),
                          positions={"x": X3, "y": Y3}, t_final=TX,
                          external_times={"x": TX, "y": TY}, component_pair=PAIR,
                          orders=[2], method=method, n_gauss=6, n_samples=1024)


def test_CA4_two_point_qmc_with_a_writing_cache_raises(expansions):
    cache = _build(_subclass("R_matrix_batch", _lag_in_place), "diag", "spatial")
    with pytest.raises(ValueError, match="read-only"):
        _two_point(_two_point_integrands(expansions, "diag"), cache,
                   {"x": X3, "y": Y3})


# ---------------------------------------------------------------------------
# CA5: a C_at_batch that writes into its position arguments
# ---------------------------------------------------------------------------

def _separation(cache, parent, t1, t2, x1, x2, write):
    """``C`` of a translation-invariant kernel from the separation, formed
    in ``x1`` in place when ``write``."""
    if write:
        x1 -= x2
        d = x1
    else:
        d = np.asarray(x1, dtype=float) - np.asarray(x2, dtype=float)
    return parent(t1, t2, d, np.zeros_like(d))


def _separation_cache(write):
    return _subclass("C_at_batch",
                     lambda c, p, *a: _separation(c, p, *a, write=write))


@pytest.mark.parametrize("method", sorted(_METHODS))
def test_CA5_a_cache_writing_into_its_positions_gives_the_fresh_value(expansions,
                                                                      method):
    totals = {}
    for write in (False, True):
        cache = _build(_separation_cache(write), "mix", "spatial")
        px, py = X3.copy(), Y3.copy()
        name, kw = _METHODS[method]
        totals[write] = expansions["mix"][1].evaluate(
            sw.propagators_from_cache(cache), positions={"x": px, "y": py},
            t_final=TX, external_times={"x": TX, "y": TY}, component_pair=PAIR,
            orders=[0, 2], method=name, **kw).total
        np.testing.assert_array_equal(px, X3)
        np.testing.assert_array_equal(py, Y3)
    plain = _evaluate(expansions, "mix", _build(PropagatorCache, "mix", "spatial"),
                      method, orders=(0, 2))
    assert totals[True] == totals[False]
    assert totals[False] == pytest.approx(plain, rel=1e-12, abs=0.0)


def test_CA5_C_value_hands_an_override_copies_of_the_positions():
    """The scalar route: ``C_value`` on a spatial table calls ``C_at_batch``
    with the positions reshaped, which were views of the caller's arrays."""
    values = {}
    for write in (False, True):
        cache = _build(_separation_cache(write), "mix", "spatial")
        px, py = X3.copy(), Y3.copy()
        values[write] = cache.C_value(px, TX, py, TY)
        np.testing.assert_array_equal(px, X3)
        np.testing.assert_array_equal(py, Y3)
    np.testing.assert_array_equal(values[True], values[False])
    np.testing.assert_allclose(values[False], _c_reference(X3, TX, Y3, TY, G_MIX),
                               rtol=2e-4)


def test_CA5_two_point_qmc_with_a_writing_cache(expansions):
    integrands = _two_point_integrands(expansions, "mix")
    totals = {}
    for write in (False, True):
        px, py = X3.copy(), Y3.copy()
        totals[write] = _two_point(integrands,
                                   _build(_separation_cache(write), "mix", "spatial"),
                                   {"x": px, "y": py})
        np.testing.assert_array_equal(px, X3)
        np.testing.assert_array_equal(py, Y3)
    assert totals[False] != 0.0
    assert totals[True] == totals[False]
