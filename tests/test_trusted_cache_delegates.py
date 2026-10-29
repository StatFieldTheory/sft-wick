r"""A package method is trusted only while the methods it calls are the package's.

The integrators skip the copies of :func:`_c_lookup_positions` and
:func:`_owned_batch` for the package's own cache methods, found by method
identity.  A subclass can keep such a method and override what it calls.

TD1: ``SpectralPropagatorCache.C_value`` passes its positions to
``_c_matrix``.  A subclass whose ``_c_matrix`` forms the separation in place
(``n1 -= n2``) received the caller's position objects, one object for both
ends of a C inside one direction group.  In one ``Expansion.evaluate`` at
orders 0 and 2 the order-0 lookup turned the caller's ``x`` into ``x - y``
and the order-2 diagrams read it: -16.8 % on ``gauss_legendre``, -16.9 % on
``qmc_vectorized``, -17.0 % on ``qmc_scalar``.  On ``nquad`` the second
point of a two-point sweep came out -90.0 %.  Without a spatial table the
caller's ``direction`` was zeroed.  ``SpectralPropagatorCache.C_value``
now counts as the package's own only while ``_c_matrix`` is.  On a direct
call ``C_value`` itself handed ``_c_matrix`` the caller's positions and
returned its array: a second call on the same ``x`` came out -90.0 %, and
a ``_c_matrix`` filling one buffer changed the first result by -50.6 % at
the next call.  It now hands an overridden ``_c_matrix`` copies and
returns a copy.

TD2: ``PropagatorCache.C_at_batch`` returns the array of ``_assemble_C``,
``_lazy_lookup`` or ``_closed_form_at_batch_diag``.  An override of one of
them that fills and returns one buffer gave every C factor the batched
integrators keep the last factor's values: at order 2 +181.8 % (mixing κ²)
and +179.9 % (diagonal κ²) on ``gauss_legendre`` and ``qmc_vectorized``,
with and without the propagator lookup bank, and in
``integrate_two_point_qmc``.  The output is now copied unless all three are
the package's own.  A direct ``C_at_batch`` passed the caller's position
arrays on to an overridden ``_lazy_lookup`` or ``_closed_form_at_batch_diag``:
one forming ``x1 -= x2`` turned the caller's ``x`` into ``x - y``, and a
second call on the same arrays came out up to 90 % off.  Those overrides
now receive copies.

TD3: the copy guard in front of ``_C_value_direct`` was a method that
``C_value``, the table builders and the closed-form lookup called on the
caller's objects.  A subclass overriding it to form ``n1 -= n2`` before
calling ``_C_value_direct`` took the order-2 value to zero (-4.3e-10
against -4.53e-2 on ``gauss_legendre``) and changed the caller's ``x``.  The
guard is now a module function: there is no method to override, and an
override of ``_C_value_direct`` receives copies.

References: the order-0 C of the spectral subclass in closed form, and the
C of the generic system of ``test_custom_cache_aliasing``.  Each override
that writes or reuses a buffer is compared with the same override that
does neither, or with the plain cache.  Inputs are generic: two components
with distinct weights and rates (equal rates where the spectral cache
needs a scalar R), a mixing κ² with entries of both signs, ``t_min != 0``,
distinct 3-D points and distinct times.
"""
from __future__ import annotations

import os
from math import exp

import numpy as np
import pytest

import sft_wick as sw
from sft_wick._propagator_lookup import wrap_propagator_cache
from sft_wick.evaluate import PropagatorCache, _owned_batch, integrate_diagrams
from sft_wick.spectral import SpectralDensity, SpectralPropagatorCache, _c_star
from tests.test_custom_cache_aliasing import (
    _BANK,
    EQUAL,
    F,
    G_DIAG,
    G_MIX,
    PAIR,
    T0,
    T_MAX,
    TX,
    TY,
    X3,
    Y3,
    _c_reference,
    _ExactC,
    _gaussian_system,
    _reuse,
    _subclass,
    _two_point,
)

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

# ---------------------------------------------------------------------------
# TD1: a SpectralPropagatorCache subclass overriding _c_matrix
# ---------------------------------------------------------------------------

ELL = 0.9
WEIGHTS_C = np.array([0.7, 1.3])
NODES, DENSITY_WEIGHTS = np.array([0.6, 1.7]), np.array([0.35, 0.65])
D_NOISE = 0.5


def _c_star_reference(t1, t2):
    """``Σ_k w_k (D/h_k)(e^{-h_k|t1-t2|} - e^{-h_k(t1+t2-2 t_min)})``."""
    return sum(w * (D_NOISE / h) * (exp(-h * abs(t1 - t2)) - exp(-h * (t1 + t2 - 2 * T0)))
               for h, w in zip(NODES, DENSITY_WEIGHTS))


def _env(d):
    d = np.asarray(d, dtype=float)
    return exp(-float(np.sum(d * d)) / (2 * ELL**2))


class _Positional(SpectralPropagatorCache):
    """``C_ab = δ_ab w_a c*(t1, t2) env(n1 - n2)``: a position-aware C on
    the spectral one.  With ``write`` the separation is formed in ``n1``
    in place.  ``C_diagonal`` follows ``_c_matrix`` so the scalar loop sees
    the same C."""

    write = False

    def _c_matrix(self, n1, t1, n2, t2):
        self.calls = getattr(self, "calls", 0) + 1
        c = float(_c_star(self.density, float(t1), float(t2), self.noise_D, self._t_min))
        if self.write:
            n1 -= n2
            d = n1
        else:
            d = np.asarray(n1, dtype=float) - np.asarray(n2, dtype=float)
        return np.diag(WEIGHTS_C) * c * _env(d)

    def C_diagonal(self, n, t1, n_prime=None, t2=None):
        n_prime = n if n_prime is None else n_prime
        t2 = t1 if t2 is None else t2
        return np.diag(self._c_matrix(n, t1, n_prime, t2)).copy()


def _positional(write, table=True):
    cls = type("_P", (_Positional,), {"write": write})
    cache = cls(SpectralDensity(NODES, DENSITY_WEIGHTS), D_NOISE, n_components=2,
                t_min=T0)
    if table:
        cache.precompute_C_table_translation(t_max=T_MAX, n_grid_t=24)
    return cache


@pytest.fixture(scope="module")
def spectral_expansion():
    system = _gaussian_system(G_DIAG, rates=EQUAL,
                              vertices=[sw.LocalVertex("F", coupling=F)])
    return system, system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2],
                                 progress=False)


_TD1_METHODS = {
    "gauss_legendre": dict(n_gauss=6),
    "qmc_vectorized": dict(n_samples=1024, seed=5),
    "qmc_scalar": dict(n_samples=64, seed=5),
}


def _evaluate(exp_, cache, method, orders, t_final=TX, positions=None, pair=(1, 1)):
    positions = {"x": X3.copy(), "y": Y3.copy()} if positions is None else positions
    return exp_.evaluate(
        sw.propagators_from_cache(cache), positions=positions, t_final=t_final,
        external_times={"x": t_final, "y": TY}, component_pair=pair,
        orders=list(orders), method=method, **_TD1_METHODS.get(method, {})).total


@pytest.mark.parametrize("pair", [(1, 1), (1, 0)])
@pytest.mark.parametrize("method", sorted(_TD1_METHODS))
def test_TD1_a_c_matrix_writing_its_positions(spectral_expansion, method, pair):
    """Orders 0 and 2 in one call: the order-2 diagrams read the positions
    the order-0 lookup had."""
    _, exp_ = spectral_expansion
    totals = {}
    for write in (False, True):
        cache = _positional(write)
        positions = {"x": X3.copy(), "y": Y3.copy()}
        totals[write] = _evaluate(exp_, cache, method, (0, 2), positions=positions,
                                  pair=pair)
        assert cache.calls > 0
        np.testing.assert_array_equal(positions["x"], X3)
        np.testing.assert_array_equal(positions["y"], Y3)
    assert totals[False] != 0.0
    assert totals[True] == totals[False]


def test_TD1_nquad_sweep_with_a_writing_c_matrix(spectral_expansion):
    """Two grid points with one positions dict; the second reads it again."""
    _, exp_ = spectral_expansion
    values = {}
    for write in (False, True):
        cache, positions = _positional(write), {"x": X3.copy(), "y": Y3.copy()}
        values[write] = [_evaluate(exp_, cache, "nquad", (0,), t_final=tf,
                                   positions=positions) for tf in (TX, 1.7)]
        np.testing.assert_array_equal(positions["x"], X3)
    assert values[True] == values[False]
    want = [WEIGHTS_C[1] * _c_star_reference(tf, TY) * _env(X3 - Y3) for tf in (TX, 1.7)]
    np.testing.assert_allclose(values[False], want, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("method", ["qmc_scalar", "nquad", "gauss_legendre"])
def test_TD1_without_a_table_the_direction_is_kept(spectral_expansion, method):
    """No spatial table: every point is the caller's ``direction`` object."""
    system, exp_ = spectral_expansion
    cv = system.build_coupling_values()
    values = {}
    for write in (False, True):
        direction = np.array([0.35, -0.2, 0.6])
        values[write], _ = integrate_diagrams(
            exp_.dts_by_order[0], cv, TX, _positional(write, table=False), t_min=T0,
            method=method, n_samples=64, seed=5, n_gauss=6, direction=direction,
            fixed_indices={"a": 1, "b": 1}, external_times={"x": TX, "y": TY})
        np.testing.assert_array_equal(direction, [0.35, -0.2, 0.6])
    assert values[True] == values[False]
    assert values[False] == pytest.approx(WEIGHTS_C[1] * _c_star_reference(TX, TY),
                                          rel=1e-13, abs=0.0)


def test_TD1_the_trust_follows_c_matrix():
    from sft_wick.evaluate import _c_lookup_positions

    x, y = X3.copy(), Y3.copy()
    plain = SpectralPropagatorCache(SpectralDensity(NODES, DENSITY_WEIGHTS), D_NOISE,
                                    n_components=2, t_min=T0)
    for cache in (plain, wrap_propagator_cache(plain, **_BANK)):
        got = _c_lookup_positions(cache, "C_value", x, y)
        assert got[0] is x and got[1] is y
    got = _c_lookup_positions(_positional(False, table=False), "C_value", x, x)
    assert got[0] is not x and got[1] is not x and got[0] is not got[1]
    np.testing.assert_array_equal(got[0], X3)


class _Buffered(_Positional):
    """``_Positional`` filling and returning one buffer."""

    def _c_matrix(self, n1, t1, n2, t2):
        buffer = self.__dict__.setdefault("buffer", np.empty((2, 2)))
        buffer[...] = _Positional._c_matrix(self, n1, t1, n2, t2)
        return buffer


def test_TD1_a_direct_C_value_copies_for_an_overridden_c_matrix():
    """A direct call has no integrator in front of it to make the copies:
    ``C_value`` hands an overridden ``_c_matrix`` copies of the positions
    and returns a copy of its array."""
    density = SpectralDensity(NODES, DENSITY_WEIGHTS)
    want = [np.diag(WEIGHTS_C) * _c_star_reference(t1, t2) * _env(X3 - Y3)
            for t1, t2 in ((TX, TY), (1.7, 0.9))]
    x, y = X3.copy(), Y3.copy()
    writing = _positional(True, table=False)
    got = [writing.C_value(x, TX, y, TY), writing.C_value(x, TX, y, TY)]
    np.testing.assert_array_equal(x, X3)
    np.testing.assert_array_equal(y, Y3)
    np.testing.assert_allclose(got, [want[0]] * 2, rtol=1e-13, atol=0.0)
    buffered = _Buffered(density, D_NOISE, n_components=2, t_min=T0)
    first = buffered.C_value(x, TX, y, TY)
    second = buffered.C_value(x, 1.7, y, 0.9)
    assert not np.shares_memory(first, buffered.buffer)
    np.testing.assert_allclose([first, second], want, rtol=1e-13, atol=0.0)
    plain = SpectralPropagatorCache(density, D_NOISE, n_components=2, t_min=T0)
    np.testing.assert_allclose(plain.C_value(x, TX, y, TY),
                               np.eye(2) * _c_star_reference(TX, TY), rtol=1e-13)


# ---------------------------------------------------------------------------
# TD2: the methods whose output C_at_batch returns
# ---------------------------------------------------------------------------

_SYSTEMS = {"diag": (G_DIAG, True), "mix": (G_MIX, False)}
_TD2_METHODS = {
    "gl": ("gauss_legendre", dict(n_gauss=6)),
    "qmc": ("qmc_vectorized", dict(n_samples=1024, seed=5)),
}


@pytest.fixture(scope="module")
def expansions():
    out = {}
    for kind, (g, diag_C) in _SYSTEMS.items():
        system = _gaussian_system(g, vertices=[sw.LocalVertex("F", coupling=F)])
        out[kind] = (system, system.expand(("phi_a(x)", "phi_b(y)"), orders=[2],
                                           diag_C=diag_C, progress=False))
    return out


def _build(cls, kind, delegate):
    g, diag_C = _SYSTEMS[kind]
    model = _gaussian_system(g).build_propagator_model(diag_C=diag_C)
    cache = cls(model, homogeneity="general", c_value_fn=_ExactC(g))
    if delegate == "_closed_form_at_batch_diag":
        cache._closed_form_only = True
    else:
        cache.precompute_C_table_general(t_max=T_MAX, n_grid_t=16)
    return cache


def _total(expansions, kind, cache, method, **extra):
    name, kw = _TD2_METHODS[method]
    return expansions[kind][1].evaluate(
        sw.propagators_from_cache(cache), positions={"x": X3, "y": Y3},
        t_final=TX, external_times={"x": TX, "y": TY}, component_pair=PAIR,
        orders=[2], method=name, **kw, **extra).total


_TD2_CASES = (
    [(d, k, m, b) for d in ("_assemble_C", "_lazy_lookup") for k in sorted(_SYSTEMS)
     for m in sorted(_TD2_METHODS) for b in (False, True)]
    + [("_closed_form_at_batch_diag", k, "gl", False) for k in sorted(_SYSTEMS)]
)


@pytest.mark.parametrize("delegate, kind, method, bank", _TD2_CASES,
                         ids=[f"{d}-{k}-{m}{'-bank' if b else ''}"
                              for d, k, m, b in _TD2_CASES])
def test_TD2_a_delegate_returning_one_buffer(expansions, delegate, kind, method,
                                             bank):
    extra = _BANK if bank else {}
    want = _total(expansions, kind, _build(PropagatorCache, kind, delegate), method,
                  **extra)
    got = _total(expansions, kind, _build(_subclass(delegate, _reuse), kind, delegate),
                 method, **extra)
    assert want != 0.0
    assert got == want


def test_TD2_two_point_qmc_with_a_buffered_assemble_C(expansions):
    system, exp_ = expansions["mix"]
    cv = system.build_coupling_values()
    integrands = [dt.build_integrand(cv, dict(zip("ab", PAIR)))
                  for dt in exp_.dts_by_order[2]]
    totals = [_two_point(integrands, _build(cls, "mix", "_assemble_C"),
                         {"x": X3, "y": Y3})
              for cls in (PropagatorCache, _subclass("_assemble_C", _reuse))]
    assert totals[0] != 0.0
    assert totals[1] == totals[0]


def test_TD2_the_fresh_value_matches_the_reference(expansions):
    """Order 2 needs the integrators, so the reference here is order 0
    through the same lazy table at 3-D points, to its interpolation
    error."""
    system, _ = expansions["mix"]
    exp0 = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0], diag_C=False,
                         progress=False)
    cache = _build(_subclass("_assemble_C", _reuse), "mix", "_assemble_C")
    got = exp0.evaluate(
        sw.propagators_from_cache(cache), positions={"x": X3, "y": Y3},
        t_final=TX, external_times={"x": TX, "y": TY}, component_pair=PAIR,
        orders=[0], method="gauss_legendre", n_gauss=6).total
    assert got == pytest.approx(_c_reference(X3, TX, Y3, TY, G_MIX)[PAIR],
                                rel=2e-4, abs=0.0)


@pytest.mark.parametrize("delegate", ["_assemble_C", "_lazy_lookup",
                                      "_closed_form_at_batch_diag"])
def test_TD2_C_at_batch_is_trusted_only_with_its_own_delegates(delegate):
    out = np.zeros((2, 2))
    plain = _build(PropagatorCache, "mix", delegate)
    for cache in (plain, wrap_propagator_cache(plain, **_BANK)):
        assert _owned_batch(cache, "C_at_batch", out) is out
    got = _owned_batch(_build(_subclass(delegate, _reuse), "mix", delegate),
                       "C_at_batch", out)
    assert got is not out
    np.testing.assert_array_equal(got, out)


def _separation_in_place(delegate):
    """The delegate's value, then ``x1 -= x2`` in its position arguments."""
    where = {"_lazy_lookup": (3, 4), "_closed_form_at_batch_diag": (2, 3)}[delegate]

    def body(cache, parent, *args):
        value = parent(*args)
        x1, x2 = (args[k] for k in where)
        x1 -= x2
        return value

    return _subclass(delegate, body)


@pytest.mark.parametrize("delegate", ["_lazy_lookup", "_closed_form_at_batch_diag"])
def test_TD2_a_direct_C_at_batch_hands_its_delegates_copies_of_the_positions(delegate):
    """The integrators pass ``C_at_batch`` copies (``_C_at_fixed_positions``);
    a direct caller's ``(n, d)`` positions went on to an overridden
    ``_lazy_lookup`` or ``_closed_form_at_batch_diag`` as they were."""
    t1, t2 = np.array([TX, 1.3]), np.array([TY, 0.8])
    xs, ys = np.array([X3, Y3]), np.array([Y3, 0.5 * X3])
    plain = _build(PropagatorCache, "mix", delegate).C_at_batch(t1, t2, xs, ys)
    cache = _build(_separation_in_place(delegate), "mix", delegate)
    got = [cache.C_at_batch(t1, t2, xs, ys) for _ in range(2)]
    np.testing.assert_array_equal(xs, [X3, Y3])
    np.testing.assert_array_equal(ys, [Y3, 0.5 * X3])
    np.testing.assert_array_equal(got[0], plain)
    np.testing.assert_array_equal(got[1], plain)
    cache.C_at_batch(t1, t2, xs, xs)
    np.testing.assert_array_equal(xs, [X3, Y3])
    want = [_c_reference(x, a, y, b, G_MIX) for x, a, y, b in zip(xs, t1, ys, t2)]
    tol = 1e-12 if delegate == "_closed_form_at_batch_diag" else 2e-4
    np.testing.assert_allclose(plain, want, rtol=tol, atol=0.0)


# ---------------------------------------------------------------------------
# TD3: the copy guard in front of _C_value_direct
# ---------------------------------------------------------------------------

def _guard_override(name):
    """A subclass whose ``name`` forms the separation in ``n1`` in place and
    computes C from it, which the translation-invariant reference allows.
    ``_C_value_direct_on_copies`` was the copy guard's method name."""
    parent = PropagatorCache._C_value_direct

    def method(self, n1, t1, n2, t2, **kwargs):
        n1 -= n2
        return parent(self, n1, t1, np.zeros_like(n1), t2, **kwargs)

    return type(f"_Override{name}", (PropagatorCache,), {name: method})


_TD3_NAMES = ["_C_value_direct_on_copies", "_C_value_direct"]


@pytest.mark.parametrize("name", _TD3_NAMES)
@pytest.mark.parametrize("method", sorted(_TD2_METHODS))
def test_TD3_an_override_in_front_of_the_C_hook(expansions, name, method):
    """Orders 0 and 2 through a lazy table at 3-D points: the table builder
    passed one position array to every cell, and the order-0 lookup the
    caller's positions."""
    system, _ = expansions["mix"]
    exp_ = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2], diag_C=False,
                         progress=False)
    totals = {}
    for key, cls in (("plain", PropagatorCache), ("override", _guard_override(name))):
        cache = _build(cls, "mix", "_assemble_C")
        positions = {"x": X3.copy(), "y": Y3.copy()}
        mname, kw = _TD2_METHODS[method]
        totals[key] = exp_.evaluate(
            sw.propagators_from_cache(cache), positions=positions, t_final=TX,
            external_times={"x": TX, "y": TY}, component_pair=PAIR, orders=[0, 2],
            method=mname, **kw).total
        np.testing.assert_array_equal(positions["x"], X3)
    assert totals["plain"] != 0.0
    assert totals["override"] == pytest.approx(totals["plain"], rel=1e-12, abs=0.0)


@pytest.mark.parametrize("name", _TD3_NAMES)
def test_TD3_direct_lookups_keep_the_callers_positions(name):
    model = _gaussian_system(G_MIX).build_propagator_model(diag_C=False)
    cache = _guard_override(name)(model, c_value_fn=_ExactC(G_MIX))
    x, y = X3.copy(), Y3.copy()
    got = cache.C_value(x, TX, y, TY)
    np.testing.assert_array_equal(x, X3)
    np.testing.assert_allclose(got, _c_reference(X3, TX, Y3, TY, G_MIX), rtol=1e-13)
    direction = np.array([0.35, -0.2])
    cache.precompute_C_table(t_max=T_MAX, n_grid=8, direction=direction)
    np.testing.assert_array_equal(direction, [0.35, -0.2])
