r"""The two spellings of an L1 observable, and the observables L1 refuses.

``System.expand`` takes the observable as strings (``"phi_a(x)"``) or as
``FieldOperator`` objects from :meth:`System.build_fields`.  The
``FieldOperator`` branch read an attribute the class does not have
(``component_indices``), so every such call raised ``AttributeError``
although the docstring advertised the form (OO1).  The component tuple of
``Expansion.evaluate`` is positional, so a set observable, whose order
follows ``PYTHONHASHSEED``, is refused as the YAML loader refuses it; on the
model below the set ``{"phi_a(x)", "phi_b(y)"}`` gave -0.0621 or -0.1454
at ``component_pair=(0, 1)`` depending on the seed, with no error (OO2).

The model is generic: two components with distinct rates, a local cubic
``F`` and a constant non-local ``κ³`` vertex ``K`` with entries of both
signs and no index symmetry, ``t_min = 0.3``, distinct points and distinct
external times.  References share no code with the package:

* order 0, ``C_aa(x, t1; y, t2) = λ e^{-(x-y)²/2σ_x²} ∫∫ e^{-γ_a(t1-s1)}
  e^{-γ_a(t2-s2)} e^{-|s1-s2|/σ_t} ds1 ds2`` with the inner integral in
  closed form and the outer one by scipy ``quad`` (:func:`_c_reference`);
* the FK channel at order 2 for a constant ``κ³``, the one-dimensional
  integral of ``tests/test_sweep_selectors.py`` (:func:`_fk_reference`).
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.integrate import quad

import sft_wick as sw
from sft_wick import Field
from sft_wick.workflow import specs as sp

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

T_MIN = 0.3
T_FINAL = 1.7
GAMMAS = (0.8, 1.3)
LAM, SIGMA_T, SIGMA_X = 0.4, 0.7, 1.3
X, Y = 0.1, 0.55
TIMES = {"x": 1.1, "y": 1.6}
F_TENSOR = np.array([[[0.2, -0.7], [-0.7, 1.1]], [[-0.4, 0.5], [0.5, 0.9]]])
K_TENSOR = np.array([[[0.3, -0.2], [-0.2, 0.6]], [[-0.2, 0.6], [0.6, -0.5]]])
GL = dict(method="gauss_legendre", n_gauss=6)
TUPLES = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _c_reference(a: int, b: int, t1: float, t2: float, x: float = X,
                 y: float = Y) -> float:
    """``C_ab(x, t1; y, t2)`` of the model: zero off the diagonal."""
    if a != b:
        return 0.0
    g, r = GAMMAS[a], 1.0 / SIGMA_T

    def inner(s1):
        # ∫_{t_min}^{t2} e^{-g (t2 - s2)} e^{-r |s1 - s2|} ds2, closed form.
        lo = min(s1, t2)
        below = np.exp(-g * t2 - r * s1) * (
            np.exp((g + r) * lo) - np.exp((g + r) * T_MIN)) / (g + r)
        above = 0.0
        if s1 < t2:
            start = max(s1, T_MIN)
            above = np.exp(-g * t2 + r * s1) * (
                np.exp((g - r) * t2) - np.exp((g - r) * start)) / (g - r)
        return below + above

    outer = quad(lambda s1: np.exp(-g * (t1 - s1)) * inner(s1), T_MIN, t1,
                 points=[t2] if T_MIN < t2 < t1 else None,
                 epsabs=0.0, epsrel=1e-13, limit=200)[0]
    return LAM * np.exp(-(x - y) ** 2 / (2 * SIGMA_X ** 2)) * outer


def _fk_reference(a: int, b: int) -> float:
    """The FK channel of ``⟨φ_a(T) φ_b(T)⟩`` for a constant ``κ³``."""
    g = np.asarray(GAMMAS)

    def G(c, s):
        return (1.0 - np.exp(-g[c] * (s - T_MIN))) / g[c]

    def side(a, b):
        total = 0.0
        for c in range(2):
            for d in range(2):
                inner = quad(
                    lambda s: np.exp(-g[a] * (T_FINAL - s)) * G(c, s) * G(d, s),
                    T_MIN, T_FINAL, epsabs=1e-15, epsrel=1e-13)[0]
                total += F_TENSOR[a, c, d] * K_TENSOR[c, d, b] * inner
        return total * G(b, T_FINAL)

    return side(a, b) + side(b, a)


@pytest.fixture(scope="module")
def system():
    return sw.System(
        field=sp.FieldSpec("phi", n_components=2),
        linear=sp.DiagonalA(gamma=list(GAMMAS)),
        noise=sp.GaussianNoise(kappa2=sp.SeparableTranslation(
            temporal=sp.ExponentialTemporal(lam=LAM, sigma_t=SIGMA_T),
            spatial=sp.GaussianSpatial(sigma_x=SIGMA_X))),
        vertices=(sp.LocalVertex("F", F_TENSOR),),
        nonlocal_vertices=(sp.NonLocalVertex("K", order=3, coupling=K_TENSOR),),
        t_min=T_MIN,
    )


@pytest.fixture(scope="module")
def props(system):
    # The built-in closed form at every lookup, not a table of it.
    return system.propagators(t_max=2.0, n_grid_t=12, c_closed_form_only=True,
                              progress=False)


def _evaluate(expansion, props, pair, **kw):
    return expansion.evaluate(props, positions={"x": X, "y": Y},
                              t_final=T_FINAL, component_pair=pair, **GL, **kw)


# ---------------------------------------------------------------------------
# OO1: FieldOperator observables at L1
# ---------------------------------------------------------------------------

def test_OO1_field_operators_match_the_exact_covariance(system, props):
    """Order 0 at distinct times: the component tuple pins each letter, so
    (0, 0) and (1, 1) give the two distinct rates' covariances.  Every
    call raised AttributeError before."""
    phi, _psi = system.build_fields()
    exp = system.expand([phi("a", "x"), phi("b", "y")], orders=[0, 2],
                        progress=False)
    for a, b in TUPLES:
        got = _evaluate(exp, props, (a, b), orders=0, external_times=TIMES)
        want = _c_reference(a, b, TIMES["x"], TIMES["y"])
        assert got.total == pytest.approx(want, rel=1e-12, abs=1e-15), (a, b)
    assert _c_reference(0, 0, *TIMES.values()) != pytest.approx(
        _c_reference(1, 1, *TIMES.values()), rel=0.1, abs=0.0)


def test_OO1_field_operators_give_the_fk_channel(system, props):
    phi, _psi = system.build_fields()
    exp = system.expand((phi("a", "x"), phi("b", "y")), orders=2,
                        progress=False)
    for a, b in TUPLES:
        got = _evaluate(exp, props, (a, b), vertex_types="FK").total
        assert got == pytest.approx(_fk_reference(a, b), rel=1e-9, abs=0.0)


def test_OO1_both_spellings_give_one_expansion(system, props):
    """The two spellings share a cache key and give the same diagrams, so
    every channel at every order agrees bit for bit."""
    phi, _psi = system.build_fields()
    ops = system.expand([phi("a", "x"), phi("b", "y")], orders=[0, 2],
                        progress=False)
    strings = system.expand(["phi_a(x)", "phi_b(y)"], orders=[0, 2],
                            progress=False)
    assert ops.observable_repr == strings.observable_repr == (
        ("phi", "a", "x"), ("phi", "b", "y"))
    for pair in TUPLES:
        r_ops = _evaluate(ops, props, pair, external_times=TIMES)
        r_str = _evaluate(strings, props, pair, external_times=TIMES)
        assert r_ops.by_order == r_str.by_order
        assert r_ops.per_diagram == r_str.per_diagram


def test_OO1_a_shared_letter_takes_one_index(system, props):
    phi, _psi = system.build_fields()
    exp = system.expand([phi("a", "x"), phi("a", "y")], orders=0,
                        progress=False)
    with pytest.raises(ValueError, match="share the component letter 'a'"):
        _evaluate(exp, props, (0, 1))
    got = _evaluate(exp, props, (1, 1), external_times=TIMES).total
    assert got == pytest.approx(_c_reference(1, 1, *TIMES.values()),
                                rel=1e-12, abs=0.0)


@pytest.mark.parametrize("make, match", [
    (lambda phi, psi: [phi("1", "x"), phi("b", "y")], "numeric literal"),
    (lambda phi, psi: [phi("a", "x"), psi("0", "y")], "numeric literal"),
    (lambda phi, psi: [Field("phi", "physical", 3)("a", "x"), phi("b", "y")],
     "neither field of this system"),
    (lambda phi, psi: [Field("chi", "physical", 2)("a", "x"), phi("b", "y")],
     "neither field of this system"),
], ids=["literal_phi", "literal_psi", "other_n", "other_name"])
def test_OO1_operators_the_string_form_cannot_spell_are_refused(system, make,
                                                                match):
    """A literal label would fix the component the tuple chooses; an
    operator of another field is not one of this system's fields."""
    phi, psi = system.build_fields()
    with pytest.raises(ValueError, match=match):
        system.expand(make(phi, psi), orders=[0, 2], progress=False)


# ---------------------------------------------------------------------------
# OO2: a set has no order
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda: {"phi_a(x)", "phi_b(y)"},
    lambda: frozenset({"phi_a(x)", "phi_b(y)"}),
    lambda: {"phi_a(x)": 0, "phi_b(y)": 1}.keys(),
], ids=["set", "frozenset", "dict_keys"])
def test_OO2_a_set_observable_is_refused(system, make):
    with pytest.raises(ValueError, match="set, which has no order"):
        system.expand(make(), orders=[0, 2], progress=False)


def test_OO2_a_set_of_field_operators_is_refused(system):
    phi, _psi = system.build_fields()
    with pytest.raises(ValueError, match="set, which has no order"):
        system.expand({phi("a", "x"), phi("b", "y")}, orders=0,
                      progress=False)


def test_OO2_ordered_spellings_are_accepted(system, props):
    """A list, a tuple and a dict's values keep their order; the swapped
    list is the other observable, so (0, 1) moves to its mirror."""
    values = {"first": "phi_a(x)", "second": "phi_b(y)"}.values()
    totals = [
        _evaluate(system.expand(obs, orders=[0, 2], progress=False), props,
                  (0, 1), external_times=TIMES).total
        for obs in (["phi_a(x)", "phi_b(y)"], ("phi_a(x)", "phi_b(y)"), values)
    ]
    assert totals[0] == totals[1] == totals[2]
    swapped = _evaluate(
        system.expand(["phi_b(y)", "phi_a(x)"], orders=[0, 2], progress=False),
        props, (0, 1), external_times=TIMES).total
    assert swapped != pytest.approx(totals[0], rel=0.1, abs=0.0)
