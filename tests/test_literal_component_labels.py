"""Numeric literal component labels read one way in every resolver.

A component label that parses as an integer is a literal component, and it
counts from 1: ``phi('1', 'x')`` is component 0.  Coupling symbols and
propagator legs have always read it so.  The Kronecker delta that
``diag_C`` / ``diag_R`` / ``iso_C`` / ``iso_R`` insert between two external
labels read it from 0, so ``<phi_1(x) phi_b(y)>`` under ``diag_C`` came out
as ``C_00`` at ``b = 1`` and as 0 at ``b = 0``, the two values swapped.  A
literal outside ``1..n_components`` (``'0'``, which indexing wrapped to the
last component) is now refused where the observable enters.

References share no code with the package: the exact covariance and
response of a two-component Ornstein-Uhlenbeck process with diagonal drift
and white noise started at ``t_min``, Isserlis' theorem, and the exact Itô
moment hierarchy of ``examples/reference/ito_moments.py``.  The model is
generic: distinct rates and noise amplitudes, a quadratic coupling with
entries of both signs and no index symmetry, distinct times, ``t_min != 0``.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

import sft_wick as sw
from sft_wick import Action, Field, Vertex
from sft_wick._coupling_plan import compile_coupling_plan
from sft_wick.evaluate import integrate_diagrams
from sft_wick.expressions import KroneckerDelta, Product, Rational, Sum, Symbol
from sft_wick.perturbation import (
    _eval_symbolic,
    _eval_symbolic_batched,
    compute_moment,
    compute_moment_numerical,
)
from sft_wick.workflow.specs import ConstantImpulse

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "examples" / "reference")
)
import ito_moments as im  # noqa: E402

GAMMA = np.array([0.7, 1.9])
S = np.diag([0.8, 1.3])
GAMMA_ISO, S_ISO = 0.9, 0.6
T_MIN, T_F = 0.4, 3.0
TIMES = {"x": 2.3, "y": 1.7, "z": 2.9, "w": 1.1}
F_BARE = np.array(
    [[[-0.829, -0.526], [0.603, 0.164]], [[-0.812, -0.134], [-0.042, -0.681]]]
)
METHODS = ("gauss_legendre", "qmc_vectorized", "qmc", "nquad")
ENGINES = ("compute_moment", "compute_moment_numerical")

phi = Field("phi", "physical", n_components=2)
psi = Field("psi", "response", n_components=2)


def C_exact(a, b, t1, t2, gamma=GAMMA, s=S):
    """Covariance of dX = -gamma X dt + dW, <dW dW^T> = s dt, X(t_min) = 0."""
    if a != b:
        return 0.0
    g = gamma[a]
    return s[a, a] / (2 * g) * (
        np.exp(-g * abs(t1 - t2)) - np.exp(-g * (t1 + t2 - 2 * T_MIN))
    )


def R_exact(a, b, t1, t2, gamma=GAMMA):
    """Response of the same process: exp(-gamma (t1 - t2)) for t1 > t2."""
    if a != b or t1 <= t2:
        return 0.0
    return float(np.exp(-gamma[a] * (t1 - t2)))


def _close(got, want, rel=1e-13):
    return abs(got - want) <= rel * max(1.0, abs(want))


def _system(gamma, s):
    return sw.System(
        field=sw.FieldSpec("phi", 2),
        linear=sw.DiagonalA(gamma=list(gamma)),
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=1.0),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            ),
            sigma2=ConstantImpulse(s),
        ),
        t_min=T_MIN,
    )


def _cache(gamma, s, diag_C):
    return (
        _system(gamma, s)
        .propagators(
            t_max=T_F + 0.5,
            c_closed_form="auto",
            c_closed_form_only=True,
            c_closed_form_vectorized=True,
            diag_C=diag_C,
            progress=False,
        )
        .cache
    )


@pytest.fixture(scope="module")
def caches():
    """Distinct rates: diagonal C, matrix R.  Keyed by the table's diag_C."""
    return {dc: _cache(GAMMA, S, dc) for dc in (False, True)}


@pytest.fixture(scope="module")
def iso_cache():
    """Equal rates and amplitudes: C = c I and a scalar R."""
    return _cache([GAMMA_ISO] * 2, S_ISO * np.eye(2), True)


def _terms(engine, ops, order, action=None, couplings=None, **flags):
    """The diagram terms of ``order`` alone, from either L0 engine."""
    action = Action([]) if action is None else action
    if engine == "compute_moment":
        return compute_moment(ops, action, order=order, **flags).diagram_terms(order)
    res = compute_moment_numerical(ops, action, order, couplings or {}, **flags)
    return res.get(order, [])


def _value(dts, cache, fixed, method, ops=None, couplings=None, n_gauss=8):
    """The integrated total; the points of ``ops`` sit at ``TIMES``, or
    every external point at ``T_F`` when ``ops`` is None."""
    points = () if ops is None else [op.spatial_arg for op in ops]
    total, _ = integrate_diagrams(
        dts,
        couplings or {},
        T_F,
        cache,
        t_min=T_MIN,
        method=method,
        fixed_indices=fixed,
        external_times={p: TIMES[p] for p in points} or None,
        n_samples=64,
        seed=3,
        n_gauss=n_gauss,
    )
    return complex(total)


# --------------------------------------------------------------------------
# Order 0 against the exact two-point functions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("table_diag_C", [False, True])
@pytest.mark.parametrize("literal_first", [True, False])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("engine", ENGINES)
def test_LK1_two_point_C_with_a_literal_label_is_exact(
    caches, engine, method, literal_first, table_diag_C
):
    """<phi_l(x) phi_b(y)> = delta_{l-1, b} C_{l-1, l-1}(t_x, t_y) under
    diag_C, for both literals, both pins and both operator orders.  Measured
    worst deviation 1.1e-16.  The delta used to read the literal from 0,
    which returned C_00 = 0.315 at b = 1 and 0 at b = 0."""
    for literal, b in itertools.product(("1", "2"), (0, 1)):
        ops = [phi(literal, "x"), phi("b", "y")]
        if not literal_first:
            ops = ops[::-1]
        dts = _terms(engine, ops, 0, diag_C=True)
        got = _value(dts, caches[table_diag_C], {"b": b}, method, ops)
        want = C_exact(int(literal) - 1, b, TIMES["x"], TIMES["y"])
        assert _close(got, want), (literal, b, got, want)
    assert C_exact(0, 0, TIMES["x"], TIMES["y"]) > 0.3


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("engine", ENGINES)
def test_LK2_two_point_R_with_a_literal_label_is_exact(caches, engine, method):
    """<phi_l(x) psi_b(y)> = R_{l-1, b}(t_x, t_y) under diag_R with a matrix
    R (distinct rates).  The integrators rotate the -i of one external psi
    onto the real axis.  The old reading gave R_00 = 0.657 at b = 1 and 0 at
    b = 0 for the literal '1'."""
    for literal, b in itertools.product(("1", "2"), (0, 1)):
        for ops in (
            [phi(literal, "x"), psi("b", "y")],
            [psi("b", "y"), phi(literal, "x")],
        ):
            dts = _terms(engine, ops, 0, diag_R=True)
            got = _value(dts, caches[False], {"b": b}, method, ops)
            want = R_exact(int(literal) - 1, b, TIMES["x"], TIMES["y"])
            assert _close(got, want), (literal, b, got, want)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("engine", ENGINES)
def test_LK3_isotropic_propagators_with_a_literal_label(iso_cache, engine, method):
    """iso_C and iso_R drop the component indices from the propagators, so
    the delta alone carries the components: <phi_l phi_b> = delta c and
    <phi_l psi_b> = delta r.  The delta used to vanish for l = 1, b = 0
    (0 against c = 0.176)."""
    gamma, s = np.array([GAMMA_ISO] * 2), S_ISO * np.eye(2)
    for literal, b in itertools.product(("1", "2"), (0, 1)):
        a = int(literal) - 1
        ops = [phi(literal, "x"), phi("b", "y")]
        dts = _terms(engine, ops, 0, iso_C=True)
        got = _value(dts, iso_cache, {"b": b}, method, ops)
        want = C_exact(a, b, TIMES["x"], TIMES["y"], gamma, s)
        assert _close(got, want), ("C", literal, b, got, want)
        ops = [phi(literal, "x"), psi("b", "y")]
        dts = _terms(engine, ops, 0, iso_R=True)
        got = _value(dts, iso_cache, {"b": b}, method, ops)
        want = R_exact(a, b, TIMES["x"], TIMES["y"], gamma)
        assert _close(got, want), ("R", literal, b, got, want)


@pytest.mark.parametrize("engine", ENGINES)
def test_LK4_four_point_isserlis_with_literal_labels(caches, engine):
    """<phi_1(x) phi_b(y) phi_2(z) phi_d(w)> under diag_C against Isserlis'
    theorem for every (b, d).  (b, d) = (0, 1) and (1, 0) used to return 0
    against 3.3e-3 and 5.4e-3."""
    labels = ("x", "y", "z", "w")
    nonzero = 0
    for b, d in itertools.product((0, 1), repeat=2):
        comps = dict(zip(labels, (0, b, 1, d)))
        want = sum(
            C_exact(comps[p], comps[q], TIMES[p], TIMES[q])
            * C_exact(comps[r], comps[u], TIMES[r], TIMES[u])
            for (p, q), (r, u) in (
                (("x", "y"), ("z", "w")),
                (("x", "z"), ("y", "w")),
                (("x", "w"), ("y", "z")),
            )
        )
        ops = [phi("1", "x"), phi("b", "y"), phi("2", "z"), phi("d", "w")]
        dts = _terms(engine, ops, 0, diag_C=True)
        got = _value(dts, caches[False], {"b": b, "d": d}, "gauss_legendre", ops)
        assert _close(got, want), (b, d, got, want)
        nonzero += want != 0.0
    assert nonzero == 2


# --------------------------------------------------------------------------
# Order 1 against the exact Itô moment hierarchy
# --------------------------------------------------------------------------


def _hierarchy_first_order(components):
    """E[X_i X_j X_k](T_F) at first order in F for
    dX = (-gamma X + F X X) dt + dW, X(t_min) = 0.  No sft_wick code."""
    sde = im.PolySDE(D=2, n_tags=1)
    sde.add_linear_drift(np.diag(-GAMMA))
    sde.add_constant_diffusion(S)
    sde.add_quadratic_drift(F_BARE, tag=(1,))
    target = im.unit(2, *components)
    return im.solve(sde, [(target, (1,))], [T_F - T_MIN])[(target, (1,))][0]


@pytest.mark.parametrize("method", ["gauss_legendre", "nquad"])
@pytest.mark.parametrize("engine", ENGINES)
def test_LK5_three_point_first_order_against_ito_hierarchy(caches, engine, method):
    """<phi_l(x) phi_b(y) phi_m(z)> at first order in F, all three at T_F,
    under diag_R and diag_C.  Its disconnected part C_{l-1, b} <phi_m>
    carries the delta.  The letter spelling with the same components is the
    control.  Measured deviation 1.1e-16 (16 Gauss-Legendre nodes) and
    3.3e-16 (nquad); the literal and letter spellings agree bit for bit.
    The old reading returned -0.138 against -0.330 for (1, b=0, 2)."""
    action = Action([Vertex(fields=[psi, phi, phi], coupling="F", local=True)])
    couplings = {"F": -1j * F_BARE}
    flags = {"diag_R": True, "diag_C": True}
    checked = 0
    for (l1, l2), b in itertools.product((("1", "2"), ("2", "1")), (0, 1)):
        a, c = int(l1) - 1, int(l2) - 1
        want = _hierarchy_first_order((a, b, c))
        ops = [phi(l1, "x"), phi("b", "y"), phi(l2, "z")]
        dts = _terms(engine, ops, 1, action, couplings, **flags)
        got = _value(dts, caches[False], {"b": b}, method, None, couplings, 16)
        control_ops = [phi("a", "x"), phi("b", "y"), phi("c", "z")]
        control = _terms(engine, control_ops, 1, action, couplings, **flags)
        pins = {"a": a, "b": b, "c": c}
        ref = _value(control, caches[False], pins, method, None, couplings, 16)
        assert _close(ref, want, 1e-9), ("control", l1, l2, b, ref, want)
        assert _close(got, want, 1e-9), (l1, l2, b, got, want)
        checked += abs(want) > 1e-3
    assert checked >= 3


# --------------------------------------------------------------------------
# The resolvers themselves
# --------------------------------------------------------------------------


def _large(expr):
    """A sum big enough (>= 16 nodes) for compile_coupling_plan to lower."""
    return Sum(tuple(Product((Rational(1, 2), expr)) for _ in range(8)))


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("planned", [False, True])
def test_LK6_delta_reads_a_literal_as_symbols_and_legs_do(planned, batched):
    """KroneckerDelta('1', 'b') is 1 exactly when b pins component 0, the
    component Symbol('A', ('1',)) reads, in the scalar and batched
    interpreters and in a compiled plan; a pin on the literal itself still
    takes precedence."""
    values = {"A": np.array([3.0, 7.0])}
    expr = Product((Symbol("A", ("1",)), KroneckerDelta("1", "b")))
    expr2 = Product((Symbol("A", ("2",)), KroneckerDelta("b", "2")))
    for e, pins, want in (
        (expr, {"b": 0}, 3.0),
        (expr, {"b": 1}, 0.0),
        (expr, {"1": 1, "b": 1}, 7.0),
        (expr2, {"b": 1}, 7.0),
        (expr2, {"b": 0}, 0.0),
    ):
        tree = _large(e) if planned else e
        scale = 4.0 if planned else 1.0
        if planned:
            plan = compile_coupling_plan(tree)
            assert plan is not None
            got = (
                plan.batched(tree, values, pins, 3)
                if batched
                else plan.scalar(tree, values, pins)
            )
        elif batched:
            got = _eval_symbolic_batched(tree, values, pins, 3)
        else:
            got = _eval_symbolic(tree, values, pins)
        assert np.all(np.asarray(got) == scale * want), (e, pins, got)


def test_LK7_literal_component_counts_from_one():
    from sft_wick.indices import literal_component

    spellings = ("1", "2", "01", "+1", " 3 ")
    assert [literal_component(s) for s in spellings] == [0, 1, 0, 0, 2]
    assert literal_component("0") == -1
    assert literal_component("a") is None and literal_component("i_0") is None


# --------------------------------------------------------------------------
# Out-of-range literals are refused where the observable enters
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_LK8_out_of_range_literal_is_refused(engine):
    """'0' was read as component -1, the last one: <phi_0(x) phi_0(y)>
    returned C_11.  A literal above n_components met a delta alone under
    iso_C and returned 0, or raised IndexError elsewhere.  Both ends of
    1..n_components are accepted."""
    for label in ("0", "3", "-1", "00"):
        for ops in (
            [phi(label, "x"), phi("b", "y")],
            [phi("b", "x"), psi(label, "y")],
        ):
            with pytest.raises(ValueError, match="literal labels count from 1"):
                _terms(engine, ops, 0, diag_C=True)
    for label in ("1", "2", "02"):
        assert _terms(engine, [phi(label, "x"), phi("b", "y")], 0, diag_C=True)
    scalar = Field("phi", "physical", n_components=1)
    assert _terms(engine, [scalar("x"), scalar("y")], 0)
