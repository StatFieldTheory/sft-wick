"""The oracle's compiler (``validation/hierarchy.py``) on the parts of a
ModelSpec that no demo exercises, against references that share no code
with it.

* A linear Gaussian model with two OU blocks loaded onto both components,
  shared white noise, two positions and ``t_min ≠ 0``: equal-time, two-time
  (both orders) and integrated moments against the Lyapunov equation of the
  same linear system, solved with ``scipy.linalg.expm``.
* A constant source, static cumulants of orders 1-3 with no index symmetry
  and a white cumulant of order 3: closed forms, which also pin the
  symmetrisation of a cumulant tensor and the joint cumulant (``target=
  'connected'``).
* Multiplicative noise at two positions, which shares its Wiener processes
  between them, and an initial law with two independent noise families:
  identities that hold exactly.
* The spatial kernels in two and three dimensions against their formulas.
* The termination guard, the JSON round trip and the refusals.
"""
from __future__ import annotations

import itertools
import json
import math
from typing import Callable

import numpy as np
import pytest
from scipy.linalg import expm

from validation.hierarchy import (Embedding, check_terminates, moments,
                                  tags_up_to)
from validation.modelspec import (Cumulant, Kernel, Leg, ModelSpec,
                                  Multiplicative, Observable, OUBlock, Poisson,
                                  model_from_dict, model_to_dict,
                                  observable_from_dict, observable_to_dict)

# ---------------------------------------------------------------------------
# Linear Gaussian model against its Lyapunov equation
# ---------------------------------------------------------------------------

GAMMA = (0.7, 1.3)
BLOCKS = (OUBlock(0.45, 0.6, Kernel("exponential", (0.8,)), (1.0, -0.4)),
          OUBlock(0.3, 1.1, Kernel("gaussian", (1.2,)), (0.2, 0.9)))
WHITE = ((0.30, 0.10), (0.10, 0.20))
T_MIN = 0.35
POS = {"x": 0.0, "y": 0.9}
LINEAR = ModelSpec(gamma=GAMMA, ou=BLOCKS, white=WHITE, t_min=T_MIN)


def _lyapunov(s1: float, s2: float) -> tuple[np.ndarray, Callable, Callable]:
    """``E[X(s2) X(s1)ᵀ]``, ``s2 ≥ s1``, for the state ``(φ_{a,i}, ζ_{b,i},
    I_{a,i})`` with ``i`` over the two positions, written out by hand, and
    the index maps of ``φ`` and ``I`` into that state."""
    pts = [POS["x"], POS["y"]]
    n = 2 * 2
    D = 3 * n
    phi = lambda a, i: 2 * a + i            # noqa: E731
    zeta = lambda b, i: n + 2 * b + i       # noqa: E731
    integ = lambda a, i: 2 * n + 2 * a + i  # noqa: E731
    A = np.zeros((D, D))
    Q = np.zeros((D, D))
    S0 = np.zeros((D, D))
    for a, i in itertools.product(range(2), repeat=2):
        A[phi(a, i), phi(a, i)] = -GAMMA[a]
        A[integ(a, i), phi(a, i)] = 1.0
        for b in range(2):
            A[phi(a, i), zeta(b, i)] = BLOCKS[b].loading[a]
        for b, j in itertools.product(range(2), repeat=2):
            Q[phi(a, i), phi(b, j)] = WHITE[a][b]
    for b, blk in enumerate(BLOCKS):
        for i, j in itertools.product(range(2), repeat=2):
            K = blk.spatial(pts[i], pts[j])
            A[zeta(b, i), zeta(b, i)] = -1.0 / blk.sigma_t
            Q[zeta(b, i), zeta(b, j)] = 2.0 * blk.lam / blk.sigma_t * K
            S0[zeta(b, i), zeta(b, j)] = blk.lam * K
    L = np.kron(np.eye(D), A) + np.kron(A, np.eye(D))
    big = np.zeros((D * D + 1, D * D + 1))
    big[:-1, :-1] = L
    big[:-1, -1] = Q.reshape(-1, order="F")
    y0 = np.append(S0.reshape(-1, order="F"), 1.0)
    sigma = (expm(big * s1) @ y0)[:-1].reshape(D, D, order="F")
    cross = expm(A * (s2 - s1)) @ sigma
    return cross, phi, integ


@pytest.mark.parametrize("a,b", [(0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("tx,ty", [(1.6, 1.6), (1.6, 0.7), (0.7, 1.6)])
def test_VH1_linear_model_matches_its_lyapunov_equation(a, b, tx, ty):
    obs = Observable([(a, "x"), (b, "y")], POS, T_MIN + max(tx, ty),
                     times={"x": T_MIN + tx, "y": T_MIN + ty})
    got = moments(LINEAR, obs, [()])[()]
    if tx >= ty:
        cross, phi, _ = _lyapunov(ty, tx)
        want = cross[phi(a, 0), phi(b, 1)]
    else:
        cross, phi, _ = _lyapunov(tx, ty)
        want = cross[phi(b, 1), phi(a, 0)]
    assert got == pytest.approx(want, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("over", [("x",), "all"])
def test_VH2_integrated_legs_match_the_lyapunov_equation(over):
    obs = Observable([(0, "x"), (1, "y")], POS, T_MIN + 1.4,
                     integrate_over=over)
    got = moments(LINEAR, obs, [()])[()]
    cross, phi, integ = _lyapunov(1.4, 1.4)
    second = integ(1, 1) if over == "all" else phi(1, 1)
    assert got == pytest.approx(cross[integ(0, 0), second], rel=1e-12,
                                abs=1e-15)


# ---------------------------------------------------------------------------
# Source, static and white cumulants against closed forms
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(20261109)
G2 = (0.6, 1.4)
K1 = (0.3, -0.2)
K2 = ((0.4, 0.1), (0.1, 0.25))
K3 = RNG.normal(size=(2, 2, 2))       # no index symmetry
W3 = RNG.normal(size=(2, 2, 2))
CUMULANT_MODEL = ModelSpec(
    gamma=G2, source=(0.5, -0.8),
    cumulants=(Cumulant(1, K1), Cumulant(2, K2), Cumulant(3, K3),
               Cumulant(3, W3, time="white")),
    t_min=-0.3)
S = 1.2                                # time from t_min
PTS = {"x": 0.0, "y": 0.7, "z": -0.5, "w": 1.1}


def _sym(t, idx):
    perms = list(itertools.permutations(idx))
    return sum(np.asarray(t)[p] for p in perms) / len(perms)


def _u(a):
    return (1.0 - np.exp(-G2[a] * S)) / G2[a]


def _tag(**powers):
    names = CUMULANT_MODEL.channels()
    return tuple(powers.get(name, 0) for name in names)


def _obs(comps, target="moment", merged=False):
    labels = "xyzw"[:len(comps)]
    pts = {**PTS, "y": PTS["x"]} if merged else PTS
    return Observable(list(zip(comps, labels)), pts, -0.3 + S, target=target)


def test_VH3_source_and_first_cumulant_shift_the_mean():
    assert CUMULANT_MODEL.channels() == ("h", "S1", "S2", "S3", "W3")
    every = tags_up_to(CUMULANT_MODEL, 3)
    assert len(every) == math.comb(5 + 3, 3) == len(set(every))
    assert max(map(sum, every)) == 3 and _tag(S1=2, W3=1) in every
    got = moments(CUMULANT_MODEL, _obs((1,)), [_tag(h=1), _tag(S1=1),
                                               _tag()])
    assert got[_tag(h=1)] == pytest.approx(-0.8 * _u(1), rel=1e-13, abs=0.0)
    assert got[_tag(S1=1)] == pytest.approx(K1[1] * _u(1), rel=1e-13, abs=0.0)
    assert got[_tag()] == 0.0


@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize("comps", [(0, 1, 1), (1, 0, 0), (1, 1, 0)])
def test_VH4_third_order_cumulants_enter_symmetrised(comps, merged):
    """``merged`` puts two legs at one position, so a jump moment sees one
    state twice."""
    got = moments(CUMULANT_MODEL, _obs(comps, merged=merged),
                  [_tag(S3=1), _tag(W3=1)])
    u = np.prod([_u(a) for a in comps])
    rate = sum(G2[a] for a in comps)
    assert got[_tag(S3=1)] == pytest.approx(_sym(K3, comps) * u, rel=1e-12,
                                            abs=0.0)
    assert got[_tag(W3=1)] == pytest.approx(
        _sym(W3, comps) * (1.0 - np.exp(-rate * S)) / rate, rel=1e-12,
        abs=0.0)


def test_VH5_four_point_moment_and_joint_cumulant_of_a_static_vector():
    comps = (0, 1, 1, 0)
    u = np.prod([_u(a) for a in comps])
    k2 = np.asarray(K2)
    pairs = (k2[0, 1] * k2[1, 0] + k2[0, 1] * k2[1, 0]
             + k2[0, 0] * k2[1, 1])
    singles = sum(K1[comps[j]] * _sym(K3, comps[:j] + comps[j + 1:])
                  for j in range(4))
    tags = [_tag(S2=2), _tag(S1=1, S3=1), _tag(S1=2, S2=1)]
    raw = moments(CUMULANT_MODEL, _obs(comps), tags)
    assert raw[_tag(S2=2)] == pytest.approx(pairs * u, rel=1e-12, abs=0.0)
    assert raw[_tag(S1=1, S3=1)] == pytest.approx(singles * u, rel=1e-12,
                                                  abs=0.0)
    assert abs(raw[_tag(S1=2, S2=1)]) > 1e-3
    connected = moments(CUMULANT_MODEL, _obs(comps, "connected"), tags)
    for t in tags:
        assert abs(connected[t]) <= 1e-15 * max(1.0, abs(raw[t]))


# ---------------------------------------------------------------------------
# Guards, JSON and refusals
# ---------------------------------------------------------------------------

def test_VH6_the_guard_refuses_a_generator_that_would_not_close():
    model = ModelSpec(gamma=(1.0,), quadratic=np.ones((1, 1, 1)))
    emb = Embedding(model, [(0.0,)])
    core = emb._parts[0]
    core.drift.append((0, 1.0, (2,), (0,)))
    with pytest.raises(ValueError, match="raises the degree"):
        check_terminates(emb)
    core.drift[-1] = (0, 1.0, (2,), (1, 0))
    with pytest.raises(ValueError, match="does not have 1 entries"):
        check_terminates(emb)

    class Unclosed(Embedding):
        def _core(self):
            sde = super()._core()
            sde.drift.append((0, 1.0, (2,), self.zero_tag()))
            return sde

    with pytest.raises(ValueError, match="raises the degree"):
        Unclosed(model, [(0.0,)])


def test_VH7_every_block_survives_a_json_round_trip():
    model = ModelSpec(
        gamma=(0.9, 1.2), source=(0.1, 0.2), quadratic=np.ones((2, 2, 2)),
        cubic=np.ones((2,) * 4), ou=BLOCKS, white=WHITE,
        multiplicative=Multiplicative(np.ones((2, 3)), np.ones((2, 3, 2)),
                                      "stratonovich"),
        poisson=Poisson(1.1, (0.5, -0.4), (0.6, 0.9), "exponential",
                        (0.4, 1.5), (4, 3)),
        cumulants=(Cumulant(3, K3), Cumulant(2, K2, "white")), t_min=0.25)
    assert model_from_dict(json.loads(json.dumps(model_to_dict(model)))) == model
    assert model.channels() == ("h", "F", "G", "g", "S3", "W2", "P3", "P4")
    obs = Observable([(0, "x"), Leg(1, "y")], {"x": (0.0, 1.0), "y": (2, 3)},
                     2.0, times={"y": 1.5}, integrate_over=("x",),
                     target="connected")
    assert observable_from_dict(json.loads(json.dumps(
        observable_to_dict(obs)))) == obs


@pytest.mark.parametrize("build,error", [
    (lambda: ModelSpec(gamma=(1.0, 1.0), white=((1.0, 0.2), (0.0, 1.0))),
     "symmetric"),
    (lambda: ModelSpec(gamma=(1.0, 1.0), quadratic=np.ones((2, 2))),
     "shape"),
    (lambda: Observable([(0, "x"), (1, "x")], {"x": 0.0}, 1.0),
     "distinct"),
    (lambda: Observable([(0, "x")], {"x": 0.0}, 1.0, times={"x": 0.5},
                        integrate_over="all"), "integrated"),
    (lambda: ModelSpec(gamma=(1.0, 1.0), white=((1.0, 2.0), (2.0, 1.0))),
     "semi-definite"),
    (lambda: ModelSpec(gamma=(1.0,), cumulants=(Cumulant(2, [[1.0]]),
                                                Cumulant(2, [[2.0]]))),
     "two cumulants"),
    (lambda: Cumulant(2, np.ones((2, 3))), "shape"),
    (lambda: Kernel("exponential", (-0.8,)), "positive"),
    (lambda: OUBlock(-0.5, 0.6, Kernel("constant"), (1.0,)), "non-negative"),
    (lambda: ModelSpec(gamma=(1.0, 1.0),
                       multiplicative=Multiplicative([[1.0]], [[[0.0]]])),
     "g0 has 2 rows"),
    (lambda: ModelSpec(gamma=(1.0, 1.0), poisson=Poisson(1.0, (1.0,), (1.0,))),
     "amplitude has 2"),
    (lambda: Poisson(1.0, (1.0,), (1.0,), "white", tau=(1.0,)), "no tau"),
    (lambda: Poisson(1.0, (1.0,), (1.0,), orders=(2,)), "m >= 3"),
    (lambda: Observable([(0, "x")], {"x": 0.0}, 1.0, times={"x": 1.5}),
     "after t_final"),
    (lambda: Observable([(0, "x")], {"x": 0.0}, 1.0, times={"y": 0.5}),
     "unknown labels"),
    (lambda: Observable([(0, "x")], {"x": 0.0}, 1.0, integrate_over=("y",)),
     "unknown labels"),
    (lambda: Observable([(0, "x"), (1, "y")], {"x": 0.0, "y": (0.0, 1.0)},
                        1.0), "dimension"),
])
def test_VH8_the_spec_refuses_ill_formed_input(build, error):
    with pytest.raises(ValueError, match=error):
        build()


@pytest.mark.parametrize("obs", [
    Observable([(0, "x"), Leg(0, "y", "psi")], POS, 1.0),
    Observable([(0, "x"), (1, "y"), (0, "z")], {**POS, "z": 0.2}, 2.0,
               times={"x": 1.0, "y": 1.5}),
])
def test_VH9_unsupported_observables_raise(obs):
    with pytest.raises(NotImplementedError):
        moments(LINEAR, obs, [()])


@pytest.mark.parametrize("tags,obs,error", [
    ([(1,)], _obs((0,)), "non-negative entries"),
    ([(0, -1, 0, 0, 1)], _obs((0,)), "non-negative entries"),
    ([_tag()], _obs((2,)), "component 2"),
    ([_tag()], Observable([(0, "x")], PTS, 1.0, times={"x": -0.5}),
     "before t_min"),
    ([_tag()], Observable([(0, "x")], PTS, -0.4), "before t_min"),
])
def test_VH10_moments_refuses_tags_and_legs_the_model_does_not_have(
        tags, obs, error):
    with pytest.raises(ValueError, match=error):
        moments(CUMULANT_MODEL, obs, tags)


# ---------------------------------------------------------------------------
# Identities: shared multiplicative noise, independent initial families
# ---------------------------------------------------------------------------

G0 = RNG.normal(scale=0.4, size=(2, 2))
G1 = RNG.normal(scale=0.3, size=(2, 2, 2))
F2 = RNG.normal(scale=0.4, size=(2, 2, 2))


@pytest.mark.parametrize("interpretation", ["ito", "stratonovich"])
def test_VH11_shared_multiplicative_noise_makes_positions_identical(
        interpretation):
    """With the Wiener processes shared by every position and no other
    noise, ``φ(x)`` and ``φ(y)`` solve one SDE from one start, so every
    moment is the same at distinct positions as at one."""
    model = ModelSpec(gamma=(0.7, 1.6), quadratic=F2,
                      multiplicative=Multiplicative(G0, G1, interpretation),
                      t_min=0.2)
    assert model.channels() == ("F", "g")
    tags = tags_up_to(model, 3)
    for comps in [(0, 1), (1, 1), (0, 1, 1)]:
        labels = "xyz"[:len(comps)]
        apart = Observable(list(zip(comps, labels)),
                           {"x": 0.0, "y": 0.8, "z": -0.5}, 1.6)
        one = Observable(list(zip(comps, labels)),
                         {"x": 0.0, "y": 0.0, "z": 0.0}, 1.6)
        got, want = moments(model, apart, tags), moments(model, one, tags)
        assert any(abs(v) > 1e-3 for v in want.values())
        for t in tags:
            assert got[t] == pytest.approx(want[t], rel=1e-12, abs=1e-16)


def test_VH12_additive_multiplicative_noise_is_white_noise():
    """``g1 = 0`` leaves the constant columns ``g0``, which is white noise
    with ``S = g0 g0ᵀ``, the same at every position, and no ``g`` channel."""
    ou = (BLOCKS[0],)
    as_mult = ModelSpec(gamma=GAMMA, quadratic=F2, ou=ou, multiplicative=
                        Multiplicative(G0, np.zeros((2, 2, 2)), "stratonovich"))
    as_white = ModelSpec(gamma=GAMMA, quadratic=F2, ou=ou, white=G0 @ G0.T)
    assert as_mult.channels() == as_white.channels() == ("F",)
    for a, b in itertools.product(range(2), repeat=2):
        obs = Observable([(a, "x"), (b, "y")], POS, 1.3, times={"x": 0.8})
        got = moments(as_mult, obs, [(0,), (2,)])
        want = moments(as_white, obs, [(0,), (2,)])
        for t in want:
            assert got[t] == pytest.approx(want[t], rel=1e-12, abs=1e-16)


@pytest.mark.parametrize("other", ["ou", "poisson"])
def test_VH13_independent_noise_families_split_the_tag(other):
    """A static first cumulant alone is a shift by ``κ1_a u_a(t)``.  With
    a second, independent family and no drift non-linearity, the coefficient
    at ``S1`` is ``Σ_j κ1_{a_j} u_{a_j}`` times the moment of the other legs
    in the model without the shift."""
    k1 = (0.3, -0.2)
    if other == "ou":
        base = ModelSpec(gamma=(0.7, 1.3), ou=(BLOCKS[0],))
        comps, tag = (0, 1, 1), ()
    else:
        base = ModelSpec(gamma=(1.0, 1.0), poisson=Poisson(
            1.5, (1.0, -0.7), (0.6, 1.1), "exponential", (0.3, 2.0), (3,)))
        comps, tag = (0, 1, 1, 0), (1,)
    shifted = ModelSpec(gamma=base.gamma, ou=base.ou, poisson=base.poisson,
                        cumulants=(Cumulant(1, k1),))
    s = 1.4
    labels = "xyzw"[:len(comps)]
    pts = {"x": 0.0, "y": 0.8, "z": -0.5, "w": 1.3}
    got = moments(shifted, Observable(list(zip(comps, labels)), pts, s),
                  [(1,) + tag])[(1,) + tag]
    want = 0.0
    for j, a in enumerate(comps):
        u = (1.0 - math.exp(-base.gamma[a] * s)) / base.gamma[a]
        rest = [(c, lab) for i, (c, lab) in enumerate(zip(comps, labels))
                if i != j]
        want += k1[a] * u * moments(base, Observable(rest, pts, s),
                                    [tag])[tag]
    assert abs(want) > 1e-4
    assert got == pytest.approx(want, rel=1e-12, abs=0.0)


# ---------------------------------------------------------------------------
# Kernels in two and three dimensions
# ---------------------------------------------------------------------------

def _legendre(coeffs, c):
    polys = (1.0, c, 0.5 * (3 * c * c - 1), 0.5 * (5 * c ** 3 - 3 * c))
    return sum(a * p for a, p in zip(coeffs, polys))


def _inhomogeneous(s, c, length, x, y):
    g = math.exp(-(sum((v - c) ** 2 for v in x)
                   + sum((v - c) ** 2 for v in y)) / (2 * length ** 2))
    return g * math.exp(-math.dist(x, y) ** 2 / (2 * s ** 2))


KERNELS = [
    ("constant", (), lambda x, y: 1.0),
    ("exponential", (0.8,), lambda x, y: math.exp(-math.dist(x, y) / 0.8)),
    ("gaussian", (1.1,),
     lambda x, y: math.exp(-math.dist(x, y) ** 2 / (2 * 1.1 ** 2))),
    ("damped_cosine", (1.5, 1.6), lambda x, y: math.exp(
        -math.dist(x, y) / 1.5) * math.cos(1.6 * math.dist(x, y))),
    ("legendre", (0.25, 0.6, 0.3, 0.15), lambda x, y: _legendre(
        (0.25, 0.6, 0.3, 0.15), sum(a * b for a, b in zip(x, y))
        / math.hypot(*x) / math.hypot(*y))),
    ("inhomogeneous", (0.9, 0.3, 1.2),
     lambda x, y: _inhomogeneous(0.9, 0.3, 1.2, x, y)),
]


@pytest.mark.parametrize("x,y", [((0.2, -0.7), (-0.4, 0.6)),
                                 ((0.2, -0.3, 0.5), (-0.4, 0.6, 0.1))])
@pytest.mark.parametrize("kind,params,formula", KERNELS,
                         ids=[k[0] for k in KERNELS])
def test_VH14_kernels_in_two_and_three_dimensions(kind, params, formula,
                                                  x, y):
    kernel = Kernel(kind, params)
    assert kernel(x, y) == pytest.approx(formula(x, y), rel=1e-14, abs=0.0)
    assert kernel(x, x) == pytest.approx(formula(x, x), rel=1e-14, abs=0.0)
