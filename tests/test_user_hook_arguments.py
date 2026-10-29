r"""User hooks outside the propagator pipeline get copies of what is read again.

UH1: ``SpectralDensity.from_callable`` called ``rho(h)`` and then built the
nodes from the same ``h``.  A density callable that shifts its argument in
place (``h -= 0.2``) moved every node by 0.2; one that fills and returns its
argument made the nodes equal to the density values.  At order 0 through
``propagators_from_cache`` and ``Expansion.evaluate`` the correlator came out
51 % high for the shift and 12.8 times too large for the returned argument,
with no error.  ``rho`` now receives a copy.  The copy changes the meaning
of one write: a ``rho`` that reverses its argument in place and returns the
density at the reversed points was right before and gave the order-0
correlator 87 % high with the copy alone.  The package cannot tell which
order the densities follow, so a ``rho`` that reorders its argument raises.

UH2: ``SpectralDensity.average`` passed the density's own node array to
``f``.  An ``f`` that doubles its argument in place returned the right
average but changed the frozen, content-hashed density: every node moved
(by up to 3.34), the hash changed, a later average came out 0.0157 against
0.0907, and the order-0 correlator of a cache built on the density before
the call dropped to 0.108 of its value.  ``f`` now receives a copy, and the
stored nodes and weights are read-only, so a direct write raises.  The flag
is set again after unpickling, because neither ``joblib`` nor
``copy.deepcopy`` keeps it.

UH3: ``solve_self_consistency`` handed a user ``distance`` the live
proposal, the live state and, on the first iteration, the caller's
``initial``.  On a contraction that converges in 27 iterations, a distance
that subtracts in place (``a -= b``) made the run diverge after 6; one that
writes its second argument overwrote the caller's ``initial``
(``[0.25, -0.6]`` became ``[-6.65, 3.95]``) and, at damping 0.35, ran out of
iterations 0.83 from the fixed point; one that sorts its arguments in place
reported convergence at a state 2.8 from the fixed point.  A user distance
now receives copies; the default ``max_abs_distance`` does not write and is
called as before.

UH4: the ``callback`` received the live post-mixing state, which the loop
reads again (distance, cycle test, next step, result).  A display callback
that rounds the state in place to three decimals stopped the run
converging: ``max_iter`` after 200 iterations with an error of 1.1e-4,
against convergence in 27 with 2.5e-13.  It now receives a copy when one is
given.

UH5: ``build_R_contracted_callable`` handed the raw kernel the caller's
position array and reused it for all ``|chi|^m`` raw calls.  A raw kernel
that centres its positions in place gave a value too small by a relative
0.997 and overwrote the caller's positions (``[0.1, -0.35, 0.6]`` became
``[-32.3, -32.75, -31.8]``); through L1 the order-1 three-point function
came out 566 times too small.  The wrapper now keeps a private copy and
passes each raw call its own.  It also held the caller's ``chi_grid`` by
reference while its trapezoid weights were fixed at build time, so a later
``chi += 0.25`` by the caller changed K[0,0,0] by a relative 0.37; it now
copies the grid when it is built.

UH6: ``solve_self_consistency`` used the caller's ``initial`` as its
iteration-1 state.  A step that fills and returns a buffer it owns, started
from that buffer (a solver that updates ``self.G`` in place and returns it,
started from ``self.G``), overwrote the state before the residual was
taken: converged after one iteration with a residual of 0.0, at a state
0.338 from the fixed point, at damping 0 and 0.35.  The loop now starts
from a copy of ``initial``.

References share no code with the package: scipy ``quad`` over the
continuous density for R and C; the closed-form fixed point
``(I - A)^{-1} b``; and the leg-factorised trapezoid sum computed with
``np.trapezoid`` for the R-contracted kernel.  Each writing hook is compared
with the same hook that does not write.  Inputs are generic: two
components, entries of both signs, ``t_min != 0``, distinct points and
distinct times.
"""
from __future__ import annotations

import copy
import io
import pickle
import warnings
from math import exp

import numpy as np
import pytest
from scipy.integrate import quad

import sft_wick as sw
from sft_wick import SpectralDensity, spectral_cache
from sft_wick.selfconsistency import solve_self_consistency
from sft_wick.workflow import build_R_contracted_callable
from sft_wick.workflow.propagators import propagators_from_cache


# ---------------------------------------------------------------------------
# UH1-UH2: spectral densities
# ---------------------------------------------------------------------------

LO, HI, N_NODES, SHIFT = 0.45, 3.2, 24, 0.15
D_NOISE, T_MIN = 0.75, 0.6
T1, T2 = 2.9, 1.4
H_OFFSET = 0.2


def _rho(h):
    """An entire, strictly positive density, so 24 Gauss-Legendre nodes
    reproduce its integrals to round-off."""
    h = np.asarray(h, dtype=float)
    return (h - 0.3) ** 2 * np.exp(-1.1 * h) + 0.05


def _rho_clean(h):
    return _rho(h)


def _rho_write(h):
    """Evaluates, then shifts its argument in place.  The value does not
    depend on the write, so the weights compare exactly on any platform
    (``(h - 0.2) + 0.2`` does not always round back to ``h``)."""
    values = _rho(h)
    h -= H_OFFSET
    return values


def _rho_return(h):
    """Fills its argument with the density and returns it."""
    values = _rho(h)
    h[...] = values
    return h


RHOS = {"clean": _rho_clean, "write": _rho_write, "return": _rho_return}


def _quad_average(g):
    """``∫ rho(h) g(h + SHIFT) dh / ∫ rho(h) dh`` on ``[LO, HI]``."""
    norm = quad(_rho, LO, HI, epsabs=0, epsrel=1e-13)[0]
    num = quad(lambda h: _rho(h) * g(h + SHIFT), LO, HI, epsabs=0,
               epsrel=1e-13, limit=200)[0]
    return num / norm


def _R_reference(dt):
    return _quad_average(lambda k: exp(-k * dt))


def _C_reference(t1, t2):
    return _quad_average(lambda k: D_NOISE / k * (
        exp(-k * abs(t1 - t2)) - exp(-k * (t1 + t2 - 2 * T_MIN))))


@pytest.fixture(scope="module")
def order_zero():
    """An order-0 two-point expansion of a two-component system; the
    spectral cache supplies R and C, so the system's own rates do not enter."""
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[0.9, 1.7]),
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.4, sigma_t=0.8),
            spatial=sw.ExponentialSpatial(sigma_x=1.2))),
        t_min=T_MIN,
    )
    return system.expand(("phi_a(x)", "phi_b(y)"), orders=0, progress=False)


def _order_zero_C(expansion, density):
    props = propagators_from_cache(
        spectral_cache(density, D_NOISE, n_components=2, t_min=T_MIN))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.real(expansion.evaluate(
            props, positions={"x": 0.3, "y": -0.45}, t_final=T1,
            external_times={"x": T1, "y": T2}, component_pair=(1, 1),
            orders=[0], method="nquad").total))


@pytest.mark.parametrize("name", sorted(RHOS))
def test_UH1_from_callable_is_independent_of_what_rho_does_to_its_argument(
        order_zero, name):
    clean = SpectralDensity.from_callable(_rho_clean, LO, HI, N_NODES,
                                          shift=SHIFT)
    dens = SpectralDensity.from_callable(RHOS[name], LO, HI, N_NODES,
                                         shift=SHIFT)
    np.testing.assert_array_equal(dens.nodes, clean.nodes)
    np.testing.assert_array_equal(dens.weights, clean.weights)
    cache = spectral_cache(dens, D_NOISE, n_components=2, t_min=T_MIN)
    np.testing.assert_allclose(cache.R_time(T1, T2), _R_reference(T1 - T2),
                               rtol=1e-12, atol=0)
    # The writing rho gave 0.0497 and the returning one 0.422 here.
    np.testing.assert_allclose(_order_zero_C(order_zero, dens),
                               _C_reference(T1, T2), rtol=1e-12, atol=0)


def _rho_reverse(h):
    """Reverses its argument in place and returns the density at the
    reversed points."""
    h[:] = h[::-1].copy()
    return _rho(h)


def _rho_reverse_after(h):
    """Returns the density at the points as given, then reverses them."""
    values = _rho(h)
    h[:] = h[::-1].copy()
    return values


@pytest.mark.parametrize("rho", [_rho_reverse, _rho_reverse_after],
                         ids=["reverse-then-evaluate", "evaluate-then-reverse"])
def test_UH1_a_rho_that_reorders_its_argument_is_refused(rho):
    """The copy fixes every write except a reordering: the densities may
    follow either order, and nothing tells the package which.  Before the
    copy the first of these was right and the second wrong; with the copy
    and no check, the first gave the order-0 C 87 % high (0.0614 against
    0.0329) and R 64 % high, with no error."""
    with pytest.raises(ValueError, match="reordered"):
        SpectralDensity.from_callable(rho, LO, HI, N_NODES, shift=SHIFT)


def test_UH1_a_rho_that_sorts_its_ascending_argument_is_accepted():
    """The nodes arrive in ascending order, so an in-place ascending sort
    does not move them and is not a reordering."""
    def rho(h):
        h.sort()
        return _rho(h)

    dens = SpectralDensity.from_callable(rho, LO, HI, N_NODES, shift=SHIFT)
    clean = SpectralDensity.from_callable(_rho_clean, LO, HI, N_NODES,
                                          shift=SHIFT)
    np.testing.assert_array_equal(dens.nodes, clean.nodes)
    np.testing.assert_array_equal(dens.weights, clean.weights)


def _f_clean(h):
    return np.exp(-1.6 * h)


def _f_write(h):
    """Doubles its argument in place; ``0.8 * 2h`` is ``1.6 h`` exactly."""
    h *= 2.0
    return np.exp(-0.8 * h)


def _f_return(h):
    values = np.exp(-1.6 * h)
    h[...] = values
    return h


FS = {"clean": _f_clean, "write": _f_write, "return": _f_return}


@pytest.mark.parametrize("name", sorted(FS))
def test_UH2_average_leaves_the_density_and_its_caches_unchanged(
        order_zero, name):
    dens = SpectralDensity.from_callable(_rho_clean, LO, HI, N_NODES,
                                         shift=SHIFT)
    nodes, weights, key = dens.nodes.copy(), dens.weights.copy(), hash(dens)
    want_C = _C_reference(T1, T2)
    before = _order_zero_C(order_zero, dens)
    cache = spectral_cache(dens, D_NOISE, n_components=2, t_min=T_MIN)
    assert cache.density is dens

    got = dens.average(FS[name])
    np.testing.assert_allclose(got, _quad_average(lambda k: exp(-1.6 * k)),
                               rtol=1e-12, atol=0)

    np.testing.assert_array_equal(dens.nodes, nodes)
    np.testing.assert_array_equal(dens.weights, weights)
    assert hash(dens) == key
    assert dens == SpectralDensity.from_callable(_rho_clean, LO, HI, N_NODES,
                                                 shift=SHIFT)
    later = dens.average(lambda h: np.exp(-1.3 * h))
    np.testing.assert_allclose(later, _quad_average(lambda k: exp(-1.3 * k)),
                               rtol=1e-12, atol=0)
    # A cache built before the call keeps its values.
    np.testing.assert_allclose(cache.R_time(T1, T2), _R_reference(T1 - T2),
                               rtol=1e-12, atol=0)
    after = _order_zero_C(order_zero, dens)
    assert after == before
    np.testing.assert_allclose(after, want_C, rtol=1e-12, atol=0)


def _assert_read_only(dens):
    for arr in (dens.nodes, dens.weights):
        assert not arr.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 1.0


def test_UH2_stored_arrays_are_read_only_and_the_caller_keeps_its_own():
    nodes = np.array([0.35, 1.2, 2.6, 0.9])
    weights = np.array([0.4, 1.1, 0.25, 0.7])
    dens = SpectralDensity(nodes, weights)
    _assert_read_only(dens)
    # The caller's arrays are neither stored nor frozen.
    assert nodes.flags.writeable and weights.flags.writeable
    assert not np.shares_memory(dens.nodes, nodes)
    assert not np.shares_memory(dens.weights, weights)
    nodes[0] = 9.0
    assert dens.nodes[0] == 0.35
    for built in (SpectralDensity.delta(0.8),
                  SpectralDensity.from_samples(np.array([0.3, 0.7, 1.9, 2.4]),
                                               n_nodes=2, shift=0.1),
                  SpectralDensity.from_callable(_rho_clean, LO, HI, 6)):
        _assert_read_only(built)
    # Building from another density's (read-only) arrays still works.
    shifted = spectral_cache(dens, D_NOISE, shift=0.2).density
    np.testing.assert_allclose(shifted.nodes, dens.nodes + 0.2, rtol=0,
                               atol=1e-15)
    _assert_read_only(shifted)


def _joblib_round_trip(obj):
    joblib = pytest.importorskip("joblib")
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    buf.seek(0)
    return joblib.load(buf)


@pytest.mark.parametrize("round_trip", [
    lambda d: pickle.loads(pickle.dumps(d)), _joblib_round_trip,
    copy.deepcopy, copy.copy,
], ids=["pickle", "joblib", "deepcopy", "copy"])
def test_UH2_round_trips_keep_value_hash_and_read_only_flag(round_trip):
    dens = SpectralDensity.from_callable(_rho_clean, LO, HI, N_NODES,
                                         shift=SHIFT)
    back = round_trip(dens)
    assert back == dens and hash(back) == hash(dens)
    _assert_read_only(back)
    _assert_read_only(dens)
    cache = round_trip(spectral_cache(dens, D_NOISE, n_components=2,
                                      t_min=T_MIN))
    _assert_read_only(cache.density)
    np.testing.assert_allclose(float(cache.C_diagonal(0, T1, 0, T2)[1]),
                               _C_reference(T1, T2), rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# UH3-UH4: the self-consistency driver's distance and callback
# ---------------------------------------------------------------------------

A_MAP = np.array([[0.31, -0.22], [0.17, -0.41]])
B_MAP = np.array([0.73, -1.19])
X0 = np.array([0.25, -0.6])
X_STAR = np.linalg.solve(np.eye(2) - A_MAP, B_MAP)
TOL = 1e-12


def _step(x):
    return A_MAP @ x + B_MAP


def _d_clean(a, b):
    return float(np.max(np.abs(a - b)))


def _d_write_first(a, b):
    a -= b
    return float(np.max(np.abs(a)))


def _d_write_second(a, b):
    b -= a
    return float(np.max(np.abs(b)))


def _d_sort_clean(a, b):
    return float(np.max(np.abs(np.sort(a) - np.sort(b))))


def _d_sort_in_place(a, b):
    a.sort()
    b.sort()
    return float(np.max(np.abs(a - b)))


def _run(initial, distance=None, damping=0.0, callback=None, step=_step):
    return solve_self_consistency(initial, step, distance=distance, tol=TOL,
                                  max_iter=200, damping=damping,
                                  callback=callback)


def _assert_same_run(got, want):
    assert (got.reason, got.n_iter, got.residuals) == (
        want.reason, want.n_iter, want.residuals)
    if isinstance(want.state, dict):
        for k in want.state:
            np.testing.assert_array_equal(got.state[k], want.state[k])
    else:
        np.testing.assert_array_equal(got.state, want.state)


@pytest.mark.parametrize("damping", [0.0, 0.35])
@pytest.mark.parametrize("writer, clean", [
    (_d_write_first, _d_clean), (_d_write_second, _d_clean),
    (_d_sort_in_place, _d_sort_clean),
], ids=["a-=b", "b-=a", "sort"])
def test_UH3_a_writing_distance_gives_the_run_of_the_clean_one(
        writer, clean, damping):
    want = _run(X0.copy(), clean, damping)
    assert want.converged, want.summary()
    np.testing.assert_allclose(want.state, X_STAR, rtol=0, atol=1e-11)
    initial = X0.copy()
    got = _run(initial, writer, damping)
    _assert_same_run(got, want)
    np.testing.assert_array_equal(initial, X0)


def test_UH3_a_dict_state_and_a_leaf_writing_distance():
    target = {"R": np.array([0.8, -0.35, 1.1]), "C": np.array([-0.6, 0.45])}

    def step(s):
        return {"R": 0.55 * s["R"][::-1] + 0.3 * target["R"],
                "C": -0.4 * s["C"] + target["C"]}

    def clean(a, b):
        return max(float(np.max(np.abs(a[k] - b[k]))) for k in a)

    def writer(a, b):
        for k in a:
            a[k] -= b[k]
        return max(float(np.max(np.abs(a[k]))) for k in a)

    init = {"R": np.array([0.2, -0.9, 0.4]), "C": np.array([1.3, -0.2])}
    for damping in (0.0, 0.35):
        want = _run(copy.deepcopy(init), clean, damping, step=step)
        assert want.converged, want.summary()
        caller = copy.deepcopy(init)
        _assert_same_run(_run(caller, writer, damping, step=step), want)
        for k in init:
            np.testing.assert_array_equal(caller[k], init[k])


def _cb_round(it, s, res):
    np.round(s, 3, out=s)


def _cb_scale(it, s, res):
    s *= 1.01


@pytest.mark.parametrize("damping", [0.0, 0.35])
@pytest.mark.parametrize("callback", [_cb_round, _cb_scale],
                         ids=["round", "scale"])
def test_UH4_a_writing_callback_does_not_change_the_run(callback, damping):
    want = _run(X0.copy(), damping=damping)
    assert want.converged, want.summary()
    np.testing.assert_allclose(want.state, X_STAR, rtol=0, atol=1e-11)
    _assert_same_run(_run(X0.copy(), damping=damping, callback=callback), want)


def test_UH4_dict_state_and_a_recording_callback():
    def step(s):
        return {"R": A_MAP @ s["R"] + B_MAP, "C": -0.5 * s["C"] + 0.3}

    def round_leaves(it, s, res):
        for v in s.values():
            np.round(v, 3, out=v)

    init = {"R": X0.copy(), "C": np.array([0.9, -0.4, 0.15])}
    want = _run(copy.deepcopy(init), step=step)
    assert want.converged, want.summary()
    _assert_same_run(_run(copy.deepcopy(init), step=step,
                          callback=round_leaves), want)

    # Recording still sees every post-mixing state.
    seen = []
    got = _run(X0.copy(), damping=0.35,
               callback=lambda it, s, res: seen.append(s))
    assert len(seen) == got.n_iter
    np.testing.assert_array_equal(seen[-1], got.state)


# ---------------------------------------------------------------------------
# UH6: an initial state that is the step's own output buffer
# ---------------------------------------------------------------------------

class _BufferStep:
    """Fills and returns an output buffer it owns, as SC7's step does."""

    def __init__(self):
        self.out = X0.copy()

    def __call__(self, x):
        self.out[:] = A_MAP @ x + B_MAP
        return self.out


class _SolverObject:
    """Updates its own ``G`` in place and returns it, the shape of a DMFT
    solver started from its current Green's function."""

    def __init__(self):
        self.G = X0.copy()

    def __call__(self, x):
        np.matmul(A_MAP, x, out=self.G)
        self.G += B_MAP
        return self.G


@pytest.mark.parametrize("damping", [0.0, 0.35])
@pytest.mark.parametrize("make", [_BufferStep, _SolverObject],
                         ids=["buffer", "solver"])
def test_UH6_initial_that_is_the_steps_output_buffer(make, damping):
    """Passed as is, ``initial`` was the iteration-1 state: the step wrote
    its buffer, which was that state, so the residual compared the buffer
    with itself.  Both steps reported converged after one iteration with a
    residual of 0.0, at a state 0.338 from the fixed point, at either
    damping."""
    want = _run(X0.copy(), damping=damping, step=make())
    assert want.converged and want.n_iter > 2, want.summary()
    np.testing.assert_allclose(want.state, X_STAR, rtol=0, atol=1e-11)
    step = make()
    buf = step.out if isinstance(step, _BufferStep) else step.G
    got = _run(buf, damping=damping, step=step)
    _assert_same_run(got, want)
    assert got.state is not buf


def test_UH6_dict_state_whose_leaves_are_the_steps_buffers():
    leaves = {"R": X0.copy(), "C": np.array([1.3, -0.2])}
    target_c = np.array([-0.6, 0.45])

    def step(s):
        np.matmul(A_MAP, s["R"], out=leaves["R"])
        leaves["R"] += B_MAP
        leaves["C"][:] = -0.4 * s["C"] + target_c
        return leaves

    init = copy.deepcopy(leaves)
    want = _run(copy.deepcopy(init), step=step)
    assert want.converged and want.n_iter > 2, want.summary()
    np.testing.assert_allclose(want.state["R"], X_STAR, rtol=0, atol=1e-11)
    np.testing.assert_allclose(want.state["C"], target_c / 1.4, rtol=0,
                               atol=1e-11)
    for k in init:
        leaves[k][:] = init[k]
    _assert_same_run(_run(leaves, step=step), want)


# ---------------------------------------------------------------------------
# UH5: the brute-force R-contracted reference
# ---------------------------------------------------------------------------

K_A = np.array([[[0.62, -0.41], [0.18, -0.77]],
                [[-0.29, 0.53], [0.94, -0.12]]])
P_LEG = np.array([1.0, 0.6, 1.7])
Q_LEG = np.array([0.4, 0.9, 0.2])
X_CENTRE = 0.15
G_R = 1.3
CHI = np.linspace(0.2, 1.7, 9)
N_PTS = np.array([0.1, -0.35, 0.6])
T_PTS = np.array([1.7, 1.25, 0.9])


def _R_kernel(t, s):
    return exp(-G_R * (t - s))


def _raw_value(x, t):
    return K_A * np.exp(-np.sum(P_LEG * x ** 2)) * np.exp(-(Q_LEG @ t))


def _raw_clean(n_list, t_list):
    x = np.asarray(n_list, dtype=float) - X_CENTRE
    return _raw_value(x, np.asarray(t_list, dtype=float))


def _raw_write(n_list, t_list):
    x = np.asarray(n_list, dtype=float)
    x -= X_CENTRE
    return _raw_value(x, np.asarray(t_list, dtype=float))


def _contracted_reference(n, t):
    """The trapezoid sum of the wrapper, factorised over legs: the raw
    kernel is a product of one factor per leg, and the causal R is zero on
    the grid nodes past each outer time."""
    x = np.asarray(n, dtype=float) - X_CENTRE
    factor = 1.0
    for i in range(3):
        f = np.where(CHI <= t[i], np.exp(-G_R * (t[i] - CHI) - Q_LEG[i] * CHI),
                     0.0)
        factor *= np.trapezoid(f, CHI)
    return K_A * np.exp(-np.sum(P_LEG * x ** 2)) * factor


@pytest.mark.parametrize("as_list", [False, True], ids=["ndarray", "list"])
def test_UH5_a_raw_kernel_writing_its_positions_gives_the_reference(as_list):
    want = _contracted_reference(N_PTS, T_PTS)
    clean = build_R_contracted_callable(_raw_clean, _R_kernel, CHI, order=3,
                                        n_components=2)
    writer = build_R_contracted_callable(_raw_write, _R_kernel, CHI, order=3,
                                         n_components=2)
    np.testing.assert_allclose(clean(N_PTS.copy(), T_PTS), want, rtol=1e-12,
                               atol=0)
    n_in = list(N_PTS) if as_list else N_PTS.copy()
    got = writer(n_in, T_PTS)
    # The writer gave K[0,0,0] = 4.52e-5 against 1.58e-2.
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(np.asarray(n_in), N_PTS)


def test_UH5_the_wrapper_keeps_its_own_chi_grid():
    """The trapezoid weights are fixed when the wrapper is built.  Holding
    the caller's grid by reference, a later ``chi += 0.25`` by the caller
    moved the nodes and not the weights: K[0,0,0] went from 1.58e-2 to
    1.00e-2."""
    want = _contracted_reference(N_PTS, T_PTS)
    chi_user = CHI.copy()
    fn = build_R_contracted_callable(_raw_clean, _R_kernel, chi_user, order=3,
                                     n_components=2)
    np.testing.assert_allclose(fn(N_PTS.copy(), T_PTS), want, rtol=1e-12,
                               atol=0)
    chi_user += 0.25
    np.testing.assert_allclose(fn(N_PTS.copy(), T_PTS), want, rtol=1e-12,
                               atol=0)


@pytest.mark.parametrize("method, kw", [
    ("gauss_legendre", dict(n_gauss=3)),
    ("qmc_vectorized", dict(n_samples=64, seed=5)),
], ids=["gl", "qmc"])
def test_UH5_L1_three_point_function_with_a_writing_raw_kernel(method, kw):
    """Order 1, one absorbed κ³ vertex: every leg time is an external time,
    so the value is the R-contracted callable at the external points."""
    totals = {}
    for name, raw in (("clean", _raw_clean), ("write", _raw_write)):
        fn = build_R_contracted_callable(raw, _R_kernel, CHI, order=3,
                                         n_components=2)
        system = sw.System(
            field=sw.FieldSpec("phi", n_components=2),
            linear=sw.DiagonalA(gamma=[G_R, G_R]),
            nonlocal_vertices=[sw.NonLocalVertex(
                name="K", order=3, coupling=fn, already_R_contracted=True)],
            noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.1, sigma_t=0.3),
                spatial=sw.ExponentialSpatial(sigma_x=1.0))),
            t_min=0.2,
        )
        exp_ = system.expand(("phi_a(x)", "phi_b(y)", "phi_c(z)"), orders=[1],
                             progress=False)
        props = system.propagators(t_max=2.0, n_grid_t=8, progress=False)
        totals[name] = [
            exp_.evaluate(props, positions=dict(zip("xyz", N_PTS)),
                          t_final=T_PTS[0],
                          external_times=dict(zip("xyz", T_PTS)),
                          component_pair=pair, orders=[1],
                          vertex_types=["K"], method=method, **kw).total
            for pair in ((0, 1, 1), (1, 0, 1), (1, 1, 0))
        ]
    assert all(v != 0.0 for v in totals["clean"])
    assert totals["write"] == totals["clean"]
