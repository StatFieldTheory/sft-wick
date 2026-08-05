"""The self-consistency driver.

Every fixed point here is known independently -- a contraction with an
analytic solution, a deliberate two-cycle, a divergent map -- so the LOOP is
what is under test, not a physics model whose answer we would be guessing.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

from sft_wick.selfconsistency import (
    SelfConsistencyResult, _copy_state, _mix, max_abs_distance,
    solve_self_consistency,
)


# --------------------------------------------------------------------- #
# SC1 — it finds a fixed point, and reports it as such
# --------------------------------------------------------------------- #

def test_SC1_scalar_contraction_reaches_its_analytic_fixed_point():
    """x -> (x + a/x)/2 is Newton for sqrt(a): the fixed point is sqrt(a)."""
    a = 7.0
    res = solve_self_consistency(1.0, lambda x: 0.5 * (x + a / x), tol=1e-14)
    assert res.converged and res.reason == "converged"
    assert float(res.state) == pytest.approx(np.sqrt(a), rel=1e-12)
    assert bool(res) is True
    # the residual history is monotone here and ends below tol
    assert res.residuals[-1] <= 1e-14
    assert len(res.residuals) == res.n_iter


def test_SC1_linear_map_matches_the_closed_form_fixed_point():
    """x -> A x + b has the fixed point (I - A)^-1 b for spectral radius < 1."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4)) * 0.1
    b = rng.normal(size=4)
    want = np.linalg.solve(np.eye(4) - A, b)
    res = solve_self_consistency(np.zeros(4), lambda x: A @ x + b, tol=1e-13,
                                 max_iter=500)
    assert res.converged, res.summary()
    assert res.state == pytest.approx(want, rel=1e-10)


def test_SC1_structured_states_work():
    """A DMFT state is a pair or a dict of two-time arrays, not a scalar."""
    target = {"R": np.linspace(0.0, 1.0, 6).reshape(2, 3), "C": np.full(4, 2.0)}

    def step(s):
        return {k: 0.5 * (s[k] + target[k]) for k in s}

    init = {"R": np.zeros((2, 3)), "C": np.zeros(4)}
    res = solve_self_consistency(init, step, tol=1e-12, max_iter=200)
    assert res.converged
    for k in target:
        assert res.state[k] == pytest.approx(target[k], abs=1e-11)

    # tuples too
    res2 = solve_self_consistency(
        (np.zeros(3), 0.0),
        lambda s: (0.5 * (s[0] + np.ones(3)), 0.5 * (s[1] + 4.0)),
        tol=1e-12, max_iter=200,
    )
    assert res2.converged
    assert res2.state[0] == pytest.approx(np.ones(3), abs=1e-11)
    assert float(res2.state[1]) == pytest.approx(4.0, abs=1e-11)


# --------------------------------------------------------------------- #
# SC2 — the failure modes are told apart.  This is the point of the module:
# a non-converged iteration looks exactly like a converged one if you only
# print the last state.
# --------------------------------------------------------------------- #

def test_SC2_divergence_is_reported_not_returned_as_an_answer():
    res = solve_self_consistency(1.0, lambda x: 3.0 * x + 1.0, tol=1e-10,
                                 max_iter=200)
    assert not res.converged
    assert res.reason == "diverged", res.summary()
    assert res.n_iter < 200, "should stop early, not burn every iteration"
    assert bool(res) is False


def test_SC2_a_two_cycle_is_called_oscillating_and_damping_fixes_it():
    """x -> -x + 4 has the fixed point 2 but a plain iteration two-cycles.

    That is exactly the case damping exists for, and the reason string is
    what tells the caller to reach for it.
    """
    step = lambda x: -x + 4.0                       # noqa: E731
    plain = solve_self_consistency(0.0, step, tol=1e-10, max_iter=60)
    assert not plain.converged
    assert plain.reason == "oscillating", plain.summary()

    damped = solve_self_consistency(0.0, step, tol=1e-12, max_iter=200,
                                    damping=0.5)
    assert damped.converged, damped.summary()
    assert float(damped.state) == pytest.approx(2.0, abs=1e-10)


def test_SC2_damping_cannot_fake_convergence():
    """The residual must measure ||step(x) - x||, not the damped movement.

    The state moves by only `(1 - damping)` times the step, so a residual read
    off the movement shrinks as damping rises -- crank damping to 0.99 and any
    iteration "converges" a hundred times sooner, at a point that is not a
    fixed point at all.  That is the classic way a DMFT loop reports a
    solution it never found, so it gets its own test.
    """
    # a map with NO fixed point reachable here: x -> x + 1 moves by 1 forever
    for damping in (0.0, 0.9, 0.99):
        res = solve_self_consistency(0.0, lambda x: x + 1.0, tol=1e-2,
                                     max_iter=20, damping=damping)
        assert not res.converged, f"damping={damping} faked convergence"
        # the residual is the step, so it is 1.0 regardless of damping
        assert res.residual == pytest.approx(1.0), damping

    # and on a genuinely converging map the residual is damping-independent
    # at the first iteration, where every run starts from the same state
    firsts = [
        solve_self_consistency(0.0, lambda x: 0.5 * (x + 1.0), tol=1e-12,
                               max_iter=1, damping=d).residuals[0]
        for d in (0.0, 0.5, 0.9)
    ]
    assert firsts == pytest.approx([firsts[0]] * 3)


def test_SC2_max_iter_while_still_improving_says_so():
    res = solve_self_consistency(0.0, lambda x: 0.5 * (x + 1.0), tol=1e-30,
                                 max_iter=5)
    assert not res.converged
    assert res.reason == "max_iter", res.summary()
    assert res.n_iter == 5
    assert len(res.residuals) == 5
    # still descending when it ran out -- more iterations WOULD have helped
    assert res.residuals[-1] < res.residuals[0]


def test_SC2_the_result_cannot_be_mistaken_for_a_bare_state():
    """`bool(result)` is `converged`, so ignoring the outcome is deliberate."""
    bad = solve_self_consistency(1.0, lambda x: 3.0 * x, max_iter=50)
    assert not bad
    assert isinstance(bad, SelfConsistencyResult)
    assert "diverged" in bad.summary()


# --------------------------------------------------------------------- #
# SC3 — mixing, callbacks, validation
# --------------------------------------------------------------------- #

def test_SC3_damping_mixes_after_the_step_not_before():
    """`step` must always see a genuine state, never a pre-mixed one.

    Mixing before would feed `step` a point that is not on the trajectory --
    for a physics step that means evaluating diagrams at propagators that no
    self-energy produced.
    """
    seen = []

    def step(x):
        seen.append(float(x))
        return 0.0            # constant map; fixed point is 0

    res = solve_self_consistency(1.0, step, damping=0.75, tol=1e-6,
                                 max_iter=4)
    # states are 1, 0.75, 0.5625, ... -- geometric in the damping factor
    assert seen[0] == pytest.approx(1.0)
    assert seen[1] == pytest.approx(0.75)
    assert seen[2] == pytest.approx(0.5625)
    assert not res.converged  # 4 iterations is not enough at damping 0.75


def test_SC3_callback_sees_every_iteration():
    log = []
    solve_self_consistency(
        0.0, lambda x: 0.5 * (x + 1.0), tol=1e-12, max_iter=10,
        callback=lambda i, s, r: log.append((i, float(s), r)),
    )
    assert [k for k, _, _ in log] == list(range(1, len(log) + 1))
    assert all(r > 0 for _, _, r in log)


def test_SC3_inputs_are_validated():
    f = lambda x: x  # noqa: E731
    with pytest.raises(ValueError, match="damping"):
        solve_self_consistency(0.0, f, damping=1.0)
    with pytest.raises(ValueError, match="damping"):
        solve_self_consistency(0.0, f, damping=-0.1)
    with pytest.raises(ValueError, match="max_iter"):
        solve_self_consistency(0.0, f, max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        solve_self_consistency(0.0, f, tol=-1.0)


def test_SC3_distance_rejects_mismatched_structures():
    """Comparing whatever happens to line up is how a wrong fixed point gets
    reported as converged."""
    with pytest.raises(ValueError, match="different keys"):
        max_abs_distance({"R": 1.0}, {"C": 1.0})
    with pytest.raises(ValueError, match="different lengths"):
        max_abs_distance((1.0, 2.0), (1.0,))
    with pytest.raises(ValueError, match="different shapes"):
        max_abs_distance(np.zeros((2, 2)), np.zeros(4))
    with pytest.raises(TypeError):
        max_abs_distance({"R": 1.0}, (1.0,))
    # nested structures do work
    assert max_abs_distance({"a": [np.zeros(2), 1.0]},
                            {"a": [np.array([0.0, 0.5]), 1.25]}) == 0.5


def test_SC3_a_custom_distance_is_used():
    calls = []

    def dist(a, b):
        calls.append(1)
        return abs(float(a) - float(b))

    res = solve_self_consistency(0.0, lambda x: 0.5 * (x + 1.0), tol=1e-12,
                                 max_iter=100, distance=dist)
    assert res.converged and calls


# --------------------------------------------------------------------- #
# SC4 — a physically-shaped fixed point, solved by the driver
# --------------------------------------------------------------------- #

def test_SC4_a_two_time_self_consistency_with_a_known_solution():
    """Shaped like a DMFT step: the state is a two-time array, the update is
    linear in it, and the fixed point is available in closed form.

    C_{n+1}(t,t') = K(t,t') + g * (M @ C_n @ M^T)(t,t') is linear, so the fixed
    point is the solution of a Sylvester-type equation -- here obtained by
    solving the flattened linear system, entirely independently of the loop.
    """
    n, g = 5, 0.15
    rng = np.random.default_rng(3)
    K = rng.normal(size=(n, n))
    K = 0.5 * (K + K.T)                      # C is symmetric, as a correlator
    M = np.tril(rng.normal(size=(n, n)) * 0.3)   # retarded: lower-triangular

    def step(C):
        return K + g * (M @ C @ M.T)

    # independent reference: vec(C) = vec(K) + g (M kron M) vec(C)
    A = np.eye(n * n) - g * np.kron(M, M)
    want = np.linalg.solve(A, K.reshape(-1)).reshape(n, n)

    res = solve_self_consistency(np.zeros((n, n)), step, tol=1e-13,
                                 max_iter=500)
    assert res.converged, res.summary()
    assert res.state == pytest.approx(want, abs=1e-11)
    # and it is genuinely a fixed point of the step, not just a converged loop
    assert step(res.state) == pytest.approx(res.state, abs=1e-11)


def test_SC4_a_supercritical_coupling_is_reported_not_silently_wrong():
    """The same map past its radius of convergence.

    A self-consistency that does not have a stable fixed point must not come
    back looking like one -- this is the physics failure the reason string
    exists for.
    """
    n, g = 5, 40.0
    rng = np.random.default_rng(3)
    K = rng.normal(size=(n, n))
    M = np.tril(rng.normal(size=(n, n)) * 0.3)
    res = solve_self_consistency(np.zeros((n, n)),
                                 lambda C: K + g * (M @ C @ M.T),
                                 tol=1e-10, max_iter=300)
    assert not res.converged
    assert res.reason in ("diverged", "max_iter"), res.summary()


# --------------------------------------------------------------------- #
# SC5 — the classifier's discriminating cases.  Each of these was a mutant
# that survived: the suite could not tell the right rule from a wrong one.
# --------------------------------------------------------------------- #

def test_SC5_a_transient_bump_is_not_condemned_as_divergence():
    """A residual that rises for several iterations and then turns over.

    Monotone growth alone is not divergence -- a physical loop often gets
    worse before it gets better, and killing it at four consecutive increases
    would throw away a run that was about to converge.
    """
    seq = iter([1.0, 1.3, 1.7, 2.2, 2.8, 1.0, 0.2, 0.01, 1e-5, 1e-9, 0.0])

    def step(x):
        return float(x) + next(seq)     # residual follows `seq` exactly

    res = solve_self_consistency(0.0, step, tol=1e-8, max_iter=11)
    assert res.converged, res.summary()
    # it really did rise for five iterations first
    assert res.residuals[:5] == pytest.approx([1.0, 1.3, 1.7, 2.2, 2.8])


def test_SC5_heavy_damping_is_not_mistaken_for_a_cycle():
    """The cycle test must measure against the state's MOVEMENT.

    Under damping d the state moves by only (1 - d) of the step, so a cycle
    test measured against the residual sees every heavily-damped iteration as
    "barely moving" and calls a perfectly healthy run oscillating.
    """
    res = solve_self_consistency(0.0, lambda x: 0.5 * (x + 1.0), tol=1e-10,
                                 max_iter=6000, damping=0.99)
    assert res.reason != "oscillating", res.summary()
    assert res.converged, res.summary()
    assert float(res.state) == pytest.approx(1.0, abs=1e-9)


def test_SC5_step_sees_the_mixed_state_not_the_raw_proposal():
    """Which point `step` is evaluated at, exactly.

    The trajectory is x_{n+1} = (1-d) F(x_n) + d x_n, and `step` is evaluated
    at x_n -- a point on that trajectory.  Feeding it the raw F(x_n) instead
    would evaluate diagrams at propagators no self-energy ever produced.
    """
    seen = []

    def step(x):
        seen.append(float(x))
        return 2.0 * float(x) + 1.0            # fixed point at -1

    solve_self_consistency(1.0, step, damping=0.5, tol=1e-12, max_iter=4)
    # x0=1; F=3, x1=0.5*3+0.5*1=2; F=5, x2=0.5*5+0.5*2=3.5; F=8, x3=5.75
    assert seen == pytest.approx([1.0, 2.0, 3.5, 5.75])


def test_SC5_an_obvious_blow_up_stops_immediately_not_eventually():
    """`divergence_factor` alone would keep iterating a 10x-per-step blow-up
    for another handful of rounds.  For a physics state that is not free:
    each round the arrays grow by the same factor, and "stop before it
    overflows" is what the diverged verdict is for.
    """
    res = solve_self_consistency(1.0, lambda x: 3.0 * x, tol=1e-10,
                                 max_iter=100)
    assert res.reason == "diverged", res.summary()
    # 3x per step needs eight iterations to clear divergence_factor=1e3 from
    # the first residual; the window test sees it at four.
    assert res.n_iter <= 4, f"took {res.n_iter} iterations to admit it"


# --------------------------------------------------------------------- #
# SC6 — defects found by adversarial review.  Each of these previously
# reported a state that was NOT a fixed point, or condemned a run that was
# converging.  They are the reason the module exists, so they are pinned.
# --------------------------------------------------------------------- #

def test_SC6_a_step_that_mutates_in_place_cannot_fake_convergence():
    """`x += dx; return x` is the standard numpy idiom -- and it returns the
    object it was given.  The residual `dist(proposed, state)` is then
    measured between an object and itself: identically zero, converged on
    iteration 1, at a state that is not remotely a fixed point.
    """
    def mutating(x):
        x += 1.0                 # in place; returns its own argument
        return x

    res = solve_self_consistency(np.zeros(3), mutating, tol=1e-8, max_iter=10)
    assert not res.converged, res.summary()
    assert res.residual == pytest.approx(1.0)

    # dicts of arrays too -- the shape a DMFT state actually has
    def mutating_dict(s):
        for k in s:
            s[k] *= 2.0
        return s

    res = solve_self_consistency({"R": np.ones(2), "C": np.ones(2)},
                                 mutating_dict, tol=1e-8, max_iter=5)
    assert not res.converged, res.summary()

    # and the copy handed to `step` must not corrupt the caller's own object
    initial = np.zeros(3)
    solve_self_consistency(initial, mutating, tol=1e-8, max_iter=3)
    assert initial == pytest.approx(np.zeros(3)), "caller's state was mutated"


def test_SC6_complex_states_are_compared_by_modulus_not_real_part():
    """`np.asarray(x, dtype=float)` does not raise on a complex array -- it
    drops the imaginary part, warning once per source line and then falling
    silent.  sft-wick's own diagram values carry ``i^(-E_psi)`` phases, so a
    DMFT state assembled from them is routinely complex, and a distance taken
    on real parts only reports convergence while Im keeps moving.
    """
    assert max_abs_distance(np.array([1 + 2j]), np.array([1 + 0j])) == 2.0

    # a map that is a fixed point of Re but not of Im
    res = solve_self_consistency(np.array([1.0 + 1.0j]),
                                 lambda z: z.real + 2j * z.imag,
                                 tol=1e-8, max_iter=5)
    assert not res.converged, res.summary()

    # mixing keeps the state complex rather than flattening it
    mixed = _mix(np.array([1 + 1j]), np.array([0 + 0j]), 0.5)
    assert mixed == pytest.approx(np.array([0.5 + 0.5j]))

    # a genuinely complex fixed point is still found
    res = solve_self_consistency(
        0j, lambda z: 0.5 * (z + (3.0 - 4.0j)), tol=1e-12, max_iter=200)
    assert res.converged and res.state == pytest.approx(3.0 - 4.0j)


@pytest.mark.parametrize("a", [-0.5, -0.909, -0.91, -0.95, -0.99])
def test_SC6_an_alternating_contraction_converges_and_is_not_called_a_cycle(a):
    """`x -> a x + b` with -1 < a < 0 alternates about its fixed point while
    contracting toward it.  Every step it returns to within |1 + a| of where
    it was two steps ago -- so a cycle test that looks only at the state's
    return distance condemns the entire slowly-alternating regime, which is
    precisely the regime near a DMFT transition.  The residual is what
    distinguishes them: a contraction's falls, a cycle's does not.
    """
    res = solve_self_consistency(0.0, lambda x: a * x + 1.0, tol=1e-10,
                                 max_iter=20000)
    assert res.reason == "converged", res.summary()
    assert float(res.state) == pytest.approx(1.0 / (1.0 - a), abs=1e-8)


def test_SC6_a_near_involution_is_called_oscillating_and_damping_fixes_it():
    """Past |a| ~ 0.995 the improvement per two steps drops below 1% and the
    map is reported oscillating.  That is the honest verdict rather than an
    arbitrary cut: plain iteration needs thousands of steps there, and the
    prescribed remedy genuinely works -- which is what makes the reason
    string actionable rather than merely a label.
    """
    plain = solve_self_consistency(0.0, lambda x: -0.999 * x + 1.0,
                                   tol=1e-10, max_iter=500)
    assert plain.reason == "oscillating", plain.summary()

    damped = solve_self_consistency(0.0, lambda x: -0.999 * x + 1.0,
                                    tol=1e-10, max_iter=500, damping=0.5)
    assert damped.converged, damped.summary()
    assert float(damped.state) == pytest.approx(1.0 / 1.999, abs=1e-9)


def test_SC6_a_run_that_drifts_off_a_repelling_start_is_not_called_diverged():
    """`x -> x^2` from just inside 1 contracts to 0, but the residual x(1-x)
    climbs from 1e-6 to 0.25 before collapsing.  Measuring growth against the
    best residual EVER seen kills it on the way up -- and "start from the
    non-interacting solution, drift off it, converge to the interacting one"
    is exactly that trajectory.
    """
    res = solve_self_consistency(1.0 - 1e-6, lambda x: x * x, tol=1e-12,
                                 max_iter=400)
    assert res.converged, res.summary()
    assert float(res.state) == pytest.approx(0.0, abs=1e-20)
    # it really did climb by five orders of magnitude first
    assert max(res.residuals) > 1e4 * res.residuals[0]


@pytest.mark.parametrize("period, step_map", [
    (3, {0.0: 1.0, 1.0: 2.0, 2.0: 0.0}),
    (4, {0.0: 1.0, 1.0: 5.0, 5.0: 2.0, 2.0: 0.0}),
    (5, {0.0: 1.0, 1.0: 5.0, 5.0: 2.0, 2.0: 7.0, 7.0: 0.0}),
])
def test_SC6_longer_cycles_are_called_oscillating_not_max_iter(period, step_map):
    """A cycle test that only looks one step back sees period 2 and nothing
    else.  Reporting a 3-cycle as "max_iter" tells the caller to raise
    max_iter -- which never terminates.  Damping is the remedy, and
    "oscillating" is the string that says so.
    """
    res = solve_self_consistency(0.0, lambda x: step_map[float(x)],
                                 tol=1e-10, max_iter=80)
    assert res.reason == "oscillating", f"period {period}: {res.summary()}"
    assert res.n_iter < 80, "should not have burned every iteration"


def test_SC6_namedtuple_states_survive_damping():
    """`type(t)(list)` works for a plain tuple and raises for a namedtuple --
    and only under damping, which is the module's own prescribed cure for an
    oscillating result.  Failing exactly when the user takes your advice is a
    bad way to fail.
    """
    NT = collections.namedtuple("NT", "R C")
    res = solve_self_consistency(
        NT(np.zeros(2), 0.0),
        lambda s: NT(0.5 * (s.R + 1.0), 0.5 * (s.C + 4.0)),
        damping=0.3, tol=1e-12, max_iter=300,
    )
    assert res.converged, res.summary()
    assert isinstance(res.state, NT)
    assert res.state.R == pytest.approx(np.ones(2), abs=1e-10)


def test_SC6_an_empty_state_is_rejected_not_declared_converged():
    """Every distance over an empty state is 0, so the loop would report
    convergence on iteration 1.  An empty state is an upstream bug -- a
    filtered-away component, an empty time grid -- and saying so beats
    returning a "solution"."""
    with pytest.raises(ValueError, match="no elements"):
        solve_self_consistency({"R": np.zeros((0, 4)), "C": np.zeros(0)},
                               lambda s: {k: v + 1.0 for k, v in s.items()})
    with pytest.raises(ValueError, match="no elements"):
        solve_self_consistency([], lambda s: s)


def test_SC6_non_numeric_states_are_rejected_with_a_clear_error():
    with pytest.raises(TypeError, match="numeric"):
        max_abs_distance(np.array(["a", "b"]), np.array(["a", "c"]))


# --------------------------------------------------------------------- #
# SC7 — a second adversarial review round.  The two CRITICALs here are the
# same root cause as SC6's: `step` was protected on the INPUT side only.
# --------------------------------------------------------------------- #

def test_SC7_a_step_returning_a_reused_output_buffer_cannot_fake_convergence():
    """The preallocated-output idiom: `step` fills and returns a buffer it
    owns.  `_mix` short-circuits to `proposed` at the DEFAULT damping=0, so
    `state` would alias that buffer; the next call then overwrites `state` in
    place BEFORE the residual is taken, and `dist(proposed, state)` compares
    the buffer with itself -- exactly 0.0, "converged", at the second iterate.

    Copying the input to `step` does not help: the aliasing is on the way out.
    Note damping > 0 hides this (mixing allocates), so it is invisible in
    precisely the default configuration.
    """
    buf = np.zeros(3)
    target = np.array([1.0, 2.0, 3.0])

    def step(x):
        buf[:] = 0.5 * (x + target)
        return buf

    res = solve_self_consistency(np.zeros(3), step, tol=1e-8, max_iter=200)
    assert res.converged, res.summary()
    assert res.state == pytest.approx(target, abs=1e-7)
    assert res.state is not buf, "the result aliases the caller's buffer"
    assert res.n_iter > 2, "converged suspiciously early -- residual faked?"

    # a slow contraction is the dangerous case: it used to report converged
    # at iteration 2, ~50x away from the answer
    b2 = np.zeros(4)
    tgt = np.array([10.0, -5.0, 3.0, 7.0])

    def slow(x):
        b2[:] = 0.99 * x + 0.01 * tgt
        return b2

    res = solve_self_consistency(np.zeros(4), slow, tol=1e-10, max_iter=20000)
    assert res.converged, res.summary()
    assert res.state == pytest.approx(tgt, abs=1e-6)

    # a class holding a workspace is the same shape
    class Solver:
        def __init__(self, n):
            self.out = np.zeros(n)

        def __call__(self, x):
            np.multiply(x + target, 0.5, out=self.out)
            return self.out

    res = solve_self_consistency(np.zeros(3), Solver(3), tol=1e-8,
                                 max_iter=200)
    assert res.state == pytest.approx(target, abs=1e-7)


def test_SC7_a_state_type_outside_the_known_containers_is_still_copied():
    """`_copy_state` used to fall through to `return x` for anything that was
    not a dict/list/tuple/ndarray, silently withdrawing the guarantee for
    exactly the states the module does not enumerate -- and the caller has no
    signal that the protection lapsed.
    """
    pd = pytest.importorskip("pandas")

    s0 = pd.Series([0.0, 0.0, 0.0])
    assert _copy_state(s0) is not s0

    def mutating(x):
        x += 1.0            # in place; returns its own argument
        return x

    res = solve_self_consistency(pd.Series([0.0, 0.0, 0.0]), mutating,
                                 tol=1e-8, max_iter=10)
    assert not res.converged, res.summary()

    # the caller's own object is untouched
    original = pd.Series([0.0, 0.0, 0.0])
    solve_self_consistency(original, mutating, tol=1e-8, max_iter=3)
    assert float(original.iloc[0]) == 0.0


def test_SC7_container_subclasses_survive_every_iteration_and_any_damping():
    """`step` must receive the type it was given on iteration 2 as on
    iteration 1, and `result.state`'s type must not depend on `damping`."""
    class MyDict(dict):
        pass

    seen = []

    def step(x):
        seen.append(type(x).__name__)
        return MyDict({k: 0.5 * (v + 1.0) for k, v in x.items()})

    for damping in (0.0, 0.3):
        seen.clear()
        res = solve_self_consistency(MyDict({"R": np.zeros(2)}), step,
                                     tol=1e-12, max_iter=200, damping=damping)
        assert res.converged, res.summary()
        assert set(seen) == {"MyDict"}, f"damping={damping}: step saw {set(seen)}"
        assert isinstance(res.state, MyDict), f"damping={damping}"


def test_SC7_integer_states_are_not_measured_through_a_wraparound():
    """`aa - bb` in the input's own integer dtype wraps modulo 2**nbits, so a
    state far from the fixed point can report a distance small enough to
    satisfy `tol` -- convergence declared by overflow."""
    assert max_abs_distance(np.array([0], dtype=np.int8),
                            np.array([-128], dtype=np.int8)) == 128.0
    assert max_abs_distance(np.array([0], dtype=np.uint8),
                            np.array([200], dtype=np.uint8)) == 200.0
    assert max_abs_distance(np.array([2 ** 62], dtype=np.int64),
                            np.array([-2 ** 62], dtype=np.int64)) \
        == pytest.approx(2.0 ** 63)


def test_SC7_a_nan_or_inf_residual_is_reported_as_diverged():
    """The non-finite guard is the first thing checked, before `res <= tol`.
    A NaN state must not be able to slip through as anything else."""
    res = solve_self_consistency(1.0, lambda x: np.nan, tol=1e-8, max_iter=10)
    assert res.reason == "diverged" and not res.converged
    assert np.isnan(res.residuals[-1])

    res = solve_self_consistency(1.0, lambda x: np.inf, tol=1e-8, max_iter=10)
    assert res.reason == "diverged" and not res.converged

    # NaN buried inside an array, with the other entries finite
    def partial_nan(x):
        out = np.array(x, dtype=float).copy()
        out[0] = np.nan
        return out

    res = solve_self_consistency(np.zeros(3), partial_nan, tol=1e-8,
                                 max_iter=10)
    assert res.reason == "diverged", res.summary()
    # and max_abs_distance itself propagates it rather than averaging it away
    assert np.isnan(max_abs_distance(np.array([np.nan, 0.0]),
                                     np.array([0.0, 0.0])))


def test_SC7_the_callback_receives_the_post_mixing_state():
    """The docstring promises the POST-mixing state.  Handing it `prev`
    instead broke nothing, because the existing callback test checked only
    the iteration indices and that the residual was positive."""
    seen = []
    solve_self_consistency(1.0, lambda x: 0.0, damping=0.75, tol=1e-9,
                           max_iter=4,
                           callback=lambda i, s, r: seen.append(float(s)))
    # states after mixing are 0.75, 0.5625, ... -- geometric in the damping
    assert seen == pytest.approx([0.75, 0.5625, 0.421875, 0.31640625])


def test_SC7_the_window_divergence_gate_has_its_own_coverage():
    """`divergence_factor` against the recent window is a gate distinct from
    the three-consecutive-increases fast path.  Both `divergence_factor`
    1e3 -> 1e12 and the window 8 -> 2 used to survive mutation, so every
    `diverged` verdict in the suite came from the fast path alone.

    A SHORTER window has a LARGER minimum and is therefore strictly more
    permissive, so discriminating the window length needs growth that clears
    the 8-window threshold but not the 2-window one -- while zig-zagging so
    the monotone fast path never fires.  Residuals 1, 10, 5, 50, 25, 250,
    125, 1250: the last is 1250x the 8-window minimum (fires) but only 10x
    its immediate predecessor (a 2-window would not fire).
    """
    seq = iter([1.0, 10.0, 5.0, 50.0, 25.0, 250.0, 125.0, 1250.0])

    def step(x):
        return float(x) + next(seq)

    res = solve_self_consistency(0.0, step, tol=1e-8, max_iter=8)
    assert res.reason == "diverged", res.summary()
    assert res.n_iter == 8

    r = res.residuals
    # it was NOT the monotone fast path: no four strictly increasing in a row
    assert not any(r[i] < r[i + 1] < r[i + 2] < r[i + 3]
                   for i in range(len(r) - 3)), r
    # and it really is the FACTOR that decides: raise it past the jump and the
    # same sequence runs to the end
    seq2 = iter([1.0, 10.0, 5.0, 50.0, 25.0, 250.0, 125.0, 1250.0])
    res2 = solve_self_consistency(0.0, lambda x: float(x) + next(seq2),
                                  tol=1e-8, max_iter=8,
                                  divergence_factor=1e6)
    assert res2.reason == "max_iter", res2.summary()


def test_SC7_a_device_array_leaf_keeps_its_type_under_damping():
    """`_mix` must not coerce the LEAF through `np.asarray`.

    The container branches preserve dict/list/tuple/namedtuple types, but the
    leaf fell through to `np.asarray`, so a state whose leaves are array-LIKE
    rather than ndarray -- a JAX or torch array, the natural choice for the
    Dyson solve this driver exists to serve -- was handed to `step` as its own
    type on iteration 1 and as a plain ndarray from iteration 2 onward.  For a
    device array that also moves the state silently back to the host.

    Only `damping > 0` reaches the coercion (at damping 0 `_mix` short-
    circuits), so this is invisible in the default configuration and appears
    exactly when the caller takes the module's advice to add damping.
    """
    class ArrayLike:
        """Array-like but NOT an ndarray subclass -- a jax.Array's branch."""

        def __init__(self, v):
            self.v = np.asarray(v, dtype=float)

        def __array__(self, dtype=None, copy=None):
            return self.v if dtype is None else self.v.astype(dtype)

        def __add__(self, o):
            return ArrayLike(self.v + np.asarray(o))

        __radd__ = __add__

        def __mul__(self, o):
            return ArrayLike(self.v * np.asarray(o))

        __rmul__ = __mul__

    seen = []

    def step(s):
        seen.append(type(s).__name__)
        return ArrayLike(0.5 * (np.asarray(s) + 1.0))

    for damping in (0.0, 0.3):
        seen.clear()
        res = solve_self_consistency(ArrayLike([0.0, 0.0]), step, tol=1e-12,
                                     max_iter=400, damping=damping)
        assert res.converged, f"damping={damping}: {res.summary()}"
        assert set(seen) == {"ArrayLike"}, f"damping={damping}: step saw {set(seen)}"
        assert isinstance(res.state, ArrayLike), f"damping={damping}"
        assert np.asarray(res.state) == pytest.approx(np.ones(2), abs=1e-10)

    # a leaf that cannot do the arithmetic still falls back rather than raising
    assert _mix(np.ones(2), np.zeros(2), 0.3) == pytest.approx(0.7 * np.ones(2))
    assert _mix(np.array([1 + 1j]), np.array([0j]), 0.5) \
        == pytest.approx(np.array([0.5 + 0.5j]))
