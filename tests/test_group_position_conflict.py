"""Two points of ONE R-connected direction group may not be placed at
two different positions.

``analyze_spatial`` union-finds the spatial points that R-propagators
join into ``direction_groups``.  R carries ``δ(n − n')``, so a group is
one spatial point and has one coordinate; the integrators read that
coordinate through ``DiagramIntegrand._resolve_group_x``, which used to
take ``positions[p]`` for the first ``p`` of the group that the caller
had named -- iterating a ``frozenset``.

With one ψ leg per vertex no R chain can join two φ externals, so the
ordinary ``("phi_a(x)", "phi_b(y)")`` observable never reaches the
branch.  Two mechanisms do:

* an observable with an external response leg, ``("phi_a(x)",
  "psi_b(y)")`` -- at order 0 the single R IS the diagram, at order 2
  the chain ``x -- v -- y`` carries the C propagators with it;
* a LOCAL vertex with two or more ψ legs, which is what
  :class:`~sft_wick.workflow.specs.MultiplicativeImpulse` lowers to
  (``Expansion.evaluate`` already refuses that one, in
  ``expansion.py::_guard_single_site``; the L0 route did not).

**What the branch produced.**  On this file's system (N = 2, γ =
(0.8, 1.15), an asymmetric F, and a closed-form C that depends on
where the pair sits rather than only on its separation), order 2 of
``⟨φ_a(x, 3.0) ψ_b(y, 1.2)⟩`` with ``positions={'x': 0.0, 'y': 0.9}``:

==========================  =================
which position was picked   value
==========================  =================
x  (i.e. 0.0)               1.1042554919e-01
y  (i.e. 0.9)               4.9123696275e-02
==========================  =================

-- a factor 2.25 apart, and each is the value of a different physical
configuration: 1.1042554919e-01 is what the same call returns with both
externals at 0.0, and 4.9123696275e-02 with both at 0.9 (locked by
``GP1``).  The request for two positions answered a question about one
position, and ``frozenset`` iteration order decided which: with the
labels ``('x', 'y')`` the package returned 4.91e-02 under
``PYTHONHASHSEED`` 0, 1 and 2 and 1.10e-01 under 3, and renaming the
same externals to ``('u', 'v')`` flipped it at a fixed seed.

**Why it is refused rather than computed.**  The correlator at two
positions is ``δ(x − y)`` times the number the package reports at
coincidence: a distribution, not a value.  Returning 0 would be no
better -- the package reports δ-coefficients (order-0 ``⟨φψ⟩`` is
``exp(-γ Δt)``, not ``∞``), so 0 would be inconsistent with what it
reports at ``x = y``.
"""
from __future__ import annotations

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.evaluate import (
    integrate_diagrams,
    integrate_moment,
    integrate_two_point_qmc,
)
import sft_wick.evaluate as ev

N = 2
GAMMA = [0.8, 1.15]
T_F = 3.0
T_Y = 1.2
SEP = {"x": 0.0, "y": 0.9}          # the refused spelling
AT_X = {"x": 0.0, "y": 0.0}         # what picking x would have answered
AT_Y = {"x": 0.9, "y": 0.9}         # what picking y would have answered
EXT_T = {"x": T_F, "y": T_Y}
N_GAUSS = 16
MATCH = "response propagators joins them"


def _C_fn(n1, t1, n2, t2):
    """A C that depends on WHERE the pair sits, not only on its
    separation -- so the group's coordinate reaches the number even
    when both C legs belong to that one group."""
    x1 = float(np.asarray(n1).sum())
    x2 = float(np.asarray(n2).sum())
    env = np.exp(-0.5 * (x1 ** 2 + x2 ** 2))
    base = 0.6 * env * np.exp(-1.2 * abs(t1 - t2))
    return np.diag([base, 0.7 * base])


def _system(sigma2=None):
    F = np.zeros((N, N, N))
    F[0, 1, 1] = 0.9
    F[1, 0, 1] = 0.5
    F[1, 1, 0] = 0.3
    return sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(gamma=GAMMA),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            ),
            sigma2=sigma2,
        ),
    )


@pytest.fixture(scope="module")
def props():
    return _system().propagators(
        t_max=4.0, n_grid_t=20,
        c_closed_form=_C_fn, c_closed_form_only=True,
    )


@pytest.fixture(scope="module")
def psi_expansion():
    """⟨φ_a(x) ψ_b(y)⟩ -- the external-response-leg mechanism."""
    return _system().expand(("phi_a(x)", "psi_b(y)"), orders=[2])


@pytest.fixture(scope="module")
def phi_expansion():
    """⟨φ_a(x) φ_b(y)⟩ -- the control: never joined by an R chain."""
    return _system().expand(("phi_a(x)", "phi_b(y)"), orders=[2])


def _l1(expansion, props, positions, *, method="gauss_legendre", **kw):
    kwargs = dict(n_gauss=N_GAUSS) if method == "gauss_legendre" else {}
    kwargs.update(kw)
    return expansion.evaluate(
        props, positions=positions, t_final=T_F, external_times=EXT_T,
        component_pair=(0, 0), orders=[2], method=method, **kwargs,
    ).total


def _integrands(expansion):
    coupling = expansion.system.build_coupling_values()
    return [dt.build_integrand(coupling, {"a": 0, "b": 0})
            for dt in expansion.diagrams(2)]


def _integrand(expansion):
    """The order-2 diagram whose C propagator joins two DISTINCT points.

    Its sibling (C on one point, ``C(y_1, y_1)``) evaluates to exactly 0
    for the ``(a, b) = (0, 0)`` component pair of this F, which would
    make the companion "still evaluates" assertions vacuous.
    """
    for ig in _integrands(expansion):
        if any(left != right
               for left, right, _il, _ir in ig.spatial.c_propagators):
            return ig
    raise AssertionError("no diagram with a two-point C propagator")


# =====================================================================
# GP1 -- what the arbitrary pick produced
# =====================================================================


def test_GP1_the_two_picks_are_two_different_physical_answers(
        psi_expansion, props, monkeypatch):
    """Disable the new guard and the old resolver is back, verbatim.  Its
    answer to ``x != y`` is bit-for-bit the answer to one of the two
    coincident configurations -- a wrong number, not a rounding.
    """
    at_x = _l1(psi_expansion, props, AT_X)
    at_y = _l1(psi_expansion, props, AT_Y)
    assert abs(at_x) > 1e-3 and abs(at_y) > 1e-3, (at_x, at_y)
    # Not equal by symmetry: the pick decides a factor 2.25 here.
    assert abs(at_x / at_y) > 2.0, (at_x, at_y)

    monkeypatch.setattr(ev, "_require_one_position_per_group",
                        lambda *a, **k: None)
    old = _l1(psi_expansion, props, SEP)
    assert (old == pytest.approx(at_x, rel=1e-12, abs=1e-15)
            or old == pytest.approx(at_y, rel=1e-12, abs=1e-15)), (
        f"pre-guard value {old!r} is neither of the two coincident "
        f"answers {at_x!r} / {at_y!r}")


def test_GP1_pinned_values(psi_expansion, props):
    """The two candidate answers, pinned, so the sizes in this file's
    docstring stay honest."""
    assert _l1(psi_expansion, props, AT_X) == pytest.approx(
        1.1042554919e-01, rel=1e-8, abs=0.0)
    assert _l1(psi_expansion, props, AT_Y) == pytest.approx(
        4.9123696275e-02, rel=1e-8, abs=0.0)


# =====================================================================
# GP2 -- every entry point refuses
# =====================================================================


def _evaluate_point_keyed(ig, cache, positions):
    """``DiagramIntegrand.evaluate`` reached with a POINT-keyed
    ``directions`` dict -- the spelling its C lookups fall back to
    (``directions.get(dir_l, directions.get(sp_l))``), and the only one
    that can express two positions in one group.  Point-keyed means
    EVERY point needs an entry, internal ones included."""
    sp = ig.spatial
    times = dict.fromkeys(sp.external_points, T_F)
    times["y"] = T_Y
    # Inside the causal region, or the Theta product is 0 and the
    # companion "still evaluates" assertion is vacuous.
    ivars = list(sp.time_integration_vars)
    earlier = {v: set() for v in ivars}
    for early, late in sp.time_orderings:
        if early in earlier and late in earlier:
            earlier[late].add(early)
    ranked = sorted(ivars, key=lambda v: (len(earlier[v]), v))
    step = (T_F - T_Y) / (len(ranked) + 1)
    times.update({v: T_Y + (i + 1) * step for i, v in enumerate(ranked)})
    directions = {p: 0.9 for p in sp.direction_map}
    directions.update(positions)
    # One point of the integrand still carries the (-i)^n_response phase
    # that the integrators project out, so this one is complex.
    return ig.evaluate(times, directions, cache)


_ENTRY_POINTS = {
    "integrate_moment": lambda ig, c, p: integrate_moment(
        ig, T_F, c, method="qmc", n_samples=2 ** 8, seed=1,
        positions=p, external_times=EXT_T),
    "integrate_moment_qmc": lambda ig, c, p: ig.integrate_moment_qmc(
        T_F, c, n_samples=2 ** 8, seed=1, positions=p, external_times=EXT_T),
    "integrate_moment_qmc_vectorized": lambda ig, c, p: (
        ig.integrate_moment_qmc_vectorized(
            T_F, c, n_samples=2 ** 10, seed=1, positions=p,
            external_times=EXT_T)),
    "integrate_moment_gauss_legendre": lambda ig, c, p: (
        ig.integrate_moment_gauss_legendre(
            T_F, c, positions=p, n_gauss=N_GAUSS, external_times=EXT_T)),
    "integrate_moment_nquad": lambda ig, c, p: ig.integrate_moment_nquad(
        T_F, c, positions=p, external_times=EXT_T),
    "evaluate": lambda ig, c, p: (_evaluate_point_keyed(ig, c, p), 0.0),
    "integrate_two_point_qmc": lambda ig, c, p: integrate_two_point_qmc(
        [ig], T_F, p, c, n_samples=2 ** 8, seed=1, external_times=EXT_T),
}


@pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
def test_GP2_every_entry_point_refuses_two_positions_in_one_group(
        entry, psi_expansion, props):
    ig = _integrand(psi_expansion)
    with pytest.raises(ValueError, match=MATCH):
        _ENTRY_POINTS[entry](ig, props.cache, dict(SEP))


@pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
def test_GP2_every_entry_point_still_takes_one_position(
        entry, psi_expansion, props):
    """Non-vacuous companion: the same call at a single position runs and
    returns something that is not zero."""
    ig = _integrand(psi_expansion)
    value, _err = _ENTRY_POINTS[entry](ig, props.cache, dict(AT_Y))
    assert np.isfinite(abs(value))
    assert abs(value) > 1e-6, value


def test_GP2_integrate_diagrams_refuses_before_starting_workers(
        psi_expansion, props, monkeypatch):
    """The batch entry point checks every term up front, so with
    ``n_jobs != 1`` the error comes from the caller's process."""
    pytest.importorskip("joblib")
    import joblib

    def no_workers(*args, **kwargs):
        raise AssertionError("joblib.Parallel was started")

    monkeypatch.setattr(joblib, "Parallel", no_workers)
    with pytest.raises(ValueError, match=MATCH):
        integrate_diagrams(
            psi_expansion.diagrams(2),
            coupling_values=psi_expansion.system.build_coupling_values(),
            lambda_f=T_F, cache=props.cache, method="gauss_legendre",
            n_gauss=8, fixed_indices={"a": 0, "b": 0}, n_jobs=-1,
            positions=dict(SEP), external_times=EXT_T,
        )


@pytest.mark.parametrize(
    "method", ["gauss_legendre", "qmc_vectorized", "nquad", "qmc"])
def test_GP2_L1_evaluate_refuses_on_every_method(
        method, psi_expansion, props):
    kw = {"n_samples": 2 ** 10, "seed": 3} if method.startswith("qmc") else {}
    with pytest.raises(ValueError, match=MATCH):
        _l1(psi_expansion, props, SEP, method=method, **kw)


# =====================================================================
# GP3 -- the label spelling does not decide it
# =====================================================================


@pytest.mark.parametrize("labels", [("x", "y"), ("u", "v"), ("aa", "bb"),
                                    ("s", "t"), ("y", "z")])
def test_GP3_refusal_is_the_same_under_every_label_spelling(labels, props):
    """The very thing the old branch was sensitive to.  Same physics,
    five spellings of the external names: all five raise, and each
    message names its own two points."""
    lx, ly = labels
    exp = _system().expand((f"phi_a({lx})", f"psi_b({ly})"), orders=[2])
    with pytest.raises(ValueError) as excinfo:
        exp.evaluate(
            props, positions={lx: 0.0, ly: 0.9}, t_final=T_F,
            external_times={lx: T_F, ly: T_Y}, component_pair=(0, 0),
            orders=[2], method="gauss_legendre", n_gauss=8,
        )
    message = str(excinfo.value)
    assert f"'{lx}'" in message and f"'{ly}'" in message, message
    assert "0.9" in message, message


# =====================================================================
# GP4 -- the control: nothing else is refused
# =====================================================================


@pytest.mark.parametrize(
    "method", ["gauss_legendre", "qmc_vectorized", "nquad", "qmc"])
def test_GP4_two_phi_externals_apart_are_untouched(
        method, phi_expansion, props):
    """φ externals are never in one group (one ψ leg per vertex), so the
    ordinary separated two-point function still runs on every integrator
    and still moves with the separation."""
    kw = {"n_samples": 2 ** 12, "seed": 3} if method.startswith("qmc") else {}
    apart = _l1(phi_expansion, props, SEP, method=method, **kw)
    together = _l1(phi_expansion, props, AT_X, method=method, **kw)
    assert np.isfinite(apart) and abs(apart) > 1e-6, apart
    assert abs(apart - together) > 1e-3 * abs(together), (apart, together)


def test_GP4_one_named_external_per_group_is_not_refused(
        psi_expansion, props):
    """Only a CONFLICT is refused.  Naming one point of the group leaves
    the group's coordinate determined, so it still evaluates."""
    value = _l1(psi_expansion, props, {"x": 0.9})
    assert value == pytest.approx(_l1(psi_expansion, props, AT_Y),
                                  rel=1e-12, abs=1e-15)


def test_GP4_equal_positions_spelled_as_vectors_are_not_refused(
        psi_expansion, props):
    """The comparison is elementwise, so d-dim positions that agree pass
    and ones that differ in a single component do not."""
    same = {"x": np.array([0.9, 0.0]), "y": [0.9, 0.0]}
    assert np.isfinite(_l1(psi_expansion, props, same))
    with pytest.raises(ValueError, match=MATCH):
        _l1(psi_expansion, props, {"x": [0.9, 0.0], "y": [0.9, 0.2]})


# =====================================================================
# GP5 -- the second mechanism: a local vertex with two ψ legs
# =====================================================================


def test_GP5_multiplicative_noise_is_refused_at_L0_too(props):
    """``Expansion.evaluate`` already refused this one
    (``_guard_single_site``); ``integrate_diagrams`` took the branch.
    Both refuse now, so the L0 route is no longer the way around it."""
    # g0 diagonal so D0 = g0 g0^T is, and the default diag_C stands; the
    # phi-dependence (g1), which is what makes the two-psi vertices, is
    # asymmetric.
    g0 = np.array([[0.7, 0.0], [0.0, 0.5]])
    g1 = np.zeros((N, 2, N))
    g1[0, 0, 1] = 0.4
    g1[1, 1, 0] = -0.3
    system = _system(sigma2=sw.MultiplicativeImpulse(g0=g0, g1=g1))
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[1])
    terms = exp.diagrams(1)
    joined = [dt for dt in terms
              if any(len(set(g) & {"x", "y"}) > 1
                     for g in dt.analyze_spatial().direction_groups)]
    assert joined, "no order-1 diagram joins the two externals"

    with pytest.raises(ValueError, match=MATCH):
        integrate_diagrams(
            joined, coupling_values=system.build_coupling_values(),
            lambda_f=T_F, cache=props.cache, method="gauss_legendre",
            n_gauss=8, fixed_indices={"a": 0, "b": 0},
            positions=dict(SEP),
        )
    # And the L1 guard still fires first, with its own message.
    with pytest.raises(ValueError, match="MultiplicativeImpulse"):
        exp.evaluate(props, positions=SEP, t_final=T_F,
                     component_pair=(0, 0), orders=[1],
                     method="gauss_legendre", n_gauss=8)
