"""The kink between an internal time and a fixed external time.

``_kink_pairs`` orders two integration variables, which covers a kink whose
two ends are both integrated.  The same three sources -- the ends of a C
propagator when C is kinked on its diagonal, the parents of a vertex with
several ψ legs, a coupling callable declaring ``has_coincident_time_kinks``
-- can put the kink between an integration variable and an external point
held at its own time (``external_times``), and ``u = t*`` with ``t*``
constant is not an ordering between two variables.  The integrators cut the
variable's range at ``t*`` instead (``_extra_bounds``), and the cut carries:

* a variable held **above** ``t*`` holds every later variable above it --
  otherwise the piece has an outer variable whose range collapses to zero
  width at ``t*``, moving the kink rather than removing it;
* a variable held **below** ``t*`` has upper bound ``min(parents, t*)``,
  which is kinked where a variable parent crosses ``t*`` -- so the parent
  takes the cut too.

With every external at one time the cut lies on the boundary of the domain
and none is made, which is what keeps every diagram without such a kink
bit-identical.

Reference: the exact Itô moment hierarchy of the same system
(``examples/reference/ito_moments.py``, transported over the lag between
the two external times), which shares no code with the package.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

import sft_wick as sw
from sft_wick.evaluate import (_applied_cut_bounds, _constant_bounds,
                               _kink_constant_cuts, _kink_orientations,
                               _kink_pairs)
from sft_wick.workflow.specs import ConstantImpulse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "examples" / "reference"))
import ito_moments as im  # noqa: E402

T_X, T_Y = 1.9, 1.3          # the two external times of the unit tests
T_MIN, T_CEIL = 0.0, 1.9
FIXED = {"x": T_X, "y": T_Y}


def _spatial(orderings, c_pairs, ivars=("u", "v")):
    return SimpleNamespace(
        equal_time_aliases=(), time_integration_vars=tuple(ivars),
        time_orderings=list(orderings),
        c_propagators=[(l, r, None, None) for l, r in c_pairs])


def _cuts(spatial, fixed=FIXED, *, c_kink=True, extra=(), coupling_pairs=()):
    bounds = _constant_bounds(spatial, spatial.time_orderings, fixed, T_MIN,
                              T_CEIL, extra)
    return _kink_constant_cuts(spatial, c_kink, bounds, fixed,
                               spatial.time_orderings, coupling_pairs, extra)


# --------------------------------------------------------------------------
# what is cut
# --------------------------------------------------------------------------

def test_a_C_propagator_to_a_fixed_external_is_cut():
    """``u ≤ t_x`` and C kinked between ``u`` and the external at ``t_y``:
    the line ``u = t_y`` runs through the domain."""
    sp = _spatial([("u", "x")], [("u", "y")], ivars=("u",))
    assert _cuts(sp) == [("u", T_Y)]


def test_no_cut_with_every_external_at_one_time():
    """The usual case: the kink is then the boundary ``u = t_final``."""
    sp = _spatial([("u", "x")], [("u", "y")], ivars=("u",))
    assert _cuts(sp, {"x": T_X, "y": T_X}) == []


def test_no_cut_when_the_causal_order_already_holds_the_variable_below():
    """``u ≤ t_y`` makes the same kink the boundary again."""
    sp = _spatial([("u", "y")], [("u", "y")], ivars=("u",))
    assert _cuts(sp) == []


def test_no_cut_without_a_kinked_C():
    sp = _spatial([("u", "x")], [("u", "y")], ivars=("u",))
    assert _cuts(sp, c_kink=False) == []


def test_no_cut_when_no_external_is_pinned():
    """``integrate_over='all'`` sweeps every external, so there is no
    constant to cut at; those kinks are pairs instead (below)."""
    sp = _spatial([("u", "x")], [("u", "y")], ivars=("u",))
    assert _cuts(sp, {}) == []


def test_a_swept_external_pairs_like_an_integration_variable():
    """A swept external is drawn before every internal variable, so an
    ordering with one is carried by the mapping: its kinks are pairs, not
    cuts.  Left unsplit they cost more than the other splits buy -- each
    piece bounds a variable by another, which bends the unsplit kink line
    into a curve."""
    sp = _spatial([("u", "x")], [("u", "y")], ivars=("u",))
    assert _kink_pairs(sp, True, sp.time_orderings) == []
    assert _kink_pairs(sp, True, sp.time_orderings, swept=("y",)) \
        == [("u", "y")]
    assert len(_kink_orientations(sp, [("u", "y")], sp.time_orderings,
                                  swept=("y",))) == 2


def test_two_swept_externals_are_a_pair():
    """Two swept externals are drawn by ``_swept_external_order``, and the
    split now pairs them: the integrators pass that function the orderings
    the split added, so the extra edge reaches the sampler.  Left unpaired
    the kink cost seven orders of magnitude on demo 7 (5.9e-08 at 12 nodes
    against 1e-12 once paired)."""
    sp = _spatial([], [("x", "y")], ivars=("u",))
    assert _kink_pairs(sp, True, sp.time_orderings, swept=("x", "y")) \
        == [("x", "y")]
    assert len(_kink_orientations(sp, [("x", "y")], sp.time_orderings,
                                  swept=("x", "y"))) == 2


def test_two_swept_externals_ordered_through_a_fixed_external_are_not_a_pair():
    """``x -> s -> y`` with ``s`` pinned: ``_swept_external_order`` already
    draws ``x`` first, so the split must not add the reverse ordering that
    the sampler would ignore.  The chain runs through a point that is not a
    split variable, which ``_later_sets`` does not follow."""
    sp = _spatial([("x", "s"), ("s", "y")], [("x", "y")], ivars=("u",))
    assert _kink_pairs(sp, True, sp.time_orderings, swept=("x", "y")) == []


def test_a_swept_external_already_ordered_is_not_a_pair():
    sp = _spatial([("u", "y")], [("u", "y")], ivars=("u",))
    assert _kink_pairs(sp, True, sp.time_orderings, swept=("y",)) == []


def test_a_multi_psi_vertex_with_a_fixed_external_parent_is_cut():
    """``v ≤ min(u, t_y)`` changes branch where ``u`` crosses ``t_y``; the
    parents rule of ``_kink_candidates`` with a constant for one parent."""
    sp = _spatial([("v", "y"), ("v", "u"), ("u", "x")], [], ivars=("u", "v"))
    assert _cuts(sp, c_kink=False) == [("u", T_Y)]


def test_a_coupling_callable_kink_against_a_fixed_external_is_cut():
    """``has_coincident_time_kinks`` on a leg whose partner is a fixed
    external (an ``already_R_contracted`` vertex aliases the leg onto it)."""
    sp = _spatial([("u", "x")], [], ivars=("u",))
    assert _cuts(sp, c_kink=False, coupling_pairs=(("u", "y"),)) \
        == [("u", T_Y)]


def test_the_parent_of_a_cut_variable_is_cut_too():
    """Cutting the inner variable at ``t*`` leaves ``v ≤ min(u, t*)``, which
    is kinked at ``u = t*``: the parent takes the same cut."""
    sp = _spatial([("v", "u"), ("u", "x")], [("v", "y")])
    assert _cuts(sp) == [("v", T_Y)]
    assert _cuts(sp, extra=(("v", "le", T_Y),)) == [("u", T_Y)]
    # ...and above t*, v carries u with it, so there is nothing left to cut.
    assert _cuts(sp, extra=(("v", "ge", T_Y),)) == []


def test_a_cut_is_not_repeated_once_it_holds():
    sp = _spatial([("u", "x")], [("u", "y")], ivars=("u",))
    assert _cuts(sp, extra=(("u", "le", T_Y),)) == []
    assert _cuts(sp, extra=(("u", "ge", T_Y),)) == []


def test_a_lower_cut_is_carried_to_the_later_variables():
    """``v ≥ t*`` and ``v ≤ u`` give ``u ≥ t*``.  The causal mapping carries
    an upper bound (``min(parents)``) but not a lower one."""
    sp = _spatial([("v", "u"), ("u", "x")], [("v", "y")])
    upper, lower = _applied_cut_bounds(sp, sp.time_orderings,
                                       (("v", "ge", T_Y),))
    assert lower == {"v": T_Y, "u": T_Y} and upper == {}
    upper, lower = _applied_cut_bounds(sp, sp.time_orderings,
                                       (("u", "le", T_Y),))
    assert upper == {"u": T_Y, "v": T_Y} and lower == {}


def test_no_cut_no_bounds():
    """Empty in, empty out: a diagram with no cut takes the mapping it
    always took, to the bit."""
    sp = _spatial([("v", "u"), ("u", "x")], [("v", "y")])
    assert _applied_cut_bounds(sp, sp.time_orderings, ()) == ({}, {})


# --------------------------------------------------------------------------
# against the exact hierarchy
# --------------------------------------------------------------------------

N = 2
GAMMA = 1.0
S2 = np.array([[0.8, 0.3], [0.3, 0.5]])
F = np.array([[[0.25, -0.40], [0.15, 0.30]],
              [[-0.35, 0.20], [0.45, -0.10]]])
POS = {"x": 0.0, "y": 0.7}


def _system():
    """White noise (kinked C) and one quadratic vertex, two points."""
    return sw.System(
        field=sw.FieldSpec("phi", N), linear=sw.DiagonalA(gamma=[GAMMA] * N),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=1.0),
                spatial=sw.ExponentialSpatial(sigma_x=1.0)),
            sigma2=ConstantImpulse(S2)))


def _sde():
    """The Markov embedding at the two points: state ``(component, point)``,
    one tag counting the vertex."""
    D = 2 * N
    idx = {(c, i): c * 2 + i for c in range(N) for i in range(2)}
    sde = im.PolySDE(D=D, n_tags=1)
    sde.add_linear_drift(-GAMMA * np.eye(D))
    S = np.zeros((D, D))
    Q = np.zeros((D, D, D))
    for (c, i), k in idx.items():
        for (d, j), l in idx.items():
            S[k, l] = S2[c, d]          # the same white noise at every point
        for d in range(N):
            for e in range(N):
                Q[k, idx[(d, i)], idx[(e, i)]] += F[c, d, e]
    sde.add_constant_diffusion(S)
    sde.add_quadratic_drift(Q, tag=(1,))
    return sde, idx, D


def _reachable(sde, root):
    """``(index, M)`` of the ``(monomial, tag)`` nodes reachable from
    ``root``, and the generator on them."""
    index, rows, queue = {}, [], [root]
    while queue:
        node = queue.pop()
        if node in index:
            continue
        index[node] = len(rows)
        alpha, tag = node
        row = []
        for beta, c, dtag in sde.generator(alpha):
            src_tag = tuple(x - y for x, y in zip(tag, dtag))
            if min(src_tag, default=0) < 0:
                continue
            row.append(((beta, src_tag), c))
            if (beta, src_tag) not in index:
                queue.append((beta, src_tag))
        rows.append(row)
    data, ri, ci = [], [], []
    for r, row in enumerate(rows):
        for src, c in row:
            data.append(c)
            ri.append(r)
            ci.append(index[src])
    return index, csr_matrix((data, (ri, ci)),
                             shape=(len(rows), len(rows)))


def _two_time(a, b, t_late, t_early, order):
    """``E[φ_a(x, t_late) φ_b(y, t_early)]`` at ``O(F^order)``: transport the
    late leg over the lag, then evaluate each monomial it reaches together
    with the early leg."""
    sde, idx, D = _sde()
    late = im.unit(D, idx[(a, 0)])
    early = im.unit(D, idx[(b, 1)])
    tag = (order,)
    index, M = _reachable(sde, (late, tag))
    e = np.zeros(len(index))
    e[index[(late, tag)]] = 1.0
    s = float(t_late) - float(t_early)
    row = expm_multiply(M.T.tocsr() * s, e) if s > 0.0 else e
    nodes = [(node, row[k]) for node, k in index.items() if row[k] != 0.0]
    targets = [(tuple(m + n for m, n in zip(mono, early)), ntag)
               for (mono, ntag), _ in nodes]
    got = im.solve(sde, targets, [t_early])
    return float(sum(
        w * got[(tuple(m + n for m, n in zip(mono, early)), ntag)][0]
        for (mono, ntag), w in nodes))


@pytest.fixture(scope="module")
def setup():
    system = _system()
    props = system.propagators(t_max=T_X + 0.1, c_closed_form="auto",
                               c_closed_form_only=True, diag_C=False,
                               progress=False)
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2], diag_C=False)
    return exp, props


def _package(setup, ab, order, method, ext, **kw):
    exp, props = setup
    return exp.evaluate(props, positions=POS, t_final=max(ext.values()),
                        component_pair=ab, orders=[order], method=method,
                        external_times=ext, **kw).total


@pytest.mark.parametrize("n_gauss", [8, 16])
@pytest.mark.parametrize("ab", [(0, 1), (1, 1)])
@pytest.mark.parametrize("order", [0, 2])
def test_two_time_gauss_legendre_matches_the_hierarchy(setup, ab, order,
                                                       n_gauss):
    """The kink at ``u = t_y`` is cut, so the rate is exhausted at 8 nodes
    rather than falling to ``n^-2``: before the cut the order-2 channel was
    4.5e-03, 9.1e-04 and 2.2e-04 off at 8, 16 and 32 nodes, and ``nquad``
    1.3e-07."""
    ext = {"x": T_X, "y": T_Y}
    ref = _two_time(ab[0], ab[1], T_X, T_Y, order)
    got = _package(setup, ab, order, "gauss_legendre", ext, n_gauss=n_gauss)
    assert got == pytest.approx(ref, rel=1e-11, abs=0.0)


@pytest.mark.parametrize("ab", [(0, 1), (1, 1)])
def test_two_time_nquad_matches_the_hierarchy(setup, ab):
    ext = {"x": T_X, "y": T_Y}
    ref = _two_time(ab[0], ab[1], T_X, T_Y, 2)
    assert _package(setup, ab, 2, "nquad", ext) == pytest.approx(
        ref, rel=1e-8, abs=0.0)


@pytest.mark.parametrize("ab", [(0, 1), (1, 1)])
def test_the_earlier_external_may_be_either_point(setup, ab):
    """``y`` late and ``x`` early: the cut follows the times, not the order
    of the observable's points."""
    ext = {"x": T_Y, "y": T_X}
    ref = _two_time(ab[1], ab[0], T_X, T_Y, 2)
    got = _package(setup, ab, 2, "gauss_legendre", ext, n_gauss=8)
    assert got == pytest.approx(ref, rel=1e-11, abs=0.0)


@pytest.mark.parametrize("ab", [(0, 1), (1, 1)])
def test_equal_external_times_take_the_unsplit_mapping(setup, ab):
    """No cut is made when the externals share a time, so the value is the
    one the unsplit mapping gives, to the bit."""
    pinned = _package(setup, ab, 2, "gauss_legendre", {"x": T_X, "y": T_X},
                      n_gauss=8)
    exp, props = setup
    default = exp.evaluate(props, positions=POS, t_final=T_X,
                           component_pair=ab, orders=[2],
                           method="gauss_legendre", n_gauss=8).total
    assert pinned == default


def test_the_comparison_is_not_vacuous():
    """The two-time value differs from the equal-time one by more than the
    tolerances above: the checks would notice a cut made at the wrong
    time."""
    equal = _two_time(0, 1, T_X, T_X, 2)
    two = _two_time(0, 1, T_X, T_Y, 2)
    assert abs(two - equal) / abs(equal) > 0.1
