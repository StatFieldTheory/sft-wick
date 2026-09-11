"""Callable non-local couplings at several sets of points (VC0-VC4).

At order >= 2 a diagram holds several copies of a vertex.  Its coupling
sum lists each copy at its own points and routes the legs of non-local
copies between them: the order-2 six-point function with two kappa3
vertices is one DiagramTerm whose 360 terms spread the six leg labels over
the two copies in all 10 partitions and 36 leg orders.  Every distinct
``(name, spatial_args)`` pair now gets a symbol of its own and is evaluated
at its own points; up to 0.5.0 ``DiagramTerm.build_integrand`` refused a
callable at more than one set of points.

Checking this exposed a defect that affected static couplings as well.
The leg structure of an ``equal_time`` vertex (its legs share one time) or
an ``already_R_contracted`` one (its legs take their partners' times and
drop their R factors) was read off the vertex instances, while the coupling
sum routes the legs of the copies differently from term to term.  Every
term was integrated with the measure of one routing; at equal external
times all routings have the same measure, which is why it went unnoticed.
``compute_moment`` now splits such a DiagramTerm by leg structure
(``perturbation._split_by_instance_structure``).

References, none of them sft-wick contraction code:

* a numpy hand contraction.  The order-2 six-point function is the sum over
  the 10 partitions of the six external points into two triples of the
  products of two order-1 three-point functions, each a sum over the 6 leg
  orders.  Per DiagramTerm, the partitions are read off the diagram's R
  propagators (which external each leg label is attached to) and the leg
  sets of its coupling-sum terms;
* demo 4's Campbell closed form ``K_R`` (``examples/demo4/poisson_noise.py``),
  the exact connected three-point function at F = 0, so that the order-2
  six-point function of demo 4 with a kappa3 vertex alone is the sum over
  the 10 partitions of ``K_R K_R``.

* **VC0** the kernels are not symmetric in their points, and evaluating
  every copy at the first copy's points moves each checked value by more
  than 1e-2.
* **VC1** a callable returning a constant tensor equals the tensor, diagram
  by diagram, for plain, ``equal_time`` and ``already_R_contracted`` copies
  on every integrator.
* **VC2** a point- and time-dependent callable against the hand
  contraction, per DiagramTerm.
* **VC3** demo 4's callables against the closed form, at distinct points and
  times.
* **VC4** static couplings: two ``equal_time`` copies, and an ``equal_time``
  or an ``already_R_contracted`` vertex next to a plain one, against the
  hand contraction at distinct external times, where the old measure is
  shown to differ; at equal times it did not.

Distinct external times and ``t_min = 0.3`` throughout; N = 3 with a
generic component tuple.  R = Theta (gamma = 0) in VC0-VC2 makes every time
integral a polynomial one, so Gauss-Legendre with a few nodes and QMC on a
time-independent kernel are exact and a 1e-12 comparison is meaningful for
every route.
"""
from __future__ import annotations

import functools
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial import Polynomial

import sft_wick as sw
from sft_wick.expressions import Product, Sum, Symbol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "demo4"))
import poisson_noise as nz  # noqa: E402
import poisson_system as dsys  # noqa: E402

N = 3
T_MIN = 0.3
LABELS = ("x1", "x2", "x3", "x4", "x5", "x6")
OBS = tuple(f"phi_{c}({lab})" for c, lab in zip("abcdef", LABELS))
POS = dict(zip(LABELS, (0.3, -0.8, 1.1, 0.05, -0.45, 0.7)))
TIMES = dict(zip(LABELS, (1.7, 0.55, 1.25, 0.4, 1.45, 0.85)))
EQUAL_TIMES = dict.fromkeys(LABELS, 1.7)
COMPS = (0, 2, 1, 1, 0, 2)
PERMS = tuple(itertools.permutations(range(3)))
FIRST, SECOND = (0, 1, 2), (3, 4, 5)
#: The 10 partitions of the six externals into two triples.
PARTITIONS = tuple((A, tuple(i for i in range(6) if i not in A))
                   for A in itertools.combinations(range(6), 3) if 0 in A)
TOL = 1e-12
MIN_GAP = 1e-2
NOISE = sw.GaussianNoise(kappa2=sw.SeparableTranslation(
    temporal=sw.ExponentialTemporal(lam=0.3, sigma_t=0.5),
    spatial=sw.GaussianSpatial(sigma_x=1.0)))
FLAGS = {"plain": {}, "equal_time": dict(equal_time=True),
         "absorbed": dict(already_R_contracted=True)}


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
class Kernel:
    """``κ_pqr(x_0,t_0; x_1,t_1; x_2,t_2)
    = Σ_k A_k[p,q,r] Π_l (a_kl + b_kl x_l)(1 + c_kl t_l)``.

    :meth:`single` is the per-sample contract, :meth:`batch` the batched
    one; positions are scalars."""

    def __init__(self, terms):
        self.terms = [tuple(np.asarray(v, dtype=float) for v in term)
                      for term in terms]

    def value(self, comps, xs, ts) -> float:
        return sum(A[tuple(comps)] * np.prod((a + b * np.asarray(xs))
                                             * (1 + c * np.asarray(ts)))
                   for A, a, b, c in self.terms)

    def single(self, n_list, t_list) -> np.ndarray:
        n = np.asarray(n_list, dtype=float).reshape(3)
        t = np.asarray(t_list, dtype=float).reshape(3)
        out = np.zeros((N,) * 3)
        for A, a, b, c in self.terms:
            out += np.prod((a + b * n) * (1 + c * t)) * A
        return out

    def batch(self, n_arr, t_arr) -> np.ndarray:
        n = np.asarray(n_arr, dtype=float)
        t = np.asarray(t_arr, dtype=float)
        out = np.zeros((t.shape[1],) + (N,) * 3)
        for A, a, b, c in self.terms:
            f = np.prod((a[:, None] + b[:, None] * n)
                        * (1 + c[:, None] * t), axis=0)
            out += f[:, None, None, None] * A[None]
        return out


@functools.lru_cache(maxsize=None)
def _kernel(time_dependent: bool, seed: int = 20260911) -> Kernel:
    rng = np.random.default_rng(seed)
    terms = []
    for _ in range(2):
        A = rng.normal(size=(N, N, N))
        a = rng.uniform(0.5, 1.5, size=3)
        b = 0.8 * rng.normal(size=3)
        c = 0.7 * rng.normal(size=3) if time_dependent else np.zeros(3)
        terms.append((A, a, b, c))
    return Kernel(terms)


@functools.lru_cache(maxsize=None)
def _constant_tensor(seed: int = 11) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(N, N, N))


class Constant:
    """A callable that ignores its points and returns one tensor."""

    def __init__(self, tensor):
        self.tensor = np.asarray(tensor)

    def single(self, n_list, t_list):  # noqa: ARG002
        return self.tensor

    def batch(self, n_arr, t_arr):  # noqa: ARG002
        S = np.asarray(t_arr).shape[1]
        return np.broadcast_to(self.tensor, (S,) + self.tensor.shape).copy()


# --------------------------------------------------------------------------- #
# Hand contraction (numpy only)
# --------------------------------------------------------------------------- #
def _h(kern: Kernel, idx, measure: str, pts=None) -> float:
    """Order-1 ``<phi phi phi>`` of the externals ``idx`` (R = Theta)::

        (1/6) Σ_β ∫ κ_{c_β(0) c_β(1) c_β(2)}(legs at the points of β)

    Leg ``l`` pairs with external ``idx[β(l)]`` and takes its component and
    its point.  ``(-1) (-i)^3 (i/6) = 1/6``: the order-1 sign, three R and
    the ``NonLocalVertex`` factor.  ``measure``: ``"plain"``, each leg time
    over ``[t_min, t_partner]``; ``"equal_time"``, one shared time over
    ``[t_min, min t_partner]``; ``"absorbed"``, each leg at its partner's
    time.  ``pts`` (default ``idx``) are the externals whose points the
    kernel is evaluated at; VC0 passes the first triple for every copy."""
    pts = idx if pts is None else pts
    total = 0.0
    for beta in PERMS:
        comps = tuple(COMPS[idx[beta[leg]]] for leg in range(3))
        xs = np.array([POS[LABELS[pts[beta[leg]]]] for leg in range(3)])
        ts = np.array([TIMES[LABELS[pts[beta[leg]]]] for leg in range(3)])
        for A, a, b, c in kern.terms:
            ang = np.prod(a + b * xs)
            if measure == "plain":
                tim = np.prod((ts - T_MIN) + 0.5 * c * (ts ** 2 - T_MIN ** 2))
            elif measure == "equal_time":
                poly = Polynomial([1.0])
                for leg in range(3):
                    poly = poly * Polynomial([1.0, c[leg]])
                prim = poly.integ()
                tim = prim(ts.min()) - prim(T_MIN)
            else:
                tim = np.prod(1 + c * ts)
            total += A[comps] * ang * tim
    return total / 6.0


def _hand(kern, measure, partitions=PARTITIONS, one_point_set=False) -> float:
    """Σ over ``partitions`` of the products of the two three-point
    functions: the order-2 six-point function.  ``one_point_set``: every
    copy evaluated at the points of the first triple (VC0)."""
    pts = FIRST if one_point_set else None
    return sum(_h(kern, A, measure, pts) * _h(kern, B, measure, pts)
               for A, B in partitions)


def _symbols(expr) -> list:
    if isinstance(expr, Symbol):
        return [expr]
    if isinstance(expr, Product):
        return [s for f in expr.factors for s in _symbols(f)]
    if isinstance(expr, Sum):
        return [s for t in expr.terms for s in _symbols(t)]
    return []


def _partitions_of(dt) -> set:
    """The partitions a DiagramTerm's coupling sum holds: each term's two K
    leg sets, mapped to externals through the diagram's R propagators."""
    external_of = {p.spatial_right: p.spatial_left for p in dt.propagators
                   if p.kind == "R"}
    index_of = {lab: k for k, lab in enumerate(LABELS)}
    terms = (dt.coupling_sum.terms if isinstance(dt.coupling_sum, Sum)
             else (dt.coupling_sum,))
    out = set()
    for term in terms:
        triples = [tuple(sorted(index_of[external_of[leg]]
                                for leg in s.spatial_args))
                   for s in _symbols(term) if s.name == "K"]
        assert len(triples) == 2
        A, B = sorted(triples)
        out.add((A, B))
    return out


# --------------------------------------------------------------------------- #
# sft-wick side
# --------------------------------------------------------------------------- #
def _system(coupling, *, vectorized: bool, kind: str) -> sw.System:
    sw.reset_uid_counter()
    return sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(gamma=[0.0] * N), noise=NOISE, t_min=T_MIN,
        nonlocal_vertices=[sw.NonLocalVertex(
            "K", order=3, coupling=coupling, coupling_vectorized=vectorized,
            **FLAGS[kind])])


#: ``key -> (coupling, coupling_vectorized)``.  ``const_*`` return one
#: tensor whatever their points; ``kernel_t`` depends on points and times,
#: ``kernel_c`` on points only (which makes the QMC routes exact).
COUPLINGS = {
    "tensor": lambda: (_constant_tensor(), False),
    "const_single": lambda: (Constant(_constant_tensor()).single, False),
    "const_batch": lambda: (Constant(_constant_tensor()).batch, True),
    "kernel_t_single": lambda: (_kernel(True).single, False),
    "kernel_t_batch": lambda: (_kernel(True).batch, True),
    "kernel_c_single": lambda: (_kernel(False).single, False),
    "kernel_c_batch": lambda: (_kernel(False).batch, True),
}


@functools.lru_cache(maxsize=None)
def _setup(coupling_key: str, kind: str):
    """``(expansion, props)``, built once per coupling and vertex kind: the
    order-2 six-point expansion takes about a second."""
    coupling, vectorized = COUPLINGS[coupling_key]()
    system = _system(coupling, vectorized=vectorized, kind=kind)
    return (system.expand(OBS, orders=[2], diag_C=False),
            system.propagators(t_max=2.0, n_grid_t=8, progress=False))


def _run(coupling_key, *, kind, method, kw):
    expansion, props = _setup(coupling_key, kind)
    res = expansion.evaluate(
        props, positions=POS, t_final=max(TIMES.values()),
        external_times=TIMES, component_pair=COMPS, orders=[2],
        method=method, **kw)
    dts = expansion.dts_by_order[2]
    return [(dts[d["diagram_idx"]], d["value"]) for d in res.per_diagram]


# (method, batched contract?, time-dependent kernel?, kwargs); a
# time-independent kernel makes the integrand constant, so QMC is exact.
ROUTES = {
    "plain": [("gauss_legendre", True, True, dict(n_gauss=2)),
              ("gauss_legendre", False, True, dict(n_gauss=2)),
              ("qmc_vectorized", True, False, dict(n_samples=2 ** 5, seed=3)),
              ("qmc_scalar", False, False, dict(n_samples=2 ** 4, seed=3))],
    "equal_time": [("gauss_legendre", True, True, dict(n_gauss=4)),
                   ("nquad", False, True, {}),
                   ("qmc_vectorized", True, False,
                    dict(n_samples=2 ** 5, seed=3)),
                   ("qmc_scalar", False, False,
                    dict(n_samples=2 ** 4, seed=3))],
    "absorbed": [("gauss_legendre", True, True, dict(n_gauss=2)),
                 ("nquad", False, True, {}),
                 ("qmc_vectorized", True, True, dict(n_samples=2 ** 4)),
                 ("qmc_scalar", False, True, dict(n_samples=2 ** 4))],
}
CASES = [(kind, *route) for kind, routes in ROUTES.items()
         for route in routes]
IDS = [f"{kind}-{method}-{'batched' if batched else 'per_sample'}"
       for kind, method, batched, _td, _kw in CASES]


# --------------------------------------------------------------------------- #
# VC0: the checks below are not vacuous
# --------------------------------------------------------------------------- #
def test_VC0_kernel_is_not_symmetric_in_its_points():
    kern = _kernel(True)
    xs, ts = np.array([0.3, -0.8, 1.1]), np.array([1.7, 1.2, 0.9])
    base = kern.value((0, 2, 1), xs, ts)
    for perm in PERMS[1:]:
        moved = kern.value((0, 2, 1), xs[list(perm)], ts[list(perm)])
        assert abs(moved - base) > MIN_GAP * abs(base), perm


@pytest.mark.parametrize("time_dependent", [True, False])
@pytest.mark.parametrize("kind", sorted(FLAGS))
def test_VC0_one_point_set_for_every_copy_is_detectable(kind, time_dependent):
    kern = _kernel(time_dependent)
    ok = _hand(kern, kind)
    wrong = _hand(kern, kind, one_point_set=True)
    assert abs(wrong - ok) > MIN_GAP * abs(ok)


# --------------------------------------------------------------------------- #
# VC1: a constant callable is the static tensor, diagram by diagram
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,method,batched,_td,kw", CASES, ids=IDS)
def test_VC1_constant_callable_equals_the_tensor(kind, method, batched, _td,
                                                 kw):
    got = _run("const_batch" if batched else "const_single", kind=kind,
               method=method, kw=kw)
    ref = _run("tensor", kind=kind, method=method, kw=kw)
    assert len(got) == len(ref) == (10 if kind == "equal_time" else 1)
    scale = max(abs(v) for _dt, v in ref)
    for (dt_g, g), (dt_r, r) in zip(got, ref):
        assert dt_g.propagators == dt_r.propagators
        assert abs(g - r) <= TOL * scale, (g, r)


# --------------------------------------------------------------------------- #
# VC2: a point-dependent callable against the hand contraction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,method,batched,time_dependent,kw", CASES,
                         ids=IDS)
def test_VC2_point_dependent_callable_matches_hand(kind, method, batched,
                                                   time_dependent, kw):
    kern = _kernel(time_dependent)
    got = _run(f"kernel_{'t' if time_dependent else 'c'}"
               f"_{'batch' if batched else 'single'}",
               kind=kind, method=method, kw=kw)
    seen = []
    scale = abs(_hand(kern, kind))
    for dt, value in got:
        parts = _partitions_of(dt)
        seen.extend(parts)
        ref = _hand(kern, kind, partitions=tuple(parts))
        assert abs(value - ref) <= TOL * scale, (sorted(parts), value, ref)
    assert sorted(seen) == sorted(PARTITIONS), "each partition exactly once"
    total = sum(v for _dt, v in got)
    assert total == pytest.approx(_hand(kern, kind), rel=TOL, abs=0.0)


# --------------------------------------------------------------------------- #
# VC3: demo 4, an exact physical case
# --------------------------------------------------------------------------- #
D4_POS = dict(zip(LABELS, (0.0, 0.8, -0.5, 1.3, 0.35, -1.1)))
D4_TIMES = dict(zip(LABELS, (1.7, 1.2, 0.6, 1.5, 0.9, 1.3)))
D4_COMPS = ((0, 1, 1, 0, 1, 0), (1, 1, 0, 0, 0, 1))
PULSES = {"white": nz.PARAMS_WHITE, "exponential": nz.PARAMS_EXP}


def _d4_closed(p, comps, one_point_set=False) -> float:
    """``Σ_partitions K_R(A) K_R(B)``: at F = 0, ``K_R`` is the exact
    connected three-point function (``poisson_noise.py``)."""
    xs = np.array([D4_POS[lab] for lab in LABELS])
    ts = np.array([D4_TIMES[lab] for lab in LABELS])
    total = 0.0
    for A, B in PARTITIONS:
        ka = [FIRST if one_point_set else A][0]
        kb = [FIRST if one_point_set else B][0]
        total += (nz.K_R(tuple(comps[i] for i in A), xs[list(ka)],
                         ts[list(ka)], p)[0]
                  * nz.K_R(tuple(comps[i] for i in B), xs[list(kb)],
                           ts[list(kb)], p)[0])
    return total


@functools.lru_cache(maxsize=None)
def _d4_setup(pulse: str, r_contracted: bool):
    p = PULSES[pulse]
    system = dsys.make_system(p, cumulants=(3,), r_contracted=r_contracted)
    props = dsys.propagators_for(system, p, t_max=2.2)
    return system.expand(OBS, orders=[2], diag_C=False), props


D4_ROUTES = [
    ("exponential", True, "gauss_legendre", dict(n_gauss=4), TOL),
    ("exponential", True, "qmc_vectorized", dict(n_samples=2 ** 4), TOL),
    ("exponential", True, "qmc_scalar", dict(n_samples=2 ** 4), TOL),
    ("exponential", True, "nquad", {}, TOL),
    ("white", True, "gauss_legendre", dict(n_gauss=4), TOL),
    ("white", True, "nquad", {}, TOL),
    ("white", False, "gauss_legendre", dict(n_gauss=16), TOL),
    ("white", False, "qmc_vectorized", dict(n_samples=2 ** 12, seed=5), 1e-3),
]
#: The raw white route on nquad is 5 s per component tuple (two shared leg
#: times, adaptive), so it runs for one tuple only.
D4_ROUTES_ONE = [("white", False, "nquad", {}, 1e-10)]


@pytest.mark.parametrize("comps", D4_COMPS)
@pytest.mark.parametrize("pulse,rc,method,kw,tol", D4_ROUTES)
def test_VC3_demo4_six_point_function_matches_closed_form(pulse, rc, method,
                                                          kw, tol, comps):
    """White pulses: the raw vertex is ``equal_time`` (two shared times);
    either pulse: the ``already_R_contracted`` ``K_R`` (no time integral)."""
    p = PULSES[pulse]
    expansion, props = _d4_setup(pulse, rc)
    got = expansion.evaluate(
        props, positions=D4_POS, t_final=max(D4_TIMES.values()),
        external_times=D4_TIMES, component_pair=comps, orders=[2],
        method=method, **kw).total
    ref = _d4_closed(p, comps)
    assert got == pytest.approx(ref, rel=tol, abs=0.0)
    wrong = _d4_closed(p, comps, one_point_set=True)
    assert abs(wrong - ref) > MIN_GAP * abs(ref)


@pytest.mark.parametrize("pulse,rc,method,kw,tol", D4_ROUTES_ONE)
def test_VC3_demo4_six_point_function_on_nquad(pulse, rc, method, kw, tol):
    test_VC3_demo4_six_point_function_matches_closed_form(
        pulse, rc, method, kw, tol, D4_COMPS[0])


# --------------------------------------------------------------------------- #
# VC4: static couplings with equal_time / already_R_contracted copies
# --------------------------------------------------------------------------- #
G = 0.8


@functools.lru_cache(maxsize=None)
def _static_tensors(seed: int = 7):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(N, N, N)), rng.normal(size=(N, N, N))


def _measure(kind: str, idx, times) -> float:
    """The leg integrals of one vertex, R = exp(-G (t - t'))."""
    ts = np.array([times[LABELS[i]] for i in idx])
    if kind == "plain":
        return float(np.prod(-np.expm1(-G * (ts - T_MIN)) / G))
    if kind == "equal_time":
        return float(np.exp(-G * ts.sum())
                     * (np.exp(3 * G * ts.min()) - np.exp(3 * G * T_MIN))
                     / (3 * G))
    return 1.0


def _coupling(kappa, idx) -> float:
    return sum(kappa[tuple(COMPS[idx[b]] for b in beta)]
               for beta in PERMS) / 6.0


def _static_hand(kinds, times, old_rule=False) -> float:
    """Order-2 six-point function of two static vertices (species ``K``,
    ``L``; one species when ``kinds[1]`` is None).  ``old_rule``: every
    term integrated with the measure of the routing that keeps the legs of
    ``x1 x2 x3`` on the first copy, which is what the code before
    ``_split_by_instance_structure`` did."""
    kk, kl = _static_tensors()
    same = kinds[1] is None
    k_kind, l_kind = kinds[0], kinds[0] if same else kinds[1]
    kl = kk if same else kl
    parts = PARTITIONS if same else [
        (A, tuple(i for i in range(6) if i not in A))
        for A in itertools.combinations(range(6), 3)]
    total = 0.0
    for A, B in parts:
        mA, mB = (FIRST, SECOND) if old_rule else (A, B)
        total += (_coupling(kk, A) * _coupling(kl, B)
                  * _measure(k_kind, mA, times) * _measure(l_kind, mB, times))
    return total


@functools.lru_cache(maxsize=None)
def _static_setup(kinds):
    kk, kl = _static_tensors()
    vertices = [sw.NonLocalVertex("K", order=3, coupling=kk,
                                  **FLAGS[kinds[0]])]
    if kinds[1] is not None:
        vertices.append(sw.NonLocalVertex("L", order=3, coupling=kl,
                                          **FLAGS[kinds[1]]))
    sw.reset_uid_counter()
    system = sw.System(field=sw.FieldSpec("phi", N),
                       linear=sw.DiagonalA(gamma=[G] * N), noise=NOISE,
                       nonlocal_vertices=vertices, t_min=T_MIN)
    return (system.expand(OBS, orders=[2], diag_C=False),
            system.propagators(t_max=2.0, n_grid_t=8, progress=False),
            "K" if kinds[1] is None else "KL")


def _static_package(kinds, times):
    expansion, props, label = _static_setup(kinds)
    res = expansion.evaluate(
        props, positions=POS, t_final=max(times.values()),
        external_times=times, component_pair=COMPS, orders=[2],
        vertex_types={label}, method="gauss_legendre", n_gauss=12)
    return res.total, len(expansion.by_vertex_type(2)[label])


STATIC = [(("equal_time", None), 10), (("absorbed", "plain"), 20),
          (("equal_time", "plain"), 20), (("absorbed", None), 1)]


@pytest.mark.parametrize("kinds,n_terms", STATIC,
                         ids=["-".join(str(k) for k in s[0]) for s in STATIC])
def test_VC4_static_copies_take_the_measure_of_their_own_legs(kinds,
                                                              n_terms):
    got, n_dt = _static_package(kinds, TIMES)
    ref = _static_hand(kinds, TIMES)
    assert got == pytest.approx(ref, rel=TOL, abs=0.0)
    assert n_dt == n_terms
    old = _static_hand(kinds, TIMES, old_rule=True)
    if kinds == ("absorbed", None):
        # every leg is absorbed whichever copy it belongs to: a control
        assert old == pytest.approx(ref, rel=TOL, abs=0.0)
    else:
        assert abs(old - ref) > MIN_GAP * abs(ref)


@pytest.mark.parametrize("kinds", [s[0] for s in STATIC[:3]],
                         ids=["-".join(str(k) for k in s[0])
                              for s in STATIC[:3]])
def test_VC4_equal_external_times_hid_the_defect(kinds):
    """At equal external times every routing has the same measure."""
    ref = _static_hand(kinds, EQUAL_TIMES)
    assert _static_hand(kinds, EQUAL_TIMES, old_rule=True) == pytest.approx(
        ref, rel=1e-13, abs=0.0)
    got, _ = _static_package(kinds, EQUAL_TIMES)
    assert got == pytest.approx(ref, rel=TOL, abs=0.0)
