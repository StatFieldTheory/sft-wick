"""Leg order of a callable non-local coupling (LO0-LO6).

A non-local vertex ``K_{i0 i1 i2}(y0, y1, y2) psi_i0(y0) psi_i1(y1)
psi_i2(y2)`` enters a diagram through a coupling sum with one term per
assignment of its legs to the fields they contract with.  Each term,
``K_{σ(abc)}(σ(y))``, keeps every component index on its own leg.  Up to
0.4.2 a callable coupling was evaluated once, at the first term's leg order,
and every term read that tensor with its own permuted indices, so sft-wick
computed ``Σ_σ κ_{σ(abc)}(y)`` where ``Σ_σ κ_{σ(abc)}(σ(y))`` is required.
The two agree when the kernel is symmetric under a permutation of its leg
points at fixed component indices, and differ in general otherwise.  A
cumulant is symmetric only when (index, point) pairs are permuted together,
so a multi-component cumulant whose value depends on the leg points was
mis-evaluated.

Every expected value below is a numpy hand contraction written for this
file; no sft-wick contraction code produces it.  The kernels are sums of
separable terms ``A[p,q,r] Π_l (a_l + b_l·n_l)(1 + c_l t_l)``:

* ``generic``: no symmetry;
* ``pair-symmetric``: invariant when (index, point) pairs are permuted
  together, the symmetry of a cumulant, and not index-symmetric at fixed
  points;
* ``idx-symmetric``: index-symmetric at fixed points, with an asymmetric
  point dependence;
* ``pts-symmetric``: point-symmetric at fixed indices, with an asymmetric
  index tensor (the old evaluation was right for it, as for a constant
  tensor);
* ``fully-symmetric``: symmetric in indices and in points separately.

The last two are controls: the old and the new evaluation agree on them, so
they check the hand normalisation independently of the fix.  For the first
three every test also asserts that the correct value differs from the old
one ("indices permuted, points fixed"), so none of these tests can pass
against the code before the fix.

* **LO0** the kernels have the stated symmetries.
* **LO1** order-1 ``<phi_a(x1) phi_b(x2) phi_c(z)>``, all 27 component
  triples, on ``gauss_legendre``, ``qmc_vectorized`` and ``qmc_scalar`` with
  both callable contracts, on a matrix-valued R, and on ``equal_time`` and
  ``already_R_contracted`` vertices.
* **LO2** the same with gamma > 0 and ``z`` at an earlier time, so leg
  times move with the legs as well as leg directions.
* **LO3** the callable is called at every leg order the contraction needs.
* **LO4** the order-2 FK channel of ``<phi_a(x) phi_b(z)>``, with a plain
  and an ``already_R_contracted`` K.
* **LO5** a single-component field (no component indices) and the L0
  ``compute_moment`` + ``integrate_diagrams`` route.
* **LO6** a static ndarray coupling (control; it never used the callable
  path).

The QMC routes use a time-independent kernel.  With R = Θ the integrand is
then constant over the causal region, every sample returns the exact value,
and a 1e-12 comparison is meaningful for QMC as well.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
import pytest
from numpy.polynomial import Polynomial

import sft_wick as sw
from sft_wick.evaluate import integrate_diagrams
from sft_wick.perturbation import _collect_symbol_spatial_args

N = 3
PERMS = tuple(permutations(range(3)))
TRIPLES = tuple(product(range(N), repeat=3))
TOL = 1e-12
KERNELS = ("generic", "pair-symmetric", "idx-symmetric", "pts-symmetric",
           "fully-symmetric")
# Kernels that are not point-symmetric at fixed indices: the old evaluation
# was wrong for these, and every test checks that it was.
POINT_ASYMMETRIC = frozenset(KERNELS[:3])
# The correct and the old value must differ by more than this (relative to
# the largest correct value) for a POINT_ASYMMETRIC kernel, or the test is
# vacuous.
MIN_GAP = 1e-2


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


N_A = _unit([0.3, -0.2, 0.93])
N_B = _unit([0.8, 0.5, -0.33])
OBS = ("phi_a(x1)", "phi_b(x2)", "phi_c(z)")
POS = {"x1": N_A, "x2": N_A, "z": N_B}      # x1 and x2: two labels, one point
POS_FK = {"x": N_A, "z": N_B}

# No C propagator enters these diagrams; the noise only fixes the spatial
# structure (unit-vector directions on the sphere).
NOISE = sw.GaussianNoise(kappa2=sw.SeparableRotation(
    temporal=sw.ExponentialTemporal(lam=1.0, sigma_t=0.5),
    angular=sw.LegendreAngular(coeffs=[1.0]),
))


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
class SepKernel:
    """``κ_pqr(u0,u1,u2) = Σ_k A_k[p,q,r] Π_l (a_kl + b_kl·n_l)(1 + c_kl t_l)``.

    :meth:`single` is the per-sample callable contract, :meth:`batch` the
    batched one (``coupling_vectorized=True``).
    """

    def __init__(self, terms):
        self.terms = [tuple(np.asarray(x, dtype=float) for x in term)
                      for term in terms]

    def single(self, n_list, t_list) -> np.ndarray:
        """``n_list`` (3, 3), ``t_list`` (3,) -> (N, N, N)."""
        n = np.asarray(n_list, dtype=float)
        t = np.asarray(t_list, dtype=float)
        out = np.zeros((N, N, N))
        for A, a, b, c in self.terms:
            f = 1.0
            for leg in range(3):
                f *= (a[leg] + n[leg] @ b[leg]) * (1.0 + c[leg] * t[leg])
            out += f * A
        return out

    def batch(self, n_arr, t_arr) -> np.ndarray:
        """``n_arr`` (3, S, 3), ``t_arr`` (3, S) -> (S, N, N, N)."""
        n = np.asarray(n_arr, dtype=float)
        t = np.asarray(t_arr, dtype=float)
        out = np.zeros((t.shape[1], N, N, N))
        for A, a, b, c in self.terms:
            f = np.ones(t.shape[1])
            for leg in range(3):
                f = f * (a[leg] + n[leg] @ b[leg]) * (1.0 + c[leg] * t[leg])
            out += f[:, None, None, None] * A[None]
        return out


def _pair_symmetrize(kern: SepKernel) -> SepKernel:
    """``(1/6) Σ_π κ_{i_π0 i_π1 i_π2}(u_π0, u_π1, u_π2)``: a cumulant's symmetry."""
    terms = []
    for A, a, b, c in kern.terms:
        for pi in PERMS:
            inv = np.argsort(pi)                       # inv[l] = π⁻¹(l)
            A_pi = np.empty_like(A)
            for i in np.ndindex(*A.shape):
                A_pi[i] = A[i[pi[0]], i[pi[1]], i[pi[2]]]
            terms.append((A_pi / 6.0, a[inv], b[inv], c[inv]))
    return SepKernel(terms)


def _make_kernels(time_dependent: bool, seed: int = 20260911) -> dict:
    rng = np.random.default_rng(seed)
    terms = []
    for _ in range(2):
        A = rng.normal(size=(N, N, N))
        a = rng.uniform(0.5, 1.5, size=3)
        b = 0.8 * rng.normal(size=(3, 3))
        c = 0.7 * rng.normal(size=3) if time_dependent else np.zeros(3)
        terms.append((A, a, b, c))
    generic = SepKernel(terms)
    A0, a0, b0, c0 = generic.terms[0]
    A_sym = sum(np.transpose(A0, p) for p in PERMS) / 6.0
    same_legs = (np.full(3, a0[0]), np.tile(b0[0], (3, 1)), np.full(3, c0[0]))
    return {
        "generic": generic,
        "pair-symmetric": _pair_symmetrize(generic),
        "idx-symmetric": SepKernel([(A_sym, a0, b0, c0)]),
        "pts-symmetric": SepKernel([(A0,) + same_legs]),
        "fully-symmetric": SepKernel([(A_sym,) + same_legs]),
    }


def _asymmetries(kern: SepKernel, seed: int = 5) -> tuple[float, float, float]:
    """Relative asymmetry at three distinct points under a permutation of
    (indices at fixed points, points at fixed indices, both together)."""
    rng = np.random.default_rng(seed)
    n = rng.normal(size=(3, 3))
    n /= np.linalg.norm(n, axis=1)[:, None]
    t = rng.uniform(0.0, 1.0, size=3)
    T = kern.single(n, t)
    idx = pts = pair = 0.0
    for s in PERMS:
        Ts = kern.single(n[list(s)], t[list(s)])
        for i in np.ndindex(*T.shape):
            i_s = tuple(i[k] for k in s)
            idx = max(idx, abs(T[i_s] - T[i]))
            pts = max(pts, abs(Ts[i] - T[i]))
            pair = max(pair, abs(Ts[i_s] - T[i]))
    scale = np.max(np.abs(T))
    return idx / scale, pts / scale, pair / scale


# --------------------------------------------------------------------------- #
# Hand contractions (numpy only)
# --------------------------------------------------------------------------- #
def _moments(gamma: float, t_e: float) -> tuple[float, float]:
    """``∫_0^t_e R dt`` and ``∫_0^t_e t R dt`` for ``R = exp(-gamma (t_e - t))``."""
    if gamma == 0.0:
        return t_e, 0.5 * t_e ** 2
    e = np.exp(-gamma * t_e)
    return ((1 - e) / gamma,
            t_e * (1 - e) / gamma - (1 - e * (1 + gamma * t_e)) / gamma ** 2)


def _hand(kern: SepKernel, ext, mode: str, measure: str = "legs",
          gamma: float = 0.0) -> float:
    """Order-1 ``<phi phi phi>``; ``ext`` = three (component, direction, time).

    Wick pairs leg ``l`` with external ``β(l)`` for every bijection ``β``;
    ``<phi_c(e) psi_p(u)> = -i δ_cp R(t_e, t) δ(n_e, n)``, and the overall
    factor is ``(-1)(-i)^3 (i/6) = 1/6`` (order-1 sign, three R, the
    ``NonLocalVertex`` factor ``-i^3/3!``)::

        (1/6) Σ_β ∫ Π_l dt_l R(t_β(l), t_l) κ_{c_β(0) c_β(1) c_β(2)}(u_0, u_1, u_2)

    Mode ``"ok"`` gives leg ``l`` the component AND the point of ``β(l)``.
    Mode ``"idx"`` (the value returned up to 0.4.2) permutes the components
    only; the points stay those of ``(x1, x2, z)``.

    ``measure``: ``"legs"``, each leg time integrated against R;
    ``"equal"``, one shared leg time in ``[0, t_e]`` (``equal_time``,
    gamma = 0); ``"absorbed"``, no leg integral, each leg at its partner's
    time (``already_R_contracted``).
    """
    total = 0.0
    for beta in PERMS:
        points_of = beta if mode == "ok" else (0, 1, 2)
        comps = tuple(ext[beta[leg]][0] for leg in range(3))
        pts = [ext[points_of[leg]] for leg in range(3)]
        for A, a, b, c in kern.terms:
            ang = np.prod([a[leg] + pts[leg][1] @ b[leg] for leg in range(3)])
            if measure == "legs":
                tim = 1.0
                for leg in range(3):
                    m0, m1 = _moments(gamma, pts[leg][2])
                    tim *= m0 + c[leg] * m1
            elif measure == "equal":
                poly = Polynomial([1.0])
                for leg in range(3):
                    poly = poly * Polynomial([1.0, c[leg]])
                t_top = min(p[2] for p in pts)
                tim = poly.integ()(t_top) - poly.integ()(0.0)
            else:
                tim = np.prod([1.0 + c[leg] * pts[leg][2] for leg in range(3)])
            total += A[comps] * ang * tim
    return total / 6.0


def _hand_fk(kern: SepKernel, F: np.ndarray, comps, mode: str,
             fixed_dirs=None, time_factor: float = 1.0 / 3.0) -> float:
    """Order-2 FK channel of ``<phi_a(x) phi_b(z)>``, both at time 1, R = Θ.

    Local F (``psi_i phi_j phi_k`` at ``w``, ``F_MSR = -i F``) and non-local K
    (``K = (i/6) κ``, κ time-independent).  F's ψ contracts with one external
    ``e``, which puts ``w`` at ``n_e``; K's legs contract bijectively with the
    other external ``e'`` and the two φ legs of F, targets
    ``[(c_e', n_e'), (j, n_e), (k, n_e)]``.  Prefactor: order-2 sign / 2! x
    multinomial 2 = 1, ``(-i)^4 = 1``, ``(-i)(i/6) = 1/6``.  Time integral
    (``time_factor``): ``∫_0^1 dt_w (∫_0^t_w dt)^2 ∫_0^1 dt = 1/3`` for a plain
    K; ``∫_0^1 dt_w = 1`` for an ``already_R_contracted`` K, whose legs sit
    at their partners' times.

    Mode ``"idx"`` evaluates κ at ``fixed_dirs[e]`` (the first coupling-sum
    term's leg order) for every term: the value returned up to 0.4.2.
    """
    ext = {"x": (comps[0], N_A), "z": (comps[1], N_B)}
    zeros = np.zeros(3)
    total = 0.0
    for e, e_other in (("x", "z"), ("z", "x")):
        (c_e, n_e), (c_other, n_other) = ext[e], ext[e_other]
        for j in range(N):
            for k in range(N):
                targets = [(c_other, n_other), (j, n_e), (k, n_e)]
                for beta in PERMS:
                    cidx = tuple(targets[beta[leg]][0] for leg in range(3))
                    dirs = (np.array([targets[beta[leg]][1] for leg in range(3)])
                            if mode == "ok" else fixed_dirs[e])
                    total += F[c_e, j, k] * kern.single(dirs, zeros)[cidx]
    return total * time_factor / 6.0


def _assert_matches(got: dict, ok: dict, old: dict, kernel: str,
                    headline) -> None:
    """sft-wick equals the hand value on every cell; the headline cell to
    ``TOL`` relative.  For a ``POINT_ASYMMETRIC`` kernel the old value must
    differ from the correct one (non-vacuity); for a control it must not."""
    scale = max(abs(v) for v in ok.values())
    worst = max(abs(got[k] - ok[k]) for k in ok) / scale
    assert worst < TOL, (
        f"[{kernel}] max |sft-wick - hand| / max |hand| = {worst:.2e}")
    g, h = got[headline], ok[headline]
    assert abs(g - h) <= TOL * abs(h), (
        f"[{kernel}] {headline}: sft-wick {g!r}, hand {h!r}, "
        f"rel {abs(g - h) / abs(h):.2e}")
    gap = max(abs(ok[k] - old[k]) for k in ok) / scale
    if kernel in POINT_ASYMMETRIC:
        assert gap > MIN_GAP, (
            f"[{kernel}] vacuous: the old value is within {gap:.1e} of the "
            f"correct one")
    else:
        assert gap < TOL, (
            f"[{kernel}] control: the old value differs from the correct one "
            f"by {gap:.1e}")


# --------------------------------------------------------------------------- #
# sft-wick side
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _EyeR:
    """``R_ab(t1, t2) = δ_ab Θ(t1 - t2)``, matrix-valued (``iso_R=False``).

    A frozen dataclass keeps the ``repr`` stable, so the expansion cache key
    does not change between processes."""

    def __call__(self, t1, t2):
        return np.eye(N) if t1 > t2 else np.zeros((N, N))


def _system(coupling, *, vectorized: bool, gamma: float = 0.0,
            equal_time: bool = False, absorbed: bool = False,
            matrix_R: bool = False) -> sw.System:
    sw.reset_uid_counter()
    linear = (sw.ExplicitR(R_time=_EyeR(), iso_R=False) if matrix_R
              else sw.DiagonalA(gamma=[gamma] * N))
    return sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=linear,
        noise=NOISE,
        nonlocal_vertices=[sw.NonLocalVertex(
            "K", order=3, coupling=coupling, coupling_vectorized=vectorized,
            equal_time=equal_time, already_R_contracted=absorbed)],
    )


# route: (method, batched contract?, time-dependent kernel?, hand measure,
#         _system flags)
ROUTES = {
    "gauss_legendre-batched": ("gauss_legendre", True, True, "legs", {}),
    "gauss_legendre-per_sample": ("gauss_legendre", False, True, "legs", {}),
    "qmc_vectorized-batched": ("qmc_vectorized", True, False, "legs", {}),
    "qmc_vectorized-per_sample": ("qmc_vectorized", False, False, "legs", {}),
    "qmc_scalar-per_sample": ("qmc_scalar", False, False, "legs", {}),
    "qmc_scalar-batched": ("qmc_scalar", True, False, "legs", {}),
    "qmc_scalar-matrix_R": ("qmc_scalar", False, False, "legs",
                            {"matrix_R": True}),
    "equal_time": ("gauss_legendre", True, True, "equal",
                   {"equal_time": True}),
    "already_R_contracted-gauss_legendre": ("gauss_legendre", True, True,
                                            "absorbed", {"absorbed": True}),
    "already_R_contracted-qmc_scalar": ("qmc_scalar", False, True,
                                        "absorbed", {"absorbed": True}),
}


# --------------------------------------------------------------------------- #
# LO0: the kernels are what the tests say they are
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("time_dependent", [True, False])
def test_LO0_kernels_have_the_stated_symmetries(time_dependent):
    # (index-symmetric, point-symmetric, pair-symmetric)
    stated = {
        "generic": (False, False, False),
        "pair-symmetric": (False, False, True),
        "idx-symmetric": (True, False, False),
        "pts-symmetric": (False, True, False),
        "fully-symmetric": (True, True, True),
    }
    kernels = _make_kernels(time_dependent)
    for name, flags in stated.items():
        for what, asym, symmetric in zip(("idx", "pts", "pair"),
                                         _asymmetries(kernels[name]), flags):
            if symmetric:
                assert asym < 1e-14, f"{name}: {what} asymmetry {asym:.1e}"
            else:
                assert asym > MIN_GAP, f"{name}: {what} asymmetry {asym:.1e}"


# --------------------------------------------------------------------------- #
# LO1: order-1 three-point function on every route
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_LO1_order1_matches_hand_on_every_route(route, kernel):
    method, batched, time_dependent, measure, flags = ROUTES[route]
    kern = _make_kernels(time_dependent)[kernel]
    system = _system(kern.batch if batched else kern.single,
                     vectorized=batched, **flags)
    expansion = system.expand(OBS, orders=[1])
    props = system.propagators(t_max=2.0, n_grid_t=8)

    got, ok, old = {}, {}, {}
    for comps in TRIPLES:
        ext = [(comps[0], N_A, 1.0), (comps[1], N_A, 1.0), (comps[2], N_B, 1.0)]
        got[comps] = expansion.evaluate(
            props, positions=POS, t_final=1.0, component_pair=comps,
            method=method, n_gauss=4, n_samples=2 ** 5, seed=7,
        ).total
        ok[comps] = _hand(kern, ext, "ok", measure)
        old[comps] = _hand(kern, ext, "idx", measure)
    _assert_matches(got, ok, old, kernel, headline=(0, 1, 2))


# --------------------------------------------------------------------------- #
# LO2: leg times travel with the legs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kernel", KERNELS)
def test_LO2_leg_times_move_with_the_legs(kernel):
    """gamma = 0.8 and ``z`` at t = 0.6 while ``x1``, ``x2`` sit at t = 1: each
    leg's time integral now depends on which external the leg pairs with."""
    gamma, t_a, t_b = 0.8, 1.0, 0.6
    kern = _make_kernels(True)[kernel]
    system = _system(kern.batch, vectorized=True, gamma=gamma)
    expansion = system.expand(OBS, orders=[1])
    props = system.propagators(t_max=2.0, n_grid_t=8)

    got, ok, old = {}, {}, {}
    for comps in TRIPLES:
        ext = [(comps[0], N_A, t_a), (comps[1], N_A, t_a), (comps[2], N_B, t_b)]
        got[comps] = expansion.evaluate(
            props, positions=POS, t_final=t_a, external_times={"z": t_b},
            component_pair=comps, method="gauss_legendre", n_gauss=12,
        ).total
        ok[comps] = _hand(kern, ext, "ok", gamma=gamma)
        old[comps] = _hand(kern, ext, "idx", gamma=gamma)
    _assert_matches(got, ok, old, kernel, headline=(0, 1, 2))


# --------------------------------------------------------------------------- #
# LO3: which leg orders reach the callable
# --------------------------------------------------------------------------- #
def test_LO3_callable_is_called_at_every_leg_order():
    """Legs of external points ``x1``, ``x2`` (direction A) and ``z`` (B): the
    contraction needs the direction patterns AAB, ABA and BAA.  Up to 0.4.2
    the callable only ever received AAB."""
    kern = _make_kernels(True)["generic"]
    seen = set()

    def spy(n_arr, t_arr):
        n = np.asarray(n_arr)
        for s in range(n.shape[1]):
            seen.add("".join("A" if np.allclose(n[leg, s], N_A) else "B"
                             for leg in range(3)))
        return kern.batch(n_arr, t_arr)

    system = _system(spy, vectorized=True)
    expansion = system.expand(OBS, orders=[1])
    props = system.propagators(t_max=2.0, n_grid_t=8)
    expansion.evaluate(props, positions=POS, t_final=1.0,
                       component_pair=(0, 1, 2), method="gauss_legendre",
                       n_gauss=4)
    assert seen == {"AAB", "ABA", "BAA"}


# --------------------------------------------------------------------------- #
# LO4: the order-2 FK channel of a two-point function
# --------------------------------------------------------------------------- #
def _fk_first_occurrence(dt) -> tuple[str, np.ndarray]:
    """(external that F's ψ contracts with, directions of K's legs in the
    first coupling-sum term): the one leg order the old code evaluated."""
    legs = _collect_symbol_spatial_args(dt.coupling_sum)["K"]
    w = next(v for v in dt.integration_vars if v not in legs)
    e = next(p.spatial_left for p in dt.propagators
             if p.kind == "R" and p.spatial_right == w)
    direction_map = dt.analyze_spatial().direction_map
    return e, np.array([POS_FK[direction_map[y][2:]] for y in legs])


@pytest.mark.parametrize("absorbed", [False, True],
                         ids=["plain", "already_R_contracted"])
@pytest.mark.parametrize("kernel", KERNELS)
def test_LO4_fk_channel_matches_hand(kernel, absorbed):
    kern = _make_kernels(False)[kernel]
    F = np.random.default_rng(99).normal(size=(N, N, N))
    sw.reset_uid_counter()
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(gamma=[0.0] * N),
        noise=NOISE,
        vertices=[sw.LocalVertex("F", coupling=F)],
        nonlocal_vertices=[sw.NonLocalVertex(
            "K", order=3, coupling=kern.batch, coupling_vectorized=True,
            already_R_contracted=absorbed)],
    )
    expansion = system.expand(("phi_a(x)", "phi_b(z)"), orders=[2])
    props = system.propagators(t_max=2.0, n_grid_t=8)
    fk = expansion.by_vertex_type(2)["FK"]
    fixed_dirs = dict(_fk_first_occurrence(dt) for dt in fk)
    assert sorted(fixed_dirs) == ["x", "z"], "one FK diagram per external"
    time_factor = 1.0 if absorbed else 1.0 / 3.0

    got, ok, old = {}, {}, {}
    for comps in product(range(N), repeat=2):
        got[comps] = expansion.evaluate(
            props, positions=POS_FK, t_final=1.0, component_pair=comps,
            orders=[2], vertex_types={"FK"}, method="gauss_legendre",
            n_gauss=8,
        ).total
        ok[comps] = _hand_fk(kern, F, comps, "ok", time_factor=time_factor)
        old[comps] = _hand_fk(kern, F, comps, "idx", fixed_dirs, time_factor)
    _assert_matches(got, ok, old, kernel, headline=(0, 1))


# --------------------------------------------------------------------------- #
# LO5: no component indices; the L0 API
# --------------------------------------------------------------------------- #
def test_LO5_single_component_field():
    """N = 1: the simplifier strips every component index, so the terms of
    the coupling sum differ only by leg order, and the kernel
    ``(1 + n0·v)(2 + n2·w)`` is not symmetric in its points."""
    v = np.array([0.4, -0.7, 0.2])
    w = np.array([-0.3, 0.5, 0.9])

    def k1(n_list, t_list):  # noqa: ARG001 -- time-independent
        n = np.asarray(n_list, dtype=float)
        return np.array((1.0 + n[0] @ v) * (2.0 + n[2] @ w)).reshape(1, 1, 1)

    sw.reset_uid_counter()
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=1),
        linear=sw.DiagonalA(gamma=[0.0]),
        noise=NOISE,
        nonlocal_vertices=[sw.NonLocalVertex("K", order=3, coupling=k1)],
    )
    expansion = system.expand(("phi(x1)", "phi(x2)", "phi(z)"), orders=[1])
    props = system.propagators(t_max=2.0, n_grid_t=8)
    got = expansion.evaluate(props, positions=POS, t_final=1.0,
                             component_pair=(0, 0, 0),
                             method="gauss_legendre", n_gauss=4).total
    dirs = (N_A, N_A, N_B)
    ok = sum(k1(np.array([dirs[b[leg]] for leg in range(3)]), None)[0, 0, 0]
             for b in PERMS) / 6.0
    old = k1(np.array(dirs), None)[0, 0, 0]
    assert abs(got - ok) <= TOL * abs(ok), f"sft-wick {got!r}, hand {ok!r}"
    assert abs(ok - old) > MIN_GAP * abs(ok), "vacuous"


def test_LO5_l0_compute_moment_route():
    """``compute_moment`` + ``integrate_diagrams`` with a raw coupling dict:
    the same ``build_integrand``, reached without the workflow layer."""
    kern = _make_kernels(False)["generic"]
    props = _system(kern.batch, vectorized=True).propagators(t_max=2.0,
                                                             n_grid_t=8)
    sw.reset_uid_counter()
    phi = sw.Field("phi", "physical", n_components=N)
    psi = sw.Field("psi", "response", n_components=N)
    obs = [phi("a", "x1"), phi("b", "x2"), phi("c", "z")]
    action = sw.Action(vertices=[
        sw.Vertex(fields=[psi, psi, psi], coupling="K", local=False)])
    dts = sw.compute_moment(obs, action, order=1, iso_R=True).diagram_terms(1)

    def k_msr(n_list, t_list):
        return (1j / 6.0) * kern.single(n_list, t_list)

    got, _ = integrate_diagrams(
        dts, {"K": k_msr}, lambda_f=1.0, cache=props.cache,
        method="gauss_legendre", fixed_indices={"a": 0, "b": 1, "c": 2},
        positions=POS,
    )
    ext = [(0, N_A, 1.0), (1, N_A, 1.0), (2, N_B, 1.0)]
    ok, old = _hand(kern, ext, "ok"), _hand(kern, ext, "idx")
    assert abs(got - ok) <= TOL * abs(ok), f"sft-wick {got!r}, hand {ok!r}"
    assert abs(ok - old) > MIN_GAP * abs(ok), "vacuous"


# --------------------------------------------------------------------------- #
# LO6: control
# --------------------------------------------------------------------------- #
def test_LO6_static_ndarray_coupling_matches_hand():
    """An asymmetric constant tensor goes through the static path, which
    reads each term's own indices from one array and was always right."""
    A = _make_kernels(True)["generic"].terms[0][0]
    const = SepKernel([(A, np.ones(3), np.zeros((3, 3)), np.zeros(3))])
    system = _system(A, vectorized=False)
    expansion = system.expand(OBS, orders=[1])
    props = system.propagators(t_max=2.0, n_grid_t=8)
    got, ok = {}, {}
    for comps in TRIPLES:
        ext = [(comps[0], N_A, 1.0), (comps[1], N_A, 1.0), (comps[2], N_B, 1.0)]
        got[comps] = expansion.evaluate(
            props, positions=POS, t_final=1.0, component_pair=comps,
            method="gauss_legendre", n_gauss=4,
        ).total
        ok[comps] = _hand(const, ext, "ok")
    scale = max(abs(v) for v in ok.values())
    worst = max(abs(got[k] - ok[k]) for k in ok) / scale
    assert worst < TOL, f"max |sft-wick - hand| / max |hand| = {worst:.2e}"
