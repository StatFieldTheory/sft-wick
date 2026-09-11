"""Callable couplings on the adaptive-quadrature integrator (NQ1-NQ5).

``DiagramIntegrand.integrate_moment_nquad`` refused a spacetime-dependent
(callable) coupling -- "method='nquad' does not support spacetime-dependent
(callable) couplings" -- and before 2026-04 it silently multiplied the zeros
placeholder and returned 0.  It now materialises the coupling at every point
the quadrature visits, through ``dynamic_coupling_array``, exactly as the
scalar QMC loop does; ``make_scipy_integrand`` does the same.

* **NQ1** nquad equals Gauss-Legendre on the same diagrams: a plain, an
  ``equal_time`` and an ``already_R_contracted`` kappa3, both callable
  contracts.
* **NQ2** with a matrix-valued R, which the batched integrators refuse,
  nquad matches a numpy hand contraction.
* **NQ3** demo 4's level A: the three-point function at F = 0 for every
  component triple, raw and R-contracted, at equal and at unequal external
  times, against Campbell's closed form.
* **NQ4** demo 4's level B channels against the exact moment hierarchy.
* **NQ5** ``make_scipy_integrand`` drives scipy's own nquad to the same
  value.

Two more routes reach nquad with a callable coupling elsewhere:
``tests/test_callable_vertex_copies.py`` (two copies of one vertex, against
a hand contraction) and ``tests/test_local_callable_coupling.py`` (a
callable local coupling, against an exact moment hierarchy).
"""
from __future__ import annotations

import functools
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

import sft_wick as sw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "demo4"))
import poisson_noise as nz  # noqa: E402
import poisson_system as dsys  # noqa: E402
from poisson_reference import Hierarchy  # noqa: E402

N = 2
GAMMA = 0.6
T_MIN = 0.25
LABELS = ("x1", "x2", "x3")
OBS = ("phi_a(x1)", "phi_b(x2)", "phi_c(x3)")
POS = dict(zip(LABELS, (0.4, -0.7, 1.05)))
TIMES = dict(zip(LABELS, (1.6, 1.15, 0.85)))
COMPS = (0, 1, 1)
PERMS = tuple(itertools.permutations(range(3)))
NOISE = sw.GaussianNoise(kappa2=sw.SeparableTranslation(
    temporal=sw.ExponentialTemporal(lam=0.3, sigma_t=0.5),
    spatial=sw.GaussianSpatial(sigma_x=1.0)))
FLAGS = {"plain": {}, "equal_time": dict(equal_time=True),
         "absorbed": dict(already_R_contracted=True)}


class Kernel:
    """``κ_pqr(x_l, t_l) = Σ_k A_k[p,q,r] Π_l (a_kl + b_kl x_l)(1 + c_kl t_l)``."""

    def __init__(self, seed: int = 4):
        rng = np.random.default_rng(seed)
        self.terms = [(rng.normal(size=(N, N, N)),
                       rng.uniform(0.5, 1.5, size=3),
                       0.8 * rng.normal(size=3),
                       0.7 * rng.normal(size=3)) for _ in range(2)]

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


KERNEL = Kernel()


def _system(kind: str, batched: bool, *, matrix_R=False, coupling=None):
    sw.reset_uid_counter()
    linear = (sw.ExplicitR(R_time=_EyeR(), iso_R=False) if matrix_R
              else sw.DiagonalA(gamma=[GAMMA] * N))
    fn = coupling if coupling is not None else (
        KERNEL.batch if batched else KERNEL.single)
    return sw.System(
        field=sw.FieldSpec("phi", n_components=N), linear=linear,
        noise=NOISE, t_min=T_MIN,
        nonlocal_vertices=[sw.NonLocalVertex(
            "K", order=3, coupling=fn, coupling_vectorized=batched,
            **FLAGS[kind])])


from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class _EyeR:
    """``R_ab(t1, t2) = δ_ab Θ(t1 - t2)``: a matrix-valued R."""

    def __call__(self, t1, t2):
        return np.eye(N) if t1 > t2 else np.zeros((N, N))


@functools.lru_cache(maxsize=None)
def _evaluate(kind: str, batched: bool, method: str, n_gauss: int = 14,
              matrix_R: bool = False, constant: bool = False) -> float:
    coupling = _Constant() if constant else None
    system = _system(kind, batched, matrix_R=matrix_R, coupling=coupling)
    expansion = system.expand(OBS, orders=[1])
    props = system.propagators(t_max=2.0, n_grid_t=8, progress=False)
    return expansion.evaluate(
        props, positions=POS, t_final=max(TIMES.values()),
        external_times=TIMES, component_pair=COMPS, orders=[1],
        method=method, n_gauss=n_gauss).total


CONST_TENSOR = np.random.default_rng(21).normal(size=(N, N, N))


@dataclass(frozen=True)
class _Constant:
    """A callable ignoring its points: the per-sample contract."""

    def __call__(self, n_list, t_list):  # noqa: ARG002
        return CONST_TENSOR


# --------------------------------------------------------------------------- #
# NQ1: nquad = Gauss-Legendre
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batched", [False, True],
                         ids=["per_sample", "batched"])
@pytest.mark.parametrize("kind", sorted(FLAGS))
def test_NQ1_nquad_matches_gauss_legendre(kind, batched):
    """Three leg times for a plain vertex, one for ``equal_time`` and none
    for ``already_R_contracted``; the integrand is smooth, so the two rules
    agree far beyond their own accuracies."""
    got = _evaluate(kind, batched, "nquad")
    ref = _evaluate(kind, batched, "gauss_legendre")
    assert got == pytest.approx(ref, rel=1e-10, abs=0.0)


# --------------------------------------------------------------------------- #
# NQ2: matrix-valued R, which only the scalar loops take
# --------------------------------------------------------------------------- #
def _hand_constant_matrix_R() -> float:
    """``(1/6) Σ_β A[c_β(0) c_β(1) c_β(2)] Π_l (t_β(l) - t_min)``: the
    order-1 three-point function of a constant kappa3 with ``R = Θ``.  The
    prefactor is ``(-1) (-i)^3 (i/6) = 1/6``."""
    total = 0.0
    for beta in PERMS:
        comps = tuple(COMPS[beta[leg]] for leg in range(3))
        widths = np.prod([TIMES[LABELS[beta[leg]]] - T_MIN
                          for leg in range(3)])
        total += CONST_TENSOR[comps] * widths
    return total / 6.0


def test_NQ2_matrix_R_with_a_callable_coupling():
    got = _evaluate("plain", False, "nquad", matrix_R=True, constant=True)
    assert got == pytest.approx(_hand_constant_matrix_R(), rel=1e-10,
                                abs=0.0)


# --------------------------------------------------------------------------- #
# NQ3 / NQ4: demo 4
# --------------------------------------------------------------------------- #
T_D4 = 1.7
POS_D4 = {"x": 0.0, "y": 0.8, "z": -0.5}
UNEQUAL_D4 = {"x": 1.7, "y": 1.2, "z": 0.6}
TRIPLES = tuple(itertools.product(range(2), repeat=3))
PULSES = {"white": nz.PARAMS_WHITE, "exponential": nz.PARAMS_EXP}


@functools.lru_cache(maxsize=None)
def _d4_three_point(pulse: str, r_contracted: bool):
    p = PULSES[pulse]
    system = dsys.make_system(p, cumulants=(3,), r_contracted=r_contracted)
    props = dsys.propagators_for(system, p, t_max=T_D4 + 0.5)
    return system.expand(("phi_a(x)", "phi_b(y)", "phi_c(z)"), orders=[1],
                         diag_C=False), props


@pytest.mark.parametrize("comps", TRIPLES)
@pytest.mark.parametrize("pulse,rc", [("white", True), ("white", False),
                                      ("exponential", True)])
def test_NQ3_demo4_level_a_three_point(pulse, rc, comps):
    """Campbell's closed form ``K_R`` is the exact connected three-point
    function at F = 0 (``examples/demo4``); the white raw route is the
    ``equal_time`` vertex, whose leg time nquad integrates."""
    p = PULSES[pulse]
    expansion, props = _d4_three_point(pulse, rc)
    got = expansion.evaluate(props, positions=POS_D4, t_final=T_D4,
                             component_pair=comps, orders=[1],
                             method="nquad").total
    xs = np.array([POS_D4[lab] for lab in ("x", "y", "z")])
    ref = nz.K_R(comps, xs, np.full(3, T_D4), p)[0]
    assert got == pytest.approx(ref, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("pulse", sorted(PULSES))
def test_NQ3_demo4_level_a_unequal_times(pulse):
    p = PULSES[pulse]
    expansion, props = _d4_three_point(pulse, True)
    xs = np.array([POS_D4[lab] for lab in ("x", "y", "z")])
    ts = np.array([UNEQUAL_D4[lab] for lab in ("x", "y", "z")])
    for comps in [(0, 1, 1), (1, 0, 1)]:
        got = expansion.evaluate(
            props, positions=POS_D4, t_final=max(UNEQUAL_D4.values()),
            external_times=UNEQUAL_D4, component_pair=comps, orders=[1],
            method="nquad").total
        assert got == pytest.approx(nz.K_R(comps, xs, ts, p)[0], rel=1e-12,
                                    abs=0.0)


@functools.lru_cache(maxsize=None)
def _d4_level_b(pulse: str):
    p = PULSES[pulse]
    system = dsys.make_system(p, f_amplitude=1.0, cumulants=(3, 4),
                              r_contracted=(pulse == "exponential"))
    props = dsys.propagators_for(system, p, t_max=T_D4 + 0.5)
    return system, props, Hierarchy(p, [POS_D4["x"], POS_D4["y"]])


@pytest.mark.parametrize("comps", [(0, 1), (1, 1)])
@pytest.mark.parametrize("pulse", sorted(PULSES))
def test_NQ4_demo4_level_b_fk3(pulse, comps):
    """The order-2 F kappa3 channel of ``<phi_a(x) phi_b(y)>`` against the
    coefficient of ``eps mu`` of the moment hierarchy."""
    system, props, H = _d4_level_b(pulse)
    ref = H.moments([(comps[0], 0), (comps[1], 1)], [(1, 1)], T_D4)[(1, 1)]
    got = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2],
                        diag_C=False).evaluate(
        props, positions=POS_D4, t_final=T_D4, component_pair=comps,
        orders=[2], vertex_types={"FK3"}, method="nquad").total
    assert got == pytest.approx(ref, rel=1e-10, abs=0.0)


@pytest.mark.slow
def test_NQ4_demo4_level_b_ffk4_exponential():
    """The order-3 F F kappa4 channel: ``K_R`` is kinked where two vertex
    times cross, so adaptive quadrature needs ~40 s for 2.8e-9."""
    system, props, H = _d4_level_b("exponential")
    ref = H.moments([(0, 0), (1, 1)], [(2, 2)], T_D4)[(2, 2)]
    got = system.expand(("phi_a(x)", "phi_b(y)"), orders=[3],
                        diag_C=False).evaluate(
        props, positions=POS_D4, t_final=T_D4, component_pair=(0, 1),
        orders=[3], vertex_types={"FK4"}, method="nquad").total
    assert got == pytest.approx(ref, rel=1e-7, abs=0.0)


# --------------------------------------------------------------------------- #
# NQ5: make_scipy_integrand
# --------------------------------------------------------------------------- #
def test_NQ5_make_scipy_integrand_drives_scipy_nquad():
    """The L0 helper materialises the callable per call too, so scipy's own
    nquad on it reproduces ``integrate_moment_nquad``."""
    from scipy.integrate import nquad as scipy_nquad

    system = _system("equal_time", True)
    expansion = system.expand(OBS, orders=[1])
    props = system.propagators(t_max=2.0, n_grid_t=8, progress=False)
    cv = system.build_coupling_values()
    fixed = dict(zip(("a", "b", "c"), COMPS))
    total_pkg = total_scipy = 0.0
    for dt in expansion.dts_by_order[1]:
        ig = dt.build_integrand(cv, fixed)
        assert ig.dynamic_coupling is not None
        dirs = {}
        for group in ig.spatial.direction_groups:
            dvar = ig.spatial.direction_map[next(iter(group))]
            external = sorted(p for p in group if p in POS)
            dirs[dvar] = POS[external[0]] if external else 0.0
        ext = {lab: TIMES[lab] for lab in ig.spatial.external_points}
        f = ig.make_scipy_integrand(ext, dirs, props.cache)
        val, _ = scipy_nquad(f, ig.integration_bounds(ext, t_min=T_MIN),
                             opts={"epsabs": 1e-12, "epsrel": 1e-10})
        total_scipy += val
        total_pkg += ig.integrate_moment_nquad(
            lambda_f=max(TIMES.values()), cache=props.cache, t_min=T_MIN,
            positions=POS, external_times=TIMES)[0]
    assert total_scipy == pytest.approx(total_pkg, rel=1e-8, abs=0.0)
    assert abs(total_pkg) > 1e-6
