"""Callable (spacetime-dependent) local couplings at L1 (LC1-LC8).

Up to 0.5.0 ``LocalVertex`` took only a tensor.  A callable coupling raised a
``TypeError`` from ``System.build_coupling_values``, although L0 has
evaluated callable local couplings since F6
(``tests/test_msr_numerics_regressions.py``) and the YAML reference
documented ``coupling_module`` for local vertices.
``LocalVertex(coupling=fn, rank=n)`` now lowers ``fn`` with the MSR factor
``-i`` applied to its output, under the two calling contracts of a
non-local coupling at one point: ``fn(n_list, t_list)`` with length-1
inputs, or ``coupling_vectorized=True`` with ``(1, n_samples)`` inputs.

The drift is ``F_abc(x, t) = F0_abc (1 + beta x) exp(-lambda t)`` at N = 2,
with no index symmetry, coloured noise plus white noise that mixes the
components, distinct observation points and ``t_min = 0.5``.  The reference
is the exact moment hierarchy of a Markov embedding in which a
deterministic state ``u = exp(-lambda t)`` makes the drift a polynomial
(``examples/reference/decaying_drift.py``, on
``examples/reference/ito_moments.py``); it shares no code with the package.
At order 2 the two F vertices of ``<phi_a phi_b>`` sit at different times,
so each copy must be evaluated at its own time, the case that was refused
up to 0.5.0.

* **LC1** the reference equals demo 5's hierarchy at lambda = 0.
* **LC2** the checked moments resolve the time and position dependence:
  freezing the drift at ``t_min`` or at ``t_final``, or dropping beta,
  moves each of them by more than 5 %.
* **LC3** construction: a callable needs ``rank``; ``rank`` is checked
  against a tensor; the MSR factor and the batched flag reach the coupling
  values; the argument shapes of both contracts.
* **LC4** a callable returning the constant tensor equals the tensor,
  diagram by diagram, on every integrator.
* **LC5** ``<phi_a>`` and ``<phi_a phi_b phi_c>`` at order 1 and
  ``<phi_a phi_b>`` at order 2 against the hierarchy, on every integrator
  and both contracts.
* **LC6** distinct rates (a diagonal matrix R) on the scalar QMC loop and
  nquad.
* **LC7** a position-dependent drift, beta = 0.45.
* **LC8** a YAML local vertex with ``coupling_module`` and ``rank``.
"""
from __future__ import annotations

import functools
import sys
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.workflow.config import build_system, load_workflow_config
from sft_wick.workflow.specs import ConstantImpulse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "reference"))
sys.path.insert(0, str(ROOT / "examples" / "demo5"))
import decaying_drift as dd  # noqa: E402
from white_reference import Hierarchy as Demo5Hierarchy  # noqa: E402

T = 1.6                                   # observed at t_min + T
POS = {"x": 0.3, "y": -0.6, "z": 0.9}
LABELS = ("x", "y", "z")
OBS = {1: ("phi_a(x)",), 2: ("phi_a(x)", "phi_b(y)"),
       3: ("phi_a(x)", "phi_b(y)", "phi_c(z)")}
PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1))

GL = ("gauss_legendre", dict(n_gauss=12))
QMC_V = ("qmc_vectorized", dict(n_samples=2 ** 12, seed=2))
QMC_S = ("qmc_scalar", dict(n_samples=2 ** 10, seed=2))
NQUAD = ("nquad", {})
TOL = {"gauss_legendre": 1e-12, "nquad": 1e-10, "qmc_vectorized": 1e-5,
       "qmc_scalar": 1e-4}


@dataclass(frozen=True)
class DecayingF:
    """``F0 (1 + beta x) exp(-decay t)``, per-sample contract:
    ``n_list``, ``t_list`` of shape ``(1,)`` -> ``(N, N, N)``."""

    f0: tuple
    decay: float
    beta: float = 0.0

    def __call__(self, n_list, t_list):
        x = float(np.asarray(n_list, dtype=float).reshape(-1)[0])
        t = float(np.asarray(t_list, dtype=float).reshape(-1)[0])
        return np.asarray(self.f0) * ((1.0 + self.beta * x)
                                      * np.exp(-self.decay * t))


@dataclass(frozen=True)
class DecayingFBatch:
    """The same drift, batched contract: ``(1, S)`` -> ``(S, N, N, N)``."""

    f0: tuple
    decay: float
    beta: float = 0.0

    def __call__(self, n_2d, t_2d):
        x = np.asarray(n_2d, dtype=float)[0]
        t = np.asarray(t_2d, dtype=float)[0]
        fac = (1.0 + self.beta * x) * np.exp(-self.decay * t)
        return fac[:, None, None, None] * np.asarray(self.f0)[None]


def _noise(p: dd.DecayParams) -> sw.GaussianNoise:
    return sw.GaussianNoise(
        kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=p.lam, sigma_t=p.sigma_t),
            spatial=sw.GaussianSpatial(sigma_x=p.sigma_x)),
        sigma2=ConstantImpulse(np.asarray(p.S, dtype=float)))


def _system(p: dd.DecayParams, coupling: str) -> sw.System:
    """``coupling``: ``"per_sample"``, ``"batched"`` or ``"tensor"`` (the
    constant ``F0``, a plain ndarray)."""
    if coupling == "tensor":
        vertex = sw.LocalVertex("F", coupling=p.F0)
    else:
        batched = coupling == "batched"
        fn = (DecayingFBatch if batched else DecayingF)(p.f0, p.decay, p.beta)
        vertex = sw.LocalVertex("F", coupling=fn, rank=3,
                                coupling_vectorized=batched)
    return sw.System(
        field=sw.FieldSpec("phi", p.n_components),
        linear=sw.DiagonalA(gamma=list(p.gamma)),
        vertices=[vertex], noise=_noise(p), t_min=p.t_min)


@functools.lru_cache(maxsize=None)
def _setup(p: dd.DecayParams, coupling: str):
    system = _system(p, coupling)
    props = system.propagators(
        t_max=p.t_min + T + 0.5, c_closed_form="auto",
        c_closed_form_only=True, c_closed_form_vectorized=True,
        diag_C=False, progress=False)
    return system, props


@functools.lru_cache(maxsize=None)
def _expansion(p: dd.DecayParams, coupling: str, n_points: int, order: int):
    system, _ = _setup(p, coupling)
    return system.expand(OBS[n_points], orders=[order], diag_C=False)


def _evaluate(p, coupling, n_points, order, comps, method, kw):
    _, props = _setup(p, coupling)
    return _expansion(p, coupling, n_points, order).evaluate(
        props, positions=POS, t_final=p.t_min + T, component_pair=comps,
        orders=[order], method=method, **kw)


@functools.lru_cache(maxsize=None)
def _reference(p: dd.DecayParams, n_points: int, order: int, comps) -> float:
    H = dd.DecayHierarchy(p, [POS[lab] for lab in LABELS[:n_points]])
    return H.moments([(c, i) for i, c in enumerate(comps)], [order],
                     T)[order]


def _rel(got: float, ref: float) -> float:
    return abs(got - ref) / abs(ref)


# (n_points, order, component tuple)
OBSERVABLES = ([(1, 1, (a,)) for a in range(2)]
               + [(2, 2, ab) for ab in PAIRS]
               + [(3, 1, (1, 0, 1)), (3, 1, (0, 1, 1))])


# --------------------------------------------------------------------------- #
# LC1-LC2: the reference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("legs,order", [(((0, 0), (1, 1)), 2),
                                        (((1, 0), (0, 1)), 2),
                                        (((1, 0), (1, 1)), 0)])
def test_LC1_reference_equals_demo5_at_zero_decay(legs, order):
    """With lambda = 0 the embedding is demo 5's (``u`` stays 1)."""
    p = replace(dd.PARAMS, decay=0.0)

    class Demo5Params:   # the attributes demo 5's hierarchy reads
        gamma = p.gamma[0]
        lam, sigma_t, sigma_x, S = p.lam, p.sigma_t, p.sigma_x, p.S
        n_components = p.n_components

    got = dd.DecayHierarchy(p, [POS["x"], POS["y"]]).moments(
        list(legs), [order], T)[order]
    ref = Demo5Hierarchy(Demo5Params, [POS["x"], POS["y"]], p.F0).moments(
        list(legs), [order], T)[order]
    assert got == pytest.approx(ref, rel=1e-13, abs=0.0)


@pytest.mark.parametrize("n_points,order,comps", OBSERVABLES)
def test_LC2_checked_moments_resolve_time_and_position(n_points, order,
                                                       comps):
    """A drift frozen at t_min or at t_final, or one without its position
    dependence, gives moments more than 5 % away: a callable evaluated at a
    wrong time or point cannot pass LC5 and LC7."""
    p = dd.PARAMS
    ref = _reference(p, n_points, order, comps)
    frozen = _reference(replace(p, decay=0.0), n_points, order, comps)
    for t_frozen in (p.t_min, p.t_min + T):
        # a constant drift c F0 scales the order-k coefficient by c**k
        c = np.exp(-p.decay * t_frozen)
        assert _rel(c ** order * frozen, ref) > 0.05, t_frozen
    tilted = replace(p, beta=0.45)
    assert _rel(_reference(tilted, n_points, order, comps), ref) > 0.05


# --------------------------------------------------------------------------- #
# LC3: construction and calling contracts
# --------------------------------------------------------------------------- #
def test_LC3_callable_needs_a_rank_and_a_tensor_rank_is_checked():
    fn = DecayingF(dd.F0_TENSOR, 0.7)
    with pytest.raises(ValueError, match="rank"):
        sw.LocalVertex("F", coupling=fn)
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="rank"):
            sw.LocalVertex("F", coupling=fn, rank=bad)
    with pytest.raises(ValueError, match="3 axes"):
        sw.LocalVertex("F", coupling=np.zeros((2, 2, 2)), rank=2)
    assert sw.LocalVertex("F", coupling=np.zeros((2, 2, 2)), rank=3).n_legs == 3
    assert sw.LocalVertex("F", coupling=np.zeros((2, 2, 2))).n_legs == 3
    assert sw.LocalVertex("F", coupling=fn, rank=3).n_legs == 3


@pytest.mark.parametrize("coupling", ["per_sample", "batched"])
def test_LC3_lowering_applies_minus_i_and_keeps_the_contract(coupling):
    p = dd.PARAMS
    system = _system(p, coupling)
    (vertex,) = system.build_action().vertices
    assert len(vertex.fields) == 3 and vertex.local
    cv = system.build_coupling_values()["F"]
    assert callable(cv)
    assert bool(getattr(cv, "vectorized", False)) == (coupling == "batched")
    n = np.array([[0.3, -0.6]]) if coupling == "batched" else np.array([0.3])
    t = np.array([[0.9, 1.7]]) if coupling == "batched" else np.array([0.9])
    bare = system.vertices[0].coupling(n, t)
    np.testing.assert_array_equal(cv(n, t), -1j * bare)


@pytest.mark.parametrize("method,coupling", [
    ("gauss_legendre", "per_sample"), ("gauss_legendre", "batched"),
    ("qmc_vectorized", "batched"), ("qmc_scalar", "per_sample"),
    ("nquad", "per_sample")])
def test_LC3_argument_shapes(method, coupling):
    """Per-sample: ``(1,)`` position and time; batched: ``(1, S)``."""
    p = dd.PARAMS
    inner = (DecayingFBatch if coupling == "batched" else DecayingF)(
        p.f0, p.decay)
    seen = set()

    def spy(n, t):
        seen.add((np.shape(n)[:1], np.shape(t)[:1], np.ndim(n), np.ndim(t)))
        return inner(n, t)

    system = sw.System(
        field=sw.FieldSpec("phi", 2), linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        vertices=[sw.LocalVertex("F", coupling=spy, rank=3,
                                 coupling_vectorized=coupling == "batched")],
        noise=_noise(p), t_min=p.t_min)
    _, props = _setup(p, "tensor")
    system.expand(OBS[1], orders=[1], diag_C=False).evaluate(
        props, positions=POS, t_final=p.t_min + T, component_pair=(0,),
        orders=[1], method=method, n_gauss=4, n_samples=2 ** 4, seed=1)
    ndim = 2 if coupling == "batched" else 1
    assert seen == {((1,), (1,), ndim, ndim)}


# --------------------------------------------------------------------------- #
# LC4: a constant callable is the tensor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("coupling", ["per_sample", "batched"])
@pytest.mark.parametrize("method,kw", [
    GL, QMC_V, ("qmc_scalar", dict(n_samples=2 ** 7, seed=2)), NQUAD])
@pytest.mark.parametrize("n_points,order,comps", [(1, 1, (1,)),
                                                  (2, 2, (0, 1))])
def test_LC4_constant_callable_equals_the_tensor(n_points, order, comps,
                                                 method, kw, coupling):
    if method == "nquad" and order == 2:
        pytest.skip("covered by the slow LC5 nquad case")
    p = replace(dd.PARAMS, decay=0.0)
    got = _evaluate(p, coupling, n_points, order, comps, method, kw)
    ref = _evaluate(p, "tensor", n_points, order, comps, method, kw)
    assert len(got.per_diagram) == len(ref.per_diagram) >= 1
    scale = max(abs(d["value"]) for d in ref.per_diagram)
    for g, r in zip(got.per_diagram, ref.per_diagram):
        assert abs(g["value"] - r["value"]) <= 1e-12 * scale, (g, r)


# --------------------------------------------------------------------------- #
# LC5-LC7: against the exact hierarchy
# --------------------------------------------------------------------------- #
def _lc5_cases():
    cases = []
    for n_points, order, comps in OBSERVABLES:
        methods = [GL, QMC_V]
        if n_points != 2:
            methods += [QMC_S, NQUAD]
        elif comps == (0, 1):
            methods += [QMC_S]
        for method, kw in methods:
            for coupling in ("per_sample", "batched"):
                if method == "qmc_scalar" and coupling == "batched" \
                        and n_points == 2:
                    continue
                cases.append((n_points, order, comps, method, kw, coupling))
    return cases


@pytest.mark.parametrize("n_points,order,comps,method,kw,coupling",
                         _lc5_cases())
def test_LC5_decaying_drift_matches_the_hierarchy(n_points, order, comps,
                                                  method, kw, coupling):
    p = dd.PARAMS
    got = _evaluate(p, coupling, n_points, order, comps, method, kw).total
    ref = _reference(p, n_points, order, comps)
    assert _rel(got, ref) < TOL[method], (got, ref)


@pytest.mark.slow
@pytest.mark.parametrize("comps", [(0, 1)])
def test_LC5_order_2_on_nquad(comps):
    p = dd.PARAMS
    got = _evaluate(p, "per_sample", 2, 2, comps, *NQUAD).total
    assert _rel(got, _reference(p, 2, 2, comps)) < TOL["nquad"]


@pytest.mark.parametrize("n_points,order,comps,method,kw,tol", [
    (1, 1, (0,), *QMC_S, 1e-4), (1, 1, (1,), *QMC_S, 1e-4),
    (1, 1, (0,), *NQUAD, 1e-10), (1, 1, (1,), *NQUAD, 1e-10),
    (3, 1, (1, 0, 1), *QMC_S, 1e-4), (3, 1, (1, 0, 1), *NQUAD, 1e-10),
    (2, 2, (0, 1), *QMC_S, 1e-3),
])
def test_LC6_matrix_R_on_the_scalar_loops(n_points, order, comps, method, kw,
                                          tol):
    """Distinct rates make R a diagonal matrix, which the batched
    integrators do not take; the scalar QMC loop and nquad do."""
    p = dd.PARAMS_MATRIX_R
    got = _evaluate(p, "per_sample", n_points, order, comps, method, kw).total
    assert _rel(got, _reference(p, n_points, order, comps)) < tol


@pytest.mark.parametrize("coupling", ["per_sample", "batched"])
@pytest.mark.parametrize("n_points,order,comps", [
    (1, 1, (0,)), (1, 1, (1,)), (2, 2, (0, 1)), (2, 2, (1, 0)),
    (3, 1, (1, 0, 1))])
def test_LC7_position_dependent_drift(n_points, order, comps, coupling):
    p = replace(dd.PARAMS, beta=0.45)
    got = _evaluate(p, coupling, n_points, order, comps, *GL).total
    assert _rel(got, _reference(p, n_points, order, comps)) < 1e-12


# --------------------------------------------------------------------------- #
# LC8: YAML
# --------------------------------------------------------------------------- #
_F_MODULE = textwrap.dedent("""
    import numpy as np

    F0 = np.array({f0!r})

    def coupling_fn(n_2d, t_2d):
        t = np.asarray(t_2d, dtype=float)[0]
        return np.exp(-{decay!r} * t)[:, None, None, None] * F0[None]
""")

_YAML = textwrap.dedent("""
    system:
      field: {{name: phi, n_components: 2}}
      linear: {{type: diagonal, gamma: [1.0, 1.0]}}
      vertices:
        - name: F
          coupling_module: ./f_decay.py
          coupling_attr: coupling_fn
          rank: 3
          coupling_vectorized: true
      noise:
        kappa2:
          type: separable_translation
          temporal: {{type: exponential, lam: {lam}, sigma_t: {sigma_t}}}
          spatial:  {{type: gaussian, sigma_x: {sigma_x}}}
        sigma2: {{type: constant, amplitude: {white}}}
      t_min: {t_min}
    expand:
      observable: ["phi_a(x)", "phi_b(y)"]
      orders: [2]
    propagators:
      t_max: 2.6
      n_grid_t: 8
    sweep:
      positions_grid: {{x: [0.3], y: [-0.6]}}
      t_final_grid: [2.1]
      component_pairs: [[0, 1]]
      method: gauss_legendre
""")


def test_LC8_yaml_local_vertex_with_coupling_module(tmp_path):
    # A scalar white-noise amplitude keeps C diagonal (so the YAML defaults
    # apply) and declares the diagonal kink, which the Gauss-Legendre split
    # needs: with ``sigma2: null`` the split does not fire and the cusp of
    # the coloured C leaves 2.6e-6 at n_gauss = 12.
    white = 0.5
    p = replace(dd.PARAMS, S=((white, 0.0), (0.0, white)))
    (tmp_path / "f_decay.py").write_text(
        _F_MODULE.format(f0=[list(map(list, m)) for m in p.f0],
                         decay=p.decay))
    (tmp_path / "c.yaml").write_text(_YAML.format(
        lam=p.lam, sigma_t=p.sigma_t, sigma_x=p.sigma_x, t_min=p.t_min,
        white=white))
    system = build_system(load_workflow_config(tmp_path / "c.yaml").system)
    (vertex,) = system.vertices
    assert callable(vertex.coupling)
    assert (vertex.rank, vertex.coupling_vectorized) == (3, True)
    props = system.propagators(
        t_max=p.t_min + T + 0.5, c_closed_form="auto",
        c_closed_form_only=True, c_closed_form_vectorized=True,
        progress=False)
    got = system.expand(OBS[2], orders=[2]).evaluate(
        props, positions=POS, t_final=p.t_min + T, component_pair=(0, 1),
        orders=[2], method="gauss_legendre", n_gauss=12).total
    assert _rel(got, _reference(p, 2, 2, (0, 1))) < 1e-10
