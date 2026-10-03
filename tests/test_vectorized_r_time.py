r"""Vectorised response callables and the scalar-R distinct-pair path.

``PropagatorCache.R_time_batch`` used to wrap ``model.R_time`` in
``np.vectorize`` and call it once per causal sample, and
``R_matrix_batch`` once per distinct causal pair.  On a tensor-product
Gauss-Legendre grid most samples repeat a time pair, and in a weak-lensing
workload (``unify_wl``) 94 to 98.5 per cent of the CPU time of an
evaluation was spent in these Python calls.

A response callable may now carry ``vectorized = True``: it then accepts
two arrays of equal shape ``(n,)`` and returns ``(n,)`` (scalar R) or
``(n, N, N)`` (matrix R), and the batch methods call it once on the causal
pairs.  The four built-in R classes of ``DiagonalA`` are vectorised.
``ExplicitR(vectorized=True)`` and the YAML key
``system.linear.R_time_vectorized`` declare a user callable as such.  A
callable without the flag is called once per distinct causal pair, for a
scalar R as for a matrix R.

* **VR1** built-in R classes: an array call equals the scalar calls
  element by element with ``==``, acausal and equal times included.
* **VR2** ``R_time_batch``: strict Θ, one call per distinct causal pair for
  a scalar callable, one array call for a vectorised one, the same values
  with ``==`` as the old ``np.vectorize`` path.
* **VR3** ``R_matrix_batch`` with a vectorised matrix R.
* **VR4** shape errors from a vectorised callable.
* **VR5** ``ExplicitR(vectorized=True)``: the flag reaches the model, the
  user's callable is not modified, and the scalar call sites (``R_product``,
  ``C_value`` quadrature) still work with it.
* **VR6** channel totals: ``DiagonalA`` (vectorised built-ins) against the
  same R hidden behind a scalar-only wrapper (distinct-pair path), with
  ``==`` on ``gauss_legendre`` and ``qmc_vectorized``: scalar and matrix R,
  constant and time-dependent rates, ``t_min != 0``, distinct points, an
  off-diagonal component pair.
* **VR7** YAML ``R_time_vectorized``: parsed, lowered, probed with an array
  call at load time; a callable whose array call has the wrong shape or
  wrong values is refused.
* **VR8** the built-in R classes and the ``ExplicitR`` wrapper survive
  pickling with the flag (``n_jobs > 1``, disk caches), and a two-worker
  evaluation equals the serial one.
* **VR9** two-dimensional and broadcast time arrays.
* **VR10** non-local vertices: a static κ³, a callable ``equal_time`` κ³
  and a callable ``already_R_contracted`` κ³ with a matrix R, the
  three-point function at order 1 and the F-K channel of the two-point
  function at order 2, vectorised against scalar-only R with ``==``.
"""
from __future__ import annotations

import pickle
import textwrap
from pathlib import Path

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.evaluate import PropagatorCache, PropagatorModel
from sft_wick.workflow.specs import (
    _StaticIsoR,
    _StaticMatR,
    _TimeDepIsoR,
    _TimeDepMatR,
)

N = 2


def _gamma_t(t):
    return np.array([0.6 + 0.3 * np.sin(t), 1.6 - 0.2 * t])


def _gamma_t_iso(t):
    g = 0.8 + 0.3 * np.sin(t)
    return np.array([g, g])


def _builtin_Rs():
    """One instance of each built-in R class."""
    return {
        "static_iso": sw.DiagonalA(gamma=[0.7, 0.7]).build_R_callable(),
        "static_mat": sw.DiagonalA(gamma=[0.6, 1.6]).build_R_callable(),
        "timedep_iso": sw.DiagonalA(
            gamma=_gamma_t_iso, t_min_cache=0.0, t_max_cache=3.0,
            n_grid_cache=40).build_R_callable(),
        "timedep_mat": sw.DiagonalA(
            gamma=_gamma_t, t_min_cache=0.0, t_max_cache=3.0,
            n_grid_cache=40).build_R_callable(),
    }


class _ScalarOnly:
    """Hide a callable's ``vectorized`` flag: forces the distinct-pair
    path while keeping the same scalar values."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def __call__(self, t1, t2):
        assert np.ndim(t1) == 0 and np.ndim(t2) == 0
        self.calls.append((float(t1), float(t2)))
        return self.fn(t1, t2)


class _VecIso:
    """A vectorised scalar R that counts its calls."""

    vectorized = True

    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.array_calls = 0
        self.scalar_calls = 0

    def __call__(self, t1, t2):
        if np.ndim(t1) == 0 and np.ndim(t2) == 0:
            self.scalar_calls += 1
            return 0.0 if t1 < t2 else float(np.exp(-self.gamma * (t1 - t2)))
        self.array_calls += 1
        t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
        return np.where(t1 < t2, 0.0, np.exp(-self.gamma * (t1 - t2)))


def _bare_cache(R_time, iso_R=True) -> PropagatorCache:
    return PropagatorCache(PropagatorModel(
        R_time=R_time, kappa2=lambda n1, t1, n2, t2: np.eye(N),
        n_components=N, iso_R=iso_R))


T1 = np.array([1.0, 1.0, 0.5, 0.7, 1.0, 0.8, 2.9, 0.0])
T2 = np.array([0.2, 0.2, 0.5, 0.9, 0.3, 0.2, 0.1, 0.0])


# --------------------------------------------------------------------------- #
# VR1
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["static_iso", "static_mat", "timedep_iso",
                                  "timedep_mat"])
def test_VR1_builtin_R_array_call_equals_scalar_calls(kind):
    R = _builtin_Rs()[kind]
    assert getattr(R, "vectorized", False) is True
    got = np.asarray(R(T1, T2))
    want = np.array([R(float(a), float(b)) for a, b in zip(T1, T2)])
    assert got.shape == want.shape
    assert got.dtype == np.float64
    assert np.array_equal(got, want)
    # The scalar contract is unchanged: a float (iso) or an (N, N) array.
    if kind.endswith("iso"):
        assert type(R(1.0, 0.5)) is float
    else:
        assert np.asarray(R(1.0, 0.5)).shape == (N, N)


# --------------------------------------------------------------------------- #
# VR2
# --------------------------------------------------------------------------- #
def test_VR2_scalar_callable_called_once_per_distinct_causal_pair():
    R = _ScalarOnly(_StaticIsoR(0.9))
    out = _bare_cache(R).R_time_batch(T1, T2)
    assert sorted(R.calls) == [(0.8, 0.2), (1.0, 0.2), (1.0, 0.3), (2.9, 0.1)]
    old = np.where(T1 > T2, np.vectorize(_StaticIsoR(0.9), otypes=[float])(
        T1, T2), 0.0)
    assert np.array_equal(out, old)
    assert out[2] == 0.0 and out[3] == 0.0 and out[7] == 0.0


def test_VR2_vectorized_callable_called_once_on_arrays():
    R = _VecIso()
    out = _bare_cache(R).R_time_batch(T1, T2)
    assert (R.array_calls, R.scalar_calls) == (1, 0)
    ref = _StaticIsoR(0.9)
    old = np.array([ref(float(a), float(b)) if a > b else 0.0
                    for a, b in zip(T1, T2)])
    assert np.array_equal(out, old)


def test_VR2_vectorized_callable_not_called_without_causal_pairs():
    R = _VecIso()
    out = _bare_cache(R).R_time_batch(np.array([0.1, 0.5]),
                                      np.array([0.4, 0.5]))
    assert R.array_calls == 0 and not out.any()


def test_VR2_acausal_values_of_a_vectorized_R_are_masked():
    class _NoTheta:
        vectorized = True

        def __call__(self, t1, t2):
            return np.exp(-np.abs(np.asarray(t1) - np.asarray(t2)))

    out = _bare_cache(_NoTheta()).R_time_batch(T1, T2)
    assert out[2] == 0.0 and out[3] == 0.0 and out[7] == 0.0
    assert out[0] == np.exp(-0.8)


# --------------------------------------------------------------------------- #
# VR3
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["static_mat", "timedep_mat"])
def test_VR3_matrix_batch_vectorized_equals_distinct_pair_path(kind):
    R = _builtin_Rs()[kind]
    vec = _bare_cache(R, iso_R=False).R_matrix_batch(T1, T2)
    scal = _bare_cache(_ScalarOnly(R), iso_R=False).R_matrix_batch(T1, T2)
    assert vec.shape == (T1.size, N, N)
    assert np.array_equal(vec, scal)


# --------------------------------------------------------------------------- #
# VR4
# --------------------------------------------------------------------------- #
def test_VR4_vectorized_scalar_R_with_wrong_shape_raises():
    class _Bad:
        vectorized = True

        def __call__(self, t1, t2):
            return np.zeros(3)

    with pytest.raises(ValueError, match=r"vectorized R_time"):
        _bare_cache(_Bad()).R_time_batch(T1, T2)


def test_VR4_vectorized_matrix_R_with_wrong_shape_raises():
    class _Bad:
        vectorized = True

        def __call__(self, t1, t2):
            return np.zeros(np.shape(t1))

    with pytest.raises(ValueError, match=r"vectorized R_time"):
        _bare_cache(_Bad(), iso_R=False).R_matrix_batch(T1, T2)


@pytest.mark.parametrize("flag", ["yes", 1, None, (True,)])
def test_VR4_explicitR_vectorized_must_be_a_bool(flag):
    with pytest.raises(TypeError, match="vectorized"):
        sw.ExplicitR(R_time=_plain_R, vectorized=flag)


# --------------------------------------------------------------------------- #
# VR5
# --------------------------------------------------------------------------- #
def _plain_R(t1, t2):
    t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
    out = np.where(t1 < t2, 0.0, np.exp(-0.9 * (t1 - t2)))
    return float(out) if out.ndim == 0 else out


def test_VR5_explicitR_vectorized_flag_reaches_the_model():
    lin = sw.ExplicitR(R_time=_plain_R, vectorized=True)
    R = lin.build_R_callable()
    assert getattr(R, "vectorized", False) is True
    assert not hasattr(_plain_R, "vectorized")  # user callable untouched
    assert R(1.0, 0.4) == _plain_R(1.0, 0.4)
    assert np.array_equal(R(T1, T2), _plain_R(T1, T2))
    # Default: no flag, the callable is passed through as before.
    assert sw.ExplicitR(R_time=_plain_R).build_R_callable() is _plain_R


def test_VR5_vectorized_R_at_the_scalar_call_sites():
    cache_v = _bare_cache(sw.ExplicitR(R_time=_plain_R, vectorized=True)
                          .build_R_callable())
    cache_s = _bare_cache(_plain_R)
    pairs = (("a", "b"), ("c", "d"))
    times = {"a": 1.0, "b": 0.3, "c": 0.8, "d": 0.1}
    assert cache_v.R_product(pairs, times) == cache_s.R_product(pairs, times)
    assert np.array_equal(
        np.asarray(cache_v.C_value(0.0, 1.0, 0.0, 0.7)),
        np.asarray(cache_s.C_value(0.0, 1.0, 0.0, 0.7)))


# --------------------------------------------------------------------------- #
# VR6
# --------------------------------------------------------------------------- #
def _C_fn(n1, t1, n2, t2):
    """A smooth symmetric stand-in for C with off-diagonal entries; the
    test compares two R paths, so C only has to be the same in both."""
    r = abs(float(np.sum(n1)) - float(np.sum(n2)))
    m = np.array([[1.0, 0.3], [0.3, 0.7]])
    return (np.exp(-0.4 * (t1 + t2) - 0.5 * (t1 - t2) ** 2 - r) * m)


def _F():
    F = np.zeros((N, N, N))
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = 0.4
    F[1, 1, 0] = 0.6
    F[0, 0, 1] = -0.3
    return F


def _system(linear):
    return sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=linear,
        vertices=[sw.LocalVertex("F", coupling=_F())],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
            spatial=sw.ExponentialSpatial(sigma_x=1.0))),
        t_min=0.3,
    )


VR6_LINEARS = {
    "static_iso": lambda: sw.DiagonalA(gamma=[0.7, 0.7]),
    "static_mat": lambda: sw.DiagonalA(gamma=[0.6, 1.6]),
    "timedep_iso": lambda: sw.DiagonalA(
        gamma=_gamma_t_iso, t_max_cache=3.0, n_grid_cache=40),
    "timedep_mat": lambda: sw.DiagonalA(
        gamma=_gamma_t, t_max_cache=3.0, n_grid_cache=40),
}


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized",
                                    "qmc_scalar", "nquad"])
@pytest.mark.parametrize("kind", list(VR6_LINEARS))
def test_VR6_channel_totals_unchanged_by_vectorisation(kind, method):
    sys_v = _system(VR6_LINEARS[kind]())
    R_built = sys_v.build_propagator_model().R_time
    assert getattr(R_built, "vectorized", False) is True
    sys_s = _system(sw.ExplicitR(R_time=_ScalarOnly(R_built),
                                 iso_R=sys_v.iso_R))
    obs = ("phi_a(x)", "phi_b(y)")
    kw = dict(positions={"x": 0.1, "y": 0.6}, t_final=2.1,
              component_pair=(0, 1), orders=[0, 2], method=method,
              n_gauss=6, n_samples=2 ** 7 if method == "qmc_scalar"
              else 2 ** 9, seed=3,
              diag_R=False, diag_C=False)
    totals = []
    for system in (sys_v, sys_s):
        props = system.propagators(t_max=2.5, n_grid_t=12,
                                   c_closed_form=_C_fn,
                                   c_closed_form_only=True, diag_C=False)
        res = system.expand(obs, orders=[0, 2], diag_R=False,
                            diag_C=False).evaluate(props, **{
                                k: v for k, v in kw.items()
                                if k not in ("diag_R", "diag_C")})
        totals.append(res.total)
    assert totals[0] != 0.0
    assert totals[0] == totals[1]


# --------------------------------------------------------------------------- #
# VR7
# --------------------------------------------------------------------------- #
_VEC_MODULE = textwrap.dedent("""
    import numpy as np

    def R_time(t1, t2):
        t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
        out = np.where(t1 < t2, 0.0, np.exp(-1.0 * (t1 - t2)))
        return float(out) if out.ndim == 0 else out

    def R_scalar_only(t1, t2):
        if t1 < t2:
            return 0.0
        return float(np.exp(-1.0 * (t1 - t2)))

    def R_wrong_shape(t1, t2):
        if np.ndim(t1) == 0:
            return R_scalar_only(t1, t2)
        return np.zeros((len(t1), 2, 2))

    def R_wrong_values(t1, t2):
        if np.ndim(t1) == 0:
            return R_scalar_only(t1, t2)
        return np.exp(-2.0 * (np.asarray(t1) - np.asarray(t2)))
""")

_YAML = textwrap.dedent("""
    system:
      field: {{name: phi, n_components: 2}}
      linear:
        type: explicit
        R_time_module: ./R_time.py
        R_time_attr: {attr}
        iso_R: true
        R_time_vectorized: {flag}
      vertices: []
      nonlocal_vertices: []
      noise:
        kappa2:
          type: separable_translation
          temporal: {{type: exponential, lam: 0.05, sigma_t: 0.3}}
          spatial:  {{type: exponential, sigma_x: 1.0}}
        sigma2: null
    expand:
      observable: ["phi_a(x)", "phi_b(y)"]
      orders: [0]
    propagators:
      t_max: 2.0
      n_grid_t: 12
    sweep:
      positions_grid: {{x: [0.0], y: [0.0]}}
      t_final_grid: [1.0]
      component_pairs: [[0, 0]]
""")


def _load(tmp_path: Path, attr: str, flag: str):
    from sft_wick.workflow.config import build_system, load_workflow_config
    (tmp_path / "R_time.py").write_text(_VEC_MODULE)
    path = tmp_path / "c.yaml"
    path.write_text(_YAML.format(attr=attr, flag=flag))
    return build_system(load_workflow_config(path).system)


def test_VR7_yaml_R_time_vectorized_lowers_to_flagged_explicitR(tmp_path):
    system = _load(tmp_path, "R_time", "true")
    assert system.linear.vectorized is True
    R = system.build_propagator_model().R_time
    assert getattr(R, "vectorized", False) is True
    assert np.array_equal(R(T1, T2),
                          np.where(T1 < T2, 0.0, np.exp(-(T1 - T2))))


def test_VR7_yaml_default_is_not_vectorized(tmp_path):
    system = _load(tmp_path, "R_scalar_only", "false")
    assert system.linear.vectorized is False


def test_VR7_yaml_probe_rejects_a_scalar_only_callable(tmp_path):
    with pytest.raises(ValueError, match=r"R_time_vectorized"):
        _load(tmp_path, "R_scalar_only", "true")


@pytest.mark.parametrize("attr, match", [("R_wrong_shape", "shape"),
                                         ("R_wrong_values", "differs")])
def test_VR7_yaml_probe_rejects_inconsistent_array_calls(tmp_path, attr,
                                                        match):
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, attr, "true")


# --------------------------------------------------------------------------- #
# VR8
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["static_iso", "static_mat", "timedep_iso",
                                  "timedep_mat"])
def test_VR8_builtin_R_pickles_with_its_flag(kind):
    R = _builtin_Rs()[kind]
    R2 = pickle.loads(pickle.dumps(R))
    assert getattr(R2, "vectorized", False) is True
    assert np.array_equal(np.asarray(R2(T1, T2)), np.asarray(R(T1, T2)))


def test_VR8_explicitR_wrapper_pickles_with_its_flag():
    R = sw.ExplicitR(R_time=_plain_R, vectorized=True).build_R_callable()
    R2 = pickle.loads(pickle.dumps(R))
    assert getattr(R2, "vectorized", False) is True
    assert np.array_equal(R2(T1, T2), R(T1, T2))


def test_VR8_two_workers_equal_serial():
    system = _system(VR6_LINEARS["timedep_mat"]())
    props = system.propagators(t_max=2.5, n_grid_t=12, c_closed_form=_C_fn,
                               c_closed_form_only=True, diag_C=False)
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2], diag_R=False,
                        diag_C=False)
    kw = dict(positions={"x": 0.1, "y": 0.6}, t_final=2.1,
              component_pair=(0, 1), orders=[2], method="gauss_legendre",
              n_gauss=5)
    serial = exp.evaluate(props, n_jobs=1, **kw).total
    parallel = exp.evaluate(props, n_jobs=2, **kw).total
    assert serial != 0.0 and parallel == serial


# --------------------------------------------------------------------------- #
# VR9
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("vectorized", [True, False])
def test_VR9_two_dimensional_and_broadcast_inputs(vectorized):
    R = _VecIso() if vectorized else _ScalarOnly(_StaticIsoR(0.9))
    cache = _bare_cache(R)
    a, b = T1.reshape(2, 4), T2.reshape(2, 4)
    out = cache.R_time_batch(a, b)
    assert out.shape == (2, 4)
    assert np.array_equal(out.reshape(-1), cache.R_time_batch(T1, T2))
    out_b = cache.R_time_batch(1.0, T2)
    want = np.array([_StaticIsoR(0.9)(1.0, float(t)) if 1.0 > t else 0.0
                     for t in T2])
    assert out_b.shape == T2.shape and np.array_equal(out_b, want)
    mat = _builtin_Rs()["static_mat"]
    for Rm in (mat, _ScalarOnly(mat)):
        out_m = _bare_cache(Rm, iso_R=False).R_matrix_batch(a, b)
        assert out_m.shape == (2, 4, N, N)
        assert np.array_equal(out_m.reshape(-1, N, N),
                              _bare_cache(mat, iso_R=False)
                              .R_matrix_batch(T1, T2))


# --------------------------------------------------------------------------- #
# VR10
# --------------------------------------------------------------------------- #
_K3 = np.random.default_rng(5).normal(size=(N, N, N))


def _k3_equal_time(n_list, t_list):
    """A callable equal-time κ³: depends on the shared leg time and on the
    three leg positions, no index symmetry."""
    t = float(np.asarray(t_list)[0])
    n = np.asarray(n_list, dtype=float)
    return _K3 * (1.0 + 0.3 * np.sin(t)) * np.exp(-0.2 * (n ** 2).sum())


def _k3_r_contracted(n_list, t_list):
    """A callable R-contracted κ³ at three partner points; any smooth
    function serves, the test compares two R paths."""
    t = np.asarray(t_list, dtype=float)
    n = np.asarray(n_list, dtype=float)
    return _K3 * np.exp(-0.3 * t.sum() - 0.1 * (n ** 2).sum()) * (
        1.0 + 0.2 * t[0] - 0.1 * t[2])


VR10_VERTICES = {
    "static": lambda: sw.NonLocalVertex("K", order=3, coupling=_K3),
    "equal_time": lambda: sw.NonLocalVertex(
        "K", order=3, coupling=_k3_equal_time, equal_time=True),
    "r_contracted": lambda: sw.NonLocalVertex(
        "K", order=3, coupling=_k3_r_contracted, already_R_contracted=True),
}

VR10_CASES = [
    (("phi_a(x)", "phi_b(y)", "phi_c(z)"), 1, (0, 1, 1)),
    (("phi_a(x)", "phi_b(y)"), 2, (1, 0)),
]


def _nonlocal_system(linear, vertex):
    return sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=linear,
        vertices=[sw.LocalVertex("F", coupling=_F())],
        nonlocal_vertices=[vertex],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
            spatial=sw.ExponentialSpatial(sigma_x=1.0))),
        t_min=0.3,
    )


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
@pytest.mark.parametrize("obs, order, comps", VR10_CASES)
@pytest.mark.parametrize("vertex", list(VR10_VERTICES))
def test_VR10_nonlocal_vertices_unchanged_by_vectorisation(vertex, obs,
                                                           order, comps,
                                                           method):
    sys_v = _nonlocal_system(VR6_LINEARS["timedep_mat"](),
                             VR10_VERTICES[vertex]())
    R_built = sys_v.build_propagator_model().R_time
    sys_s = _nonlocal_system(sw.ExplicitR(R_time=_ScalarOnly(R_built),
                                          iso_R=False),
                             VR10_VERTICES[vertex]())
    positions = {"x": 0.1, "y": 0.6, "z": -0.4}
    totals = []
    for system in (sys_v, sys_s):
        props = system.propagators(t_max=2.5, n_grid_t=12,
                                   c_closed_form=_C_fn,
                                   c_closed_form_only=True, diag_C=False)
        exp = system.expand(obs, orders=[order], diag_R=False, diag_C=False)
        totals.append(exp.evaluate(
            props, positions={k: positions[k] for k in "xyz"[:len(obs)]},
            t_final=2.1, component_pair=comps, orders=[order],
            method=method, n_gauss=4, n_samples=2 ** 8, seed=7).total)
    assert totals[0] != 0.0
    assert totals[0] == totals[1]


@pytest.mark.parametrize("t_max, span", [(2.5, 2.0), (None, 1.0),
                                         ("auto", 1.0), (0.2, 1.0),
                                         (float("inf"), 1.0)])
def test_VR7_probe_window(t_max, span):
    from sft_wick.workflow.config import _probe_span
    assert _probe_span(0.5, t_max) == span
