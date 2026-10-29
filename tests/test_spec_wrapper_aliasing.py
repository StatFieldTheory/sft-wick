r"""Spec wrappers own the values of user callables they keep.

SW1-SW3: ``DiagonalA`` with a callable rate collected every ``gamma(t)``
return value in a list and stacked the list after the last call.  A rate
that fills and returns one buffer therefore gave every row of the spline
grid the rate at ``t_max_cache``, with no error.  For the rate below on the
grid ``[0.35, 4]``, R(2.7, 1.1) came out 0.1264 and 0.4643 against the
exact 0.1389 and 0.2666 (-9.0 % and +74 %).  Through ``Expansion.evaluate``
the order-0 values were -8.9 % and +53 % off, the order-2 value at (1, 1)
+129 %, and the order-2 value at (0, 1) had the wrong sign (0.00368 against
-0.00112).  Through YAML ``gamma_module``, a rate whose two components meet
only at ``t_max_cache`` turned a matrix R into a scalar one (``iso_R``
true), and component 1 got component 0's response.  Each value is now
copied as it is collected.

SW4-SW5: ``SeparableTranslation`` and ``SeparableRotation`` given raw
callables kept the temporal kernel's value while they called the spatial
(angular) kernel.  Two kernels that fill one shared buffer made κ² the
square of the second factor: 0.3282 against 0.0767 in translation and
0.3145 against 0.0751 in rotation.  At order 0 and distinct points, C_00
came out 76 % high and C_11 78 % high on the lazy translation table, whose
base is built at zero separation where the spatial factor is 1, and 6.5 %
and 8.2 % high on the full table.  In rotation C_00 came out 1.5 % low and
C_11 within 3e-5 of the fresh value.  Order 2 moved at every component
pair, by 9.8 % to 141 %.  ``CustomKernel`` and the built-in kernels return
Python floats and were not affected.  The temporal value is now copied when
it is an array; a float passes unchanged.

SW6-SW7: the YAML probe of ``R_time_vectorized: true`` collected the scalar
reference values in a list and built them from the time arrays it had just
passed to the user's R.  An R that fills one buffer on scalar calls, or that
uses its time arrays as scratch space, was refused at load with "differs
from its scalar calls" although it integrates correctly.  The reference is
now built first, from floats, with each value copied.

References share no code with the package: the rates have closed-form
integrals Γ_a, and the order-0 C is a nested scipy ``quad`` over the two
source times with the response in closed form.  Every comparison of a
buffer-reusing or argument-writing callable is against the same callable
without that behaviour, with ``==``.  Inputs are generic: two components
with distinct rates, a vertex with entries of both signs, ``t_min != 0``,
distinct points and distinct external times, an off-diagonal component pair.
"""
from __future__ import annotations

import os
import textwrap
from math import cos, exp, sin
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

import sft_wick as sw

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

T0 = 0.35
F = np.array([[[0.45, -0.3], [-0.3, 0.2]], [[-0.25, 0.15], [0.15, 0.55]]])
X, Y = 0.3, -0.6
TX, TY = 2.1, 1.6
PAIRS = ((0, 0), (1, 1), (0, 1))
# The C tables span [T0, T_MAX] at spacing 0.25, so TX, TY and TX - TY are
# nodes, as is |X - Y| = 0.9 on the full translation table's r grid.
T_MAX, N_GRID_T = 2.35, 9


def _nested(resp, kt, t1, t2):
    """``∫_{T0}^{t1} dl1 ∫_{T0}^{t2} dl2 resp(t1, l1) resp(t2, l2) kt(l1 - l2)``,
    with the inner integral split at the cusp ``l2 = l1``."""

    def inner(l1):
        pts = [l1] if T0 < l1 < t2 else None
        return quad(lambda l2: resp(t2, l2) * kt(l1 - l2), T0, t2,
                    points=pts, epsabs=0, epsrel=1e-12, limit=200)[0]

    return quad(lambda l1: resp(t1, l1) * inner(l1), T0, t1,
                epsabs=0, epsrel=1e-12, limit=200)[0]


def _totals(system, props, **kw):
    exp_ = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2],
                         progress=False)
    out = {}
    for order in (0, 2):
        out[order] = [
            exp_.evaluate(props, positions=kw.get("positions", {"x": X, "y": Y}),
                          t_final=TX, external_times={"x": TX, "y": TY},
                          component_pair=pair, orders=[order],
                          method="gauss_legendre", n_gauss=6).total
            for pair in PAIRS
        ]
    return out


# ---------------------------------------------------------------------------
# SW1-SW3: a callable rate that fills and returns one buffer
# ---------------------------------------------------------------------------

def _rates(t):
    return 1.3 - 0.4 * exp(-t), 0.5 + 0.35 * sin(0.8 * t)


def _Gamma(a, t):
    """Closed-form antiderivative of component ``a`` of :func:`_rates`."""
    if a == 0:
        return 1.3 * t + 0.4 * exp(-t)
    return 0.5 * t - 0.4375 * cos(0.8 * t)


def _R_exact(a, t1, l1):
    return exp(-(_Gamma(a, t1) - _Gamma(a, l1))) if t1 >= l1 else 0.0


class _FreshRate:
    def __call__(self, t):
        return np.array(_rates(t))

    def __repr__(self):
        return "_FreshRate()"


class _BufferRate:
    """The same rate, written into one array that every call returns."""

    def __init__(self):
        self.buffer = np.empty(2)

    def __call__(self, t):
        self.buffer[:] = _rates(t)
        return self.buffer

    def __repr__(self):
        return "_BufferRate()"


def _rate_system(rate):
    # t_min_cache above t_min: System._effective_linear extends the grid,
    # so the replaced copy of the operator is the one that builds it.
    return sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=rate, t_min_cache=0.9, t_max_cache=4.0,
                            n_grid_cache=121),
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.6, sigma_t=0.5),
            spatial=sw.GaussianSpatial(sigma_x=1.1))),
        vertices=[sw.LocalVertex("F", coupling=F)],
        t_min=T0,
    )


_R_PAIRS = ((2.7, 1.1), (1.9, 0.4), (3.8, 3.1), (1.2, 2.0))


def test_SW1_buffer_rate_gives_the_fresh_grid_and_response():
    """The rate buffer gave every grid row gamma(t_max_cache), and R(2.7,
    1.1) came out -9.0 % and +74 % off the exact value."""
    grids, Rs = {}, {}
    for name, rate in (("fresh", _FreshRate()), ("buffer", _BufferRate())):
        system = _rate_system(rate)
        assert system.iso_R is False
        grids[name] = system._effective_linear._gamma_grid
        R = system.build_propagator_model().R_time
        Rs[name] = [np.asarray(R(t1, t2)) for t1, t2 in _R_PAIRS]
    np.testing.assert_array_equal(grids["buffer"][1], grids["fresh"][1])
    want_grid = np.array([_rates(t) for t in grids["fresh"][0]])
    np.testing.assert_allclose(grids["fresh"][1], want_grid, rtol=1e-15)
    for got_fresh, got_buffer, (t1, t2) in zip(Rs["fresh"], Rs["buffer"],
                                               _R_PAIRS):
        np.testing.assert_array_equal(got_buffer, got_fresh)
        want = np.diag([_R_exact(a, t1, t2) for a in range(2)])
        # The Γ spline is O(h^4) at h = 0.026.
        np.testing.assert_allclose(got_fresh, want, rtol=1e-8, atol=0)


@pytest.fixture(scope="module")
def rate_totals():
    out = {}
    for name, rate in (("fresh", _FreshRate()), ("buffer", _BufferRate())):
        system = _rate_system(rate)
        props = system.propagators(t_max=T_MAX, n_grid_t=N_GRID_T, progress=False)
        out[name] = _totals(system, props)
    return out


def test_SW2_buffer_rate_gives_the_fresh_totals(rate_totals):
    """System -> propagators -> Expansion.evaluate at orders 0 and 2: the
    buffer rate gives the fresh rate's totals bit for bit (it gave -8.9 %
    and +53 % at order 0, and the off-diagonal order-2 value of the wrong
    sign), and the fresh order-0 values match the nested-quad reference
    (9.5e-11 measured; the external times are table nodes)."""
    fresh, buffer = rate_totals["fresh"], rate_totals["buffer"]
    assert all(v != 0.0 for v in fresh[2])
    assert buffer == fresh
    kx = exp(-((X - Y) ** 2) / (2 * 1.1**2))
    want = [
        kx * _nested(lambda t, s, a=a: _R_exact(a, t, s),
                     lambda d: 0.6 * exp(-abs(d) / 0.5), TX, TY)
        for a in range(2)
    ]
    np.testing.assert_allclose(fresh[0], [want[0], want[1], 0.0],
                               rtol=1e-8, atol=1e-15)


_GAMMA_MODULE = textwrap.dedent("""
    import numpy as np

    T_CROSS = 4.0

    def _rates(t):
        g0 = 1.0 + 0.3 * np.sin(t)
        # component 1 equals component 0 only at t = T_CROSS (t_max_cache)
        return g0, g0 + 0.15 * (t - T_CROSS)

    def gamma_fresh(t):
        return np.array(_rates(t))

    _BUFFER = np.empty(2)

    def gamma_buffer(t):
        _BUFFER[0], _BUFFER[1] = _rates(t)
        return _BUFFER
""")

_GAMMA_YAML = textwrap.dedent("""
    system:
      field: {{name: phi, n_components: 2}}
      t_min: 0.35
      linear:
        type: diagonal
        gamma_module: ./sw_gamma_rate.py
        gamma_attr: {attr}
        t_min_cache: 0.35
        t_max_cache: 4.0
        n_grid_cache: 121
      vertices: []
      nonlocal_vertices: []
      noise:
        kappa2:
          type: separable_translation
          temporal: {{type: exponential, lam: 0.6, sigma_t: 0.5}}
          spatial:  {{type: gaussian, sigma_x: 1.1}}
        sigma2: null
    expand:
      observable: ["phi_a(x)", "phi_b(y)"]
      orders: [0]
    propagators:
      t_max: 2.4
      n_grid_t: 12
    sweep:
      positions_grid: {{x: [0.3], y: [-0.6]}}
      t_final_grid: [2.2]
      component_pairs: [[1, 1]]
""")


def test_SW3_yaml_buffer_rate_keeps_a_matrix_response(tmp_path):
    """Through ``gamma_module``: a buffer rate whose components cross at
    t_max_cache made every grid row equal there, so iso_R came out true and
    component 1 got component 0's response."""
    from sft_wick.workflow.config import build_system, load_workflow_config

    (tmp_path / "sw_gamma_rate.py").write_text(_GAMMA_MODULE)
    Rs = {}
    for attr in ("gamma_fresh", "gamma_buffer"):
        path = tmp_path / f"{attr}.yaml"
        path.write_text(_GAMMA_YAML.format(attr=attr))
        system = build_system(load_workflow_config(path).system)
        assert system.iso_R is False, attr
        Rs[attr] = np.asarray(system.build_propagator_model().R_time(2.7, 1.1))
    np.testing.assert_array_equal(Rs["gamma_buffer"], Rs["gamma_fresh"])

    def Gamma(a, t):
        g0 = t - 0.3 * cos(t)
        return g0 if a == 0 else g0 + 0.075 * (t - 4.0) ** 2

    want = np.diag([exp(-(Gamma(a, 2.7) - Gamma(a, 1.1))) for a in range(2)])
    np.testing.assert_allclose(Rs["gamma_fresh"], want, rtol=1e-8, atol=0)


# ---------------------------------------------------------------------------
# SW4-SW5: separable κ² kernels that share one output buffer
# ---------------------------------------------------------------------------

_SHARED = np.empty(())


class _Kernel:
    """A 1-D kernel returning a 0-d array: a fresh one, or the module
    buffer ``_SHARED`` that every shared kernel fills."""

    def __init__(self, name, fn, shared):
        self.name, self.fn, self.shared = name, fn, shared

    def __call__(self, x):
        value = self.fn(float(x))
        if not self.shared:
            return np.array(value)
        _SHARED[...] = value
        return _SHARED

    def __repr__(self):
        return f"_Kernel({self.name!r}, shared={self.shared})"


def _kt(dt):
    return 1.3 * exp(-abs(dt) / 0.55)


def _kx(r):
    return exp(-(r * r) / (2 * 0.9**2))


def _kang(c):
    return 0.5 + 0.4 * c


V1 = np.array([0.0, 0.6, 0.8])
V2 = np.array([0.48, -0.6, 0.64])
COS12 = float(V1 @ V2)
GAMMA_SEP = (0.65, 1.45)


def _separable_system(kind, shared):
    temporal = _Kernel("t", _kt, shared)
    if kind == "rotation":
        kappa2 = sw.SeparableRotation(temporal=temporal,
                                      angular=_Kernel("ang", _kang, shared))
    else:
        kappa2 = sw.SeparableTranslation(temporal=temporal,
                                         spatial=_Kernel("x", _kx, shared))
    return sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=list(GAMMA_SEP)),
        noise=sw.GaussianNoise(kappa2=kappa2),
        vertices=[sw.LocalVertex("F", coupling=F)],
        t_min=T0,
    )


@pytest.mark.parametrize("kind", ["translation", "rotation"])
def test_SW4_shared_kernel_buffer_gives_the_product(kind):
    """kappa2 = kappa_t(dt) kappa_x(r) I; a shared buffer gave
    kappa_x(r)^2 (0.3282 against 0.0767, and 0.3145 against 0.0751)."""
    if kind == "rotation":
        n1, n2, spatial = V1, V2, _kang(COS12)
    else:
        n1, n2, spatial = 0.35, -0.6, _kx(0.95)
    want = _kt(1.7 - 0.45) * spatial * np.eye(2)
    for shared in (False, True):
        k2 = _separable_system(kind, shared).build_propagator_model().kappa2
        got = np.asarray(k2(n1, 1.7, n2, 0.45))
        np.testing.assert_allclose(got, want, rtol=1e-15, atol=0)


_SEPARABLE_PATHS = {
    "translation-lazy-gl": ("translation", dict(c_method="gauss_legendre")),
    "translation-lazy-dblquad": ("translation", dict(c_method="dblquad")),
    "translation-full-gl": ("translation", dict(c_method="gauss_legendre",
                                                r_max=2.7, n_grid_r=4)),
    "rotation-lazy-gl": ("rotation", dict(c_method="gauss_legendre")),
}


@pytest.mark.parametrize("path", sorted(_SEPARABLE_PATHS))
def test_SW5_shared_kernel_buffer_gives_the_fresh_totals(path):
    """System -> propagators -> Expansion.evaluate at orders 0 and 2 on the
    lazy and full tables: the shared-buffer kernels give the fresh kernels'
    totals bit for bit (at order 0, C_00 and C_11 were 76 % and 78 % high
    on the lazy translation table and 6.5 % and 8.2 % high on the full one;
    in rotation C_00 was 1.5 % low, C_11 within 3e-5 and order 2 off by
    9.8 % to 16 %), and the fresh order-0 values match the nested-quad
    reference (7e-14 measured; the external times and the separation are
    table nodes)."""
    kind, kw = _SEPARABLE_PATHS[path]
    positions = {"x": V1, "y": V2} if kind == "rotation" else {"x": X, "y": Y}
    totals = {}
    for shared in (False, True):
        system = _separable_system(kind, shared)
        props = system.propagators(t_max=T_MAX, n_grid_t=N_GRID_T,
                                   progress=False, **kw)
        totals[shared] = _totals(system, props, positions=positions)
    fresh = totals[False]
    assert all(v != 0.0 for v in fresh[2])
    assert totals[True] == fresh
    spatial = _kang(COS12) if kind == "rotation" else _kx(abs(X - Y))
    want = [
        spatial * _nested(lambda t, s, g=g: exp(-g * (t - s)) if t >= s else 0.0,
                          _kt, TX, TY)
        for g in GAMMA_SEP
    ]
    np.testing.assert_allclose(fresh[0], [want[0], want[1], 0.0],
                               rtol=1e-10, atol=1e-15)


# ---------------------------------------------------------------------------
# SW6-SW7: the YAML probe of R_time_vectorized
# ---------------------------------------------------------------------------

_R_MODULE = textwrap.dedent("""
    import numpy as np

    G = (0.9, 1.6)
    TAU = 1.9

    def _values(t1, t2, iso):
        # A causal R with time-dependent rates; dense and lower triangular
        # for a matrix R.
        t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
        d = (t1 - t2) * (1.0 + 0.1 * (t1 + t2))
        c = t1 >= t2
        if iso:
            return np.where(c, np.exp(-G[0] * d), 0.0)
        out = np.zeros(np.shape(t1) + (2, 2))
        out[..., 0, 0] = np.where(c, np.exp(-G[0] * d), 0.0)
        out[..., 1, 1] = np.where(c, np.exp(-G[1] * d), 0.0)
        out[..., 1, 0] = np.where(c, -0.35 * d * np.exp(-G[0] * d), 0.0)
        return out

    def _make(iso):
        buffer = np.zeros(() if iso else (2, 2))

        def fresh(t1, t2):
            return _values(t1, t2, iso)

        def buffered(t1, t2):
            # scalar calls fill and return one buffer
            if np.ndim(t1) == 0:
                buffer[...] = _values(t1, t2, iso)
                return buffer
            return _values(t1, t2, iso)

        def writes(t1, t2):
            # array calls reuse their time arrays as scratch space once
            # they have read them
            if np.ndim(t1) == 0:
                return _values(t1, t2, iso)
            out = _values(t1, t2, iso)
            t1 /= TAU
            t2 /= TAU
            return out

        def buffered_wrong(t1, t2):
            # buffered, and wrong on arrays: must still be refused
            if np.ndim(t1) == 0:
                return buffered(t1, t2)
            return 1.01 * _values(t1, t2, iso)

        return fresh, buffered, writes, buffered_wrong

    iso_fresh, iso_buffered, iso_writes, iso_buffered_wrong = _make(True)
    mat_fresh, mat_buffered, mat_writes, mat_buffered_wrong = _make(False)
""")

_R_YAML = textwrap.dedent("""
    system:
      field: {{name: phi, n_components: 2}}
      t_min: 0.35
      linear:
        type: explicit
        R_time_module: ./sw_vectorized_R.py
        R_time_attr: {attr}
        iso_R: {iso}
        R_time_vectorized: true
      vertices: []
      nonlocal_vertices: []
      noise:
        kappa2:
          type: separable_translation
          temporal: {{type: exponential, lam: 0.6, sigma_t: 0.5}}
          spatial:  {{type: gaussian, sigma_x: 1.1}}
        sigma2: null
    expand:
      observable: ["phi_a(x)", "phi_b(y)"]
      orders: [0]
      diag_R: false
      diag_C: false
    propagators:
      t_max: 2.6
      n_grid_t: 12
      diag_C: false
    sweep:
      positions_grid: {{x: [0.3], y: [-0.6]}}
      t_final_grid: [2.2]
      component_pairs: [[1, 0]]
""")


def _load_R(tmp_path: Path, attr: str):
    from sft_wick.workflow.config import build_system, load_workflow_config

    (tmp_path / "sw_vectorized_R.py").write_text(_R_MODULE)
    path = tmp_path / f"{attr}.yaml"
    iso = "true" if attr.startswith("iso") else "false"
    path.write_text(_R_YAML.format(attr=attr, iso=iso))
    return build_system(load_workflow_config(path).system)


_PROBE_T1 = np.array([1.925, 2.375, 1.475, 0.9])
_PROBE_T2 = np.array([0.8, 0.575, 1.25, 1.3])


@pytest.mark.parametrize("variant", ["buffered", "writes"])
@pytest.mark.parametrize("kind", ["iso", "mat"])
def test_SW6_vectorized_probe_accepts_a_consistent_R(tmp_path, kind, variant):
    """A vectorised R that fills one buffer on scalar calls, or uses its
    time arrays as scratch space, gives the same values as the fresh R on
    floats and on arrays; the probe used to refuse it at load."""
    fresh = _load_R(tmp_path, f"{kind}_fresh").build_propagator_model().R_time
    system = _load_R(tmp_path, f"{kind}_{variant}")
    R = system.build_propagator_model().R_time
    assert system.iso_R is (kind == "iso")
    for t1, t2 in zip(_PROBE_T1, _PROBE_T2):
        np.testing.assert_array_equal(np.asarray(R(float(t1), float(t2))),
                                      np.asarray(fresh(float(t1), float(t2))))
    np.testing.assert_array_equal(R(_PROBE_T1.copy(), _PROBE_T2.copy()),
                                  fresh(_PROBE_T1.copy(), _PROBE_T2.copy()))


@pytest.mark.parametrize("kind", ["iso", "mat"])
def test_SW7_vectorized_probe_still_refuses_a_wrong_array_call(tmp_path, kind):
    """The reordered probe still refuses an array call 1 % off its scalar
    calls, when the scalar calls fill one buffer."""
    with pytest.raises(ValueError, match="differs from its scalar calls"):
        _load_R(tmp_path, f"{kind}_buffered_wrong")
