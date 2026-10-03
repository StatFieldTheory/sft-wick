r"""The change of integration variable of ``docs/user_guide/time_variable.rst``.

sft-wick integrates in the time of the system.  A caller who wants another
variable ``u``, ``t = f(u)`` increasing with ``J = dt/du``, poses the system
in ``u`` with the fields rescaled as ``φ~ = J φ`` and ``ψ~ = ψ / J``.  Each
ingredient then carries a fixed power of ``J``; the page lists them.  Here
one system is built in ``t`` and the same system in ``u`` with a curved map
``t = u + 0.4 u²`` and ``t_min = 0.3``, two components, cumulants and
couplings without index symmetry, and every row of the table is checked:

* **TV1** C from κ² and a white σ² by the package's own quadrature, R from
  ``DiagonalA`` with a time-dependent rate: ``γ~ = J γ − d ln J / du``,
  ``κ²~ = J₁² J₂² κ²``, ``σ²~ = J³ σ²``, ``C~ = J₁ J₂ C``.
* **TV2** diagrams with a closed-form R and C: a static cubic F (unchanged),
  a static quartic G (``J⁻¹ G``), a raw κ³ static and callable
  (``Π J(u_i)² κ``), an ``equal_time`` κ³ (``J⁴ κ``), an
  ``already_R_contracted`` κ³ (``Π J(u'_i) K_R`` at the partner points);
  integrated externals with unit weight and fixed externals divided by
  ``J``; three- and two-point functions, orders 0 to 2.
* **TV3** each rule with one factor of ``J`` missing or wrong: the two
  systems then disagree, so TV1 and TV2 test the factors.
* **TV4** why: an Einstein-de Sitter lensing toy (``λ`` the affine
  parameter, ``u = χ`` the comoving distance) at a source at z = 1100.

The reference for each value is the same quantity in the other variable,
computed by the package; the map and every kernel are closed forms.
"""
from __future__ import annotations

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.evaluate import PropagatorCache

A = 0.4
N = 2
T_MIN, T = 0.3, 2.2
U_MIN = float((-1.0 + np.sqrt(1.0 + 4.0 * A * T_MIN)) / (2.0 * A))


def t_of(u):
    return u + A * u * u


def u_of(t):
    return (-1.0 + np.sqrt(1.0 + 4.0 * A * t)) / (2.0 * A)


def J(u):
    return 1.0 + 2.0 * A * u


def dlnJ(u):
    return 2.0 * A / (1.0 + 2.0 * A * u)


def _ones(u):
    return np.ones_like(np.asarray(u, dtype=float))


# --------------------------------------------------------------------------- #
# TV1: C from κ² and σ² by quadrature, DiagonalA with a time-dependent rate
# --------------------------------------------------------------------------- #
M2 = np.array([[1.0, 0.4], [0.4, 0.6]])
S2 = np.array([[0.5, 0.1], [0.1, 0.3]])


def gamma_t(t):
    return np.array([0.7 + 0.2 * t, 1.3 - 0.1 * t])


def k2_t(n1, t1, n2, t2):
    return (M2 * np.exp(-(t1 - t2) ** 2 / 0.8) * (1.0 + 0.3 * t1 * t2)
            * np.exp(-0.5 * abs(n1 - n2)))


def s2_t(n1, t, n2):
    return S2 * (1.0 + 0.5 * t) * np.exp(-abs(n1 - n2))


class _GammaU:
    def __init__(self, shift=True):
        self.shift = shift

    def __call__(self, u):
        return J(u) * gamma_t(t_of(u)) - (dlnJ(u) if self.shift else 0.0)


class _K2U:
    def __init__(self, power=2):
        self.power = power

    def __call__(self, n1, u1, n2, u2):
        return ((J(u1) * J(u2)) ** self.power
                * k2_t(n1, t_of(u1), n2, t_of(u2)))


class _S2U:
    def __init__(self, power=3):
        self.power = power

    def __call__(self, n1, u, n2):
        return J(u) ** self.power * s2_t(n1, t_of(u), n2)


def _c_system(variable, *, gamma=None, k2=None, s2=None):
    if variable == "t":
        linear = sw.DiagonalA(gamma=gamma_t, t_max_cache=3.0,
                              n_grid_cache=3000)
        noise = sw.GaussianNoise(kappa2=sw.GeneralKappa2(fn=k2_t),
                                 sigma2=sw.CustomImpulse(fn=s2_t))
        t_min = T_MIN
    else:
        linear = sw.DiagonalA(gamma=gamma or _GammaU(),
                              t_max_cache=float(u_of(3.0)),
                              n_grid_cache=3000)
        noise = sw.GaussianNoise(kappa2=sw.GeneralKappa2(fn=k2 or _K2U()),
                                 sigma2=sw.CustomImpulse(fn=s2 or _S2U()))
        t_min = U_MIN
    return sw.System(field=sw.FieldSpec("phi", n_components=N),
                     linear=linear, noise=noise, t_min=t_min)


C_POINTS = [(0.1, 2.1, 0.6, 1.4), (0.0, 0.9, -0.3, 1.9),
            (0.2, 1.5, 0.2, 1.5)]


def _c_pair(u_system):
    """C at the C_POINTS in t, and from the u system divided by J J."""
    ct = PropagatorCache(_c_system("t").build_propagator_model(diag_C=False))
    cu = PropagatorCache(u_system.build_propagator_model(diag_C=False))
    for x, t1, y, t2 in C_POINTS:
        u1, u2 = u_of(t1), u_of(t2)
        yield (np.asarray(ct.C_value(x, t1, y, t2)),
               np.asarray(cu.C_value(x, u1, y, u2)) / (J(u1) * J(u2)))


def _max_rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(a)))


def test_TV1_C_from_kappa2_and_white_noise_in_u_equals_C_in_t():
    for a, b in _c_pair(_c_system("u")):
        assert np.all(np.abs(a) > 0)
        assert _max_rel(a, b) < 1e-10


def test_TV1_white_part_is_not_negligible():
    """The σ² term moves C by more than 1 per cent at every point, so TV1
    tests its factor."""
    no_white = sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(gamma=gamma_t, t_max_cache=3.0,
                            n_grid_cache=3000),
        noise=sw.GaussianNoise(kappa2=sw.GeneralKappa2(fn=k2_t)),
        t_min=T_MIN)
    cw = PropagatorCache(_c_system("t").build_propagator_model(diag_C=False))
    cn = PropagatorCache(no_white.build_propagator_model(diag_C=False))
    for x, t1, y, t2 in C_POINTS:
        assert _max_rel(np.asarray(cw.C_value(x, t1, y, t2)),
                        np.asarray(cn.C_value(x, t1, y, t2))) > 1e-2


# --------------------------------------------------------------------------- #
# TV2: diagrams, closed-form R and C
# --------------------------------------------------------------------------- #
def Gamma_t(t):
    return np.array([0.7 * t + 0.1 * t * t, 1.3 * t - 0.05 * t * t])


def R_t(t, s):
    if t < s:
        return np.zeros((N, N))
    return np.diag(np.exp(-(Gamma_t(t) - Gamma_t(s))))


class _RU:
    def __init__(self, left=J, right=J):
        self.left, self.right = left, right

    def __call__(self, u, v):
        return self.left(u) * R_t(t_of(u), t_of(v)) / self.right(v)


MC = np.array([[1.0, 0.35], [0.35, 0.8]])


def c_t(n1, t1, n2, t2):
    return (MC * (t1 - 0.1) * (t2 - 0.1)
            * np.exp(-(t1 - t2) ** 2 / 0.7 - 0.4 * abs(n1 - n2)))


def c_u(n1, u1, n2, u2):
    return J(u1) * J(u2) * c_t(n1, t_of(u1), n2, t_of(u2))


_rng = np.random.default_rng(11)
F = _rng.normal(size=(N, N, N))
G = _rng.normal(size=(N, N, N, N))
K3 = _rng.normal(size=(N, N, N))


def _space(n):
    """exp(-0.1 Σ n²) over the legs, per sample: shape (n_samples,)."""
    return np.exp(-0.1 * np.sum(np.asarray(n, dtype=float) ** 2, axis=0))


def _tensor(scale, rank):
    return scale.reshape((-1,) + (1,) * rank)


def k3_callable_t(n, t):
    """Raw κ³ at three leg times, no symmetry in the legs (vectorised)."""
    t = np.asarray(t, dtype=float)
    s = np.exp(-0.3 * t.sum(axis=0)) * (1 + 0.2 * t[0] - 0.1 * t[2])
    return _tensor(s * _space(n), 3) * K3


def k3_equal_time_t(n, t):
    s = 1.0 + 0.3 * np.sin(np.asarray(t, dtype=float)[0])
    return _tensor(s * _space(n), 3) * K3


def k3_r_contracted_t(n, t):
    return k3_callable_t(n, t)


class _LegFactor:
    """``Π_i J(u_i)^power`` times a vectorised κ³ evaluated at ``t(u)``;
    ``inner=None`` stands for the static ``K3``."""

    def __init__(self, inner, power):
        self.inner, self.power = inner, power

    def __call__(self, n, u):
        u = np.asarray(u, dtype=float)
        fac = np.prod(J(u) ** self.power, axis=0)
        if self.inner is None:
            return _tensor(fac, 3) * K3
        return _tensor(fac, 3) * self.inner(n, t_of(u))


class _EqualTimeU:
    def __init__(self, power=4):
        self.power = power

    def __call__(self, n, u):
        u = np.asarray(u, dtype=float)
        return _tensor(J(u[0]) ** self.power, 3) * k3_equal_time_t(n, t_of(u))


class _QuarticU:
    """The quartic local vertex in u: ``J^(1 + q - p) G`` with one ψ leg and
    three φ legs, ``J⁻¹ G``."""

    def __init__(self, power=-1):
        self.power = power

    def __call__(self, n, u):
        u0 = np.asarray(u, dtype=float)[0]
        return _tensor(J(u0) ** self.power, 4) * G


def _vertex_system(variable, kind, *, k_u=None, r_u=None, g_u=None):
    """``kind`` picks the non-local vertex; the cubic F is always there,
    the quartic G only for kind 'quartic'."""
    if variable == "t":
        linear = sw.ExplicitR(R_time=R_t, iso_R=False)
        k = {"raw_static": K3, "raw_callable": k3_callable_t,
             "equal_time": k3_equal_time_t,
             "r_contracted": k3_r_contracted_t}.get(kind, K3)
        g = G
        t_min = T_MIN
    else:
        linear = sw.ExplicitR(R_time=r_u or _RU(), iso_R=False)
        k = k_u or {"raw_static": _LegFactor(None, 2),
                    "raw_callable": _LegFactor(k3_callable_t, 2),
                    "equal_time": _EqualTimeU(),
                    "r_contracted": _LegFactor(k3_r_contracted_t, 1),
                    }.get(kind, _LegFactor(None, 2))
        g = g_u or _QuarticU()
        t_min = U_MIN
    vertices = [sw.LocalVertex("F", coupling=F)]
    if kind == "quartic":
        vertices.append(sw.LocalVertex(
            "G", coupling=g, rank=4,
            coupling_vectorized=variable == "u"))
    nonlocal_ = [sw.NonLocalVertex(
        "K", order=3, coupling=k,
        coupling_vectorized=callable(k),
        equal_time=kind == "equal_time",
        already_R_contracted=kind == "r_contracted")]
    return sw.System(
        field=sw.FieldSpec("phi", n_components=N), linear=linear,
        vertices=vertices, nonlocal_vertices=nonlocal_,
        noise=sw.GaussianNoise(kappa2=sw.GeneralKappa2(fn=_zero_k2)),
        t_min=t_min)


def _zero_k2(n1, t1, n2, t2):
    return np.zeros((N, N))


POS = {"x": 0.1, "y": 0.6, "z": -0.4}
OBS3 = ("phi_a(x)", "phi_b(y)", "phi_c(z)")
OBS2 = ("phi_a(x)", "phi_b(y)")


def _evaluate(system, variable, obs, order, comps, ext, vertex_types, n,
              c_fn=None):
    """The moment in t; for the u system the external factors 1/J are
    applied here, which is part of the recipe under test."""
    labels = [o[o.index("(") + 1:-1] for o in obs]
    if variable == "t":
        to_var, t_max, c_fn = (lambda t: t), 2.5, c_fn or c_t
    else:
        to_var, t_max, c_fn = u_of, float(u_of(2.5)), c_fn or c_u
    props = system.propagators(t_max=t_max, n_grid_t=12, c_closed_form=c_fn,
                               c_closed_form_only=True, diag_C=False)
    expansion = system.expand(obs, orders=[order], diag_R=False,
                              diag_C=False)
    kw, factor = {}, 1.0
    if ext == "all":
        kw["integrate_over"] = "all"
        t_final = float(to_var(T))
    else:
        times = {lab: float(to_var(t)) for lab, t in zip(labels, ext)}
        if variable == "u":
            factor = float(np.prod([1.0 / J(v) for v in times.values()]))
        kw["external_times"] = times
        t_final = max(times.values())
    value = expansion.evaluate(
        props, positions={k: POS[k] for k in labels}, t_final=t_final,
        component_pair=comps, orders=[order], vertex_types=vertex_types,
        method="gauss_legendre", n_gauss=n, **kw).total
    return float(np.real(value)) * factor


# (id, kind, observable, order, components, externals, vertex_types, nodes)
TV2_CASES = [
    ("F_order2_fixed", "raw_static", OBS2, 2, (1, 0), (2.1, 1.7), ["F"], 16),
    ("F_order2_integrated", "raw_static", OBS2, 2, (1, 0), "all", ["F"], 12),
    ("order0_fixed", "raw_static", OBS2, 0, (1, 0), (2.1, 1.7), None, 4),
    ("G_order1_fixed", "quartic", OBS2, 1, (0, 1), (2.1, 1.7), ["G"], 16),
    ("K_raw_static_equal_times", "raw_static", OBS3, 1, (0, 1, 1),
     (2.2, 2.2, 2.2), ["K"], 16),
    ("K_raw_static_integrated", "raw_static", OBS3, 1, (0, 1, 1), "all",
     ["K"], 8),
    ("K_raw_callable_equal_times", "raw_callable", OBS3, 1, (1, 0, 1),
     (2.2, 2.2, 2.2), ["K"], 16),
    ("K_equal_time_fixed", "equal_time", OBS3, 1, (0, 1, 1),
     (2.1, 1.6, 1.2), ["K"], 16),
    ("K_r_contracted_fixed", "r_contracted", OBS3, 1, (0, 1, 1),
     (2.1, 1.6, 1.2), ["K"], 16),
    ("FK_r_contracted_fixed", "r_contracted", OBS2, 2, (1, 0), (2.1, 1.7),
     None, 16),
    ("FK_r_contracted_integrated", "r_contracted", OBS2, 2, (1, 0), "all",
     None, 12),
]


@pytest.mark.parametrize(
    "kind, obs, order, comps, ext, vtypes, n",
    [c[1:] for c in TV2_CASES], ids=[c[0] for c in TV2_CASES])
def test_TV2_diagrams_in_u_equal_diagrams_in_t(kind, obs, order, comps, ext,
                                               vtypes, n):
    ref = _evaluate(_vertex_system("t", kind), "t", obs, order, comps, ext,
                    vtypes, n)
    got = _evaluate(_vertex_system("u", kind), "u", obs, order, comps, ext,
                    vtypes, n)
    assert abs(ref) > 1e-3
    assert got == pytest.approx(ref, rel=1e-10, abs=0.0)


# --------------------------------------------------------------------------- #
# TV3: a missing or wrong factor breaks the agreement
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("what", ["gamma_shift", "kappa2_power",
                                  "sigma2_power"])
def test_TV3_wrong_C_factor_is_detected(what):
    u_system = {
        "gamma_shift": lambda: _c_system("u", gamma=_GammaU(shift=False)),
        "kappa2_power": lambda: _c_system("u", k2=_K2U(power=1)),
        "sigma2_power": lambda: _c_system("u", s2=_S2U(power=2)),
    }[what]()
    worst = max(_max_rel(a, b) for a, b in _c_pair(u_system))
    assert worst > 1e-3


TV3_VERTEX_CASES = [
    ("R_without_right_J", "raw_static", dict(r_u=_RU(right=_ones)),
     OBS2, 2, (1, 0), (2.1, 1.7), ["F"], 8),
    ("raw_kappa_power_1", "raw_static", dict(k_u=_LegFactor(None, 1)),
     OBS3, 1, (0, 1, 1), (2.2, 2.2, 2.2), ["K"], 8),
    ("equal_time_power_3", "equal_time", dict(k_u=_EqualTimeU(power=3)),
     OBS3, 1, (0, 1, 1), (2.1, 1.6, 1.2), ["K"], 8),
    ("r_contracted_power_0", "r_contracted",
     dict(k_u=_LegFactor(k3_r_contracted_t, 0)),
     OBS3, 1, (0, 1, 1), (2.1, 1.6, 1.2), ["K"], 8),
    ("quartic_power_0", "quartic", dict(g_u=_QuarticU(power=0)),
     OBS2, 1, (0, 1), (2.1, 1.7), ["G"], 8),
]


@pytest.mark.parametrize(
    "kind, override, obs, order, comps, ext, vtypes, n",
    [c[1:] for c in TV3_VERTEX_CASES], ids=[c[0] for c in TV3_VERTEX_CASES])
def test_TV3_wrong_vertex_factor_is_detected(kind, override, obs, order,
                                             comps, ext, vtypes, n):
    ref = _evaluate(_vertex_system("t", kind), "t", obs, order, comps, ext,
                    vtypes, n)
    got = _evaluate(_vertex_system("u", kind, **override), "u", obs, order,
                    comps, ext, vtypes, n)
    assert abs(got / ref - 1.0) > 1e-3


def test_TV3_fixed_external_without_its_J_is_detected():
    ref = _evaluate(_vertex_system("t", "raw_static"), "t", OBS2, 0, (1, 0),
                    (2.1, 1.7), None, 4)
    got = _evaluate(_vertex_system("u", "raw_static"), "u", OBS2, 0, (1, 0),
                    (2.1, 1.7), None, 4)
    # _evaluate divides by J(u_x) J(u_y); undo it.
    undone = got * J(u_of(2.1)) * J(u_of(1.7))
    assert got == pytest.approx(ref, rel=1e-12, abs=0.0)
    assert abs(undone / ref - 1.0) > 1e-1


# --------------------------------------------------------------------------- #
# TV4: a steep time variable (Einstein-de Sitter lensing toy)
# --------------------------------------------------------------------------- #
# a = (1 - χ)², the affine parameter λ = [1 - (1 - χ)⁵] / 5, dλ/dχ = a²,
# D = a χ, the Sachs response R(λ, λ') = (D(λ') / D(λ))², a smooth C in χ,
# F = -1, the observable <Φ(x) Φ(y)> with Φ = ∫ φ dλ.  At z = 1100,
# dχ/dλ = (1 + z)² is 1.2e6 at the source.
Z_SOURCE = 1100.0
CHI_S = 1.0 - (1.0 + Z_SOURCE) ** -0.5
LAM_S = (1.0 - (1.0 - CHI_S) ** 5) / 5.0


def _chi_of_lam(lam):
    return 1.0 - np.maximum(1.0 - 5.0 * np.asarray(lam, dtype=float),
                            0.0) ** 0.2


def _jac(chi):
    return (1.0 - chi) ** 4


def _dbar(chi):
    return (1.0 - chi) ** 2 * chi


def _c_of_chi(x1, c1, x2, c2):
    u1, u2 = c1 / CHI_S, c2 / CHI_S
    h1, h2 = u1 * (1.0 - 0.5 * u1), u2 * (1.0 - 0.5 * u2)
    w = 0.3 * CHI_S
    return np.cos(x1 - x2) * h1 * h2 * np.exp(-0.5 * ((c1 - c2) / w) ** 2)


def _r_lam(t, s):
    if t < s:
        return 0.0
    return float((_dbar(_chi_of_lam(s)) / _dbar(_chi_of_lam(t))) ** 2)


def _c_lam(n1, t1, n2, t2):
    return np.array([[_c_of_chi(n1, _chi_of_lam(t1), n2, _chi_of_lam(t2))]])


def _r_chi(t, s):
    if t < s:
        return 0.0
    return float(_jac(t) * (_dbar(s) / _dbar(t)) ** 2 / _jac(s))


def _c_chi(n1, t1, n2, t2):
    return np.array([[_jac(t1) * _jac(t2) * _c_of_chi(n1, t1, n2, t2)]])


def _eds(variable, n):
    r_fn, c_fn, t_end = ((_r_lam, _c_lam, LAM_S) if variable == "lambda"
                         else (_r_chi, _c_chi, CHI_S))
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=1),
        linear=sw.ExplicitR(R_time=r_fn),
        vertices=[sw.LocalVertex("F", coupling=np.full((1, 1, 1), -1.0))],
        noise=sw.GaussianNoise(kappa2=sw.GeneralKappa2(
            fn=lambda n1, t1, n2, t2: np.zeros((1, 1)))))
    props = system.propagators(t_max=t_end, c_closed_form=c_fn,
                               c_closed_form_only=True, diag_C=False)
    return float(np.real(system.expand(("phi(x)", "phi(y)"), orders=[2],
                                       diag_C=False).evaluate(
        props, positions={"x": 0.3, "y": 1.1}, t_final=t_end,
        component_pair=(0, 0), orders=[2], integrate_over="all",
        method="gauss_legendre", n_gauss=n).total))


def test_TV4_steep_variable_needs_the_change_of_variable():
    chi_12, chi_16 = _eds("chi", 12), _eds("chi", 16)
    lam_16 = _eds("lambda", 16)
    assert abs(chi_12 / chi_16 - 1.0) < 1e-8
    assert abs(lam_16 / chi_16 - 1.0) > 0.1
