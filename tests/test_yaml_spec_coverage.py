r"""What the YAML layer can express, against the L1 specs it lowers to.

Four L1 specs had no YAML spelling: a :class:`CustomKernel` on any of the
three kernel axes (demo 3's spatial envelope had to be declared through the
general ``callable_module`` hatch, with a closed-form C alongside), a
matrix-valued :class:`ExplicitR` (refused with "set iso_R: true"), and a
matrix :class:`ConstantImpulse` (accepted, never checked, and undocumented).
Each is exercised here twice: once as YAML and once as the equivalent L1
``System``, which must give the same numbers to the last bit, and once
against a reference that shares no code with the package:

* a two-exponential temporal kernel: the defining double integral of C by
  ``scipy.integrate.dblquad`` on the two triangles;
* demo 3's spatial envelope: the Campbell closed form ``K_R`` of
  ``examples/demo3/shot_noise.py``, which is exact at ``F = 0``;
* a dense (non-normal) drift: ``C(T, T) = ∫ e^{A(T-s)} S e^{A'(T-s)} ds``
  by ``scipy.integrate.quad``, and the 3-point function
  ``K_ijk M_ai M_bj M_ck`` with ``M = A^{-1}(e^{A Δ} - I)``;
* a matrix white-noise amplitude: the same quadrature for the κ² part plus
  ``S_ab (1 - e^{-(γ_a+γ_b)Δ}) / (γ_a + γ_b)`` for the white part.

The rest of the file is input validation: every refusal below used to be a
silently ignored key, a silently wrong tensor, or an obscure failure much
later.
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.integrate import dblquad, quad
from scipy.linalg import expm, solve_continuous_lyapunov

import sft_wick as sw
from sft_wick.workflow import specs as sp
from sft_wick.workflow.closed_forms import builtin_closed_form_for
from sft_wick.workflow.config import (build_system, load_workflow_config,
                                      run_workflow)

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

ROOT = Path(__file__).resolve().parents[1]
DEMO3 = ROOT / "examples" / "demo3"
sys.path.insert(0, str(DEMO3))
import shot_noise as sn  # noqa: E402

T_MIN = 0.3
T_FINAL = 1.5
GAMMAS = [0.9, 1.4]            # component-dependent rates
S_MATRIX = [[0.30, 0.12], [0.12, 0.20]]
A_DENSE = [[-1.0, 0.4], [-0.3, -1.6]]   # non-normal: A A' != A' A


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


def _load(path: Path, attr: str):
    """Import a written module and take one attribute, so an L1 reference
    run uses the very object the YAML run loads."""
    spec = importlib.util.spec_from_file_location(f"_ref_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr)


def _run(tmp_path: Path, cfg: dict, name: str = "c.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg))
    _sweep, totals = run_workflow(load_workflow_config(path), progress=False)
    return totals


def _value(totals, comp, **coords) -> float:
    mask = np.ones(len(totals), dtype=bool)
    for col, c in zip("abcd", comp):
        mask &= (totals[col] == c).to_numpy()
    for col, v in coords.items():
        mask &= np.isclose(totals[col].to_numpy(dtype=float), v, atol=1e-12)
    assert mask.sum() == 1, (comp, coords)
    return float(totals.loc[mask, "value"].iloc[0])


def _ou_double_integral(kernel, g_a, g_b, t0=T_MIN, T=T_FINAL) -> float:
    """``∫∫_{t0}^{T} e^{-g_a (T-s1)} κ_t(s1-s2) e^{-g_b (T-s2)} ds1 ds2``,
    split on the diagonal where κ_t has its cusp."""
    def f(s2, s1):
        return (np.exp(-g_a * (T - s1)) * float(kernel(s1 - s2))
                * np.exp(-g_b * (T - s2)))

    below = dblquad(f, t0, T, lambda s1: t0, lambda s1: s1,
                    epsabs=1e-13, epsrel=1e-13)[0]
    above = dblquad(f, t0, T, lambda s1: s1, lambda s1: T,
                    epsabs=1e-13, epsrel=1e-13)[0]
    return below + above


def _base_config(**overrides) -> dict:
    """A small generic 2-point config: distinct rates, t_min != 0, the two
    points apart."""
    cfg = {
        "system": {
            "field": {"name": "phi", "n_components": 2},
            "linear": {"type": "diagonal", "gamma": list(GAMMAS)},
            "t_min": T_MIN,
            "noise": {"kappa2": {
                "type": "separable_translation",
                "temporal": {"type": "exponential", "lam": 0.25,
                             "sigma_t": 0.45},
                "spatial": {"type": "gaussian", "sigma_x": 0.8}}},
        },
        "expand": {"observable": ["phi_a(x)", "phi_b(y)"], "orders": [0]},
        "propagators": {"t_max": T_FINAL, "n_grid_t": 8},
        "sweep": {"positions_grid": {"x": [0.0], "y": [0.45]},
                  "t_final_grid": [T_FINAL],
                  "component_tuples": [[0, 0], [0, 1], [1, 1]],
                  "method": "gauss_legendre", "n_gauss": 4},
    }
    for dotted, value in overrides.items():
        node = cfg
        *path, leaf = dotted.split("__")
        for key in path:
            node = node[key]
        node[leaf] = value
    return cfg


# ---------------------------------------------------------------------------
# YC1 -- a custom temporal kernel
# ---------------------------------------------------------------------------

_TEMPORAL_MODULE = '''
    from dataclasses import dataclass

    import numpy as np


    @dataclass(frozen=True)
    class TwoExponentials:
        """kappa_t(dt) = l1 e^{-|dt|/tau1} + l2 e^{-|dt|/tau2}.

        Not a member of the package's exponential family, so this kernel
        has no YAML spelling other than 'custom'.  A frozen dataclass keeps
        the repr stable, which the caches key on.
        """

        lam1: float
        tau1: float
        lam2: float
        tau2: float

        def __call__(self, dt):
            d = abs(float(dt))
            return (self.lam1 * np.exp(-d / self.tau1)
                    + self.lam2 * np.exp(-d / self.tau2))


    temporal = TwoExponentials(0.35, 0.4, 0.15, 1.3)
'''


def _custom_temporal_config(module: Path) -> dict:
    return _base_config(system__noise__kappa2__temporal={
        "type": "custom", "module": str(module), "attr": "temporal"})


def test_YC1_custom_temporal_kernel_matches_L1_and_quadrature(tmp_path):
    module = _write(tmp_path, "yc_temporal.py", _TEMPORAL_MODULE)
    kernel = _load(module, "temporal")
    totals = _run(tmp_path, _custom_temporal_config(module))

    # Same physics through L1, with the very same kernel object.
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=list(GAMMAS)), t_min=T_MIN,
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sp.CustomKernel(fn=kernel),
            spatial=sw.GaussianSpatial(sigma_x=0.8))))
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0])
    props = system.propagators(t_max=T_FINAL, n_grid_t=8, progress=False)
    l1 = exp.sweep(props, positions_grid={"x": [0.0], "y": [0.45]},
                   t_final_grid=[T_FINAL],
                   component_tuples=[[0, 0], [0, 1], [1, 1]],
                   method="gauss_legendre", n_gauss=4).totals()
    np.testing.assert_array_equal(totals["value"].to_numpy(),
                                  l1["value"].to_numpy())

    # C_ab(T, T) = delta_ab kappa_x(r) * the double integral of the kernel.
    kappa_x = float(np.exp(-0.45 ** 2 / (2 * 0.8 ** 2)))
    for a in (0, 1):
        ref = kappa_x * _ou_double_integral(kernel, GAMMAS[a], GAMMAS[a])
        assert _value(totals, (a, a)) == pytest.approx(ref, rel=1e-6,
                                                       abs=0.0)
    assert _value(totals, (0, 1)) == 0.0


def test_YC1_custom_kernel_is_not_the_exponential_family(tmp_path):
    """The check above would pass for any kernel if the two agreed: the
    package's own exponential kernel gives a different C."""
    module = _write(tmp_path, "yc_temporal.py", _TEMPORAL_MODULE)
    kernel = _load(module, "temporal")
    custom = _ou_double_integral(kernel, GAMMAS[0], GAMMAS[0])
    single = _ou_double_integral(sw.ExponentialTemporal(lam=0.5, sigma_t=0.4),
                                 GAMMAS[0], GAMMAS[0])
    assert abs(custom - single) / abs(custom) > 1e-2


# ---------------------------------------------------------------------------
# YC2 -- a custom spatial kernel: demo 3's envelope, level A from YAML
# ---------------------------------------------------------------------------

_ENVELOPE_MODULE = f'''
    import sys

    sys.path.insert(0, {str(DEMO3)!r})

    from shot_noise import PARAMS, Kappa2Spatial, RContractedCoupling

    spatial = Kappa2Spatial(params=PARAMS)
    k3 = RContractedCoupling(m=3, params=PARAMS)
'''

_DEMO3_POS = {"x": 0.0, "y": 0.7, "z": -0.4}
_DEMO3_T = 2.5


def _demo3_config(module: Path, triples) -> dict:
    p = sn.PARAMS
    return {
        "system": {
            "field": {"name": "phi", "n_components": p.n_components},
            "linear": {"type": "diagonal", "gamma": [p.gamma] * 2},
            "nonlocal_vertices": [{
                "name": "K3", "order": 3, "coupling_module": str(module),
                "coupling_attr": "k3", "coupling_vectorized": True,
                "already_R_contracted": True}],
            "noise": {"kappa2": {
                "type": "separable_translation",
                "temporal": {"type": "exponential",
                             "lam": float(sn.kappa2_lam(p)),
                             "sigma_t": p.sigma_t},
                "spatial": {"type": "custom", "module": str(module),
                            "attr": "spatial"}}},
        },
        "expand": {"observable": ["phi_a(x)", "phi_b(y)", "phi_c(z)"],
                   "orders": [1]},
        "propagators": {"t_max": 4.0, "n_grid_t": 20,
                        "c_closed_form": "auto",
                        "c_closed_form_only": True},
        "sweep": {"positions_grid": {k: [v] for k, v in _DEMO3_POS.items()},
                  "t_final_grid": [_DEMO3_T],
                  "component_tuples": [list(t) for t in triples],
                  "method": "gauss_legendre", "n_gauss": 8},
    }


def test_YC2_custom_spatial_kernel_runs_demo3_level_a_from_yaml(tmp_path):
    """Demo 3's envelope is ``σ_x (1 + r/σ_x) e^{-r/σ_x}``, neither
    exponential nor Gaussian; its configs had to declare κ² through the
    general callable hatch and supply C in closed form.  As a
    ``SeparableTranslation`` with a custom spatial kernel the built-in
    closed form applies, so no quadrature runs."""
    module = _write(tmp_path, "yc_envelope.py", _ENVELOPE_MODULE)
    triples = [(0, 0, 0), (1, 1, 1), (0, 1, 1)]
    cfg = _demo3_config(module, triples)
    system = build_system(load_workflow_config(
        _write(tmp_path, "probe.yaml", yaml.safe_dump(cfg))).system)
    assert builtin_closed_form_for(system) is not None

    totals = _run(tmp_path, cfg)
    xs = np.array([_DEMO3_POS[k] for k in "xyz"])
    ref = float(sn.K_R(xs, np.full(3, _DEMO3_T), sn.PARAMS)[0])
    for comp in [(0, 0, 0), (1, 1, 1)]:
        got = _value(totals, comp)
        assert got == pytest.approx(ref, rel=1e-12, abs=0.0), comp
    # The cumulant is delta_{abc}, so a mixed triple vanishes identically.
    assert _value(totals, (0, 1, 1)) == 0.0

    # The same system through L1.
    import system as demo3_system  # noqa: PLC0415 -- examples/demo3
    l1_system = demo3_system.make_system(sn.PARAMS, cumulants=(3,))
    exp = l1_system.expand(("phi_a(x)", "phi_b(y)", "phi_c(z)"), orders=[1])
    props = l1_system.propagators(t_max=4.0, n_grid_t=20,
                                  c_closed_form="auto",
                                  c_closed_form_only=True, progress=False)
    l1 = exp.sweep(props, positions_grid={k: [v] for k, v in
                                          _DEMO3_POS.items()},
                   t_final_grid=[_DEMO3_T],
                   component_tuples=[list(t) for t in triples],
                   method="gauss_legendre", n_gauss=8).totals()
    np.testing.assert_array_equal(totals["value"].to_numpy(),
                                  l1["value"].to_numpy())


# ---------------------------------------------------------------------------
# YC3 -- a custom angular kernel
# ---------------------------------------------------------------------------

_ANGULAR_MODULE = '''
    from dataclasses import dataclass


    @dataclass(frozen=True)
    class LegendreSum:
        """C_0 + C_1 P_1 + C_2 P_2, written out."""

        c0: float
        c1: float
        c2: float

        def __call__(self, cos_theta):
            x = float(cos_theta)
            return self.c0 + self.c1 * x + self.c2 * 0.5 * (3.0 * x * x - 1.0)


    angular = LegendreSum(1.0, 0.4, 0.25)
'''

_DIRECTIONS = {"x": [[1.0, 0.0]], "y": [[0.6, 0.8]]}


def _rotation_config(angular: dict) -> dict:
    cfg = _base_config()
    cfg["system"]["noise"]["kappa2"] = {
        "type": "separable_rotation",
        "temporal": {"type": "exponential", "lam": 0.25, "sigma_t": 0.45},
        "angular": angular}
    cfg["sweep"]["positions_grid"] = _DIRECTIONS
    cfg["sweep"]["component_tuples"] = [[0, 0], [1, 1]]
    return cfg


def test_YC3_custom_angular_kernel_matches_legendre_and_L1(tmp_path):
    module = _write(tmp_path, "yc_angular.py", _ANGULAR_MODULE)
    kernel = _load(module, "angular")
    custom = _run(tmp_path, _rotation_config(
        {"type": "custom", "module": str(module), "attr": "angular"}))
    legendre = _run(tmp_path, _rotation_config(
        {"type": "legendre", "coeffs": [1.0, 0.4, 0.25]}), name="leg.yaml")
    np.testing.assert_allclose(custom["value"].to_numpy(),
                               legendre["value"].to_numpy(), rtol=1e-6)

    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=list(GAMMAS)), t_min=T_MIN,
        noise=sw.GaussianNoise(kappa2=sw.SeparableRotation(
            temporal=sw.ExponentialTemporal(lam=0.25, sigma_t=0.45),
            angular=sp.CustomKernel(fn=kernel))))
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0])
    props = system.propagators(t_max=T_FINAL, n_grid_t=8, progress=False)
    l1 = exp.sweep(props, positions_grid=_DIRECTIONS,
                   t_final_grid=[T_FINAL], component_tuples=[[0, 0], [1, 1]],
                   method="gauss_legendre", n_gauss=4).totals()
    np.testing.assert_array_equal(custom["value"].to_numpy(),
                                  l1["value"].to_numpy())


# ---------------------------------------------------------------------------
# YC4 -- a matrix-valued explicit R
# ---------------------------------------------------------------------------

_DENSE_R_MODULE = f'''
    from dataclasses import dataclass

    import numpy as np
    from scipy.linalg import expm, solve_continuous_lyapunov

    A = np.array({A_DENSE!r})
    S = np.array({S_MATRIX!r})
    T_MIN = {T_MIN!r}
    X = solve_continuous_lyapunov(A, -S)


    @dataclass(frozen=True)
    class DenseR:
        """R(t1, t2) = Theta(t1 - t2) exp(A (t1 - t2)), a dense matrix."""

        def __call__(self, t1, t2):
            if t1 < t2:
                return np.zeros(A.shape)
            return expm(A * (float(t1) - float(t2)))


    @dataclass(frozen=True)
    class LyapunovC:
        """C of the white noise S: C(t, t) = X - e^{{A d}} X e^{{A' d}} with
        A X + X A' = -S and d = t - T_MIN, continued to t1 != t2 by
        C(t1, t2) = e^{{A (t1-m)}} C(m, m) e^{{A' (t2-m)}}, m = min(t1, t2)."""

        def __call__(self, n1, t1, n2, t2):
            t1, t2 = float(t1), float(t2)
            m = max(min(t1, t2), T_MIN)
            d = m - T_MIN
            Cm = X - expm(A * d) @ X @ expm(A.T * d)
            return expm(A * (t1 - m)) @ Cm @ expm(A.T * (t2 - m))


    R_time = DenseR()
    C_fn = LyapunovC()
'''

_SCALAR_R_MODULE = '''
    import numpy as np


    def R_time(t1, t2):
        return 0.0 if t1 < t2 else float(np.exp(-(t1 - t2)))
'''

_ACAUSAL_R_MODULE = '''
    import numpy as np


    def R_time(t1, t2):
        return float(np.exp(-abs(t1 - t2)))     # the Heaviside is missing
'''


def _dense_config(module: Path) -> dict:
    cfg = _base_config()
    cfg["system"]["linear"] = {"type": "explicit", "iso_R": False,
                               "R_time_module": str(module)}
    cfg["system"]["noise"] = {
        "kappa2": {"type": "separable_translation",
                   "temporal": {"type": "exponential", "lam": 0.0,
                                "sigma_t": 1.0},
                   "spatial": {"type": "exponential", "sigma_x": 1.0}},
        "sigma2": {"type": "constant", "amplitude": S_MATRIX}}
    cfg["expand"]["diag_R"] = False
    cfg["propagators"] = {"t_max": T_FINAL, "n_grid_t": 8,
                          "c_closed_form_module": str(module),
                          "c_closed_form_attr": "C_fn",
                          "c_closed_form_only": True, "diag_C": False}
    cfg["sweep"]["method"] = "nquad"
    cfg["sweep"].pop("n_gauss")
    return cfg


def _white_noise_C(T=T_FINAL, A=None) -> np.ndarray:
    """``∫_{T_MIN}^{T} e^{A(T-s)} S e^{A'(T-s)} ds`` by adaptive quadrature
    (no Lyapunov identity, no package code)."""
    A = np.asarray(A_DENSE if A is None else A, float)
    S = np.asarray(S_MATRIX, float)
    out = np.zeros((2, 2))
    for a, b in itertools.product(range(2), repeat=2):
        out[a, b] = quad(
            lambda s: (expm(A * (T - s)) @ S @ expm(A.T * (T - s)))[a, b],
            T_MIN, T, epsabs=1e-13, epsrel=1e-13)[0]
    return out


def test_YC4_matrix_explicit_R_matches_L1_and_quadrature(tmp_path):
    """A dense drift used to be refused by the YAML layer ("currently
    supports only scalar R_time callables")."""
    module = _write(tmp_path, "yc_dense_r.py", _DENSE_R_MODULE)
    totals = _run(tmp_path, _dense_config(module))
    ref = _white_noise_C()
    for a, b in itertools.product(range(2), repeat=2):
        if (a, b) == (1, 0):
            continue                     # not in the config's tuples
        assert _value(totals, (a, b)) == pytest.approx(ref[a, b], rel=1e-9,
                                                       abs=0.0), (a, b)
    # The off-diagonal entries of A matter: dropping them moves C.
    diagonal_only = _white_noise_C(A=np.diag(np.diag(A_DENSE)))
    assert np.max(np.abs(diagonal_only - ref) / np.abs(ref)) > 1e-2, (
        "a diagonal drift would give the same C, so this is not a test of "
        "the matrix R")

    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.ExplicitR(R_time=_load(module, "R_time"), iso_R=False),
        t_min=T_MIN,
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=1.0),
                spatial=sw.ExponentialSpatial(sigma_x=1.0)),
            sigma2=sw.ConstantImpulse(np.asarray(S_MATRIX, float))))
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0], diag_R=False,
                        diag_C=False)
    props = system.propagators(t_max=T_FINAL, n_grid_t=8,
                               c_closed_form=_load(module, "C_fn"),
                               c_closed_form_only=True, diag_C=False,
                               progress=False)
    l1 = exp.sweep(props, positions_grid={"x": [0.0], "y": [0.45]},
                   t_final_grid=[T_FINAL],
                   component_tuples=[[0, 0], [0, 1], [1, 1]],
                   method="nquad").totals()
    np.testing.assert_array_equal(totals["value"].to_numpy(),
                                  l1["value"].to_numpy())


def test_YC4_matrix_explicit_R_three_point_function(tmp_path):
    """The 3-point function of a static symmetric κ³ under the dense R:
    ``K_ijk M_ai M_bj M_ck`` with ``M = A^{-1}(e^{A d} - I)``."""
    module = _write(tmp_path, "yc_dense_r.py", _DENSE_R_MODULE)
    rng = np.random.default_rng(11)
    K = rng.normal(size=(2, 2, 2))
    K = sum(np.transpose(K, s) for s in itertools.permutations(range(3))) / 6
    triples = [(0, 1, 1), (1, 0, 0), (1, 1, 1)]
    cfg = _dense_config(module)
    cfg["system"]["nonlocal_vertices"] = [
        {"name": "K", "order": 3, "coupling": K.tolist()}]
    cfg["expand"]["observable"] = ["phi_a(x)", "phi_b(y)", "phi_c(z)"]
    cfg["expand"]["orders"] = [1]
    cfg["sweep"]["positions_grid"] = {"x": [0.0], "y": [0.45], "z": [-0.3]}
    cfg["sweep"]["component_tuples"] = [list(t) for t in triples]
    totals = _run(tmp_path, cfg)

    A = np.asarray(A_DENSE, float)
    M = np.linalg.solve(A, expm(A * (T_FINAL - T_MIN)) - np.eye(2))
    ref = np.einsum("ijk,ai,bj,ck->abc", K, M, M, M)
    for comp in triples:
        assert _value(totals, comp) == pytest.approx(ref[comp], rel=1e-9,
                                                     abs=0.0), comp


@pytest.mark.parametrize("module_src,iso_R,match", [
    (_SCALAR_R_MODULE, False, r"has shape \(\), but iso_R: false"),
    (_DENSE_R_MODULE, True, r"has shape \(2, 2\), but iso_R: true"),
    (_ACAUSAL_R_MODULE, True, "causal"),
])
def test_YC4_explicit_R_is_probed_for_shape_and_causality(
        tmp_path, module_src, iso_R, match):
    module = _write(tmp_path, "yc_probe_r.py", module_src)
    cfg = _base_config()
    cfg["system"]["linear"] = {"type": "explicit", "iso_R": iso_R,
                               "R_time_module": str(module)}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match=match):
        load_workflow_config(path)


# ---------------------------------------------------------------------------
# YC5 -- a matrix white-noise amplitude
# ---------------------------------------------------------------------------

def _sigma2_config(amplitude) -> dict:
    cfg = _base_config()
    cfg["system"]["noise"]["sigma2"] = {"type": "constant",
                                        "amplitude": amplitude}
    cfg["propagators"].update({"c_closed_form": "auto",
                               "c_closed_form_only": True, "diag_C": False})
    return cfg


def test_YC5_matrix_constant_impulse_matches_L1_and_the_closed_forms(tmp_path):
    totals = _run(tmp_path, _sigma2_config(S_MATRIX))

    kernel = sw.ExponentialTemporal(lam=0.25, sigma_t=0.45)
    kappa_x = float(np.exp(-0.45 ** 2 / (2 * 0.8 ** 2)))
    S = np.asarray(S_MATRIX, float)
    for a, b in [(0, 0), (0, 1), (1, 1)]:
        white = S[a, b] * (1.0 - np.exp(-(GAMMAS[a] + GAMMAS[b])
                                        * (T_FINAL - T_MIN))) / (
            GAMMAS[a] + GAMMAS[b])
        coloured = (kappa_x * _ou_double_integral(kernel, GAMMAS[a], GAMMAS[b])
                    if a == b else 0.0)
        assert _value(totals, (a, b)) == pytest.approx(white + coloured,
                                                       rel=1e-9, abs=0.0)
    # The off-diagonal entry is carried by sigma2 alone: a diagonal
    # amplitude would leave it 0.
    assert abs(_value(totals, (0, 1))) > 1e-3

    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=list(GAMMAS)), t_min=T_MIN,
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=kernel, spatial=sw.GaussianSpatial(sigma_x=0.8)),
            sigma2=sw.ConstantImpulse(np.asarray(S_MATRIX, float))))
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0], diag_C=False)
    props = system.propagators(t_max=T_FINAL, n_grid_t=8,
                               c_closed_form="auto", c_closed_form_only=True,
                               diag_C=False, progress=False)
    l1 = exp.sweep(props, positions_grid={"x": [0.0], "y": [0.45]},
                   t_final_grid=[T_FINAL],
                   component_tuples=[[0, 0], [0, 1], [1, 1]],
                   method="gauss_legendre", n_gauss=4).totals()
    np.testing.assert_array_equal(totals["value"].to_numpy(),
                                  l1["value"].to_numpy())


@pytest.mark.parametrize("amplitude,match", [
    ([0.3, 0.2], r"shape \(2,\)"),
    ([[0.3, 0.1, 0.0], [0.1, 0.2, 0.0], [0.0, 0.0, 0.1]], r"shape \(3, 3\)"),
    ([[0.30, 0.12], [0.05, 0.20]], "symmetric"),
    ([[0.30, float("inf")], [float("inf"), 0.20]], "non-finite"),
    ("0.3x", "must be a number"),
])
def test_YC5_bad_amplitude_is_refused(tmp_path, amplitude, match):
    with pytest.raises(ValueError, match=match):
        _run(tmp_path, _sigma2_config(amplitude))


def test_YC5_constant_sigma2_requires_an_amplitude(tmp_path):
    cfg = _base_config()
    cfg["system"]["noise"]["sigma2"] = {"type": "constant"}
    with pytest.raises(ValueError, match="requires 'amplitude'"):
        _run(tmp_path, cfg)


def test_YC5_scalar_amplitude_still_works(tmp_path):
    """The isotropic spelling is unchanged: amplitude * I_N."""
    totals = _run(tmp_path, _sigma2_config(0.2))
    S = 0.2 * np.eye(2)
    kernel = sw.ExponentialTemporal(lam=0.25, sigma_t=0.45)
    kappa_x = float(np.exp(-0.45 ** 2 / (2 * 0.8 ** 2)))
    for a in (0, 1):
        white = S[a, a] * (1.0 - np.exp(-2 * GAMMAS[a] * (T_FINAL - T_MIN))
                           ) / (2 * GAMMAS[a])
        coloured = kappa_x * _ou_double_integral(kernel, GAMMAS[a], GAMMAS[a])
        assert _value(totals, (a, a)) == pytest.approx(white + coloured,
                                                       rel=1e-9, abs=0.0)
    assert _value(totals, (0, 1)) == 0.0


# ---------------------------------------------------------------------------
# YC6 -- vertex blocks
# ---------------------------------------------------------------------------

_F_TENSOR = [[[0.0, 0.0], [0.0, 0.7]], [[0.0, -0.4], [-0.4, 0.0]]]


def _vertex_config(vertices=None, nonlocal_vertices=None) -> dict:
    cfg = _base_config()
    if vertices is not None:
        cfg["system"]["vertices"] = vertices
    if nonlocal_vertices is not None:
        cfg["system"]["nonlocal_vertices"] = nonlocal_vertices
    return cfg


@pytest.mark.parametrize("vertices,nonlocal_vertices,match", [
    ([{"coupling": _F_TENSOR}], None, "'name'"),
    ([{"name": "F"}], None, "exactly one of"),
    ([{"name": "F", "coupling": _F_TENSOR, "coupling_path": "f.npy"}], None,
     "exactly one of"),
    ([{"name": "F", "coupling": _F_TENSOR, "coupling_attr": "fn"}], None,
     "coupling_attr"),
    ([{"name": "F", "coupling": _F_TENSOR}],
     [{"name": "F", "order": 3, "coupling": np.zeros((2, 2, 2)).tolist()}],
     "used more than once"),
    (None, [{"name": "K", "coupling": np.zeros((2, 2, 2)).tolist()}],
     "'order'"),
    (None, [{"name": "K", "order": 0,
             "coupling": np.zeros((2, 2, 2)).tolist()}], "'order'"),
    (None, [{"name": "K", "order": 3,
             "coupling": np.zeros((2, 2)).tolist()}], r"shape \(2, 2\)"),
    ([{"name": "F", "coupling": np.zeros((3, 3, 3)).tolist()}], None,
     r"shape \(3, 3, 3\)"),
    ([{"name": "F", "coupling": _F_TENSOR, "already_R_contracted": True}],
     None, "unknown key"),
    ([{"name": "F", "coupling": _F_TENSOR, "coupling_vectorised": True}],
     None, "unknown key"),
    (None, [{"name": "K", "order": 3, "coupling_vectorized": True,
             "coupling": np.zeros((2, 2, 2)).tolist()}],
     "applies to a callable coupling"),
    (None, [{"name": "K", "order": 3, "already_R_contracted": "maybe",
             "coupling": np.zeros((2, 2, 2)).tolist()}],
     "must be true or false"),
])
def test_YC6_bad_vertex_block_is_refused(tmp_path, vertices,
                                         nonlocal_vertices, match):
    cfg = _vertex_config(vertices, nonlocal_vertices)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match=match):
        load_workflow_config(path)


def test_YC6_quoted_false_is_read_as_false(tmp_path):
    """``bool("false")`` is True, so a quoted flag used to switch the
    R-contracted interpretation on and change the value."""
    cfg = _vertex_config(None, [{
        "name": "K", "order": 3, "coupling": np.zeros((2, 2, 2)).tolist(),
        "already_R_contracted": "false", "equal_time": "no"}])
    system = build_system(load_workflow_config(
        _write(tmp_path, "c.yaml", yaml.safe_dump(cfg))).system)
    vertex = system.nonlocal_vertices[0]
    assert vertex.already_R_contracted is False
    assert vertex.equal_time is False


def test_YC6_every_vertex_field_is_reachable_from_yaml(tmp_path):
    """The vertex blocks forward every dataclass field of the spec, so no
    field of LocalVertex / NonLocalVertex is out of reach."""
    import dataclasses

    cfg = _vertex_config(
        [{"name": "F", "coupling": _F_TENSOR}],
        [{"name": "K", "order": 3, "coupling": np.zeros((2, 2, 2)).tolist(),
          "equal_time": True}])
    system = build_system(load_workflow_config(
        _write(tmp_path, "c.yaml", yaml.safe_dump(cfg))).system)
    assert system.nonlocal_vertices[0].equal_time is True
    from sft_wick.workflow.config import _vertex_keys
    for cls in (sp.LocalVertex, sp.NonLocalVertex):
        assert {f.name for f in dataclasses.fields(cls)} <= _vertex_keys(cls)


# ---------------------------------------------------------------------------
# YC7 -- kernel blocks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("temporal,match", [
    ({"type": "exponential", "lam": 0.2, "sigma_x": 1.0}, "unknown key"),
    ({"type": "exponential", "lam": 0.2}, r"requires \['sigma_t'\]"),
    ({"type": "exponential", "lam": 0.2, "sigma_t": 0.0}, "must be positive"),
    ({"type": "exponential", "lam": "x", "sigma_t": 0.4}, "must be a number"),
    ({"type": "ornstein"}, "Unsupported time-kernel type"),
    ({"type": "custom", "attr": "temporal"}, "requires 'module"),
])
def test_YC7_bad_kernel_block_is_refused(tmp_path, temporal, match):
    cfg = _base_config(system__noise__kappa2__temporal=temporal)
    with pytest.raises(ValueError, match=match):
        _run(tmp_path, cfg)


def test_YC7_scientific_notation_without_a_dot_is_read_as_a_number(tmp_path):
    """PyYAML reads ``1e-3`` as a string; it used to reach the kernel and
    fail inside the C quadrature."""
    cfg = _base_config()
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg).replace("lam: 0.25", "lam: 1e-3"))
    system = build_system(load_workflow_config(path).system)
    assert float(system.noise.kappa2.temporal.lam) == pytest.approx(1e-3)


def test_YC7_unsupported_type_names_the_custom_escape(tmp_path):
    cfg = _base_config(system__noise__kappa2__spatial={"type": "lorentzian",
                                                       "sigma_x": 1.0})
    with pytest.raises(ValueError, match="custom"):
        _run(tmp_path, cfg)


# ---------------------------------------------------------------------------
# YC8 / YC9 -- overrides into lists, and an unseeded sweep
# ---------------------------------------------------------------------------

def test_YC8_override_reaches_a_list_entry(tmp_path):
    cfg = _base_config()
    cfg["output"] = [{"type": "table", "format": "csv", "path": "out.csv"},
                     {"type": "npz", "path": "out.npz"}]
    cfg["system"]["vertices"] = [{"name": "F", "coupling": _F_TENSOR}]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    loaded = load_workflow_config(path, overrides={
        "output[0].path": "patched.csv",
        "system.vertices[0].name": "G",
        "system.linear.gamma": 1.25,
    })
    assert loaded.output[0].path == "patched.csv"
    assert loaded.system.vertices[0]["name"] == "G"
    assert loaded.system.linear["gamma"] == [1.25]


@pytest.mark.parametrize("key", ["output[2].path", "output[0].missing",
                                 "system[0].linear"])
def test_YC8_override_of_a_missing_entry_raises(tmp_path, key):
    cfg = _base_config()
    cfg["output"] = [{"type": "table", "format": "csv", "path": "out.csv"}]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(KeyError, match="does not exist in config"):
        load_workflow_config(path, overrides={key: "x"})


def test_YC9_seed_null_runs_unseeded(tmp_path):
    cfg = _base_config()
    cfg["sweep"].update({"method": "qmc_vectorized", "n_samples": 64,
                         "seed": None})
    cfg["sweep"].pop("n_gauss")
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    loaded = load_workflow_config(path)
    assert loaded.sweep.seed is None
    _sweep, totals = run_workflow(loaded, progress=False)
    assert np.all(np.isfinite(totals["value"].to_numpy()))
