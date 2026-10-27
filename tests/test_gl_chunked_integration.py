"""Chunked GL acceptance against elementary integrals and independent ODEs.

These references do not enumerate GL nodes or reproduce the causal map.
Changing chunk size may regroup floating-point additions, so comparisons
use explicit error budgets rather than requiring bit equality for GL.
"""

from dataclasses import dataclass
from math import factorial

import numpy as np
import pytest
import yaml

import sft_wick as sw
from sft_wick.evaluate import integrate_diagrams, integrate_moment
from sft_wick.workflow.config import load_workflow_config, run_workflow
from tests import test_matrix_r_batched as mb

T_MIN = 0.2
K_RAW = np.array([[[0.7, -0.3], [-0.3, 0.5]], [[-0.3, 0.5], [0.5, -0.8]]])
OBS3 = ("phi_a(x)", "phi_b(y)", "phi_c(z)")
POS3 = {"x": 0.0, "y": 0.3, "z": -0.4}


@dataclass
class ZeroC:
    n_components: int
    has_diagonal_kink = False

    def __call__(self, x1, t1, x2, t2):
        shape = np.broadcast(np.asarray(t1), np.asarray(t2)).shape
        return np.zeros(shape + (self.n_components, self.n_components))


def _noise():
    return sw.GaussianNoise(
        kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=0.5),
            spatial=sw.ExponentialSpatial(sigma_x=1.0),
        )
    )


def _cumulant(
    coupling,
    *,
    n=2,
    gamma=None,
    linear=None,
    vectorized=False,
    equal_time=False,
    absorbed=False,
):
    if linear is None:
        linear = sw.DiagonalA(gamma=[0.0] * n if gamma is None else list(gamma))
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=n),
        linear=linear,
        nonlocal_vertices=[
            sw.NonLocalVertex(
                "K",
                order=3,
                coupling=coupling,
                coupling_vectorized=vectorized,
                equal_time=equal_time,
                already_R_contracted=absorbed,
            )
        ],
        noise=_noise(),
        t_min=T_MIN,
    )
    obs = OBS3 if n > 1 else ("phi(x)", "phi(y)", "phi(z)")
    exp = system.expand(obs, orders=[1])
    props = system.propagators(
        t_max=2.0,
        c_closed_form=ZeroC(n),
        c_closed_form_only=True,
        c_closed_form_vectorized=True,
        progress=False,
    )
    return exp, props


def _eval3(
    exp,
    props,
    chunk,
    *,
    t_final=1.4,
    comps=(0, 1, 1),
    n_gauss=8,
    method="gauss_legendre",
    **kw,
):
    return exp.evaluate(
        props,
        positions=POS3,
        t_final=t_final,
        component_pair=comps,
        orders=[1],
        method=method,
        n_gauss=n_gauss,
        gl_chunk_size=chunk,
        **kw,
    )


@pytest.mark.parametrize("chunk", [1, 3, 7, 63, 64, 65, np.int64(11)])
def test_static_rectangle_weights_tail_and_exponentials(chunk):
    rates = np.array([0.7, 1.6])
    exp, props = _cumulant(K_RAW.copy(), gamma=rates)
    times = {"x": 0.8, "y": 1.1, "z": 1.4}
    comps = (1, 0, 1)
    # Exact cumulant of phi=M xi with constant random source xi.
    M = {
        label: -np.expm1(-rates * (time - T_MIN)) / rates
        for label, time in times.items()
    }
    ref = K_RAW[comps] * np.prod([M[label][c] for label, c in zip("xyz", comps)])
    eager = _eval3(exp, props, None, comps=comps, external_times=times)
    chunked = _eval3(exp, props, chunk, comps=comps, external_times=times)
    assert chunked.total == pytest.approx(ref, rel=2e-12, abs=0.0)
    assert chunked.total == pytest.approx(eager.total, rel=2e-14, abs=0.0)
    assert [r["error"] for r in chunked.per_diagram] == [0.0]


@dataclass
class PolynomialKappa:
    calls: list

    def __call__(self, n_array, t_array):
        t = np.asarray(t_array) - T_MIN
        self.calls.append(t.shape[1])
        powers = np.array([1, 2])
        fs = [t[leg, :, None] ** powers[None, :] for leg in range(3)]
        return (
            K_RAW[None]
            * fs[0][:, :, None, None]
            * fs[1][:, None, :, None]
            * fs[2][:, None, None, :]
        )


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 24, 26, 27, 28])
def test_component_polynomials_integrate_exactly_across_tail_blocks(chunk):
    calls = []
    exp, props = _cumulant(PolynomialKappa(calls), vectorized=True)
    times = {"x": 0.8, "y": 1.1, "z": 1.4}
    comps = (0, 1, 0)
    powers = [1, 2]
    ref = K_RAW[comps] * np.prod(
        [
            (times[label] - T_MIN) ** (powers[c] + 1) / (powers[c] + 1)
            for label, c in zip("xyz", comps)
        ]
    )
    got = _eval3(exp, props, chunk, comps=comps, external_times=times, n_gauss=3)
    assert got.total == pytest.approx(ref, rel=3e-14, abs=0.0)
    assert calls and max(calls) <= chunk
    tail = 27 % chunk
    if tail:
        assert tail in calls


def _linear_chain():
    gamma, b = 0.7, -0.4
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=1),
        linear=sw.DiagonalA(gamma=[gamma]),
        vertices=[sw.LocalVertex("B", coupling=np.array([[b]]))],
        noise=_noise(),
        t_min=T_MIN,
    )
    exp = system.expand(("phi(x)", "psi(y)"), orders=[0, 1, 2, 3])
    props = system.propagators(
        t_max=2.0,
        c_closed_form=ZeroC(1),
        c_closed_form_only=True,
        c_closed_form_vectorized=True,
        progress=False,
    )
    return exp, props, gamma, b


@pytest.mark.parametrize("chunk", [1, 7, 124, 125, 126])
def test_response_chain_simplex_matches_exact_shifted_linear_drift(chunk):
    exp, props, gamma, b = _linear_chain()
    T, source = 1.4, 0.45
    lag = T - source
    got = exp.evaluate(
        props,
        positions={"x": 0.0, "y": 0.0},
        t_final=T,
        external_times={"y": source},
        component_pair=(0, 0),
        orders=[0, 1, 2, 3],
        method="gauss_legendre",
        n_gauss=5,
        gl_chunk_size=chunk,
    )
    # R_alpha=exp[(-gamma+alpha*b)(T-source)]. Taylor coefficients in alpha.
    for order in range(4):
        ref = np.exp(-gamma * lag) * b**order * lag**order / factorial(order)
        assert got.by_order[order] == pytest.approx(ref, rel=2e-12, abs=0.0)
    assert all(row["error"] == 0.0 for row in got.per_diagram)


@dataclass
class PairExponentialKappa:
    calls: list
    beta: float = 1.3
    has_coincident_time_kinks = True

    def __call__(self, n_array, t_array):
        t = np.asarray(t_array)
        self.calls.append(t.shape[1])
        envelope = (
            sum(
                np.exp(-self.beta * np.abs(t[i] - t[j]))
                for i, j in ((0, 1), (0, 2), (1, 2))
            )
            / 3
        )
        return envelope[:, None, None, None] * K_RAW[None]


def test_chunk_bound_is_preserved_through_all_internal_kink_pieces():
    calls = []
    kernel = PairExponentialKappa(calls)
    exp, props = _cumulant(kernel, vectorized=True)
    L, beta, comps = 1.2, kernel.beta, (0, 1, 1)
    # Integral over cube of the symmetric mean of three pair kernels.
    pair_area = 2 * L / beta + 2 * np.expm1(-beta * L) / beta**2
    ref = K_RAW[comps] * L * pair_area
    eager = _eval3(exp, props, None, comps=comps, n_gauss=8)
    calls.clear()
    got = _eval3(exp, props, 7, comps=comps, n_gauss=8)
    assert got.total == pytest.approx(ref, rel=2e-12, abs=0.0)
    assert got.total == pytest.approx(eager.total, rel=2e-14, abs=0.0)
    assert calls and max(calls) <= 7


@pytest.mark.parametrize(
    "method,chunk", [("gauss_legendre", None), ("gauss_legendre", 3), ("nquad", 3)]
)
def test_chunk_bound_is_preserved_through_fixed_external_time_cuts(method, chunk):
    calls = []
    kernel = PairExponentialKappa(calls)
    exp, props = _cumulant(kernel, vectorized=True, absorbed=True)
    L, a, beta, comps = 1.2, 0.45, kernel.beta, (0, 1, 1)
    integral_at_a = (2 - np.exp(-beta * a) - np.exp(-beta * (L - a))) / beta
    integral_at_L = -np.expm1(-beta * L) / beta
    ref = (
        K_RAW[comps] * (integral_at_a + integral_at_L + L * np.exp(-beta * (L - a))) / 3
    )
    got = _eval3(
        exp,
        props,
        chunk,
        comps=comps,
        n_gauss=8,
        method=method,
        integrate_over={"x"},
        external_times={"y": T_MIN + a, "z": T_MIN + L},
    )
    assert got.total == pytest.approx(ref, rel=2e-12, abs=0.0)
    assert calls
    if method == "gauss_legendre" and chunk is not None:
        assert max(calls) <= chunk


@pytest.mark.parametrize("absorbed,dynamic", [(True, False), (True, True)])
def test_zero_dimensional_static_and_dynamic_bypass_grid(absorbed, dynamic):
    coupling = (lambda n, t: K_RAW.copy()) if dynamic else K_RAW.copy()
    exp, props = _cumulant(coupling, absorbed=absorbed)
    got = _eval3(exp, props, 1, n_gauss=8)
    assert got.total == pytest.approx(K_RAW[0, 1, 1], rel=1e-14, abs=0.0)
    assert all(row["error"] == 0.0 for row in got.per_diagram)


def test_zero_width_causal_domain_returns_zero():
    exp, props, _, _ = _linear_chain()
    got = exp.evaluate(
        props,
        positions={"x": 0.0, "y": 0.0},
        t_final=T_MIN,
        external_times={"y": T_MIN},
        component_pair=(0, 0),
        orders=[1, 2, 3],
        method="gauss_legendre",
        n_gauss=4,
        gl_chunk_size=3,
    )
    assert got.total == 0.0
    assert all(row["value"] == 0.0 and row["error"] == 0.0 for row in got.per_diagram)


@dataclass
class RandomSourceC:
    G: np.ndarray
    has_diagonal_kink = False

    def __call__(self, x1, t1, x2, t2):
        x1a, x2a = np.asarray(x1), np.asarray(x2)
        diff = x1a - x2a
        distance_sq = (
            np.sum(diff * diff, axis=-1) if diff.ndim > 1 else np.sum(diff * diff)
        )
        temporal = (np.asarray(t1) - T_MIN) * (np.asarray(t2) - T_MIN)
        prefactor = temporal * np.exp(-distance_sq)
        return prefactor[..., None, None] * self.G


@pytest.mark.parametrize("chunk", [3, 6])
@pytest.mark.parametrize("vectorized", [False, True])
def test_fixed_3d_C_positions_are_not_misread_when_block_or_tail_has_three_samples(
    chunk, vectorized
):
    G = np.array([[0.8, -0.25], [-0.25, 0.5]])
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[0.0, 0.0]),
        noise=_noise(),
        t_min=T_MIN,
    )
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0], diag_C=False)
    props = system.propagators(
        t_max=2.0,
        c_closed_form=RandomSourceC(G),
        c_closed_form_only=True,
        c_closed_form_vectorized=vectorized,
        diag_C=False,
        progress=False,
    )
    x = np.array([0.1, -0.2, 0.4])
    y = np.array([0.8, 0.3, -0.1])
    L = 1.2
    # phi_i(t,x)=xi_i(x)*(t-t_min), with covariance G exp(-||x-y||^2).
    # Each integrated external contributes integral_0^L t dt = L^2/2.
    ref = G[0, 1] * np.exp(-np.dot(x - y, x - y)) * L**4 / 4
    got = exp.evaluate(
        props,
        positions={"x": x, "y": y},
        t_final=T_MIN + L,
        component_pair=(0, 1),
        orders=[0],
        integrate_over="all",
        method="gauss_legendre",
        n_gauss=3,
        gl_chunk_size=chunk,
    )
    assert got.total == pytest.approx(ref, rel=2e-14, abs=0.0)


@dataclass
class NearZeroPhaseKappa:
    epsilon: float
    n: int = 1

    def __call__(self, n_array, t_array):
        t = np.asarray(t_array)[0] - T_MIN
        val = t**8 + 1j * self.epsilon
        return val.reshape((len(t),) + (1,) * 3) * np.ones((self.n,) * 3)


@pytest.mark.parametrize("matrix_R", [False, True])
def test_reality_projection_uses_entire_leaf_scale_across_chunks(matrix_R):
    n = 2 if matrix_R else 1
    linear = None
    if matrix_R:

        def matrix(t, s):
            return np.eye(n) if t >= s else np.zeros((n, n))

        linear = sw.ExplicitR(R_time=matrix, iso_R=False)
    exp, props = _cumulant(
        NearZeroPhaseKappa(1e-11, n),
        n=n,
        linear=linear,
        vectorized=True,
        equal_time=True,
    )
    comps = (0, 1, 1) if n == 2 else (0, 0, 0)
    # Integral_0^1 t^8 dt=1/9. Tiny imaginary roundoff is judged against
    # the leaf's largest real coefficient, not its first near-zero node.
    eager = _eval3(exp, props, None, t_final=T_MIN + 1, comps=comps, n_gauss=6)
    got = _eval3(exp, props, 1, t_final=T_MIN + 1, comps=comps, n_gauss=6)
    assert got.total == pytest.approx(1 / 9, rel=3e-14, abs=0.0)
    assert got.total == pytest.approx(eager.total, rel=3e-14, abs=0.0)


def test_reality_projection_still_rejects_a_bad_phase_at_leaf_completion():
    exp, props = _cumulant(
        NearZeroPhaseKappa(1e-3), n=1, vectorized=True, equal_time=True
    )
    for chunk in (None, 1, 5):
        with pytest.raises(ValueError, match="imaginary|Im"):
            _eval3(exp, props, chunk, t_final=T_MIN + 1, comps=(0, 0, 0), n_gauss=6)


@dataclass
class AssignmentPhaseKappa:
    def __call__(self, n_array, t_array):
        n_samples = np.asarray(t_array).shape[1]
        K = np.zeros((2, 2, 2), dtype=complex)
        K[0, 0, 0] = 1e10
        K[1, 1, 1] = 1e-3j
        return np.broadcast_to(K, (n_samples,) + K.shape)


def test_component_assignments_cannot_borrow_each_others_reality_scale():
    def matrix(t, s):
        return np.eye(2) if t >= s else np.zeros((2, 2))

    exp, props = _cumulant(
        AssignmentPhaseKappa(),
        n=2,
        linear=sw.ExplicitR(R_time=matrix, iso_R=False),
        vectorized=True,
        equal_time=True,
    )
    for chunk in (None, 1):
        with pytest.raises(ValueError, match="imaginary|Im"):
            _eval3(exp, props, chunk, comps=(1, 1, 1), n_gauss=4)


@dataclass
class CancellingKappa:
    offset: float
    amplitude: float = 1e4

    def __call__(self, n_array, t_array):
        t = np.asarray(t_array) - T_MIN
        envelope = self.amplitude * (np.sum(t, axis=0) - 1.5) + self.offset
        return envelope[:, None, None, None] * K_RAW[None]


@pytest.mark.parametrize("offset", [0.0, 1e-3])
def test_signed_node_cancellation_has_an_absolute_error_budget(offset):
    exp, props = _cumulant(CancellingKappa(offset), vectorized=True)
    # Over [0,1]^3, integral(sum(t)-1.5)=0; remaining value is K*offset.
    ref = K_RAW[0, 1, 1] * offset
    for chunk in (1, 7, 63, None):
        got = _eval3(exp, props, chunk, t_final=T_MIN + 1, n_gauss=4).total
        # The tolerance follows the integrand scale under cancellation;
        # no relative tolerance can describe the exact-zero case.
        assert got == pytest.approx(ref, rel=0.0, abs=2e-10)


@pytest.mark.parametrize("scale", [1.0, 1e-21, 1e-100])
@pytest.mark.parametrize("dynamic", [False, True])
def test_small_coupling_large_response_remains_finite_in_complete_GL_path(
    scale, dynamic
):
    gain = scale ** (-1 / 3)

    def R(t, s):
        return gain if t >= s else 0.0

    coupling = scale * K_RAW.copy()
    if dynamic:
        fixed = coupling

        def coupling(n, t):
            return fixed.copy()

    exp, props = _cumulant(coupling, linear=sw.ExplicitR(R_time=R, iso_R=True))
    got = _eval3(exp, props, 7, t_final=T_MIN + 1, n_gauss=4)
    ref = K_RAW[0, 1, 1] * scale * gain**3
    assert np.isfinite(got.total)
    assert got.total == pytest.approx(ref, rel=3e-14, abs=0.0)


@pytest.mark.parametrize("n_jobs,evaluate_n_jobs", [(2, 1), (1, 2)])
@pytest.mark.parametrize("reuse", [False, True])
def test_parallel_and_reused_sweeps_with_dense_R_C_match_independent_hierarchy(
    n_jobs, evaluate_n_jobs, reuse
):
    system, props = mb._dense("F")
    exp = system.expand(mb.OBS2, orders=[0, 2], diag_R=False, diag_C=False)
    assert sum(len(exp.diagrams(o)) for o in exp.orders) > 2
    kw = dict(
        positions_grid={"x": [0.0], "y": [0.0, 0.7]},
        t_final_grid=[mb.T_F],
        component_pairs=[(0, 1), (1, 1)],
        orders=[0, 2],
        method="gauss_legendre",
        n_gauss=8,
        progress=False,
    )
    eager = exp.sweep(props, **kw)
    got = exp.sweep(
        props,
        gl_chunk_size=7,
        reuse_integrands=reuse,
        n_jobs=n_jobs,
        evaluate_n_jobs=evaluate_n_jobs,
        **kw,
    )
    assert len(got.rows) == len(eager.rows)
    for actual, expected in zip(got.rows, eager.rows):
        assert {k: v for k, v in actual.items() if k != "value"} == {
            k: v for k, v in expected.items() if k != "value"
        }
        assert actual["value"] == pytest.approx(expected["value"], rel=2e-13, abs=1e-15)
    for row in got.totals().to_dict("records"):
        ref = mb._dense_moment((row["a"], row["b"]), row["order"])
        assert row["value"] == pytest.approx(ref, rel=3e-10, abs=1e-14)


@pytest.mark.parametrize(
    "route", ["direct", "integrate_moment", "integrate_diagrams", "evaluate"]
)
def test_public_L0_and_L1_routes_forward_GL_chunk_size(route):
    exp, props = _cumulant(K_RAW.copy())
    comps = (0, 1, 1)
    fixed = dict(zip("abc", comps))
    terms = exp.diagrams(1)
    if route == "direct":
        vals = [
            dt.build_integrand(
                exp.system.build_coupling_values(), fixed
            ).integrate_moment_gauss_legendre(
                1.4,
                props.cache,
                t_min=T_MIN,
                positions=POS3,
                n_gauss=3,
                gl_chunk_size=7,
            )[
                0
            ]
            for dt in terms
        ]
        value = sum(vals)
    elif route == "integrate_moment":
        vals = [
            integrate_moment(
                dt.build_integrand(exp.system.build_coupling_values(), fixed),
                1.4,
                props.cache,
                t_min=T_MIN,
                positions=POS3,
                method="gauss_legendre",
                n_gauss=3,
                gl_chunk_size=7,
            )[0]
            for dt in terms
        ]
        value = sum(vals)
    elif route == "integrate_diagrams":
        value = integrate_diagrams(
            terms,
            exp.system.build_coupling_values(),
            1.4,
            props.cache,
            t_min=T_MIN,
            fixed_indices=fixed,
            positions=POS3,
            method="gauss_legendre",
            n_gauss=3,
            gl_chunk_size=7,
        )[0]
    else:
        value = _eval3(exp, props, 7, n_gauss=3).total
    assert value == pytest.approx(K_RAW[comps] * 1.2**3, rel=2e-14, abs=0.0)


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, "3", [], {}])
def test_gl_chunk_size_requires_a_positive_nonbool_integer(bad):
    exp, props = _cumulant(K_RAW.copy())
    with pytest.raises((ValueError, TypeError), match="gl_chunk_size"):
        _eval3(exp, props, bad, n_gauss=3)


def test_chunk_option_does_not_change_QMC_samples_or_uncertainty():
    exp, props = _cumulant(K_RAW.copy(), gamma=[0.7, 1.6])
    kw = dict(
        positions_grid={k: [v] for k, v in POS3.items()},
        t_final_grid=[1.4],
        component_tuples=[(0, 1, 1)],
        orders=[1],
        method="qmc_vectorized",
        n_samples=128,
        seed=17,
        progress=False,
    )
    eager = exp.sweep(props, **kw)
    got = exp.sweep(props, gl_chunk_size=3, **kw)
    assert got.rows == eager.rows


_YAML_BASE = {
    "system": {
        "field": {"name": "phi", "n_components": 1},
        "linear": {"type": "diagonal", "gamma": [1.0]},
        "vertices": [],
        "nonlocal_vertices": [{"name": "K", "order": 3, "coupling": [[[2.0]]]}],
        "noise": {
            "kappa2": {
                "type": "separable_translation",
                "temporal": {"type": "exponential", "lam": 0.0, "sigma_t": 0.5},
                "spatial": {"type": "exponential", "sigma_x": 1.0},
            }
        },
    },
    "expand": {"observable": ["phi(x)", "phi(y)", "phi(z)"], "orders": [1]},
    "propagators": {"t_max": 2.0, "n_grid_t": 6},
    "sweep": {
        "positions_grid": {"x": [0.0], "y": [0.3], "z": [-0.4]},
        "t_final_grid": [0.8, 1.2],
        "component_tuples": [[0, 0, 0]],
        "orders": [1],
        "method": "gauss_legendre",
        "n_gauss": 8,
    },
    "output": [],
}


def test_yaml_default_none_and_chunked_execution_against_exact_cumulant(tmp_path):
    import copy

    cfg = copy.deepcopy(_YAML_BASE)
    path = tmp_path / "chunked.yaml"
    path.write_text(yaml.safe_dump(cfg))
    default_cfg = load_workflow_config(path)
    assert default_cfg.sweep.gl_chunk_size is None
    eager, _ = run_workflow(default_cfg, progress=False)
    cfg["sweep"]["gl_chunk_size"] = 7
    path.write_text(yaml.safe_dump(cfg))
    chunk_cfg = load_workflow_config(path)
    assert chunk_cfg.sweep.gl_chunk_size == 7
    got, totals = run_workflow(chunk_cfg, progress=False)
    assert len(got.rows) == len(eager.rows)
    for row in totals.to_dict("records"):
        ref = 2 * (-np.expm1(-row["t_final"])) ** 3
        assert row["value"] == pytest.approx(ref, rel=2e-12, abs=0.0)


@pytest.mark.parametrize("bad", [True, 3.5, "8", -2, 0])
def test_yaml_chunk_bound_does_not_silently_coerce_invalid_values(tmp_path, bad):
    import copy

    cfg = copy.deepcopy(_YAML_BASE)
    cfg["sweep"]["gl_chunk_size"] = bad
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises((TypeError, ValueError), match="gl_chunk_size"):
        load_workflow_config(path)


@pytest.mark.parametrize("chunk", [None, 7])
def test_cli_dry_run_times_the_configured_GL_batches(
    tmp_path, monkeypatch, capsys, chunk
):
    import copy

    import sft_wick.evaluate as evaluation
    from sft_wick.workflow import Expansion
    from sft_wick.workflow.cli import main

    cfg = copy.deepcopy(_YAML_BASE)
    cfg["sweep"]["gl_chunk_size"] = chunk
    path = tmp_path / "dry-run.yaml"
    path.write_text(yaml.safe_dump(cfg))
    blocks, results = [], []
    original_blocks = evaluation._gl_tensor_blocks
    original_evaluate = Expansion.evaluate

    def observe_blocks(*args, **kwargs):
        for nodes, weights in original_blocks(*args, **kwargs):
            blocks.append(len(nodes))
            yield nodes, weights

    def observe_evaluate(self, *args, **kwargs):
        result = original_evaluate(self, *args, **kwargs)
        results.append(result.total)
        return result

    monkeypatch.setattr(evaluation, "_gl_tensor_blocks", observe_blocks)
    monkeypatch.setattr(Expansion, "evaluate", observe_evaluate)
    assert main(["run", str(path), "--dry-run", "--quiet"]) == 0
    output = capsys.readouterr().out
    assert "cost estimate" in output and "n_gauss=8" in output
    # Both the warm-up and measured diagram evaluation use the same
    # configured bound and retain the exact scalar cumulant reference.
    assert len(results) >= 2
    for value in results:
        assert value == pytest.approx(2 * (-np.expm1(-1.2)) ** 3, rel=2e-12, abs=0.0)
    if chunk is None:
        assert blocks and set(blocks) == {8**3}
        assert "gl_chunk_size=" not in output
    else:
        assert blocks and max(blocks) == chunk
        assert f"gl_chunk_size={chunk}" in output


@dataclass
class CappedR:
    limit: int
    calls: list
    vectorized = True

    def __call__(self, t, s):
        t, s = np.broadcast_arrays(t, s)
        if t.ndim:
            self.calls.append(t.size)
            if t.size > self.limit:
                raise AssertionError(
                    f"R callback received {t.size} samples, limit={self.limit}"
                )
        return np.where(t > s, 1.0, 0.0)


def test_high_dimensional_tensor_rule_never_passes_a_full_product_to_callback():
    calls = []
    cap = 257
    linear = sw.ExplicitR(R_time=CappedR(cap, calls), iso_R=True, vectorized=True)
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=1),
        linear=linear,
        nonlocal_vertices=[sw.NonLocalVertex("K", order=5, coupling=np.ones((1,) * 5))],
        noise=_noise(),
        t_min=T_MIN,
    )
    labels = "xyzwv"
    exp = system.expand(tuple(f"phi({p})" for p in labels), orders=[1])
    props = system.propagators(
        t_max=2.0,
        c_closed_form=ZeroC(1),
        c_closed_form_only=True,
        c_closed_form_vectorized=True,
        progress=False,
    )
    calls.clear()
    got = exp.evaluate(
        props,
        positions=dict.fromkeys(labels, 0.0),
        t_final=T_MIN + 1.0,
        component_pair=(0,) * 5,
        orders=[1],
        method="gauss_legendre",
        n_gauss=10,
        gl_chunk_size=cap,
    )
    assert got.total == pytest.approx(1.0, rel=3e-13, abs=0.0)
    assert calls and max(calls) <= cap
    assert len(calls) > 100
