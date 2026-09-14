"""Demo2 analytic kernel and diagram channels against independent moments."""
from itertools import permutations
from pathlib import Path
import sys
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/reference"))
sys.path.insert(0, str(ROOT / "examples/demo2"))
sys.path.insert(0, str(ROOT / "examples/paper_assets/demo2_kappa4"))

from demo2_moments import moments
import k3_R_coupling as kernel
import run_budget as budget
import k4_R_contracted as kernel4
import sft_wick as sw


def test_reference_covariance_and_gaussian_parity():
    times = [0.0, 0.05, 1.0, 15.0]
    for r in (0.0, 0.6):
        result = moments(times, r, channels=("o0", "fk", "fffk", "fk5"))
        exact = np.array([budget.C_eff_exact(0.0, t, r, t)[0, 0] for t in times])
        np.testing.assert_allclose(result["o0"][:, 0], exact, rtol=2e-12, atol=1e-17)
        np.testing.assert_allclose(result["o0"][:, 2], exact, rtol=2e-12, atol=1e-17)
        np.testing.assert_array_equal(result["o0"][:, 1], 0)
        gaussian = moments(times, r, alpha=0, channels=("fk", "fffk", "fk5"))
        for values in gaussian.values():
            np.testing.assert_allclose(values, 0, atol=1e-18)


def test_reference_two_site_limit_and_replica_extraction():
    times = [0.1, 1.0, 3.0, 15.0, 50.0]
    single = moments(times, channels=("fk", "fffk", "fk5"))
    two = moments(times, r=1e-10, channels=("fk", "fffk", "fk5"))
    for channel in single:
        np.testing.assert_allclose(two[channel], single[channel], rtol=5e-9, atol=1e-19)
    # Obtained by an independent, untagged six-variable M=1/M=2 generator:
    # A = 2*sqrt(2)*moment(M=2) - moment(M=1) isolates kappa3*C.
    expected = [1.60682140668e-12, 3.089920628869201e-6,
                4.403914387897749e-5, 5.50315822178252e-5, 5.503163023890467e-5]
    np.testing.assert_allclose(single["fffk"][:, 1], expected, rtol=5e-10, atol=0)


@pytest.mark.parametrize("sigma_t", [0.3, 1.0, 1.00000001, 1e20])
def test_kernel_permutation_short_time_and_resonance(sigma_t):
    times = np.array([[1.3, 0.1, 1e-10], [0.4, 0.1, 2e-10], [0.9, 0.1, 3e-10]])
    positions = np.array([0.0, 0.4, 1.3])

    def evaluate(perm):
        x = positions[list(perm)]
        spatial = [np.exp(-abs(x[i]-x[j])) for i, j in ((0, 1), (0, 2), (1, 2))]
        return kernel.k3_R(*times[list(perm)], *spatial, sigma_t=sigma_t)

    reference = evaluate((0, 1, 2))
    for perm in permutations(range(3)):
        np.testing.assert_allclose(evaluate(perm), reference, rtol=1e-11, atol=0)
    instantaneous = kernel.kappa3_raw(0, 0, 0, *positions, sigma_t=sigma_t)
    assert reference[-1] == pytest.approx(instantaneous*np.prod(times[:, -1]), rel=2e-9, abs=0)
    if sigma_t == 1e20:
        expected = instantaneous*np.prod(-np.expm1(-times), axis=0)
        np.testing.assert_allclose(reference, expected, rtol=1e-11, atol=0)


@pytest.mark.parametrize("r,t,n_gauss", [(0.0, 0.1, 10), (0.6, 3.0, 18), (0.0, 15.0, 28)])
def test_fffk_all_components_match_independent_moments(r, t, n_gauss):
    system = budget.make_system(budget.LAM, sw.NonLocalVertex(
        "K", 3, coupling=kernel.coupling_fn_vectorized,
        coupling_vectorized=True, already_R_contracted=True))
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[4], progress=False)
    props = budget.props_for(system, True)
    expected = moments([t], r, channels=("fffk",))["fffk"][0]
    got = [expansion.evaluate(props, positions={"x": 0., "y": r}, t_final=t,
                              component_pair=pair, vertex_types=["FK"],
                              method="gauss_legendre", n_gauss=n_gauss,
                              n_jobs=1).total for pair in budget.PAIRS]
    np.testing.assert_allclose(got, expected, rtol=2e-7, atol=1e-20)


@pytest.mark.slow
def test_fffk_late_time_matches_independent_moments():
    test_fffk_all_components_match_independent_moments(0.0, 50.0, 40)


@pytest.mark.parametrize("r,t", [(0.0, 0.2), (0.6, 3.0)])
def test_ffk4_matches_independent_moments(r, t):
    system = budget.make_system(budget.LAM, sw.NonLocalVertex(
        "K4", 4, coupling=kernel4.coupling_fn_vectorized,
        coupling_vectorized=True, already_R_contracted=True))
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[3], progress=False)
    props = budget.props_for(system, True)
    expected = moments([t], r, channels=("ffk4",))["ffk4"][0]
    got = [expansion.evaluate(props, positions={"x": 0., "y": r}, t_final=t,
                              component_pair=pair, vertex_types=["FK4"],
                              method="gauss_legendre", n_gauss=18,
                              n_jobs=1).total for pair in budget.PAIRS]
    np.testing.assert_allclose(got, expected, rtol=2e-7, atol=1e-20)


def test_kappa4_permutation_and_short_time_limits():
    times = np.array([[1.3, 1e-9], [0.4, 2e-9], [0.9, 3e-9], [0.6, 4e-9]])
    positions = np.array([0.0, 0.4, 1.3, 0.7])

    def evaluate(perm, sigma_t=0.3):
        x = positions[list(perm)]
        spatial = {(i, j): np.full(2, np.exp(-abs(x[i]-x[j])))
                   for i in range(4) for j in range(4) if i != j}
        return kernel4.k4_R(times[list(perm)], spatial, sigma_t=sigma_t)

    identity = (0, 1, 2, 3)
    for sigma_t in (0.3, 1.0, 1e20):
        reference = evaluate(identity, sigma_t)
        for perm in ((1, 0, 3, 2), (2, 3, 0, 1), (3, 1, 2, 0)):
            np.testing.assert_allclose(evaluate(perm, sigma_t), reference, rtol=1e-10, atol=0)
        import k4_coupling
        instantaneous = k4_coupling.kappa4_amplitude(positions[:, None], np.zeros((4, 1)))[0]
        assert reference[-1] == pytest.approx(instantaneous*np.prod(times[:, -1]), rel=1e-7, abs=0)
        if sigma_t == 1e20:
            np.testing.assert_allclose(reference, instantaneous*np.prod(-np.expm1(-times), axis=0),
                                       rtol=1e-10, atol=0)


def test_analytic_kernel_serializes_without_example_import_path(tmp_path):
    from joblib.externals import cloudpickle
    cloudpickle.register_pickle_by_value(kernel)
    path = tmp_path / "kernel.pkl"
    path.write_bytes(cloudpickle.dumps(kernel.coupling_fn_vectorized))
    code = ("from joblib.externals import cloudpickle; import sys; import numpy as np; "
            "fn=cloudpickle.load(open(sys.argv[1], 'rb')); "
            "value=fn(np.zeros((3,1)),np.ones((3,1)))[0,0,0,0]; "
            "assert abs(value/0.0004550447956000525-1)<1e-10")
    completed = subprocess.run([sys.executable, "-c", code, str(path)], cwd=tmp_path,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
