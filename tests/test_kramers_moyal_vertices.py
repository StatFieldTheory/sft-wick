"""Polynomial Itô generator terms through the Python and YAML entry points."""

import numpy as np
import pytest

import sft_wick as sw


@pytest.mark.parametrize("n_psi", [3, 4])
def test_yaml_polynomial_km_coefficients_match_response_integral(n_psi, tmp_path):
    import yaml
    from sft_wick.workflow.config import load_workflow_config, run_workflow

    start, early, late = 0.3, 0.61, 0.87
    gamma, variance = 0.8, 0.6
    doc = {
        "system": {
            "t_min": start,
            "linear": {"type": "diagonal", "gamma": gamma},
            "noise": {
                "kappa2": {
                    "type": "separable_translation",
                    "temporal": {"type": "exponential", "lam": 0, "sigma_t": 1},
                    "spatial": {"type": "exponential", "sigma_x": 1},
                },
                "sigma2": {"type": "constant", "amplitude": variance},
            },
            "km_vertices": [
                {"name": f"D{n_psi}{degree}", "n_psi": n_psi,
                 "coupling": np.full((1,) * (n_psi + degree), coefficient).tolist()}
                for degree, coefficient in enumerate((0.19, -0.12, 0.31))
            ],
        },
        "expand": {"observable": [f"phi({x})" for x in "xyzw"[:n_psi]], "orders": [1]},
        "propagators": {"t_max": late},
        "sweep": {"positions_grid": dict.fromkeys("xyzw"[:n_psi], [0]),
                  "t_final_grid": [late], "external_times_grid": {"x": [early]},
                  "component_tuples": [[0] * n_psi], "method": "gauss_legendre",
                  "n_gauss": 20},
    }
    path = tmp_path / "km.yaml"
    path.write_text(yaml.safe_dump(doc))
    _, result = run_workflow(load_workflow_config(path))

    # With exactly m external legs and m psi legs, all response lines
    # go to the externals. Only the constant and the Wick-paired quadratic
    # part of D survive at first order from a centered Gaussian start.
    from scipy.integrate import quad

    def integrand(t):
        covariance = -variance * np.expm1(-2 * gamma * (t - start)) / (2 * gamma)
        response_product = np.exp(-gamma * (early + (n_psi - 1) * late - n_psi * t))
        return (0.19 + 0.31 * covariance) * response_product

    expected = quad(integrand, start, early, epsabs=1e-14)[0]
    assert result["value"].iloc[0] == pytest.approx(expected, rel=0, abs=1e-14)

    # A missing 1/m! leaves the phase real but multiplies this amplitude.
    from math import factorial

    for term in doc["system"]["km_vertices"]:
        term["coupling"] = (np.asarray(term["coupling"]) * factorial(n_psi)).tolist()
    path.write_text(yaml.safe_dump(doc))
    _, wrong = run_workflow(load_workflow_config(path))
    assert abs(wrong["value"].iloc[0] - expected) > 0.1


def test_dynamic_km_tensor_must_match_declared_components():
    bad = sw.KramersMoyalVertex(
        "D3", 3, lambda x, t: np.ones((np.shape(t)[1], 1, 1, 1)),
        rank=3, coupling_vectorized=True,
    )
    with pytest.raises(ValueError, match="callable coupling returned shape"):
        bad.build_msr_coupling(2)(np.zeros((1, 2)), np.zeros((1, 2)))


def test_yaml_quadratic_noise_uses_alpha_in_the_ito_drift(tmp_path):
    from scipy.integrate import quad
    import yaml
    from sft_wick.workflow.config import load_workflow_config, run_workflow

    start, end, gamma = 0.3, 0.95, 0.8
    g0, g1, g2 = 0.7, -0.3, 0.2
    sigma2 = {
        "type": "multiplicative", "g0": [[g0]], "g1": [[[g1]]], "g2": [[[[g2]]]],
        "alpha": 0.5,
    }
    doc = {
        "system": {
            "t_min": start, "linear": {"type": "diagonal", "gamma": gamma},
            "noise": {
                "kappa2": {
                    "type": "separable_translation",
                    "temporal": {"type": "exponential", "lam": 0, "sigma_t": 1},
                    "spatial": {"type": "exponential", "sigma_x": 1},
                },
                "sigma2": sigma2,
            },
        },
        "expand": {"observable": ["phi(x)"], "orders": [1]},
        "propagators": {"t_max": end},
        "sweep": {"positions_grid": {"x": [0]}, "t_final_grid": [end],
                  "component_tuples": [[0]], "method": "gauss_legendre", "n_gauss": 20},
    }
    path = tmp_path / "alpha.yaml"
    path.write_text(yaml.safe_dump(doc))
    _, result = run_workflow(load_workflow_config(path))

    def integrand(t):
        covariance = -g0**2 * np.expm1(-2 * gamma * (t - start)) / (2 * gamma)
        return 0.5 * np.exp(-gamma * (end - t)) * (g0 * g1 + 3 * g1 * g2 * covariance)

    expected = quad(integrand, start, end, epsabs=1e-14)[0]
    assert result["value"].iloc[0] == pytest.approx(expected, rel=0, abs=1e-14)
    sigma2["alpha"] = 0.0
    path.write_text(yaml.safe_dump(doc))
    _, wrong = run_workflow(load_workflow_config(path))
    assert wrong["value"].iloc[0] == 0.0
    assert abs(expected) > 1e-2
