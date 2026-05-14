"""Regression tests for matrix-valued response propagator evaluation."""

import numpy as np
import pytest

from sft_wick import Action, Field, compute_moment
from sft_wick.evaluate import PropagatorCache, PropagatorModel


def test_non_iso_r_integrand_evaluate_uses_matrix_element():
    """Matrix R should not be routed through scalar-only R_product."""

    phi = Field("phi", "physical", n_components=2)
    psi = Field("psi", "response", n_components=2)
    result = compute_moment(
        [phi("1", "x"), psi("2", "y")],
        Action([]),
        order=0,
        response_phase=False,
        collect_topology=True,
        diag_R=False,
        diag_C=False,
        iso_R=False,
    )
    dt = result.diagram_terms(0)[0]
    ig = dt.build_integrand({})

    def R_matrix(t1, t2):  # noqa: ARG001
        return np.array([[1.0, 2.0], [3.0, 4.0]])

    def kappa2(n1, t1, n2, t2):  # noqa: ARG001
        return np.eye(2)

    cache = PropagatorCache(
        PropagatorModel(
            R_time=R_matrix,
            kappa2=kappa2,
            n_components=2,
            iso_R=False,
            diag_C=False,
        )
    )

    val = ig.evaluate({"x": 1.0, "y": 0.0}, {}, cache)

    assert val == pytest.approx(-2.0j)
