"""Static Gaussian disorder and the two times of its averaged MSR vertex."""

import numpy as np
import pytest
from scipy.integrate import quad

from sft_wick import (
    ConstantImpulse,
    DiagonalA,
    ExponentialSpatial,
    ExponentialTemporal,
    FieldSpec,
    GaussianNoise,
    SeparableTranslation,
    System,
)
from sft_wick.disorder import RandomCoupling


def test_disorder_vertex_has_independent_clocks_and_a_shared_matrix():
    law = RandomCoupling(2, variance=1.4, ensemble="goe")
    vertex = law.vertex(copies=2)
    system = System(
        FieldSpec("phi", 4),
        DiagonalA([1.0]),
        GaussianNoise(
            SeparableTranslation(ExponentialTemporal(0, 1), ExponentialSpatial(1)),
            ConstantImpulse(1.0),
        ),
        grouped_vertices=[vertex],
    )
    (raw,) = system.build_action().vertices
    assert raw.leg_groups == (0, 0, 1, 1)
    assert [f.is_response for f in raw.fields] == [True, False, True, False]
    coupling = system.build_coupling_values()["J"]
    np.testing.assert_allclose(
        [
            coupling[0, 1, 2, 3],
            coupling[0, 1, 3, 2],
            coupling[0, 0, 2, 2],
            coupling[0, 3, 2, 1],
        ],
        [0.35, 0.35, 0.7, 0],
        atol=0,
        rtol=1e-15,
    )
    with pytest.raises(ValueError, match="requires 2 components"):
        System(
            system.field, system.linear, system.noise, grouped_vertices=[law.vertex()]
        )


@pytest.mark.parametrize(
    "n,kind,expected",
    [
        (6, "iid", (0.04157221367654879, 0.004530008517361823)),
        (20, "iid", (0.03501873918661763, 0.0028568900472686408)),
        (6, "goe", (0.10311320337206652, 0.027350122268941346)),
        (20, "goe", (0.09280188303485985, 0.020819322044909088)),
    ],
)
def test_two_time_response_coefficients_against_matrix_exponential(n, kind, expected):
    # Independent Isserlis contractions of the two matrix exponential series
    # about g=0 with impulse 1, damping 1 and elapsed times .5 and 1.
    law = RandomCoupling(n, ensemble=kind)
    readout = [(0, 2, 0.7), (0, 2, 1.2)]
    actual = [law.coefficient(readout, pairs=k, t_min=0.2, n_gauss=6) for k in [1, 2]]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def test_shared_disorder_covariance_of_thermal_second_moments():
    n = 6
    law = RandomCoupling(n)
    observations = [(replica, 1, t) for replica in (0, 1) for t in (0.7, 1.2)]
    value = law.coefficient(observations, pairs=1, copies=2, connected=True, t_min=0.2)
    first = quad(lambda v: np.exp(-(1.9 - 2 * v)) * (1.9 - 2 * v), 0.2, 0.7)[0]
    np.testing.assert_allclose(value, first**2 / n, rtol=1e-11, atol=0)
    assert law.coefficient([(0, 1, 0.7), (1, 1, 1.2)], pairs=1, copies=2) == 0


def test_scalar_cubic_coefficient_retains_shared_disorder_across_lines():
    law = RandomCoupling(1)
    actual = law.coefficient([(0, 0, 1.0)] * 2, pairs=1, cubic=1)
    np.testing.assert_allclose(actual, -0.48879239428114074, rtol=1e-11, atol=0)


def test_ordered_tensor_has_only_direct_element_pairings():
    law = RandomCoupling(2, rank=3)
    bare = law.vertex().coupling(None, None)
    assert bare[1, 0, 1, 1, 0, 1] == 0.25
    assert bare[1, 0, 1, 1, 1, 0] == 0
    # Sum of the independent tagged Itô responses with each of the eight
    # unit tensor entries turned on separately at variance 1/4.
    result = law.coefficient([(0, 0, 0.65)] * 2, pairs=1)
    np.testing.assert_allclose(result, 0.03581351575748211, rtol=1e-12, atol=0)
    assert RandomCoupling(2, variance=0).coefficient([(0, 0, 0.2)] * 2, pairs=1) == 0
    with pytest.raises(ValueError, match="matrix"):
        RandomCoupling(2, rank=3, ensemble="goe")


def test_independent_thermal_copies_in_the_general_system():
    law = RandomCoupling(2)
    system = System(
        FieldSpec("phi", 4),
        DiagonalA([0.6]),
        GaussianNoise(
            SeparableTranslation(ExponentialTemporal(0, 1), ExponentialSpatial(1)),
            ConstantImpulse(1.7),
        ),
        grouped_vertices=[law.vertex(copies=2)],
        t_min=0.1,
    )
    expansion = system.expand(
        ["phi_a(x)", "phi_b(y)", "phi_c(z)", "phi_d(w)"],
        orders=[1],
        target="connected",
        routing="multiplicity",
        progress=False,
    )
    times = [0.5, 0.8, 0.5, 0.8]
    direct = expansion.evaluate(
        system.propagators(t_max=0.8, progress=False),
        positions=dict.fromkeys("xyzw", 0.0),
        external_times=dict(zip("xyzw", times)),
        component_pair=(0, 0, 2, 2),
        method="gauss_legendre",
        n_gauss=5,
        t_final=0.8,
    ).total
    projected = law.coefficient(
        [(c, 0, t) for c, t in zip((0, 0, 1, 1), times)],
        pairs=1,
        copies=2,
        connected=True,
        decay=0.6,
        noise=1.7,
        t_min=0.1,
        n_gauss=5,
    )
    np.testing.assert_allclose(projected, direct, rtol=1e-12, atol=0)
