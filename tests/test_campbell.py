"""Joint Poisson source statistics and the quadratic response of a smooth readout."""

from math import factorial

import numpy as np
import pytest

from sft_wick.campbell import CampbellDriver, ExponentialMark


@pytest.mark.parametrize("compensated", [False, True])
def test_joint_characteristic_and_quadratic_insertion(compensated):
    # Both projections of the same source share N; independent copies fail.
    mu = 0.7
    law = CampbellDriver([mu], [[1.0, -2.0]], compensated=compensated)
    n = np.arange(25)
    probabilities = np.exp(-mu) * np.array([mu**k / factorial(k) for k in map(int, n)])
    x = n - (mu if compensated else 0)
    phases = np.exp(1j * (0.6 * x - 0.8 * x))
    expected = probabilities @ phases
    np.testing.assert_allclose(law.characteristic([0.6, 0.4]), expected, atol=2e-15)
    np.testing.assert_allclose(
        law.quadratic_characteristic([0.6, 0.4], 0, 1),
        probabilities @ (-2 * x**2 * phases),
        atol=3e-15,
    )
    np.testing.assert_allclose(law.mixed_cumulant([0, 1, 1]), 4 * mu)
    assert law.mixed_cumulant([0]) == (0 if compensated else mu)
    np.testing.assert_allclose(
        law.fourier_expectation([[0.0, 0.0], [0.6, 0.4], [-0.6, -0.4]], [1, 0.5, 0.5]),
        1 + expected.real,
    )


def test_saturating_mean_and_quadratic_drift_insertion():
    mu = 0.8
    law = CampbellDriver([mu], [[1.0, 2.0, -3.0]])
    n = np.arange(35)
    weights = np.exp(-mu) * np.array([mu**k / factorial(k) for k in map(int, n)])
    assert law.saturating_mean() == pytest.approx(weights @ (n / (1 + n)), abs=2e-13)
    assert law.saturating_quadratic_response(1, 2) == pytest.approx(
        weights @ (-6 * n * n / (1 + n) ** 2), abs=2e-13
    )


def test_exponential_marks_and_small_compensated_exponent():
    law = CampbellDriver([3.0], [[1.0, 2.0]], mark=ExponentialMark(), compensated=True)
    q = 1.0e-12
    z = 1j * q
    np.testing.assert_allclose(
        law.log_characteristic([0.0, q / 2]), 3 * z * z / (1 - z), rtol=2e-15, atol=0
    )
    assert law.mixed_cumulant([1, 0, 1, 0]) == 3 * 24 * 4
    zero = CampbellDriver([0.0], [[1.0, -2.0]])
    assert zero.saturating_mean() == 0
    with pytest.raises(ValueError, match="nonnegative"):
        law.saturating_mean()


def test_characteristic_without_mark_moments():
    # A compound Poisson process with independent symmetric Cauchy marks.
    class Cauchy:
        def increment(self, z, *, compensated):
            if compensated:
                raise ValueError("a Cauchy mark has no finite mean")
            return np.expm1(-np.abs(z.imag))

    law = CampbellDriver([2.0], [[1.0, -2.0]], mark=Cauchy())
    assert law.characteristic([3, 2]) == pytest.approx(np.exp(2 * np.expm1(-1)))
    with pytest.raises(ValueError, match="moment"):
        law.mixed_cumulant([0])


def test_many_weak_or_frequent_events_remain_visible():
    # For Poisson N the integral of its PGF is E[1/(1+N)]=(1-exp(-mu))/mu.
    mu = 1e6
    law = CampbellDriver([mu], [[1.0]])
    assert law.saturating_mean() == pytest.approx(1 - 1 / mu, abs=3e-13)
    assert 1 - 2 / mu < law.saturating_quadratic_response(0, 0) < 1 - 2 / mu + 2 / mu**2

    tiny = CampbellDriver([1e300], [[1e-200]], compensated=True)
    np.testing.assert_allclose(
        tiny.log_characteristic(1).real, -5e-101, rtol=1e-14, atol=0
    )
    np.testing.assert_allclose(tiny.mixed_cumulant([0, 0]), 1e-100, rtol=1e-14, atol=0)


def test_intensity_validation_and_event_snapshot():
    weights = np.array([0.0, 1.0])
    profiles = np.array([[-10.0], [0.2]])
    law = CampbellDriver(weights, profiles)
    weights[:] = 9
    profiles[:] = 9
    assert law.mixed_cumulant([0]) == 0.2
    with pytest.raises(ValueError):
        law.responses.setflags(write=True)
    with pytest.raises(ValueError, match="nonnegative"):
        CampbellDriver([-1], [[1]])
    with pytest.raises(ValueError, match="one nonempty"):
        CampbellDriver([1], [[1], [2]])
    with pytest.raises(ValueError, match="frequency"):
        law.characteristic([0, 1])
    with pytest.raises(ValueError, match="outside"):
        law.saturating_quadratic_response(0, 1)
    with pytest.raises(ValueError, match="nonnegative"):
        CampbellDriver([2.0], [[-1.0]]).saturating_mean()


@pytest.mark.parametrize(
    ("rate", "jump"), [(1.0, 1e155), (1e155, 1.0), (1e-100, 1e255), (1e255, 1e155)]
)
def test_saturating_quadratic_response_combines_the_scale_before_rounding(rate, jump):
    law = CampbellDriver([rate], [[jump, jump, -jump]])
    expected = np.expm1(-rate)  # negative occupied-source probability
    np.testing.assert_allclose(
        law.saturating_quadratic_response(1, 2), expected, rtol=2e-12, atol=0
    )


def test_characteristic_combines_event_damping_with_quadratic_response():
    law = CampbellDriver([750.0], [[1.0, 1e125]])
    # For Poisson N, E[N² s^N]=exp(mu*(s-1))*((mu*s)^2+mu*s).
    s = 1j
    expected = (
        np.exp(1j * 750)
        * np.exp(-750 + 250 * np.log(10.0))
        * ((750 * s) ** 2 + 750 * s)
    )
    np.testing.assert_allclose(
        law.quadratic_characteristic([np.pi / 2, 0], 1, 1), expected, rtol=5e-13, atol=0
    )
    h = 1e110
    exponential = CampbellDriver([1.0], [[h]], mark=ExponentialMark())
    # E[X² exp(iX)] ~ -2i/(e*h) for a unit-rate exponentially marked source.
    np.testing.assert_allclose(
        exponential.quadratic_characteristic(1, 0, 0),
        -2j / (np.e * h),
        rtol=5e-13,
        atol=0,
    )


@pytest.mark.parametrize("mark", [None, ExponentialMark()])
@pytest.mark.parametrize("rate", [0.3, 0.75])
def test_large_compensated_rotation_preserves_complex_phase(mark, rate):
    h = 1.009326171875e16
    z = 1j * h
    unit = mark is None
    first = np.exp(z) if unit else 1 / (1 - z) ** 2
    second = first if unit else 2 / (1 - z) ** 3
    increment = np.expm1(z) if unit else z / (1 - z)
    # The centering angle is formed from exact binary inputs. Scaling by
    # .75 in a single float would lose the quarter-radian residual.
    from fractions import Fraction

    angle = Fraction(rate) * Fraction(h)
    high = float(angle)
    rot = np.exp(-1j * high) * np.exp(-1j * float(angle - Fraction(high)))
    phi = np.exp(rate * increment) * rot
    law = CampbellDriver([rate], [[h]], mark=mark, compensated=True)
    np.testing.assert_allclose(law.characteristic(1), phi, rtol=2e-13, atol=0)
    expected = phi * h * h * (rate * second + (rate * (first - 1)) ** 2)
    np.testing.assert_allclose(
        law.quadratic_characteristic(1, 0, 0), expected, rtol=2e-13, atol=0
    )


def test_nearly_gaussian_zero_of_the_quadratic_characteristic():
    from fractions import Fraction

    rate, jump = 1e200, 1e-100
    law = CampbellDriver([rate], [[jump]], compensated=True)
    # For unit jumps the insertion is phi*h²*mu*exp(ih)*(1-4*mu*sin(h/2)²).
    # Here the next sine correction is below 1e-199 in the last factor.
    variance = Fraction(rate) * Fraction(jump) ** 2
    expected = np.exp(-float(variance) / 2) * float(variance * (1 - variance))
    np.testing.assert_allclose(
        law.quadratic_characteristic(1, 0, 0), expected, rtol=2e-13, atol=0
    )


@pytest.mark.parametrize(
    ("mark", "compensated", "phi", "insert"),
    [
        (
            None,
            False,
            0.9916778870666417 - 0.06140656155870213j,
            1.0599131680680716 - 0.34800821306925195j,
        ),
        (
            None,
            True,
            -0.9664834539817737 - 0.23044637267460374j,
            -0.8500332190533518 - 0.022476540600624033j,
        ),
        (ExponentialMark(), False, 0.7408182206817179 + 1.842203e-317j, 0j),
        (
            ExponentialMark(),
            True,
            -0.7086200710802327 - 0.21603062504254422j,
            -0.18431208048796852 - 0.05618956557356574j,
        ),
    ],
)
def test_large_projected_event_angle(mark, compensated, phi, insert):
    # 650-decimal-digit compound Poisson formula at the exact dyadic
    # angle 2**1050; the second output has ordinary amplitude 1.7.
    law = CampbellDriver([0.3], [[2.0**950, 1.7]], mark=mark, compensated=compensated)
    k = [2.0**100, 0.0]
    np.testing.assert_allclose(law.characteristic(k), phi, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(
        law.quadratic_characteristic(k, 1, 1), insert, rtol=1e-13, atol=0.0
    )


def test_large_exponential_mark_projection_retains_subnormal_insertion():
    law = CampbellDriver([0.3], [[2.0**1022]], mark=ExponentialMark())
    expected = -1.931689753998e-311
    assert abs(
        law.quadratic_characteristic(8.0, 0, 0).imag - expected
    ) <= 3 * np.nextafter(0.0, 1.0)
