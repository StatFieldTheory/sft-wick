"""Small coefficients must survive multiplication by large kernels.

The independent reference is ordinary multiplication and summation.  Neither
the coefficient's units nor the presence of a summation index changes it.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from sft_wick.evaluate import DiagramIntegrand


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("scale", [1.0, 1e-21, 1e-100])
def test_rescaling_coupling_and_kernel_preserves_contribution(indexed, dynamic,
                                                            scale):
    dt = SimpleNamespace(propagator_indices=(("i", 2),) if indexed else (),
                         n_external_response=0)
    coeff = np.array([scale, -0.25 * scale]) if indexed else np.array(scale)
    ig = DiagramIntegrand(dt, None, coeff)
    r = np.full(3, 1.0 / scale)
    if dynamic:
        samples = np.broadcast_to(coeff, (3,) + coeff.shape)
        got = ig._dynamic_values(samples, [], r, np.ones(3))
    else:
        got = ig._static_values([], r, np.ones(3))
    expected = np.full(3, 0.75 if indexed else 1.0)
    np.testing.assert_allclose(got, expected, rtol=2e-15, atol=0.0)


@pytest.mark.parametrize("dynamic", [False, True])
def test_small_nonzero_wrong_phase_is_not_silently_pruned(dynamic):
    dt = SimpleNamespace(propagator_indices=(("i", 1),),
                         n_external_response=0)
    ig = DiagramIntegrand(dt, None, np.array([1e-21j]))
    with pytest.raises(ValueError):
        if dynamic:
            ig._dynamic_values(np.full((2, 1), 1e-21j), [], np.ones(2))
        else:
            ig._static_values([], np.ones(2), np.ones(2))
