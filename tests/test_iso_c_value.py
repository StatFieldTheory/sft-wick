"""An index-free C propagator stands for ``c`` in ``C_ab = δ_ab c``.

``compute_moment(..., iso_C=True)`` strips the equal component indices from
a C propagator and keeps any index sum in the coupling.  Up to 0.4.2 every
evaluator then multiplied by the trace ``N c``: at N = 2 the order-0
``⟨φ_0 φ_0⟩`` came out as ``2 C_00`` and an order-2 diagram with two C
propagators 4 times too large, on every backend and at L0 and L1.

Reference: the default ``iso_C=False`` expansion of the same system, whose
C propagators keep their indices; ``iso_C`` is a simplification and must
not change any value.  A C that is not a multiple of the identity has no
single ``c`` and is refused.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.workflow.specs import GeneralKappa2

N = 2
T_FINAL = 1.6
POS = {"x": 0.0, "y": 0.35}
F = np.array([[[0.2, -0.7], [0.4, 0.1]],
              [[0.9, 0.3], [-0.5, 0.6]]])

METHODS = ["gauss_legendre", "qmc_vectorized", "qmc_scalar", "nquad"]


def _system(kappa2=None):
    kappa2 = kappa2 if kappa2 is not None else sw.SeparableTranslation(
        temporal=sw.ExponentialTemporal(lam=0.6, sigma_t=0.5),
        spatial=sw.ExponentialSpatial(sigma_x=1.0))
    return sw.System(field=sw.FieldSpec("phi", N),
                     linear=sw.DiagonalA(gamma=[1.0, 1.0]),
                     vertices=[sw.LocalVertex("F", coupling=F)],
                     noise=sw.GaussianNoise(kappa2=kappa2))


@pytest.fixture(scope="module")
def isotropic():
    system = _system()
    props = system.propagators(t_max=2.0, n_grid_t=12, c_closed_form="auto",
                               c_closed_form_only=True, progress=False)
    return system, props


def _evaluate(system, props, obs, order, comps, method, iso_C):
    exp = system.expand(obs, orders=[order], iso_C=iso_C)
    return exp.evaluate(props, positions=POS, t_final=T_FINAL,
                        component_pair=comps, orders=[order], method=method,
                        n_gauss=8, n_samples=2**9, seed=5).total


CASES = [
    (("phi_a(x)", "phi_b(y)"), 0, (0, 0)),
    (("phi_a(x)", "phi_b(y)"), 0, (0, 1)),
    (("phi_a(x)",), 1, (0,)),
    (("phi_a(x)",), 1, (1,)),
    (("phi_a(x)", "phi_b(y)"), 2, (0, 0)),
    (("phi_a(x)", "phi_b(y)"), 2, (0, 1)),
]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("obs,order,comps", CASES,
                         ids=[f"order{o}-{c}" for _, o, c in CASES])
def test_iso_C_changes_no_value(isotropic, obs, order, comps, method):
    system, props = isotropic
    if method in ("qmc_scalar", "nquad") and order == 2:
        pytest.skip("scalar-loop cost; order 2 is covered by the batched "
                    "backends, and the scalar loop at orders 0-1")
    plain = _evaluate(system, props, obs, order, comps, method, iso_C=False)
    iso = _evaluate(system, props, obs, order, comps, method, iso_C=True)
    if comps == (0, 1) and order == 0:
        assert plain == 0.0 and iso == 0.0
        return
    assert plain != 0.0
    assert iso == pytest.approx(plain, rel=1e-10, abs=0.0)


@dataclass(frozen=True)
class UnequalDiagonal:
    def __call__(self, n1, t1, n2, t2):
        return np.exp(-abs(t1 - t2) / 0.5) * np.diag([0.6, 0.35])


def test_a_non_isotropic_C_is_refused_under_iso_C():
    system = _system(GeneralKappa2(fn=UnequalDiagonal()))
    props = system.propagators(t_max=2.0, n_grid_t=12, progress=False)
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0], iso_C=True)
    with pytest.raises(ValueError, match="iso_C"):
        exp.evaluate(props, positions=POS, t_final=T_FINAL,
                     component_pair=(0, 0), orders=[0],
                     method="gauss_legendre")
