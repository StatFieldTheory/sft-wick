"""Invocation-local reductions: independent arithmetic and semantic gates."""

from types import SimpleNamespace

import numpy as np
import pytest

from sft_wick import evaluate as ev


def integrand(coeff=None, phase=0):
    dt = SimpleNamespace(
        propagator_indices=(("i", 2), ("j", 2)), n_external_response=phase
    )
    if coeff is None:
        coeff = np.array([[0.7, -0.3], [0.2, 1.4]])
    return ev.DiagramIntegrand(dt, None, coeff)


def values(ig, c, rb=(), dynamic=False):
    if dynamic:
        return ig._dynamic_values(
            np.broadcast_to(ig.coupling_array, (3, 2, 2)),
            c,
            np.ones(3),
            np.ones(3),
            r_batches=rb,
        )
    return ig._static_values(c, np.ones(3), np.ones(3), r_batches=rb)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("kind", ["iso", "ctrace", "rtrace"])
def test_reduction_once_and_independent_contraction(monkeypatch, dynamic, kind):
    ig = integrand()
    R = np.broadcast_to(np.array([[0.7, -0.4], [0.25, 1.3]]), (3, 2, 2)).copy()
    C = np.broadcast_to(np.array([[2.0, -0.8], [0.4, 3.0]]), (3, 2, 2)).copy()
    if kind == "iso":
        C[:] = 2 * np.eye(2)
    cb = (
        [(C, None, None)]
        if kind == "iso"
        else [(C, "missing" if kind == "ctrace" else "i", "j")]
    )
    rb = [(R, "missing" if kind == "rtrace" else "i", "j", np.ones((2, 2), dtype=bool))]
    calls = {"iso": 0, "trace": 0}
    original_iso, original_einsum = ev._isotropic_C, np.einsum

    def iso(*a, **kw):
        calls["iso"] += 1
        return original_iso(*a, **kw)

    def einsum(spec, *a, **kw):
        if spec == "iaa->i":
            calls["trace"] += 1
        return original_einsum(spec, *a, **kw)

    monkeypatch.setattr(ev, "_isotropic_C", iso)
    monkeypatch.setattr(np, "einsum", einsum)
    # Explicit scalar contraction, independent of package selectors/reductions.
    expected = 0.0
    for a in range(2):
        for b in range(2):
            rf = 2.0 if kind == "rtrace" else R[0, a, b]
            cf = 2.0 if kind == "iso" else (5.0 if kind == "ctrace" else C[0, a, b])
            expected += ig.coupling_array[a, b] * rf * cf
    np.testing.assert_allclose(values(ig, cb, rb, dynamic), expected, rtol=2e-15)
    assert calls == {"iso": int(kind == "iso"), "trace": int(kind != "iso")}


@pytest.mark.parametrize("dynamic", [False, True])
def test_unused_invalid_c_and_phase_order(dynamic):
    malformed = [(np.ones((3, 1, 2, 2)), None, None)]
    zeros = np.zeros((3, 2, 2))
    rb = [(zeros, "i", "j", np.zeros((2, 2), dtype=bool))]
    np.testing.assert_array_equal(
        values(integrand(np.zeros((2, 2))), malformed, dynamic=dynamic), 0
    )
    np.testing.assert_array_equal(values(integrand(), malformed, rb, dynamic), 0)
    with pytest.raises(ValueError, match="phase|real|imaginary"):
        values(integrand(np.ones((2, 2)) * 1j), malformed, rb, dynamic)


@pytest.mark.parametrize("dynamic", [False, True])
def test_no_retained_numeric_state(dynamic):
    ig = integrand(np.ones((2, 2)))
    C = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()
    cb = [(C, None, None)]
    np.testing.assert_array_equal(values(ig, cb, dynamic=dynamic), 4.0)
    ig.coupling_array[0, 0] = 0.0
    C *= 3.0
    np.testing.assert_array_equal(values(ig, cb, dynamic=dynamic), 9.0)
    ig.coupling_array[0, 0] = 1e-100
    C[:] = 1e100 * np.eye(2)
    ig.coupling_array[0, 1:] = 0.0
    ig.coupling_array[1, :] = 0.0
    np.testing.assert_allclose(values(ig, cb, dynamic=dynamic), 1.0, rtol=2e-15)


def test_scalar_static_coefficient_broadcasts_over_all_assignments(monkeypatch):
    ig = integrand(np.array(1.0))
    C = np.broadcast_to(2.0 * np.eye(2), (3, 2, 2)).copy()
    calls = []
    original = ev._isotropic_C

    def iso(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(ev, "_isotropic_C", iso)
    np.testing.assert_array_equal(values(ig, [(C, None, None)]), 8.0)
    assert len(calls) == 1


def test_ndarray_subclass_can_mutate_borrowed_c_buffer():
    C = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()

    class StatefulCoefficients(np.ndarray):
        reads = 0

        def __getitem__(self, key):
            self.reads += 1
            C[:] = self.reads * np.eye(2)
            return super().__getitem__(key)

    coeff = np.ones((2, 2)).view(StatefulCoefficients)
    ig = integrand(coeff)
    np.testing.assert_array_equal(values(ig, [(C, None, None)]), 10.0)


@pytest.mark.parametrize("dynamic", [False, True])
def test_free_labels_override_pins_and_literals_are_one_indexed(dynamic):
    ig = integrand(np.ones((2, 2)))
    ig.fixed_indices.update(i=1, pinned=0)
    C = np.broadcast_to(np.diag([2.0, 3.0]), (3, 2, 2)).copy()
    cb = [(C, "i", "1"), (C, "missing", "pinned"), (C, "pinned", "pinned")]
    # i=0 appears twice, each contributing C00 * trace(C); i=1 gives zero.
    np.testing.assert_array_equal(values(ig, cb, dynamic=dynamic), 40.0)
    ig.fixed_indices["pinned"] = 1
    np.testing.assert_array_equal(values(ig, cb, dynamic=dynamic), 60.0)


def test_custom_reality_projector_retains_uncached_mutation():
    C = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()

    class Projector:
        calls = 0

        def project(self, raw, phase, where, key):
            self.calls += 1
            C[:] = self.calls * np.eye(2)
            return raw

    ig = integrand(np.ones((2, 2)))
    got = ig._dynamic_values(
        np.ones((3, 2, 2)), [(C, None, None)], np.ones(3), _reality=Projector()
    )
    np.testing.assert_array_equal(got, 10.0)


def test_builtin_projector_with_instance_override_uses_original_path():
    C = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()
    projector = ev._BatchRealityAccumulator()
    calls = []

    def project(raw, phase, where, key):
        calls.append(key)
        C[:] = len(calls) * np.eye(2)
        return raw

    class Project:
        def __call__(self, *args):
            return project(*args)

        def __getattr__(self, name):
            raise AssertionError("eligibility must not inspect a custom callable")

    projector.project = Project()
    ig = integrand(np.ones((2, 2)))
    got = ig._dynamic_values(
        np.ones((3, 2, 2)), [(C, None, None)], np.ones(3), _reality=projector
    )
    np.testing.assert_array_equal(got, 10.0)


def test_instance_helper_override_preserves_original_path():
    C = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()
    ig = integrand(np.ones((2, 2)))

    class RFactor:
        def __call__(self, r, batches, idx):
            C[:] *= 2
            return r

        def __eq__(self, other):
            raise AssertionError("eligibility must use function identity")

    object.__setattr__(ig, "_r_factor", RFactor())
    np.testing.assert_array_equal(values(ig, [(C, None, None)]), 30.0)


@pytest.mark.parametrize("dynamic", [False, True])
def test_custom_component_dimension_is_not_preinspected(dynamic):
    C = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()

    class Dimension(int):
        calls = 0

        def __gt__(self, other):
            self.calls += 1
            C[:] = self.calls * np.eye(2)
            return super().__gt__(other)

    dim = Dimension(2)
    dt = SimpleNamespace(
        propagator_indices=(("i", dim), ("j", 2)), n_external_response=0
    )
    ig = ev.DiagramIntegrand(dt, None, np.ones((2, 2)))
    np.testing.assert_array_equal(values(ig, [(C, None, None)], dynamic=dynamic), 4.0)


@pytest.mark.parametrize("dynamic", [False, True])
def test_custom_phase_can_mutate_borrowed_c(dynamic):
    C = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()

    class Phase(int):
        calls = 0

        def __int__(self):
            self.calls += 1
            C[:] = self.calls * np.eye(2)
            return super().__int__()

    ig = integrand(np.ones((2, 2)), phase=Phase(0))
    np.testing.assert_array_equal(values(ig, [(C, None, None)], dynamic=dynamic), 10.0)


def test_custom_integrand_helper_retains_original_signature():
    C = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()

    class CustomIntegrand(ev.DiagramIntegrand):
        @staticmethod
        def _c_product(cb, idx, n):
            C[:] *= 2.0
            return ev.DiagramIntegrand._c_product(cb, idx, n)

    original = integrand(np.ones((2, 2)))
    ig = CustomIntegrand(original.diagram_term, None, original.coupling_array)
    np.testing.assert_array_equal(values(ig, [(C, None, None)]), 30.0)


@pytest.mark.parametrize("dynamic", [False, True])
def test_diagonal_trace_fallback(dynamic):
    ig = integrand(np.ones((2, 2)))
    C = np.broadcast_to(np.array([2.0, 3.0]), (3, 2)).copy()
    np.testing.assert_array_equal(
        values(ig, [(C, "missing", "j")], dynamic=dynamic), 20.0
    )


@pytest.mark.parametrize("dynamic", [False, True])
def test_custom_integer_pin_can_mutate_borrowed_c(dynamic):
    C = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()

    class Pin(int):
        calls = 0

        def __ne__(self, other):
            self.calls += 1
            C[:] = self.calls * np.eye(2)
            return super().__ne__(other)

    ig = integrand(np.ones((2, 2)))
    ig.fixed_indices["pin"] = Pin(0)
    cb = [(np.ones((3, 2)), "pin", "pin"), (C, None, None)]
    np.testing.assert_array_equal(values(ig, cb, dynamic=dynamic), 10.0)


def test_coefficient_r_arithmetic_precedes_c_validation():
    coeff = np.zeros((2, 2))
    coeff[0, 0] = 1e308
    ig = integrand(coeff)
    with np.errstate(over="raise"), pytest.raises(FloatingPointError, match="multiply"):
        ig._static_values(
            [(np.ones((3, 1, 2, 2)), None, None)], np.full(3, 1e308), np.ones(3)
        )


@pytest.mark.parametrize("iso_c", [False, True])
def test_public_mean_against_independent_white_noise_contraction(iso_c):
    from scipy.integrate import quad
    import sft_wick as sw
    from sft_wick.workflow.specs import ConstantImpulse

    F = np.array([[[0.7, -0.3], [0.2, 1.4]], [[0.1, 0.6], [-0.8, 0.4]]])

    def response(t, s):
        d = t - s
        cs, sn = np.cos(0.3 * d), np.sin(0.3 * d)
        return np.exp(-d) * np.array([[cs, -sn], [sn, cs]])

    def covariance(x, t, y, s):
        t, s = np.broadcast_arrays(np.asarray(t), np.asarray(s))
        c = np.exp(-np.abs(t - s)) * (-np.expm1(-2 * np.minimum(t, s))) / 2
        angle = 0.3 * (t - s)
        out = np.zeros(t.shape + (2, 2))
        out[..., 0, 0] = out[..., 1, 1] = c * np.cos(angle)
        out[..., 0, 1], out[..., 1, 0] = -c * np.sin(angle), c * np.sin(angle)
        return out

    system = sw.System(
        field=sw.FieldSpec("phi", 2),
        linear=sw.ExplicitR(R_time=response, iso_R=False),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=1.0),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            ),
            sigma2=ConstantImpulse(np.eye(2)),
        ),
    )
    props = system.propagators(
        t_max=1.1,
        c_closed_form=covariance,
        c_closed_form_only=True,
        c_closed_form_vectorized=True,
        diag_C=False,
        progress=False,
    )
    exp = system.expand(
        ("phi_a(x)",), orders=[1], iso_C=iso_c, diag_R=False, diag_C=False
    )
    expected, error = quad(
        lambda u: (1 - np.exp(-2 * u))
        / 2
        * np.exp(-(1 - u))
        * (
            np.cos(0.3 * (1 - u)) * (F[0, 0, 0] + F[0, 1, 1])
            - np.sin(0.3 * (1 - u)) * (F[1, 0, 0] + F[1, 1, 1])
        ),
        0.0,
        1.0,
        epsabs=1e-13,
        epsrel=1e-13,
    )
    assert error < 1e-12
    for method, extra in (
        ("gauss_legendre", {}),
        ("gauss_legendre", {"gl_chunk_size": 7}),
        ("qmc_vectorized", {"n_samples": 2**12, "seed": 17}),
    ):
        result = exp.evaluate(
            props,
            positions={"x": 0.0},
            t_final=1.0,
            component_pair=(0,),
            orders=[1],
            method=method,
            n_gauss=16,
            **extra,
        )
        assert result.total == pytest.approx(
            expected, rel=3e-4 if method == "qmc_vectorized" else 2e-13, abs=0.0
        )


@pytest.mark.parametrize("dynamic", [False, True])
def test_repeated_factors_and_nonfinite_zero_behavior(dynamic):
    ig = integrand(np.array([[1.0, 0.0], [0.0, 0.0]]))
    C = np.broadcast_to(2.0 * np.eye(2), (3, 2, 2)).copy()
    cb = [(C, None, None), (C, None, None)]
    np.testing.assert_array_equal(values(ig, cb, dynamic=dynamic), 4.0)
    # The C-product is still multiplied in the old order. No zero-C prune.
    inf = np.broadcast_to(np.full((2, 2), np.inf), (3, 2, 2)).copy()
    zero = np.zeros((3, 2))
    with np.errstate(invalid="ignore"):
        got = values(ig, [(zero, "i", "j"), (inf, "missing", "j")], dynamic=dynamic)
    assert np.all(np.isnan(got))
