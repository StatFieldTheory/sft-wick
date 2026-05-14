"""Regression tests for ``NonLocalVertex(equal_time=True)``.

Verifies that the m time legs of an equal-time non-local vertex
collapse into a single integration variable while the m spatial legs
remain independent — matching the cosmological equal-shell cumulant
convention (e.g. ``canoes.sachs.compute_kappa3_zeta_table``).

Failure modes guarded against:
    1. SpatialStructure leaks the m aliased labels into
       ``time_integration_vars`` (Jacobian over-counts by ``width^{m-1}``).
    2. The dynamic-coupling callable receives m distinct times instead of
       one shared time across the m legs.
    3. Static-coupling ``DiagramIntegrand.evaluate`` raises ``KeyError``
       on an aliased label that's absent from the per-sample ``times``
       dict.
    4. Existing ``equal_time=False`` (default) behaviour is unchanged.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sft_wick.evaluate import SpatialStructure, analyze_spatial
from sft_wick.expressions import Symbol
from sft_wick.fields import Field, reset_uid_counter
from sft_wick.indices import IndexContext
from sft_wick.perturbation import DiagramTerm, Propagator, Rational
from sft_wick.vertices import Vertex, VertexInstance
from sft_wick.workflow.specs import NonLocalVertex


@pytest.fixture(autouse=True)
def _reset_uids():
    """Reset the global FieldOperator UID counter before every test.

    Several assertions below depend on the ordering of spatial-variable
    labels (e.g. ``inst.spatial_variables`` -> ``(legs[1], legs[2])``
    aliasing to ``legs[0]``). Without this reset, test ordering can
    silently change the labels and break the alias assertions.
    """
    reset_uid_counter()
    yield


# ----------------------------------------------------------------------
# Layer 1: spec / vertex / instance plumbing
# ----------------------------------------------------------------------


def test_nonlocal_vertex_accepts_equal_time_flag():
    spec = NonLocalVertex(
        name="K", order=3, coupling=np.zeros((3, 3, 3)), equal_time=True,
    )
    assert spec.equal_time is True
    # Default is False.
    default_spec = NonLocalVertex(
        name="K", order=3, coupling=np.zeros((3, 3, 3)),
    )
    assert default_spec.equal_time is False


def test_vertex_carries_equal_time():
    psi = Field("psi", "response", n_components=3)
    v = Vertex(
        fields=[psi, psi, psi], coupling="K",
        local=False, equal_time=True,
    )
    assert v.equal_time is True


def test_vertex_instance_builds_alias_map_for_equal_time():
    psi = Field("psi", "response", n_components=3)
    v = Vertex(
        fields=[psi, psi, psi], coupling="K",
        local=False, equal_time=True,
    )
    inst = VertexInstance.instantiate(v, IndexContext(), copy_id=0)
    legs = inst.spatial_variables
    assert len(legs) == 3
    assert inst.equal_time_aliases == ((legs[1], legs[0]), (legs[2], legs[0]))


def test_vertex_instance_alias_empty_when_not_equal_time():
    psi = Field("psi", "response", n_components=3)
    v = Vertex(
        fields=[psi, psi, psi], coupling="K",
        local=False, equal_time=False,
    )
    inst = VertexInstance.instantiate(v, IndexContext(), copy_id=0)
    assert inst.equal_time_aliases == ()


def test_local_vertex_alias_always_empty():
    phi = Field("phi", "physical", n_components=3)
    psi = Field("psi", "response", n_components=3)
    v = Vertex(fields=[psi, phi, phi], coupling="F", local=True)
    inst = VertexInstance.instantiate(v, IndexContext(), copy_id=0)
    assert inst.equal_time_aliases == ()


def test_local_vertex_with_equal_time_true_raises():
    """``equal_time=True`` is meaningless on a local vertex (one shared
    spatial / time leg already) — must fail loudly at construction
    rather than silently leaving the spurious ``t_max^(m-1)`` factor in
    place that the equal-time mode is supposed to remove."""
    phi = Field("phi", "physical", n_components=3)
    psi = Field("psi", "response", n_components=3)
    with pytest.raises(ValueError, match="non-local"):
        Vertex(
            fields=[psi, phi, phi], coupling="F",
            local=True, equal_time=True,
        )


def test_vertex_instance_alias_empty_for_order_1_nonlocal():
    """Order-1 (m=1) non-local vertex has nothing to collapse — alias
    map must stay empty even with ``equal_time=True``."""
    psi = Field("psi", "response", n_components=1)
    v = Vertex(
        fields=[psi], coupling="K",
        local=False, equal_time=True,
    )
    inst = VertexInstance.instantiate(v, IndexContext(), copy_id=0)
    assert inst.equal_time_aliases == ()


def test_vertex_instance_alias_for_order_2_nonlocal():
    """m=2 is the smallest non-trivial collapse case — one alias pair."""
    psi = Field("psi", "response", n_components=1)
    v = Vertex(
        fields=[psi, psi], coupling="K",
        local=False, equal_time=True,
    )
    inst = VertexInstance.instantiate(v, IndexContext(), copy_id=0)
    legs = inst.spatial_variables
    assert len(legs) == 2
    assert inst.equal_time_aliases == ((legs[1], legs[0]),)


# ----------------------------------------------------------------------
# Layer 2: SpatialStructure filtering
# ----------------------------------------------------------------------


def _make_diagram_with_equal_time_aliases(aliases):
    """Build a minimal DiagramTerm carrying the requested alias tuple."""
    # 3 internal labels: s_a (rep), s_b, s_c. R-propagators pair them
    # arbitrarily so all 3 are real internal labels in integration_vars.
    props = (
        Propagator("R", None, None, "s_a", "x_ext"),
        Propagator("R", None, None, "s_b", "x_ext"),
        Propagator("R", None, None, "s_c", "x_ext"),
    )
    return DiagramTerm(
        propagators=props,
        coupling_sum=Symbol(
            "K", indices=(), spatial_args=("s_a", "s_b", "s_c"),
        ),
        rational_prefactor=Rational(1, 1),
        integration_vars=("s_a", "s_b", "s_c"),
        summation_indices=(),
        n_response=0,
        equal_time_aliases=aliases,
    )


class _UnitCache:
    """Tiny cache double for measure-only equal-time tests."""

    def __init__(self):
        self.model = SimpleNamespace(iso_R=True, diag_C=True)
        self.r_calls = []
        self.c_calls = []

    def R_product(self, r_propagators, times):
        out = 1.0
        for sl, sr in r_propagators:
            self.r_calls.append((sl, sr, times[sl], times[sr]))
            out *= 1.0
        return out

    def R_time_batch(self, t_left, t_right):  # noqa: ARG002
        return np.ones_like(np.asarray(t_left, dtype=float))

    def C_value(self, n_left, t_left, n_right, t_right):  # noqa: ARG002
        self.c_calls.append((t_left, t_right))
        return np.ones((1, 1))

    def C_diagonal_batch(self, t_left, t_right):  # noqa: ARG002
        return np.ones((len(t_left), 1))


def test_analyze_spatial_filters_aliased_legs_from_time_integration():
    dt = _make_diagram_with_equal_time_aliases(
        (("s_b", "s_a"), ("s_c", "s_a")),
    )
    ss = analyze_spatial(dt)
    # Only the canonical representative remains as an integration var.
    assert ss.time_integration_vars == ("s_a",)
    # The aliases are exposed on SpatialStructure.
    assert dict(ss.equal_time_aliases) == {"s_b": "s_a", "s_c": "s_a"}
    # Spatial structure (direction_map) still includes all labels —
    # only the time integration is collapsed, not the spatial.
    assert {"s_a", "s_b", "s_c"}.issubset(ss.direction_map.keys())


def test_analyze_spatial_no_filter_when_no_aliases():
    dt = _make_diagram_with_equal_time_aliases(())
    ss = analyze_spatial(dt)
    assert set(ss.time_integration_vars) == {"s_a", "s_b", "s_c"}
    assert ss.equal_time_aliases == ()


def test_static_evaluate_fills_equal_time_aliases_without_keyerror():
    dt = _make_diagram_with_equal_time_aliases(
        (("s_b", "s_a"), ("s_c", "s_a")),
    )
    ig = dt.build_integrand({"K": np.array(1.0)})
    cache = _UnitCache()

    val = ig.evaluate({"s_a": 0.25, "x_ext": 1.0}, {}, cache)

    assert val == pytest.approx(1.0 + 0.0j)
    assert ("s_b", "x_ext", 0.25, 1.0) in cache.r_calls
    assert ("s_c", "x_ext", 0.25, 1.0) in cache.r_calls


def test_dynamic_callable_receives_shared_times_for_equal_time_legs():
    seen_t_lists: list[np.ndarray] = []

    def K_dynamic(n_list, t_list):  # noqa: ARG001
        seen_t_lists.append(np.asarray(t_list, dtype=float))
        return 1.0

    dt = _make_diagram_with_equal_time_aliases(
        (("s_b", "s_a"), ("s_c", "s_a")),
    )
    ig = dt.build_integrand({"K": K_dynamic})

    val, err = ig.integrate_moment_gauss_legendre(
        lambda_f=2.0,
        cache=_UnitCache(),
        t_min=0.0,
        n_gauss=2,
    )

    assert err == 0.0
    assert val == pytest.approx(2.0)
    assert seen_t_lists
    for t_list in seen_t_lists:
        assert t_list.shape == (3,)
        assert t_list[0] == pytest.approx(t_list[1])
        assert t_list[0] == pytest.approx(t_list[2])


def test_equal_time_jacobian_ratio_for_constant_integrand():
    span = 2.0
    dt_full = _make_diagram_with_equal_time_aliases(())
    dt_equal_time = _make_diagram_with_equal_time_aliases(
        (("s_b", "s_a"), ("s_c", "s_a")),
    )
    ig_full = dt_full.build_integrand({"K": np.array(1.0)})
    ig_equal_time = dt_equal_time.build_integrand({"K": np.array(1.0)})

    full, _ = ig_full.integrate_moment_gauss_legendre(
        lambda_f=span,
        cache=_UnitCache(),
        t_min=0.0,
        n_gauss=2,
    )
    equal_time, _ = ig_equal_time.integrate_moment_gauss_legendre(
        lambda_f=span,
        cache=_UnitCache(),
        t_min=0.0,
        n_gauss=2,
    )

    # full integrates m=3 independent times over [0, span], so volume
    # is span^3. equal_time collapses all 3 to one shared time, so
    # volume is span^1. The ratio span^(1-m) is the spurious Jacobian
    # factor (t_max)^(m-1) that the equal_time flag removes.
    assert full == pytest.approx(span ** 3)
    assert equal_time == pytest.approx(span)


def test_equal_time_jacobian_ratio_qmc_vectorized():
    """Same Jacobian-ratio check as above, but exercising the
    ``integrate_moment_qmc_vectorized`` path — the production default
    integrator (``SweepConfig.method='qmc_vectorized'``). The GL test
    above does NOT cover this code path; the ``_times`` resolver in
    the QMC kernel has its own alias-redirect block."""
    span = 2.0
    dt_full = _make_diagram_with_equal_time_aliases(())
    dt_equal_time = _make_diagram_with_equal_time_aliases(
        (("s_b", "s_a"), ("s_c", "s_a")),
    )
    ig_full = dt_full.build_integrand({"K": np.array(1.0)})
    ig_equal_time = dt_equal_time.build_integrand({"K": np.array(1.0)})

    full, _ = ig_full.integrate_moment_qmc_vectorized(
        lambda_f=span, cache=_UnitCache(), t_min=0.0,
        n_samples=2**12, seed=42,
    )
    equal_time, _ = ig_equal_time.integrate_moment_qmc_vectorized(
        lambda_f=span, cache=_UnitCache(), t_min=0.0,
        n_samples=2**12, seed=42,
    )

    # Constant integrand (R=C=1, coupling=1) so QMC integrates the
    # raw simplex volume exactly up to QMC bias. 2**12 Sobol samples
    # nail span^3 = 8 and span = 2 to ~0.1% on this trivial fixture.
    assert full == pytest.approx(span ** 3, rel=1e-3)
    assert equal_time == pytest.approx(span, rel=1e-3)


def test_equal_time_qmc_vectorized_alias_redirect_in_times():
    """Check the ``_times`` resolver inside the QMC-vectorized kernel
    actually redirects aliased leg labels to their canonical rep —
    not just produces the right total. Uses a tracking cache that
    records every ``R_time_batch`` (left, right) pair; for the
    equal_time diagram the three R-propagators must all see the
    SAME t_left array (since s_a, s_b, s_c collapse to s_a)."""

    class _Tracker:
        def __init__(self):
            self.model = SimpleNamespace(iso_R=True, diag_C=True)
            self.r_left_arrays: list[np.ndarray] = []

        def R_time_batch(self, t_left, t_right):  # noqa: ARG002
            self.r_left_arrays.append(np.asarray(t_left, dtype=float))
            return np.ones_like(np.asarray(t_left, dtype=float))

        def C_diagonal_batch(self, t_left, t_right):  # noqa: ARG002
            return np.ones((len(t_left), 1))

    dt = _make_diagram_with_equal_time_aliases(
        (("s_b", "s_a"), ("s_c", "s_a")),
    )
    ig = dt.build_integrand({"K": np.array(1.0)})
    tracker = _Tracker()
    ig.integrate_moment_qmc_vectorized(
        lambda_f=2.0, cache=tracker, t_min=0.0,
        n_samples=2**10, seed=7,
    )
    # Diagram has 3 R-propagators (s_a, s_b, s_c) -> x_ext. All three
    # left-times must be identical after alias redirect.
    assert len(tracker.r_left_arrays) == 3
    np.testing.assert_allclose(
        tracker.r_left_arrays[0], tracker.r_left_arrays[1],
    )
    np.testing.assert_allclose(
        tracker.r_left_arrays[0], tracker.r_left_arrays[2],
    )


# ----------------------------------------------------------------------
# Layer 3: end-to-end measure correction (constant coupling, no R/C
# variation in time)
# ----------------------------------------------------------------------
#
# For a uniform integrand over the m time legs of a single equal_time
# vertex, the equal_time=True result should differ from the equal_time
# =False result by a factor of (lambda_f - t_min)^(m-1). This catches
# Jacobian errors directly.


def _equal_time_jacobian_ratio_test_setup():
    """Build a minimal toy environment so the unit test runs in <1s.

    Returns the result-pair (equal_time, full) for a single FK-like
    diagram with constant ``R(t,t') = 1`` and a constant ``ζ`` callable.
    """
    from sft_wick.workflow.specs import (
        ConstantImpulse, DiagonalA, FieldSpec, GaussianNoise,
        NonLocalVertex, SeparableTranslation,
        ExponentialSpatial, ExponentialTemporal,
    )
    from sft_wick.workflow.system import System

    # Use a constant κ³ to expose the measure-only difference.
    def kappa3_constant(n_list, t_list):
        # Returns a (1,1,1) tensor with constant value 1.0; the legs
        # carry no time dependence so the integrand reduces to the
        # product of R-products and the time-volume measure.
        return np.ones((1, 1, 1))

    common = dict(
        field=FieldSpec(name="phi", n_components=1),
        linear=DiagonalA(gamma=[0.0]),  # R(t, t') = Theta(t - t')
        noise=GaussianNoise(
            kappa2=SeparableTranslation(
                temporal=ExponentialTemporal(lam=0.0, sigma_t=1.0),
                spatial=ExponentialSpatial(sigma_x=1.0),
            ),
            sigma2=ConstantImpulse(amplitude=1.0),
        ),
        vertices=(),
    )
    sys_et = System(
        nonlocal_vertices=(
            NonLocalVertex(
                name="K", order=3, coupling=kappa3_constant,
                equal_time=True,
            ),
        ),
        **common,
    )
    sys_full = System(
        nonlocal_vertices=(
            NonLocalVertex(
                name="K", order=3, coupling=kappa3_constant,
                equal_time=False,
            ),
        ),
        **common,
    )
    return sys_et, sys_full


def test_equal_time_collapses_time_integrations_in_diagram_term():
    """Confirm that DiagramTerm.equal_time_aliases is propagated through
    a full expansion (collect_topology path)."""
    sys_et, _ = _equal_time_jacobian_ratio_test_setup()
    # Run the expansion at Order 1 (single K insertion). The simplest
    # diagram with a non-trivial K vertex is ⟨ψ ψ ψ⟩ at Order 1 — but
    # ⟨φ φ⟩ at Order 0 plus a K loop is also possible. Stick with
    # ⟨ψ ψ ψ⟩ Order-1 K: this is a single K-vertex with 3 internal
    # legs, none collapsed via R to an external. The 3 legs are tied
    # by the equal_time flag.
    exp = sys_et.expand(
        observable=["phi(x)", "phi(y)", "phi(z)"],
        orders=[1],
        response_phase=True,
        ito=True,
        collect_topology=True,
        iso_R=True,
        diag_R=True,
        diag_C=True,
    )
    # At least one DiagramTerm at order=1 should carry the alias map.
    dterms = exp.dts_by_order.get(1, [])
    assert dterms, "Expected at least one Order-1 ⟨φφφ⟩ K diagram term."
    aliased = [dt for dt in dterms if dt.equal_time_aliases]
    assert aliased, (
        "Order-1 K diagram lost equal_time_aliases when expanded — "
        "this means the alias info isn't being threaded from "
        "VertexInstance into DiagramTerm."
    )
    # Check the alias structure: 3 K legs → 2 entries (non-rep → rep).
    for dt in aliased:
        assert len(dt.equal_time_aliases) == 2, dt.equal_time_aliases
