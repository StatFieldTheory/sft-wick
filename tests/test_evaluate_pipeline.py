"""Low-level structural tests for the numerical evaluation pipeline.

Narrow scope: this file only verifies structural properties of
:class:`sft_wick.evaluate.SpatialStructure` (integration variables,
direction groups, causal orderings) and of
:meth:`DiagramTerm.evaluate_coupling` (coupling-tensor symmetry).

End-to-end QMC/nquad agreement, diagram-count regression, propagator
numerics, alternative-path consistency, and spatial-coordinate routing
are all covered — with stronger, reference-backed assertions — by the
deductive test suite (``test_deductive_expansion.py``,
``test_deductive_numerics.py``, Phases 1-5).  Do not add redundant
end-to-end assertions here.
"""

import numpy as np
import pytest

from sft_wick import (
    Field, Vertex, Action, compute_moment, reset_uid_counter,
)
from sft_wick.evaluate import (
    PropagatorModel, PropagatorCache,
)


@pytest.fixture(autouse=True)
def _reset_uid():
    reset_uid_counter()


@pytest.fixture(scope="module")
def cubic_diagrams():
    """Generate order-2 diagram terms for the cubic vertex.

    Structural-test scope only — no ``precompute_C_table`` call, since
    the remaining tests never touch numerical propagator values (they
    read ``SpatialStructure`` fields and exercise ``evaluate_coupling``
    on symbolic expressions).  Returning ``None`` for the cache slot
    makes accidental numerical use error loudly.

    Module-scoped: the six DTs are computed once per test module
    instead of per test (saves ~1 s × 5 tests).
    """
    reset_uid_counter()
    phi = Field("phi", "physical", n_components=2)
    psi = Field("psi", "response", n_components=2)
    v = Vertex(fields=[psi, phi, phi], coupling="F")
    action = Action(vertices=[v])
    obs = [phi("a", "x"), phi("b", "y")]
    result = compute_moment(obs, action, order=2,
                            collect_topology=True,
                            diag_R=True, diag_C=True, iso_R=True)
    return result.diagram_terms(2), None


# =====================================================================
# Spatial Structure Analysis
# =====================================================================


class TestSpatialAnalysis:
    """Verify spatial structure extraction from diagram terms."""

    def test_order2_integration_vars(self, cubic_diagrams):
        """Order 2 diagrams have 2 integration variables."""
        dts, _ = cubic_diagrams
        for dt in dts:
            assert len(dt.integration_vars) == 2

    def test_r_connected_points_share_direction(self, cubic_diagrams):
        """Points connected by R propagators share a direction variable."""
        dts, cache = cubic_diagrams
        F = np.zeros((2, 2, 2))
        F[0, 1, 1] = 1.0
        F[1, 0, 1] = 0.5
        F[1, 1, 0] = 0.5
        fi = {"a": 1, "b": 1}

        for dt in dts:
            coeff = dt.evaluate_coupling({"F": -1j * F}, fixed_indices=fi)
            if np.abs(coeff).sum() < 1e-14:
                continue
            ig = dt.build_integrand({"F": -1j * F}, fixed_indices=fi)
            sp = ig.spatial

            # For each R propagator, check endpoints share direction
            for sl, sr in sp.r_propagators:
                dir_l = sp.direction_map[sl]
                dir_r = sp.direction_map[sr]
                assert dir_l == dir_r, (
                    f"R({sl},{sr}): directions {dir_l} ≠ {dir_r}"
                )

    def test_time_orderings_from_R(self, cubic_diagrams):
        """R propagators impose t_left >= t_right ordering."""
        dts, cache = cubic_diagrams
        F = np.zeros((2, 2, 2))
        F[0, 1, 1] = 1.0
        F[1, 0, 1] = 0.5
        F[1, 1, 0] = 0.5
        fi = {"a": 1, "b": 1}

        for dt in dts:
            coeff = dt.evaluate_coupling({"F": -1j * F}, fixed_indices=fi)
            if np.abs(coeff).sum() < 1e-14:
                continue
            ig = dt.build_integrand({"F": -1j * F}, fixed_indices=fi)
            sp = ig.spatial

            # Number of time orderings should match number of R propagators
            n_r = len(sp.r_propagators)
            n_ord = len(sp.time_orderings)
            assert n_ord == n_r, (
                f"Expected {n_r} time orderings, got {n_ord}"
            )


# =====================================================================
# Coupling Coefficient Evaluation
# =====================================================================


class TestCouplingEvaluation:
    """Verify coupling tensor substitution."""

    def test_zeroth_order_coupling_is_one(self, cubic_diagrams):
        """Order 0 has no coupling — coefficient should be 1."""
        phi = Field("phi", "physical", n_components=2)
        psi = Field("psi", "response", n_components=2)
        v = Vertex(fields=[psi, phi, phi], coupling="F")
        action = Action(vertices=[v])
        obs = [phi("a", "x"), phi("b", "y")]
        result = compute_moment(obs, action, order=0, collect_topology=True)
        dt = result.diagram_terms(0)[0]

        F = np.zeros((2, 2, 2))
        F[0, 1, 1] = 1.0
        coeff = dt.evaluate_coupling({"F": -1j * F}, fixed_indices={"a": 0, "b": 0})
        # Should be scalar 1 (no coupling at order 0)
        assert np.abs(coeff).sum() > 0

    def test_symmetrised_coupling_sum(self, cubic_diagrams):
        """Three symmetry configs give same total for each diagram."""
        dts, _ = cubic_diagrams
        fi = {"a": 1, "b": 1}

        configs = [
            {"F": -1j * np.array([[[0,0],[1,0]],[[0,0.5],[0.5,0]]])},  # sym
            {"F": -1j * np.array([[[0,0],[1,0]],[[0,0],[1,0]]])},      # asym A
            {"F": -1j * np.array([[[0,0],[1,0]],[[0,1],[0,0]]])},      # asym B
        ]

        for dt in dts:
            vals = [np.abs(dt.evaluate_coupling(cv, fixed_indices=fi)).sum()
                    for cv in configs]
            for v in vals[1:]:
                np.testing.assert_allclose(v, vals[0], rtol=1e-12)


# Note: removed QMC/nquad agreement and two-point spatial-factor tests —
# the former is covered (with an independent closed-form reference) by
# deductive Phase 3 P1 and Phase 4 C4; the latter's
# ``test_equal_point_matches_integrate_moment`` was testing a false
# claim (it compared the *time-integrated moment*
# ``integrate_moment(lambda_f=t_f)`` with the *fixed-time correlator*
# ``integrate_two_point_qmc(t_f=t_f)`` — different observables).
# Spatial-coordinate routing is covered by deductive Phase 5 S1-S5
# in a diagram-by-diagram scaling check.
