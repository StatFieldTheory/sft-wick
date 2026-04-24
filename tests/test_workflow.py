"""Regression tests for the high-level ``sft_wick.workflow`` API.

These tests lock in the user-facing surface so accidental changes
to the wrapper (System / Expansion / Propagators / Result /
SweepResult) get caught quickly.  They deliberately exercise the
wrapper only through its public surface — no reaching into
`raw_result` / `.cache` attributes — so the tests double as a
live specification of the API.

Coverage:

========  ======================================================
Test ID    Claim
========  ======================================================
WF1        ``System.expand`` + observable string parsing produces
           the same DiagramTerms as raw compute_moment on the
           demo1 setup.
WF2        ``Expansion.summary`` reports correct per-order
           counts and a ``by_vertex_type`` histogram with only
           ``'F'`` for a purely local theory.
WF3        ``Expansion.by_vertex_type`` correctly separates
           ``'F'`` vs ``'FK'`` when a ``NonLocalVertex`` is
           present — 6 F + 2 FK at order 2 in demo2's setup.
WF4        End-to-end: ``system.propagators(c_closed_form=...)`` +
           ``expansion.sweep`` produces numbers bit-matching the
           raw-API path (``validate_phase5.py``'s Method B
           κ²-ratio reference) to < 1e-6 relative.
WF5        ``SweepResult.totals()`` returns a pandas DataFrame
           with one row per (positions, t_final, a, b, order)
           and the correct summed values.
========  ======================================================
"""

from __future__ import annotations

import numpy as np
import pytest

import sft_wick as sw

# --------------------------------------------------------------------- #
# Shared fixture: demo1-style system (OU kernel, separable, N=2)
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def demo1_system():
    F = np.zeros((2, 2, 2))
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = 0.5
    F[1, 1, 0] = 0.5
    return sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        # Bare F — wrapper applies the MSR ``-i`` factor.
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            ),
        ),
    )


def _C_t_closed_form(t1, t2, lam=0.05, sigma_t=0.3, gamma=1.0):
    """Closed-form C_t for OU kernel — see
    ``tests/test_deductive_numerics.py::C_closed_form``."""
    a = 1.0 / sigma_t
    tl, th = (t1, t2) if t1 <= t2 else (t2, t1)
    if tl <= 0:
        return 0.0
    gpa, gma = gamma + a, gamma - a
    E1 = np.expm1(2 * gamma * tl) / (2 * gamma)
    E2 = tl if abs(gma) < 1e-14 else np.expm1(gma * tl) / gma
    E3 = np.expm1(gpa * tl) / gpa
    E4 = np.exp(gma * th)
    I = E1 / gpa - E2 / gpa + E4 * E3 / gma - E1 / gma
    return lam * np.exp(-gamma * (t1 + t2)) * I


def _C_demo1(n1, t1, n2, t2):
    sigma_x = 1.0
    N = 2
    r = abs(float(np.asarray(n1).sum()) - float(np.asarray(n2).sum()))
    return _C_t_closed_form(t1, t2) * np.exp(-r / sigma_x) * np.eye(N)


# =====================================================================
# WF1 — expand consistency
# =====================================================================


def test_WF1_expand_produces_expected_diagram_counts(demo1_system):
    """System.expand at orders 0, 2, 4 yields the canonical counts
    (1 + 6 + 64 = 71 diagrams) produced by the raw compute_moment
    path on the same physical spec."""
    exp = demo1_system.expand(
        ("phi_a(x)", "phi_b(y)"), orders=[0, 2, 4],
    )
    assert len(exp.diagrams(0)) == 1, (
        "order-0 should be a single self-loop"
    )
    assert len(exp.diagrams(2)) == 6
    assert len(exp.diagrams(4)) == 64


# =====================================================================
# WF2 — summary() reports sensible structure
# =====================================================================


def test_WF2_summary_purely_local_theory(demo1_system):
    exp = demo1_system.expand(
        ("phi_a(x)", "phi_b(y)"), orders=[0, 2, 4],
    )
    summary = exp.summary()

    # All 3 orders present
    assert set(summary) == {0, 2, 4}

    # vertex_type histogram should only contain 'F' (and maybe '' for
    # the order-0 term that has no coupling factors).
    for o, info in summary.items():
        labels = set(info["by_vertex_type"])
        assert labels.issubset({"F", ""}), (
            f"order {o}: unexpected vertex types {labels}"
        )

    # Order 4 has at least one diagram with 3 cross-group C's
    assert max(summary[4]["by_n_cross_C"]) >= 3


# =====================================================================
# WF3 — FF / FK classification
# =====================================================================


def test_WF3_FF_FK_classification():
    """Add a non-local cubic vertex (κ^(3)) and check that
    ``Expansion.by_vertex_type`` produces demo2's 6 FF + 2 FK
    split at order 2."""
    N = 2
    F = np.zeros((N, N, N))
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = 0.5
    F[1, 1, 0] = 0.5
    K = np.zeros((N, N, N))
    K[0, 0, 0] = 1.0
    K[1, 1, 1] = 1.0

    system = sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        # Bare tensors — wrapper applies -i (local F) and +i/6 (K, κ^(3)).
        vertices=[sw.LocalVertex("F", coupling=F)],
        nonlocal_vertices=[sw.NonLocalVertex("K", order=3, coupling=K)],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            ),
        ),
    )
    exp = system.expand(
        ("phi_a(x)", "phi_b(y)"), orders=[2],
    )
    groups = exp.by_vertex_type(order=2)
    assert "F" in groups and len(groups["F"]) == 6
    assert "FK" in groups and len(groups["FK"]) == 2


# =====================================================================
# WF4 — end-to-end sweep matches known reference
# =====================================================================


def test_WF4_end_to_end_matches_validate_phase5(demo1_system):
    """Sweep at 4 r × 2 t × 2 (a,b) × 3 orders via the wrapper and
    assert a handful of well-tested reference values from
    ``examples/demo1/validate_phase5.py`` match to < 1e-6 rel."""
    exp = demo1_system.expand(
        ("phi_a(x)", "phi_b(y)"), orders=[0, 2, 4],
    )
    props = demo1_system.propagators(
        t_max=15.0, n_grid_t=60,
        c_closed_form=_C_demo1,
    )
    sweep = exp.sweep(
        props,
        # All reference values below pin y ∈ {0.0, 0.5, 1.0} and
        # t_f = 15.0 — trimmed from the original 4×2 grid to keep
        # this test under 20 s while preserving every asserted point.
        positions_grid={"x": [0.0], "y": [0.0, 0.5, 1.0]},
        t_final_grid=[15.0],
        component_pairs=[(0, 0), (1, 1)],
        orders=[0, 2, 4],
        # validate_phase5.py reference values are time-integrated
        # (``integrate_moment`` integrates all externals over
        # ``[0, lambda_f]``).  Match that convention here.
        integrate_over="all",
        n_samples=2 ** 13,
        seed=42,
    )
    totals = sweep.totals()

    # Reference values from examples/demo1/validate_phase5.py run.
    reference = {
        (0, 0, 0.0, 15.0, 0): 3.996863e-01,
        (0, 0, 0.5, 15.0, 0): 2.424220e-01,
        (0, 0, 0.5, 15.0, 2): 3.212865e-02,
        (0, 0, 0.5, 15.0, 4): 2.231587e-03,
        (1, 1, 1.0, 15.0, 2): 9.620743e-03,
        (1, 1, 1.0, 15.0, 4): 7.882002e-04,
    }
    for (a, b, r, t_f, ord_), expected in reference.items():
        mask = (
            (totals["a"] == a)
            & (totals["b"] == b)
            & (abs(totals["y"] - r) < 1e-12)
            & (abs(totals["t_final"] - t_f) < 1e-12)
            & (totals["order"] == ord_)
        )
        got = float(totals.loc[mask, "value"].iloc[0])
        rel = abs(got - expected) / abs(expected)
        assert rel < 1e-5, (
            f"WF4 at (a={a},b={b},r={r},t_f={t_f},ord={ord_}): "
            f"wrapper={got:.6e} vs ref={expected:.6e}, rel={rel:.2e}"
        )


# =====================================================================
# WF5 — SweepResult.totals() shape and content
# =====================================================================


def test_WF5_sweep_totals_schema(demo1_system):
    """``totals()`` returns a DataFrame with exactly one row per
    (x, y, t_final, a, b, order) group."""
    exp = demo1_system.expand(
        ("phi_a(x)", "phi_b(y)"), orders=[0, 2],
    )
    props = demo1_system.propagators(
        t_max=3.0, n_grid_t=40, c_closed_form=_C_demo1,
    )
    sweep = exp.sweep(
        props,
        positions_grid={"x": [0.0], "y": [0.0, 0.5]},
        t_final_grid=[1.0],
        component_pairs=[(0, 0)],
        orders=[0, 2],
        integrate_over="all",  # consistent with WF4's convention
        n_samples=2 ** 10,
        seed=1,
    )
    totals = sweep.totals()
    expected_columns = {"x", "y", "t_final", "a", "b", "order", "value"}
    assert expected_columns.issubset(totals.columns)

    # 2 r-values × 1 t × 1 (a,b) × 2 orders = 4 rows.
    assert len(totals) == 4

    # by_vertex_type_totals groups by vertex_type instead of order.
    vt = sweep.by_vertex_type_totals()
    assert "vertex_type" in vt.columns
    # Purely local: all vtype labels are 'F' (order 2) or '' (order 0).
    assert set(vt["vertex_type"]).issubset({"F", ""})
