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

import warnings

import numpy as np
import pandas as pd
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


@pytest.fixture(scope="module")
def demo1_expansion(demo1_system):
    """The orders-[0, 2, 4] expansion of ``demo1_system``.

    WF1, WF2 and WF4 each asked for *this same* expansion — same system,
    same observable, same orders — and ``System.expand`` has no in-memory
    memo (its ``cache_path`` cache is opt-in and off here), so the
    identical 1.7 s ``compute_moment`` ran three times.  ``Expansion`` is
    a frozen dataclass over frozen ``DiagramTerm``s and every consumer
    below is read-only, so one build serves all three.
    """
    return demo1_system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2, 4])


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


def test_WF1_expand_produces_expected_diagram_counts(demo1_expansion):
    """System.expand at orders 0, 2, 4 yields the canonical counts
    (1 + 6 + 64 = 71 diagrams) produced by the raw compute_moment
    path on the same physical spec."""
    exp = demo1_expansion
    assert len(exp.diagrams(0)) == 1, (
        "order-0 should be a single self-loop"
    )
    assert len(exp.diagrams(2)) == 6
    assert len(exp.diagrams(4)) == 64


# =====================================================================
# WF2 — summary() reports sensible structure
# =====================================================================


def test_WF2_summary_purely_local_theory(demo1_expansion):
    summary = demo1_expansion.summary()

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


def test_WF4_end_to_end_matches_validate_phase5(demo1_expansion, demo1_system):
    """Sweep via the wrapper and assert a handful of well-tested
    reference values from ``examples/demo1/validate_phase5.py`` match to
    < 1e-6 rel."""
    exp = demo1_expansion
    props = demo1_system.propagators(
        t_max=15.0, n_grid_t=60,
        c_closed_form=_C_demo1,
    )

    # Reference values from examples/demo1/validate_phase5.py run.
    #
    # The four INTERACTING entries were re-pinned when the C-table diagonal
    # kink was fixed (see test_F21_* in test_msr_numerics_regressions.py).
    # C(t1,t2) has a derivative discontinuity of exactly -sigma2(t) on
    # t1 == t2, which a tensor-product spline cannot represent; on the
    # diagonal the table did not converge at all (22.3% relative error at
    # n_grid=41, still 21.4% at 321), while staying clean O(h^4) off it.
    # Harvesting the grid's own i == j entries into a 1-D spline restores
    # O(h^4) there.  Re-pinned against the MORE ACCURATE value, not by
    # loosening the 1e-5 tolerance:
    #
    #   (0,0,0.5,2): 3.212865e-02 -> 3.222453e-02   (rel 2.98e-03)
    #   (0,0,0.5,4): 2.231587e-03 -> 2.236403e-03   (rel 2.16e-03)
    #   (1,1,1.0,2): 9.620743e-03 -> 9.627814e-03   (rel 7.35e-04)
    #   (1,1,1.0,4): 7.882002e-04 -> 7.887824e-04   (rel 7.39e-04)
    #
    # Both order-0 entries are UNCHANGED, which is the consistency check:
    # order 0 has no tadpole, so it never evaluates C on the diagonal.
    reference = {
        (0, 0, 0.0, 15.0, 0): 3.996863e-01,
        (0, 0, 0.5, 15.0, 0): 2.424220e-01,
        (0, 0, 0.5, 15.0, 2): 3.222453e-02,
        (0, 0, 0.5, 15.0, 4): 2.236403e-03,
        (1, 1, 1.0, 15.0, 2): 9.627814e-03,
        (1, 1, 1.0, 15.0, 4): 7.887824e-04,
    }

    # The grid is DERIVED from the reference table rather than being a
    # Cartesian box drawn around it.  The box (3 separations x 2 component
    # pairs x 3 orders) evaluated 18 cells; only these 6 are ever read, so
    # 12 QMC integrations at ~2.3 s each were computed and discarded.
    #
    # Splitting the box into one sweep per (component pair, order) block
    # cannot move a retained value: `Expansion.sweep` flattens the grid to
    # independent `evaluate` tasks, each re-seeded with the same `seed`,
    # and `integrate_diagrams` integrates each order's diagrams
    # separately.  Verified rather than assumed -- all six values are
    # bit-identical to the 3x2x3 box's, e.g. (0,0,0.5,4) is
    # 0.002236402700096169 either way.
    blocks: dict[tuple, list] = {}
    for (a, b, r, t_f, ord_) in reference:
        blocks.setdefault((a, b, t_f, ord_), []).append(r)

    frames = []
    for (a, b, t_f, ord_), separations in blocks.items():
        frames.append(exp.sweep(
            props,
            positions_grid={"x": [0.0], "y": sorted(set(separations))},
            t_final_grid=[t_f],
            component_pairs=[(a, b)],
            orders=[ord_],
            # validate_phase5.py reference values are time-integrated
            # (``integrate_moment`` integrates all externals over
            # ``[0, lambda_f]``).  Match that convention here.
            integrate_over="all",
            n_samples=2 ** 13,
            seed=42,
        ).totals())
    # Each block above carries a single component pair, so none of them
    # exercises `sweep`'s Cartesian product over `component_pairs` -- the
    # 3x2x3 box did, and no other test in the suite passes a multi-entry
    # `component_pairs`.  Order 0 restores that axis for ~0.5 s, and can
    # be *asserted* rather than merely run: gamma_0 == gamma_1 and
    # C_{ab} = delta_{ab} C_t(t1,t2) e^{-r}, so the two diagonal pairs are
    # the same number.  Both (1,1) rows are therefore pinned by the (0,0)
    # references already in the table -- a symmetry consequence, not a
    # fresh snapshot of today's output.  (The box computed these two cells
    # too, and asserted nothing about them.)
    pair_axis = exp.sweep(
        props,
        positions_grid={"x": [0.0], "y": [0.0, 0.5]},
        t_final_grid=[15.0],
        component_pairs=[(0, 0), (1, 1)],
        orders=[0],
        integrate_over="all",
        n_samples=2 ** 13,
        seed=42,
    ).totals()
    frames.append(pair_axis)
    for r in (0.0, 0.5):
        vals = {}
        for (a, b) in ((0, 0), (1, 1)):
            m = ((pair_axis["a"] == a) & (pair_axis["b"] == b)
                 & (abs(pair_axis["y"] - r) < 1e-12))
            assert len(pair_axis.loc[m]) == 1, (
                f"component_pairs axis: {len(pair_axis.loc[m])} rows for "
                f"(a={a}, b={b}, r={r})"
            )
            vals[(a, b)] = float(pair_axis.loc[m, "value"].iloc[0])
        assert vals[(1, 1)] == vals[(0, 0)], (
            f"order-0 component symmetry broken at r={r}: "
            f"(0,0)={vals[(0, 0)]!r} vs (1,1)={vals[(1, 1)]!r}"
        )
        # ... and the shared value is the one validate_phase5.py pins.
        ref = reference[(0, 0, r, 15.0, 0)]
        assert abs(vals[(1, 1)] - ref) / abs(ref) < 1e-5

    totals = pd.concat(frames, ignore_index=True)

    bad = []
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
        if rel >= 1e-5:
            bad.append(
                f"  (a={a},b={b},r={r},t_f={t_f},ord={ord_}): "
                f"got {got:.6e} ref {expected:.6e} rel {rel:.2e}"
            )
    assert not bad, "WF4 reference mismatches:\n" + "\n".join(bad)


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


# =====================================================================
# WF11 — two-time observables through the declarative sweep API
# =====================================================================
#
# Every integrator now takes per-point `external_times`, but until this the
# workflow layer could not reach it: `sweep()` took `t_final_grid` scalars and
# pinned all externals there.  With every external at one time Theta kills the
# R joining them, so *every* observable carrying an external response leg came
# back identically 0 -- which makes R(t,t') and C(t,t'), the DMFT order
# parameters, unreachable from the documented API.
#
# `external_times_grid` mirrors `positions_grid`: one list per external point,
# swept as a further Cartesian axis.  These reuse `demo1_system` at order 0,
# where the coupling is irrelevant, with the closed-form C so no dblquad runs.

_WF11_GAMMA = 1.0


def _wf11_props(demo1_system, t_max=6.0, n_grid_t=30):
    return demo1_system.propagators(
        t_max=t_max, n_grid_t=n_grid_t, c_closed_form=_C_demo1,
    )


def test_WF11_sweep_reaches_two_time_response(demo1_system):
    """R(T, t') must come out of `sweep`, not the identically-zero value."""
    exp = demo1_system.expand(("phi_a(x)", "psi_b(y)"), orders=[0])
    props = _wf11_props(demo1_system)

    T = 4.0
    tprimes = [1.0, 2.0, 3.0]
    sweep = exp.sweep(
        props,
        positions_grid={"x": [0.0], "y": [0.0]},
        t_final_grid=[T],
        external_times_grid={"x": [T], "y": tprimes},
        component_pairs=[(0, 0)],
        orders=[0],
        method="nquad",
    )
    tot = sweep.totals()
    assert "t_y" in tot.columns, tot.columns.tolist()

    for tp in tprimes:
        row = tot[(abs(tot["t_y"] - tp) < 1e-12) & (tot["order"] == 0)]
        assert len(row) == 1, f"t'={tp}: {len(row)} rows"
        got = float(row["value"].iloc[0])
        assert got == pytest.approx(np.exp(-_WF11_GAMMA * (T - tp)), rel=1e-9, abs=0.0)
        assert got != 0.0


def test_WF11_equal_times_still_give_zero_for_a_response(demo1_system):
    """Theta is untouched: pinning both externals together stays exactly 0.

    This is the behaviour the feature works around, not one it changes.
    """
    exp = demo1_system.expand(("phi_a(x)", "psi_b(y)"), orders=[0])
    props = _wf11_props(demo1_system)
    # ... and the user is told why, instead of getting a silent table of zeros.
    with pytest.warns(UserWarning, match="response leg"):
        tot = exp.sweep(
            props,
            positions_grid={"x": [0.0], "y": [0.0]},
            t_final_grid=[4.0],
            component_pairs=[(0, 0)],
            orders=[0],
            method="nquad",
        ).totals()
    assert float(tot["value"].iloc[0]) == pytest.approx(0.0, abs=1e-14)


def test_WF11_default_sweep_is_unchanged(demo1_system):
    """Omitting `external_times_grid` must reproduce the old rows exactly.

    Same guarantee as `external_times=None` at L0: the feature adds an axis,
    it does not move anything that already worked.
    """
    exp = demo1_system.expand(("phi_a(x)", "phi_b(y)"), orders=[0])
    props = _wf11_props(demo1_system)
    kw = dict(
        positions_grid={"x": [0.0], "y": [0.0, 0.5]},
        t_final_grid=[2.0],
        component_pairs=[(0, 0)],
        orders=[0],
        method="nquad",
    )
    base = exp.sweep(props, **kw).totals()
    same = exp.sweep(
        props, external_times_grid={"x": [2.0], "y": [2.0]}, **kw,
    ).totals()
    assert np.array_equal(
        np.sort(base["value"].to_numpy()), np.sort(same["value"].to_numpy())
    )
    assert "t_x" not in base.columns and "t_x" in same.columns


def test_WF11_two_time_correlator_matches_the_closed_form(demo1_system):
    """C(t, t') off the equal-time diagonal, against the OU closed form."""
    exp = demo1_system.expand(("phi_a(x)", "phi_b(y)"), orders=[0])
    props = _wf11_props(demo1_system, t_max=6.0, n_grid_t=60)

    tot = exp.sweep(
        props,
        positions_grid={"x": [0.0], "y": [0.0]},
        t_final_grid=[5.0],
        external_times_grid={"x": [4.0], "y": [1.5, 3.0]},
        component_pairs=[(0, 0)],
        orders=[0],
        method="nquad",
    ).totals()
    for tp in (1.5, 3.0):
        got = float(tot[abs(tot["t_y"] - tp) < 1e-12]["value"].iloc[0])
        want = _C_t_closed_form(4.0, tp)
        assert got == pytest.approx(want, rel=2e-3, abs=0.0), (
            f"C(4.0, {tp}) = {got:.8f} vs closed form {want:.8f}"
        )


@pytest.mark.parametrize("method", ["nquad", "qmc_vectorized", "gauss_legendre"])
def test_WF11_two_time_at_interacting_order_agrees_across_backends(
    demo1_system, method,
):
    """The other WF11 tests are all `orders=[0]`, where every diagram has zero
    time-integration variables and each backend short-circuits before its
    sampler runs.  So none of them exercises an integrator through the sweep
    path at all -- exactly the gap that would hide a backend-specific error at
    an interacting order.

    This runs order 2 with unequal external times and requires the three
    backends to agree.  Cross-backend rather than closed-form: the O(g^2)
    two-time response has no convenient closed form, but a defect in one
    backend's causal mapping shows up as a disagreement, and a factor-2 error
    of the kind a nested internal ordering could produce cannot hide.
    """
    exp = demo1_system.expand(("phi_a(x)", "psi_b(y)"), orders=[0, 2])
    props = _wf11_props(demo1_system, t_max=6.0, n_grid_t=40)
    kw = ({"n_samples": 2 ** 14, "seed": 7} if method.startswith("qmc")
          else {"n_gauss": 24} if method == "gauss_legendre" else {})

    def run(m, **extra):
        return exp.sweep(
            props,
            positions_grid={"x": [0.0], "y": [0.0]},
            t_final_grid=[4.0],
            external_times_grid={"x": [4.0], "y": [1.5]},
            component_pairs=[(0, 0)],
            orders=[0, 2],
            method=m,
            **extra,
        ).totals()

    tot = run(method, **kw)
    ref = run("nquad")

    got2 = float(tot[tot["order"] == 2]["value"].iloc[0])
    ref2 = float(ref[ref["order"] == 2]["value"].iloc[0])
    # The order-2 term must actually be there -- a zero would make the
    # comparison vacuous, which is the failure mode of the order-0 tests.
    assert abs(ref2) > 1e-6, f"order-2 reference is ~0 ({ref2:.3e})"
    assert got2 == pytest.approx(ref2, rel=5e-3, abs=0.0), (
        f"{method} order 2 = {got2:.8f} vs nquad {ref2:.8f} "
        f"(ratio {got2 / ref2:.4f})"
    )
    # And order 0 is still the exact retarded propagator.
    got0 = float(tot[tot["order"] == 0]["value"].iloc[0])
    assert got0 == pytest.approx(np.exp(-1.0 * (4.0 - 1.5)), rel=1e-6, abs=0.0)


def test_WF11_guards_reject_the_three_silent_failures(demo1_system):
    """Each of these previously produced a plausible answer, not an error.

    Found by adversarial review of this feature.
    """
    exp = demo1_system.expand(("phi_a(x)", "phi_b(y)"), orders=[0])
    props = _wf11_props(demo1_system, t_max=6.0, n_grid_t=30)
    base = dict(
        positions_grid={"x": [0.0], "y": [0.0]},
        t_final_grid=[4.0],
        component_pairs=[(0, 0)],
        orders=[0],
        method="nquad",
    )

    # 1. A time past the propagator horizon used to clamp to the table edge.
    with pytest.raises(ValueError, match="exceed the propagator table horizon"):
        exp.sweep(props, external_times_grid={"x": [99.0], "y": [1.0]}, **base)

    # 2. An empty list gave zero rows and then an opaque pandas KeyError.
    with pytest.raises(ValueError, match="empty list"):
        exp.sweep(props, external_times_grid={"x": [], "y": [1.0]}, **base)

    # 3. A `t_<point>` column that shadows an existing one used to overwrite
    #    it silently.  `t_final` is the reserved name most likely to collide.
    exp_f = demo1_system.expand(("phi_a(final)", "phi_b(y)"), orders=[0])
    with pytest.raises(ValueError, match="collide with existing sweep"):
        exp_f.sweep(
            props,
            positions_grid={"final": [0.0], "y": [0.0]},
            t_final_grid=[4.0],
            external_times_grid={"final": [4.0], "y": [1.0]},
            component_pairs=[(0, 0)], orders=[0], method="nquad",
        )


# =====================================================================
# WF12 — the spatially-structureless-cache guard, rebuilt after review
# =====================================================================

def _wf12_setup():
    from sft_wick.workflow.propagators import propagators_from_cache
    dens = sw.SpectralDensity.from_samples(np.array([0.8, 1.0, 1.4]))
    props = propagators_from_cache(
        sw.spectral_cache(dens, noise_D=1.0, n_components=2))
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
            spatial=sw.ExponentialSpatial(sigma_x=1.0))),
    )
    return system, props


def _spatial_warnings(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [w for w in caught if "no spatial structure" in str(w.message)]


_WF12_KW = dict(t_final_grid=[1.0], component_pairs=[(0, 0)], orders=[0],
                method="nquad")


def test_WF12_a_structureless_cache_warns_once_per_multi_separation_sweep():
    """A cache with no spatial table returns the same C at every separation,
    so the sweep produces a column of identical numbers next to a varying
    position column.

    The question is about the SWEEP -- "does this grid cover more than one
    position configuration" -- and is answerable only where the grid is.
    Asked per grid point instead ("are this point's two externals at different
    places?") it gets both halves wrong, which is what the first version did:
    silent on a sweep over one position key, and firing on every ordinary
    single-separation ``evaluate(x != y)``.
    """
    system, props = _wf12_setup()
    exp2 = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0])

    hits = _spatial_warnings(lambda: exp2.sweep(
        props, positions_grid={"x": [0.0], "y": [0.0, 0.5, 1.0]},
        n_jobs=1, **_WF12_KW))
    assert len(hits) == 1, f"expected exactly one warning, got {len(hits)}"

    # and it must survive n_jobs > 1: the per-grid-point path runs inside a
    # loky subprocess, whose warnings never reach the parent
    hits = _spatial_warnings(lambda: exp2.sweep(
        props, positions_grid={"x": [0.0], "y": [0.0, 0.5, 1.0]},
        n_jobs=2, **_WF12_KW))
    assert len(hits) == 1, "the warning was lost to the worker processes"

    # a sweep over a SINGLE position key still varies the separation
    exp1 = system.expand(("phi_a(x)", "phi_b(x)"), orders=[0])
    hits = _spatial_warnings(lambda: exp1.sweep(
        props, positions_grid={"x": [0.0, 0.5, 1.0]}, n_jobs=1, **_WF12_KW))
    assert len(hits) == 1, "a one-key sweep over three positions was missed"


def test_WF12_the_guard_does_not_fire_on_the_ordinary_cases():
    """A warning that fires on normal use is worse than none: users learn to
    filter it, and it is then absent when it matters."""
    system, props = _wf12_setup()
    exp2 = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0])

    # a plain single-separation evaluate, x != y -- no sweep, no column
    assert not _spatial_warnings(lambda: exp2.evaluate(
        props, positions={"x": 0.0, "y": 1.0}, t_final=1.0,
        component_pair=(0, 0), orders=[0], method="nquad"))

    # a sweep pinned to one position configuration
    assert not _spatial_warnings(lambda: exp2.sweep(
        props, positions_grid={"x": [0.0], "y": [1.0]}, n_jobs=1, **_WF12_KW))

    # a cache that DOES carry spatial structure
    spatial_props = system.propagators(t_max=2.0, n_grid_t=9)
    assert not _spatial_warnings(lambda: exp2.sweep(
        spatial_props, positions_grid={"x": [0.0], "y": [0.0, 0.5]},
        t_final_grid=[1.0], component_pairs=[(0, 0)], orders=[0]))
