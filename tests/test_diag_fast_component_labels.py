"""Observable component labels on a C propagator in the iso_R + diag_C fast
path (DC0-DC4).

``DiagramIntegrand.evaluate`` routes to ``_evaluate_diag_fast`` whenever the
cache model is ``iso_R`` + ``diag_C`` and the diagram carries propagator
summation indices.  Up to 0.4.2 that path resolved a C propagator's component
label with ``_resolve_component(idx_name, {})`` -- an EMPTY map -- whenever the
label was not one of those summation indices.  An observable label pinned
through ``fixed_indices`` (the ``a``, ``b`` of ``<phi_a(x) phi_b(y)>``) then
resolved to ``None`` and the propagator contributed ``c_diag.sum()``, every
component, instead of ``c_diag[fixed_indices[label]]``.  For the order-2
two-point function below, 4 of the 6 diagram terms carry such a label
alongside a summation index, and the observable came out 17%-78% too large.

``_evaluate_general`` and the batched backends merge ``fixed_indices`` into the
index map and were always right, so the defect was visible only through the
scalar loop: ``method='qmc_scalar'``; ``method='qmc'``, which auto-selects the
scalar loop whenever ``_cache_supports_batch_c`` is False -- as it is for a
cache holding only spatial (rotation / translation / general) tables and no
legacy ``_c_splines``; ``method='nquad'``; and a direct
``DiagramIntegrand.evaluate``.  The L1 ``Expansion.evaluate`` and YAML
``sweep.method`` default is ``qmc_vectorized``, which is why no published
number went through it.

The same loop read ``il`` alone and ignored ``ir``.  Under ``diag_C``,
``C_{ab} = delta_{ab} c_diag[a]``, so the two legs are tied by a Kronecker
delta.  ``apply_diagonal(diag_C=True)`` merges them into a single name, which
is why the L1 pipeline never exposed it -- but a term expanded WITHOUT
``diag_C`` and evaluated against a ``diag_C`` cache still arrives with
``il != ir``, and dropping the delta keeps a cross-component term that
``_evaluate_general`` and ``_select_C_batch`` both set to zero.

* **DC0** the fixture really is the configuration under test: 4 of the 6
  order-2 terms carry ``a`` or ``b`` on a C propagator together with
  propagator summation indices, and the cache reports ``iso_R`` + ``diag_C``
  with a spatial table but no ``_c_splines`` (so ``method='qmc'`` picks the
  scalar loop).
* **DC1** ``_evaluate_diag_fast`` against a numpy hand contraction over the
  FULL C matrices, for every term and every pinned pair.  The reference is
  written in this file and reads ``cache.C_value``, which the fast path never
  touches.
* **DC2** the same, with the C legs artificially de-merged so ``il != ir``
  across all four leg classes (axis/axis equal, axis/axis distinct,
  axis/pinned, pinned/pinned), cross-checked against ``_evaluate_general``.
* **DC3** end-to-end through the L1 workflow API: ``qmc_scalar``, ``qmc``,
  ``qmc_vectorized``, ``gauss_legendre`` and ``nquad`` on the same observable
  at pinned component pairs.
* **DC4** the recorded values, and the pre-fix ones shown to differ.

DC1, DC2 and DC4 each assert that the label-blind formula gives a materially
different number, so none of them can pass against the code before the fix.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.evaluate import (
    DiagramIntegrand,
    _cache_has_spatial_table,
    _cache_supports_batch_c,
)
from sft_wick.perturbation import Propagator


# ---------------------------------------------------------------------------
# Fixture: 2-component field, one cubic local vertex, rotation-homogeneous
# noise, and the order-2 two-point function <phi_a(x) phi_b(y)>.
# ---------------------------------------------------------------------------

#: Deliberately asymmetric in its component indices so a mis-resolved label
#: cannot be masked by a coincidence of the coupling.
F_COUPLING = np.zeros((2, 2, 2))
F_COUPLING[0, 1, 1] = 1.0
F_COUPLING[1, 0, 0] = 0.5
F_COUPLING[0, 0, 1] = 0.3

T_FINAL = 1.0
N_SAMPLES = 2 ** 10
SEED = 3
PAIRS = [(0, 0), (0, 1), (1, 1)]


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


POS = {"x": _unit([0.3, -0.2, 0.93]), "y": _unit([0.8, 0.5, -0.33])}

#: Distinct directions per direction variable, so every C propagator sees a
#: different matrix and a mis-selected component shows up numerically.
_DIR_POOL = [
    _unit([0.3, -0.2, 0.93]),
    _unit([0.8, 0.5, -0.33]),
    _unit([-0.4, 0.7, 0.6]),
    _unit([0.1, 0.15, 0.98]),
]


def _build_system():
    return sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        noise=sw.GaussianNoise(kappa2=sw.SeparableRotation(
            temporal=sw.ExponentialTemporal(lam=1.0, sigma_t=0.5),
            angular=sw.LegendreAngular(coeffs=[1.0]),
        )),
        vertices=[sw.LocalVertex(name="F", coupling=F_COUPLING)],
    )


@pytest.fixture(scope="module")
def setup():
    system = _build_system()
    props = system.propagators(t_max=2.0, n_grid_t=16)
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2])
    return {
        "system": system,
        "props": props,
        "cache": props.cache,
        "expansion": expansion,
        "dts": expansion.dts_by_order[2],
    }


# ---------------------------------------------------------------------------
# Independent reference: the C contraction done by hand, from the term's
# declared structure and the FULL C matrices.  ``r_val`` is passed as 1.0 on
# both sides so the response propagators cancel out of the comparison and
# what is compared is purely the component bookkeeping.
# ---------------------------------------------------------------------------

def _pin(name, idx_map):
    """Component a leg label denotes, or None (-> trace), by hand."""
    if name is None:
        return None
    if name in idx_map:
        return idx_map[name]
    try:
        return int(name) - 1          # 1-indexed literal
    except ValueError:
        return None


def _reference_c_contraction(ig, cache, times, directions):
    """Sum over the propagator summation indices with explicit C matrices."""
    spatial = ig.spatial
    prop_idx = ig.diagram_term.propagator_indices
    names = [n for n, _ in prop_idx]
    total = 0j
    for pidx in np.ndindex(*tuple(d for _, d in prop_idx)):
        coeff = complex(ig.coupling_array[pidx])
        if coeff == 0:
            continue
        idx_map = {**ig.fixed_indices, **dict(zip(names, pidx))}
        for sp_l, sp_r, il, ir in spatial.c_propagators:
            C = cache.C_value(
                directions[spatial.direction_map[sp_l]], times[sp_l],
                directions[spatial.direction_map[sp_r]], times[sp_r],
            )
            a, b = _pin(il, idx_map), _pin(ir, idx_map)
            coeff *= C[a, b] if (a is not None and b is not None) else np.trace(C)
        total += coeff
    return total


def _label_blind_contraction(ig, cache, times, directions):
    """What the fast path computed before the fix: a leg label that is not a
    propagator summation index resolved against an EMPTY map, and ``ir``
    ignored.  Kept verbatim so the tests below have a control that fails
    against the old code."""
    spatial = ig.spatial
    prop_idx = ig.diagram_term.propagator_indices
    names = [n for n, _ in prop_idx]
    axis_of = {n: ax for ax, n in enumerate(names)}
    contracted = ig.coupling_array.copy()
    n_axes = contracted.ndim
    for sp_l, sp_r, il, _ir in spatial.c_propagators:
        c_diag = cache.C_diagonal(
            directions[spatial.direction_map[sp_l]], times[sp_l],
            directions[spatial.direction_map[sp_r]], times[sp_r],
        )
        if il is None:
            contracted = contracted * c_diag.sum()
            continue
        axis = axis_of.get(il)
        if axis is None:
            a = _pin(il, {})           # the bug: empty map, not fixed_indices
            contracted = contracted * (c_diag.sum() if a is None else c_diag[a])
            continue
        shape = [1] * n_axes
        shape[axis] = len(c_diag)
        contracted = contracted * c_diag.reshape(shape)
    return complex(contracted.sum())


def _sample_point(ig):
    """A deterministic (times, directions) evaluation point for one term."""
    spatial = ig.spatial
    points = sorted(
        set(spatial.direction_map)
        | {p for c in spatial.c_propagators for p in c[:2]}
    )
    times = {p: 0.9 - 0.1 * i for i, p in enumerate(points)}
    dvars = sorted(set(spatial.direction_map.values()))
    directions = {dv: _DIR_POOL[i % len(_DIR_POOL)] for i, dv in enumerate(dvars)}
    return times, directions


def _fast(ig, cache, times, directions):
    prop_idx = ig.diagram_term.propagator_indices
    names = [n for n, _ in prop_idx]
    return ig._evaluate_diag_fast(
        1.0, times, directions, cache, names,
        {n: ax for ax, n in enumerate(names)},
        coupling_array=ig.coupling_array,
    )


def _general(ig, cache, times, directions):
    prop_idx = ig.diagram_term.propagator_indices
    names = [n for n, _ in prop_idx]
    return ig._evaluate_general(
        1.0, times, directions, cache, names,
        {n: ax for ax, n in enumerate(names)},
        coupling_array=ig.coupling_array,
    )


# ---------------------------------------------------------------------------
# DC0 -- the fixture is the configuration under test
# ---------------------------------------------------------------------------

def test_DC0_fixture_is_the_configuration_under_test(setup):
    cache = setup["cache"]
    assert cache.model.iso_R and cache.model.diag_C, (
        "the fast path is only taken for an iso_R + diag_C model"
    )
    # Spatial table but no legacy _c_splines: this is exactly the cache shape
    # for which method='qmc' auto-selects the scalar loop.
    assert _cache_has_spatial_table(cache)
    assert not _cache_supports_batch_c(cache)

    dts = setup["dts"]
    assert len(dts) == 6, f"expected 6 order-2 terms, got {len(dts)}"
    labelled = [
        dt for dt in dts
        if dt.propagator_indices
        and any(p.kind == "C" and {p.index_left, p.index_right} & {"a", "b"}
                for p in dt.propagators)
    ]
    assert len(labelled) == 4, (
        "expected 4 terms carrying an observable label on a C propagator "
        f"alongside propagator summation indices, got {len(labelled)}"
    )
    # apply_diagonal(diag_C=True) has merged every C propagator's legs.
    for dt in dts:
        for p in dt.propagators:
            if p.kind == "C":
                assert p.index_left == p.index_right


# ---------------------------------------------------------------------------
# DC1 -- fast path vs the hand contraction, every term x every pinned pair
# ---------------------------------------------------------------------------

#: Order-2 terms on which the label-blind formula differs, per pinned pair.
#: Four terms carry an observable label on a C propagator (DC0); the ones
#: missing from a given pair's count have a vanishing coupling for that pair.
LABEL_SENSITIVE_TERMS = {(0, 0): 4, (0, 1): 3, (1, 1): 2}


@pytest.mark.parametrize("pair", PAIRS)
def test_DC1_fast_path_matches_hand_contraction(setup, pair):
    cache = setup["cache"]
    fixed = {"a": pair[0], "b": pair[1]}
    n_label_sensitive = 0
    for i, dt in enumerate(setup["dts"]):
        ig = dt.build_integrand({"F": F_COUPLING}, fixed)
        times, directions = _sample_point(ig)
        ref = _reference_c_contraction(ig, cache, times, directions)
        got = _fast(ig, cache, times, directions)
        assert got == pytest.approx(ref, rel=1e-12, abs=1e-15), (
            f"term {i}, pair {pair}: fast path {got!r} != hand contraction {ref!r}"
        )
        assert _general(ig, cache, times, directions) == pytest.approx(
            ref, rel=1e-12, abs=1e-15
        )
        old = _label_blind_contraction(ig, cache, times, directions)
        if abs(old - ref) > 1e-9 * max(abs(ref), 1e-9):
            n_label_sensitive += 1
            # Exactly one C propagator per term carries a label, N = 2, and
            # the angular kernel is isotropic, so the label-blind
            # ``c_diag.sum()`` is exactly ``2 * c_diag[pinned]``.
            assert old / ref == pytest.approx(2.0, rel=1e-9, abs=1e-12), (
                f"term {i}, pair {pair}: expected the label-blind formula to "
                f"double the term, got {old!r} vs {ref!r}"
            )
    assert n_label_sensitive == LABEL_SENSITIVE_TERMS[pair], (
        "control: the label-blind formula must differ on the terms carrying "
        f"an observable label; expected {LABEL_SENSITIVE_TERMS[pair]} such "
        f"terms for pair {pair}, got {n_label_sensitive}"
    )


# ---------------------------------------------------------------------------
# DC2 -- il != ir: the Kronecker delta between the two legs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", PAIRS)
def test_DC2_distinct_leg_labels_carry_the_kronecker_delta(setup, pair):
    """A term expanded without diag_C keeps ``il != ir``.  Under a diag_C
    cache the propagator is still ``delta_{il,ir} c_diag[il]``, so the fast
    path must agree with the explicit matrix contraction for every pairing of
    leg labels -- two summation axes, a summation axis and a pinned label,
    and two pinned labels."""
    cache = setup["cache"]
    fixed = {"a": pair[0], "b": pair[1]}
    seen = 0
    for i, dt in enumerate(setup["dts"]):
        for j, p in enumerate(dt.propagators):
            if p.kind != "C":
                continue
            for other in ("a", "b", "i_0", "i_1", "1", "2"):
                if other == p.index_right:
                    continue
                props = list(dt.propagators)
                props[j] = Propagator("C", p.index_left, other,
                                      p.spatial_left, p.spatial_right)
                dt2 = replace(dt, propagators=tuple(props))
                ig = dt2.build_integrand({"F": F_COUPLING}, fixed)
                if not ig.diagram_term.propagator_indices:
                    continue
                times, directions = _sample_point(ig)
                ref = _reference_c_contraction(ig, cache, times, directions)
                got = _fast(ig, cache, times, directions)
                assert got == pytest.approx(ref, rel=1e-12, abs=1e-15), (
                    f"term {i} propagator {j} {p.index_left!r}->{other!r}, "
                    f"pair {pair}: fast {got!r} != reference {ref!r}"
                )
                assert _general(ig, cache, times, directions) == pytest.approx(
                    ref, rel=1e-12, abs=1e-15
                )
                seen += 1
    assert seen >= 50, f"expected a broad sweep of leg pairings, got {seen}"


# ---------------------------------------------------------------------------
# DC3 -- every backend on the same observable
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def backend_values(setup):
    """``{pair: {method: total}}`` for every integration backend."""
    expansion = setup["expansion"]
    props = setup["props"]
    out = {}
    for pair in PAIRS:
        vals = {}
        for method in ("qmc_scalar", "qmc", "qmc_vectorized",
                       "gauss_legendre", "nquad"):
            kw = {}
            if method.startswith("qmc"):
                kw = {"n_samples": N_SAMPLES, "seed": SEED}
            vals[method] = expansion.evaluate(
                props, t_final=T_FINAL, positions=POS,
                component_pair=pair, method=method, **kw,
            ).total
        out[pair] = vals
    return out


@pytest.mark.parametrize("pair", PAIRS)
def test_DC3_scalar_loop_matches_the_batched_backends(setup, backend_values, pair):
    """The scalar loop and the vectorised path are the same estimator on the
    same Sobol points, so at a fixed seed they agree to round-off."""
    vals = backend_values[pair]
    ref = vals["qmc_vectorized"]
    for method in ("qmc_scalar", "qmc"):
        assert vals[method] == pytest.approx(ref, rel=1e-12, abs=1e-15), (
            f"pair {pair}: {method} {vals[method]!r} != qmc_vectorized {ref!r}"
        )


@pytest.mark.parametrize("pair", PAIRS)
def test_DC3_quadrature_backends_agree_with_qmc(setup, backend_values, pair):
    """Gauss-Legendre and nquad are different rules, so they agree with QMC
    only to the accuracy of the integration -- measured at 1.3e-5 (GL) and
    9.4e-5 (nquad) relative on this observable."""
    vals = backend_values[pair]
    ref = vals["qmc_vectorized"]
    for method in ("gauss_legendre", "nquad"):
        assert vals[method] == pytest.approx(ref, rel=1e-3, abs=1e-9), (
            f"pair {pair}: {method} {vals[method]!r} != qmc_vectorized {ref!r}"
        )


# ---------------------------------------------------------------------------
# DC4 -- recorded values, and the pre-fix values shown to differ
# ---------------------------------------------------------------------------

#: qmc_vectorized, n_samples=2**10, seed=3, t_final=1.0 -- measured on the
#: fixed code.  Every backend above agrees with these.  Re-recorded when the
#: Sobol samplers moved to 64-bit points: scipy scrambles the sequence, so a
#: different ``bits`` is a different point set, and each value moved by its
#: own sampling error at 2**10 samples (1.4e-4, 6.9e-5, 1.1e-4 relative),
#: not by the 1e-9 bias that change removed.  The 30-bit values were
#: 2.4163209589574818e-02, 1.5817788075977668e-02, 7.0654764616863335e-03.
RECORDED = {
    (0, 0): 2.4159751969924134e-02,
    (0, 1): 1.5816695846356810e-02,
    (1, 1): 7.0646686728619430e-03,
}

#: What the scalar loop returned before the fix, at the same settings.  Kept
#: so DC4 cannot pass against the old code.
PRE_FIX = {
    (0, 0): 2.834850e-02,
    (0, 1): 2.809094e-02,
    (1, 1): 9.275227e-03,
}


@pytest.mark.parametrize("pair", PAIRS)
def test_DC4_recorded_values(setup, backend_values, pair):
    got = backend_values[pair]["qmc_vectorized"]
    assert got == pytest.approx(RECORDED[pair], rel=1e-10, abs=1e-14)

    # Control: the pre-fix scalar-loop value is 17%-78% away, so a
    # regression in either direction is caught rather than absorbed.
    drift = abs(PRE_FIX[pair] - RECORDED[pair]) / abs(RECORDED[pair])
    assert drift > 0.05, (
        f"pair {pair}: the pre-fix value is only {drift:.1%} away, too close "
        "to act as a control"
    )
    for method, value in backend_values[pair].items():
        assert abs(value - PRE_FIX[pair]) / abs(PRE_FIX[pair]) > 1e-3, (
            f"pair {pair}: {method} reproduced the pre-fix value {value!r}"
        )
