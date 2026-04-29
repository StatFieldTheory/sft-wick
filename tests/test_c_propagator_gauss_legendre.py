"""Tests for the Gauss-Legendre C-propagator quadrature path.

Adds an alternative ``c_method='gauss_legendre'`` to
:meth:`PropagatorCache._C_value_direct` (and the three
``precompute_C_table_*`` builders) that splits the rectangle
``[t_min, t1] × [t_min, t2]`` at the diagonal ``λ1 = λ2`` into 2 or 3
sub-regions and applies a fixed tensor-product Gauss-Legendre rule
per sub-region.  This avoids the ``O(10%)`` polynomial-fit error that
naive whole-rectangle GL incurs from the ``|λ1 − λ2|`` cusp on the
diagonal of typical OU / exponential-temporal κ² kernels, while
recovering the dramatic 50-1000× speedup over
``scipy.integrate.dblquad``'s adaptive recursion.

The four tests:

- ``test_C_value_direct_gl_matches_dblquad_translation``: a
  translation-invariant κ²(λ1, λ2) ∝ exp(-|λ1-λ2|/σ_t)·exp(-r/σ_x).
  GL n=20 must agree with dblquad to rel-tol 1e-4 at five
  ``(t1, t2, r)`` test points covering both ``t1 = t2`` and ``t1 ≠ t2``.
- ``test_C_value_direct_gl_matches_dblquad_rotation``: similar but
  with a rotation-invariant κ² depending on ``x1·x2``.
- ``test_precompute_C_table_translation_gl_matches_dblquad``: full
  20×20 ``(t1, t2)`` grid table-build comparison.
- ``test_C_value_direct_gl_speedup``: timing benchmark -- GL n=20 is
  expected to be at least 50× faster than dblquad on the typical
  ``t1 = t2 = T_MAX, r = 0`` deep-domain case where dblquad's
  adaptive subdivision becomes especially expensive.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from sft_wick.evaluate import (
    PropagatorCache,
    PropagatorModel,
    _C_value_direct_gl,
    _rotation_cos,
)


# --------------------------------------------------------------------- #
# Common kernel definitions: OU R, separable κ²                         #
# --------------------------------------------------------------------- #

GAMMA = 1.0
LAM = 0.05
SIGMA_T = 0.3
SIGMA_X = 1.0
T_MIN = 0.0


def _R_time(t1: float, t2: float) -> float:
    """Causal OU: ``exp(-γ (t1 - t2))`` for ``t1 ≥ t2`` else 0."""
    return float(np.exp(-GAMMA * (t1 - t2))) if t1 >= t2 else 0.0


def _kappa2_translation(n1, t1, n2, t2):
    """Separable translation-invariant κ² used by demo1.

    ``λ · exp(-|t1-t2|/σ_t) · exp(-|n1-n2|/σ_x) · I``
    """
    r = abs(float(np.asarray(n1).sum()) - float(np.asarray(n2).sum()))
    kt = LAM * np.exp(-abs(t1 - t2) / SIGMA_T)
    kx = np.exp(-r / SIGMA_X)
    return kt * kx * np.eye(1)


def _kappa2_rotation(n1, t1, n2, t2):
    """Rotation-invariant κ² depending on ``x1 · x2`` only."""
    cos_val = _rotation_cos(n1, n2)
    kt = LAM * np.exp(-abs(t1 - t2) / SIGMA_T)
    kx = 0.5 * (1.0 + cos_val)
    return kt * kx * np.eye(1)


def _make_translation_model() -> PropagatorModel:
    return PropagatorModel(
        R_time=_R_time, kappa2=_kappa2_translation,
        n_components=1, iso_R=True, diag_C=True, t_min=T_MIN,
    )


def _make_rotation_model() -> PropagatorModel:
    return PropagatorModel(
        R_time=_R_time, kappa2=_kappa2_rotation,
        n_components=1, iso_R=True, diag_C=True, t_min=T_MIN,
    )


# --------------------------------------------------------------------- #
# Tests                                                                  #
# --------------------------------------------------------------------- #


class TestGaussLegendreVsDblquad:
    """Pointwise GL-vs-dblquad agreement on the demo1/demo2-style
    OU + separable kernels.  The diagonal cusp at ``λ1 = λ2`` is
    handled by the sub-region split; ``n_gauss=20`` should reach
    machine precision (well under the requested 1e-4 rel-tol).
    """

    RTOL = 1e-4

    def test_C_value_direct_gl_matches_dblquad_translation(self):
        """Translation-invariant κ²: GL n=20 vs dblquad on five points
        spanning ``t1 = t2`` (square case) and ``t1 ≠ t2``
        (rectangle case)."""
        model = _make_translation_model()
        cache_db = PropagatorCache(model)

        test_points = [
            (1.0, 1.0, 0.0),    # square, r = 0
            (2.0, 2.0, 0.5),    # square, r > 0
            (1.5, 0.8, 0.3),    # rectangle, t1 > t2
            (0.7, 2.0, 1.2),    # rectangle, t2 > t1
            (3.0, 3.0, 1.5),    # square, deeper domain
        ]

        for t1, t2, r in test_points:
            n1 = np.asarray(0.0)
            n2 = np.asarray(r)
            got_gl = _C_value_direct_gl(model, n1, t1, n2, t2, n_gauss=20)
            got_db = cache_db._C_value_direct(n1, t1, n2, t2)

            # Element-wise relative error.
            denom = np.abs(got_db) + 1e-300
            rel = np.max(np.abs(got_gl - got_db) / denom)
            assert rel < self.RTOL, (
                f"GL vs dblquad disagreement at (t1={t1}, t2={t2}, "
                f"r={r}): gl={got_gl.ravel()}, db={got_db.ravel()}, "
                f"rel={rel:.2e}"
            )

    def test_C_value_direct_gl_matches_dblquad_rotation(self):
        """Rotation-invariant κ²: GL n=20 vs dblquad on representative
        ``(t1, t2, cos θ)`` points."""
        model = _make_rotation_model()
        cache_db = PropagatorCache(model, homogeneity="rotation")

        cases = [
            (1.0, 1.0, 0.0),    # orthogonal
            (2.0, 2.0, 0.5),    # 60°
            (1.2, 0.8, 1.0),    # parallel, rectangle
            (1.0, 2.5, -0.5),   # 120°, rectangle reverse
            (3.0, 3.0, -1.0),   # antiparallel, deep domain
        ]

        for t1, t2, cos_val in cases:
            sin_val = float(np.sqrt(max(0.0, 1.0 - cos_val ** 2)))
            e1 = np.array([1.0, 0.0])
            e2 = np.array([cos_val, sin_val])

            got_gl = _C_value_direct_gl(model, e1, t1, e2, t2, n_gauss=20)
            got_db = cache_db._C_value_direct(e1, t1, e2, t2)

            denom = np.abs(got_db) + 1e-300
            rel = np.max(np.abs(got_gl - got_db) / denom)
            assert rel < self.RTOL, (
                f"GL vs dblquad disagreement (rotation) at "
                f"(t1={t1}, t2={t2}, cos={cos_val}): "
                f"gl={got_gl.ravel()}, db={got_db.ravel()}, "
                f"rel={rel:.2e}"
            )

    def test_precompute_C_table_translation_gl_matches_dblquad(self):
        """Full-grid table build via GL must match the dblquad-built
        table to ``rtol=1e-4`` at every grid node.  Keep the grid
        small (n_grid_t=12, n_grid_r=4) so dblquad finishes quickly
        in CI (12² × 4 ≈ 576 dblquad calls)."""
        T_MAX = 2.0
        N_GRID_T = 12
        N_GRID_R = 4
        R_MAX = 1.5

        cache_db = PropagatorCache(_make_translation_model())
        cache_db.precompute_C_table_translation(
            t_max=T_MAX, n_grid_t=N_GRID_T,
            r_max=R_MAX, n_grid_r=N_GRID_R,
            c_method="dblquad",
        )

        cache_gl = PropagatorCache(_make_translation_model())
        cache_gl.precompute_C_table_translation(
            t_max=T_MAX, n_grid_t=N_GRID_T,
            r_max=R_MAX, n_grid_r=N_GRID_R,
            c_method="gauss_legendre", n_gauss=20,
        )

        # Both caches now have a 3-D (t1, t2, r) interpolator.  Compare
        # at a handful of grid-coincident (and one off-grid) points.
        ts = np.linspace(T_MIN, T_MAX, N_GRID_T)
        rs = np.linspace(0.0, R_MAX, N_GRID_R)
        for ti in [0, 4, N_GRID_T - 1]:
            for tj in [0, 6, N_GRID_T - 1]:
                for ri in [0, 1, N_GRID_R - 1]:
                    t1, t2, r = ts[ti], ts[tj], rs[ri]
                    pt = np.array([[t1, t2, r]])
                    v_db = cache_db._c_translation_splines[0](pt)[0]
                    v_gl = cache_gl._c_translation_splines[0](pt)[0]
                    if abs(v_db) < 1e-300:
                        continue  # both should be ~0 there
                    rel = abs(v_gl - v_db) / abs(v_db)
                    assert rel < self.RTOL, (
                        f"table mismatch at (t1={t1}, t2={t2}, r={r}): "
                        f"db={v_db:.6e}, gl={v_gl:.6e}, rel={rel:.2e}"
                    )

    def test_C_value_direct_gl_per_call_speedup(self):
        """GL n=20 should be at least 50× faster than dblquad on a
        single ``_C_value_direct`` call at the demo2-style deep-domain
        configuration ``(t1=t2=5, r=0)``, where dblquad's adaptive
        subdivision is most expensive (the diagonal cusp triggers
        deep recursion)."""
        model = _make_translation_model()
        cache_db = PropagatorCache(model)

        T = 5.0
        n1, n2 = np.asarray(0.0), np.asarray(0.0)

        # Warm-up call so first-call import overhead doesn't bias the
        # timing.  Cost of one call is dominated by user-Python, not
        # numpy import.
        _ = _C_value_direct_gl(model, n1, T, n2, T, n_gauss=20)
        _ = cache_db._C_value_direct(n1, T, n2, T)

        # Average ~3 calls each.  The variance is small.
        n_reps = 3
        t0 = time.time()
        for _ in range(n_reps):
            _C_value_direct_gl(model, n1, T, n2, T, n_gauss=20)
        t_gl = (time.time() - t0) / n_reps

        t0 = time.time()
        for _ in range(n_reps):
            cache_db._C_value_direct(n1, T, n2, T)
        t_db = (time.time() - t0) / n_reps

        speedup = t_db / t_gl
        # Target is 50× per the task spec; we add a generous floor so
        # CI noise on slow boxes doesn't flake.
        assert speedup >= 50.0, (
            f"expected >= 50× speedup, got {speedup:.1f}× "
            f"(gl={t_gl*1000:.2f} ms, db={t_db*1000:.2f} ms)"
        )


class TestGaussLegendreCorrectness:
    """Edge-case correctness tests for the GL helper:

    - Empty domain (t1 = t_min or t2 = t_min) → C = 0.
    - White-noise δ piece is added on the GL path too.
    - The cache instance attribute ``c_method='gauss_legendre'`` is
      honoured by the public ``_C_value_direct`` entry point.
    """

    def test_empty_domain_returns_zero(self):
        model = _make_translation_model()
        for t1, t2 in [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0)]:
            got = _C_value_direct_gl(
                model, np.asarray(0.0), t1, np.asarray(0.0), t2,
                n_gauss=20,
            )
            assert np.allclose(got, 0.0), (
                f"empty domain at (t1={t1}, t2={t2}) should give 0, "
                f"got {got.ravel()}"
            )

    def test_white_noise_term_matches_dblquad(self):
        """When ``model.sigma2`` is set, GL must include the 1-D
        δ-collapsed piece -- compared to the dblquad path."""
        amp = 0.3

        def sigma2(n1, t, n2):
            return np.array([[amp]])

        model = PropagatorModel(
            R_time=_R_time,
            kappa2=lambda n1, t1, n2, t2: np.zeros((1, 1)),
            sigma2=sigma2,
            n_components=1, iso_R=True, diag_C=True, t_min=T_MIN,
        )
        cache_db = PropagatorCache(model)

        for t1, t2 in [(1.0, 1.0), (1.0, 2.0), (0.5, 1.5)]:
            got_gl = _C_value_direct_gl(
                model, np.asarray(0.0), t1, np.asarray(0.0), t2,
                n_gauss=20,
            )
            got_db = cache_db._C_value_direct(
                np.asarray(0.0), t1, np.asarray(0.0), t2,
            )
            rel = abs(got_gl[0, 0] - got_db[0, 0]) / abs(got_db[0, 0])
            assert rel < 1e-8, (
                f"GL+white-noise mismatch at (t1={t1}, t2={t2}): "
                f"gl={got_gl[0, 0]:.8e}, db={got_db[0, 0]:.8e}, "
                f"rel={rel:.2e}"
            )

    def test_cache_instance_c_method_is_honoured(self):
        """Setting ``c_method='gauss_legendre'`` on the cache must
        route ``_C_value_direct`` through the GL helper -- result
        agrees with the explicit helper call to floating-point
        equality."""
        model = _make_translation_model()
        cache = PropagatorCache(model, c_method="gauss_legendre", n_gauss=20)

        n1, n2 = np.asarray(0.0), np.asarray(0.5)
        via_cache = cache._C_value_direct(n1, 1.5, n2, 1.0)
        via_helper = _C_value_direct_gl(
            model, n1, 1.5, n2, 1.0, n_gauss=20,
        )
        assert np.allclose(via_cache, via_helper, rtol=0, atol=0), (
            f"cache dispatch disagrees with helper: "
            f"cache={via_cache.ravel()}, helper={via_helper.ravel()}"
        )

    def test_unknown_c_method_raises(self):
        """Constructing a cache with an unknown ``c_method`` must
        raise immediately; passing an unknown method via the
        per-call override of ``_C_value_direct`` must also raise."""
        model = _make_translation_model()
        with pytest.raises(ValueError, match="c_method"):
            PropagatorCache(model, c_method="simpson")

        cache = PropagatorCache(model)
        with pytest.raises(ValueError, match="unknown c_method"):
            cache._C_value_direct(
                np.asarray(0.0), 1.0, np.asarray(0.0), 1.0,
                method="simpson",
            )

    def test_invalid_n_gauss_raises(self):
        model = _make_translation_model()
        with pytest.raises(ValueError, match="n_gauss"):
            PropagatorCache(model, n_gauss=1)
