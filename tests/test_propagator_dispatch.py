"""Dispatcher-level tests for the C-propagator build path.

* separability: the lazy translation cache builds ONE temporal table and
  rescales it by ``κ_x(r)`` -- numbers identical to the per-``r`` build;
* time symmetry: the upper-triangle build reproduces the full one;
* ``c_method='auto'``: the Gauss-Legendre node count is refined until the
  rule is converged for the requested ``t_max``, and dblquad is used
  where it never converges;
* progress reporting never changes a number and is silent by default.
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import pytest

import sft_wick as sw
from sft_wick import progress as prog
from sft_wick.evaluate import PropagatorCache, select_gl_node_count


def _demo1_like(sigma_t=0.5, gamma=(1.0, 1.0), temporal="exponential"):
    F = np.zeros((2, 2, 2))
    F[0, 1, 1] = 1.0
    if temporal == "exponential":
        kt = sw.ExponentialTemporal(lam=0.05, sigma_t=sigma_t)
    else:
        kt = sw.GaussianTemporal(lam=0.05, sigma_t=sigma_t)
    return sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=list(gamma)),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=kt, spatial=sw.ExponentialSpatial(sigma_x=1.0))),
    )


# --------------------------------------------------------------------- #
# Separability and symmetry in the lazy translation cache               #
# --------------------------------------------------------------------- #


class _NoShortcuts(PropagatorCache):
    """Reference: the per-r, full-square build the lazy cache used to do."""

    def _lazy_spatial_factor(self):
        return None

    def _c_time_symmetric(self):
        return False


@pytest.mark.parametrize("c_method", ["gauss_legendre", "dblquad"])
def test_lazy_translation_shared_temporal_table_matches_per_r_build(c_method):
    system = _demo1_like()
    model = system.build_propagator_model()
    kw = dict(homogeneity="translation", c_method=c_method, n_gauss=16)
    fast = PropagatorCache(model, **kw)
    slow = _NoShortcuts(model, **kw)
    # Cost note (2026-09): n_grid_t was 12.  ``fast`` and ``slow`` build
    # the SAME grid by two routes, so the resolution is not what is
    # asserted -- the agreement is.  Measured: worst relative gap over
    # the four r's is 3.436e-16 (GL) / 1.718e-16 (dblquad) at n_grid_t
    # = 12, and bit-identically the same at 8 and at 6, while the build
    # cost falls as n_grid_t**2 (dblquad: 6.4 s -> 2.6 s).  Tolerances
    # below are unchanged.
    n_grid_t, t_max = 8, 4.0
    fast.precompute_C_table_translation(t_max=t_max, n_grid_t=n_grid_t)
    slow.precompute_C_table_translation(t_max=t_max, n_grid_t=n_grid_t)

    rs = [0.0, 0.5, 1.0, 2.5]
    t1 = np.array([0.3, 1.0, 2.5, 4.0, 4.0, 3.1])
    t2 = np.array([0.3, 2.0, 2.5, 4.0, 0.2, 0.9])
    for r in rs:
        a = fast.C_at_batch(t1, t2, np.zeros(6), np.full(6, r))
        b = slow.C_at_batch(t1, t2, np.zeros(6), np.full(6, r))
        scale = np.max(np.abs(b))
        # GL is a fixed rule, so the two builds differ only by rounding;
        # dblquad re-adapts per cell (its epsabs floor moves with the
        # magnitude), so agreement there is bounded by ITS tolerance.
        tol = 1e-12 if c_method == "gauss_legendre" else 1e-6
        assert np.max(np.abs(a - b)) <= tol * scale, (r, np.max(np.abs(a - b)) / scale)

    # The shortcut ran the quadrature grid exactly once for the four r's;
    # the reference ran it four times.
    assert fast._lazy_translation.n_grid_builds == 1
    assert slow._lazy_translation.n_grid_builds == len(rs)


def test_separability_shortcut_is_off_with_white_noise_or_user_c():
    system = _demo1_like()
    model = system.build_propagator_model()
    with_white = sw.System(
        field=system.field, linear=system.linear, vertices=system.vertices,
        noise=sw.GaussianNoise(kappa2=system.noise.kappa2,
                               sigma2=sw.ConstantImpulse(0.2)),
    ).build_propagator_model()
    assert PropagatorCache(model)._lazy_spatial_factor() is not None
    assert PropagatorCache(with_white)._lazy_spatial_factor() is None
    user = PropagatorCache(model, c_value_fn=lambda n1, t1, n2, t2: np.eye(2))
    assert user._lazy_spatial_factor() is None
    assert user._c_time_symmetric() is False


def test_builtin_closed_form_cache_uses_the_shortcut_too():
    system = _demo1_like()
    props = system.propagators(t_max=4.0, n_grid_t=10)
    lazy = props.cache._lazy_translation
    t = np.array([1.0, 2.0, 4.0])
    for r in (0.0, 0.7, 1.9):
        props.cache.C_at_batch(t, t, np.zeros(3), np.full(3, r))
    assert lazy.n_grid_builds == 1


# --------------------------------------------------------------------- #
# c_method='auto': node-count selection                                  #
# --------------------------------------------------------------------- #


def test_select_gl_node_count_escalates_with_the_horizon():
    """The demo1 kernel (σ_t = 0.3) is converged at n=20 for t_max=10 but
    not for t_max=100, where a fixed n=20 is ~2% off (documented in
    PropagatorCache).  The selector must notice."""
    model = _demo1_like(sigma_t=0.3).build_propagator_model()
    assert select_gl_node_count(model, 10.0) == 20
    n_long = select_gl_node_count(model, 100.0)
    assert n_long is None or n_long > 20


def test_auto_resolves_per_system_family_at_l1():
    gaussian = _demo1_like(temporal="gaussian", sigma_t=0.5)
    props = gaussian.propagators(t_max=6.0, n_grid_t=10)
    assert props.c_source.startswith("quadrature:gauss_legendre(n="), props.c_source
    n = props.cache.n_gauss_resolved
    assert n >= 20

    # Its table must agree with dblquad at the deep corner: the whole
    # point of the convergence check.
    ref = PropagatorCache(
        gaussian.build_propagator_model(), c_method="dblquad",
        quad_opts={"epsabs": 1e-13, "epsrel": 1e-10},
    )
    got = props.cache._C_value_direct(np.asarray(0.0), 6.0, np.asarray(0.0), 6.0)
    exp = ref._C_value_direct(np.asarray(0.0), 6.0, np.asarray(0.0), 6.0)
    assert np.max(np.abs(got - exp)) <= 1e-6 * np.max(np.abs(exp))


def test_direct_calls_before_any_build_use_dblquad_under_auto():
    model = _demo1_like().build_propagator_model()
    cache = PropagatorCache(model)          # c_method='auto'
    assert cache.c_method == "auto"
    assert cache.c_method_resolved == "dblquad"
    cache.precompute_C_table_translation(t_max=3.0, n_grid_t=6)
    assert cache.c_method_resolved == "gauss_legendre"
    assert cache.n_gauss_resolved == 20


def test_l0_rejects_unknown_c_method():
    model = _demo1_like().build_propagator_model()
    with pytest.raises(ValueError):
        PropagatorCache(model, c_method="simpson")


# --------------------------------------------------------------------- #
# Progress reporting                                                     #
# --------------------------------------------------------------------- #


def test_progress_callback_receives_every_stage_and_numbers_are_unchanged():
    system = _demo1_like()
    events: list = []

    def cb(desc, done, total):
        events.append((desc, done, total))

    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2])
    props = system.propagators(t_max=3.0, n_grid_t=8, c_closed_form=None)
    quiet = exp.sweep(
        props, positions_grid={"x": [0.0], "y": [0.0, 0.5]},
        t_final_grid=[1.0, 3.0], component_pairs=[(0, 0)],
        n_samples=256, progress=False,
    ).totals()
    loud = exp.sweep(
        props, positions_grid={"x": [0.0], "y": [0.0, 0.5]},
        t_final_grid=[1.0, 3.0], component_pairs=[(0, 0)],
        n_samples=256, progress=cb,
    ).totals()
    assert loud.equals(quiet)
    descs = {d for d, _, _ in events}
    assert any(d.startswith("sweep") for d in descs), descs
    finished = [(d, n, t) for d, n, t in events if n == t]
    assert finished, "callback never saw the bar complete"
    # 4 grid points x 7 diagrams (1 at order 0, 6 at order 2).
    assert max(t for _, _, t in events) == 4 * 7


def test_progress_is_silent_when_disabled_and_prints_when_forced(monkeypatch):
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setenv("SFT_WICK_PROGRESS", "0")
    with prog.progress_bar(3, "thing") as tick:
        tick(); tick(); tick()
    assert err.getvalue() == ""

    monkeypatch.setenv("SFT_WICK_PROGRESS", "1")
    monkeypatch.setattr(prog, "_DELAY", 0.0)   # bars hide sub-second loops
    err2 = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err2)
    with prog.progress_bar(3, "thing", unit="cell") as tick:
        tick(); tick(); tick()
    out = err2.getvalue()
    assert "thing" in out and "3/3" in out

    # A loop that finishes inside the delay prints nothing at all.
    monkeypatch.setattr(prog, "_DELAY", 60.0)
    err4 = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err4)
    with prog.progress_bar(3, "fast", unit="cell") as tick:
        tick(); tick(); tick()
    assert err4.getvalue() == ""

    # An explicit scope beats the environment.
    err3 = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err3)
    with prog.progress(False):
        with prog.progress_bar(2, "quiet") as tick:
            tick(); tick()
    assert err3.getvalue() == ""


def test_progress_map_parallel_matches_serial_and_ticks():
    tasks = list(range(12))
    ticks: list = []
    with prog.progress(lambda d, n, t: ticks.append((n, t))):
        serial = prog.progress_map(lambda x: x * x, tasks, "sq", n_jobs=1)
        parallel = prog.progress_map(lambda x: x * x, tasks, "sq", n_jobs=2)
    assert serial == parallel == [x * x for x in tasks]
    assert (12, 12) in ticks
