"""C-table error probes and transactional publication."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import joblib
import numpy as np
import pytest

import sft_wick as sw
from sft_wick import evaluate as ev
from sft_wick.evaluate import CTableAccuracyError, PropagatorCache
from sft_wick.workflow.cache import load_or_compute


def _system():
    return sw.System(
        field=sw.FieldSpec("phi", n_components=1),
        linear=sw.DiagonalA(gamma=[1.0]),
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.1, sigma_t=0.25),
            spatial=sw.ExponentialSpatial(sigma_x=1.0),
        )),
    )


def _exact_c(_x1, t1, _x2, t2):
    time = min(t1, t2)
    return np.array([[time + max(0.0, time - 1.0) ** 2]])


def test_workflow_table_tolerance_defaults_to_one_part_per_million():
    system = _system()
    candidate = system.propagators(4.0, 8, c_closed_form_only=False)
    assert candidate.cache.table_rtol == 1e-6
    assert candidate.cache.table_certificates == {}
    with pytest.raises(CTableAccuracyError, match="table_rtol=1e-06") as exc:
        candidate.cache.C_value(0.0, 1.0, 0.0, 1.0)
    assert exc.value.certificate.time.normalized > 1e-3
    assert exc.value.certificate.space is None
    assert candidate.cache.table_certificates == {}
    assert candidate.cache._lazy_translation._splines_by_key == {}
    intentional = system.propagators(4.0, 8, c_closed_form_only=False, table_rtol=None)
    assert np.isfinite(intentional.cache.C_value(0.0, 1.0, 0.0, 1.0)).all()
    assert intentional.cache.table_certificates == {}


def test_old_coarse_long_horizon_table_is_refused():
    system = _system()
    candidate = system.propagators(100.0, 60, c_closed_form_only=False)
    with pytest.raises(CTableAccuracyError) as exc:
        candidate.cache.C_value(0.0, 100.0, 0.37, 100.0)
    assert exc.value.certificate.estimate > 1e-2


@pytest.mark.parametrize(
    "mode,kwargs,x1,x2",
    [
        ("translation", {"r_max": 1.0, "n_grid_r": 3}, 0.0, 0.5),
        ("rotation", {"n_grid_cos": 3}, np.array([1.0, 0]), np.array([0.0, 1])),
        ("general", {"x_max": 1.0, "n_grid_x": 3}, 0.0, 0.5),
    ],
)
def test_failed_complete_table_rebuild_retains_old_values_and_range(mode, kwargs, x1, x2):
    cache = PropagatorCache(
        _system().build_propagator_model(),
        homogeneity=mode, c_value_fn=_exact_c, table_rtol=1e-6,
    )
    build = getattr(cache, "precompute_C_table_" + mode)
    build(1.0, 6, c_method="gauss_legendre", n_gauss=4, **kwargs)
    value = cache.C_value(x1, 0.75, x2, 0.75)
    certificates = cache.table_certificates
    assert certificates[None].estimate < 1e-12
    rule = cache._table_rules
    with pytest.raises(CTableAccuracyError):
        build(1.5, 6, c_method="gauss_legendre", n_gauss=20, **kwargs)
    assert cache._c_table_range == (0.0, 1.0)
    assert cache._table_rules is rule
    assert cache.table_certificates == certificates
    np.testing.assert_array_equal(cache.C_value(x1, 0.75, x2, 0.75), value)
    with pytest.raises(ValueError, match="table"):
        cache.C_value(x1, 1.375, x2, 1.375)


def test_spatial_probe_finds_off_node_error_in_an_exact_time_table():
    def c(x1, t1, x2, t2):
        return np.array([[min(t1, t2) * np.exp(-float(x1 - x2) ** 2)]])

    cache = PropagatorCache(
        _system().build_propagator_model(), c_value_fn=c, table_rtol=1e-6,
    )
    with pytest.raises(CTableAccuracyError, match="worst space residual") as exc:
        cache.precompute_C_table_translation(1.0, 6, 1.0, 3)
    error = exc.value.certificate
    assert error.time.normalized < 1e-12
    assert error.nodes.normalized < 1e-12
    assert error.space.normalized > 1e-2
    assert cache._c_translation_splines is None


@pytest.mark.parametrize("temporal_interior", [False, True])
def test_general_table_probes_covariance_inside_spatial_cells(temporal_interior):
    def c(x1, t1, x2, t2):
        left, right = x1 * (1 - x1 * x1), x2 * (1 - x2 * x2)
        time = t1 * (1 - t1) * t2 * (1 - t2) if temporal_interior else min(t1, t2)
        return np.array([[left * right * time]])

    n_grid_t = 2 if temporal_interior else 6
    with pytest.raises(CTableAccuracyError) as exc:
        _system().propagators(
            1.0, n_grid_t, homogeneity="general", x_max=1.0, n_grid_x=3,
            c_closed_form=c,
        )
    error = exc.value.certificate
    assert error.time.normalized == error.nodes.normalized == 0.0
    if temporal_interior:
        assert error.space.normalized == 0.0
        assert error.joint.normalized > 0.5
    else:
        assert error.space.normalized > 0.5


def test_lazy_node_probe_catches_inaccurate_quadrature_and_rebuilds_after_refusal():
    cache = PropagatorCache(_system().build_propagator_model(), table_rtol=1e-6)
    cache.precompute_C_table_translation(4.0, 8, c_method="gauss_legendre", n_gauss=2)
    cold = cache._lazy_translation
    original_rules = cache._table_rules.copy()
    for builds in (1, 2):
        with pytest.raises(CTableAccuracyError) as exc:
            cache.C_value(0.0, 1.0, 0.5, 1.0)
        assert exc.value.certificate.nodes.normalized > 1e-2
        assert cold._base_grids is None
        assert cold._splines_by_key == {}
        assert cache.table_certificates == {}
        assert cache._table_rules == original_rules
        assert cold.n_grid_builds == builds


def test_legacy_time_table_rejection_preserves_previous_interpolator(monkeypatch):
    model = _system().build_propagator_model()
    cache = PropagatorCache(model, c_value_fn=_exact_c)
    cache.precompute_C_table(1.0, n_grid=8)
    prior = cache.C_value(0.0, 0.5, 0.0, 0.5)
    old_splines = cache._c_splines

    def refuse(*args):
        raise RuntimeError("diagonal fit was rejected")

    monkeypatch.setattr(ev, "_diag_line_interp", refuse)
    with pytest.raises(RuntimeError, match="diagonal fit was rejected"):
        cache.precompute_C_table(1.5, n_grid=8)
    assert cache._c_splines is old_splines
    assert cache._c_table_range == (0.0, 1.0)
    np.testing.assert_array_equal(cache.C_value(0.0, 0.5, 0.0, 0.5), prior)


def test_legacy_tensor_grid_probe_sees_cusp_off_the_diagonal():
    cache = PropagatorCache(
        _system().build_propagator_model(), c_value_fn=_exact_c, table_rtol=1e-6,
    )
    with pytest.raises(CTableAccuracyError) as exc:
        cache.precompute_C_table(1.0, n_grid=8)
    assert exc.value.certificate.nodes.normalized < 1e-12
    assert exc.value.certificate.time.normalized > 1e-2
    assert cache._c_splines is None


def test_secondary_legacy_table_keeps_published_spatial_bounds_and_certificate():
    def affine_c(_x1, t1, _x2, t2):
        return np.array([[1.0 + t1 + t2]])

    cache = PropagatorCache(
        _system().build_propagator_model(), c_value_fn=affine_c, table_rtol=1e-6,
    )
    cache.precompute_C_table_translation(1.0, 6, 1.0, 3)
    spatial = cache.table_certificates
    cache.precompute_C_table(0.5, n_grid=6)
    assert cache.table_certificates == spatial
    t = np.array([0.75])
    np.testing.assert_allclose(
        cache.C_at_batch(t, t, 0.0, 0.0), [[2.5]], atol=0.0, rtol=1e-14,
    )
    with pytest.raises(ValueError, match="table"):
        cache.C_diagonal_batch(t, t)


def test_lazy_table_persistence_stores_a_cold_recipe(tmp_path):
    path = tmp_path / "props.joblib"
    system = _system()
    props = system.propagators(
        1.0, 8, c_closed_form_only=False, table_rtol=0.1, cache_path=path,
    )
    cold_bytes = path.read_bytes()
    assert props.cache.table_certificates == {}
    assert np.isfinite(props.cache.C_value(0.0, 0.5, 0.5, 0.8)).all()
    assert len(props.cache.table_certificates) == 1
    other = system.propagators(
        1.0, 8, c_closed_form_only=False, table_rtol=0.1, cache_path=path,
    )
    assert other.cache.table_certificates == {}
    assert path.read_bytes() == cold_bytes
    assert other.cache.table_rtol == 0.1


def test_simultaneous_lazy_reads_publish_one_certified_key():
    cache = PropagatorCache(
        _system().build_propagator_model(), c_value_fn=_exact_c, table_rtol=1e-6,
    )
    cache.precompute_C_table_translation(1.0, 6)
    ready = Barrier(2)

    def read():
        ready.wait(timeout=5)
        return cache.C_value(0.0, 0.3, 0.5, 0.7)

    with ThreadPoolExecutor(max_workers=2) as pool:
        readers = [pool.submit(read) for _ in range(2)]
        first, second = (reader.result(timeout=5) for reader in readers)
    np.testing.assert_array_equal(first, second)
    assert len(cache.table_certificates) == 1
    assert cache._lazy_translation.n_grid_builds == 1


def test_direct_batch_reads_bypass_table_lock(monkeypatch):
    cache = _system().propagators(1.0).cache
    assert cache._closed_form_only
    monkeypatch.setattr(cache, "_table_lock", None)
    result = cache.C_at_batch(np.array([0.5]), np.array([0.5]), 0.0, 0.0)
    assert (result > 0).all()


def test_reader_uses_one_table_when_rebuild_changes_bounds(monkeypatch):
    cache = PropagatorCache(
        _system().build_propagator_model(), c_value_fn=_exact_c, table_rtol=1e-6,
    )
    cache.precompute_C_table_translation(1.0, 6, 1.0, 3)
    checked = Event()
    release = Event()
    check_times = cache._refuse_times_past_table

    def pause_after_bounds(t1, t2):
        check_times(t1, t2)
        checked.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(cache, "_refuse_times_past_table", pause_after_bounds)
    t = np.array([0.75])
    with ThreadPoolExecutor(max_workers=2) as pool:
        read = pool.submit(cache.C_at_batch, t, t, 0.0, 0.5)
        try:
            assert checked.wait(timeout=5)
            acquired = cache._table_lock.acquire(blocking=False)
            if acquired:
                cache._table_lock.release()
            assert not acquired
            build = pool.submit(cache.precompute_C_table_translation, 0.5, 6, 1.0, 3)
        finally:
            release.set()
        np.testing.assert_allclose(read.result(timeout=5), [[0.75]], rtol=1e-14, atol=0.0)
        build.result(timeout=5)
    with pytest.raises(ValueError, match="table"):
        cache.C_at_batch(t, t, 0.0, 0.5)


def test_failed_eager_rebuild_preserves_the_last_disk_table(tmp_path):
    path = tmp_path / "props.joblib"
    kwargs = dict(r_max=1.0, n_grid_r=3, cache_path=path)
    system = _system()
    good = system.propagators(1.0, 8, table_rtol=0.1, **kwargs)
    prior_bytes = path.read_bytes()
    assert good.cache.table_certificates[None].estimate < 0.1
    with pytest.raises(CTableAccuracyError):
        system.propagators(1.0, 8, **kwargs)
    assert path.read_bytes() == prior_bytes
    loaded = system.propagators(1.0, 8, table_rtol=0.1, **kwargs)
    assert loaded.cache.table_certificates == good.cache.table_certificates


def test_interrupted_disk_cache_dump_preserves_last_complete_data(tmp_path, monkeypatch):
    path = tmp_path / "cache.joblib"
    assert load_or_compute(path, {"key": 1}, lambda: 2) == 2
    original = path.read_bytes()

    def fail_after_partial_dump(_value, destination):
        destination.write_bytes(b"partial")
        raise OSError("write interrupted")

    monkeypatch.setattr(joblib, "dump", fail_after_partial_dump)
    with pytest.raises(OSError, match="write interrupted"):
        load_or_compute(path, {"key": 1}, lambda: 3, refresh=True)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
