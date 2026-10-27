"""Independent quadrature and compatibility checks for bounded GL batches."""

import numpy as np
import pytest


@pytest.mark.parametrize("chunk", [1, 3, 17, 1000])
def test_tensor_blocks_integrate_separable_polynomial(chunk):
    from sft_wick.evaluate import _gauss_legendre_unit_rule, _gl_tensor_blocks

    nodes, weights = _gauss_legendre_unit_rule(4)
    total = 0.0
    count = 0
    for u, w in _gl_tensor_blocks(nodes, weights, 3, chunk):
        assert len(u) <= chunk
        count += len(u)
        total += float(np.sum(w * u[:, 0] ** 3 * u[:, 1] ** 2 * u[:, 2]))
    assert count == 4**3
    assert total == pytest.approx(1 / (4 * 3 * 2), abs=2e-16, rel=0.0)


def test_large_theoretical_grid_stays_lazy(monkeypatch):
    from sft_wick.evaluate import _gauss_legendre_unit_rule, _gl_tensor_blocks

    def forbid_mesh(*args, **kwargs):
        raise AssertionError("Chunked GL must not construct a Cartesian mesh")

    monkeypatch.setattr(np, "meshgrid", forbid_mesh)
    nodes, weights = _gauss_legendre_unit_rule(2)
    blocks = _gl_tensor_blocks(nodes, weights, 70, 3)
    u, w = next(blocks)  # 2**70 total nodes; only three are generated.
    assert u.shape == (3, 70)
    assert w.shape == (3,)
    assert np.all((u > 0) & (u < 1))


def test_unit_rules_are_reused_and_read_only(monkeypatch):
    from numpy.polynomial import legendre
    from sft_wick.evaluate import _gauss_legendre_unit_rule

    _gauss_legendre_unit_rule.cache_clear()
    original = legendre.leggauss
    calls = []

    def record(n):
        calls.append(n)
        return original(n)

    monkeypatch.setattr(legendre, "leggauss", record)
    first = _gauss_legendre_unit_rule(5)
    second = _gauss_legendre_unit_rule(5)
    assert calls == [5]
    assert first is second
    assert not first[0].flags.writeable
    assert not first[1].flags.writeable
    assert _gauss_legendre_unit_rule.cache_info().maxsize == 32


def test_one_node_rule_retains_volume():
    from sft_wick.evaluate import _gauss_legendre_unit_rule, _gl_tensor_blocks

    nodes, weights = _gauss_legendre_unit_rule(1)
    [(u, w)] = list(_gl_tensor_blocks(nodes, weights, 4, 3))
    assert np.array_equal(u, np.full((1, 4), 0.5))
    assert np.array_equal(w, [1.0])
