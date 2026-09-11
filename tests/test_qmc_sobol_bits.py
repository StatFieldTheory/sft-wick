"""The Sobol samplers use 64-bit points (SB1-SB3).

``scipy.stats.qmc.Sobol`` defaults to ``bits=30``: its points are multiples
of 2^-30 and the first point of every dimension is 0, so the sample mean of
a smooth integrand carries a left-Riemann bias

    mean - integral = -(f(1) - f(0)) / 2^31 + O(statistical),

the same for every seed and every sample count.  It is a floor, not a rate:
it shows wherever the QMC statistical error is already below ~1e-8.  Demo 7
measured it at 1.38e-9 relative on the order-1 channel of
``<phi_0(x, t) phi_1(y, t')>``, where ``nquad``, Gauss-Legendre and the
exact moment hierarchy agree to 3.6e-16.

* **SB1** the bias of the default is the formula, and 64-bit points do not
  have it (scipy only, no sft-wick).
* **SB2** every ``Sobol`` construction in ``src/`` passes ``bits=64``
  explicitly, in the spirit of the ``approx(rel=)`` guard: a new call site
  cannot inherit the default silently.
* **SB3** (slow) the floor is gone from the channel that measured it.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_SB1_default_sobol_bits_carry_a_left_riemann_bias():
    """exp(2u) on [0, 1]: the default's error is the formula, to 1 %, at
    every sample count and seed; 64-bit points reach round-off."""
    exact = (np.e ** 2 - 1) / 2
    predicted = -(np.e ** 2 - 1) / 2 ** 31 / exact
    for m in (14, 18):
        for seed in (1, 3):
            u = qmc.Sobol(d=1, seed=seed).random(2 ** m)[:, 0]
            rel = (np.mean(np.exp(2 * u)) - exact) / exact
            assert rel == pytest.approx(predicted, rel=1e-2, abs=0.0), (m, seed, rel)

            u = qmc.Sobol(d=1, seed=seed, bits=64).random(2 ** m)[:, 0]
            rel = (np.mean(np.exp(2 * u)) - exact) / exact
            assert abs(rel) < 1e-13, (m, seed, rel)


def _sobol_calls(path: Path):
    """Every ``Sobol(...)`` call in ``path``, as ``(lineno, keywords)``."""
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = (fn.attr if isinstance(fn, ast.Attribute)
                else fn.id if isinstance(fn, ast.Name) else None)
        if name == "Sobol":
            out.append((node.lineno, {kw.arg: kw.value for kw in node.keywords}))
    return out


def test_SB2_every_sobol_construction_in_src_passes_bits_64():
    calls = [(p, lineno, kws)
             for p in sorted(SRC.rglob("*.py"))
             for lineno, kws in _sobol_calls(p)]
    assert calls, "no Sobol construction found in src/ -- has it moved?"
    for path, lineno, kws in calls:
        where = f"{path.relative_to(ROOT)}:{lineno}"
        assert "bits" in kws, f"{where}: Sobol without an explicit bits="
        value = kws["bits"]
        assert isinstance(value, ast.Constant) and value.value == 64, where


@pytest.mark.slow
def test_SB3_the_qmc_floor_is_gone_from_the_channel_that_measured_it():
    """Demo 7's two-time order-1 channel: both QMC backends used to stop at
    1.38e-9 relative for every seed and sample count, against 3.6e-16 for
    ``nquad`` and the hierarchy."""
    sys.path.insert(0, str(ROOT / "examples" / "demo7"))
    import space7_model as m7          # noqa: E402
    import space7_run as run7          # noqa: E402

    setup = run7.Setup(m7.CONFIGS["translation-exp"])
    cfg = setup.cfg
    H = cfg.hierarchy([run7.POS["x"], run7.POS["y"]])
    ref = H.two_time([(0, 0)], [(1, 1)], run7.TAGS[1], run7.T, run7.T_EARLY)
    ext = {"x": cfg.t_min + run7.T, "y": cfg.t_min + run7.T_EARLY}
    for route in (run7.QV(18), run7.QS(16)):
        row = run7.run_case(setup, run7.TWO, (0, 1), 1, route, ref=ref,
                            positions=run7.POS, t_final=cfg.t_min + run7.T,
                            orders=(0, 1, 2), external_times=ext)
        assert row["status"] != "refused", row
        assert row["error"] < 1e-11, row
