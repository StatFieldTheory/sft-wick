r"""n-point sweeps: ``Expansion.sweep`` over component tuples.

The sweep used to unpack every entry of ``component_pairs`` as a pair
(``for (a, b) in component_pairs``), so a 3- or 4-point observable could be
evaluated one component tuple at a time through ``Expansion.evaluate`` but
could not be swept or run from a YAML config.  On that code every sweep and
every CLI run below raises ``ValueError: too many values to unpack``.

References, none of which uses sft-wick code:

* demo 4, level A (``examples/demo4``): at F = 0 the order-1 κ^m diagram is
  the whole connected m-point function and equals the Campbell closed form
  ``K_R`` of ``poisson_noise.py``.  Its cumulants are symmetric only when
  (component, point) pairs are permuted together, and the two components
  have different amplitudes (of both signs), widths and decay times, so a
  component tuple routed to the wrong operator changes the value.
* a static symmetric κ³ with ``R_a = Θ e^{-γ_a Δt}`` and ``t_min = 0.25``:
  the 3-point function is ``K_ijk M_ai M_bj M_ck`` with
  ``M = diag((1 - e^{-γ_a (T - t_min)}) / γ_a)``, a numpy einsum.

Every sweep row is also compared with ``Expansion.evaluate`` at the same
point, which must agree to the last bit: the sweep calls it, so any
difference is a routing error.
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

import sft_wick as sw
from sft_wick.workflow import cli
from sft_wick.workflow.config import load_workflow_config, run_workflow

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

ROOT = Path(__file__).resolve().parents[1]
DEMO4 = ROOT / "examples" / "demo4"
sys.path.insert(0, str(DEMO4))
import poisson_noise as nz  # noqa: E402
import poisson_system as dsys  # noqa: E402

T = 1.7
POS = {"x": 0.0, "y": 0.8, "z": -0.5, "w": 1.3}
LABELS = ("x", "y", "z", "w")
UNEQUAL = {"x": 1.7, "y": 1.2, "z": 0.6}
PULSES = {"white": nz.PARAMS_WHITE, "exponential": nz.PARAMS_EXP}


def _rel(a, b):
    return abs(a - b) / abs(b)


def _level_a(p, m, rc):
    """Demo 4's level-A expansion and propagators (as ``level_a.py``)."""
    system = dsys.make_system(p, cumulants=(m,), r_contracted=rc)
    props = dsys.propagators_for(system, p, t_max=T + 0.5)
    obs = tuple(f"phi_{c}({lab})" for c, lab in zip("abcd", LABELS[:m]))
    return props, system.expand(obs, orders=[1], diag_C=False)


def _grid(m):
    return {lab: [POS[lab]] for lab in LABELS[:m]}


def _row_value(totals, comp, **coords):
    """The ``value`` of the one totals row at ``comp`` (and ``coords``)."""
    mask = np.ones(len(totals), dtype=bool)
    for col, c in zip("abcd", comp):
        mask &= (totals[col] == c).to_numpy()
    for col, v in coords.items():
        mask &= np.isclose(totals[col].to_numpy(dtype=float), v,
                           rtol=0.0, atol=1e-12)
    assert mask.sum() == 1, (comp, coords, int(mask.sum()))
    return float(totals.loc[mask, "value"].iloc[0])


def _closed(p, comp, labels, times):
    xs = np.array([POS[lab] for lab in labels])
    return float(nz.K_R(comp, xs, np.array(times, float), p)[0])


# ---------------------------------------------------------------------------
# Demo 4, level A through Expansion.sweep
# ---------------------------------------------------------------------------

#: (pulse, R-contracted, integrator, kwargs, tolerance against K_R).  The
#: R-contracted diagram has no time integral left; the raw white-pulse
#: vertex is ``equal_time`` and leaves a 1-D integral.
LEVEL_A_ROUTES = [
    ("exponential", True, "gauss_legendre", dict(n_gauss=8), 1e-12),
    ("exponential", True, "qmc_scalar", dict(n_samples=2 ** 6, seed=3),
     1e-12),
    ("white", True, "qmc_vectorized", dict(n_samples=2 ** 6, seed=3), 1e-12),
    ("white", False, "gauss_legendre", dict(n_gauss=16), 1e-12),
    ("white", False, "qmc_vectorized", dict(n_samples=2 ** 12, seed=3),
     5e-3),
]


@pytest.mark.parametrize(
    "pulse,rc,method,kw,tol", LEVEL_A_ROUTES,
    ids=[f"{r[0]}-{'rc' if r[1] else 'raw'}-{r[2]}" for r in LEVEL_A_ROUTES])
def test_NT1_three_point_sweep_every_triple(pulse, rc, method, kw, tol):
    p = PULSES[pulse]
    props, exp = _level_a(p, 3, rc)
    triples = list(itertools.product(range(2), repeat=3))
    sweep = exp.sweep(props, positions_grid=_grid(3), t_final_grid=[T],
                      component_tuples=triples, orders=[1], method=method,
                      **kw)
    assert sweep.component_keys == ("a", "b", "c")
    totals = sweep.totals()
    assert len(totals) == 8
    for comp in triples:
        got = _row_value(totals, comp)
        direct = exp.evaluate(props, positions=POS, t_final=T,
                              component_pair=comp, orders=[1],
                              method=method, **kw).total
        assert got == direct, comp
        ref = _closed(p, comp, LABELS[:3], [T] * 3)
        assert _rel(got, ref) < tol, (comp, got, ref)


@pytest.mark.parametrize("pulse", sorted(PULSES))
def test_NT2_four_point_sweep_every_quadruple(pulse):
    p = PULSES[pulse]
    props, exp = _level_a(p, 4, rc=True)
    quads = list(itertools.product(range(2), repeat=4))
    totals = exp.sweep(props, positions_grid=_grid(4), t_final_grid=[T],
                       component_tuples=quads, orders=[1],
                       method="gauss_legendre", n_gauss=8).totals()
    assert list(totals.columns) == ["x", "y", "z", "w", "t_final", "a", "b",
                                    "c", "d", "order", "value"]
    for comp in quads:
        got = _row_value(totals, comp)
        direct = exp.evaluate(props, positions=POS, t_final=T,
                              component_pair=comp, orders=[1],
                              method="gauss_legendre", n_gauss=8).total
        assert got == direct, comp
        assert _rel(got, _closed(p, comp, LABELS, [T] * 4)) < 1e-12, comp


@pytest.mark.parametrize("pulse", sorted(PULSES))
def test_NT3_unequal_external_times(pulse):
    """Each absorbed leg takes its partner's time; the rows carry t_x,
    t_y, t_z beside the component columns."""
    p = PULSES[pulse]
    props, exp = _level_a(p, 3, rc=True)
    triples = [(0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 1)]
    sweep = exp.sweep(props, positions_grid=_grid(3), t_final_grid=[1.7],
                      external_times_grid={k: [v] for k, v in UNEQUAL.items()},
                      component_tuples=triples, orders=[1],
                      method="gauss_legendre", n_gauss=8)
    totals = sweep.totals()
    assert {"t_x", "t_y", "t_z", "a", "b", "c"} <= set(totals.columns)
    for comp in triples:
        got = _row_value(totals, comp)
        direct = exp.evaluate(props, positions=POS, t_final=1.7,
                              component_pair=comp, orders=[1],
                              external_times=UNEQUAL,
                              method="gauss_legendre", n_gauss=8).total
        assert got == direct, comp
        ref = _closed(p, comp, LABELS[:3], [UNEQUAL[k] for k in "xyz"])
        assert _rel(got, ref) < 1e-12, comp


def test_NT3_the_rows_are_not_interchangeable():
    """The eight triples give eight different values, so a row carrying
    another triple's value would fail the checks above."""
    props, exp = _level_a(nz.PARAMS_EXP, 3, rc=True)
    triples = list(itertools.product(range(2), repeat=3))
    totals = exp.sweep(props, positions_grid=_grid(3), t_final_grid=[T],
                       component_tuples=triples, orders=[1],
                       method="gauss_legendre", n_gauss=8).totals()
    values = sorted(totals["value"])
    gaps = np.diff(values) / np.abs(values[1:])
    assert np.min(np.abs(gaps)) > 1e-3, values


# ---------------------------------------------------------------------------
# A static κ³ with component-dependent rates and t_min != 0
# ---------------------------------------------------------------------------

_K = np.random.default_rng(3).normal(size=(2, 2, 2))
K_SYM = sum(np.transpose(_K, s) for s in itertools.permutations(range(3))) / 6
T_MIN = 0.25
STATIC_POS = {"x": [0.0], "y": [0.3], "z": [-0.4]}
STATIC_TUPLES = [(0, 1, 1), (1, 0, 0), (1, 1, 1), (0, 0, 0)]


def _static(gammas):
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=gammas),
        nonlocal_vertices=[sw.NonLocalVertex("K", order=3, coupling=K_SYM)],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.3, sigma_t=0.5),
            spatial=sw.ExponentialSpatial(sigma_x=1.0))),
        t_min=T_MIN,
    )
    exp = system.expand(("phi_a(x)", "phi_b(y)", "phi_c(z)"), orders=[1])
    props = system.propagators(t_max=2.0, n_grid_t=10, progress=False)
    g = np.asarray(gammas, float)
    M = np.diag((1.0 - np.exp(-g * (1.2 - T_MIN))) / g)
    return exp, props, np.einsum("ijk,ai,bj,ck->abc", K_SYM, M, M, M)


#: A component-dependent rate makes R a matrix, which the batched
#: integrators refuse today; they get equal rates.
STATIC_ROUTES = [
    ([1.0, 1.4], "nquad", {}, 1e-12),
    ([1.0, 1.4], "qmc_scalar", dict(n_samples=2 ** 9, seed=5), 1e-3),
    ([1.1, 1.1], "gauss_legendre", dict(n_gauss=10), 1e-12),
    ([1.1, 1.1], "qmc_vectorized", dict(n_samples=2 ** 10, seed=5), 1e-3),
]


@pytest.mark.parametrize("gammas,method,kw,tol", STATIC_ROUTES,
                         ids=[r[1] for r in STATIC_ROUTES])
def test_NT4_static_kappa3_every_integrator(gammas, method, kw, tol):
    exp, props, ref = _static(gammas)
    totals = exp.sweep(props, positions_grid=STATIC_POS, t_final_grid=[1.2],
                       component_tuples=STATIC_TUPLES, orders=[1],
                       method=method, **kw).totals()
    for comp in STATIC_TUPLES:
        got = _row_value(totals, comp)
        direct = exp.evaluate(
            props, positions={k: v[0] for k, v in STATIC_POS.items()},
            t_final=1.2, component_pair=comp, orders=[1], method=method,
            **kw).total
        assert got == direct, comp
        assert _rel(got, ref[comp]) < tol, (comp, got, ref[comp])


def test_NT5_parallel_sweep_rows_equal_the_serial_rows():
    exp, props, _ = _static([1.1, 1.1])
    kw = dict(positions_grid={"x": [0.0, 0.2], "y": [0.3], "z": [-0.4]},
              t_final_grid=[1.2, 1.5], component_tuples=STATIC_TUPLES,
              orders=[1], method="gauss_legendre", n_gauss=6)
    serial = exp.sweep(props, **kw).rows
    parallel = exp.sweep(props, n_jobs=2, **kw).rows
    assert len(serial) == 2 * 2 * len(STATIC_TUPLES)
    assert parallel == serial


# ---------------------------------------------------------------------------
# Two-point sweeps are unchanged; defaults
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_point():
    F = np.zeros((2, 2, 2))
    F[0, 1, 1] = 0.7
    F[1, 0, 1] = -0.4
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[1.2, 1.2]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.2, sigma_t=0.4),
            spatial=sw.GaussianSpatial(sigma_x=0.8))))
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2])
    props = system.propagators(t_max=2.0, n_grid_t=12, progress=False)
    return exp, props


_TWO_KW = dict(positions_grid={"x": [0.0], "y": [0.0, 0.6]},
               t_final_grid=[1.0], orders=[0, 2], method="gauss_legendre",
               n_gauss=6)


def test_NT6_component_pairs_rows_and_columns_are_unchanged(two_point):
    exp, props = two_point
    pairs = [(0, 1), (1, 1)]
    by_pairs = exp.sweep(props, component_pairs=pairs, **_TWO_KW)
    by_tuples = exp.sweep(props, component_tuples=pairs, **_TWO_KW)
    assert by_pairs.rows == by_tuples.rows
    assert by_pairs.component_keys == ("a", "b")
    assert list(by_pairs.to_dataframe().columns) == [
        "x", "y", "t_final", "a", "b", "order", "diagram_idx",
        "vertex_type", "n_cross_C", "value", "error"]
    totals = by_pairs.totals()
    assert list(totals.columns) == ["x", "y", "t_final", "a", "b", "order",
                                    "value"]
    for (a, b), y, order in itertools.product(pairs, (0.0, 0.6), (0, 2)):
        direct = exp.evaluate(props, positions={"x": 0.0, "y": y},
                              t_final=1.0, component_pair=(a, b),
                              orders=[order], method="gauss_legendre",
                              n_gauss=6).total
        # The sweep sums the diagrams through pandas and ``evaluate``
        # through Python, so the two differ in the last bit once a
        # diagram is integrated in pieces (the kink split).
        assert _row_value(totals, (a, b), y=y, order=order) == pytest.approx(
            direct, rel=1e-14, abs=0.0)


def test_NT7_defaults_are_the_zero_tuple(two_point):
    exp, props = two_point
    default = exp.sweep(props, **_TWO_KW)
    assert default.rows == exp.sweep(props, component_pairs=[(0, 0)],
                                     **_TWO_KW).rows
    exp3, props3, _ = _static([1.1, 1.1])
    kw = dict(positions={"x": 0.0, "y": 0.3, "z": -0.4}, t_final=1.2,
              orders=[1], method="gauss_legendre", n_gauss=6)
    assert (exp3.evaluate(props3, **kw).total
            == exp3.evaluate(props3, component_pair=(0, 0, 0), **kw).total)
    res = exp3.evaluate(props3, **kw)
    assert res.component_pair == (0, 0, 0)


def test_NT7_one_point_sweep_has_one_component_column():
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        vertices=[sw.LocalVertex("F", coupling=np.full((2, 2, 2), 0.3))],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.2, sigma_t=0.4),
            spatial=sw.GaussianSpatial(sigma_x=0.8))))
    exp = system.expand(("phi_a(x)",), orders=[1])
    props = system.propagators(t_max=2.0, n_grid_t=10, progress=False)
    sweep = exp.sweep(props, positions_grid={"x": [0.0]}, t_final_grid=[1.0],
                      component_tuples=[(0,), (1,)], orders=[1],
                      method="gauss_legendre", n_gauss=6)
    assert sweep.component_keys == ("a",)
    assert list(sweep.totals().columns) == ["x", "t_final", "a", "order",
                                            "value"]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def static3():
    exp, props, _ = _static([1.1, 1.1])
    return exp, props


_SWEEP_KW = dict(positions_grid=STATIC_POS, t_final_grid=[1.2], orders=[1],
                 method="gauss_legendre", n_gauss=4)


@pytest.mark.parametrize("kwargs,match", [
    (dict(component_tuples=[(0, 1)]), r"has 2 entries, but the observable "
     r"<phi_a\(x\) phi_b\(y\) phi_c\(z\)> has 3 operators"),
    (dict(component_tuples=[(0, 1, 1, 0)]), "has 4 entries"),
    (dict(component_pairs=[(0, 0)]), "has 2 entries"),
    (dict(component_tuples=[(0, 2, 1)]), r"outside 0\.\.1"),
    (dict(component_tuples=[(0, -1, 1)]), r"outside 0\.\.1"),
    (dict(component_tuples=[(0, True, 1)]), "not an integer"),
    (dict(component_tuples=[(0, 1.0, 1)]), "not an integer"),
    (dict(component_tuples=(0, 1, 1)), "list of component tuples"),
    (dict(component_tuples=[]), "empty list"),
    (dict(component_tuples=[(0, 1, 1), (0, 1, 1)]), "listed twice"),
    (dict(component_tuples=[(0, 1, 1)], component_pairs=[(0, 1, 1)]),
     "not both"),
])
def test_NT8_bad_component_axis_is_refused(static3, kwargs, match):
    exp, props = static3
    with pytest.raises(ValueError, match=match):
        exp.sweep(props, **kwargs, **_SWEEP_KW)


@pytest.mark.parametrize("comp,match", [
    ((0, 1), "has 2 entries"), ((0, 1, 1, 0), "has 4 entries"),
    ((0, 0, 5), r"outside 0\.\.1"), ((0, 0, -1), r"outside 0\.\.1"),
])
def test_NT8_evaluate_refuses_a_bad_component_tuple(static3, comp, match):
    exp, props = static3
    with pytest.raises(ValueError, match=match):
        exp.evaluate(props, positions={"x": 0.0, "y": 0.3, "z": -0.4},
                     t_final=1.2, component_pair=comp, orders=[1],
                     method="gauss_legendre", n_gauss=4)


def test_NT9_grid_axes_and_columns_are_checked(static3, two_point):
    exp, props = static3
    base = dict(component_tuples=[(0, 1, 1)], orders=[1],
                method="gauss_legendre", n_gauss=4)
    with pytest.raises(ValueError, match="twice"):
        exp.sweep(props, positions_grid=STATIC_POS, t_final_grid=[1.2, 1.2],
                  **base)
    with pytest.raises(ValueError, match="twice"):
        exp.sweep(props, positions_grid={**STATIC_POS, "y": [0.3, 0.3]},
                  t_final_grid=[1.2], **base)
    with pytest.raises(ValueError, match="empty list"):
        exp.sweep(props, positions_grid={**STATIC_POS, "y": []},
                  t_final_grid=[1.2], **base)
    # A spatial label named like a component column would be overwritten.
    exp2, props2 = two_point
    exp_c = exp2.system.expand(("phi_a(b)", "phi_b(y)"), orders=[0])
    with pytest.raises(ValueError, match="coincide with sweep column"):
        exp_c.sweep(props2, positions_grid={"b": [0.0], "y": [0.0]},
                    t_final_grid=[1.0], component_tuples=[(0, 1)],
                    orders=[0], method="gauss_legendre", n_gauss=4)


def test_NT10_numpy_position_arrays_are_accepted(two_point):
    """A positions list given as a numpy array used to raise
    ``TypeError: 'float' object is not iterable``: the row code called
    ``tuple(v.tolist())`` on each numpy scalar."""
    exp, props = two_point
    kw = dict(_TWO_KW, component_tuples=[(0, 1)])
    kw_np = dict(kw, positions_grid={"x": np.array([0.0]),
                                     "y": np.linspace(0.0, 0.6, 2)})
    as_lists = exp.sweep(props, **kw).totals()
    as_arrays = exp.sweep(props, **kw_np).totals()
    np.testing.assert_array_equal(as_arrays["value"].to_numpy(),
                                  as_lists["value"].to_numpy())


# ---------------------------------------------------------------------------
# YAML and the CLI
# ---------------------------------------------------------------------------

def _run_cli(config: Path, workdir: Path, monkeypatch, *extra) -> int:
    monkeypatch.chdir(workdir)
    return cli.main(["run", str(config), "--quiet", *extra])


@pytest.mark.parametrize("config,m", [("config_level_a.yaml", 3),
                                      ("config_level_a_4pt.yaml", 4)])
def test_NT11_cli_reproduces_level_a(tmp_path, monkeypatch, config, m):
    """``sft-wick run`` of the shipped demo-4 configs gives the values of
    ``level_a.py``'s ``Expansion.evaluate`` calls, bit for bit."""
    assert _run_cli(DEMO4 / config, tmp_path, monkeypatch) == 0
    stem = Path(config).stem.replace("config_", "")
    out = np.load(tmp_path / "results" / f"{stem}.npz")
    p = nz.PARAMS_EXP
    props, exp = _level_a(p, m, rc=True)
    tuples = list(itertools.product(range(2), repeat=m))
    assert len(out["value"]) == len(tuples)
    got = {tuple(int(out[c][i]) for c in "abcd"[:m]): float(out["value"][i])
           for i in range(len(out["value"]))}
    for comp in tuples:
        direct = exp.evaluate(props, positions=POS, t_final=T,
                              component_pair=comp, orders=[1],
                              method="gauss_legendre", n_gauss=8).total
        assert got[comp] == direct, comp
        assert _rel(got[comp], _closed(p, comp, LABELS[:m], [T] * m)) < 1e-12


def test_NT11_cli_dry_run_counts_the_tuples(tmp_path, monkeypatch, capsys):
    assert _run_cli(DEMO4 / "config_level_a.yaml", tmp_path, monkeypatch,
                    "--dry-run") == 0
    printed = capsys.readouterr().out
    assert "grid points: 8" in printed
    assert "component_tuples=[(0, 0, 0), (0, 0, 1)" in printed


def _demo4_yaml(tmp_path: Path, edit) -> Path:
    """The shipped 3-point config with absolute module paths, edited and
    written to ``tmp_path``."""
    cfg = yaml.safe_load((DEMO4 / "config_level_a.yaml").read_text())
    hooks = str(DEMO4 / "poisson_l2.py")
    cfg["system"]["nonlocal_vertices"][0]["coupling_module"] = hooks
    cfg["system"]["noise"]["kappa2"]["module"] = hooks
    cfg["propagators"]["c_closed_form_module"] = hooks
    cfg["output"] = []
    edit(cfg)
    path = tmp_path / "level_a.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_NT12_yaml_white_pulses_at_unequal_times(tmp_path):
    """White pulses (``sigma2: callable_module``) and unequal external
    times through YAML, against the L1 evaluate and the closed form."""
    triples = [(0, 1, 1), (1, 0, 1), (1, 1, 0)]

    def edit(cfg):
        cfg["system"]["nonlocal_vertices"][0]["coupling_attr"] = "K3_R_WHITE"
        cfg["system"]["noise"] = {
            "kappa2": {"type": "separable_translation",
                       "temporal": {"type": "exponential", "lam": 0.0,
                                    "sigma_t": 1.0},
                       "spatial": {"type": "gaussian", "sigma_x": 1.0}},
            "sigma2": {"type": "callable_module",
                       "module": str(DEMO4 / "poisson_l2.py"),
                       "attr": "SIGMA2_WHITE"}}
        cfg["propagators"]["c_closed_form_attr"] = "C_WHITE"
        cfg["sweep"]["external_times_grid"] = {k: [v] for k, v in
                                               UNEQUAL.items()}
        cfg["sweep"]["component_tuples"] = [list(c) for c in triples]

    _, totals = run_workflow(load_workflow_config(_demo4_yaml(tmp_path, edit)),
                             progress=False)
    p = nz.PARAMS_WHITE
    props, exp = _level_a(p, 3, rc=True)
    for comp in triples:
        direct = exp.evaluate(props, positions=POS, t_final=1.7,
                              component_pair=comp, orders=[1],
                              external_times=UNEQUAL,
                              method="gauss_legendre", n_gauss=8).total
        got = _row_value(totals, comp)
        assert got == direct, comp
        ref = _closed(p, comp, LABELS[:3], [UNEQUAL[k] for k in "xyz"])
        assert _rel(got, ref) < 1e-12, comp


def test_NT12_yaml_component_pairs_spelling_takes_tuples_too(tmp_path):
    def edit(cfg):
        cfg["sweep"]["component_pairs"] = cfg["sweep"].pop(
            "component_tuples")[:2]

    _, totals = run_workflow(load_workflow_config(_demo4_yaml(tmp_path, edit)),
                             progress=False)
    assert len(totals) == 2 and {"a", "b", "c"} <= set(totals.columns)


@pytest.mark.parametrize("edit,match", [
    (lambda c: c["sweep"].update(component_pairs=[[0, 1, 1]]), "not both"),
    (lambda c: c["sweep"].pop("component_tuples"),
     "component_tuples is required"),
    (lambda c: c["sweep"].update(component_tuples=[[0, 1]]),
     r"has 2 entries, but the observable <phi_a\(x\) phi_b\(y\) "
     r"phi_c\(z\)> has 3"),
    (lambda c: c["sweep"].update(component_tuples=[0, 1, 1]),
     "each entry is a list"),
    (lambda c: c["sweep"].update(component_tuples=[[0, 1, 2]]),
     r"outside 0\.\.1"),
])
def test_NT12_yaml_bad_component_axis_fails_before_the_expansion(
        tmp_path, monkeypatch, edit, match):
    """Refused at load time or before ``System.expand`` runs."""
    import sft_wick.workflow.system as system_mod

    def no_expand(*a, **k):
        raise AssertionError("the expansion ran before the check")

    monkeypatch.setattr(system_mod.System, "expand", no_expand)
    with pytest.raises(ValueError, match=match):
        run_workflow(load_workflow_config(_demo4_yaml(tmp_path, edit)),
                     progress=False)
