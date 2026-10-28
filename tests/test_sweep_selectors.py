r"""The diagram selectors of a sweep (``orders``, ``vertex_types``), the
points of ``sweep.positions_grid``, and the shape of the YAML blocks.

Up to 0.7.0 ``Expansion.evaluate`` / ``Expansion.sweep`` filtered with
``set(vertex_types)``, so a bare string was read one character at a time:
``vertex_types="FK"`` selected the label ``F`` and dropped ``FK``.  The
string is what ``--override sweep.vertex_types=FK`` and a YAML
``vertex_types: FK`` produce.  On the model below that run returned the F
channel, -1.03e-2, in place of the FK channel, -2.05e-1, without an error.
A label no diagram carries selected nothing: ``evaluate`` returned 0 and a
sweep ended in ``KeyError: 'x'`` after every integral had run.  An integer
``orders`` (``--override sweep.orders=2``) loaded and then raised
``TypeError: 'int' object is not iterable`` once the expansion and the
propagators were built, a string ``"2"`` was read as the order 2, and an
order 2.5 was truncated to 2.

The same rules reach the other entry points: a Python override may give a
grid, the observable or the component axis as a tuple or a NumPy array
(SS9); ``System.expand`` reads ``orders`` as ``Expansion.evaluate`` does
(SS10); under the connected target a label a diagram could carry selects
the computed zero rather than raise (SS11); ``vertex_types`` takes any
iterable of strings (SS12); and ``sweep.integrate_over`` is checked against
the observable when the file is loaded, one string being one point (SS13).

The model: a local cubic ``F`` and a non-local ``κ³`` vertex ``K`` with a
constant tensor, both with entries of both signs, two components with
distinct rates, ``t_min = 0.3`` and two distinct points.  For a constant
``κ³`` the FK channel of ``⟨φ_a(t) φ_b(t)⟩`` is a one-dimensional integral,

    Σ_cd F_acd K_cdb J_b ∫_{t_min}^{t} e^{-γ_a (t - s)} G_c(s) G_d(s) ds
        + (a ↔ b),

with ``G_c(s) = (1 - e^{-γ_c (s - t_min)}) / γ_c`` and ``J_b = G_b(t)``
(:func:`_fk_reference`, scipy ``quad``; it shares no code with the
package).  The F channel at ``(a, b) = (0, 1)`` is 20 times smaller, so a
selector that picks the wrong channel misses the reference by 95 %.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.integrate import quad

import sft_wick as sw
from sft_wick.workflow import specs as sp
from sft_wick.workflow.cli import _parse_overrides
from sft_wick.workflow.config import load_workflow_config, run_workflow
from sft_wick.workflow.estimate import estimate_cost
from sft_wick.workflow.system import System

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

T_MIN = 0.3
T_FINAL = 1.7
GAMMAS = [0.8, 1.3]
X, Y = 0.1, 0.55
F_TENSOR = np.array([[[0.2, -0.7], [-0.7, 1.1]], [[-0.4, 0.5], [0.5, 0.9]]])
K_TENSOR = np.array([[[0.3, -0.2], [-0.2, 0.6]], [[-0.2, 0.6], [0.6, -0.5]]])
PAIRS = [(0, 1), (1, 1)]
GL = dict(method="gauss_legendre", n_gauss=6)


def _fk_reference(a: int, b: int) -> float:
    """The FK channel of ``⟨φ_a(T) φ_b(T)⟩`` for a constant ``κ³``."""
    g = np.asarray(GAMMAS)

    def G(c, s):
        return (1.0 - np.exp(-g[c] * (s - T_MIN))) / g[c]

    def side(a, b):
        total = 0.0
        for c in range(2):
            for d in range(2):
                inner = quad(
                    lambda s: np.exp(-g[a] * (T_FINAL - s)) * G(c, s) * G(d, s),
                    T_MIN, T_FINAL, epsabs=1e-15, epsrel=1e-13)[0]
                total += F_TENSOR[a, c, d] * K_TENSOR[c, d, b] * inner
        return total * G(b, T_FINAL)

    return side(a, b) + side(b, a)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def system():
    return sw.System(
        field=sp.FieldSpec("phi", n_components=2),
        linear=sp.DiagonalA(gamma=list(GAMMAS)),
        noise=sp.GaussianNoise(kappa2=sp.SeparableTranslation(
            temporal=sp.ExponentialTemporal(lam=0.4, sigma_t=0.7),
            spatial=sp.GaussianSpatial(sigma_x=1.3))),
        vertices=(sp.LocalVertex("F", F_TENSOR),),
        nonlocal_vertices=(sp.NonLocalVertex("K", order=3,
                                             coupling=K_TENSOR),),
        t_min=T_MIN,
    )


@pytest.fixture(scope="module")
def expansion(system):
    return system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2],
                         progress=False)


@pytest.fixture(scope="module")
def props(system):
    return system.propagators(t_max=2.0, n_grid_t=12, progress=False)


def _evaluate(expansion, props, pair=(0, 1), **kw):
    return expansion.evaluate(props, positions={"x": X, "y": Y},
                              t_final=T_FINAL, component_pair=pair,
                              **GL, **kw)


def _sweep(expansion, props, **kw):
    return expansion.sweep(props, positions_grid={"x": [X], "y": [Y]},
                           t_final_grid=[T_FINAL], component_tuples=PAIRS,
                           **GL, **kw)


def _config(**sweep) -> dict:
    cfg = {
        "system": {
            "field": {"name": "phi", "n_components": 2},
            "t_min": T_MIN,
            "linear": {"type": "diagonal", "gamma": list(GAMMAS)},
            "vertices": [{"name": "F", "coupling": F_TENSOR.tolist()}],
            "nonlocal_vertices": [{"name": "K", "order": 3,
                                   "coupling": K_TENSOR.tolist()}],
            "noise": {"kappa2": {
                "type": "separable_translation",
                "temporal": {"type": "exponential", "lam": 0.4,
                             "sigma_t": 0.7},
                "spatial": {"type": "gaussian", "sigma_x": 1.3}}},
        },
        "expand": {"observable": ["phi_a(x)", "phi_b(y)"], "orders": [0, 2]},
        "propagators": {"t_max": 2.0, "n_grid_t": 12},
        "sweep": {"positions_grid": {"x": [X], "y": [Y]},
                  "t_final_grid": [T_FINAL],
                  "component_tuples": [list(p) for p in PAIRS], **GL},
    }
    cfg["sweep"].update(sweep)
    return cfg


def _write(tmp_path: Path, cfg: dict) -> Path:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _channel(totals, a: int, b: int) -> float:
    rows = totals[(totals["a"] == a) & (totals["b"] == b)]
    return float(np.real(rows["value"].sum()))


def _assert_fk(totals) -> None:
    for a, b in PAIRS:
        assert _channel(totals, a, b) == pytest.approx(
            _fk_reference(a, b), rel=1e-9, abs=0.0), (a, b)


# ---------------------------------------------------------------------------
# SS1 -- the reference, and a string is one label at L1
# ---------------------------------------------------------------------------

def test_SS1_fk_channel_matches_the_reference(expansion, props):
    """The list spelling against the independent integral, and the F
    channel far from it, so the comparisons below can tell the channels
    apart."""
    for a, b in PAIRS:
        fk = _evaluate(expansion, props, (a, b), vertex_types=["FK"]).total
        assert fk == pytest.approx(_fk_reference(a, b), rel=1e-9, abs=0.0)
    f = _evaluate(expansion, props, vertex_types=["F"]).total
    assert abs(f) < 0.1 * abs(_fk_reference(0, 1))


@pytest.mark.parametrize("label", ["FK", "F", ""])
def test_SS1_a_string_is_one_label(expansion, props, label):
    """``"FK"`` used to select ``{"F", "K"}``; ``""`` (the order-0 label)
    used to select nothing."""
    by_label = {o: expansion.by_vertex_type(o).get(label, [])
                for o in expansion.orders}
    as_str = _evaluate(expansion, props, vertex_types=label)
    as_list = _evaluate(expansion, props, vertex_types=[label])
    assert as_str.per_diagram == as_list.per_diagram
    assert {r["vertex_type"] for r in as_str.per_diagram} == {label}
    assert len(as_str.per_diagram) == sum(len(v) for v in by_label.values())
    if label == "FK":
        assert as_str.total == pytest.approx(_fk_reference(0, 1), rel=1e-9,
                                             abs=0.0)


def test_SS1_the_channels_add_up_to_the_total(expansion, props):
    total = _evaluate(expansion, props).total
    parts = sum(_evaluate(expansion, props, vertex_types=label).total
                for label in ("", "F", "FK"))
    assert parts == pytest.approx(total, rel=1e-13, abs=1e-15)


def test_SS1_sweep_takes_a_string_too(expansion, props):
    _assert_fk(_sweep(expansion, props, vertex_types="FK").totals())


# ---------------------------------------------------------------------------
# SS2 -- the YAML spellings and the CLI override
# ---------------------------------------------------------------------------

def test_SS2_yaml_string_selects_one_label(tmp_path):
    sweep, totals = run_workflow(
        load_workflow_config(_write(tmp_path, _config(vertex_types="FK"))),
        progress=False)
    assert {r["vertex_type"] for r in sweep.rows} == {"FK"}
    _assert_fk(totals)


def test_SS2_cli_override_selects_one_label(tmp_path):
    """``--override sweep.vertex_types=FK`` on a file without the key: the
    run behind the CRITICAL finding."""
    overrides = _parse_overrides(["sweep.vertex_types=FK"])
    assert overrides == {"sweep.vertex_types": "FK"}
    cfg = load_workflow_config(_write(tmp_path, _config()), overrides=overrides)
    assert cfg.sweep.vertex_types == ["FK"]
    sweep, totals = run_workflow(cfg, progress=False)
    assert {r["vertex_type"] for r in sweep.rows} == {"FK"}
    _assert_fk(totals)


def test_SS2_dry_run_counts_the_selected_diagrams(tmp_path, expansion):
    cfg = load_workflow_config(
        _write(tmp_path, _config()),
        overrides=_parse_overrides(["sweep.vertex_types=FK"]))
    est = estimate_cost(cfg, bench_seconds=0.05)
    assert est.diagrams_per_order == {
        0: 0, 2: len(expansion.by_vertex_type(2)["FK"])}


def test_SS2_dry_run_benches_each_order_on_its_own_labels(tmp_path,
                                                          expansion):
    """The cost estimate evaluates one order at a time; a label of the
    selection that only another order carries must not be refused there."""
    cfg = load_workflow_config(
        _write(tmp_path, _config(vertex_types=["", "FK"])))
    est = estimate_cost(cfg, bench_seconds=0.05)
    assert est.diagrams_per_order == {
        0: 1, 2: len(expansion.by_vertex_type(2)["FK"])}


# ---------------------------------------------------------------------------
# SS3 -- vertex_types of another type, or empty
# ---------------------------------------------------------------------------

_BAD_VERTEX_TYPES = [3, 2.5, {"FK": 1}, ["FK", 2], [None], b"FK",
                     ("F", ["K"])]


@pytest.mark.parametrize("value", _BAD_VERTEX_TYPES, ids=repr)
def test_SS3_vertex_types_of_another_type_raise_TypeError(expansion, props,
                                                          value):
    with pytest.raises(TypeError, match="vertex_types"):
        _evaluate(expansion, props, vertex_types=value)
    with pytest.raises(TypeError, match="vertex_types"):
        _sweep(expansion, props, vertex_types=value)


@pytest.mark.parametrize("value", [3, {"FK": 1}, ["FK", 2], [None]],
                         ids=repr)
def test_SS3_yaml_vertex_types_of_another_type_raise_at_load(tmp_path,
                                                             value):
    with pytest.raises(TypeError, match=r"^sweep\.vertex_types"):
        load_workflow_config(_write(tmp_path, _config(vertex_types=value)))


def test_SS3_empty_vertex_types_is_refused(tmp_path, expansion, props):
    """An empty selection used to give ``evaluate`` a total of 0 and a
    sweep a ``KeyError``."""
    with pytest.raises(ValueError, match="selects no diagram"):
        _evaluate(expansion, props, vertex_types=[])
    with pytest.raises(ValueError, match="selects no diagram"):
        _sweep(expansion, props, vertex_types=set())
    with pytest.raises(ValueError, match=r"^sweep\.vertex_types"):
        load_workflow_config(_write(tmp_path, _config(vertex_types=[])))


# ---------------------------------------------------------------------------
# SS4 -- a label no diagram carries is refused before any integral
# ---------------------------------------------------------------------------

@pytest.fixture
def no_integration(monkeypatch):
    import sft_wick.evaluate as ev

    def boom(*args, **kwargs):
        raise AssertionError("an integral ran before the refusal")

    monkeypatch.setattr(ev, "integrate_diagrams", boom)


@pytest.mark.parametrize("value,orders,absent", [
    (["KF"], None, ["KF"]),
    ("FF", None, ["FF"]),
    (["FK", "KF"], None, ["KF"]),
    (["F"], [0], ["F"]),
])
def test_SS4_absent_label_is_refused(expansion, props, no_integration,
                                     value, orders, absent):
    present = sorted({lbl for o in (orders or expansion.orders)
                      for lbl in expansion.by_vertex_type(o)})
    match = (rf"vertex type\(s\) {absent!r}.*present.*{present!r}"
             .replace("[", r"\[").replace("]", r"\]"))
    with pytest.raises(ValueError, match=match):
        _evaluate(expansion, props, vertex_types=value, orders=orders)
    with pytest.raises(ValueError, match=match):
        _sweep(expansion, props, vertex_types=value, orders=orders)


def test_SS4_yaml_absent_label_is_refused_before_the_propagators(
        tmp_path, monkeypatch):
    def boom(self, *args, **kwargs):
        raise AssertionError("the propagators were built before the refusal")

    monkeypatch.setattr(System, "propagators", boom)
    cfg = load_workflow_config(_write(tmp_path, _config()),
                               overrides={"sweep.vertex_types": "KF"})
    with pytest.raises(ValueError, match=r"^sweep: .*\['KF'\]"):
        run_workflow(cfg, progress=False)


# ---------------------------------------------------------------------------
# SS5 -- orders: an int is one order; other scalars, empty and absent orders
# ---------------------------------------------------------------------------

def test_SS5_an_int_is_one_order(tmp_path, expansion, props):
    res = _evaluate(expansion, props, orders=2, vertex_types="FK")
    assert res.by_order.keys() == {2}
    assert res.total == pytest.approx(_fk_reference(0, 1), rel=1e-9, abs=0.0)
    assert res.per_diagram == _evaluate(
        expansion, props, orders=np.int64(2), vertex_types="FK").per_diagram
    _assert_fk(_sweep(expansion, props, orders=2, vertex_types="FK").totals())

    _sweep_res, totals = run_workflow(load_workflow_config(
        _write(tmp_path, _config(orders=2, vertex_types="FK"))),
        progress=False)
    assert set(totals["order"]) == {2}
    _assert_fk(totals)

    cfg = load_workflow_config(
        _write(tmp_path, _config()),
        overrides=_parse_overrides(["sweep.orders=2",
                                    "sweep.vertex_types=FK"]))
    assert cfg.sweep.orders == [2]
    _assert_fk(run_workflow(cfg, progress=False)[1])


def test_SS5_order_zero_is_an_order_not_none(expansion, props):
    """0 is falsy; it must select order 0, not every order."""
    res = _evaluate(expansion, props, pair=(1, 1), orders=0)
    assert res.by_order.keys() == {0}
    assert res.per_diagram == _evaluate(expansion, props, pair=(1, 1),
                                        orders=[0]).per_diagram
    assert res.total != 0.0


def test_SS5_expand_orders_takes_an_int(tmp_path):
    path = _write(tmp_path, _config(vertex_types="FK"))
    cfg = load_workflow_config(path, overrides=_parse_overrides(
        ["expand.orders=2"]))
    assert cfg.expand.orders == (2,)
    _assert_fk(run_workflow(cfg, progress=False)[1])


_BAD_ORDERS = [True, np.bool_(False), "2", 2.0, [2.5], [True], [None],
               {2: 1}, b"2"]


@pytest.mark.parametrize("value", _BAD_ORDERS, ids=repr)
def test_SS5_orders_of_another_type_raise_TypeError(expansion, props, value):
    """``"2"`` used to select order 2 and ``[2.5]`` order 2; ``True`` and
    ``2.0`` raised ``'bool' object is not iterable``."""
    with pytest.raises(TypeError, match="orders"):
        _evaluate(expansion, props, orders=value)
    with pytest.raises(TypeError, match="orders"):
        _sweep(expansion, props, orders=value)


@pytest.mark.parametrize("value", [True, "2", 2.0, [2.5], {2: 1}], ids=repr)
@pytest.mark.parametrize("block", ["sweep", "expand"])
def test_SS5_yaml_orders_of_another_type_raise_at_load(tmp_path, block,
                                                       value):
    cfg = _config()
    cfg[block]["orders"] = value
    with pytest.raises(TypeError, match=rf"^{block}\.orders"):
        load_workflow_config(_write(tmp_path, cfg))


def test_SS5_empty_or_absent_order_is_refused(tmp_path, expansion, props,
                                              no_integration):
    """``orders=[]`` used to return 0 from ``evaluate``, and an order the
    expansion does not hold raised ``KeyError``."""
    for call in (_evaluate, _sweep):
        with pytest.raises(ValueError, match="selects no order"):
            call(expansion, props, orders=[])
        with pytest.raises(ValueError,
                           match=r"order\(s\) \[1\].*orders \[0, 2\]"):
            call(expansion, props, orders=[1, 2])
    with pytest.raises(ValueError, match=r"^sweep\.orders"):
        load_workflow_config(_write(tmp_path, _config(orders=[])))


def test_SS5_yaml_absent_order_is_refused_before_the_propagators(
        tmp_path, monkeypatch):
    def boom(self, *args, **kwargs):
        raise AssertionError("the propagators were built before the refusal")

    monkeypatch.setattr(System, "propagators", boom)
    cfg = load_workflow_config(_write(tmp_path, _config(orders=[4])))
    with pytest.raises(ValueError, match=r"^sweep: order\(s\) \[4\]"):
        run_workflow(cfg, progress=False)


# ---------------------------------------------------------------------------
# SS6 -- positions_grid names every point of the observable (YAML only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grid,missing", [
    ({"x": [X]}, ["y"]),
    ({"y": [Y]}, ["x"]),
])
def test_SS6_yaml_positions_grid_must_name_every_point(tmp_path, grid,
                                                       missing):
    """An omitted point sat at the default coordinate 0: ``{x: [0.1]}``
    returned the value at a separation of 0.1 in place of 0.45."""
    with pytest.raises(ValueError,
                       match=rf"^sweep\.positions_grid: no position for "
                             rf"{missing!r}".replace("[", r"\[")
                             .replace("]", r"\]")):
        load_workflow_config(_write(tmp_path, _config(positions_grid=grid)))


# ---------------------------------------------------------------------------
# SS7 -- labels YAML does not read as strings
# ---------------------------------------------------------------------------

def _relabelled(a: str, b: str, quoted: bool) -> str:
    cfg = _config(vertex_types=["FK"], orders=[2])
    cfg["expand"]["observable"] = [f"phi_a({a})", f"phi_b({b})"]
    cfg["sweep"]["positions_grid"] = {"__A__": [X], "__B__": [Y]}
    text = yaml.safe_dump(cfg)
    q = "'" if quoted else ""
    return (text.replace("__A__", f"{q}{a}{q}")
            .replace("__B__", f"{q}{b}{q}"))


@pytest.mark.parametrize("a,b", [("1", "2"), ("on", "off"), ("yes", "no")])
def test_SS7_unquoted_number_or_boolean_label_is_explained(tmp_path, a, b):
    """``positions_grid: {1: [0.1], 2: [0.55]}`` used to be refused with
    "given for ['1', '2'] ... its points are ['1', '2']"."""
    path = tmp_path / "c.yaml"
    path.write_text(_relabelled(a, b, quoted=False))
    with pytest.raises(ValueError, match=r"^sweep\.positions_grid: .*quote"):
        load_workflow_config(path)


@pytest.mark.parametrize("a,b", [("1", "2"), ("on", "off")])
def test_SS7_quoted_label_runs(tmp_path, a, b):
    path = tmp_path / "c.yaml"
    path.write_text(_relabelled(a, b, quoted=True))
    _assert_fk(run_workflow(load_workflow_config(path), progress=False)[1])


# ---------------------------------------------------------------------------
# SS8 -- a block of the wrong shape is refused with its name
# ---------------------------------------------------------------------------

def _set(cfg, path, value):
    node = cfg
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return cfg


@pytest.mark.parametrize("path,value,match", [
    (("output",), {"type": "table", "format": "csv"},
     r"^output must be a list"),
    (("output",), "table", r"^output must be a list"),
    (("output",), ["table"], r"^output\[0\] must be a mapping"),
    (("system", "field"), "phi", r"^system\.field must be a mapping"),
    (("system", "vertices"), {"name": "F", "coupling": F_TENSOR.tolist()},
     r"^system\.vertices must be a list"),
    (("system", "nonlocal_vertices"), {"name": "K", "order": 3},
     r"^system\.nonlocal_vertices must be a list"),
    (("expand", "observable"), "phi_a(x)", r"^expand\.observable must be a list"),
    (("sweep", "positions_grid"), [X, Y],
     r"^sweep\.positions_grid must be a mapping"),
    (("sweep", "positions_grid"), {"x": X, "y": [Y]},
     r"^sweep\.positions_grid\['x'\] must be a list"),
    (("sweep", "t_final_grid"), T_FINAL, r"^sweep\.t_final_grid must be a list"),
    (("sweep", "external_times_grid"), [1.0],
     r"^sweep\.external_times_grid must be a mapping"),
    (("sweep", "external_times_grid"), {"x": 1.0},
     r"^sweep\.external_times_grid\['x'\] must be a list"),
])
def test_SS8_block_of_the_wrong_shape_is_refused(tmp_path, path, value,
                                                 match):
    """These used to fail with ``AttributeError: 'str' object has no
    attribute 'get'``, ``'float' object is not iterable``, "each output
    entry must specify a 'type'" for a whole mapping, or (for a string
    observable) a component tuple refused against an observable of eight
    one-character operators."""
    cfg = _set(_config(), path, value)
    with pytest.raises(ValueError, match=match):
        load_workflow_config(_write(tmp_path, cfg))


# ---------------------------------------------------------------------------
# SS9 -- a Python override may give a list-shaped field as a sequence
# ---------------------------------------------------------------------------

def _fk_config(**sweep) -> dict:
    return _config(orders=2, vertex_types="FK", **sweep)


def _totals(path, overrides=None):
    cfg = load_workflow_config(path, overrides=overrides)
    return run_workflow(cfg, progress=False)[1]


@pytest.fixture(scope="module")
def fk_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("fk")


_COLUMNS = ["x", "y", "t_final", "a", "b", "order", "value"]


@pytest.mark.parametrize("overrides", [
    {"sweep.t_final_grid": np.array([T_FINAL])},
    {"sweep.t_final_grid": (T_FINAL,)},
    {"sweep.positions_grid.y": (Y,)},
    {"sweep.positions_grid.y": np.linspace(Y, Y, 1)},
    {"sweep.positions_grid": {"x": np.array([X]), "y": (Y,)}},
    {"expand.observable": ("phi_a(x)", "phi_b(y)")},
    {"expand.observable": np.array(["phi_a(x)", "phi_b(y)"])},
    {"sweep.component_tuples": np.array(PAIRS)},
    {"sweep.component_tuples": tuple(tuple(p) for p in PAIRS)},
    {"sweep.orders": np.arange(2, 3)},
    {"sweep.vertex_types": np.array(["FK"])},
], ids=repr)
def test_SS9_python_override_sequence_matches_the_list_run(fk_dir, overrides):
    """``load_workflow_config(path, overrides=...)`` passes Python values to
    the parser without YAML; a tuple or a NumPy grid used to run (``list()``
    of it) and must give the value of the list spelling."""
    path = _write(fk_dir, _fk_config())
    ref = _totals(path)[_COLUMNS]
    got = _totals(path, overrides)[_COLUMNS]
    assert got.shape == ref.shape
    np.testing.assert_array_equal(got[["a", "b", "order"]].to_numpy(),
                                  ref[["a", "b", "order"]].to_numpy())
    np.testing.assert_allclose(got["value"].to_numpy(),
                               ref["value"].to_numpy(), rtol=1e-14, atol=0)
    _assert_fk(got)


def test_SS9_python_override_of_external_times_takes_an_array(fk_dir):
    path = _write(fk_dir, _fk_config(external_times_grid={"y": [1.2]}))
    ref = _totals(path)
    got = _totals(path, {"sweep.external_times_grid.y": np.array([1.2])})
    assert list(got["t_y"]) == list(ref["t_y"]) == [1.2] * len(PAIRS)
    np.testing.assert_allclose(got["value"].to_numpy(),
                               ref["value"].to_numpy(), rtol=1e-14, atol=0)


@pytest.mark.parametrize("overrides,match", [
    ({"sweep.t_final_grid": np.array(T_FINAL)},
     r"^sweep\.t_final_grid must be a list"),
    ({"sweep.t_final_grid": {"t": T_FINAL}},
     r"^sweep\.t_final_grid must be a list"),
    ({"sweep.t_final_grid": "1.7"}, r"^sweep\.t_final_grid must be a list"),
    ({"sweep.positions_grid.y": b"0.55"},
     r"^sweep\.positions_grid\['y'\] must be a list"),
    ({"expand.observable": {"phi_a(x)", "phi_b(y)"}},
     r"^expand\.observable: .*set"),
    ({"sweep.component_tuples": "01"},
     r"^sweep\.component_tuples must be a non-empty list"),
    ({"sweep.component_tuples": np.zeros((0, 2), dtype=int)},
     r"^sweep\.component_tuples must be a non-empty list"),
], ids=repr)
def test_SS9_override_of_another_shape_is_refused(fk_dir, overrides, match):
    """A string, bytes, a mapping, a scalar (a 0-d array too) and an
    unordered set are still refused: ``list()`` would split the first two
    into characters, keep a mapping's keys, and give a set's operators in
    hash order."""
    with pytest.raises(ValueError, match=match):
        load_workflow_config(_write(fk_dir, _fk_config()), overrides=overrides)


# ---------------------------------------------------------------------------
# SS10 -- System.expand reads orders as Expansion.evaluate does
# ---------------------------------------------------------------------------

def test_SS10_system_expand_takes_an_int(system, expansion, props):
    """``System.expand(obs, orders=2)`` used to raise ``'int' object is not
    iterable`` while ``Expansion.evaluate(orders=2)`` ran."""
    e2 = system.expand(("phi_a(x)", "phi_b(y)"), orders=2, progress=False)
    assert e2.orders == (2,)
    assert len(e2.diagrams(2)) == len(expansion.diagrams(2))
    for a, b in PAIRS:
        assert _evaluate(e2, props, (a, b), vertex_types="FK").total == (
            pytest.approx(_fk_reference(a, b), rel=1e-9, abs=0.0))
    assert system.expand(("phi_a(x)", "phi_b(y)"), orders=np.int64(0),
                         progress=False).orders == (0,)


@pytest.mark.parametrize("value", [
    "2", "02", [2.5], [True, 2], True, np.bool_(True), 2.0, [None], {2: 1},
    None,
], ids=repr)
def test_SS10_system_expand_refuses_other_orders(system, value):
    """``"2"`` gave the orders (2,), ``"02"`` (0, 2), ``[2.5]`` (2,) and
    ``[True, 2]`` (1, 2).  (``b"2"`` gave the order 50, which this test
    leaves out: before the fix it expands for minutes rather than fail.)"""
    with pytest.raises(TypeError, match=r"^System\.expand: orders"):
        system.expand(("phi_a(x)", "phi_b(y)"), orders=value, progress=False)


def test_SS10_empty_orders_are_refused_with_the_right_advice(system, tmp_path):
    """``expand.orders`` and ``System.expand`` have no "every order" to
    fall back on, so the message must not suggest ``None``."""
    with pytest.raises(ValueError, match=r"^System\.expand: orders is empty"
                                         r"(?!.*None)"):
        system.expand(("phi_a(x)", "phi_b(y)"), orders=[], progress=False)
    cfg = _config()
    cfg["expand"]["orders"] = []
    with pytest.raises(ValueError, match=r"^expand\.orders is empty(?!.*None)"):
        load_workflow_config(_write(tmp_path, cfg))


# ---------------------------------------------------------------------------
# SS11 -- connected target: a label a diagram could carry is a zero
# ---------------------------------------------------------------------------

POS4 = {"x": 0.1, "y": 0.55, "z": -0.3, "w": 0.9}
COMP4 = (0, 1, 1, 0)


@pytest.fixture(scope="module")
def connected4(system):
    """The connected 4-point function: the incidence bound leaves no
    diagram at order 1 and no K-only diagram at order 2."""
    exp = system.expand(("phi_a(x)", "phi_b(y)", "phi_c(z)", "phi_d(w)"),
                        orders=[1, 2], target="connected", progress=False)
    assert exp.diagrams(1) == []
    assert sorted(exp.by_vertex_type(2)) == ["F", "FK"]
    return exp


def _evaluate4(exp, props, **kw):
    return exp.evaluate(props, positions=POS4, t_final=T_FINAL,
                        component_pair=COMP4, **GL, **kw)


def _sweep4(exp, props, **kw):
    return exp.sweep(props, positions_grid={k: [v] for k, v in POS4.items()},
                     t_final_grid=[T_FINAL], component_tuples=[COMP4],
                     **GL, **kw).totals()


@pytest.mark.parametrize("orders,labels", [
    ([1], ["F"]),
    ([1], "K"),
    ([1, 2], ["K"]),
    ([2], "K"),
], ids=repr)
def test_SS11_connected_label_without_a_diagram_is_a_zero(
        connected4, props, orders, labels):
    """The connected target reports an order no diagram reaches as a
    computed zero; selecting a channel there gave that zero up to 0.7.0
    and must not raise."""
    res = _evaluate4(connected4, props, orders=orders, vertex_types=labels)
    assert res.by_order == {o: 0.0 for o in orders}
    assert res.per_diagram == [] and res.total == 0.0
    totals = _sweep4(connected4, props, orders=orders, vertex_types=labels)
    assert sorted(totals["order"]) == orders
    assert (totals["value"] == 0.0).all()


@pytest.mark.parametrize("orders,labels,absent", [
    ([1, 2], ["KF"], ["KF"]),
    ([1, 2], "FF", ["FF"]),
    ([1], "FK", ["FK"]),
    ([1, 2], ["F", "G"], ["G"]),
    ([1, 2], "", [""]),
], ids=repr)
def test_SS11_connected_label_no_diagram_could_carry_is_refused(
        connected4, props, no_integration, orders, labels, absent):
    """A misspelt label, a vertex the system lacks, two names at order 1
    and the order-0 label at an interacting order are still refused."""
    match = (rf"vertex type\(s\) {absent!r}"
             .replace("[", r"\[").replace("]", r"\]"))
    with pytest.raises(ValueError, match=match):
        _evaluate4(connected4, props, orders=orders, vertex_types=labels)
    with pytest.raises(ValueError, match=match):
        _sweep4(connected4, props, orders=orders, vertex_types=labels)


def test_SS11_moment_target_still_refuses_a_label_without_a_diagram(
        expansion, props, no_integration):
    """The moment target has no zero rows, so the K channel the order-2
    expansion lacks is refused there."""
    assert "K" not in expansion.by_vertex_type(2)
    with pytest.raises(ValueError, match=r"vertex type\(s\) \['K'\]"):
        _evaluate(expansion, props, vertex_types="K")


# ---------------------------------------------------------------------------
# SS12 -- vertex_types may be any iterable of strings
# ---------------------------------------------------------------------------

def _gen():
    yield "FK"


@pytest.mark.parametrize("make", [
    lambda: np.array(["FK"]),
    lambda: __import__("pandas").Index(["FK"]),
    _gen,
    lambda: iter(["FK"]),
    lambda: {"FK": None}.keys(),
    lambda: np.str_("FK"),
    lambda: ("FK",),
], ids=["ndarray", "Index", "generator", "iterator", "dict_keys", "np.str_",
        "tuple"])
def test_SS12_any_iterable_of_labels_selects_them(expansion, props, make):
    """``df['vertex_type'].unique()`` is a NumPy array; it used to select
    the FK channel and must still."""
    ref = _evaluate(expansion, props, vertex_types=["FK"])
    assert _evaluate(expansion, props,
                     vertex_types=make()).per_diagram == ref.per_diagram
    _assert_fk(_sweep(expansion, props, vertex_types=make()).totals())


@pytest.mark.parametrize("value", [
    np.array("FK"), bytearray(b"FK"), [b"FK"], np.array([1, 2]),
], ids=repr)
def test_SS12_iterables_of_other_entries_are_refused(expansion, props, value):
    with pytest.raises(TypeError, match="vertex_types"):
        _evaluate(expansion, props, vertex_types=value)


# ---------------------------------------------------------------------------
# SS13 -- sweep.integrate_over is checked when the file is loaded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,exc,match", [
    ("z", ValueError, r"\['z'\].*points are \['x', 'y'\]"),
    ("xy", ValueError, r"\['xy'\].*points are \['x', 'y'\]"),
    (["x", "z"], ValueError, r"\['z'\].*points are \['x', 'y'\]"),
    (["all"], ValueError, r"\['all'\].*points are \['x', 'y'\]"),
    (True, TypeError, r"all"),
    (3, TypeError, r"all"),
    ({"x": 1}, TypeError, r"all"),
    ([1], TypeError, r"the entry 1"),
    ([["x"]], TypeError, r"the entry \['x'\]"),
], ids=repr)
def test_SS13_yaml_integrate_over_is_checked_at_load(fk_dir, value, exc,
                                                     match):
    """``integrate_over: x`` loaded and then failed in ``evaluate`` after
    the propagators were built; a name the observable does not use loaded
    too."""
    with pytest.raises(exc, match=r"^sweep\.integrate_over.*" + match):
        load_workflow_config(_write(fk_dir, _fk_config(integrate_over=value)))


def test_SS13_one_point_name_is_that_point(fk_dir, expansion, props):
    """One string other than ``all`` is one point name, as one string is
    one vertex-type label, so ``--override sweep.integrate_over=x`` runs."""
    ref = _sweep(expansion, props, orders=2, vertex_types="FK",
                 integrate_over={"x"}).totals()
    fixed = _sweep(expansion, props, orders=2, vertex_types="FK").totals()
    assert np.all(np.abs(ref["value"].to_numpy()
                         - fixed["value"].to_numpy()) > 1e-3)
    configs = [
        load_workflow_config(_write(fk_dir, _fk_config(integrate_over="x"))),
        load_workflow_config(_write(fk_dir, _fk_config()),
                             overrides=_parse_overrides(
                                 ["sweep.integrate_over=x"])),
        load_workflow_config(_write(fk_dir, _fk_config(integrate_over=["x"]))),
        load_workflow_config(_write(fk_dir, _fk_config()),
                             overrides={"sweep.integrate_over": ("x",)}),
    ]
    for cfg in configs:
        assert cfg.sweep.integrate_over == ["x"]
        got = run_workflow(cfg, progress=False)[1]
        np.testing.assert_allclose(got["value"].to_numpy(),
                                   ref["value"].to_numpy(), rtol=1e-13, atol=0)


@pytest.mark.parametrize("value", [None, "all", []], ids=repr)
def test_SS13_null_all_and_empty_load_as_given(fk_dir, value):
    cfg = load_workflow_config(_write(fk_dir, _fk_config(integrate_over=value)))
    assert cfg.sweep.integrate_over == value
