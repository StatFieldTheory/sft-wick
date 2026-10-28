"""Observable labels spelled like the names the expansion generates.

The expansion names the points of its vertex copies ``y_0, y_1, ...`` and
their component indices ``i_0, i_1, ...``, and the Wick engines key
operators by these strings.  An observable label spelled the same way used
to be merged with the vertex point (or index) of that name: at order 2
``<phi_a(y_0) phi_b(y_1)>`` returned the order-0 value alone, and
``<phi_a(x) phi_b(y_0)>`` raised an error about a response chain joining
two points.  The labels are names for fixed points and fixed components,
and the vertex points and indices are bound variables, so renaming a label
cannot change any value.

The tests compare every spelling with the same observable written with
ordinary labels, order by order, through L0 (both engines, and the
operator-level path), L1 and YAML, and check one spelling against the
exact moment hierarchy of ``examples/reference/ito_moments.py``, which
shares no code with the package.  The model is generic: distinct rates,
a coupling with entries of both signs and no index symmetry, distinct
points, ``t_min != 0``.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

import sft_wick as sw
from sft_wick.evaluate import integrate_diagrams
from sft_wick.expressions import (
    IntegralOver,
    KroneckerDelta,
    Product,
    Propagator,
    Sum,
    SumOverIndex,
    Symbol,
)
from sft_wick.fields import reset_uid_counter
from sft_wick import indices
from sft_wick.perturbation import compute_moment, compute_moment_numerical
from sft_wick.workflow.config import load_workflow_config, run_workflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "examples" / "reference"))
import ito_moments as im  # noqa: E402

GAMMA = (0.7, 1.4)
LAM, SIG_T, SIG_X = 0.6, 0.4, 0.8
T_MIN, T_F = 0.3, 1.8
POINTS = (0.1, 0.65, -0.4, 1.2)
F_BARE = np.array([[[-0.829, -0.526], [0.603, 0.164]],
                   [[-0.812, -0.134], [-0.042, -0.681]]])
K3_BARE = np.array([[[0.31, -0.57], [0.12, 0.44]],
                    [[-0.26, 0.08], [0.73, -0.19]]])
ORDINARY = ("x", "y", "z", "w")
ORDINARY_COMPONENTS = ("a", "b", "c", "d")

TWO_POINT_SPELLINGS = [
    ("y_0", "y_1"), ("y_1", "y_0"), ("x", "y_0"), ("y_0", "x"),
    ("y", "y_1"), ("x", "y_1"), ("y_2", "y_0"),
]


def _rel_close(got: float, want: float, rel: float = 1e-12) -> bool:
    return abs(got - want) <= rel * max(abs(want), abs(got), 1e-300)


def _noise():
    return sw.GaussianNoise(kappa2=sw.SeparableTranslation(
        temporal=sw.ExponentialTemporal(lam=LAM, sigma_t=SIG_T),
        spatial=sw.ExponentialSpatial(sigma_x=SIG_X)))


@pytest.fixture(scope="module")
def system():
    return sw.System(
        field=sw.FieldSpec("phi", 2),
        linear=sw.DiagonalA(gamma=list(GAMMA)),
        vertices=[sw.LocalVertex("F", coupling=F_BARE)],
        noise=_noise(),
        t_min=T_MIN,
    )


@pytest.fixture(scope="module")
def system_k3():
    return sw.System(
        field=sw.FieldSpec("phi", 2),
        linear=sw.DiagonalA(gamma=list(GAMMA)),
        vertices=[sw.LocalVertex("F", coupling=F_BARE)],
        nonlocal_vertices=[sw.NonLocalVertex("K3", order=3, coupling=K3_BARE)],
        noise=_noise(),
        t_min=T_MIN,
    )


@pytest.fixture(scope="module")
def props(system):
    return system.propagators(t_max=2.0, c_closed_form="auto", progress=False)


def _l1_by_order(system, props, labels, components, orders, n_gauss=12):
    obs = tuple(f"phi_{c}({lab})" for c, lab in zip(ORDINARY_COMPONENTS, labels))
    exp = system.expand(obs, orders=orders, progress=False)
    res = exp.evaluate(
        props,
        positions=dict(zip(labels, POINTS)),
        t_final=T_F,
        component_pair=tuple(components),
        orders=orders,
        method="gauss_legendre",
        n_gauss=n_gauss,
    )
    counts = {o: len(exp.diagrams(o)) for o in orders}
    values = {o: complex(res.by_order.get(o, 0.0)).real for o in orders}
    return counts, values


def _assert_same(got, want, rel=1e-12):
    (gc, gv), (wc, wv) = got, want
    assert gc == wc
    for o in wv:
        assert _rel_close(gv[o], wv[o], rel), (o, gv[o], wv[o])


# --------------------------------------------------------------------------
# L1: System.expand / Expansion.evaluate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("components", [(0, 1), (1, 1)])
@pytest.mark.parametrize("labels", TWO_POINT_SPELLINGS)
def test_LC1_l1_two_point_value_independent_of_label_spelling(
        system, props, labels, components):
    """Orders 0-2 of <phi_a phi_b> spelled with y_k labels equal the
    (x, y) spelling, diagram counts included.  Earlier versions returned
    the order-0 value alone for (y_0, y_1) and raised for (x, y_0)."""
    orders = [0, 1, 2]
    want = _l1_by_order(system, props, ORDINARY[:2], components, orders)
    got = _l1_by_order(system, props, labels, components, orders)
    assert want[1][2] != 0.0
    _assert_same(got, want)


@pytest.mark.parametrize("labels", [("y_0", "y_1"), ("x", "y_2")])
def test_LC2_l1_two_point_order_four(system, props, labels):
    """Order 4 (the vertex points reach y_3), same rule on both sides of
    the comparison, so a coarse rule suffices."""
    orders = [4]
    want = _l1_by_order(system, props, ORDINARY[:2], (0, 1), orders, n_gauss=4)
    got = _l1_by_order(system, props, labels, (0, 1), orders, n_gauss=4)
    assert want[1][4] != 0.0
    _assert_same(got, want)


@pytest.mark.parametrize("labels", [
    ("y_0", "y_1", "y_2"), ("y_2", "x", "y_0"), ("x", "y", "y_1"),
])
def test_LC3_l1_three_point_with_nonlocal_vertex(system_k3, labels):
    """<phi_a phi_b phi_c> at orders 1 and 3 with a local F and a static
    non-local kappa^3: the y_k spellings equal (x, y, z)."""
    props = system_k3.propagators(t_max=2.0, c_closed_form="auto", progress=False)
    orders = [1, 3]
    want = _l1_by_order(system_k3, props, ORDINARY[:3], (0, 1, 1), orders,
                        n_gauss=4)
    got = _l1_by_order(system_k3, props, labels, (0, 1, 1), orders, n_gauss=4)
    assert want[1][1] != 0.0 and want[1][3] != 0.0
    _assert_same(got, want)


@pytest.mark.parametrize("labels", [
    ("y_0", "y_1", "y_2", "y_3"), ("y_1", "x", "y_0", "y"),
])
def test_LC4_l1_four_point(system, props, labels):
    """<phi_a phi_b phi_c phi_d> at orders 0 and 2."""
    orders = [0, 2]
    want = _l1_by_order(system, props, ORDINARY, (0, 1, 0, 1), orders, n_gauss=6)
    got = _l1_by_order(system, props, labels, (0, 1, 0, 1), orders, n_gauss=6)
    assert want[1][0] != 0.0 and want[1][2] != 0.0
    _assert_same(got, want)


# --------------------------------------------------------------------------
# L0: compute_moment, compute_moment_numerical, operator-level Wick
# --------------------------------------------------------------------------

L0_SPELLINGS = [
    # (component labels, spatial labels)
    (("a", "b"), ("y_0", "y_1")),
    (("a", "b"), ("y_1", "y_0")),
    (("a", "b"), ("x", "y_0")),
    (("a", "b"), ("y", "y_1")),
    (("i_0", "i_1"), ("x", "y")),
    (("i_1", "i_0"), ("x", "y")),
    (("a", "i_2"), ("x", "y")),
    (("i_0", "i_1"), ("y_0", "y_1")),
    (("y_0", "i_0"), ("i_0", "y_1")),
]
L0_ENGINES = ["compute_moment", "compute_moment_numerical"]
L0_FLAGS = {
    "plain": {},
    "diag": {"diag_R": True, "diag_C": True},
}


def _l0_by_order(system, props, components, labels, engine, flags, order=2):
    reset_uid_counter()
    phi, _psi = system.build_fields()
    obs = [phi(components[0], labels[0]), phi(components[1], labels[1])]
    action = system.build_action()
    cv = system.build_coupling_values()
    fixed = {components[0]: 0, components[1]: 1}
    if engine == "compute_moment_numerical":
        by_order = compute_moment_numerical(
            obs, action, order, cv, fixed_indices=fixed, **flags)
    else:
        res = compute_moment(obs, action, order, **flags)
        by_order = {o: res.diagram_terms(o) for o in range(order + 1)}
    out = {}
    for o in range(order + 1):
        dts = by_order.get(o, [])
        if not dts:
            out[o] = 0.0
            continue
        total, _ = integrate_diagrams(
            dts, coupling_values=cv, lambda_f=T_F, t_min=T_MIN,
            cache=props.cache, method="gauss_legendre", n_gauss=12,
            fixed_indices=fixed, positions=dict(zip(labels, POINTS)))
        out[o] = complex(total).real
    return out


@pytest.mark.parametrize("flags", list(L0_FLAGS))
@pytest.mark.parametrize("engine", L0_ENGINES)
@pytest.mark.parametrize("components,labels", L0_SPELLINGS)
def test_LC5_l0_value_independent_of_label_spelling(
        system, props, components, labels, engine, flags):
    """Spatial labels y_k and component labels i_k (and each spelled in
    the other's namespace) give the (a, b), (x, y) values at orders 0-2.
    Earlier versions lost the order-2 diagrams to the spatial collisions,
    and the component collisions returned values from 0.039 to 0.074 in
    place of 0.0254."""
    fl = L0_FLAGS[flags]
    want = _l0_by_order(system, props, ("a", "b"), ("x", "y"), engine, fl)
    got = _l0_by_order(system, props, components, labels, engine, fl)
    assert want[2] != 0.0
    for o in want:
        assert _rel_close(got[o], want[o]), (o, got[o], want[o])


def _free_names(expr, bound=frozenset()):
    """``(indices, points)`` of ``expr`` not bound by an enclosing sum or
    integral."""
    if isinstance(expr, SumOverIndex):
        return _free_names(expr.body, bound | {expr.index_name})
    if isinstance(expr, IntegralOver):
        return _free_names(expr.body, bound | {expr.variable})
    if isinstance(expr, (Sum, Product)):
        kids = expr.terms if isinstance(expr, Sum) else expr.factors
        parts = [_free_names(k, bound) for k in kids]
        return (set().union(*(p[0] for p in parts)),
                set().union(*(p[1] for p in parts)))
    if isinstance(expr, Propagator):
        idx = {expr.index_left, expr.index_right} - {None}
        pts = {expr.spatial_left, expr.spatial_right}
    elif isinstance(expr, Symbol):
        idx, pts = set(expr.indices), set(expr.spatial_args)
    elif isinstance(expr, KroneckerDelta):
        idx, pts = {expr.index1, expr.index2}, set()
    else:
        idx, pts = set(), set()
    return idx - bound, pts - bound


def _binders(expr):
    """Names bound by the sums and integrals of ``expr``."""
    if isinstance(expr, (SumOverIndex, IntegralOver)):
        name = expr.index_name if isinstance(expr, SumOverIndex) else expr.variable
        return {name} | _binders(expr.body)
    if isinstance(expr, (Sum, Product)):
        kids = expr.terms if isinstance(expr, Sum) else expr.factors
        return set().union(*(_binders(k) for k in kids))
    return set()


SYMBOLIC_SPELLINGS = [
    (("a", "b"), ("y_0", "y_1")),
    (("a", "b"), ("x", "y_1")),
    (("i_0", "i_1"), ("x", "y")),
    (("i_1", "a"), ("y_0", "y")),
]


@pytest.mark.parametrize("flags", [{}, {"diag_R": True, "diag_C": True},
                                   {"iso_R": True}])
@pytest.mark.parametrize("collect_topology", [True, False])
@pytest.mark.parametrize("components,labels", SYMBOLIC_SPELLINGS)
def test_LC6_symbolic_expression_keeps_observable_labels_free(
        system, components, labels, collect_topology, flags):
    """The order-2 expression of compute_moment, on the spatial-topology
    and the operator-level path: the observable's labels are its free
    names, nothing it sums or integrates over is one of them, and the
    numbers of summed and integrated names and of terms match the (a, b),
    (x, y) spelling.  The vertex points used to bind the y_k points, and
    the canonical renaming of the summed indices to bind the i_k
    indices."""
    phi, _psi = system.build_fields()
    action = system.build_action()

    def shape(comps, labs):
        reset_uid_counter()
        obs = [phi(comps[0], labs[0]), phi(comps[1], labs[1])]
        res = compute_moment(obs, action, 2, collect_topology=collect_topology,
                             **flags)
        expr = res.order(2)
        n_terms = len(expr.terms) if isinstance(expr, Sum) else 1
        return _free_names(expr), _binders(expr), n_terms

    (f_idx_ref, f_pts_ref), b_ref, n_ref = shape(("a", "b"), ("x", "y"))
    (f_idx, f_pts), binders, n_terms = shape(components, labels)
    assert (f_idx_ref, f_pts_ref) == ({"a", "b"}, {"x", "y"})
    assert f_idx == set(components)
    assert f_pts == set(labels)
    assert not binders & (set(components) | set(labels))
    assert len(binders) == len(b_ref)
    assert n_terms == n_ref


@pytest.mark.parametrize("engine", ["compute_moment", "compute_moment_numerical"])
@pytest.mark.parametrize("vertex_kind", ["local", "nonlocal", "absorbed",
                                         "equal_time"])
def test_LC7_internal_names_avoid_observable_labels_to_order_four(
        engine, vertex_kind):
    """Structure at order 4 with local vertices and order 3 with a rank-3
    non-local vertex (vertex points up to y_8, so y_8, y_9, y_10 push them
    past two digits): no integration point and no summation index of any
    diagram is an observable label, and the diagram count per order equals
    that of the ordinary spelling."""
    phi = sw.Field("phi", "physical", 2)
    psi = sw.Field("psi", "response", 2)
    vertices = [sw.Vertex([psi, phi, phi], "F")]
    if vertex_kind != "local":
        vertices.append(sw.Vertex(
            [psi, psi, psi], "K", local=False,
            already_R_contracted=vertex_kind == "absorbed",
            equal_time=vertex_kind == "equal_time"))
    action = sw.Action(vertices)
    order = 4 if vertex_kind == "local" else 3
    rng = np.random.default_rng(7)
    cv = {"F": rng.normal(size=(2,) * 3), "K": rng.normal(size=(2,) * 3)}

    def run(comps, labs):
        reset_uid_counter()
        obs = [phi(c, lab) for c, lab in zip(comps, labs)]
        if engine == "compute_moment":
            res = compute_moment(obs, action, order, iso_R=True)
            return {o: res.diagram_terms(o) for o in range(order + 1)}
        return compute_moment_numerical(
            obs, action, order, cv, fixed_indices=dict.fromkeys(comps, 0),
            iso_R=True)

    want = run(("a", "b", "c"), ("x", "y", "z"))
    for comps, labs in [(("a", "b", "c"), ("y_0", "y_3", "y_5")),
                        (("a", "b", "c"), ("y_8", "y_9", "y_10")),
                        (("i_0", "i_2", "c"), ("x", "y_1", "z")),
                        (("i_1", "b", "i_0"), ("y_2", "i_0", "y_0"))]:
        got = run(comps, labs)
        reserved = set(comps) | set(labs)
        assert {o: len(v) for o, v in got.items()} == {
            o: len(v) for o, v in want.items()}
        for dts in got.values():
            for dt in dts:
                assert not set(dt.integration_vars) & reserved
                assert not {n for n, _ in dt.summation_indices} & reserved


# --------------------------------------------------------------------------
# L2: YAML
# --------------------------------------------------------------------------


def _yaml_by_order(tmp_path, labels, components):
    cfg = {
        "system": {
            "field": {"name": "phi", "n_components": 2},
            "t_min": T_MIN,
            "linear": {"type": "diagonal", "gamma": list(GAMMA)},
            "vertices": [{"name": "F", "coupling": F_BARE.tolist()}],
            "noise": {"kappa2": {
                "type": "separable_translation",
                "temporal": {"type": "exponential", "lam": LAM, "sigma_t": SIG_T},
                "spatial": {"type": "exponential", "sigma_x": SIG_X}}},
        },
        "expand": {"observable": [f"phi_a({labels[0]})", f"phi_b({labels[1]})"],
                   "orders": [0, 2]},
        "propagators": {"t_max": 2.0, "c_closed_form": "auto"},
        "sweep": {"positions_grid": {labels[0]: [POINTS[0]], labels[1]: [POINTS[1]]},
                  "t_final_grid": [T_F], "component_pairs": [list(components)],
                  "method": "gauss_legendre", "n_gauss": 12},
    }
    path = tmp_path / f"cfg_{'_'.join(labels)}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    _sweep, totals = run_workflow(load_workflow_config(str(path)), progress=False)
    by = totals.groupby("order")["value"].sum()
    return {int(o): complex(v).real for o, v in by.items()}


@pytest.mark.parametrize("labels", [("y_0", "y_1"), ("x", "y_0"), ("y_1", "x")])
def test_LC8_yaml_value_independent_of_label_spelling(tmp_path, labels):
    want = _yaml_by_order(tmp_path, ("x", "y"), (1, 0))
    got = _yaml_by_order(tmp_path, labels, (1, 0))
    assert set(got) == set(want) == {0, 2}
    assert want[2] != 0.0
    for o in want:
        assert _rel_close(got[o], want[o]), (o, got[o], want[o])


# --------------------------------------------------------------------------
# Independent reference
# --------------------------------------------------------------------------


def _reference(components, kmax=2):
    """E[phi_a(p) phi_b(q)] from the exact Ito moment hierarchy of the
    Markov embedding (phi at two points, exponentially coloured noise
    with spatial correlation exp(-|p - q| / SIG_X)), order by order in the
    coupling.  No sft_wick code."""
    a, b = components
    n_c, n_p = 2, 2
    phi_k = {(c, i): c * n_p + i for c in range(n_c) for i in range(n_p)}
    eta_k = {(c, i): n_c * n_p + c * n_p + i for c in range(n_c) for i in range(n_p)}
    dim = 2 * n_c * n_p
    x = np.array(POINTS[:2])
    k_space = np.exp(-np.abs(x[:, None] - x[None, :]) / SIG_X)
    sde = im.PolySDE(D=dim, n_tags=1)
    drift = np.zeros((dim, dim))
    for (c, i), k in phi_k.items():
        drift[k, k] = -GAMMA[c]
        drift[k, eta_k[(c, i)]] = 1.0
    for k in eta_k.values():
        drift[k, k] = -1.0 / SIG_T
    sde.add_linear_drift(drift)
    diff = np.zeros((dim, dim))
    cov = np.zeros((dim, dim))
    for (c, i), k in eta_k.items():
        for (d, j), m in eta_k.items():
            if c == d:
                diff[k, m] = 2 * LAM / SIG_T * k_space[i, j]
                cov[k, m] = LAM * k_space[i, j]
    sde.add_constant_diffusion(diff)
    quad = np.zeros((dim,) * 3)
    for i in range(n_p):
        for c, d, e in itertools.product(range(n_c), repeat=3):
            quad[phi_k[(c, i)], phi_k[(d, i)], phi_k[(e, i)]] += F_BARE[c, d, e]
    sde.add_quadratic_drift(quad, tag=(1,))

    def initial(alpha, tag):
        if tag != (0,) or any(alpha[k] for k in phi_k.values()):
            return 0.0
        idx = [k for k in range(dim) for _ in range(alpha[k])]
        if not idx:
            return 1.0
        return im.moments_from_cumulants(
            lambda s: cov[s[0], s[1]] if len(s) == 2 else 0.0, idx)

    target = im.unit(dim, phi_k[(a, 0)], phi_k[(b, 1)])
    res = im.solve(sde, [(target, (k,)) for k in range(kmax + 1)],
                   [T_F - T_MIN], initial=initial)
    return [res[(target, (k,))][0] for k in range(kmax + 1)]


@pytest.mark.parametrize("components", [(0, 1), (1, 1)])
def test_LC9_colliding_spelling_matches_ito_hierarchy(system, props, components):
    """(y_1, y_0) against the exact hierarchy at orders 0 and 2.  Measured
    agreement 7.6e-9 at 20 Gauss-Legendre nodes for (0, 1).  Earlier
    versions returned 0 at order 2 for this spelling."""
    ref = _reference(components)
    _counts, got = _l1_by_order(system, props, ("y_1", "y_0"), components,
                                [0, 2], n_gauss=20)
    assert ref[2] != 0.0
    assert got[2] == pytest.approx(ref[2], rel=1e-6, abs=0.0)
    assert got[0] == pytest.approx(ref[0], rel=1e-9, abs=1e-15)


# --------------------------------------------------------------------------
# The generators and the expansion cache key
# --------------------------------------------------------------------------


def test_LC10_index_context_skips_reserved_names():
    IndexContext, fresh_names = indices.IndexContext, indices.fresh_names
    plain = IndexContext()
    assert [plain.fresh_spatial_variable() for _ in range(3)] == ["y_0", "y_1", "y_2"]
    assert [plain.fresh_component_index() for _ in range(3)] == ["i_0", "i_1", "i_2"]
    ctx = IndexContext({"y_0", "y_2", "i_1", "x"})
    assert [ctx.fresh_spatial_variable() for _ in range(3)] == ["y_1", "y_3", "y_4"]
    assert [ctx.fresh_component_index() for _ in range(3)] == ["i_0", "i_2", "i_3"]
    same_prefix = IndexContext()
    names = [same_prefix.fresh_spatial_variable("i"),
             same_prefix.fresh_component_index("i")]
    assert names == ["i_0", "i_1"]
    assert fresh_names("i", 3, ()) == ["i_0", "i_1", "i_2"]
    assert fresh_names("i", 3, {"i_0", "y_1"}) == ["i_1", "i_2", "i_3"]
    phi = sw.Field("phi", "physical", 2)
    assert indices.observable_labels([phi("a", "x"), phi("i_0", "y_0")]) == {
        "a", "x", "i_0", "y_0"}


@pytest.mark.parametrize("label,expected", [
    ("y_0", True), ("y_12", True), ("i_0", True), ("i_7", True),
    ("y_01", False), ("y_", False), ("y_-1", False), ("y_1a", False),
    ("x_0", False), ("yy_0", False), ("y0", False), ("Y_0", False),
    ("y_٣", False), ("y", False), ("i", False),
])
def test_LC11_generated_name_predicate(label, expected):
    assert indices.is_generated_name(label) is expected


def test_LC12_cache_key_unchanged_unless_a_label_is_generated_form(
        system, tmp_path, monkeypatch):
    """Only an observable with a y_k / i_k label gets a new expansion
    cache key, so an entry written by an earlier version for such an
    observable (which holds the merged, wrong expansion) is not read
    back, and every other key is the one it was."""
    from sft_wick.workflow import cache as cache_mod

    seen = []
    real = cache_mod.load_or_compute

    def capture(cache_path, spec_obj, compute_fn, **kw):
        seen.append(dict(spec_obj))
        return real(cache_path, spec_obj, compute_fn, **kw)

    monkeypatch.setattr(cache_mod, "load_or_compute", capture)
    system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2], progress=False)
    assert "labels_reserved" not in seen[-1]
    assert set(seen[-1]) == {"system_hash", "observable", "orders", "flags"}

    # A stale entry under the key the observable had before the marker.
    obs = ("phi_a(y_0)", "phi_b(y_1)")
    system.expand(obs, orders=[0, 2], progress=False)
    assert seen[-1].pop("labels_reserved") is True
    stale = {"key": cache_mod.hash_spec(seen[-1]),
             "value": {"dts_by_order": {0: [], 2: []}, "raw_result": None}}
    from joblib import dump
    dump(stale, tmp_path / f"expansion_{stale['key']}.joblib")
    exp = system.expand(obs, orders=[0, 2], cache_path=tmp_path, progress=False)
    assert len(exp.diagrams(2)) == 6
