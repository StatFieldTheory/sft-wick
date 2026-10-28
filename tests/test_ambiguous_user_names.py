"""Names the user chooses that the workflow layer read in two ways.

Up to 0.7.0 three spellings returned a number without an error, and a
fourth raised an error about the wrong thing:

* two observable operators sharing one component letter and given
  different component indices (``<phi_a(x) phi_a(y)>`` with ``(0, 1)``):
  the second index replaced the first, so the value and the rows were the
  ``(1, 1)`` ones;
* two vertices with one name: coupling values are keyed by name, so the
  second coupling replaced the first for both vertices;
* vertex names whose sorted concatenations agree for two different sets
  (``F``, ``K`` and ``FK``): the vertex-type label ``"FK"`` named both
  the ``F K`` channel and the ``FK`` channel, so ``by_vertex_type``
  merged them and ``vertex_types=["FK"]`` selected both;
* a ``positions`` key spelled like a vertex point (``y_0``): the L0
  resolver read it as that point's position, so it moved the point's
  group or raised an error about a response chain joining two points.

The first three are refused with a message naming the user's spelling.
A positions key for a vertex point is ignored, like any key that names
no external point, so one positions dict can serve observables whose
labels are spelled like vertex points.  The label check is compared with
a brute-force enumeration of the subsets of each name set, which shares
no code with the package.  The model is generic: distinct rates,
couplings with entries of both signs and no index symmetry, distinct
points, ``t_min != 0``.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import yaml

import sft_wick as sw
from sft_wick.evaluate import integrate_diagrams, integrate_moment
from sft_wick.fields import reset_uid_counter
from sft_wick.perturbation import compute_moment
from sft_wick.workflow.config import load_workflow_config, run_workflow

GAMMA = (0.7, 1.4)
LAM, SIG_T, SIG_X = 0.6, 0.4, 0.8
T_MIN, T_F = 0.3, 1.8
POS = {"x": 0.1, "y": 0.65, "z": -0.4}
F_BARE = np.array([[[-0.829, -0.526], [0.603, 0.164]],
                   [[-0.812, -0.134], [-0.042, -0.681]]])
G_BARE = np.array([[[0.212, -0.447], [0.391, -0.058]],
                   [[-0.305, 0.517], [0.126, 0.274]]])
K3_BARE = np.array([[[0.31, -0.57], [0.12, 0.44]],
                    [[-0.26, 0.08], [0.73, -0.19]]])


def _noise():
    return sw.GaussianNoise(kappa2=sw.SeparableTranslation(
        temporal=sw.ExponentialTemporal(lam=LAM, sigma_t=SIG_T),
        spatial=sw.ExponentialSpatial(sigma_x=SIG_X)))


def _system(vertices, nonlocal_vertices=()):
    return sw.System(
        field=sw.FieldSpec("phi", 2),
        linear=sw.DiagonalA(gamma=list(GAMMA)),
        vertices=list(vertices),
        nonlocal_vertices=list(nonlocal_vertices),
        noise=_noise(),
        t_min=T_MIN,
    )


@pytest.fixture(scope="module")
def system():
    return _system([sw.LocalVertex("F", coupling=F_BARE)],
                   [sw.NonLocalVertex("K", order=3, coupling=K3_BARE)])


@pytest.fixture(scope="module")
def props(system):
    return system.propagators(t_max=2.0, c_closed_form="auto", progress=False)


def _l1(system, props, observable, components, orders, n_gauss=6):
    exp = system.expand(observable, orders=orders, progress=False)
    labels = [op.split("(")[1].rstrip(")") for op in observable]
    res = exp.evaluate(
        props, positions={lab: POS[lab] for lab in labels}, t_final=T_F,
        component_pair=components, orders=orders, method="gauss_legendre",
        n_gauss=n_gauss)
    counts = {o: len(exp.diagrams(o)) for o in orders}
    values = {o: complex(res.by_order.get(o, 0.0)).real for o in orders}
    return counts, values


def _yaml_config(tmp_path, observable, components, vertices=None):
    cfg = {
        "system": {
            "field": {"name": "phi", "n_components": 2},
            "t_min": T_MIN,
            "linear": {"type": "diagonal", "gamma": list(GAMMA)},
            "vertices": vertices or [{"name": "F", "coupling": F_BARE.tolist()}],
            "noise": {"kappa2": {
                "type": "separable_translation",
                "temporal": {"type": "exponential", "lam": LAM, "sigma_t": SIG_T},
                "spatial": {"type": "exponential", "sigma_x": SIG_X}}},
        },
        "expand": {"observable": list(observable), "orders": [0, 2]},
        "propagators": {"t_max": 2.0, "c_closed_form": "auto"},
        "sweep": {"positions_grid": {"x": [0.1], "y": [0.65]},
                  "t_final_grid": [T_F],
                  "component_pairs": [list(c) for c in components],
                  "method": "gauss_legendre", "n_gauss": 8},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return load_workflow_config(str(path))


# --------------------------------------------------------------------------
# One component letter on two operators
# --------------------------------------------------------------------------

SHARED_LETTER_CONFLICTS = [
    (("phi_a(x)", "phi_a(y)"), (0, 1)),
    (("phi_a(x)", "phi_a(y)"), (1, 0)),
    (("phi_a(x)", "phi_b(y)", "phi_a(z)"), (0, 1, 1)),
    (("phi_b(x)", "phi_a(y)", "phi_a(z)"), (1, 1, 0)),
]


@pytest.mark.parametrize("observable,components", SHARED_LETTER_CONFLICTS)
def test_AN1_shared_component_letter_with_two_indices_raises_at_l1(
        system, props, observable, components):
    """Up to 0.7.0 the later index replaced the earlier one: measured on
    this model at 12 Gauss-Legendre nodes, <phi_a(x) phi_a(y)> with
    (0, 1) returned 0.0527235 at order 0 and 0.1077365 at order 2, the
    (1, 1) values, where (0, 1) is 0 and -0.0343829."""
    orders = [0, 1] if len(observable) == 3 else [0, 2]
    exp = system.expand(observable, orders=orders, progress=False)
    labels = [op.split("(")[1].rstrip(")") for op in observable]
    positions = {lab: POS[lab] for lab in labels}
    with pytest.raises(ValueError, match="component letter 'a'"):
        exp.evaluate(props, positions=positions, t_final=T_F,
                     component_pair=components, orders=orders,
                     method="gauss_legendre", n_gauss=4)
    grid = {lab: [POS[lab]] for lab in labels}
    same = (components[labels.index(labels[0])],) * len(observable)
    with pytest.raises(ValueError, match="component letter 'a'"):
        exp.sweep(props, positions_grid=grid, t_final_grid=[T_F],
                  component_tuples=[same, components], orders=orders,
                  method="gauss_legendre", n_gauss=4)


def test_AN2_shared_component_letter_with_two_indices_raises_in_yaml(tmp_path):
    """The YAML run used to emit rows labelled a=0, b=1 holding the (1, 1)
    values; it now stops before building anything."""
    cfg = _yaml_config(tmp_path, ["phi_a(x)", "phi_a(y)"], [(0, 1)])
    with pytest.raises(ValueError, match="component letter 'a'"):
        run_workflow(cfg, progress=False)


@pytest.mark.parametrize("observable,shared,components,orders", [
    (("phi_a(x)", "phi_b(y)"), ("phi_a(x)", "phi_a(y)"), (0, 0), [0, 1, 2]),
    (("phi_a(x)", "phi_b(y)"), ("phi_a(x)", "phi_a(y)"), (1, 1), [0, 1, 2]),
    (("phi_a(x)", "phi_b(y)", "phi_c(z)"), ("phi_a(x)", "phi_a(y)", "phi_c(z)"),
     (1, 1, 0), [1]),
    (("phi_a(x)", "phi_b(y)", "phi_c(z)"), ("phi_c(x)", "phi_a(y)", "phi_c(z)"),
     (0, 1, 0), [1]),
])
def test_AN3_shared_component_letter_with_one_index_is_unchanged(
        system, props, observable, shared, components, orders):
    """A shared letter given one index stays legal and equals the
    distinct-letter spelling, diagram counts included (the distinct-letter
    two-point value is checked against the exact moment hierarchy in
    test_observable_label_collision.py::test_LC9)."""
    want = _l1(system, props, observable, components, orders)
    got = _l1(system, props, shared, components, orders)
    assert got[0] == want[0]
    assert any(v != 0.0 for v in want[1].values())
    for o in orders:
        assert got[1][o] == pytest.approx(want[1][o], rel=1e-12, abs=0.0)


# --------------------------------------------------------------------------
# Vertex names
# --------------------------------------------------------------------------


@pytest.mark.parametrize("vertices,nonlocal_vertices", [
    ([sw.LocalVertex("F", coupling=F_BARE), sw.LocalVertex("F", coupling=G_BARE)],
     []),
    ([sw.LocalVertex("F", coupling=F_BARE)],
     [sw.NonLocalVertex("F", order=3, coupling=K3_BARE)]),
])
def test_AN4_two_vertices_with_one_name_raise(vertices, nonlocal_vertices):
    """Coupling values are keyed by name, so the second coupling replaced
    the first for both vertices: measured with local couplings F and G
    both named F, order 2 gave -0.0109881 at (0, 1), where one vertex with
    the coupling F + G, or the two vertices named apart, give 0.0186744."""
    with pytest.raises(ValueError, match="vertex name 'F'"):
        _system(vertices, nonlocal_vertices)


def test_AN4c_multiplicative_vertex_names_count_for_the_labels():
    """A MultiplicativeImpulse lowers to vertices named G and H, so a
    local vertex named GH would share the label "GH" with the {G, H}
    channel."""
    g0 = np.array([[0.4, -0.2], [0.1, 0.3]])
    g1 = np.zeros((2, 2, 2))
    g1[0, 0, 1], g1[1, 1, 0] = 0.25, -0.15
    with pytest.raises(ValueError, match="vertex-type label") as info:
        sw.System(
            field=sw.FieldSpec("phi", 2), linear=sw.DiagonalA(gamma=list(GAMMA)),
            vertices=[sw.LocalVertex("GH", coupling=F_BARE)],
            noise=sw.GaussianNoise(kappa2=_noise().kappa2,
                                   sigma2=sw.MultiplicativeImpulse(g0=g0, g1=g1)),
            t_min=T_MIN)
    assert "{G, H}" in str(info.value) and "{GH}" in str(info.value)


def _brute_force_label_collision(names):
    """Two distinct non-empty subsets of ``names`` whose sorted names
    concatenate to one string, or None."""
    seen = {}
    for k in range(1, len(names) + 1):
        for subset in itertools.combinations(sorted(names), k):
            label = "".join(subset)
            if label in seen:
                return seen[label], subset
            seen[label] = subset
    return None


def _local_system(names):
    return _system([sw.LocalVertex(n, coupling=F_BARE) for n in names])


@pytest.mark.parametrize("local,nonlocal_name,sets", [
    (("F", "FK"), "K", ("{F, K}", "{FK}")),
    (("A", "AB", "BC", "C"), None, ("{A, BC}", "{AB, C}")),
])
def test_AN5_ambiguous_vertex_type_labels_raise(local, nonlocal_name, sets):
    """With local F and FK and non-local K, measured at order 2 on this
    model: vertex_types=["FK"] gave -0.0624845, the F K channel plus the
    FK FK channel, where the F K channel alone (FK renamed G) is
    -0.0597375.  {A, BC} and {AB, C} share "ABC" although no name is a
    concatenation of others."""
    vertices = [sw.LocalVertex(n, coupling=F_BARE if i == 0 else G_BARE)
                for i, n in enumerate(local)]
    nonlocal_vertices = (
        [] if nonlocal_name is None
        else [sw.NonLocalVertex(nonlocal_name, order=3, coupling=K3_BARE)])
    with pytest.raises(ValueError, match="vertex-type label") as info:
        _system(vertices, nonlocal_vertices)
    for s in sets:
        assert s in str(info.value)


NAME_POOL = ("A", "B", "AB", "BA", "ABA", "C", "BC", "K", "K3", "3", "F", "FK")


def test_AN6_label_check_matches_brute_force_enumeration():
    """Over every set of one to five names from a pool rich in prefixes,
    the System refuses exactly the sets in which two subsets share a
    label (227 of the 1585 sets)."""
    n_refused = n_built = 0
    for k in range(1, 6):
        for names in itertools.combinations(NAME_POOL, k):
            collision = _brute_force_label_collision(names)
            if collision is None:
                _local_system(names)
                n_built += 1
            else:
                with pytest.raises(ValueError, match="vertex-type label"):
                    _local_system(names)
                n_refused += 1
    assert (n_refused, n_built) == (227, 1358)


def test_AN7_unambiguous_names_keep_their_labels(system):
    """The names the examples use (single letters, K3, W2) still build,
    and the labels are the sorted concatenations they always were."""
    for names in [("F", "G", "K"), ("F", "K3"), ("K", "K3", "F"), ("W2", "F")]:
        _local_system(names)
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2], progress=False)
    assert set(exp.by_vertex_type(2)) == {"F", "FK"}


# --------------------------------------------------------------------------
# positions keys that name vertex points
# --------------------------------------------------------------------------


def _l0_by_order(system, props, labels, positions, max_order=2):
    reset_uid_counter()
    phi, _psi = system.build_fields()
    res = compute_moment([phi("a", labels[0]), phi("b", labels[1])],
                         system.build_action(), max_order)
    cv = system.build_coupling_values()
    out = {}
    for o in range(max_order + 1):
        dts = res.diagram_terms(o)
        total = 0.0
        if dts:
            total, _ = integrate_diagrams(
                dts, coupling_values=cv, lambda_f=T_F, t_min=T_MIN,
                cache=props.cache, method="gauss_legendre", n_gauss=6,
                fixed_indices={"a": 0, "b": 1}, positions=positions)
        out[o] = complex(total).real
    return out


@pytest.mark.parametrize("extra", [{"y_0": 0.65}, {"y_1": 0.1}, {"y_0": 3.0, "y_1": -2.0}])
def test_AN8_positions_key_naming_a_vertex_point_is_ignored_at_l1(
        system, props, extra):
    """L1 accepts positions for points the observable does not use, so one
    dict can serve several observables (test_yaml_spec_coverage.py::YC12);
    a key spelled like a vertex point is now one of those.  Up to 0.7.0
    the L0 resolver read it: {'y_0': 0.65} raised an error about a response
    chain joining 'x' and 'y_0', a point the user never declared."""
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2], progress=False)
    kw = dict(t_final=T_F, component_pair=(0, 1), orders=[0, 2],
              method="gauss_legendre", n_gauss=6)
    want = exp.evaluate(props, positions={"x": 0.1, "y": 0.65}, **kw)
    got = exp.evaluate(props, positions={"x": 0.1, "y": 0.65, **extra}, **kw)
    assert want.by_order[2] != 0.0
    assert got.by_order == want.by_order


def test_AN9_one_positions_dict_serves_observables_named_like_vertex_points(
        system, props):
    """With y_0 and y_1 legal observable labels, one dict can hold them as
    the points of one observable and still serve another whose vertex
    points have those names.  Up to 0.7.0 the (x, y) observable raised."""
    shared = {"x": 0.1, "y": 0.65, "y_0": 0.1, "y_1": 0.65}
    want = _l0_by_order(system, props, ("x", "y"), {"x": 0.1, "y": 0.65})
    assert want[2] != 0.0
    assert _l0_by_order(system, props, ("x", "y"), shared) == want
    assert _l0_by_order(system, props, ("y_0", "y_1"), shared) == pytest.approx(
        want, rel=1e-12, abs=1e-300)


def test_AN10_l0_vertex_point_key_does_not_place_its_group(system, props):
    """With no entry for 'x', its group sits at the default position 0.  A
    key naming a vertex point used to be read as its group's position:
    measured per diagram at order 2 through integrate_moment (F only, 6
    Gauss-Legendre nodes), positions {'y': 0.65, 'y_0': 0.4} moved one
    diagram from 0.0023243 to 0.0063181 with no error, and raised for the
    diagrams that put y_0 in the group of y."""
    want = _l0_by_order(system, props, ("x", "y"), {"y": 0.65})
    got = _l0_by_order(system, props, ("x", "y"), {"y": 0.65, "y_0": 0.4})
    assert want[2] != 0.0
    assert got == want
    reset_uid_counter()
    phi, _psi = system.build_fields()
    dts = compute_moment([phi("a", "x"), phi("b", "y")], system.build_action(),
                         2).diagram_terms(2)
    cv = system.build_coupling_values()
    common = dict(t_min=T_MIN, method="gauss_legendre", n_gauss=6)
    for dt in dts:
        ig = dt.build_integrand(cv, fixed_indices={"a": 0, "b": 1})
        plain = integrate_moment(ig, T_F, props.cache, positions={"y": 0.65},
                                 **common)
        extra = integrate_moment(ig, T_F, props.cache,
                                 positions={"y": 0.65, "y_0": 0.4}, **common)
        assert extra == plain


def test_AN11_l0_positions_key_naming_no_point_is_still_ignored(system, props):
    """A key that names no point of the diagram has no effect, as before."""
    want = _l0_by_order(system, props, ("x", "y"), {"x": 0.1, "y": 0.65})
    got = _l0_by_order(system, props, ("x", "y"), {"x": 0.1, "y": 0.65, "w": 3.0})
    assert want[2] != 0.0
    assert got == want
