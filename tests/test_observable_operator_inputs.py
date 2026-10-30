r"""The two spellings of an L1 observable, and the observables L1 refuses.

``System.expand`` takes the observable as strings (``"phi_a(x)"``) or as
``FieldOperator`` objects from :meth:`System.build_fields`.  The
``FieldOperator`` branch read an attribute the class does not have
(``component_indices``), so every such call raised ``AttributeError``
although the docstring advertised the form (OO1).  The component tuple of
``Expansion.evaluate`` is positional, so a set observable, whose order
follows ``PYTHONHASHSEED``, is refused as the YAML loader refuses it; on the
model below the set ``{"phi_a(x)", "phi_b(y)"}`` gave -0.0621 or -0.1454
at ``component_pair=(0, 1)`` depending on the seed, with no error (OO2).
A dict's keys view iterates in insertion order and is accepted; the keys
view of another mapping, a ``ChainMap`` included, is refused, since its
order cannot be checked.  A physical field named ``psi`` shared its name with the response field, so a
``phi`` and a ``psi`` operator of one label shared a cache key (OO3).

The model is generic: two components with distinct rates, a local cubic
``F`` and a constant non-local ``κ³`` vertex ``K`` with entries of both
signs and no index symmetry, ``t_min = 0.3``, distinct points and distinct
external times.  References share no code with the package:

* order 0, ``C_aa(x, t1; y, t2) = λ e^{-(x-y)²/2σ_x²} ∫∫ e^{-γ_a(t1-s1)}
  e^{-γ_a(t2-s2)} e^{-|s1-s2|/σ_t} ds1 ds2`` with the inner integral in
  closed form and the outer one by scipy ``quad`` (:func:`_c_reference`);
* the FK channel at order 2 for a constant ``κ³``, the one-dimensional
  integral of ``tests/test_sweep_selectors.py`` (:func:`_fk_reference`).
"""
from __future__ import annotations

import collections.abc
import os
import types
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

import sft_wick as sw
from sft_wick import Field
from sft_wick.workflow import specs as sp

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

T_MIN = 0.3
T_FINAL = 1.7
GAMMAS = (0.8, 1.3)
LAM, SIGMA_T, SIGMA_X = 0.4, 0.7, 1.3
X, Y = 0.1, 0.55
TIMES = {"x": 1.1, "y": 1.6}
F_TENSOR = np.array([[[0.2, -0.7], [-0.7, 1.1]], [[-0.4, 0.5], [0.5, 0.9]]])
K_TENSOR = np.array([[[0.3, -0.2], [-0.2, 0.6]], [[-0.2, 0.6], [0.6, -0.5]]])
GL = dict(method="gauss_legendre", n_gauss=6)
TUPLES = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _c_reference(a: int, b: int, t1: float, t2: float, x: float = X,
                 y: float = Y) -> float:
    """``C_ab(x, t1; y, t2)`` of the model: zero off the diagonal."""
    if a != b:
        return 0.0
    g, r = GAMMAS[a], 1.0 / SIGMA_T

    def inner(s1):
        # ∫_{t_min}^{t2} e^{-g (t2 - s2)} e^{-r |s1 - s2|} ds2, closed form.
        lo = min(s1, t2)
        below = np.exp(-g * t2 - r * s1) * (
            np.exp((g + r) * lo) - np.exp((g + r) * T_MIN)) / (g + r)
        above = 0.0
        if s1 < t2:
            start = max(s1, T_MIN)
            above = np.exp(-g * t2 + r * s1) * (
                np.exp((g - r) * t2) - np.exp((g - r) * start)) / (g - r)
        return below + above

    outer = quad(lambda s1: np.exp(-g * (t1 - s1)) * inner(s1), T_MIN, t1,
                 points=[t2] if T_MIN < t2 < t1 else None,
                 epsabs=0.0, epsrel=1e-13, limit=200)[0]
    return LAM * np.exp(-(x - y) ** 2 / (2 * SIGMA_X ** 2)) * outer


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


@pytest.fixture(scope="module")
def system():
    return sw.System(
        field=sp.FieldSpec("phi", n_components=2),
        linear=sp.DiagonalA(gamma=list(GAMMAS)),
        noise=sp.GaussianNoise(kappa2=sp.SeparableTranslation(
            temporal=sp.ExponentialTemporal(lam=LAM, sigma_t=SIGMA_T),
            spatial=sp.GaussianSpatial(sigma_x=SIGMA_X))),
        vertices=(sp.LocalVertex("F", F_TENSOR),),
        nonlocal_vertices=(sp.NonLocalVertex("K", order=3, coupling=K_TENSOR),),
        t_min=T_MIN,
    )


@pytest.fixture(scope="module")
def props(system):
    # The built-in closed form at every lookup, not a table of it.
    return system.propagators(t_max=2.0, n_grid_t=12, c_closed_form_only=True,
                              progress=False)


def _evaluate(expansion, props, pair, **kw):
    return expansion.evaluate(props, positions={"x": X, "y": Y},
                              t_final=T_FINAL, component_pair=pair, **GL, **kw)


# ---------------------------------------------------------------------------
# OO1: FieldOperator observables at L1
# ---------------------------------------------------------------------------

def test_OO1_field_operators_match_the_exact_covariance(system, props):
    """Order 0 at distinct times: the component tuple pins each letter, so
    (0, 0) and (1, 1) give the two distinct rates' covariances.  Every
    call raised AttributeError before."""
    phi, _psi = system.build_fields()
    exp = system.expand([phi("a", "x"), phi("b", "y")], orders=[0, 2],
                        progress=False)
    for a, b in TUPLES:
        got = _evaluate(exp, props, (a, b), orders=0, external_times=TIMES)
        want = _c_reference(a, b, TIMES["x"], TIMES["y"])
        assert got.total == pytest.approx(want, rel=1e-12, abs=1e-15), (a, b)
    assert _c_reference(0, 0, *TIMES.values()) != pytest.approx(
        _c_reference(1, 1, *TIMES.values()), rel=0.1, abs=0.0)


def test_OO1_field_operators_give_the_fk_channel(system, props):
    phi, _psi = system.build_fields()
    exp = system.expand((phi("a", "x"), phi("b", "y")), orders=2,
                        progress=False)
    for a, b in TUPLES:
        got = _evaluate(exp, props, (a, b), vertex_types="FK").total
        assert got == pytest.approx(_fk_reference(a, b), rel=1e-9, abs=0.0)


def test_OO1_both_spellings_give_one_expansion(system, props):
    """The two spellings share a cache key and give the same diagrams, so
    every channel at every order agrees bit for bit."""
    phi, _psi = system.build_fields()
    ops = system.expand([phi("a", "x"), phi("b", "y")], orders=[0, 2],
                        progress=False)
    strings = system.expand(["phi_a(x)", "phi_b(y)"], orders=[0, 2],
                            progress=False)
    assert ops.observable_repr == strings.observable_repr == (
        ("phi", "a", "x"), ("phi", "b", "y"))
    for pair in TUPLES:
        r_ops = _evaluate(ops, props, pair, external_times=TIMES)
        r_str = _evaluate(strings, props, pair, external_times=TIMES)
        assert r_ops.by_order == r_str.by_order
        assert r_ops.per_diagram == r_str.per_diagram


def test_OO1_a_shared_letter_takes_one_index(system, props):
    phi, _psi = system.build_fields()
    exp = system.expand([phi("a", "x"), phi("a", "y")], orders=0,
                        progress=False)
    with pytest.raises(ValueError, match="share the component letter 'a'"):
        _evaluate(exp, props, (0, 1))
    got = _evaluate(exp, props, (1, 1), external_times=TIMES).total
    assert got == pytest.approx(_c_reference(1, 1, *TIMES.values()),
                                rel=1e-12, abs=0.0)


@pytest.mark.parametrize("make, match", [
    (lambda phi, psi: [phi("1", "x"), phi("b", "y")], "numeric literal"),
    (lambda phi, psi: [phi("a", "x"), psi("0", "y")], "numeric literal"),
    (lambda phi, psi: [Field("phi", "physical", 3)("a", "x"), phi("b", "y")],
     "neither field of this system"),
    (lambda phi, psi: [Field("chi", "physical", 2)("a", "x"), phi("b", "y")],
     "neither field of this system"),
], ids=["literal_phi", "literal_psi", "other_n", "other_name"])
def test_OO1_operators_the_string_form_cannot_spell_are_refused(system, make,
                                                                match):
    """A literal label would fix the component the tuple chooses; an
    operator of another field is not one of this system's fields."""
    phi, psi = system.build_fields()
    with pytest.raises(ValueError, match=match):
        system.expand(make(phi, psi), orders=[0, 2], progress=False)


# ---------------------------------------------------------------------------
# OO2: a set has no order
# ---------------------------------------------------------------------------

class _SetBackedMapping(collections.abc.Mapping):
    """A mapping whose keys come in hash order: its ``keys()`` is a
    ``KeysView`` but not a dict's."""

    def __init__(self, keys):
        self._keys = set(keys)

    def __getitem__(self, key):
        if key in self._keys:
            return 0
        raise KeyError(key)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)


@pytest.mark.parametrize("make", [
    lambda: {"phi_a(x)", "phi_b(y)"},
    lambda: frozenset({"phi_a(x)", "phi_b(y)"}),
], ids=["set", "frozenset"])
def test_OO2_a_set_observable_is_refused(system, make):
    with pytest.raises(ValueError, match="set, which has no order"):
        system.expand(make(), orders=[0, 2], progress=False)


@pytest.mark.parametrize("make", [
    lambda: _SetBackedMapping(["phi_a(x)", "phi_b(y)"]).keys(),
    lambda: collections.ChainMap({"phi_a(x)": 0, "phi_b(y)": 1}).keys(),
], ids=["set_backed_keys", "chainmap_keys"])
def test_OO2_the_keys_view_of_another_mapping_is_refused(system, make):
    """The package cannot check the order of a mapping that is not a dict:
    the set-backed one iterates in hash order, the ChainMap in a fixed one,
    and both raise with the advice to pass a list."""
    with pytest.raises(ValueError, match="keys view of a mapping that is not "
                                         "a dict.*list\\(mapping.keys\\(\\)\\)"):
        system.expand(make(), orders=[0, 2], progress=False)


def test_OO2_a_set_of_field_operators_is_refused(system):
    phi, _psi = system.build_fields()
    with pytest.raises(ValueError, match="set, which has no order"):
        system.expand({phi("a", "x"), phi("b", "y")}, orders=0,
                      progress=False)


def test_OO2_ordered_spellings_are_accepted(system, props):
    """A list, a tuple, a dict's values and a dict's keys keep their order
    (a dict iterates in insertion order, so its keys view is a set with an
    order; a MappingProxyType over a dict gives the same view); the swapped
    list is the other observable, so (0, 1) moves to its mirror."""
    values = {"first": "phi_a(x)", "second": "phi_b(y)"}.values()
    keys = {"phi_a(x)": 0, "phi_b(y)": 1}.keys()
    proxy_keys = types.MappingProxyType({"phi_a(x)": 0, "phi_b(y)": 1}).keys()
    totals = [
        _evaluate(system.expand(obs, orders=[0, 2], progress=False), props,
                  (0, 1), external_times=TIMES).total
        for obs in (["phi_a(x)", "phi_b(y)"], ("phi_a(x)", "phi_b(y)"), values,
                    keys, proxy_keys)
    ]
    assert len(set(totals)) == 1
    swapped = _evaluate(
        system.expand(["phi_b(y)", "phi_a(x)"], orders=[0, 2], progress=False),
        props, (0, 1), external_times=TIMES).total
    assert swapped != pytest.approx(totals[0], rel=0.1, abs=0.0)


# ---------------------------------------------------------------------------
# OO3: the physical field may not take the response field's name
# ---------------------------------------------------------------------------

def test_OO3_a_physical_field_named_psi_is_refused():
    """``FieldSpec("psi")`` gave the physical and the response field one
    name, so ``phi`` and ``psi`` operators of the same label shared a cache
    key, and a cached ``C`` diagram was returned for the response
    observable, whose diagram is ``R``."""
    with pytest.raises(ValueError, match="response field"):
        sp.FieldSpec("psi", n_components=2)
    assert sp.FieldSpec("chi", n_components=2).response_name == "psi"


# ---------------------------------------------------------------------------
# OO4: a drawing reads no uid, and System.expand leaves the uids alone
# ---------------------------------------------------------------------------

G_TENSOR = np.arange(1.0, 17.0).reshape(2, 2, 2, 2)
LOCAL_F = (sp.LocalVertex("F", F_TENSOR),)
LOCAL_G = (sp.LocalVertex("G", G_TENSOR),)
NONLOCAL_K = (sp.NonLocalVertex("K", order=3, coupling=K_TENSOR),)
TWO_POINT = ("phi_a(x)", "phi_b(y)")


def _model(vertices=LOCAL_G, nonlocal_vertices=()):
    return sw.System(
        field=sp.FieldSpec("phi", n_components=2),
        linear=sp.DiagonalA(gamma=list(GAMMAS)),
        noise=sp.GaussianNoise(kappa2=sp.SeparableTranslation(
            temporal=sp.ExponentialTemporal(lam=LAM, sigma_t=SIGMA_T),
            spatial=sp.GaussianSpatial(sigma_x=SIGMA_X))),
        vertices=vertices,
        nonlocal_vertices=nonlocal_vertices,
        t_min=T_MIN,
    )


def _draw_each(expansion, order):
    """Every record of ``order``: a collision fails all those of one vertex."""
    records = expansion.raw_result.diagrams_by_order[order]
    assert records
    for record in records:
        record.to_feynman_diagram()


@pytest.mark.parametrize("vertices,nonlocal_vertices,labels", [
    (LOCAL_G, (), "ab"),
    (LOCAL_F, (), "abc"),
    ((), NONLOCAL_K, "abc"),
], ids=["quartic-two-point", "cubic-three-point", "kappa3-three-point"])
def test_OO4_field_operators_at_the_first_uids_draw(vertices, nonlocal_vertices,
                                                    labels):
    """The reset in ``System.expand`` put the vertices on the uids of the
    operators the caller built before it."""
    system = _model(vertices, nonlocal_vertices)
    sw.reset_uid_counter()
    phi, _ = system.build_fields()
    operators = [phi(a, x) for a, x in zip(labels, "xyz")]
    _draw_each(system.expand(operators, orders=[1], progress=False), 1)


def test_OO4_each_expansion_draws_after_the_one_before():
    """The observable kept the uids it took after the last expansion, and
    the reset then gave them to the vertices of the next order."""
    system = _model()
    sw.reset_uid_counter()
    for order in (1, 2):
        _draw_each(system.expand(TWO_POINT, orders=[order], progress=False),
                   order)


def test_OO4_a_cached_expansion_draws_by_either_spelling(tmp_path):
    system = _model()
    sw.reset_uid_counter()
    phi, _ = system.build_fields()
    operators = [phi("a", "x"), phi("b", "y")]
    system.expand(operators, orders=[1], cache_path=tmp_path, progress=False)
    _draw_each(system.expand(TWO_POINT, orders=[1], cache_path=tmp_path,
                             progress=False), 1)


def _cache_hit(system, cache_path):
    system.expand(TWO_POINT, orders=[1], cache_path=cache_path, progress=False)


def _refused_call(system, cache_path):
    with pytest.raises(ValueError):
        system.expand(("nope_a(x)",), orders=[1], cache_path=cache_path)


def test_OO4_no_call_rewinds_the_uids(tmp_path):
    """The reset gave operators the caller still held to the next vertices.
    The first uids are taken here so that the test sees a reset run alone:
    a reset and one expansion stop short of them."""
    system = _model()
    phi, _ = system.build_fields()
    last = max(phi("a", "x").uid for _ in range(32))
    for call in (_cache_hit, _cache_hit, _refused_call):  # miss, hit, refused
        call(system, tmp_path)
        uid = phi("a", "x").uid
        assert uid > last
        last = uid


def _forms(operators):
    sw.reset_uid_counter()
    action = _model().build_action()
    records = sw.compute_moment(operators, action, order=1).diagrams_by_order[1]
    return Counter(r.to_feynman_diagram().canonical_form() for r in records)


@pytest.mark.parametrize("uids", [(0, 1), (2, 3), (9, 9), (0, 0)],
                         ids=["vertices", "shifted", "each-other", "both"])
def test_OO4_a_drawing_reads_no_uid(uids):
    """A uid need not be unique: operators pickled in another process or
    read from a cache can hold the uids of the vertices, or share one."""
    phi, _ = _model().build_fields()
    operators = (phi("a", "x"), phi("b", "y"))
    shifted = tuple(replace(op, uid=uid) for op, uid in zip(operators, uids))
    assert _forms(shifted) == _forms(operators)


def test_OO4_one_operator_twice_draws_two_points():
    phi, _ = _model().build_fields()
    operator = phi("a", "x")
    result = sw.compute_moment((operator, operator), _model().build_action(),
                               order=0)
    diagram = result.diagrams_by_order[0][0].to_feynman_diagram()
    assert [diagram.graph.degree(n) for n in diagram.external_nodes] == [1, 1]


def _edges_of_pairing(record, diagram):
    """The edges of ``record`` from its operators, with no ``from_pairing``.
    An external is found by its point and a vertex by its copy, so a slip in
    the map from positions to nodes shows whatever the order of the nodes."""
    graph = diagram.graph
    externals = {graph.nodes[n]["spatial"]: n for n in diagram.external_nodes}
    vertices = {graph.nodes[n]["copy_id"]: n for n in diagram.vertex_nodes}
    operators = [(op, externals[op.spatial_arg]) for op in record.observable_ops]
    for vertex in record.vertex_instances:
        operators += [(op, vertices[vertex.copy_id])
                      for op in vertex.field_operators]
    edges = Counter()
    for i, j in record.pairing:
        (left, a), (right, b) = operators[i], operators[j]
        if left.is_physical and right.is_physical:
            edges["C", frozenset((a, b)), None, None] += 1
        else:
            phi_end, psi_end = (a, b) if left.is_physical else (b, a)
            edges["R", frozenset((a, b)), phi_end, psi_end] += 1
    return edges


def _two_phi(phi, psi):
    return phi("a", "x"), phi("b", "y")


def _phi_psi(phi, psi):
    return phi("a", "x"), psi("b", "y")


def _psi_phi(phi, psi):
    return psi("a", "x"), phi("b", "y")


def _three_phi(phi, psi):
    return phi("a", "x"), phi("b", "y"), phi("c", "z")


@pytest.mark.parametrize("vertices,nonlocal_vertices,observable,order", [
    (LOCAL_G, (), _two_phi, 1),
    (LOCAL_G, (), _two_phi, 2),
    (LOCAL_F, (), _two_phi, 2),
    (LOCAL_F, NONLOCAL_K, _two_phi, 2),
    (LOCAL_G, (), _phi_psi, 1),
    (LOCAL_F, (), _psi_phi, 2),
    (LOCAL_F + LOCAL_G, (), _three_phi, 2),
], ids=["quartic-first-order", "quartic-second-order", "cubic",
        "cubic-and-kappa3", "response-external-last", "response-external-first",
        "vertices-of-two-leg-counts"])
def test_OO4_each_edge_joins_the_operators_of_its_pair(vertices,
                                                       nonlocal_vertices,
                                                       observable, order):
    """The arrow of an R and the point of each external, which the two sides
    of ``_forms`` share and the records of this model hide."""
    sw.reset_uid_counter()
    system = _model(vertices, nonlocal_vertices)
    operators = observable(*system.build_fields())
    records = sw.compute_moment(operators, system.build_action(),
                                order=order).diagrams_by_order[order]
    assert records
    for record in records:
        diagram = record.to_feynman_diagram()
        drawn = Counter((data["kind"], frozenset((u, v)), data.get("phi_end"),
                         data.get("psi_end"))
                        for u, v, data in diagram.graph.edges(data=True))
        assert drawn == _edges_of_pairing(record, diagram)
