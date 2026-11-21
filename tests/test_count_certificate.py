"""Count certificates of ``compute_moment``, ``compute_moment_numerical``,
``System.expand`` and the routing count, with ``certify=True``.

A certificate holds one row per vertex sequence of every order: the number
of valid operator pairings from the counter of :mod:`sft_wick._certify`,
which imports nothing from the package, the pairings the engine output
stands for (summed term weight, ``DiagramInfo`` records, routings), the
number of R classes the engine enumerates, and the labelled topologies,
typed classes, ``|G|`` and orbit sum of its canonical grouping.

The reference rows are computed here from the description of each model:
``tests/brute_wick.py`` enumerates every pairing of up to 14 operators with
its own rules, a union-find on owners in this file gives the connected
target, and ``tests/brute_relabel.py`` tries every relabelling of the
integrated points.  None of them imports the package.  Rows of 18 operators
are compared with the counter pins of ``tests/test_pairing_counter.py``.

The tests that replace an engine function (R classes of one label, the
shared-label guard, the routing of psi) check that the certificate raises
on that fault, with the numbers of the failing rows.  The other checks of
the certificate are tested on its own data.
"""
from __future__ import annotations

import dataclasses
import functools
import gc
import pickle
import warnings
from collections import Counter
from itertools import combinations_with_replacement, groupby, permutations, product
from math import factorial, prod
from typing import NamedTuple

import numpy as np
import pytest

import sft_wick as sw
from sft_wick import (
    Action,
    CertificateError,
    CountCertificate,
    Field,
    Vertex,
    compute_moment,
    compute_moment_numerical,
    reset_uid_counter,
)
from sft_wick import _certificate, _certify, perturbation, wick
from sft_wick._routing_count import routing_count_certificate
from sft_wick.expressions import DiracDelta, Product, Propagator, Rational
from sft_wick.workflow import system as system_module

from . import brute_relabel, brute_wick

PHI = Field("phi", "physical")
PSI = Field("psi", "response")

#: A vertex: (coupling name, legs, local); legs are "f" for phi, "s" for psi.
F = ("F", "sff", True)
G = ("G", "sff", True)
Q = ("Q", "sfff", True)
M = ("M", "ssff", True)
K = ("K", "sss", False)
K_EQ = ("Keq", "sss", False)
K_R = ("Kr", "sss", False)
E = ("E", "", True)
FLAGS = {"Keq": {"equal_time": True}, "Kr": {"already_R_contracted": True}}

XY = (("phi", "x"), ("phi", "y"))
XYZ = XY + (("phi", "z"),)

#: name -> (vertices of the action, observable, highest order).
MODELS = {
    "cubic": ((F,), XY, 4),
    "quartic": ((Q,), XY, 3),
    "double_psi": ((M,), XY, 3),
    "F_G": ((F, G), XY, 4),
    "F_M": ((F, M), XY, 3),
    "F_K_xyz": ((F, K), XYZ, 3),
    "F_Keq_xyz": ((F, K_EQ), XYZ, 3),
    "F_Kr_xyz": ((F, K_R), XYZ, 3),
    "F_K_xy": ((F, K), XY, 4),
    "F_E": ((F, E), XY, 3),
    "F_listed_twice": ((F, F), XY, 3),
}
OPERATOR_PATH_MODELS = ("quartic", "F_M", "F_K_xyz")

ENGINE_FIELDS = ("term_weight", "diagram_infos", "routings")
CLASS_FIELDS = ("topologies", "classes", "group_order", "orbit_sum")
#: The fields each path fills, at order 0 and above it.
PATH_FIELDS = {
    "numerical": ({"term_weight", "r_classes"},
                  {"term_weight", "r_classes", *CLASS_FIELDS}),
    "moment": ({"term_weight", "diagram_infos", "r_classes"},
               {"term_weight", "diagram_infos", "r_classes", *CLASS_FIELDS}),
    "operator": ({"term_weight", "diagram_infos", "r_classes"},
                 {"diagram_infos", "r_classes"}),
    "routings": ({"routings", "r_classes"},
                 {"routings", "r_classes", *CLASS_FIELDS}),
}
ENGINE_NAMES = {
    "numerical": "compute_moment_numerical",
    "moment": "compute_moment",
    "operator": "compute_moment/operator",
    "routings": "routings",
}


# --------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------- #


def make_vertex(description, phi=PHI, psi=PSI) -> Vertex:
    name, legs, local = description
    fields = [phi if leg == "f" else psi for leg in legs]
    return Vertex(fields, name, local=local, **FLAGS.get(name, {}))


def make_action(descriptions) -> Action:
    return Action([make_vertex(d) for d in descriptions])


def make_observable(spec) -> list:
    return [(PHI if kind == "phi" else PSI)(label) for kind, label in spec]


def certificate_of(engine, observable, action, order, target="moment", **kwargs):
    """The certificate of one certified call; ``engine`` is a key of
    :data:`ENGINE_NAMES`."""
    reset_uid_counter()
    if engine == "numerical":
        kwargs.setdefault("canonical", "builtin")
        out = compute_moment_numerical(observable, action, order, target=target,
                                       certify=True, **kwargs)
        return out.certificate
    if engine == "moment":
        return compute_moment(observable, action, order, target=target,
                              certify=True, **kwargs).certificate
    if engine == "operator":
        return compute_moment(observable, action, order, target=target,
                              collect_topology=False, certify=True,
                              **kwargs).certificate
    assert engine == "routings"
    return routing_count_certificate(observable, action, order, target=target)


# --------------------------------------------------------------------- #
# Reference rows
# --------------------------------------------------------------------- #


class Tally(NamedTuple):
    pairings: int
    topologies: int
    classes: int
    orbit_sum: int
    group_order: int


class BruteRow(NamedTuple):
    vertices: tuple[str, ...]
    multinomial: int
    moment: Tally
    connected: Tally
    r_classes: int


def _one_component(pairing, owners, n_owners) -> bool:
    parent = list(range(n_owners))

    def find(node):
        while parent[node] != node:
            node = parent[node]
        return node

    for pair in pairing:
        a, b = (find(owners[uid]) for uid in pair)
        if a != b:
            parent[a] = b
    return len({find(node) for node in range(n_owners)}) == 1


@functools.lru_cache(maxsize=None)
def brute_sequence(externals, sequence) -> tuple[Tally, Tally, int]:
    """Moment and connected tallies and the R-class count of one sequence
    of vertex descriptions.  Labels: ``v{i}`` for local vertex ``i`` and
    ``v{i}.{l}`` for leg ``l`` of non-local vertex ``i``."""
    ops, owners = [], []
    for k, (kind, label) in enumerate(externals):
        ops.append(brute_wick.Op(kind, label, None, len(ops)))
        owners.append(k)
    local, instances, points = {}, [], []
    for i, (name, legs, is_local) in enumerate(sequence):
        owner = len(externals) + i
        labels = ([f"v{i}"] * len(legs) if is_local
                  else [f"v{i}.{leg}" for leg in range(len(legs))])
        if is_local:
            local[f"v{i}"] = name
            points.append(f"v{i}")
        else:
            instances.append((name, tuple(labels)))
            points.extend(labels)
        for leg, label in zip(legs, labels):
            kind = "phi" if leg == "f" else "psi"
            ops.append(brute_wick.Op(kind, label, None, len(ops)))
            owners.append(owner)
    n_owners = len(externals) + len(sequence)
    group_order = prod(factorial(n) for n in Counter(sequence).values())
    moment, connected, r_sets = Counter(), Counter(), set()
    if not brute_wick.vanishes_by_leg_count(ops):
        assert len(ops) <= 14 or len(ops) % 2
        for pairing, props in brute_wick.iter_kept_pairings(ops):
            sig = brute_wick.spatial_signature(props)
            moment[sig] += 1
            r_sets.add(tuple(sorted((p[1], p[2]) for p in props if p[0] == "R")))
            if _one_component(pairing, owners, n_owners):
                connected[sig] += 1
    group = brute_relabel.relabelling_group(points, local, instances) if moment else []
    if moment:
        assert len(group) == group_order

    def tally(counts) -> Tally:
        classes: dict = {}
        for sig in counts:
            cls, stabiliser = brute_relabel.orbit_data(sig, group)
            classes.setdefault(cls, []).append(stabiliser)
        return Tally(sum(counts.values()), len(counts), len(classes),
                     sum(group_order // s[0] for s in classes.values()),
                     group_order)

    return tally(moment), tally(connected), len(r_sets)


@functools.lru_cache(maxsize=None)
def brute_rows(externals, vertices, order) -> tuple[BruteRow, ...]:
    """One row per term of the multinomial expansion, in the order of the
    combinations with repetition of the vertex positions."""
    rows = []
    for combo in combinations_with_replacement(range(len(vertices)), order):
        counts = Counter(combo)
        multinomial = factorial(order) // prod(factorial(c) for c in counts.values())
        sequence = tuple(vertices[i] for i in combo)
        moment, connected, r_classes = brute_sequence(externals, sequence)
        rows.append(BruteRow(tuple(v[0] for v in sequence), multinomial,
                             moment, connected, r_classes))
    return tuple(rows)


def check_against_brute(cert, engine, vertices, externals, top, target):
    assert isinstance(cert, CountCertificate)
    assert cert.engine == ENGINE_NAMES[engine]
    assert cert.target == target
    assert [o.order for o in cert.orders] == list(range(top + 1))
    for n, order in enumerate(cert.orders):
        expected = brute_rows(externals, vertices, n)
        assert order.n_vertices == len(vertices)
        assert [(r.index, r.vertices, r.multinomial) for r in order.sequences] == [
            (k, b.vertices, b.multinomial) for k, b in enumerate(expected)]
        wanted = PATH_FIELDS[engine][n > 0]
        for row, ref in zip(order.sequences, expected):
            want = ref.moment if target == "moment" else ref.connected
            present = {name for name in (*ENGINE_FIELDS, "r_classes", *CLASS_FIELDS)
                       if getattr(row, name) is not None}
            assert present <= wanted, (row, wanted)
            assert wanted - {"r_classes"} <= present, (row, wanted)
            assert row.counter == want.pairings, row
            for name in ENGINE_FIELDS:
                if getattr(row, name) is not None:
                    assert getattr(row, name) == want.pairings, row
            if row.r_classes is None:
                assert target == "connected" and want.pairings == 0, row
            else:
                assert row.r_classes == ref.r_classes, row
            for name in CLASS_FIELDS:
                if getattr(row, name) is not None:
                    assert getattr(row, name) == getattr(want, name), row


# (model, order, vertices, target) -> (pairings, topologies, classes,
# orbit sum, R classes); None where the reference has no such number.
ANCHORS = {
    ("cubic", 4, ("F",) * 4, "moment"): (15_120, 1_416, 64, 1_416, 216),
    ("cubic", 4, ("F",) * 4, "connected"): (14_400, 1_272, 56, 1_272, 216),
    ("F_G", 4, ("F", "F", "G", "G"), "moment"): (15_120, 1_416, 364, 1_416, None),
    ("F_M", 3, ("F", "F", "M"), "moment"): (564, 66, 34, 66, 33),
    ("F_M", 3, ("F", "F", "M"), "connected"): (556, 62, 32, 62, 33),
}


def test_CC0_the_reference_rows_hold_the_known_numbers():
    for (model, order, names, target), numbers in ANCHORS.items():
        vertices, externals, _top = MODELS[model]
        (row,) = [r for r in brute_rows(externals, vertices, order) if r.vertices == names]
        tally = row.moment if target == "moment" else row.connected
        got = (tally.pairings, tally.topologies, tally.classes, tally.orbit_sum,
               row.r_classes)
        assert all(w is None or g == w for g, w in zip(got, numbers)), (got, numbers)
    twice = brute_rows(XY, (F, F), 2)
    assert [(r.vertices, r.multinomial, r.moment.pairings) for r in twice] == [
        (("F", "F"), 1, 30), (("F", "F"), 2, 30), (("F", "F"), 1, 30)]


# --------------------------------------------------------------------- #
# Positive tests
# --------------------------------------------------------------------- #

_CC1_CASES = [
    pytest.param(model, engine, target, id=f"{model}-{engine}-{target}")
    for model in MODELS
    for engine in ("numerical", "moment", "routings")
    + (("operator",) if model in OPERATOR_PATH_MODELS else ())
    for target in ("moment", "connected")
]


@pytest.mark.parametrize("model, engine, target", _CC1_CASES)
def test_CC1_certificate_rows_equal_the_brute_force_rows(model, engine, target):
    vertices, externals, top = MODELS[model]
    cert = certificate_of(engine, make_observable(externals), make_action(vertices),
                          top, target)
    check_against_brute(cert, engine, vertices, externals, top, target)
    assert cert.schema == _certificate.CERTIFICATE_SCHEMA
    assert cert.observable == tuple(repr(op) for op in make_observable(externals))
    assert (cert.encoding is None) == (engine == "operator")
    for n, order in enumerate(cert.orders):
        expected = brute_rows(externals, vertices, n)
        counts = [(b.moment if target == "moment" else b.connected).pairings
                  for b in expected]
        assert order.pairings == sum(counts)
        assert order.weighted_pairings == sum(
            b.multinomial * count for b, count in zip(expected, counts))
    for (name, order, names, anchor_target), numbers in ANCHORS.items():
        if name != model or anchor_target != target:
            continue
        (row,) = [r for r in cert.order(order).sequences if r.vertices == names]
        assert row.counter == numbers[0]
        if row.topologies is not None:
            assert (row.topologies, row.classes, row.orbit_sum) == numbers[1:4]


PHI2 = Field("phi", "physical", n_components=2)
PSI2 = Field("psi", "response", n_components=2)


def _two_component_case():
    action = Action([make_vertex(F, PHI2, PSI2), make_vertex(K, PHI2, PSI2)])
    return [PHI2("a", "x"), PHI2("b", "y"), PHI2("c", "z")], action


@pytest.mark.parametrize("engine", ["numerical", "moment"])
@pytest.mark.parametrize("flags", [
    {"diag_R": True, "diag_C": True},
    {"iso_R": True, "iso_C": True},
], ids=["diag", "iso"])
def test_CC2_term_weights_under_the_diagonal_flags(engine, flags):
    observable, action = _two_component_case()
    plain = certificate_of(engine, observable, action, 3)
    observable, action = _two_component_case()
    flagged = certificate_of(engine, observable, action, 3, **flags)
    assert flagged == plain
    assert sum(o.pairings for o in plain.orders) > 0
    check_against_brute(plain, engine, (F, K), XYZ, 3, "moment")


_CC3_CASES = [
    pytest.param("double_psi", engine, target, 4, id=f"double_psi-{engine}-{target}")
    for engine in ("numerical", "moment", "routings")
    for target in ("moment", "connected")
] + [
    pytest.param("quartic", "routings", "moment", 4, id="quartic-routings",
                 marks=pytest.mark.heavy),
    # Measured once on the development machine: 485 s and a peak physical
    # footprint of 2.27 GiB.  The topology dict of the one sequence holds
    # 738,900 entries of ten propagators, about 3,150 bytes each.
    pytest.param("cubic", "routings", "moment", 6, id="cubic6-routings",
                 marks=pytest.mark.slow),
]
#: (model, order) -> counter pin of tests/test_pairing_counter.py.
_CC3_PINS = {("double_psi", 4): 384, ("quartic", 4): 3_243_240,
             ("cubic", 6): 32_432_400}


@pytest.mark.parametrize("model, engine, target, order", _CC3_CASES)
def test_CC3_rows_beyond_the_brute_force_size(model, engine, target, order):
    vertices, externals, _top = MODELS[model]
    cert = certificate_of(engine, make_observable(externals), make_action(vertices),
                          order, target)
    (row,) = cert.order(order).sequences
    value = row.routings if engine == "routings" else row.term_weight
    assert value == row.counter
    if target == "moment":
        assert row.counter == _CC3_PINS[(model, order)]
    assert row.orbit_sum == row.topologies
    if model == "cubic":
        assert (row.topologies, row.classes) == (738_900, 1_088)


# --------------------------------------------------------------------- #
# System.expand
# --------------------------------------------------------------------- #

_N = 2
_RNG = np.random.default_rng(20261119)
_F_TENSOR = _RNG.normal(size=(_N,) * 3)
_K_TENSOR = _RNG.normal(size=(_N,) * 3)
_G0 = np.array([[0.55, -0.20, 0.10], [0.15, 0.40, -0.25]])
_G1 = np.array([[[0.20, -0.15], [-0.10, 0.25], [0.05, 0.10]],
                [[-0.12, 0.18], [0.22, -0.08], [-0.15, 0.06]]])


def _noise(**kwargs):
    return sw.GaussianNoise(
        kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
            spatial=sw.ExponentialSpatial(sigma_x=1.0)),
        **kwargs)


def _system_f_k():
    return sw.System(
        field=sw.FieldSpec("phi", n_components=_N),
        linear=sw.DiagonalA(gamma=[1.0, 1.3]),
        vertices=[sw.LocalVertex("F", coupling=_F_TENSOR)],
        nonlocal_vertices=[sw.NonLocalVertex("K", order=3, coupling=_K_TENSOR)],
        noise=_noise(),
    )


def _system_multiplicative():
    return sw.System(
        field=sw.FieldSpec("phi", n_components=_N),
        linear=sw.DiagonalA(gamma=[1.1, 1.3]),
        vertices=[sw.LocalVertex("F", coupling=_F_TENSOR)],
        noise=_noise(sigma2=sw.MultiplicativeImpulse(
            g0=_G0, g1=_G1, interpretation="stratonovich")),
    )


_CC4_CASES = {
    "F_K": (_system_f_k, ("phi_a(x)", "phi_b(y)", "phi_c(z)"), XYZ,
            (F, K), [0, 1, 2, 3], {}),
    "multiplicative": (_system_multiplicative, ("phi_a(x)", "phi_b(y)"), XY,
                       (F, ("G", "ssf", True), ("H", "ssff", True),
                        ("B", "s", True), ("L", "sf", True)),
                       [0, 1, 2], {"diag_C": False}),
}


def _description(vertex: Vertex):
    legs = "".join("f" if f.is_physical else "s" for f in vertex.fields)
    return (vertex.coupling, legs, vertex.local)


@pytest.mark.parametrize("target", ["moment", "connected"])
@pytest.mark.parametrize("case", list(_CC4_CASES))
def test_CC4_system_expand_certifies(case, target):
    build, observable, externals, vertices, orders, kwargs = _CC4_CASES[case]
    system = build()
    assert tuple(_description(v) for v in system.build_action().vertices) == vertices
    reset_uid_counter()
    expansion = system.expand(observable, orders=orders, progress=False,
                              target=target, certify=True, **kwargs)
    cert = expansion.certificate
    assert cert is expansion.raw_result.certificate
    check_against_brute(cert, "moment", vertices, externals, max(orders), target)


def test_CC5_certify_recomputes_instead_of_reading_the_cache(tmp_path, monkeypatch):
    system = _system_f_k()
    calls = []
    original = perturbation.compute_moment

    def counting(*args, **kwargs):
        calls.append(kwargs.get("certify", False))
        return original(*args, **kwargs)

    monkeypatch.setattr(perturbation, "compute_moment", counting)
    observable = ("phi_a(x)", "phi_b(y)")
    first = system.expand(observable, orders=[0, 2], cache_path=tmp_path,
                          progress=False)
    assert first.certificate is None and calls == [False]
    second = system.expand(observable, orders=[0, 2], cache_path=tmp_path,
                           progress=False, certify=True)
    assert calls == [False, True]
    assert isinstance(second.certificate, CountCertificate)
    third = system.expand(observable, orders=[0, 2], cache_path=tmp_path,
                          progress=False)
    assert calls == [False, True]
    assert third.certificate == second.certificate


@pytest.mark.parametrize("model, order", [("F_G", 4), ("F_K_xyz", 3)])
def test_CC6_pynauty_keys_give_the_same_certificate(pynauty_module, model, order):
    vertices, externals, _top = MODELS[model]
    certs = {
        method: certificate_of("numerical", make_observable(externals),
                               make_action(vertices), order, canonical=method)
        for method in ("builtin", "pynauty")
    }
    assert certs["builtin"] == certs["pynauty"]
    assert certs["builtin"].encoding == certs["pynauty"].encoding is not None
    assert certs["builtin"].canonical != certs["pynauty"].canonical
    assert certs["builtin"].canonical == "builtin/1"


def test_CC7_workers_give_the_serial_certificate():
    vertices, externals, _top = MODELS["cubic"]
    serial = certificate_of("numerical", make_observable(externals),
                            make_action(vertices), 4)
    parallel = certificate_of("numerical", make_observable(externals),
                              make_action(vertices), 4, n_jobs=2)
    assert parallel == serial
    assert serial.order(4).sequences[0].topologies == 1_416


@pytest.mark.parametrize("engine", ["numerical", "moment", "routings"])
def test_CC8_crossing_labels_at_order_zero_are_compared(engine):
    observable = make_observable((("phi", "x"), ("psi", "x"), ("phi", "y"),
                                  ("psi", "y")))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cert = certificate_of(engine, observable, make_action((F,)), 0, "connected")
    (row,) = cert.order(0).sequences
    assert row.counter == 0
    for name in ENGINE_FIELDS:
        assert getattr(row, name) in (None, 0)
    assert (row.routings if engine == "routings" else row.term_weight) == 0


@pytest.mark.parametrize("engine", ["numerical", "moment", "routings"])
def test_CC23_order_zero_r_classes_with_mixed_label_types(engine):
    observable = [PSI("w"), PSI("z"), PHI(1), PHI("a")]
    cert = certificate_of(engine, observable, Action([]), 0)
    (row,) = cert.order(0).sequences
    assert row.counter == row.r_classes == 2


def test_CC13_certify_is_off_by_default_and_validated():
    observable = make_observable(XY)
    action = make_action((F,))
    reset_uid_counter()
    plain = compute_moment(observable, action, 2)
    assert plain.certificate is None
    assert "certificate" not in vars(plain)
    assert type(compute_moment_numerical(observable, action, 2,
                                         canonical="builtin")) is dict
    system = _system_f_k()
    assert system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2],
                         progress=False).certificate is None
    for call in (
        lambda: compute_moment(observable, action, 2, certify="yes"),
        lambda: compute_moment_numerical(observable, action, 2, certify="yes"),
        lambda: system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2],
                              progress=False, certify="yes"),
    ):
        with pytest.raises(TypeError, match="certify"):
            call()
    for call in (
        lambda: compute_moment([], action, 1, certify=True),
        lambda: compute_moment_numerical([], action, 1, certify=True),
        lambda: routing_count_certificate([], action, 1),
    ):
        with pytest.raises(ValueError, match="at least one observable operator"):
            call()
    assert compute_moment([], Action([]), 0).certificate is None

    reset_uid_counter()
    out = compute_moment_numerical(observable, action, 2, certify=True,
                                   canonical="builtin")
    assert isinstance(out, dict) and type(out) is not dict
    back = pickle.loads(pickle.dumps(out))
    assert type(back) is type(out)
    assert back.certificate == out.certificate
    assert list(back) == list(out) and [len(v) for v in back.values()] == [
        len(v) for v in out.values()]
    assert type(dict(out)) is dict

    result = compute_moment(observable, action, 1)
    result.__dict__.pop("certificate", None)
    restored = pickle.loads(pickle.dumps(result))
    assert "certificate" not in restored.__dict__
    assert restored.certificate is None


# --------------------------------------------------------------------- #
# Engine faults
# --------------------------------------------------------------------- #


def _failing_rows(cert):
    """``{(order, vertices): (counter, engine values)}`` of the rows whose
    engine fields differ from the counter."""
    out = {}
    for order in cert.orders:
        for row in order.sequences:
            values = tuple(getattr(row, n) for n in ENGINE_FIELDS
                           if getattr(row, n) is not None)
            if any(v != row.counter for v in values):
                out[(order.order, row.vertices)] = (row.counter, values)
    return out


def _r_class_rows(cert):
    return {(o.order, r.vertices): r.r_classes for o in cert.orders
            for r in o.sequences if r.r_classes is not None}


def _reordered_r_classes(original):
    """Each class of ``original`` also with every distinct reordering of the
    phi labels taken by the psi of one label."""
    def r_classes(phi_labels, psi_labels):
        for cls in original(phi_labels, psi_labels):
            runs = [list(g) for _, g in groupby(cls, key=lambda edge: edge[1])]
            orders = [sorted(set(permutations([edge[0] for edge in run])))
                      for run in runs]
            for combo in product(*orders):
                yield tuple((phi, run[0][1]) for run, phis in zip(runs, combo)
                            for phi in phis)
    return r_classes


_CC9_CASES = {
    "operator_path": ("operator", XY, (M,), 1,
                      {(1, ("M",)): (2, (4,))}, {(1, ("M",)): 2}),
    "order_zero": ("numerical", (("psi", "x"), ("psi", "x"), ("phi", "y"),
                                 ("phi", "z")), (M,), 0,
                   {(0, ()): (2, (4,))}, {(0, ()): 2}),
    "spatial_path": ("numerical", XY, (M,), 2, {},
                     {(1, ("M",)): 2, (2, ("M", "M")): 4}),
}


@pytest.mark.parametrize("case", list(_CC9_CASES))
def test_CC9_r_classes_repeated_at_one_label_raise(case, monkeypatch):
    engine, externals, vertices, order, counts, r_rows = _CC9_CASES[case]
    certificate_of(engine, make_observable(externals), make_action(vertices), order)
    monkeypatch.setattr(wick, "r_classes", _reordered_r_classes(wick.r_classes))
    with pytest.raises(CertificateError) as excinfo:
        certificate_of(engine, make_observable(externals), make_action(vertices),
                       order)
    cert = excinfo.value.certificate
    assert _failing_rows(cert) == counts
    assert {k: v for k, v in _r_class_rows(cert).items() if k in r_rows} == r_rows
    failures = excinfo.value.failures
    assert len(failures) == len(counts) + len(r_rows)
    assert sum("R classes" in f for f in failures) == len(r_rows)


def _disable_guard(monkeypatch):
    def no_guard(observable, order):
        return None
    monkeypatch.setattr(perturbation, "check_distinct_external_labels", no_guard)
    monkeypatch.setattr(system_module, "check_distinct_external_labels", no_guard)


@pytest.mark.parametrize("target, counter, engine_count",
                         [("moment", 30, 15), ("connected", 28, 14)])
@pytest.mark.parametrize("engine", ["numerical", "moment", "system"])
def test_CC10_shared_external_labels_raise(engine, target, counter, engine_count,
                                           monkeypatch):
    def run():
        if engine == "system":
            reset_uid_counter()
            return _system_f().expand(
                ("phi_a(x)", "phi_b(x)"), orders=[0, 2], progress=False,
                target=target, certify=True).certificate
        observable = make_observable((("phi", "x"), ("phi", "x")))
        return certificate_of(engine, observable, make_action((F,)), 2, target)

    with pytest.raises(ValueError, match="share the same spatial label"):
        run()
    _disable_guard(monkeypatch)
    with pytest.raises(CertificateError) as excinfo:
        run()
    rows = _failing_rows(excinfo.value.certificate)
    n_fields = 1 if engine == "numerical" else 2
    assert rows == {(2, ("F", "F")): (counter, (engine_count,) * n_fields)}
    assert len(excinfo.value.failures) == n_fields


def _system_f():
    return sw.System(
        field=sw.FieldSpec("phi", n_components=_N),
        linear=sw.DiagonalA(gamma=[1.0, 1.3]),
        vertices=[sw.LocalVertex("F", coupling=_F_TENSOR)],
        noise=_noise(),
    )


def _psi_partners(props, pairing, operators):
    return {psi: operators[phi].spatial_arg
            for (phi, psi), prop in zip(pairing, props) if prop.kind == "R"}


def _psi_held(original):
    """Routings in which every psi keeps the phi label of its partner in
    the representative pairing."""
    def routings(ref_props, rep_pairing, operators, points):
        want = _psi_partners(ref_props, rep_pairing, operators)
        return [(props, pairing)
                for props, pairing in original(ref_props, rep_pairing, operators, points)
                if _psi_partners(props, pairing, operators) == want]
    return routings


@pytest.mark.parametrize("target", ["moment", "connected"])
@pytest.mark.parametrize("engine, n_fields", [("moment", 2), ("numerical", 1),
                                              ("routings", 1)])
def test_CC11_psi_held_on_representative_slots_raises(engine, n_fields, target,
                                                      monkeypatch):
    vertices = (F, M)
    certificate_of(engine, make_observable(XY), make_action(vertices), 3, target)
    monkeypatch.setattr(perturbation, "_enumerate_component_routings",
                        _psi_held(perturbation._enumerate_component_routings))
    with pytest.raises(CertificateError) as excinfo:
        certificate_of(engine, make_observable(XY), make_action(vertices), 3, target)
    ffm = (564, 342) if target == "moment" else (556, 334)
    expected = {
        (1, ("M",)): (2, 1),
        (2, ("M", "M")): (8, 4),
        (3, ("F", "F", "M")): ffm,
        (3, ("M", "M", "M")): (48, 24),
    }
    assert _failing_rows(excinfo.value.certificate) == {
        key: (counter, (value,) * n_fields) for key, (counter, value) in expected.items()}
    assert len(excinfo.value.failures) == 4 * n_fields
    unchanged = excinfo.value.certificate.order(2).sequences[0]
    assert unchanged.vertices == ("F", "F")
    assert unchanged.counter == (30 if target == "moment" else 28)


# --------------------------------------------------------------------- #
# Checks on the certificate's own data
# --------------------------------------------------------------------- #


def test_CC12_counter_inputs_come_from_the_templates():
    observable = [PHI("x"), PHI("y_0")]
    vertices = tuple(make_vertex(d) for d in (F, F, K))
    inputs = _certificate.counter_inputs(observable, vertices)
    assert set(inputs) == {"points", "phi_counts", "psi_owner", "owner_of_group"}
    assert all(type(value) is list for value in inputs.values())
    assert inputs["points"][:2] == ["x", "y_0"]
    assert not {"x", "y_0"} & set(inputs["points"][2:])
    assert len(set(inputs["points"])) == len(inputs["points"])
    owners = inputs["owner_of_group"]
    assert owners.count(4) == 3
    k_groups = [g for g, owner in enumerate(owners) if owner == 4]
    assert [inputs["phi_counts"][g] for g in k_groups] == [0, 0, 0]
    assert sorted(inputs["psi_owner"]) == sorted([2, 3] + k_groups)
    pair = _certificate.counter_inputs(observable, vertices[:2])
    assert _certify.count_valid_pairings(**pair) == 30


class _Layout:
    """A layout whose automorphism order is fixed, or raises."""

    def __init__(self, order):
        self.order = order

    def point_automorphism_order(self, props):
        if isinstance(self.order, Exception):
            raise self.order
        return self.order


def _matched_tally(run, n):
    tally = run.begin_order(n)
    for index, counter in enumerate(tally.counters):
        tally.add_routings(index, counter)
    return tally


def _run(observable, vertices, engine="routings", target="moment"):
    return _certificate.CertificateRun(engine, observable, make_action(vertices),
                                       target)


def _cubic_order_two_terms():
    reset_uid_counter()
    return compute_moment_numerical(make_observable(XY), make_action((F,)), 2,
                                    canonical="builtin")[2]


def _kind_count(observable):
    run = _run(observable, (F,))
    tally = run.begin_order(2)
    tally.add_routings(0, 15)
    return run, tally


def _kind_half_prefactor(observable):
    run = _run(observable, (F,), "compute_moment_numerical")
    tally = run.begin_order(2)
    terms = _cubic_order_two_terms()
    prefactor = terms[0].rational_prefactor
    terms[0] = dataclasses.replace(
        terms[0], rational_prefactor=Rational(prefactor.numerator,
                                                 2 * prefactor.denominator))
    tally.add_terms(0, terms, prefactor)
    return run, tally


def _kind_negative_weight(observable):
    run = _run(observable, (F,), "compute_moment_numerical")
    tally = run.begin_order(2)
    terms = _cubic_order_two_terms()
    prefactor = terms[0].rational_prefactor
    negative = dataclasses.replace(
        terms[0], rational_prefactor=Rational(-prefactor.numerator, prefactor.denominator))
    tally.add_terms(0, [negative], prefactor)
    return run, tally


def _kind_unreadable(observable):
    run = _run(observable, (F,), "compute_moment_numerical")
    tally = run.begin_order(2)
    terms = _cubic_order_two_terms()
    terms[0] = dataclasses.replace(
        terms[0], coupling_sum=Product((DiracDelta("x", "y"), terms[0].coupling_sum)))
    tally.add_terms(0, terms, terms[1].rational_prefactor)
    return run, tally


def _kind_class_size(observable):
    run = _run(observable, (F, G))
    tally = _matched_tally(run, 4)
    index = [k for k, (seq, _m) in enumerate(tally.combinations)
             if [v.coupling for v in seq] == ["F", "F", "G", "G"]][0]
    for _ in range(3):
        tally.add_topology(index, _Layout(2), b"one class", [])
    return run, tally


def _kind_not_a_divisor(observable):
    run = _run(observable, (F,))
    tally = _matched_tally(run, 2)
    tally.add_topology(0, _Layout(RuntimeError("does not divide")), b"key", [])
    return run, tally


def _kind_repeated_r_class(observable):
    run = _run(observable, (F,))
    tally = _matched_tally(run, 2)
    tally.add_r_classes(0, 4, 2)
    return run, tally


def _kind_refused_layout(_observable):
    observable = make_observable((("phi", "x"), ("psi", "x"), ("phi", "y"),
                                  ("psi", "y")))
    run = _run(observable, (M,), target="connected")
    tally = run.begin_order(1)
    return run, tally


_CC14_KINDS = {
    "count": (_kind_count, "the counter gives 30 valid pairings"),
    "non_integral_weight": (_kind_half_prefactor, "not a non-negative integer"),
    "negative_weight": (_kind_negative_weight,
                        "stand for -1 pairings, which is not a non-negative integer"),
    "unreadable_node": (_kind_unreadable, "DiracDelta"),
    "class_size": (_kind_class_size,
                   "a class of 3 labelled topologies has |Aut_V| = 2 and |G| = 4"),
    "not_a_divisor": (_kind_not_a_divisor, "does not divide"),
    "repeated_r_class": (_kind_repeated_r_class, "yields 4 R classes, 2 of them"),
    "refused_layout": (_kind_refused_layout, "the counter cannot take this layout"),
}


@pytest.mark.parametrize("kind", list(_CC14_KINDS))
def test_CC14_each_failure_kind_raises(kind):
    build, message = _CC14_KINDS[kind]
    run, tally = build(make_observable(XY))
    run.end_order(tally)
    with pytest.raises(CertificateError) as excinfo:
        run.finish()
    err = excinfo.value
    assert len(err.failures) == 1, err.failures
    assert message in err.failures[0]
    assert message in str(err)
    (order,) = err.certificate.orders
    assert len(order.sequences) == len(tally.combinations)
    if kind in ("class_size", "not_a_divisor"):
        (failed,) = [row for row in order.sequences if row.classes]
        assert failed.group_order is not None and failed.orbit_sum is None
    back = pickle.loads(pickle.dumps(err))
    assert back.failures == err.failures and back.certificate == err.certificate


class _Dropping(Action):
    """An action whose order-2 expansion leaves out the sequence (F, G)."""

    def all_vertex_combinations(self, order):
        for seq, multinomial in super().all_vertex_combinations(order):
            if order == 2 and [v.coupling for v in seq] == ["F", "G"]:
                continue
            yield seq, multinomial


def test_CC15_a_dropped_sequence_raises():
    action = _Dropping([make_vertex(F), make_vertex(G)])
    reset_uid_counter()
    with pytest.raises(CertificateError) as excinfo:
        compute_moment_numerical(make_observable(XY), action, 2, certify=True,
                                 canonical="builtin")
    err = excinfo.value
    assert len(err.failures) == 1
    assert "add up to 2, not 2**2 = 4" in err.failures[0]
    rows = err.certificate.order(2).sequences
    assert [(r.vertices, r.counter, r.term_weight) for r in rows] == [
        (("F", "F"), 30, 30), (("G", "G"), 30, 30)]


# --------------------------------------------------------------------- #
# The output of a certified call
# --------------------------------------------------------------------- #

_CC16_CASES = [
    pytest.param(model, engine, id=f"{model}-{engine}")
    for model in ("cubic",) + OPERATOR_PATH_MODELS
    for engine in ("numerical", "moment") + (
        ("operator",) if model in OPERATOR_PATH_MODELS else ())
]


def _output_of(engine, observable, action, order, certify):
    """The returned terms of one call, as reprs in their order."""
    reset_uid_counter()
    if engine == "numerical":
        out = compute_moment_numerical(observable, action, order, canonical="builtin",
                                       certify=certify)
        return {n: [repr(t) for t in terms] for n, terms in out.items()}
    result = compute_moment(observable, action, order, certify=certify,
                            collect_topology=engine == "moment")
    return {
        "terms": {n: [repr(t) for t in terms]
                  for n, terms in result.diagram_terms_by_order.items()},
        "infos": {n: len(infos) for n, infos in result.diagrams_by_order.items()},
        "order_terms": repr(result.order_terms),
        "total": repr(result.total),
    }


@pytest.mark.parametrize("model, engine", _CC16_CASES)
def test_CC16_a_certified_call_returns_the_uncertified_terms(model, engine):
    vertices, externals, top = MODELS[model]
    plain, certified = (
        _output_of(engine, make_observable(externals), make_action(vertices), top,
                   certify)
        for certify in (False, True))
    assert certified == plain
    # Every model here has pairings above order 0.  The check looks there
    # because with XY externals order 0 alone is not empty.
    if engine == "numerical":
        assert any(_above_order_0(plain).values())
    else:
        assert sum(_above_order_0(plain["infos"]).values()) > 0
        if engine == "moment":
            assert any(_above_order_0(plain["terms"]).values())


def _above_order_0(by_order):
    """The entries of a per-order dict at orders 1 and above."""
    return {n: v for n, v in by_order.items() if n > 0}


def test_CC17_a_certified_expansion_holds_the_uncertified_terms():
    observable = ("phi_a(x)", "phi_b(y)", "phi_c(z)")
    dumps = []
    for certify in (False, True):
        reset_uid_counter()
        expansion = _system_f_k().expand(observable, orders=[0, 1, 2, 3],
                                         progress=False, certify=certify)
        dumps.append({n: [repr(t) for t in terms]
                      for n, terms in expansion.dts_by_order.items()})
    assert dumps[1] == dumps[0]
    assert sum(len(terms) for terms in dumps[0].values()) > 0


def test_CC18_a_coupling_name_that_is_not_a_str_leaves_the_classes_empty():
    action = Action([Vertex([PSI, PHI, PHI], 5)])
    reset_uid_counter()
    plain = compute_moment(make_observable(XY), action, 2)
    reset_uid_counter()
    cert = compute_moment(make_observable(XY), action, 2, certify=True).certificate
    (row,) = cert.order(2).sequences
    assert row.note.startswith("classes not computed")
    assert (row.topologies, row.classes, row.group_order, row.orbit_sum) == (None,) * 4
    assert row.counter == row.term_weight == row.diagram_infos == 30
    assert len(plain.diagrams_by_order[2]) == 30


class _Alternating(Action):
    """An action that yields the sequences of an order in reverse on every
    second call."""

    def __init__(self, vertices):
        super().__init__(vertices)
        self.calls = 0

    def all_vertex_combinations(self, order):
        out = list(super().all_vertex_combinations(order))
        self.calls += 1
        return iter(out[::-1] if self.calls % 2 == 0 else out)


@pytest.mark.parametrize("engine", ["numerical", "moment", "routings"])
def test_CC19_rows_belong_to_the_sequences_the_engine_contracts(engine):
    action = _Alternating([make_vertex(F), make_vertex(M)])
    cert = certificate_of(engine, make_observable(XY), action, 2)
    for n, order in enumerate(cert.orders):
        expected = sorted((b.vertices, b.multinomial, b.moment.pairings)
                          for b in brute_rows(XY, (F, M), n))
        assert sorted((r.vertices, r.multinomial, r.counter)
                      for r in order.sequences) == expected


def test_CC20_a_counter_error_that_is_not_a_refusal_propagates():
    run = _run(make_observable(XY), (F,), target="cumulant")
    with pytest.raises(ValueError, match="target must be one of") as excinfo:
        run.begin_order(1)
    assert not isinstance(excinfo.value, _certify.UnsupportedLayout)


def test_CC21_the_message_shows_ten_failures_and_counts_the_rest():
    vertices = tuple((f"V{i}", "sff", True) for i in range(12))
    run = _run(make_observable(XY), vertices)
    tally = _matched_tally(run, 1)
    for index in range(len(vertices)):
        tally.add_r_classes(index, 2, 1)
    run.end_order(tally)
    with pytest.raises(CertificateError) as excinfo:
        run.finish()
    lines = str(excinfo.value).splitlines()
    assert len(excinfo.value.failures) == 12
    assert sum(line.startswith("  order 1, sequence") for line in lines) == 10
    assert lines[-2] == "  and 2 more"


def _alive(types) -> int:
    gc.collect()
    return sum(isinstance(obj, types) for obj in gc.get_objects())


@pytest.mark.parametrize("engine", ["numerical", "moment"])
def test_CC22_a_failed_certificate_keeps_no_terms(engine, monkeypatch):
    counter = _certificate.count_valid_pairings
    monkeypatch.setattr(_certificate, "count_valid_pairings",
                        lambda **kwargs: counter(**kwargs) + 1)
    try:
        certificate_of(engine, make_observable(XY), make_action((F,)), 2)
    except CertificateError as exc:
        error = exc
    else:
        pytest.fail("no CertificateError")
    assert "the counter gives 31 valid pairings" in str(error)
    types = (perturbation.DiagramTerm, perturbation.DiagramInfo, Propagator)
    held = _alive(types)
    error.__traceback__ = None
    assert held == _alive(types)
