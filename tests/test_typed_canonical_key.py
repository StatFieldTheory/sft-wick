"""Typed canonical keys of labelled topologies, and their point automorphism
order.

``compute_moment_numerical`` groups the labelled topologies of a vertex
sequence by a key of :mod:`sft_wick._canonical`.  Two topologies share a key
when a relabelling of the integrated points that keeps every vertex maps one
onto the other: the points of one local vertex value permute among
themselves, and the instances of one non-local vertex permute as whole leg
tuples, leg by leg.  ``|Aut_V|`` is the number of these relabellings that fix
a topology.  It counts no swap of parallel propagators.

The reference is :mod:`tests.brute_relabel`, which tries the relabellings
one by one and shares no code with the package.  A test that takes
``canonical_method_name`` runs once with the built-in canonicaliser and
once with pynauty.  The pynauty run is skipped without pynauty, and fails
instead when ``SFT_WICK_REQUIRE_PYNAUTY=1``.  Method selection, the mixing
guards, hash seeds and import rules are in ``tests/test_canonical_method.py``.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import pickle
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from fractions import Fraction
from math import factorial
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pytest

import sft_wick
from sft_wick import Action, Field, Vertex, reset_uid_counter
from sft_wick._canonical import (
    BuiltinKeyer,
    CanonicalMethod,
    SequenceLayout,
    _point_automorphism_order,
    make_keyer,
    resolve_canonical_method,
)
from sft_wick.ir import diagram_ids, from_ir, to_ir
from sft_wick.expressions import Product, Propagator, Rational, Sum, Symbol
from sft_wick.perturbation import (
    DiagramTerm,
    _diagram_terms,
    _pure_coupling,
    _vertex_sequences,
    compute_moment,
    compute_moment_numerical,
)

from . import _canonical_models as models
from . import brute_relabel as brute

SRC = Path(sft_wick.__file__).resolve().parent


def _layout(case: models.Case) -> SequenceLayout:
    if case.seq is not None:
        return SequenceLayout.from_sequence(case.seq, case.observable)
    return SequenceLayout.from_plain(**case.plain.kwargs())


# ----------------------------------------------------------------------
# classes and |Aut_V|
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, order, typed, one_type",
    [("F,G", 2, 12, 6), ("F,F,F,G", 4, 246, 64), ("F,F,G,G", 4, 364, 64)],
)
def test_vertex_types_separate_classes(label, order, typed, one_type, canonical_method_name):
    """F and G have one shape.  A key that drops the vertex type merges (F, G)
    into 6 classes and (F, F, G, G) into 64; the same topologies on a layout
    whose points are all F give those counts."""
    method = resolve_canonical_method(canonical_method_name)
    case = models.engine_case("F_G", order, label)
    keyer = make_keyer(_layout(case), method)
    assert len({keyer.key(p) for p in case.topologies}) == typed
    one = case.plain.kwargs()
    one["local_points"] = tuple((point, models.F) for point, _vertex in one["local_points"])
    one_keyer = make_keyer(SequenceLayout.from_plain(**one), method)
    assert len({one_keyer.key(p) for p in case.topologies}) == one_type


@pytest.mark.parametrize(
    "order, label, orbit_sum, full_order_sum",
    [(2, "F,F", 12, 11), (4, "F,F,F,F", 1416, 1308)],
)
def test_point_automorphism_order_excludes_parallel_propagators(
    order, label, orbit_sum, full_order_sum, canonical_method_name
):
    """Each class has |G| / |Aut_V| members, so the classes sum to the number
    of labelled topologies.  pynauty's order of the dummy-node graph also
    counts swaps of parallel propagators: used in place of |Aut_V| it gives
    11 and 1,308, and divided by the product of m! it gives |Aut_V|."""
    method = resolve_canonical_method(canonical_method_name)
    case = models.engine_case("cubic", order, label)
    layout = _layout(case)
    keyer = make_keyer(layout, method)
    _group, data = models.brute_orbits(case)
    keys = [keyer.key(p) for p in case.topologies]
    auts = [layout.point_automorphism_order(p) for p in case.topologies]
    assert auts == [stabiliser for _class, stabiliser in data]
    assert auts == [_point_automorphism_order(p, layout) for p in case.topologies]
    if isinstance(keyer, BuiltinKeyer):
        pairs = [keyer.key_and_point_automorphism_order(p) for p in case.topologies]
        assert pairs == list(zip(keys, auts))
    members = defaultdict(list)
    for key, aut in zip(keys, auts):
        members[key].append(aut)
    assert all(len(m) * m[0] == layout.group_order for m in members.values())
    assert sum(layout.group_order // m[0] for m in members.values()) == orbit_sum
    assert orbit_sum == len(case.topologies)
    if method.name == "pynauty":
        import pynauty

        full = [models.dummy_graph_group_order(pynauty, case.plain, p) for p in case.topologies]
        assert full == [aut * models.twin_factor(p) for aut, p in zip(auts, case.topologies)]
        first_full = {}
        for key, order_of_graph in zip(keys, full):
            first_full.setdefault(key, order_of_graph)
        assert sum(Fraction(layout.group_order, f) for f in first_full.values()) == full_order_sum


@pytest.mark.heavy
def test_cubic_order_six_classes_are_the_orbits(pynauty_module):
    """All 738,900 labelled topologies of the cubic vertex at order 6, one
    at a time: both methods give the same 1,088 classes, and each class is
    one orbit of the 720 relabellings."""
    obs = models.observable(models.XY)
    (seq,) = list(_vertex_sequences(obs, Action([models.F]), 6, "moment"))
    layout = SequenceLayout.from_sequence(seq, obs)
    keyers = [make_keyer(layout, resolve_canonical_method(n)) for n in ("builtin", "pynauty")]
    class_ids: tuple[dict, dict] = ({}, {})
    paired: dict[int, int] = {}
    sizes: list[int] = []
    representatives: list = []
    count = 0
    for props in models.stream_topologies(seq):
        count += 1
        ids = [found.setdefault(k.key(props), len(found)) for found, k in zip(class_ids, keyers)]
        assert paired.setdefault(ids[0], ids[1]) == ids[1]
        if ids[0] == len(sizes):
            sizes.append(0)
            representatives.append(props)
        sizes[ids[0]] += 1
    assert count == 738_900
    assert len(class_ids[0]) == len(class_ids[1]) == len(set(paired.values())) == 1_088
    group = brute.relabelling_group(*models.brute_structure(models.plain_layout(seq, obs)))
    assert len(group) == layout.group_order == 720
    orbit_sum = 0
    for size, representative in zip(sizes, representatives):
        _class, stabiliser = brute.orbit_data(models.plain_topology(representative), group)
        assert layout.point_automorphism_order(representative) == stabiliser
        assert size * stabiliser == 720
        orbit_sum += 720 // stabiliser
    assert orbit_sum == 738_900


#: The cases compared with the reference: ``(model, order, sequence)`` of
#: an engine model, where ``None`` takes every sequence of the model, or the
#: label of a hand-built case.  The engine cases with two or more instances
#: of one non-local vertex need the instance nodes; the hand-built ones are
#: graphs the engine does not produce, with large symmetric groups, regular
#: graphs that colour refinement cannot split, and parallel propagators.
BRUTE_CASES = [
    ("cubic", 2, None),
    ("cubic", 4, None),
    ("F_G", 2, None),
    ("F_G", 4, None),
    ("quartic", 3, None),
    ("M", 4, None),
    ("psi_phi5", 2, None),
    ("F_K", 3, None),
    ("F_Kr", 3, None),
    ("F_L", 3, None),
    ("cubic_response", 2, None),
    ("K_Keq", 2, None),
    ("F_Fn2", 2, None),
    ("K_Kr", 2, None),
    ("F_K2", 4, (models.F, models.F, models.K2, models.K2)),
    "star k=6",
    "tree 3x2",
    "two triangles, hexagon",
    "cube, Wagner, two K4",
    "twin C 3x(m=5)",
    "twin R and C 3 pairs",
    "R 3-cycle and its reverse",
]

#: Class counts of engine sequences that the reference must reproduce.
CLASS_PINS = {
    ("cubic", 2, "F,F"): 6,
    ("cubic", 4, "F,F,F,F"): 64,
    ("F_G", 2, "F,G"): 12,
    ("F_G", 4, "F,F,F,G"): 246,
    ("F_G", 4, "F,F,G,G"): 364,
    ("F_K", 3, "F,F,K"): 153,
    ("F_L", 3, "L,L,L"): 14,
    ("K_Keq", 2, "K,K"): 360,
    ("K_Keq", 2, "K,K[eq]"): 720,
    ("K_Keq", 2, "K[eq],K[eq]"): 360,
    # Same coupling name, other fields or another flag: two types.
    ("F_Fn2", 2, "F,F[N2]"): 12,
    ("K_Kr", 2, "K,K[R]"): 720,
    ("F_K2", 4, "F,F,K2,K2"): 16,
}

#: |Aut_V| of each topology of the hand-built cases.
HAND_BUILT_AUT = {
    "star k=6": [720],
    "tree 3x2": [48],
    "two triangles, hexagon": [72, 12],
    "cube, Wagner, two K4": [48, 16, 1152],
    "twin C 3x(m=5)": [48],
    "twin R and C 3 pairs": [6],
    "R 3-cycle and its reverse": [1, 1],
}


def _brute_case_id(spec) -> str:
    if isinstance(spec, str):
        return spec
    model, order, select = spec
    return f"{model}-{order}" + ("-" + ",".join(v.coupling for v in select) if select else "")


def _check_against_brute(case: models.Case, method: CanonicalMethod) -> None:
    layout = _layout(case)
    keyer = make_keyer(layout, method)
    group, data = models.brute_orbits(case)
    assert layout.group_order == len(group), case.label
    assert len(layout.digest) == 16 and keyer.prefix == method.tag + layout.digest
    if case.seq is not None:
        assert SequenceLayout.from_plain(**case.plain.kwargs()).digest == layout.digest
    keys = [keyer.key(p) for p in case.topologies]
    assert all(key.startswith(keyer.prefix) for key in keys)
    assert models.partition(keys) == models.partition([c for c, _s in data]), case.label
    auts = [layout.point_automorphism_order(p) for p in case.topologies]
    assert auts == [stabiliser for _class, stabiliser in data], case.label
    rng = random.Random(case.label)
    for props, key in zip(case.topologies, keys):
        assert keyer.key(models.relabel(props, rng.choice(group))) == key, case.label
    if case.closed:
        per_class = dict(data)
        assert sum(len(group) // s for s in per_class.values()) == len(case.topologies)


@pytest.mark.parametrize("spec", BRUTE_CASES, ids=[_brute_case_id(s) for s in BRUTE_CASES])
def test_keys_match_brute_relabelling(spec, canonical_method_name):
    """Partition, |G|, |Aut_V| of every topology and the orbit sum equal the
    reference, and a relabelled topology keeps its key."""
    method = resolve_canonical_method(canonical_method_name)
    if isinstance(spec, str):
        (case,) = [c for c in models.hand_built_cases() if c.label == spec]
        _check_against_brute(case, method)
        assert [s for _c, s in models.brute_orbits(case)[1]] == HAND_BUILT_AUT[spec]
        return
    cases = models.engine_cases(*spec)
    assert cases
    for case in cases:
        _check_against_brute(case, method)
        pin = CLASS_PINS.get(spec[:2] + (case.label,))
        if pin is not None:
            assert len({c for c, _s in models.brute_orbits(case)[1]}) == pin, case.label


def test_classes_hold_equal_coupling_values(canonical_method_name):
    """Two copies of a non-local kappa3 with a generic static tensor, six
    externals, N = 2: the legs of kappa3 are not interchangeable, so the 360
    classes are the reference orbits and the coupling value is one number
    inside each class.  Coarser keys merge classes whose values differ."""
    phi = Field("phi", "physical", n_components=2)
    psi = Field("psi", "response", n_components=2)
    action = Action([Vertex([psi] * 3, "K", local=False)])
    labels = "abcdef"
    obs = [phi(c, f"x{k}") for k, c in enumerate(labels, 1)]
    reset_uid_counter()
    terms = compute_moment_numerical(
        obs, action, 2, diag_R=True, iso_R=True, canonical=canonical_method_name
    )[2]
    tensor = np.random.default_rng(7).normal(size=(2, 2, 2))
    fixed = dict(zip(labels, (0, 1, 1, 0, 1, 0)))
    values = [complex(t.evaluate_coupling({"K": tensor}, fixed)) for t in terms]
    (seq,) = list(_vertex_sequences(obs, action, 2, "moment"))
    keyer = make_keyer(
        SequenceLayout.from_sequence(seq, obs), resolve_canonical_method(canonical_method_name)
    )
    keys = [keyer.key(list(t.propagators)) for t in terms]
    group = brute.relabelling_group(*models.brute_structure(models.plain_layout(seq, obs)))
    classes = [brute.orbit_data(models.plain_topology(t.propagators), group)[0] for t in terms]
    assert len(set(keys)) == 360
    assert models.partition(keys) == models.partition(classes)
    by_key = defaultdict(list)
    for key, value in zip(keys, values):
        by_key[key].append(value)
    scale = max(abs(v) for v in values)
    assert max(abs(v - vs[0]) for vs in by_key.values() for v in vs) <= 1e-12 * scale
    assert max(abs(v - values[0]) for v in values) > 0.1 * scale


#: The best of three key times and then |Aut_V|, in a process of its own,
#: so that a search that blows up fails at the timeout instead of hanging.
#: The child unpickles only the keyer and topology this test pickled.
TIMED_KEY_SCRIPT = """
import json, pickle, sys, time
keyer, props = pickle.load(sys.stdin.buffer)
best = float("inf")
for _ in range(3):
    start = time.perf_counter()
    keyer.key(props)
    best = min(best, time.perf_counter() - start)
aut = keyer.layout.point_automorphism_order(props)
print(json.dumps({"seconds": best, "aut": aut}))
"""

#: Symmetric graphs with |Aut_V|.  The previous key of the engine, pynauty
#: on a directed dummy-node graph, takes about 1.3 s per key on the star, and
#: the built-in search without orbit pruning and without the jump back about
#: 0.4 s.  On the loops and 2-cycles, a refinement that does not tell a
#: self-loop from an arc to a node of the same colour takes about 0.9 s.
TIMED_CASES = {
    "tadpole star 2x5": (models.tadpole_star(2, 5), 2 * factorial(5) ** 2),
    "8 R loops, 8 R 2-cycles": (models.loops_and_two_cycles(8), factorial(8) ** 2 * 2**8),
}


@pytest.mark.parametrize("label", list(TIMED_CASES))
def test_symmetric_graphs_key_within_time_cap(label, canonical_method_name):
    """Each key of a symmetric graph takes under 0.2 s, and a child process
    that has not finished in 30 s fails the test."""
    case, aut = TIMED_CASES[label]
    layout = SequenceLayout.from_plain(**case.plain.kwargs())
    keyer = make_keyer(layout, resolve_canonical_method(canonical_method_name))
    (props,) = case.topologies
    try:
        done = subprocess.run(
            [sys.executable, "-c", TIMED_KEY_SCRIPT],
            input=pickle.dumps((keyer, props)),
            capture_output=True,
            timeout=30,
            check=False,
            env={**os.environ, "PYTHONPATH": str(SRC.parent)},
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{label}: no key and |Aut_V| within 30 s")
    assert done.returncode == 0, done.stderr.decode()
    timed = json.loads(done.stdout.decode().strip().splitlines()[-1])
    assert timed["aut"] == aut
    assert timed["seconds"] < 0.2


def test_strongly_regular_join_keeps_its_key_and_group(canonical_method_name):
    """32 points of one type, the complement of the Shrikhande graph beside
    the 4x4 rook graph, both strongly regular (16, 6, 2, 2), so refinement
    splits nothing and the search needs generators found deep in the tree.
    |Aut_V| = 192 * 1152, and 20 random relabellings keep the key.  Orbits
    built from generators that move the current path split the class, and a
    group order from generators that move the prefix is six times too
    large."""
    case = models.strongly_regular_join()
    layout = _layout(case)
    keyer = make_keyer(layout, resolve_canonical_method(canonical_method_name))
    (props,) = case.topologies
    assert layout.point_automorphism_order(props) == 192 * 1152
    key = keyer.key(props)
    labels = [label for label, _vertex in case.plain.local_points]
    for seed in range(20):
        images = list(labels)
        random.Random(seed).shuffle(images)
        assert keyer.key(models.relabel(props, dict(zip(labels, images)))) == key, seed


# ----------------------------------------------------------------------
# the engine
# ----------------------------------------------------------------------

PHI2 = Field("phi", "physical", n_components=2)
PSI2 = Field("psi", "response", n_components=2)
F2 = Vertex([PSI2, PHI2, PHI2], "F")
G2 = Vertex([PSI2, PHI2, PHI2], "G")


class EngineRun(NamedTuple):
    vertices: tuple[Vertex, ...]
    observable: Callable[[], list]
    order: int
    target: str
    #: Typed classes of (F, F, G, G), each emitted as one block of terms.
    blocks: int | None


ENGINE_RUNS = {
    "F,G order 4 moment": EngineRun(
        (F2, G2), lambda: [PHI2("a", "x"), PHI2("b", "y")], 4, "moment", 364
    ),
    "F,G order 4 connected": EngineRun(
        (F2, G2), lambda: [PHI2("a", "x"), PHI2("b", "y")], 4, "connected", 324
    ),
    "cubic order 4": EngineRun(
        (models.F,), lambda: models.observable(models.XY), 4, "moment", None
    ),
    "F,K order 3 connected": EngineRun(
        (models.F, models.K), lambda: models.observable(models.XYZ), 3, "connected", None
    ),
    "equal-time K order 2": EngineRun(
        (models.K_EQ,), lambda: models.observable(models.SIX), 2, "moment", None
    ),
}


def _engine_terms(run: EngineRun, canonical: str) -> list:
    reset_uid_counter()
    out = compute_moment_numerical(
        run.observable(),
        Action(list(run.vertices)),
        run.order,
        target=run.target,
        canonical=canonical,
    )
    return [term for n in sorted(out) for term in out[n]]


@pytest.mark.parametrize("run", list(ENGINE_RUNS))
def test_engine_emits_each_class_as_one_block(run, canonical_method_name, monkeypatch):
    """``compute_moment_numerical`` emits the terms class by class: each
    typed class of (F, F, G, G) is one block (the untyped key gave 1,015 and
    913 blocks).  The terms are those of a key that puts each topology in a
    class of its own, and both methods give the same list in the same order."""
    spec = ENGINE_RUNS[run]
    terms = _engine_terms(spec, canonical_method_name)
    if spec.blocks is not None:
        ffgg = models.classes_and_blocks(terms, ["F", "F", "G", "G"])
        assert ffgg == (spec.blocks, spec.blocks)
    if canonical_method_name == "builtin":
        serial = itertools.count()

        def own_key(prefix: bytes) -> bytes:
            return prefix + next(serial).to_bytes(8, "big")

        with monkeypatch.context() as patch:
            patch.setattr(BuiltinKeyer, "key", lambda self, _props: own_key(self.prefix))
            own_classes = _engine_terms(spec, "builtin")
        assert next(serial) > 0
        assert Counter(own_classes) == Counter(terms)
    else:
        assert terms == _engine_terms(spec, "builtin")


#: sha256 of the cubic terms up to order 4 in emission order, as the
#: previous untyped key emitted them, with and without pynauty.  With one
#: local vertex type the typed classes are the untyped ones, so the order is
#: the same.
CUBIC_ORDER_4_TERMS = "2c674ea874e4d182fcedb62c74e64d0e9fce1d5a28916cfb329dd6792ef1a6c3"


def test_one_local_type_keeps_the_previous_term_order(canonical_method_name):
    """Only sequences with two or more vertex types or a non-local vertex
    can change the order of the terms."""
    out = _engine_terms(ENGINE_RUNS["cubic order 4"], canonical_method_name)
    digest = hashlib.sha256()
    for term in out:
        digest.update(repr((term.propagators, term.coupling_sum)).encode())
    assert digest.hexdigest() == CUBIC_ORDER_4_TERMS


def test_engine_accepts_degenerate_vertices(canonical_method_name):
    """A component count given as the float 1.0, and a non-local vertex
    with no fields, are accepted as by ``compute_moment``; the counts are
    those of the previous release."""
    phi1 = Field("phi", "physical", n_components=1.0)
    psi1 = Field("psi", "response", n_components=1.0)
    as_float = Action([Vertex([psi1, phi1, phi1], "F")])
    fieldless = Action([models.F, Vertex([], "V", local=False)])
    runs = {}
    for name, action in (("int", Action([models.F])), ("float", as_float), ("fieldless", fieldless)):
        reset_uid_counter()
        out = compute_moment_numerical(
            models.observable(models.XY), action, 2, canonical=canonical_method_name
        )
        runs[name] = {n: terms for n, terms in sorted(out.items())}
    assert {n: len(t) for n, t in runs["float"].items()} == {0: 1, 2: 30}
    assert runs["float"] == runs["int"]
    assert {n: len(t) for n, t in runs["fieldless"].items()} == {0: 1, 1: 1, 2: 31}


@pytest.mark.parametrize("model,order,label", [
    ("F_G", 2, "F,G"),
    ("F_G", 4, "F,F,G,G"),
    ("K_Keq", 2, "K,K[eq]"),
    ("K_Kr", 2, "K,K[R]"),
    ("F_Fn2", 2, "F,F[N2]"),
])
def test_ir_ids_match_the_brute_orbit_of_each_typed_topology(model, order, label):
    case = models.engine_case(model, order, label)
    coupling = _pure_coupling(case.seq.instances)
    ids = [diagram_ids(term) for props in case.topologies
           for term in _diagram_terms(case.seq, tuple(props), coupling, 0,
                                      aligned_instances=True)]
    assert len(ids) == len(case.topologies)
    assert all(len(identities) == 1 for identities in ids)
    _group, data = models.brute_orbits(case)
    assert models.partition(ids) == models.partition([orbit for orbit, _size in data])
    assert len(set(ids)) == CLASS_PINS[model, order, label]


@pytest.mark.parametrize("spec", list(ENGINE_RUNS))
def test_ir_ids_match_between_engines_and_backends(spec, canonical_method_name):
    run = ENGINE_RUNS[spec]
    obs = run.observable()
    action = Action(list(run.vertices))
    reference = compute_moment(obs, action, run.order, target=run.target,
                               routing="multiplicity")
    result = compute_moment_numerical(obs, action, run.order, target=run.target,
                                     routing="multiplicity", canonical=canonical_method_name)
    for order, terms in result.items():
        assert Counter(map(diagram_ids, terms)) == Counter(
            map(diagram_ids, reference.diagram_terms(order))
        )


def test_ir_ids_are_stable_under_uids_seed_and_action_order():
    probe = """
import json
import sys
from sft_wick import Action, Field, Vertex, compute_moment, reset_uid_counter
from sft_wick.ir import diagram_ids

phi = Field('phi', 'physical')
psi = Field('psi', 'response')
for _ in range(20):
    phi('discard')
if sys.argv[1] == 'reset':
    reset_uid_counter()
names = ('F', 'G') if sys.argv[2] == 'forward' else ('G', 'F')
action = Action([Vertex([psi, phi, phi], name) for name in names])
result = compute_moment([phi('x'), phi('y')], action, 2, routing='multiplicity')
print(json.dumps(sorted(diagram_ids(term) for term in result.diagram_terms(2))))
"""
    src = Path(__file__).resolve().parents[1] / "src"
    runs = [("0", "reset", "forward"), ("1", "skip", "forward"),
            ("7", "reset", "reverse"), ("42", "skip", "reverse"),
            ("113", "reset", "forward")]
    results = [subprocess.run(
        [sys.executable, "-c", probe, reset, orientation], text=True, check=True,
        env={"PYTHONPATH": str(src), "PYTHONHASHSEED": seed},
        capture_output=True, timeout=30,
    ).stdout for seed, reset, orientation in runs]
    assert results == [results[0]] * 5
    assert len(set(map(tuple, json.loads(results[0])))) == 24


def test_ir_ids_preserve_r_orientation_ordered_legs_and_time_constraints():
    term = DiagramTerm(
        (Propagator("R", None, None, "x", "u"), Propagator("R", None, None, "y", "v"),
         Propagator("C", None, None, "x", "y")),
        Symbol("K", (), ("u", "v")), Rational(1), ("u", "v"), (), 2,
    )
    (base,) = diagram_ids(term)
    (reversed_c,) = diagram_ids(replace(term, propagators=(
        *term.propagators[:2], Propagator("C", None, None, "y", "x"),
    )))
    assert reversed_c == base
    assert diagram_ids(replace(term, propagators=(
        Propagator("R", None, None, "u", "x"), *term.propagators[1:],
    ))) != (base,)
    reverse_legs = Symbol("K", (), ("v", "u"))
    (swapped,) = diagram_ids(replace(term, coupling_sum=reverse_legs))
    assert swapped != base
    assert diagram_ids(replace(term, coupling_sum=Sum((
        term.coupling_sum, Product((Rational(2), reverse_legs)), term.coupling_sum,
    )))) == tuple(sorted((base, swapped)))
    equal = diagram_ids(replace(term, equal_time_aliases=(("v", "u"),)))
    assert equal != (base,)
    assert equal == diagram_ids(replace(term, equal_time_aliases=(("u", "v"),)))
    same_time = replace(term, equal_time_aliases=(("u", "x"),))
    absorbed = replace(same_time, r_absorbed_pairs=(("x", "u"),))
    assert diagram_ids(same_time) != diagram_ids(absorbed)
    assert diagram_ids(replace(term, rational_prefactor=Rational(-7),
                               response_phase=False)) == (base,)


def test_ir_full_types_preserve_same_name_with_different_field_order():
    phi = Field("phi", "physical", 2)
    psi = Field("psi", "response", 2)
    action = Action([Vertex([psi, phi, phi], "F"), Vertex([phi, psi, phi], "F")])
    obs = [phi("x", "a"), phi("y", "b")]
    result = compute_moment(obs, action, 2, routing="multiplicity")
    typed = [diagram_ids(term) for term in result.diagram_terms(2)]
    assert len(typed) == len(set(typed)) == 24
    restored = [from_ir(to_ir(term)) for term in result.diagram_terms(2)]
    assert [diagram_ids(term) for term in restored] == typed
