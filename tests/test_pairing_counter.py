"""The pairing counter of ``sft_wick._certify`` against brute-force enumeration.

A valid pairing is a complete Wick pairing of phi and psi operators in which
no psi pairs with a psi, no response propagator joins two operators on one
point, and the response propagators form no directed cycle on the points.

The references are ``iter_kept_pairings`` of ``tests/brute_wick.py``,
``_pruned_count`` in this file, and the two algorithms of the counter
checked against each other.

``iter_kept_pairings`` enumerates all ``(2n-1)!!`` pairings and classifies
each.  It imports nothing from the package.  It is run here on every case
of at most 14 operators (135,135 pairings) and is not run on larger ones.

``_pruned_count`` applies the pair rule of ``brute_wick`` during the search
and its cycle test at each leaf, so it visits only pairings without a
psi-psi pair or a same-point response propagator.  It is compared with
``iter_kept_pairings`` on the small cases.  It is the reference for the
pins of 18 operators (34,459,425 unfiltered pairings each: the double-psi
vertex at order 4 and quartic order 4, ``heavy``) and of cubic order 6 (20
operators, 654,729,075 unfiltered pairings, ``slow``).  The 32,432,400 pin
of cubic order 6 is also checked in the default run by the two algorithms
of the counter.

Quartic orders 5 and 6 (908,107,200 and 377,123,947,200), the connected
counts of cubic order 6 and quartic order 4, and the two pins at 13 psi
have the two algorithms of the counter as their only check in this suite.
During review a separate count of injective psi-to-phi maps with its own
cycle test, sharing no code with the counter or with ``brute_wick``,
reproduced each of them.

The connected counts are compared with ``iter_kept_pairings`` filtered by a
union-find on owners written in this file.  The contraction engine gives
one owner to each external operator and one to each vertex instance, a
non-local vertex included.  A ``Site`` with several external phi and one
owner is a layout the counter accepts that the engine does not produce.
"""
from __future__ import annotations

from typing import NamedTuple

import pytest

from sft_wick import _certify
from sft_wick._certify import count_valid_pairings

from . import brute_wick as brute
from .brute_wick import Op

#: Largest operator count given to ``iter_kept_pairings`` (13!! pairings).
BRUTE_MAX_OPERATORS = 14

# (n_phi, n_psi) of one external occurrence or one local vertex.
PHI = (1, 0)
PSI = (0, 1)
CUBIC = (2, 1)
QUARTIC = (3, 1)
DOUBLE_PSI = (2, 2)
TWO_POINT = (PHI, PHI)


class Site(NamedTuple):
    """Operators of one owner on one point."""

    label: str
    owner: str
    n_phi: int
    n_psi: int


def local_case(observable, vertices) -> list[Site]:
    """One point and one owner per external occurrence and per vertex."""
    sites = [Site(f"x{k}", f"ext{k}", n_phi, n_psi)
             for k, (n_phi, n_psi) in enumerate(observable)]
    sites += [Site(f"z{k}", f"v{k}", n_phi, n_psi)
              for k, (n_phi, n_psi) in enumerate(vertices)]
    return sites


def nonlocal_vertex(name: str, legs: str) -> list[Site]:
    """A vertex with one leg per point; ``legs`` is a string of ``f``/``p``
    for phi and psi."""
    return [Site(f"{name}_{k}", name, int(leg == "f"), int(leg == "p"))
            for k, leg in enumerate(legs)]


def n_operators(sites) -> int:
    return sum(s.n_phi + s.n_psi for s in sites)


def brute_feasible(sites) -> bool:
    """An odd number of operators has no pairing to enumerate."""
    n = n_operators(sites)
    return n <= BRUTE_MAX_OPERATORS or n % 2 == 1


def counter_inputs(sites) -> dict:
    """The plain data the counter takes, built from ``sites``."""
    psi_owner = [k for k, s in enumerate(sites) for _ in range(s.n_psi)]
    return {"points": [s.label for s in sites],
            "phi_counts": [s.n_phi for s in sites],
            "psi_owner": psi_owner,
            "owner_of_group": [s.owner for s in sites]}


def count(sites, target="moment", method="auto") -> int:
    return count_valid_pairings(**counter_inputs(sites), target=target,
                                method=method)


def brute_operators(sites) -> tuple[list[Op], list[str]]:
    """``brute_wick`` operators and the owner of each uid."""
    ops, owners = [], []
    for s in sites:
        for kind, n in (("phi", s.n_phi), ("psi", s.n_psi)):
            for _ in range(n):
                ops.append(Op(kind, s.label, None, len(ops)))
                owners.append(s.owner)
    return ops, owners


def _one_component(pairing, owners, nodes) -> bool:
    """Whether the pairs join every owner in ``nodes`` into one component."""
    if not nodes:
        return False
    parent = {node: node for node in nodes}

    def find(node):
        while parent[node] != node:
            node = parent[node]
        return node

    for pair in pairing:
        a, b = (find(owners[uid]) for uid in pair)
        if a != b:
            parent[a] = b
    return len({find(node) for node in nodes}) == 1


def brute_count(sites, target="moment") -> int:
    assert brute_feasible(sites)
    ops, owners = brute_operators(sites)
    nodes = {s.owner for s in sites}
    kept = brute.iter_kept_pairings(ops)
    if target == "moment":
        return sum(1 for _ in kept)
    return sum(1 for pairing, _ in kept
               if _one_component(pairing, owners, nodes))


def _pruned_count(sites) -> int:
    """Kept pairings, with ``brute_wick``'s pair rule applied during the
    search and its cycle test at each leaf."""
    ops, _ = brute_operators(sites)

    def extend(remaining, edges):
        if not remaining:
            return 0 if brute._has_r_cycle(list(edges)) else 1
        first, rest = remaining[0], remaining[1:]
        total = 0
        for i, partner in enumerate(rest):
            prop, _ = brute._contract(ops[first], ops[partner])
            if prop is None:
                continue
            grown = edges + (prop,) if prop[0] == "R" else edges
            total += extend(rest[:i] + rest[i + 1:], grown)
        return total

    return extend(tuple(range(len(ops))), ())


# --------------------------------------------------------------------- #
# Moment pins
# --------------------------------------------------------------------- #

MOMENT_PINS = [
    ("order 0", local_case(TWO_POINT, []), 1),
    ("cubic order 1", local_case(TWO_POINT, [CUBIC]), 0),
    ("cubic order 2", local_case(TWO_POINT, [CUBIC] * 2), 30),
    ("cubic order 3", local_case(TWO_POINT, [CUBIC] * 3), 0),
    ("cubic order 4", local_case(TWO_POINT, [CUBIC] * 4), 15_120),
    ("cubic order 5", local_case(TWO_POINT, [CUBIC] * 5), 0),
    ("cubic order 6", local_case(TWO_POINT, [CUBIC] * 6), 32_432_400),
    ("quartic order 1", local_case(TWO_POINT, [QUARTIC]), 6),
    ("quartic order 2", local_case(TWO_POINT, [QUARTIC] * 2), 210),
    ("quartic order 3", local_case(TWO_POINT, [QUARTIC] * 3), 18_900),
    ("quartic order 4", local_case(TWO_POINT, [QUARTIC] * 4), 3_243_240),
    ("double-psi order 1", local_case(TWO_POINT, [DOUBLE_PSI]), 2),
    ("double-psi order 2", local_case(TWO_POINT, [DOUBLE_PSI] * 2), 8),
    ("double-psi order 3", local_case(TWO_POINT, [DOUBLE_PSI] * 3), 48),
    ("double-psi order 4", local_case(TWO_POINT, [DOUBLE_PSI] * 4), 384),
    ("cubic cubic double-psi",
     local_case(TWO_POINT, [CUBIC, CUBIC, DOUBLE_PSI]), 564),
]

_PIN_IDS = [name for name, _, _ in MOMENT_PINS]
_SMALL_PINS = [pin for pin in MOMENT_PINS if brute_feasible(pin[1])]
_LARGE_PINS = [pin for pin in MOMENT_PINS if not brute_feasible(pin[1])]


@pytest.mark.parametrize("method", ["direct", "dp"])
@pytest.mark.parametrize("name, sites, expected", MOMENT_PINS, ids=_PIN_IDS)
def test_PC1_moment_pins(name, sites, expected, method):
    assert count(sites, method=method) == expected


@pytest.mark.parametrize("name, sites, expected", _SMALL_PINS,
                         ids=[name for name, _, _ in _SMALL_PINS])
def test_PC2_moment_pins_against_all_pairings(name, sites, expected):
    """Every pin of at most 14 operators equals the count of
    ``iter_kept_pairings``."""
    assert brute_count(sites) == expected


def test_PC3_the_pins_beyond_all_pairings_are_the_three_expected():
    assert [name for name, _, _ in _LARGE_PINS] == [
        "cubic order 6", "quartic order 4", "double-psi order 4"]


@pytest.mark.parametrize("name, sites, expected", _SMALL_PINS,
                         ids=[name for name, _, _ in _SMALL_PINS])
def test_PC4_pruned_search_equals_all_pairings(name, sites, expected):
    """The reference of the 18-operator pins agrees with
    ``iter_kept_pairings`` wherever both run."""
    assert _pruned_count(sites) == expected


#: Cost of the pruned search on the development machine: about 700 s at
#: cubic order 6, 50 s at quartic order 4 and 4 s at double-psi order 4.
_LARGE_MARKS = {"cubic order 6": [pytest.mark.slow],
                "quartic order 4": [pytest.mark.heavy]}


@pytest.mark.parametrize("name, sites, expected", [
    pytest.param(*pin, id=pin[0], marks=_LARGE_MARKS.get(pin[0], []))
    for pin in _LARGE_PINS])
def test_PC5_large_pins_against_the_pruned_search(name, sites, expected):
    """18 and 20 operators (34,459,425 and 654,729,075 unfiltered
    pairings), where ``iter_kept_pairings`` is not run.  The pruned search
    visits the pairings that survive the pair rule."""
    assert _pruned_count(sites) == expected


@pytest.mark.parametrize("order, expected",
                         [(5, 908_107_200), (6, 377_123_947_200)])
def test_PC6_quartic_orders_5_and_6_by_both_algorithms(order, expected):
    """No enumeration of pairings runs at this size; the two algorithms of
    the counter are the only check in this suite."""
    sites = local_case(TWO_POINT, [QUARTIC] * order)
    assert count(sites, method="direct") == expected
    assert count(sites, method="dp") == expected


# --------------------------------------------------------------------- #
# Connected pins
# --------------------------------------------------------------------- #

CONNECTED_PINS = [
    ("order 0", local_case(TWO_POINT, []), 1),
    ("cubic order 1", local_case(TWO_POINT, [CUBIC]), 0),
    ("cubic order 2", local_case(TWO_POINT, [CUBIC] * 2), 28),
    ("cubic order 3", local_case(TWO_POINT, [CUBIC] * 3), 0),
    ("cubic order 4", local_case(TWO_POINT, [CUBIC] * 4), 14_400),
    ("quartic order 1", local_case(TWO_POINT, [QUARTIC]), 6),
    ("quartic order 2", local_case(TWO_POINT, [QUARTIC] * 2), 210),
    ("quartic order 3", local_case(TWO_POINT, [QUARTIC] * 3), 18_900),
    ("double-psi order 2", local_case(TWO_POINT, [DOUBLE_PSI] * 2), 8),
    ("cubic cubic double-psi",
     local_case(TWO_POINT, [CUBIC, CUBIC, DOUBLE_PSI]), 556),
]
_CONNECTED_IDS = [name for name, _, _ in CONNECTED_PINS]


@pytest.mark.parametrize("method", ["direct", "dp"])
@pytest.mark.parametrize("name, sites, expected", CONNECTED_PINS,
                         ids=_CONNECTED_IDS)
def test_PC7_connected_pins_from_the_subset_recursion(name, sites, expected,
                                                      method):
    assert count(sites, "connected", method) == expected


@pytest.mark.parametrize("name, sites, expected", CONNECTED_PINS,
                         ids=_CONNECTED_IDS)
def test_PC8_connected_pins_against_filtered_pairings(name, sites, expected):
    assert brute_count(sites, "connected") == expected


def test_PC9_connected_counts_beyond_all_pairings():
    """Cubic order 6 and quartic order 4, from the subset recursion with
    each algorithm.  No enumeration of pairings checks these two values.

    In the quartic pins the connected count equals the moment count.  A
    component with one external is left with an odd number of phi, and in a
    component of vertices alone each vertex sends its psi to another, which
    closes a response cycle.
    The subtraction in the recursion is exercised by the cubic and
    double-psi pins and by the cases of PC10."""
    cubic = local_case(TWO_POINT, [CUBIC] * 6)
    quartic = local_case(TWO_POINT, [QUARTIC] * 4)
    assert count(cubic, "connected") == 31_363_200
    assert count(quartic, "connected", "direct") == 3_243_240
    assert count(quartic, "connected", "dp") == 3_243_240


# --------------------------------------------------------------------- #
# Cases without the symmetry of the pins
# --------------------------------------------------------------------- #

TRIPLE_PSI = (3, 3)
PSI_ONLY = (0, 2)

ASYMMETRIC_CASES = {
    "externals with unequal phi counts":
        local_case([(3, 0), (1, 0)], [CUBIC, CUBIC]),
    "composite external and a quartic vertex":
        local_case([(2, 0), (1, 0)], [QUARTIC, CUBIC]),
    "vertex with three psi":
        local_case([(2, 0), (1, 0)], [TRIPLE_PSI, CUBIC]),
    "external psi":
        local_case([PHI, PSI], [CUBIC, CUBIC]),
    "external phi and psi on one occurrence":
        local_case([(1, 1), PHI], [CUBIC, QUARTIC]),
    "vertices with zero phi":
        local_case([(3, 0), (2, 0)], [PSI_ONLY, (0, 1), (1, 1)]),
    "psi outnumbering phi":
        local_case(TWO_POINT, [PSI_ONLY, PSI_ONLY]),
    "empty observable, every vertex with psi":
        local_case([], [CUBIC, CUBIC, DOUBLE_PSI]),
    "empty observable, one vertex without psi":
        local_case([], [CUBIC, QUARTIC, (3, 0)]),
    "vertex with no operators":
        local_case(TWO_POINT, [CUBIC, CUBIC, (0, 0)]),
    "three vertex types":
        local_case([PHI], [CUBIC, QUARTIC, DOUBLE_PSI]),
}

NONLOCAL_CASES = {
    "three psi legs beside a cubic vertex":
        local_case(TWO_POINT, [CUBIC]) + nonlocal_vertex("k", "ppp"),
    "one mixed vertex":
        local_case(TWO_POINT, []) + nonlocal_vertex("k", "pfff"),
    "two mixed vertices":
        local_case(TWO_POINT, []) + nonlocal_vertex("k", "pff")
        + nonlocal_vertex("m", "pff"),
    "mixed vertex beside a local cubic":
        local_case(TWO_POINT, [CUBIC]) + nonlocal_vertex("k", "pff"),
    "two vertices of two legs, no observable":
        nonlocal_vertex("k", "pf") + nonlocal_vertex("m", "pf")
        + nonlocal_vertex("n", "pf"),
    "phi legs with a double-psi vertex":
        local_case(TWO_POINT, [DOUBLE_PSI, CUBIC])
        + nonlocal_vertex("k", "fff"),
}

SHARED_LABEL_CASES = {
    "two externals on one label":
        [Site("x", "ext0", 1, 0), Site("x", "ext1", 1, 0)]
        + local_case([], [CUBIC, CUBIC]),
    "three externals, two on one label":
        [Site("x", "ext0", 1, 0), Site("x", "ext1", 1, 0),
         Site("y", "ext2", 1, 0)] + local_case([], [CUBIC, QUARTIC]),
    "an external on the label of a vertex":
        [Site("z0", "ext0", 1, 0), Site("y", "ext1", 1, 0)]
        + local_case([], [CUBIC, CUBIC]),
    "three external operators on one label, one owner each":
        [Site("x", f"ext{k}", 1, 0) for k in range(3)]
        + [Site("y", "ext3", 1, 0)] + local_case([], [CUBIC, CUBIC]),
    # Two shared labels, one with phi only: one label carries both phi and
    # psi, so the connected target accepts the layout.
    "two vertices on one label, two externals on another":
        [Site("x", "ext0", 1, 0), Site("x", "ext1", 1, 0),
         Site("z", "v0", 2, 1), Site("z", "v1", 2, 1)],
    # Label w holds a vertex and an empty group of another owner: the empty
    # group does not make w a second shared label with phi and psi.
    "an empty group on the label of a vertex":
        [Site("x", "ext0", 1, 0), Site("y", "ext1", 1, 0),
         Site("u", "ext2", 1, 0), Site("z", "v0", 2, 1),
         Site("z", "v1", 2, 1), Site("w", "v2", 2, 1),
         Site("w", "v0", 0, 0)],
}

_ALL_CASES = {**ASYMMETRIC_CASES, **NONLOCAL_CASES, **SHARED_LABEL_CASES}

#: Counts fixed from ``iter_kept_pairings``: (moment, connected).
CASE_PINS = {
    "externals with unequal phi counts": (420, 402),
    "vertex with three psi": (270, 270),
    "vertices with zero phi": (300, 240),
    "psi outnumbering phi": (0, 0),
    "empty observable, every vertex with psi": (0, 0),
    "empty observable, one vertex without psi": (315, 315),
    "vertex with no operators": (30, 0),
    "three psi legs beside a cubic vertex": (12, 12),
    "one mixed vertex": (15, 12),
    "two mixed vertices": (90, 60),
    "two vertices of two legs, no observable": (6, 2),
    "phi legs with a double-psi vertex": (2_850, 2_604),
    "two externals on one label": (30, 28),
    "an external on the label of a vertex": (15, 14),
    "three external operators on one label, one owner each": (420, 192),
    "two vertices on one label, two externals on another": (6, 4),
    "an empty group on the label of a vertex": (900, 696),
}


@pytest.mark.parametrize("target", ["moment", "connected"])
@pytest.mark.parametrize("method", ["direct", "dp"])
@pytest.mark.parametrize("name", list(_ALL_CASES))
def test_PC10_cases_against_all_pairings(name, method, target):
    sites = _ALL_CASES[name]
    expected = brute_count(sites, target)
    assert count(sites, target, method) == expected
    if name in CASE_PINS:
        assert expected == CASE_PINS[name][target == "connected"]


def test_PC11_the_cases_are_not_all_zero():
    """A comparison of zeros with zeros would check nothing."""
    moments = {name: brute_count(sites) for name, sites in _ALL_CASES.items()}
    assert [name for name, value in moments.items() if value == 0] == [
        "psi outnumbering phi", "empty observable, every vertex with psi"]
    connected = {name: brute_count(sites, "connected")
                 for name, sites in _ALL_CASES.items()}
    assert sum(1 for name in _ALL_CASES
               if 0 < connected[name] < moments[name]) >= 8


def test_PC12_a_nonlocal_vertex_is_one_owner():
    """With each leg an owner of its own the connected count differs, so
    the owner map is read."""
    sites = _ALL_CASES["two mixed vertices"]
    split = [s._replace(owner=s.label) for s in sites]
    assert count(sites, "connected") == 60
    assert count(split, "connected") == brute_count(split, "connected") == 0
    assert count(split) == count(sites) == 90


def test_PC13_default_owner_is_the_point():
    sites = local_case(TWO_POINT, [CUBIC] * 2)
    data = counter_inputs(sites)
    del data["owner_of_group"]
    assert count_valid_pairings(**data, target="connected") == 28


# --------------------------------------------------------------------- #
# The subset recursion over in-degree vectors at 13 psi
# --------------------------------------------------------------------- #

THIRTEEN_PSI = {
    "five points, one without psi":
        local_case([], [(6, 0), (5, 4), (5, 3), (4, 3), (3, 3)]),
    "externals and a point without phi":
        local_case([(2, 0), (2, 0)], [(4, 4), (0, 2), (5, 4), (4, 3)]),
}


@pytest.mark.parametrize("name", list(THIRTEEN_PSI))
def test_PC15_thirteen_psi_dp_against_direct(name):
    sites = THIRTEEN_PSI[name]
    assert sum(s.n_psi for s in sites) == 13 > _certify.DIRECT_MAX_PSI
    direct = count(sites, method="direct")
    assert direct > 0
    assert count(sites, method="dp") == direct


def test_PC16_auto_takes_the_dp_above_the_threshold(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("direct enumeration was called")

    sites = THIRTEEN_PSI["five points, one without psi"]
    expected = count(sites, method="dp")
    monkeypatch.setattr(_certify, "_count_direct", refuse)
    assert count(sites) == expected
    with pytest.raises(AssertionError, match="direct enumeration"):
        count(local_case(TWO_POINT, [CUBIC] * 2))


def test_PC17_thirteen_psi_pins():
    """Values produced by the counter; PC15 compares its two algorithms."""
    assert count(THIRTEEN_PSI["five points, one without psi"]) == 545_080_741_632_000
    assert count(THIRTEEN_PSI["externals and a point without phi"]) == (
        195_955_200)


# --------------------------------------------------------------------- #
# Empty input and refusals
# --------------------------------------------------------------------- #


def test_PC18_empty_input():
    assert count_valid_pairings([], [], []) == 1
    assert count_valid_pairings([], [], [], target="connected") == 0


def test_PC19_one_owner_without_operators_is_connected():
    assert count([Site("x", "ext0", 0, 0)], "connected") == 1
    assert count([Site("x", "ext0", 2, 0)], "connected") == 1
    assert count([Site("x", "e0", 1, 0), Site("y", "e1", 1, 0),
                  Site("z", "v0", 0, 0)], "connected") == 0


def test_PC20_connected_refuses_cycles_through_shared_labels(monkeypatch):
    """Two labels, each shared by two owners and each with phi and psi.

    Joining each psi to the phi of its own owner is one valid pairing per
    owner taken alone and a response cycle a -> b -> a taken together, so
    the count of the union is not a product over components.  Without the
    refusal the recursion on owners returns -1.
    """
    sites = [Site("a", "o0", 0, 1), Site("b", "o0", 1, 0),
             Site("a", "o1", 1, 0), Site("b", "o1", 0, 1)]
    assert count(sites) == brute_count(sites) == 0
    assert brute_count(sites, "connected") == 0
    with pytest.raises(ValueError, match="shared by several owners"):
        count(sites, "connected")
    monkeypatch.setattr(_certify, "_require_cycles_within_components",
                        lambda groups: None)
    assert count(sites, "connected") == -1


@pytest.mark.parametrize("kwargs, message", [
    ({"points": "xy"}, "points must be a sequence"),
    ({"points": [["x"], "y"]}, "hashable"),
    ({"phi_counts": [1]}, "phi_counts has 1 entries for 2 points"),
    ({"phi_counts": [1, -1]}, "phi_counts"),
    ({"phi_counts": [1, 1.0]}, "phi_counts"),
    ({"phi_counts": [1, True]}, "phi_counts"),
    ({"phi_counts": 2}, "phi_counts must be a sequence"),
    ({"psi_owner": [2]}, "psi_owner"),
    ({"psi_owner": [-1]}, "psi_owner"),
    ({"psi_owner": ["x"]}, "psi_owner"),
    ({"psi_owner": [False]}, "psi_owner"),
    ({"owner_of_group": ["a"]}, "owner_of_group has 1 entries for 2 points"),
    ({"owner_of_group": [["a"], "b"]}, "hashable"),
    ({"target": "cumulant"}, "target"),
    ({"method": "fast"}, "method"),
])
def test_PC21_invalid_input_raises(kwargs, message):
    data = {"points": ["x", "y"], "phi_counts": [1, 1], "psi_owner": []}
    data.update(kwargs)
    with pytest.raises(ValueError, match=message):
        count_valid_pairings(**data)
