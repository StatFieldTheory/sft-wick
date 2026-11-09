"""The oracle reproduces every reference value committed for demos 4, 5
and 7.

Each demo stores the value of its own hand-written moment hierarchy.  Here
``validation/demo_cases.py`` states each demo's model as a ModelSpec, and
``validation/hierarchy.py`` compiles it into a hierarchy with its own state
layout and channel tags.  Every stored value must come back to 1e-12
relative, and the stored exact zeros (odd parity, a channel with no
diagram) to 1e-15 absolute.  Demo 4's level A, which stores only summary
errors, is checked against Campbell's closed form
(``examples/demo4/poisson_noise.py``) instead, including the joint
cumulant of the four-point function.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

from validation import demo_cases as dc
from validation.hierarchy import moments
from validation.modelspec import Observable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "examples" / "demo4"))
import poisson_noise as nz  # noqa: E402


def _check(cases) -> int:
    assert cases, "no case selected"
    values = {key: moments(*key, tags)
              for key, tags in dc.unique_queries(cases).items()}
    bad = []
    for case in cases:
        got = sum(values[(case.model, case.observable)][t] for t in case.tags)
        tol = 1e-12 * abs(case.expected) if case.expected else 1e-15
        if not abs(got - case.expected) <= tol:
            bad.append(f"{case.source}: {got!r} != {case.expected!r}")
    assert not bad, "\n".join(bad[:20])
    return len(cases)


#: Stored values per demo part; together every value in the five results
#: files that holds a hierarchy value.
GROUPS = {"demo4": 44, "demo5 part A": 117, "demo5 part C": 560,
          "demo7 item a": 432, "demo7 item b": 132, "demo7 item c": 76,
          "demo7 item e": 276, "demo7 item d": 135}


@pytest.mark.parametrize("group", list(GROUPS))
def test_VD1_every_committed_hierarchy_value_is_reproduced(group):
    cases = [c for c in dc.all_cases() if c.group == group]
    assert _check(cases) == GROUPS[group]


def test_VD2_the_groups_cover_every_case():
    assert sum(GROUPS.values()) == len(dc.all_cases()) == 1772


@pytest.mark.parametrize("pulse", ["white", "exponential"])
def test_VD3_demo4_level_a_matches_campbells_closed_form(pulse):
    """At F = 0 the m-point function is ``K_R`` exactly: m = 3 at tag P3,
    m = 4 at tag P4, and the four-point joint cumulant at the zero tag
    (the Gaussian part) vanishes."""
    model = dc.demo4_model(pulse, with_f=False)
    p = nz.Params(pulse=pulse)
    labels = ("x", "y", "z", "w")
    xs = np.array([dc.POS_DEMO4[lab] for lab in labels])
    for m, tag in ((3, (1, 0)), (4, (0, 1))):
        quads = (list(itertools.product(range(2), repeat=3)) if m == 3
                 else [(0, 1, 1, 0), (1, 1, 1, 1), (0, 0, 1, 1), (1, 0, 0, 0)])
        for comps in quads:
            legs = list(zip(comps, labels))
            obs = Observable(legs, dc.POS_DEMO4, dc.T_DEMO4,
                             target="connected")
            got = moments(model, obs, [tag, (0, 0)])
            raw = moments(model, Observable(legs, dc.POS_DEMO4, dc.T_DEMO4),
                          [(0, 0)])[(0, 0)]
            want = float(nz.K_R(comps, xs[:m], np.full(m, dc.T_DEMO4), p)[0])
            assert got[tag] == pytest.approx(want, rel=1e-12, abs=0.0)
            assert abs(got[(0, 0)]) <= 1e-14 * max(1.0, abs(raw))
