"""The lowering onto sft-wick 0.4.2 (``validation/lower_v042.py``), run on the
current release.

The 0.4.2 replay itself needs the 0.4.2 wheel and is evidence, not a test.
The 0.4.2 API is a subset of the current one for what the lowering uses, so
here the same lowering runs on the current package, which has none of the
0.4.2 bugs, and must agree with the oracle: a wrong lowering would show
here before it could be blamed on 0.4.2.  It refuses what 0.4.2 lacks
(multiplicative noise, the connected target), and it never imports
``validation/lower.py`` or the worker, which the 0.4.2 process could not
load.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import validation.lower_v042 as lv
from tests.test_validation_imports import _imports
from validation.compare import compare_values, tolerance
from validation.draw import draw
from validation.fuzz import METHODS, features
from validation.hierarchy import moments_with_scale, tags_up_to

ROOT = Path(__file__).resolve().parents[1]


def _expressible(seed: int) -> bool:
    case = draw(seed)
    return (case.model.multiplicative is None
            and case.observable.target == "moment")


#: Pinned seeds of tests/test_validation_fuzz.py that 0.4.2 can express, and
#: 7, 20, 88 for an integrated leg, Gaussian and damped-cosine OU kernels,
#: and white and exponential Poisson pulses, which those seeds lack.
SEEDS = [s for s in (3, 7, 20, 31, 33, 36, 88, 90, 111, 114, 115, 126, 134,
                     146, 151, 152, 172, 202, 207, 214, 233, 266, 292)
         if _expressible(s)]


@pytest.mark.parametrize("seed", SEEDS)
def test_VR1_the_0_4_2_lowering_agrees_with_the_oracle_on_this_release(seed):
    case = draw(seed)
    assert case.model.multiplicative is None
    assert case.observable.target == "moment"
    tags = tags_up_to(case.model, case.max_weight)
    ref, scale = moments_with_scale(case.model, case.observable, tags)
    got = lv.Lowered(case.model, case.observable, case.max_weight,
                     case.flags).evaluate("gauss_legendre",
                                          **METHODS["gauss_legendre"])
    rows = compare_values(ref, got, tolerance("gauss_legendre", "oracle"),
                          scale=scale)
    assert all(r.ok for r in rows), [r for r in rows if not r.ok]
    assert any(sum(r.tag) and r.oracle for r in rows)


def test_VR2_what_0_4_2_lacks_is_refused():
    multiplicative = next(draw(s) for s in range(300)
                          if draw(s).model.multiplicative is not None)
    with pytest.raises(NotImplementedError, match="multiplicative"):
        lv.Lowered(multiplicative.model, multiplicative.observable, 1)
    connected = next(draw(s) for s in range(300)
                     if draw(s).observable.target == "connected"
                     and draw(s).model.multiplicative is None)
    with pytest.raises(NotImplementedError, match="connected"):
        lv.Lowered(connected.model, connected.observable, 1)


def test_VR3_the_0_4_2_lowering_imports_nothing_of_the_current_api():
    path = ROOT / "validation" / "lower_v042.py"
    found = {module for _, module in _imports(path)}
    assert "validation.lowering_parts" in found
    assert not {m for m in found
                if m in ("validation.lower", "validation.worker")
                or m.startswith(("validation.lower.", "validation.worker."))
                or m.endswith(".MultiplicativeImpulse")}, found
    names = {node.id for node in ast.walk(ast.parse(path.read_text()))
             if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(ast.parse(path.read_text()))
              if isinstance(node, ast.Attribute)}
    assert "MultiplicativeImpulse" not in names


def test_VR4_the_seeds_cover_every_feature_0_4_2_can_express():
    drawn = set().union(*(features(draw(s)) for s in range(300)
                          if _expressible(s)))
    covered = set().union(*(features(draw(s)) for s in SEEDS))
    assert drawn <= covered, sorted(drawn - covered)
