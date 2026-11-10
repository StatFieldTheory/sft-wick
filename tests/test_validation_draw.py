"""Properties of the fuzzer's sampler (``validation/draw.py``) over seeds
that hypothesis chooses.

Each property holds for every case :func:`validation.draw.draw` returns,
and none needs the oracle or the package, so each example costs a
millisecond.  The settings derandomize hypothesis, so the examples change
only with its version and with the modules already imported, whose
constants it also draws; a random search belongs to the fuzz runs, which are
evidence and not tests.  A failure prints the seed and the kind, and
``draw(seed, kind)`` gives the same case on any machine.  Checked:

* a seed gives one case, the same on a second draw, and the case survives
  the JSON round trip of its model and observable; a forced kind is kept;
* the flags agree with the model (``iso_R`` exactly when the rates are
  equal, ``iso_C`` only with ``diag_C`` and ``iso_R``) and with the kind
  (every flag on an isotropic model, ``diag_C`` on a diagonal one), and
  Poisson noise is drawn as the sampler says;
* the observable is one the oracle and the package both take: distinct
  labels, components in range, at most one pinned time inside the window,
  an integrated leg not pinned, a connected target with two legs or more,
  positions apart from the origin unless multiplicative noise makes them
  one point, and two or more dimensions for a Legendre kernel;
* the largest weight is 1, 2 or 3, and 3 only on a small model, and the
  tags up to it are the ones of that weight or less.
"""
from __future__ import annotations

import json
import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from validation.draw import KINDS, draw
from validation.hierarchy import tags_up_to
from validation.modelspec import (model_from_dict, model_to_dict,
                                  observable_from_dict, observable_to_dict)

SEEDS = st.integers(min_value=0, max_value=2 ** 63 - 1)
KIND = st.sampled_from([None, *KINDS])
#: Derandomized, with no example database; the examples still change with
#: the hypothesis version and with the modules already imported.
DERANDOMIZED = settings(max_examples=300, derandomize=True, database=None,
                         deadline=None,
                         suppress_health_check=[HealthCheck.too_slow])


def _blob(case) -> str:
    return json.dumps([model_to_dict(case.model),
                       observable_to_dict(case.observable), case.max_weight,
                       case.flags], sort_keys=True)


@DERANDOMIZED
@given(seed=SEEDS, kind=KIND)
def test_VS1_a_seed_gives_one_case_and_it_survives_json(seed, kind):
    case = draw(seed, kind)
    assert _blob(draw(seed, kind)) == _blob(case)
    if kind is not None:
        assert case.kind == kind
    model = model_from_dict(json.loads(json.dumps(model_to_dict(case.model))))
    obs = observable_from_dict(json.loads(json.dumps(
        observable_to_dict(case.observable))))
    assert model_to_dict(model) == model_to_dict(case.model)
    assert observable_to_dict(obs) == observable_to_dict(case.observable)


@DERANDOMIZED
@given(seed=SEEDS, kind=KIND)
def test_VS2_the_flags_agree_with_the_model_and_its_kind(seed, kind):
    case = draw(seed, kind)
    model, flags = case.model, case.flags
    assert flags["diag_R"]
    assert flags["iso_R"] == (len(set(model.gamma)) == 1)
    if flags["iso_C"]:
        assert flags["diag_C"] and flags["iso_R"]
    if case.kind == "isotropic":
        assert all(flags.values())
    if case.kind == "diagonal":
        assert flags["diag_C"]
    if model.poisson is not None:
        assert case.kind == "general"
        if model.poisson.pulse == "white":
            assert model.multiplicative is None
        else:
            assert set(model.poisson.orders) <= {3}


@DERANDOMIZED
@given(seed=SEEDS, kind=KIND)
def test_VS3_the_observable_is_one_both_sides_take(seed, kind):
    case = draw(seed, kind)
    model, obs = case.model, case.observable
    labels = [leg.label for leg in obs.legs]
    assert 1 <= len(labels) <= 3 and len(set(labels)) == len(labels)
    assert set(labels) <= set("xyz")
    assert all(0 <= leg.comp < model.n_components for leg in obs.legs)
    assert 0.05 <= abs(model.t_min) <= 0.6
    span = obs.t_final - model.t_min
    assert 0.8 <= span <= 1.8
    times = dict(obs.times)
    assert len(times) <= 1 and (not times or len(labels) >= 2)
    for t in times.values():
        assert model.t_min < t < obs.t_final
    assert len(obs.integrate_over) <= 1
    assert not set(obs.integrate_over) & set(times)
    if obs.target == "connected":
        assert len(labels) >= 2
    points = [np.asarray(p) for _, p in obs.positions]
    if any(blk.spatial.kind == "legendre" for blk in model.ou):
        assert len(points[0]) >= 2
    if model.multiplicative is not None:
        assert all(np.array_equal(p, points[0]) for p in points)
    else:
        assert all(np.linalg.norm(p) > 0.2 for p in points)


@DERANDOMIZED
@given(seed=SEEDS, kind=KIND)
def test_VS4_the_largest_weight_and_its_tags(seed, kind):
    case = draw(seed, kind)
    model, w = case.model, case.max_weight
    assert w in (1, 2, 3)
    if w == 3:
        assert len(case.observable.legs) <= 2 and model.cubic is None
        assert len(model.channels()) <= 3
    n = len(model.channels())
    tags = tags_up_to(model, w)
    assert len(tags) == len(set(tags)) == math.comb(n + w, w)
    assert all(len(t) == n and sum(t) <= w and min(t, default=0) >= 0
               for t in tags)
