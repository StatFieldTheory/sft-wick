"""A C lookup outside its table is refused at every level.

A C table covers ``[t_min, t_max]`` in time and, on a full spatial grid,
``[0, r_max]`` in separation or ``[-x_max, x_max]`` in position.  Past those
limits a lazy table clamped to the edge and a full-grid table extrapolated,
and both returned a number with no error (up to 28 orders of magnitude off,
with the wrong sign, at ``r = 50``).

TL1 -- at each limit and just inside it the table agrees with the reference.
TL2 -- a time just past the table, and far past it, is refused at L1 on every
       table kind, on both sides, for ``t_final`` and for a pinned point.
TL3 -- ``t_final`` is refused only where it is read: with every external
       point pinned it is not, and the number does not change.
TL4 -- a separation past ``r_max`` is refused; scalars and vectors, both
       orders of the points, and rounding at the limit is not.
TL5 -- a position past ``x_max`` is refused on either side and at either
       point of the general full grid.
TL6 -- no limit where the lookup has none: cos on a rotation table, any
       position on a lazy table, anything under ``c_closed_form_only``.
TL7 -- ``C_at_batch``, ``C_value`` and the legacy batch accessors refuse too,
       so a direct L0 caller is covered.
TL8 -- ``Expansion.sweep`` refuses before its first grid point.
TL9 -- ``sft-wick validate`` and ``run`` stop with status 2 and one line
       naming the dotted path, before the expansion is paid for.
TL10 -- the limits themselves pass ``validate``, and a run at them agrees with
        the reference.

Reference: ``scipy.integrate.dblquad`` of the integral that defines C, which
imports nothing from sft-wick.  The parameters are asymmetric: unequal rates,
``t_min != 0``, and a general kernel that changes under ``x -> -x``.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.integrate import dblquad

import sft_wick as sw
import sft_wick.evaluate as ev
from sft_wick.workflow import specs as sp
from sft_wick.workflow.cli import main
from sft_wick.workflow.config import (
    ConfigError,
    check_before_run,
    load_workflow_config,
    run_workflow,
)
from sft_wick.workflow.propagators import propagators_from_cache

GAMMA = (0.7, 1.9)
LAM, SIGMA_T, SIGMA_X = 0.35, 0.45, 0.8
LEGENDRE = (0.6, 0.3, 0.1)
T_MIN, T_MAX = 0.4, 2.3
R_MAX, X_MAX = 1.2, 1.5
#: Table nodes, so that a value read at one is not an interpolation.
N_GRID_T, N_GRID_R, N_GRID_X, N_GRID_COS = 7, 9, 4, 5
#: Just past a limit, relative to it: far above rounding, far below any
#: distance a table could resolve.
DELTA = 1e-6
TABLE_ERROR = 1e-6
REFUSED = "propagator table"
#: The up-front check, as opposed to the lookup, which would refuse as well.
AT_L1 = r"Expansion\.evaluate: .*propagator table"

_NODES_T = np.linspace(T_MIN, T_MAX, N_GRID_T)
T_EARLIER, T_NODE = float(_NODES_T[2]), float(_NODES_T[3])
_NODE_X = X_MAX / 3.0  # a node of the 4-point general grid
_MID_R = R_MAX / 2.0  # a node of the 9-point separation grid


def _kappa_t(dt: float) -> float:
    return LAM * math.exp(-abs(dt) / SIGMA_T)


@lru_cache(maxsize=None)
def _time_integral(t1: float, t2: float, gamma: float = GAMMA[0]) -> float:
    """``int_{t_min}^{t1} da int_{t_min}^{t2} db R(t1,a) k(a-b) R(t2,b)``,
    split on the ``a = b`` cusp of the kernel."""
    if t1 <= T_MIN or t2 <= T_MIN:
        return 0.0

    def f(b: float, a: float) -> float:
        return (
            math.exp(-gamma * (t1 - a))
            * _kappa_t(a - b)
            * math.exp(-gamma * (t2 - b))
        )

    opts = dict(epsabs=1e-10, epsrel=1e-10)
    below, _ = dblquad(f, T_MIN, t1, T_MIN, lambda a: min(a, t2), **opts)
    above, _ = dblquad(f, T_MIN, t1, lambda a: min(a, t2), t2, **opts)
    return below + above


def _translation_kernel(r: float) -> float:
    return math.exp(-abs(r) / SIGMA_X)


def _rotation_kernel(cos: float) -> float:
    p2 = 0.5 * (3.0 * cos * cos - 1.0)
    return LEGENDRE[0] + LEGENDRE[1] * cos + LEGENDRE[2] * p2


def _general_kernel(a: float, b: float) -> float:
    return math.exp(-abs(a - b) / SIGMA_X) * (1.0 + 0.3 * a) * (1.0 + 0.3 * b)


class _GeneralKappa2:
    """Depends on the two positions apart, not on their difference alone."""

    def __call__(self, n1, t1, n2, t2):
        a = float(np.asarray(n1).ravel()[0])
        b = float(np.asarray(n2).ravel()[0])
        return np.eye(2) * _kappa_t(t1 - t2) * _general_kernel(a, b)


def _system(kappa2, vertices=()) -> sw.System:
    return sw.System(
        field=sp.FieldSpec("phi", n_components=2),
        linear=sp.DiagonalA(gamma=list(GAMMA)),
        noise=sp.GaussianNoise(kappa2=kappa2),
        vertices=tuple(vertices),
        t_min=T_MIN,
    )


def _translation(vertices=()) -> sw.System:
    return _system(
        sp.SeparableTranslation(
            temporal=sp.ExponentialTemporal(lam=LAM, sigma_t=SIGMA_T),
            spatial=sp.ExponentialSpatial(sigma_x=SIGMA_X),
        ),
        vertices,
    )


def _rotation() -> sw.System:
    return _system(
        sp.SeparableRotation(
            temporal=sp.ExponentialTemporal(lam=LAM, sigma_t=SIGMA_T),
            angular=sp.LegendreAngular(coeffs=list(LEGENDRE)),
        )
    )


def _general() -> sw.System:
    return _system(sp.GeneralKappa2(fn=_GeneralKappa2()))


#: kind -> (system, table options, in-range positions, spatial reference)
_U = np.array([1.0, 0.0])
_V = np.array([0.5, math.sqrt(3.0) / 2.0])  # cos = 0.5, a node of the cos grid
KINDS = {
    "translation-lazy": (_translation, {}, (0.0, 0.3), _translation_kernel(0.3)),
    "translation-full": (
        _translation,
        {"r_max": R_MAX, "n_grid_r": N_GRID_R},
        (0.0, _MID_R),
        _translation_kernel(_MID_R),
    ),
    "rotation-lazy": (_rotation, {}, (_U, _V), _rotation_kernel(0.5)),
    "rotation-full": (
        _rotation,
        {"n_grid_cos": N_GRID_COS},
        (_U, _V),
        _rotation_kernel(0.5),
    ),
    "general-lazy": (
        _general,
        {},
        (_NODE_X, -_NODE_X),
        _general_kernel(_NODE_X, -_NODE_X),
    ),
    "general-full": (
        _general,
        {"x_max": X_MAX, "n_grid_x": N_GRID_X},
        (_NODE_X, -_NODE_X),
        _general_kernel(_NODE_X, -_NODE_X),
    ),
}
FULL_KINDS = ("translation-full", "rotation-full", "general-full")


@lru_cache(maxsize=None)
def _built(kind: str):
    make, options, _positions, _spatial = KINDS[kind]
    system = make()
    expansion = system.expand(("phi_a(x)", "phi_a(y)"), orders=[0])
    props = system.propagators(
        t_max=T_MAX,
        n_grid_t=N_GRID_T,
        c_closed_form=None,
        c_method="gauss_legendre",
        c_n_gauss=24,
        **options,
    )
    return expansion, props


def _value(expansion, props, positions, t_final, external_times=None):
    x, y = positions
    return expansion.evaluate(
        props,
        positions={"x": x, "y": y},
        t_final=t_final,
        component_pair=(0, 0),
        orders=[0],
        external_times=external_times,
        n_samples=64,
    ).total


COUPLING = np.array([[[0.2, -0.7], [0.4, 0.1]], [[0.9, 0.3], [-0.5, 0.6]]])


# ---------------------------------------------------------------------------
# TL1: at each limit and just inside it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "time",
    [T_MAX, T_MAX * (1.0 - DELTA), T_NODE, T_MIN * (1.0 + DELTA), T_MIN],
    ids=["t_max", "t_max-delta", "inside", "t_min+delta", "t_min"],
)
def test_TL1_time_at_the_limit_agrees_with_the_reference(kind, time):
    _make, _options, positions, spatial = KINDS[kind]
    expansion, props = _built(kind)
    got = _value(expansion, props, positions, time)
    reference = spatial * _time_integral(time, time)
    # C vanishes at t_min, so a relative comparison there says nothing; the
    # floor is one cell's slope across DELTA.  Every other cell is relative.
    floor = 1e-7 if time < T_MIN * (1.0 + 2.0 * DELTA) else 0.0
    assert got == pytest.approx(reference, rel=TABLE_ERROR, abs=floor)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("limit,towards", [(T_MAX, np.inf), (T_MIN, -np.inf)],
                         ids=["t_max", "t_min"])
def test_TL1_rounding_at_a_time_limit_is_not_refused(kind, limit, towards):
    _make, _options, positions, _spatial = KINDS[kind]
    expansion, props = _built(kind)
    at = _value(expansion, props, positions, limit)
    past = _value(expansion, props, positions, np.nextafter(limit, towards))
    assert past == pytest.approx(at, rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# TL2: a time outside the table
# ---------------------------------------------------------------------------

PAST_TIMES = [
    T_MAX * (1.0 + DELTA),
    2.0 * T_MAX,
    5.0,
    50.0,
    1e6,
    math.inf,
    T_MIN * (1.0 - DELTA),
    0.0,
    -0.5,
    -1e6,
    -math.inf,
]
PAST_TIME_IDS = [
    "t_max+delta",
    "twice-t_max",
    "5",
    "50",
    "1e6",
    "inf",
    "t_min-delta",
    "zero",
    "minus-half",
    "minus-1e6",
    "minus-inf",
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("time", PAST_TIMES, ids=PAST_TIME_IDS)
def test_TL2_t_final_past_the_table_is_refused(kind, time):
    _make, _options, positions, _spatial = KINDS[kind]
    expansion, props = _built(kind)
    with pytest.raises(ValueError, match=AT_L1):
        _value(expansion, props, positions, time)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("time", PAST_TIMES, ids=PAST_TIME_IDS)
@pytest.mark.parametrize("point", ["x", "y"])
def test_TL2_a_pinned_time_past_the_table_is_refused(kind, time, point):
    _make, _options, positions, _spatial = KINDS[kind]
    expansion, props = _built(kind)
    times = {"x": 1.1, "y": T_EARLIER, point: time}
    with pytest.raises(ValueError, match=AT_L1):
        _value(expansion, props, positions, 1.2, external_times=times)


def test_TL2_the_time_refused_is_named():
    _make, _options, positions, _spatial = KINDS["translation-full"]
    expansion, props = _built("translation-full")
    with pytest.raises(ValueError, match=r"'t_final': 5\.0.*t_max=2\.3"):
        _value(expansion, props, positions, 5.0)
    with pytest.raises(ValueError, match=r"'y': 0\.1.*t_min=0\.4"):
        _value(expansion, props, positions, 1.0,
               external_times={"x": 1.0, "y": 0.1})


# ---------------------------------------------------------------------------
# TL3: t_final is refused only where it is read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["translation-lazy", "translation-full"])
@pytest.mark.parametrize("t_final", [T_MAX + 0.5, 50.0, 0.0],
                         ids=["past-t_max", "far-past", "below-t_min"])
def test_TL3_t_final_is_not_read_when_every_point_is_pinned(kind, t_final):
    _make, _options, positions, spatial = KINDS[kind]
    expansion, props = _built(kind)
    pinned = {"x": T_NODE, "y": T_EARLIER}
    got = _value(expansion, props, positions, t_final, external_times=pinned)
    assert got == _value(expansion, props, positions, 2.0, external_times=pinned)
    assert got == pytest.approx(
        spatial * _time_integral(T_NODE, T_EARLIER), rel=TABLE_ERROR, abs=0.0
    )


def test_TL3_order_two_is_unchanged_too():
    """At order 2 the vertex time is bounded by the pinned points, so a
    ``t_final`` past the table reads nothing past it."""
    coupling = np.array([[[0.2, -0.7], [0.4, 0.1]], [[T_EARLIER, 0.3], [-0.5, 0.6]]])
    system = _translation([sp.LocalVertex("F", coupling)])
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2])
    props = system.propagators(t_max=T_MAX, n_grid_t=11)
    values = {
        t_final: expansion.evaluate(
            props,
            positions={"x": 0.0, "y": 0.4},
            t_final=t_final,
            component_pair=(0, 1),
            external_times={"x": 2.0, "y": 1.2},
            n_samples=2**8,
            seed=3,
        ).total
        for t_final in (2.1, T_MAX + 0.5, 9.0)
    }
    assert len(set(values.values())) == 1
    assert values[2.1] != 0.0


def test_TL3_one_free_point_reads_t_final():
    _make, _options, positions, _spatial = KINDS["translation-full"]
    expansion, props = _built("translation-full")
    with pytest.raises(ValueError, match=AT_L1):
        _value(expansion, props, positions, T_MAX + 0.5,
               external_times={"y": 1.0})


# ---------------------------------------------------------------------------
# TL4: separation past r_max
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "r",
    [0.0, _MID_R, R_MAX, R_MAX * (1.0 - DELTA)],
    ids=["zero", "inside", "r_max", "r_max-delta"],
)
@pytest.mark.parametrize("flip", [False, True], ids=["x<y", "x>y"])
def test_TL4_separation_at_the_limit_agrees_with_the_reference(r, flip):
    expansion, props = _built("translation-full")
    positions = (r, 0.0) if flip else (0.0, r)
    got = _value(expansion, props, positions, T_NODE)
    reference = _translation_kernel(r) * _time_integral(T_NODE, T_NODE)
    assert got == pytest.approx(reference, rel=TABLE_ERROR, abs=0.0)


def test_TL4_a_vector_separation_at_the_limit_agrees_with_the_reference():
    expansion, props = _built("translation-full")
    y = R_MAX * np.array([2.0, 3.0, 6.0]) / 7.0  # norm R_MAX
    got = _value(expansion, props, (np.zeros(3), y), T_NODE)
    reference = _translation_kernel(R_MAX) * _time_integral(T_NODE, T_NODE)
    assert got == pytest.approx(reference, rel=TABLE_ERROR, abs=0.0)


@pytest.mark.parametrize("r", [np.nextafter(R_MAX, np.inf), R_MAX * (1.0 + 5e-13)],
                         ids=["one-ulp", "inside-the-slack"])
def test_TL4_rounding_at_the_limit_is_not_refused(r):
    expansion, props = _built("translation-full")
    at = _value(expansion, props, (0.0, R_MAX), T_NODE)
    assert _value(expansion, props, (0.0, r), T_NODE) == pytest.approx(
        at, rel=1e-9, abs=0.0
    )


@pytest.mark.parametrize(
    "r",
    [R_MAX * (1.0 + DELTA), 1.5, 3.0, 50.0, 1e6, math.inf],
    ids=["r_max+delta", "1.5", "3", "50", "1e6", "inf"],
)
@pytest.mark.parametrize("flip", [False, True], ids=["x<y", "x>y"])
def test_TL4_separation_past_the_table_is_refused(r, flip):
    expansion, props = _built("translation-full")
    positions = (r, 0.0) if flip else (0.0, r)
    with pytest.raises(ValueError, match=AT_L1):
        _value(expansion, props, positions, T_NODE)


def test_TL4_a_vector_separation_past_the_table_is_refused():
    expansion, props = _built("translation-full")
    y = (R_MAX + 0.1) * np.array([2.0, 3.0, 6.0]) / 7.0
    with pytest.raises(ValueError, match=AT_L1):
        _value(expansion, props, (np.zeros(3), y), T_NODE)


def test_TL4_an_omitted_position_is_refused():
    """A point left out next to one away from 0 is refused before a lookup."""
    expansion, props = _built("translation-full")
    with pytest.raises(ValueError, match=r"no position for \['y'\]"):
        expansion.evaluate(props, positions={"x": 5.0}, t_final=T_NODE,
                           component_pair=(0, 0), orders=[0])


def test_TL4_two_far_points_a_short_way_apart_are_in_range():
    """``r`` is what the table reads, not either position."""
    expansion, props = _built("translation-full")
    got = _value(expansion, props, (100.0, 100.0 + _MID_R), T_NODE)
    reference = _translation_kernel(_MID_R) * _time_integral(T_NODE, T_NODE)
    assert got == pytest.approx(reference, rel=TABLE_ERROR, abs=0.0)


# ---------------------------------------------------------------------------
# TL5: position past x_max
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x,y,rel",
    [
        (X_MAX, -X_MAX, TABLE_ERROR),
        (-X_MAX, X_MAX, TABLE_ERROR),
        (X_MAX, X_MAX, TABLE_ERROR),
        (-X_MAX, -X_MAX, TABLE_ERROR),
        # Linear interpolation over a cell of width 1, DELTA from a node.
        (X_MAX * (1.0 - DELTA), -_NODE_X, 1e-5),
        (-_NODE_X, -X_MAX * (1.0 - DELTA), 1e-5),
    ],
    ids=["+-", "-+", "++", "--", "x_max-delta", "-x_max+delta"],
)
def test_TL5_position_at_the_limit_agrees_with_the_reference(x, y, rel):
    expansion, props = _built("general-full")
    got = _value(expansion, props, (x, y), T_NODE)
    reference = _general_kernel(x, y) * _time_integral(T_NODE, T_NODE)
    assert got == pytest.approx(reference, rel=rel, abs=0.0)


@pytest.mark.parametrize(
    "x,y",
    [
        (X_MAX * (1.0 + DELTA), 0.0),
        (-X_MAX * (1.0 + DELTA), 0.0),
        (0.0, X_MAX * (1.0 + DELTA)),
        (0.0, -X_MAX * (1.0 + DELTA)),
        (1.6, 0.0),
        (-1.6, 0.0),
        (3.0, 3.0),
        (-3.0, -3.0),
        (1e6, 0.0),
        (0.0, -1e6),
        (math.inf, 0.0),
        (0.0, -math.inf),
    ],
    ids=[
        "x-high-delta",
        "x-low-delta",
        "y-high-delta",
        "y-low-delta",
        "x-high",
        "x-low",
        "both-high",
        "both-low",
        "x-1e6",
        "y-minus-1e6",
        "x-inf",
        "y-minus-inf",
    ],
)
def test_TL5_position_past_the_table_is_refused(x, y):
    expansion, props = _built("general-full")
    with pytest.raises(ValueError, match=AT_L1):
        _value(expansion, props, (x, y), T_NODE)


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["x_max", "-x_max"])
def test_TL5_rounding_at_the_limit_is_not_refused(sign):
    expansion, props = _built("general-full")
    at = _value(expansion, props, (sign * X_MAX, 0.0), T_NODE)
    past = float(np.nextafter(sign * X_MAX, sign * np.inf))
    assert _value(expansion, props, (past, 0.0), T_NODE) == pytest.approx(
        at, rel=1e-9, abs=0.0
    )


# ---------------------------------------------------------------------------
# TL6: no limit where the lookup has none
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["rotation-lazy", "rotation-full"])
@pytest.mark.parametrize(
    "x,y,cos",
    [
        (_U, _U, 1.0),
        (_U, -_U, -1.0),
        (40.0 * _U, 7.0 * _V, 0.5),
        (1e6 * _U, -1e-6 * _V, -0.5),
    ],
    ids=["parallel", "antipodal", "long-vectors", "extreme-lengths"],
)
def test_TL6_rotation_has_no_position_limit(kind, x, y, cos):
    expansion, props = _built(kind)
    got = _value(expansion, props, (x, y), T_NODE)
    reference = _rotation_kernel(cos) * _time_integral(T_NODE, T_NODE)
    assert got == pytest.approx(reference, rel=TABLE_ERROR, abs=0.0)


def test_TL6_a_lazy_table_has_no_separation_limit():
    expansion, props = _built("translation-lazy")
    got = _value(expansion, props, (0.0, 50.0), T_NODE)
    reference = _translation_kernel(50.0) * _time_integral(T_NODE, T_NODE)
    assert got == pytest.approx(reference, rel=TABLE_ERROR, abs=0.0)


def test_TL6_a_lazy_general_table_has_no_position_limit():
    expansion, props = _built("general-lazy")
    got = _value(expansion, props, (3.0, -3.0), T_NODE)
    reference = _general_kernel(3.0, -3.0) * _time_integral(T_NODE, T_NODE)
    assert got == pytest.approx(reference, rel=TABLE_ERROR, abs=0.0)


@pytest.mark.parametrize(
    "positions,t_final",
    [((0.0, 0.5), 5.0), ((0.0, 50.0), T_NODE), ((0.0, 50.0), 5.0)],
    ids=["time", "separation", "both"],
)
def test_TL6_closed_form_only_has_no_table_and_no_limit(positions, t_final):
    system = _translation()
    expansion = system.expand(("phi_a(x)", "phi_a(y)"), orders=[0])
    props = system.propagators(t_max=T_MAX, c_closed_form_only=True)
    got = _value(expansion, props, positions, t_final)
    reference = (
        _translation_kernel(positions[1]) * _time_integral(t_final, t_final)
    )
    assert got == pytest.approx(reference, rel=TABLE_ERROR, abs=0.0)


def test_TL6_closed_form_only_is_exact_below_t_min():
    system = _translation()
    expansion = system.expand(("phi_a(x)", "phi_a(y)"), orders=[0])
    props = system.propagators(t_max=T_MAX, c_closed_form_only=True)
    assert _value(expansion, props, (0.0, 0.5), 0.1) == 0.0


# ---------------------------------------------------------------------------
# TL7: the L0 accessors
# ---------------------------------------------------------------------------


def _batch(cache, t1, t2, positions):
    x, y = positions
    return cache.C_at_batch(np.asarray(t1), np.asarray(t2), x, y)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("bad", [T_MAX + 0.5, T_MIN - 0.1], ids=["high", "low"])
@pytest.mark.parametrize("side", [0, 1], ids=["t1", "t2"])
def test_TL7_C_at_batch_refuses_one_bad_sample_in_a_batch(kind, bad, side):
    _make, _options, positions, _spatial = KINDS[kind]
    _expansion, props = _built(kind)
    times = [np.array([1.0, 1.1, 1.2]), np.array([1.2, 1.1, 1.0])]
    times[side][1] = bad
    with pytest.raises(ValueError, match=REFUSED):
        _batch(props.cache, times[0], times[1], positions)


@pytest.mark.parametrize("kind", KINDS)
def test_TL7_C_at_batch_reads_the_limits_and_an_empty_batch(kind):
    _make, _options, positions, _spatial = KINDS[kind]
    _expansion, props = _built(kind)
    got = _batch(props.cache, [T_MIN, T_MAX, T_MAX], [T_MIN, T_MAX, T_MIN],
                 positions)
    assert np.isfinite(got).all()
    assert len(_batch(props.cache, [], [], positions)) == 0


def test_TL7_C_at_batch_refuses_a_separation_past_the_table():
    _expansion, props = _built("translation-full")
    with pytest.raises(ValueError, match=REFUSED):
        _batch(props.cache, [1.0], [1.0], (0.0, R_MAX * (1.0 + DELTA)))


@pytest.mark.parametrize("x,y", [(1.6, 0.0), (0.0, -1.6)], ids=["high", "low"])
def test_TL7_C_at_batch_refuses_a_position_past_the_table(x, y):
    _expansion, props = _built("general-full")
    with pytest.raises(ValueError, match=REFUSED):
        _batch(props.cache, [1.0], [1.0], (x, y))


@pytest.mark.parametrize("kind", FULL_KINDS)
def test_TL7_C_value_refuses_a_time_past_a_spatial_table(kind):
    _make, _options, positions, _spatial = KINDS[kind]
    _expansion, props = _built(kind)
    with pytest.raises(ValueError, match=REFUSED):
        props.cache.C_value(positions[0], T_MAX + 0.5, positions[1], 1.0)


def _legacy_cache(diag_C: bool):
    model = _translation().build_propagator_model(diag_C=diag_C)
    cache = ev.PropagatorCache(model, homogeneity="translation")
    cache.precompute_C_table(t_max=T_MAX, n_grid=N_GRID_T)
    return cache


@pytest.mark.parametrize("method,diag_C", [("C_diagonal_batch", True),
                                           ("C_matrix_batch", False)])
def test_TL7_a_time_only_table_refuses_too(method, diag_C):
    cache = _legacy_cache(diag_C)
    read = getattr(cache, method)
    for t in (T_MAX, T_MIN):
        assert np.isfinite(read(np.array([t]), np.array([t]))).all()
    for t in (T_MAX + 0.5, T_MIN - 0.1):
        with pytest.raises(ValueError, match=REFUSED):
            read(np.array([t]), np.array([1.0]))


def test_TL7_a_time_only_table_still_falls_back_in_C_value():
    """``C_value`` on a legacy table goes to direct quadrature past the
    table, which is correct, so it stays accepted."""
    cache = _legacy_cache(True)
    got = cache.C_value(0.0, 5.0, 0.0, 5.0)[0, 0]
    assert got == pytest.approx(_time_integral(5.0, 5.0), rel=1e-5, abs=0.0)


def test_TL7_L1_leaves_a_time_only_table_to_its_accessors():
    """Order 0 reads ``C_value``, which is right past the table, so L1 must
    not refuse it; order 2 reads the batch accessor, which refuses."""
    props = propagators_from_cache(_legacy_cache(True))
    expansion = _translation().expand(("phi_a(x)", "phi_a(y)"), orders=[0])
    got = _value(expansion, props, (0.0, 0.0), 3.0)
    assert got == pytest.approx(_time_integral(3.0, 3.0), rel=1e-5, abs=0.0)
    cubic = _translation([sp.LocalVertex("F", COUPLING)]).expand(
        ("phi_a(x)", "phi_b(y)"), orders=[2]
    )
    with pytest.raises(ValueError, match="C lookup at t"):
        cubic.evaluate(props, positions={"x": 0.0, "y": 0.0}, t_final=3.0,
                       component_pair=(0, 1), n_samples=2**5)


@pytest.mark.parametrize("options", [{}, {"r_max": R_MAX, "n_grid_r": N_GRID_R}],
                         ids=["lazy", "full"])
def test_TL7_clear_cache_keeps_the_limits_of_a_spatial_table(options):
    """It clears the time-only table and leaves the spatial one, so the
    range must stay for as long as that table can be read."""
    system = _translation()
    expansion = system.expand(("phi_a(x)", "phi_a(y)"), orders=[0])
    props = system.propagators(t_max=T_MAX, n_grid_t=N_GRID_T, **options)
    before = _value(expansion, props, (0.0, 0.5), T_NODE)
    rule = props.cache.c_method_resolved
    props.cache.clear_cache()
    assert props.cache.c_method_resolved == rule
    assert _value(expansion, props, (0.0, 0.5), T_NODE) == before
    with pytest.raises(ValueError, match=AT_L1):
        _value(expansion, props, (0.0, 0.5), T_MAX + 0.5)
    with pytest.raises(ValueError, match=REFUSED):
        _batch(props.cache, [T_MAX + 0.5], [1.0], (0.0, 0.5))


def test_TL7_clear_cache_still_empties_a_cache_with_no_spatial_table():
    cache = _legacy_cache(True)
    assert cache.c_method_resolved != "auto"
    cache.clear_cache()
    assert cache._c_table_range is None
    assert cache.c_method_resolved == "auto"


# ---------------------------------------------------------------------------
# TL8: the sweep refuses before its first grid point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,grid",
    [
        ("translation-full", dict(t_final_grid=[1.0, T_MAX + 0.5])),
        ("translation-lazy", dict(t_final_grid=[1.0, T_MIN - 0.1])),
        (
            "translation-full",
            dict(t_final_grid=[1.0],
                 external_times_grid={"x": [1.0, T_MAX + 0.5], "y": [1.0]}),
        ),
        (
            "translation-full",
            dict(positions_grid={"x": [0.0], "y": [0.5, R_MAX + 0.5]}),
        ),
        ("general-full", dict(positions_grid={"x": [0.5, -1.6], "y": [0.0]})),
    ],
    ids=["t_final-high", "t_final-low", "external-time", "separation", "position"],
)
def test_TL8_sweep_refuses_before_its_first_point(monkeypatch, kind, grid):
    import sft_wick.evaluate as module

    def unreachable(*args, **kwargs):
        raise AssertionError("a grid point was integrated")

    monkeypatch.setattr(module, "integrate_diagrams", unreachable)
    expansion, props = _built(kind)
    request = dict(
        positions_grid={"x": [0.0], "y": [0.5]},
        t_final_grid=[1.0],
        component_pairs=[(0, 0)],
        orders=[0],
    )
    request.update(grid)
    with pytest.raises(ValueError, match=REFUSED):
        expansion.sweep(props, **request)


def test_TL8_sweep_with_every_point_pinned_keeps_its_t_final_column():
    expansion, props = _built("translation-full")
    result = expansion.sweep(
        props,
        positions_grid={"x": [0.0], "y": [0.5]},
        t_final_grid=[T_MAX + 0.5],
        external_times_grid={"x": [T_NODE], "y": [T_EARLIER]},
        component_pairs=[(0, 0)],
        orders=[0],
    )
    assert len(result.rows) == 1


# ---------------------------------------------------------------------------
# TL9: validate and run stop before the expansion
# ---------------------------------------------------------------------------


def _doc() -> dict:
    return {
        "system": {
            "t_min": T_MIN,
            "field": {"name": "phi", "n_components": 2},
            "linear": {"type": "diagonal", "gamma": list(GAMMA)},
            "noise": {
                "kappa2": {
                    "type": "separable_translation",
                    "temporal": {
                        "type": "exponential",
                        "lam": LAM,
                        "sigma_t": SIGMA_T,
                    },
                    "spatial": {"type": "exponential", "sigma_x": SIGMA_X},
                }
            },
        },
        "expand": {"observable": ["phi_a(x)", "phi_b(y)"], "orders": [0]},
        "propagators": {
            "t_max": T_MAX,
            "n_grid_t": N_GRID_T,
            "r_max": R_MAX,
            "n_grid_r": N_GRID_R,
        },
        "sweep": {
            "positions_grid": {"x": [0.0], "y": [0.0, _MID_R]},
            "t_final_grid": [1.0, T_MAX],
            "component_pairs": [[0, 0], [1, 1]],
            "n_samples": 64,
        },
        "output": [{"type": "table", "format": "markdown"}],
    }


def _general_doc() -> dict:
    doc = _doc()
    props = doc["propagators"]
    del props["r_max"], props["n_grid_r"]
    props.update(homogeneity="general", x_max=X_MAX, n_grid_x=N_GRID_X)
    return doc


def _lazy_doc() -> dict:
    doc = _doc()
    del doc["propagators"]["r_max"], doc["propagators"]["n_grid_r"]
    return doc


def _rotation_doc() -> dict:
    doc = _lazy_doc()
    doc["system"]["noise"]["kappa2"] = {
        "type": "separable_rotation",
        "temporal": {"type": "exponential", "lam": LAM, "sigma_t": SIGMA_T},
        "angular": {"type": "legendre", "coeffs": list(LEGENDRE)},
    }
    doc["propagators"]["n_grid_cos"] = N_GRID_COS
    doc["sweep"]["positions_grid"] = {"x": [[1.0, 0.0]], "y": [[40.0, 7.0]]}
    return doc


def _with(doc: dict, **changes) -> dict:
    """``doc`` with dotted-path keys (``__`` for the dots) set."""
    for path, value in changes.items():
        *blocks, key = path.split("__")
        node = doc
        for block in blocks:
            node = node[block]
        node[key] = value
    return doc


#: case -> (config, dotted path, reason)
REFUSALS = {
    "t_final-high": (
        _with(_doc(), sweep__t_final_grid=[1.0, 5.0]),
        "sweep.t_final_grid[1]",
        "t_max",
    ),
    "t_final-just-high": (
        _with(_doc(), sweep__t_final_grid=[T_MAX * (1.0 + DELTA)]),
        "sweep.t_final_grid[0]",
        "t_max",
    ),
    "t_final-extreme": (
        _with(_doc(), sweep__t_final_grid=[1.0, 1e6]),
        "sweep.t_final_grid[1]",
        "t_max",
    ),
    "t_final-low": (
        _with(_doc(), sweep__t_final_grid=[0.1, 1.0]),
        "sweep.t_final_grid[0]",
        "t_min",
    ),
    "t_final-lazy": (
        _with(_lazy_doc(), sweep__t_final_grid=[5.0]),
        "sweep.t_final_grid[0]",
        "t_max",
    ),
    "t_final-rotation": (
        _with(_rotation_doc(), sweep__t_final_grid=[5.0]),
        "sweep.t_final_grid[0]",
        "t_max",
    ),
    "t_final-general": (
        _with(_general_doc(), sweep__t_final_grid=[5.0]),
        "sweep.t_final_grid[0]",
        "t_max",
    ),
    "external-time-high": (
        _with(_doc(), sweep__external_times_grid={"x": [1.0, 5.0], "y": [1.0]}),
        "sweep.external_times_grid.x[1]",
        "t_max",
    ),
    "external-time-low": (
        _with(_doc(), sweep__external_times_grid={"x": [1.0], "y": [0.1]}),
        "sweep.external_times_grid.y[0]",
        "t_min",
    ),
    "separation": (
        _with(_doc(), sweep__positions_grid={"x": [0.0], "y": [0.0, 3.0]}),
        "sweep.positions_grid.y[1]",
        "r_max",
    ),
    "separation-just-high": (
        _with(_doc(), sweep__positions_grid={"x": [0.0],
                                             "y": [R_MAX * (1.0 + DELTA)]}),
        "sweep.positions_grid.y[0]",
        "r_max",
    ),
    "separation-extreme": (
        _with(_doc(), sweep__positions_grid={"x": [1e6], "y": [0.0]}),
        "sweep.positions_grid.y[0]",
        "r_max",
    ),
    "separation-vector": (
        _with(_doc(), sweep__positions_grid={"x": [[0.0, 0.0]],
                                             "y": [[3.0, 4.0]]}),
        "sweep.positions_grid.y[0]",
        "r_max",
    ),
    "position-high": (
        _with(_general_doc(), sweep__positions_grid={"x": [0.0, 1.6], "y": [0.0]}),
        "sweep.positions_grid.x[1]",
        "x_max",
    ),
    "position-low": (
        _with(_general_doc(), sweep__positions_grid={"x": [0.0], "y": [-1.6]}),
        "sweep.positions_grid.y[0]",
        "x_max",
    ),
}


@pytest.mark.parametrize("command", [["validate"], ["run", "-q"]],
                         ids=["validate", "run"])
@pytest.mark.parametrize("case", REFUSALS)
def test_TL9_refused_before_the_expansion(tmp_path, capsys, monkeypatch,
                                          command, case):
    from sft_wick.workflow.system import System

    def unreachable(*args, **kwargs):
        raise AssertionError("the expansion was paid for")

    monkeypatch.setattr(System, "expand", unreachable)
    doc, path_, reason = REFUSALS[case]
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(doc, sort_keys=False))
    status = main([command[0], str(config), *command[1:]])
    out = capsys.readouterr()
    assert status == 2, out
    lines = out.err.strip().splitlines()
    assert len(lines) == 1, out.err
    prefix = f"sft-wick: config error [invalid_value] {path_}: "
    assert lines[0].startswith(prefix), lines[0]
    assert reason in lines[0]
    assert "Traceback" not in out.err


def test_TL9_an_override_can_move_the_limit_under_the_grid(tmp_path, capsys):
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(_doc(), sort_keys=False))
    status = main(["validate", str(config), "--override", "propagators.r_max=0.5"])
    err = capsys.readouterr().err
    assert status == 2
    assert "sweep.positions_grid.y[1]" in err


def test_TL9_python_callers_get_a_config_error(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(
        yaml.safe_dump(_with(_doc(), sweep__t_final_grid=[5.0]), sort_keys=False)
    )
    cfg = load_workflow_config(config)
    with pytest.raises(ConfigError) as caught:
        check_before_run(cfg)
    assert caught.value.path == "sweep.t_final_grid[0]"
    with pytest.raises(ConfigError):
        run_workflow(cfg, progress=False)


# ---------------------------------------------------------------------------
# TL10: the limits themselves pass
# ---------------------------------------------------------------------------

ACCEPTED = {
    "the-limits": _with(
        _doc(),
        sweep__t_final_grid=[T_MIN, T_MAX],
        sweep__positions_grid={"x": [0.0], "y": [0.0, R_MAX]},
    ),
    "just-inside": _with(
        _doc(),
        sweep__t_final_grid=[T_MIN * (1.0 + DELTA), T_MAX * (1.0 - DELTA)],
        sweep__positions_grid={"x": [0.0], "y": [R_MAX * (1.0 - DELTA)]},
    ),
    "general-limits": _with(
        _general_doc(),
        sweep__positions_grid={"x": [-X_MAX, X_MAX], "y": [X_MAX, -X_MAX]},
    ),
    "rounding": _with(
        _doc(),
        sweep__t_final_grid=[float(np.nextafter(T_MAX, np.inf))],
        sweep__positions_grid={"x": [0.0],
                               "y": [float(np.nextafter(R_MAX, np.inf))]},
    ),
    "general-rounding": _with(
        _general_doc(),
        sweep__positions_grid={"x": [float(np.nextafter(X_MAX, np.inf))],
                               "y": [float(np.nextafter(-X_MAX, -np.inf))]},
    ),
    "lazy-separation": _with(
        _lazy_doc(), sweep__positions_grid={"x": [0.0], "y": [50.0]}
    ),
    "rotation-positions": _rotation_doc(),
    "closed-form-only": _with(
        _lazy_doc(),
        propagators__c_closed_form_only=True,
        sweep__t_final_grid=[0.1, 5.0],
        sweep__positions_grid={"x": [0.0], "y": [50.0]},
    ),
    "every-point-pinned": _with(
        _doc(),
        sweep__t_final_grid=[5.0],
        sweep__external_times_grid={"x": [T_NODE], "y": [T_EARLIER]},
    ),
}


@pytest.mark.parametrize("case", ACCEPTED)
def test_TL10_the_limits_are_valid(tmp_path, capsys, case):
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(ACCEPTED[case], sort_keys=False))
    status = main(["validate", str(config)])
    out = capsys.readouterr()
    assert status == 0, out.err


def test_TL10_a_run_at_the_limits_agrees_with_the_reference(tmp_path):
    doc = _with(
        _doc(),
        sweep__t_final_grid=[T_MAX],
        sweep__positions_grid={"x": [0.0], "y": [R_MAX]},
        propagators__c_closed_form=None,
        propagators__c_method="gauss_legendre",
        propagators__c_n_gauss=24,
        output=[],
    )
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(doc, sort_keys=False))
    _sweep, totals = run_workflow(load_workflow_config(config), progress=False)
    got = {(r["a"], r["b"]): r["value"] for r in totals.to_dict("records")}
    for comp, gamma in enumerate(GAMMA):
        reference = (
            _translation_kernel(R_MAX) * _time_integral(T_MAX, T_MAX, gamma)
        )
        assert got[(comp, comp)] == pytest.approx(
            reference, rel=TABLE_ERROR, abs=0.0
        )


def test_TL10_the_shipped_quickstart_sits_at_its_limit():
    quickstart = Path(__file__).resolve().parents[1] / "examples" / "quickstart.yaml"
    doc = yaml.safe_load(quickstart.read_text())
    assert max(doc["sweep"]["t_final_grid"]) == doc["propagators"]["t_max"]
    check_before_run(load_workflow_config(quickstart))
