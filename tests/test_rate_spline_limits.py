"""An R read outside the cumulative-rate spline is refused at every level.

A callable ``DiagonalA.gamma`` is held as a spline of ``Gamma`` on
``[t_min_cache, t_max_cache]``.  ``System.propagators`` keeps ``t_max`` inside
it, so a C table bounds every time, but nothing bounded a time when there was
no table: past the end of the spline R extrapolated with no error, and at
``t = 6`` on a spline ending at 3 it returned 6.5e-14 against 0.0398.

RS1 -- at each end of the spline and just inside it R agrees with the
       reference, for a scalar and a matrix R.
RS2 -- a causal read just past the spline, and far past it, is refused, as a
       scalar pair and in a batch.
RS3 -- a read with no spline value in it is not refused: an acausal pair is 0,
       a constant rate has no spline, and an ``ExplicitR`` is the user's.
RS4 -- L1 refuses before anything is read, also a sweep before its first point,
       under ``c_closed_form_only``, where no C table bounds the time.
RS5 -- what holds no R is left alone: order 0 of two phi legs, any order of a
       system with no vertex, and ``t_final`` with every external point pinned.
RS6 -- a hand-built table past the spline is refused at its build or first read.
RS7 -- ``sft-wick validate`` and ``run`` stop with status 2 and one line naming
       the dotted path, before the expansion is paid for.
RS8 -- L1 and L2 share one rule, which goes by the request, so they agree and
       can refuse a time no R reaches, which the R callable itself would not.

Reference: ``scipy.integrate.quad`` of the rate, which imports nothing from
sft-wick.  The rates are nonlinear on purpose.  A cubic spline integrates a
rate linear in ``t`` to a quadratic ``Gamma``, which it extrapolates exactly,
so a linear rate cannot show this defect.
"""
from __future__ import annotations

import math
import pickle
import warnings

import numpy as np
import pytest
import yaml
from scipy.integrate import quad

import sft_wick as sw
import sft_wick.evaluate as ev
from sft_wick.workflow import specs as sp
from sft_wick.workflow.cli import main
from sft_wick.workflow.config import (
    ConfigError,
    check_before_run,
    load_workflow_config,
)

T_MIN, T_CACHE, T_TABLE = 0.4, 3.0, 2.0
N_GRID = 301
#: Just past a limit, relative to it: far above rounding, far below any
#: distance the spline could resolve.
DELTA = 1e-6
SPLINE_ERROR = 1e-7
REFUSED = "cumulative-rate spline"
AT_L1 = r"Expansion\.(evaluate|sweep): .*cumulative-rate spline"
COUPLING = np.array([[[0.2, -0.7], [0.4, 0.1]], [[0.9, 0.3], [-0.5, 0.6]]])


class _Rate:
    """Unequal components, so R is a matrix."""

    def __call__(self, t):
        return np.array([0.7 + 0.5 * np.sin(3.0 * t), 1.9 + 0.3 * np.cos(2.0 * t)])


class _SharedRate:
    """Equal components, so R is a scalar."""

    def __call__(self, t):
        rate = 0.7 + 0.5 * np.sin(3.0 * t)
        return np.array([rate, rate])


class _FixedC:
    """Any closed form: a time past the spline must not depend on it."""

    def __call__(self, n1, t1, n2, t2):
        return np.eye(2) * 0.1


class _OneComponentC:
    def __call__(self, n1, t1, n2, t2):
        return np.array([[0.1]])


def _exact_R(t1: float, t2: float, rate=_Rate(), component: int = 0) -> float:
    integral, _ = quad(
        lambda s: rate(s)[component], t2, t1, epsabs=1e-12, epsrel=1e-12, limit=400
    )
    return math.exp(-integral)


def _linear(rate=None) -> sp.DiagonalA:
    return sp.DiagonalA(
        gamma=rate or _Rate(),
        t_max_cache=T_CACHE,
        n_grid_cache=N_GRID,
        t_min_cache=T_MIN,
    )


def _R(kind: str):
    return _linear(_Rate() if kind == "matrix" else _SharedRate()).build_R_callable()


def _entry(R, kind: str, t1, t2):
    value = R(t1, t2)
    return value[0, 0] if kind == "matrix" else value


def _system(vertices=()) -> sw.System:
    return sw.System(
        field=sp.FieldSpec("phi", n_components=2),
        linear=_linear(),
        noise=sp.GaussianNoise(
            kappa2=sp.SeparableTranslation(
                temporal=sp.ExponentialTemporal(lam=0.35, sigma_t=0.45),
                spatial=sp.ExponentialSpatial(sigma_x=0.8),
            )
        ),
        vertices=tuple(vertices),
        t_min=T_MIN,
    )


def _closed_form_only(system: sw.System):
    return system.propagators(
        t_max=T_TABLE, c_closed_form=_FixedC(), c_closed_form_only=True
    )


def _response(t_later, t_earlier, t_final=None):
    """Order 0 of ``<phi psi>`` is R itself."""
    system = _system()
    expansion = system.expand(("phi_a(x)", "psi_a(y)"), orders=[0])
    return expansion.evaluate(
        _closed_form_only(system),
        positions={"x": 0.0, "y": 0.0},
        t_final=t_later if t_final is None else t_final,
        external_times={"x": t_later, "y": t_earlier},
        component_pair=(0, 0),
        n_samples=64,
    ).total


# ---------------------------------------------------------------------------
# RS1: at each end of the spline
# ---------------------------------------------------------------------------

KINDS = ["matrix", "scalar"]
INSIDE = [
    (T_CACHE, T_MIN),
    (T_CACHE, 1.7),
    (T_CACHE * (1.0 - DELTA), T_MIN * (1.0 + DELTA)),
    (2.2, 0.9),
    (T_MIN, T_MIN),
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("t1,t2", INSIDE,
                         ids=["both-ends", "t_max", "just-inside", "bulk", "t_min"])
def test_RS1_inside_the_spline_agrees_with_the_reference(kind, t1, t2):
    got = _entry(_R(kind), kind, t1, t2)
    assert got == pytest.approx(_exact_R(t1, t2), rel=SPLINE_ERROR, abs=0.0)


@pytest.mark.parametrize("kind", KINDS)
def test_RS1_rounding_at_an_end_is_not_refused(kind):
    R = _R(kind)
    high = np.nextafter(T_CACHE, np.inf)
    low = np.nextafter(T_MIN, -np.inf)
    at = _entry(R, kind, T_CACHE, T_MIN)
    assert _entry(R, kind, high, low) == pytest.approx(at, rel=1e-9, abs=0.0)


def test_RS1_the_second_component_is_right_too():
    got = _R("matrix")(T_CACHE, T_MIN)[1, 1]
    assert got == pytest.approx(
        _exact_R(T_CACHE, T_MIN, component=1), rel=SPLINE_ERROR, abs=0.0
    )


def test_RS1_a_rate_of_a_linear_time_shows_why_the_tests_use_a_sine():
    """Past the spline the linear rate is exact and the sine is not, so a
    check with the linear rate alone would have passed on the defect."""

    class Linear:
        def __call__(self, t):
            return np.array([0.7 + 0.1 * t, 1.9])

    spline = _linear(Linear()).build_R_callable().splines[0]
    gamma = float(spline(6.0) - spline(1.0))
    assert gamma == pytest.approx(0.7 * 5.0 + 0.05 * 35.0, rel=1e-9, abs=0.0)


# ---------------------------------------------------------------------------
# RS2: a causal read past the spline
# ---------------------------------------------------------------------------

LATER = [T_CACHE * (1.0 + DELTA), 4.0, 6.0, 20.0, 1e6, math.inf]
LATER_IDS = ["t_max+delta", "4", "6", "20", "1e6", "inf"]
EARLIER = [T_MIN * (1.0 - DELTA), 0.0, -1.0, -1e6, -math.inf]
EARLIER_IDS = ["t_min-delta", "zero", "minus-1", "minus-1e6", "minus-inf"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("t1", LATER, ids=LATER_IDS)
def test_RS2_a_later_time_past_the_spline_is_refused(kind, t1):
    R = _R(kind)
    with pytest.raises(ValueError, match=REFUSED):
        R(t1, 1.0)
    with pytest.raises(ValueError, match=REFUSED):
        R(np.array([2.0, t1]), np.array([1.0, 1.0]))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("t2", EARLIER, ids=EARLIER_IDS)
def test_RS2_an_earlier_time_past_the_spline_is_refused(kind, t2):
    R = _R(kind)
    with pytest.raises(ValueError, match=REFUSED):
        R(1.0, t2)
    with pytest.raises(ValueError, match=REFUSED):
        R(np.array([2.0, 1.0]), np.array([1.0, t2]))


def test_RS2_the_refusal_names_the_time_and_the_limit():
    with pytest.raises(ValueError, match=r"t=6\.0.*t_max_cache=3\.0"):
        _R("matrix")(6.0, 1.0)
    with pytest.raises(ValueError, match=r"t=0\.1.*t_min_cache=0\.4"):
        _R("matrix")(1.0, 0.1)


@pytest.mark.parametrize("kind", KINDS)
def test_RS2_a_pickled_R_still_refuses(kind):
    R = pickle.loads(pickle.dumps(_R(kind)))
    assert np.isfinite(R(2.0, 1.0)).all()
    with pytest.raises(ValueError, match=REFUSED):
        R(6.0, 1.0)


# ---------------------------------------------------------------------------
# RS3: a read with no spline value in it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("t1,t2", [(1.0, 6.0), (-1.0, 1.0), (-1.0, 6.0)],
                         ids=["later-past", "earlier-past", "both-past"])
def test_RS3_an_acausal_pair_is_zero_wherever_it_sits(kind, t1, t2):
    R = _R(kind)
    assert np.all(R(t1, t2) == 0.0)
    batch = R(np.array([t1, 2.0]), np.array([t2, 1.0]))
    assert np.all(batch[0] == 0.0)
    assert np.any(batch[1] != 0.0)


def test_RS3_an_empty_batch_reads_nothing():
    assert _R("scalar")(np.array([]), np.array([])).shape == (0,)
    assert _R("matrix")(np.array([]), np.array([])).shape == (0, 2, 2)


@pytest.mark.parametrize("gamma", [[0.7, 0.7], [0.7, 1.9]], ids=["scalar", "matrix"])
def test_RS3_a_constant_rate_has_no_spline_and_no_limit(gamma):
    R = sp.DiagonalA(gamma=gamma, t_max_cache=T_CACHE).build_R_callable()
    assert np.isfinite(R(1e3, -1e3)).all()


def test_RS3_explicit_R_is_not_checked():
    system = sw.System(
        field=sp.FieldSpec("phi", n_components=1),
        linear=sp.ExplicitR(R_time=lambda t1, t2: float(t1 >= t2)),
        noise=_system().noise,
    )
    expansion = system.expand(("phi(x)", "psi(y)"), orders=[0])
    props = system.propagators(
        t_max=T_TABLE, c_closed_form=_OneComponentC(), c_closed_form_only=True
    )
    got = expansion.evaluate(
        props,
        positions={"x": 0.0, "y": 0.0},
        t_final=50.0,
        external_times={"x": 50.0, "y": 40.0},
        n_samples=64,
    ).total
    assert got == 1.0


# ---------------------------------------------------------------------------
# RS4: L1 refuses before anything is read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t1", [T_CACHE, T_CACHE * (1.0 - DELTA), 2.2],
                         ids=["t_max", "t_max-delta", "bulk"])
def test_RS4_at_the_limit_the_response_agrees_with_the_reference(t1):
    assert _response(t1, 1.0) == pytest.approx(
        _exact_R(t1, 1.0), rel=SPLINE_ERROR, abs=0.0
    )


@pytest.mark.parametrize("t1", LATER, ids=LATER_IDS)
def test_RS4_a_pinned_time_past_the_spline_is_refused(t1):
    with pytest.raises(ValueError, match=AT_L1):
        _response(t1, 1.0, t_final=1.5)


@pytest.mark.parametrize("t2", EARLIER, ids=EARLIER_IDS)
def test_RS4_a_pinned_time_below_the_spline_is_refused(t2):
    with pytest.raises(ValueError, match=AT_L1):
        _response(1.5, t2)


@pytest.mark.parametrize("t_final", LATER, ids=LATER_IDS)
def test_RS4_order_two_past_the_spline_is_refused(t_final):
    system = _system([sp.LocalVertex("F", COUPLING)])
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2])
    with pytest.raises(ValueError, match=AT_L1):
        expansion.evaluate(
            _closed_form_only(system),
            positions={"x": 0.0, "y": 0.4},
            t_final=t_final,
            component_pair=(0, 1),
            n_samples=64,
        )


def test_RS4_a_sweep_refuses_before_its_first_point(monkeypatch):
    def unreachable(*args, **kwargs):
        raise AssertionError("a grid point was integrated")

    monkeypatch.setattr(ev, "integrate_diagrams", unreachable)
    system = _system([sp.LocalVertex("F", COUPLING)])
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2])
    with pytest.raises(ValueError, match=AT_L1):
        expansion.sweep(
            _closed_form_only(system),
            positions_grid={"x": [0.0], "y": [0.4]},
            t_final_grid=[1.0, 6.0],
            component_pairs=[(0, 1)],
        )


def test_RS4_the_R_of_a_built_model_refuses():
    """What a caller gets who builds a model and integrates at L0."""
    model = _system().build_propagator_model()
    with pytest.raises(ValueError, match=REFUSED):
        model.R_time(6.0, 1.0)


# ---------------------------------------------------------------------------
# RS5: L1 refuses only what is read
# ---------------------------------------------------------------------------


def test_RS5_order_zero_of_two_phi_legs_reads_no_R():
    """A closed form is right at any time, and no R enters this number."""
    system = _system([sp.LocalVertex("F", COUPLING)])
    expansion = system.expand(("phi_a(x)", "phi_a(y)"), orders=[0, 2])
    got = expansion.evaluate(
        _closed_form_only(system),
        positions={"x": 0.0, "y": 0.4},
        t_final=20.0,
        component_pair=(0, 0),
        orders=[0],
        n_samples=64,
    ).total
    assert got == pytest.approx(0.1, rel=1e-12, abs=0.0)


def test_RS5_a_system_with_no_vertex_reads_no_R_at_any_order():
    system = _system()
    expansion = system.expand(("phi_a(x)", "phi_a(y)"), orders=[0, 2])
    got = expansion.evaluate(
        _closed_form_only(system),
        positions={"x": 0.0, "y": 0.4},
        t_final=20.0,
        component_pair=(0, 0),
        n_samples=64,
    ).total
    assert got == pytest.approx(0.1, rel=1e-12, abs=0.0)


def test_RS5_t_final_is_not_read_when_every_point_is_pinned():
    assert _response(2.2, 0.9, t_final=20.0) == _response(2.2, 0.9)


def test_RS5_order_two_with_every_point_pinned_reads_inside_the_spline():
    system = _system([sp.LocalVertex("F", COUPLING)])
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2])
    props = _closed_form_only(system)
    values = {
        t_final: expansion.evaluate(
            props,
            positions={"x": 0.0, "y": 0.4},
            t_final=t_final,
            external_times={"x": 2.0, "y": 1.2},
            component_pair=(0, 1),
            n_samples=2**7,
            seed=3,
        ).total
        for t_final in (2.1, 20.0)
    }
    assert values[2.1] == values[20.0] != 0.0


def test_RS5_a_table_inside_the_spline_is_unchanged():
    """With a table ``t_max <= t_max_cache`` bounds every time already."""
    system = _system()
    expansion = system.expand(("phi_a(x)", "psi_a(y)"), orders=[0])
    props = system.propagators(t_max=T_CACHE, n_grid_t=5)
    got = expansion.evaluate(
        props,
        positions={"x": 0.0, "y": 0.0},
        t_final=T_CACHE,
        external_times={"x": T_CACHE, "y": 1.0},
        component_pair=(0, 0),
        n_samples=64,
    ).total
    assert got == pytest.approx(_exact_R(T_CACHE, 1.0), rel=SPLINE_ERROR, abs=0.0)


@pytest.mark.parametrize(
    "linear,t_min,t_max",
    [
        (dict(t_max_cache=0.01, n_grid_cache=11), 0.0, 0.01),
        (dict(t_max_cache=0.5, n_grid_cache=51), 0.0, 0.5),
        (dict(t_max_cache=5.0, n_grid_cache=101, t_min_cache=3.0), 3.0, 5.0),
        (dict(t_max_cache=2.0, n_grid_cache=51), -1.0, 2.0),
    ],
    ids=["tiny", "shorter-than-one", "starts-late", "extended-below-zero"],
)
@pytest.mark.parametrize("method", ["qmc_vectorized", "gauss_legendre"])
def test_RS5_nothing_reads_R_at_a_time_of_its_own(linear, t_min, t_max, method):
    """A table build, its probes and both integrators stay inside a spline
    of any length and start, at a ``t_final`` on its end."""
    system = sw.System(
        field=sp.FieldSpec("phi", n_components=2),
        linear=sp.DiagonalA(gamma=_Rate(), **linear),
        noise=_system().noise,
        vertices=(sp.LocalVertex("F", COUPLING),),
        t_min=t_min,
    )
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2])
    got = expansion.evaluate(
        system.propagators(t_max=t_max, n_grid_t=5),
        positions={"x": 0.0, "y": 0.3},
        t_final=t_max,
        component_pair=(0, 1),
        method=method,
        n_samples=64,
        n_gauss=3,
    ).total
    assert np.isfinite(got) and got != 0.0


# ---------------------------------------------------------------------------
# RS6: a hand-built table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder,options",
    [
        ("precompute_C_table", {"n_grid": 3}),
        ("precompute_C_table_translation",
         {"n_grid_t": 3, "r_max": 1.0, "n_grid_r": 2}),
    ],
    ids=["time-table", "full-grid"],
)
def test_RS6_a_table_past_the_spline_is_refused_at_its_build(builder, options):
    """``System.propagators`` checks ``t_max`` and these do not, and a table
    built on an extrapolated R would be wrong in every cell past the end."""
    model = _system().build_propagator_model()
    cache = ev.PropagatorCache(model, homogeneity="translation")
    with pytest.raises(ValueError, match=REFUSED), warnings.catch_warnings(
        record=True
    ) as caught:
        warnings.simplefilter("always")
        getattr(cache, builder)(t_max=T_CACHE + 1.0, **options)
    # The rule search warns before the build stops; the warning said C
    # "falls back to dblquad", which raises the same error here.
    texts = [str(w.message) for w in caught if "dblquad" in str(w.message)]
    assert texts and all("raises the same error" in t for t in texts)


def test_RS6_a_lazy_table_refuses_at_its_first_read():
    """A lazy table reads nothing at its build, so that is where it stops."""
    model = _system().build_propagator_model()
    cache = ev.PropagatorCache(model, homogeneity="translation")
    cache.precompute_C_table_translation(t_max=T_CACHE + 1.0, n_grid_t=3)
    with pytest.raises(ValueError, match=REFUSED):
        cache.C_at_batch(np.array([1.0]), np.array([1.0]), 0.0, 0.5)


# ---------------------------------------------------------------------------
# RS7: validate and run stop before the expansion
# ---------------------------------------------------------------------------


def _write_doc(tmp_path, *, observable=("phi_a(x)", "psi_a(y)"), orders=(0,),
               sweep=None, propagators=None):
    (tmp_path / "rate.py").write_text(
        "import numpy as np\n\n\n"
        "def gamma(t):\n"
        "    return np.array([0.7 + 0.5 * np.sin(3.0 * t), 1.9])\n"
    )
    (tmp_path / "c_fn.py").write_text(
        "import numpy as np\n\n\n"
        "def C_fn(n1, t1, n2, t2):\n"
        "    return np.eye(2) * 0.1\n"
    )
    doc = {
        "system": {
            "t_min": T_MIN,
            "field": {"name": "phi", "n_components": 2},
            "linear": {
                "type": "diagonal",
                "gamma_module": "rate.py",
                "t_max_cache": T_CACHE,
                "t_min_cache": T_MIN,
                "n_grid_cache": N_GRID,
            },
            "noise": {
                "kappa2": {
                    "type": "separable_translation",
                    "temporal": {"type": "exponential", "lam": 0.35, "sigma_t": 0.45},
                    "spatial": {"type": "exponential", "sigma_x": 0.8},
                }
            },
            "vertices": [{"name": "F", "coupling": COUPLING.tolist()}],
        },
        "expand": {"observable": list(observable), "orders": list(orders)},
        "propagators": {
            "t_max": T_TABLE,
            "c_closed_form_module": "c_fn.py",
            "c_closed_form_only": True,
            **(propagators or {}),
        },
        "sweep": {
            "positions_grid": {"x": [0.0], "y": [0.0]},
            "t_final_grid": [1.0],
            "external_times_grid": {"x": [1.5], "y": [1.0]},
            "component_pairs": [[0, 0]],
            "n_samples": 64,
            **(sweep or {}),
        },
    }
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def _free(**sweep) -> dict:
    """Every point left free, so ``t_final_grid`` is read."""
    return {"external_times_grid": None, **sweep}


REFUSALS = {
    "pinned-high": (
        dict(sweep={"external_times_grid": {"x": [1.5, 6.0], "y": [1.0]}}),
        "sweep.external_times_grid.x[1]",
        "t_max_cache",
    ),
    "pinned-just-high": (
        dict(sweep={"external_times_grid": {"x": [T_CACHE * (1.0 + DELTA)],
                                            "y": [1.0]}}),
        "sweep.external_times_grid.x[0]",
        "t_max_cache",
    ),
    "pinned-low": (
        dict(sweep={"external_times_grid": {"x": [1.5], "y": [0.1]}}),
        "sweep.external_times_grid.y[0]",
        "t_min_cache",
    ),
    "t_final": (
        dict(sweep=_free(t_final_grid=[1.0, 6.0]), orders=(2,),
             observable=("phi_a(x)", "phi_b(y)")),
        "sweep.t_final_grid[1]",
        "t_max_cache",
    ),
    "t_final-extreme": (
        dict(sweep=_free(t_final_grid=[1e6]), orders=(0, 2),
             observable=("phi_a(x)", "phi_b(y)")),
        "sweep.t_final_grid[0]",
        "t_max_cache",
    ),
    "t_max": (
        dict(propagators={"t_max": 4.0}),
        "propagators.t_max",
        "t_max_cache",
    ),
}


@pytest.mark.parametrize("command", [["validate"], ["run", "-q"]],
                         ids=["validate", "run"])
@pytest.mark.parametrize("case", REFUSALS)
def test_RS7_refused_before_the_expansion(tmp_path, capsys, monkeypatch, command,
                                          case):
    from sft_wick.workflow.system import System

    def unreachable(*args, **kwargs):
        raise AssertionError("the expansion was paid for")

    monkeypatch.setattr(System, "expand", unreachable)
    options, path_, reason = REFUSALS[case]
    config = _write_doc(tmp_path, **options)
    status = main([command[0], str(config), *command[1:]])
    out = capsys.readouterr()
    assert status == 2, out
    lines = out.err.strip().splitlines()
    assert len(lines) == 1, out.err
    assert lines[0].startswith(
        f"sft-wick: config error [invalid_value] {path_}: "
    ), lines[0]
    assert reason in lines[0]
    assert "Traceback" not in out.err


ACCEPTED = {
    "the-end-of-the-spline": dict(
        sweep={"external_times_grid": {"x": [T_CACHE], "y": [T_MIN]}}
    ),
    "rounding": dict(
        sweep={"external_times_grid": {"x": [float(np.nextafter(T_CACHE, np.inf))],
                                       "y": [T_MIN]}}
    ),
    "order-zero-of-phi-reads-no-R": dict(
        sweep=_free(t_final_grid=[20.0]), orders=(0,),
        observable=("phi_a(x)", "phi_a(y)"),
    ),
    "unread-t_final": dict(sweep={"t_final_grid": [20.0]}),
}


@pytest.mark.parametrize("case", ACCEPTED)
def test_RS7_what_is_not_past_the_spline_is_valid(tmp_path, capsys, case):
    config = _write_doc(tmp_path, **ACCEPTED[case])
    status = main(["validate", str(config)])
    assert status == 0, capsys.readouterr().err


def test_RS7_python_callers_get_a_config_error(tmp_path):
    options, path_, _reason = REFUSALS["t_final"]
    cfg = load_workflow_config(_write_doc(tmp_path, **options))
    with pytest.raises(ConfigError) as caught:
        check_before_run(cfg)
    assert caught.value.path == path_


def test_RS7_a_run_at_the_end_of_the_spline_agrees_with_the_reference(tmp_path):
    from sft_wick.workflow.config import run_workflow

    def rate(t):
        return np.array([0.7 + 0.5 * math.sin(3.0 * t), 1.9])

    config = _write_doc(
        tmp_path,
        orders=(0,),
        sweep={"external_times_grid": {"x": [T_CACHE], "y": [1.0]}},
    )
    _sweep, totals = run_workflow(load_workflow_config(config), progress=False)
    assert float(totals["value"].iloc[0]) == pytest.approx(
        _exact_R(T_CACHE, 1.0, rate), rel=SPLINE_ERROR, abs=0.0
    )


# ---------------------------------------------------------------------------
# RS8: L1 and L2 go by the request
# ---------------------------------------------------------------------------


def test_RS8_a_pinned_psi_later_than_its_phi_is_refused_up_front():
    """R(1, 6) is 0 and reads nothing (RS3).  The request still names a time
    past the spline, and no check before the run can do better than that
    without the time ordering of every diagram."""
    assert np.all(_R("matrix")(1.0, 6.0) == 0.0)
    with pytest.raises(ValueError, match=AT_L1):
        _response(1.0, 6.0)


def test_RS8_an_integrated_point_capped_by_a_pinned_one_is_refused_up_front():
    system = _system()
    expansion = system.expand(("phi_a(x)", "psi_a(y)"), orders=[0])
    with pytest.raises(ValueError, match=AT_L1):
        expansion.evaluate(
            _closed_form_only(system),
            positions={"x": 0.0, "y": 0.0},
            t_final=20.0,
            external_times={"x": 2.0},
            integrate_over={"y"},
            component_pair=(0, 0),
            n_samples=64,
        )


def test_RS8_an_order_with_no_diagram_counts_at_both_levels(tmp_path, capsys):
    """Order 1 of a cubic vertex with two legs holds no diagram, which only
    the expansion could show.  L1 could see it and L2 cannot, so L1 refuses
    too: ``validate`` and ``run`` must agree."""
    config = _write_doc(
        tmp_path,
        observable=("phi_a(x)", "phi_a(y)"),
        orders=(0, 1),
        sweep=_free(t_final_grid=[20.0]),
    )
    assert main(["validate", str(config)]) == 2
    assert "sweep.t_final_grid[0]" in capsys.readouterr().err
    system = _system([sp.LocalVertex("F", COUPLING)])
    expansion = system.expand(("phi_a(x)", "phi_a(y)"), orders=[0, 1])
    with pytest.raises(ValueError, match=AT_L1):
        expansion.evaluate(
            _closed_form_only(system),
            positions={"x": 0.0, "y": 0.4},
            t_final=20.0,
            component_pair=(0, 0),
            n_samples=64,
        )


@pytest.mark.parametrize(
    "orders,observable,refused",
    [
        ([0], ("phi", "phi"), False),
        ([2], ("phi", "phi"), True),
        ([0], ("phi", "psi"), True),
        ([2], ("psi", "psi"), True),
    ],
    ids=["order-zero", "order-two", "response-leg", "response-only"],
)
def test_RS8_one_rule_at_both_levels(orders, observable, refused):
    from sft_wick.workflow.expansion import _reads_R, _request_reads_R

    system = _system([sp.LocalVertex("F", COUPLING)])
    assert _request_reads_R(system, orders, set(observable)) is refused
    expansion = system.expand(
        (f"{observable[0]}_a(x)", f"{observable[1]}_a(y)"), orders=orders
    )
    assert _reads_R(expansion, orders) is refused


@pytest.mark.parametrize("excess, refused", [(5e-13, False), (1e-9, True)])
def test_t_max_within_rounding_of_the_spline_end_is_at_the_end(excess, refused):
    """``System.propagators`` compared with no slack, so it refused a computed
    ``t_max`` that R itself reads."""
    system = sw.System(
        field=sw.FieldSpec("phi", 1),
        linear=sw.DiagonalA(gamma=lambda t: np.array([1.0 + 0.1 * t]),
                            t_max_cache=10.0),
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=1.0, sigma_t=0.5),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            )
        ),
        t_min=0.25,
    )
    t_max = 10.0 * (1.0 + excess)
    if refused:
        with pytest.raises(ValueError, match="t_max_cache"):
            system._require_gamma_grid(t_max)
    else:
        system._require_gamma_grid(t_max)
