"""Connected incidence, independent cumulant partitions and public routing."""

from __future__ import annotations

import itertools
import math
import pickle
from fractions import Fraction

import numpy as np
import pytest

import sft_wick as sw
from sft_wick._connected import ConnectedIncidence
from sft_wick.expressions import (
    ImaginaryUnit,
    IntegralOver,
    Product,
    Propagator,
    Rational,
    Sum,
    Symbol,
)
from sft_wick.indices import IndexContext
from sft_wick.perturbation import _enumerate_component_routings
from sft_wick.vertices import VertexInstance
from sft_wick.wick import wick_contract_spatial
from sft_wick.workflow.config import _parse_expand


@pytest.mark.parametrize("engine", ["symbolic", "operator", "numerical"])
def test_connected_cycle_cache_lifetime_is_per_contraction(engine):
    import sft_wick.wick as wick_module

    phi = sw.Field("phi", "physical")
    psi = sw.Field("psi", "response")
    action = sw.Action([])
    for index in range(100):
        observable = [phi(f"x{index}"), psi(f"y{index}")]
        if engine == "numerical":
            sw.compute_moment_numerical(observable, action, 0, {}, target="connected")
        else:
            sw.compute_moment(
                observable,
                action,
                0,
                target="connected",
                collect_topology=engine != "operator",
            )
        assert len(wick_module._r_cycle_cache) <= 1


def test_connected_numerical_real_parallel_routing_matches_serial(monkeypatch):
    import joblib

    phi = sw.Field("phi", "physical", 2)
    psi = sw.Field("psi", "response", 2)
    action = sw.Action(
        [
            sw.Vertex([psi, phi, phi], "F"),
            sw.Vertex([psi] * 4, "K4", local=False),
        ]
    )
    observable = [phi(chr(97 + k), f"x{k}") for k in range(4)]
    couplings = {"F": np.ones((2, 2, 2)), "K4": np.ones((2,) * 4)}
    serial = sw.compute_moment_numerical(
        observable,
        action,
        3,
        couplings,
        target="connected",
        n_jobs=1,
    )
    native_parallel = joblib.Parallel
    parallel_jobs = []

    def observed_parallel(*args, **kwargs):
        parallel_jobs.append(kwargs.get("n_jobs"))
        return native_parallel(*args, **kwargs)

    monkeypatch.setattr(joblib, "Parallel", observed_parallel)
    parallel = sw.compute_moment_numerical(
        observable,
        action,
        3,
        couplings,
        target="connected",
        n_jobs=2,
    )
    # The surviving FF K4 incidence topologies exceed the real >100 threshold.
    assert 2 in parallel_jobs
    assert serial.keys() == parallel.keys()
    for order in serial:
        assert len(serial[order]) == len(parallel[order])
        for index, (first, second) in enumerate(zip(serial[order], parallel[order])):
            for name in first.__dataclass_fields__:
                assert getattr(first, name) == getattr(second, name)
            if index in {0, len(serial[order]) // 2, len(serial[order]) - 1}:
                np.testing.assert_array_equal(
                    first.evaluate_coupling(couplings),
                    second.evaluate_coupling(couplings),
                )


def _partitions(items):
    """Independent set-partition oracle; no graph/package contraction calls."""
    if not items:
        yield ()
        return
    first, rest = items[0], items[1:]
    for blocks in _partitions(rest):
        yield ((first,),) + blocks
        for k, block in enumerate(blocks):
            yield blocks[:k] + ((first,) + block,) + blocks[k + 1 :]


def _multiply(a, b, order=2):
    return [
        sum((a[k] * b[n - k] for k in range(n + 1)), Fraction())
        for n in range(order + 1)
    ]


def _partition_cumulant(q, moments):
    out = [Fraction()] * 3
    for blocks in _partitions(tuple(range(q))):
        value = [Fraction(1), Fraction(), Fraction()]
        for block in blocks:
            value = _multiply(value, moments[len(block)])
        weight = (-1) ** (len(blocks) - 1) * math.factorial(len(blocks) - 1)
        out = [a + weight * b for a, b in zip(out, value)]
    return out


def _expr_value(expr, couplings):
    """Unit propagators and unit vertex measures, for scalar algebra pins."""
    if isinstance(expr, Rational):
        return expr.numerator / expr.denominator
    if isinstance(expr, ImaginaryUnit):
        return 1j
    if isinstance(expr, Propagator):
        return 1
    if isinstance(expr, Symbol):
        return np.asarray(couplings[expr.name]).item()
    if isinstance(expr, Product):
        return math.prod(_expr_value(x, couplings) for x in expr.factors)
    if isinstance(expr, Sum):
        return sum(_expr_value(x, couplings) for x in expr.terms)
    if isinstance(expr, IntegralOver):
        return _expr_value(expr.body, couplings)
    raise AssertionError(type(expr))


def _coefficients(q, action, coupling_values, engine, target):
    phi = (
        action.vertices[0].fields[-1]
        if action.vertices[0].fields[-1].is_physical
        else sw.Field("phi", "physical")
    )
    obs = [phi(f"x{k}") for k in range(q)]
    if engine == "operator":
        result = sw.compute_moment(
            obs, action, 2, collect_topology=False, target=target
        )
        return [_expr_value(result.order(n), coupling_values) for n in range(3)]
    if engine == "symbolic":
        result = sw.compute_moment(obs, action, 2, target=target)
        dts = result.diagram_terms_by_order
    else:
        dts = sw.compute_moment_numerical(
            obs, action, 2, coupling_values, target=target
        )
    return [
        sum(
            complex(np.sum(dt.evaluate_coupling(coupling_values)))
            for dt in dts.get(n, [])
        )
        for n in range(3)
    ]


@pytest.mark.parametrize("engine", ["symbolic", "operator", "numerical"])
@pytest.mark.parametrize("q", [1, 2, 3, 4, 6])
def test_connected_matches_independent_source_moment_partitions(engine, q):
    """exp(C J²/2 + λ(bJ + κ3 J³/6 + κ4 J⁴/24)) is independent."""
    psi = sw.Field("psi", "response")
    bare = {1: Fraction(2, 5), 3: Fraction(3, 10), 4: Fraction(1, 5)}
    vertices = [sw.Vertex([psi] * rank, f"K{rank}", local=False) for rank in bare]
    couplings = {
        f"K{rank}": np.asarray(-(1j**rank) * float(value) / math.factorial(rank))
        for rank, value in bare.items()
    }
    action = sw.Action(vertices)
    moments = {}
    for size in range(1, q + 1):
        total = [Fraction()] * 3
        for blocks in _partitions(tuple(range(size))):
            term = [Fraction(1), Fraction(), Fraction()]
            for block in blocks:
                rank = len(block)
                value = [Fraction(), Fraction(), Fraction()]
                if rank == 2:
                    value[0] = Fraction(1)
                elif rank in bare:
                    value[1] = bare[rank]
                term = _multiply(term, value)
            total = [a + b for a, b in zip(total, term)]
        moments[size] = total
    expected = _partition_cumulant(q, moments)
    # Each coefficient is a float64 sum over many diagram terms; its
    # summation-order roundoff differs between platforms (2e-14 relative
    # on Linux CI at q = 6), so the tolerance is relative.
    assert _coefficients(q, action, couplings, engine, "moment") == pytest.approx(
        [float(x) for x in moments[q]], rel=1e-12, abs=2e-13
    )
    assert _coefficients(q, action, couplings, engine, "connected") == pytest.approx(
        [float(x) for x in expected], rel=1e-12, abs=2e-13
    )


@pytest.mark.parametrize("engine", ["symbolic", "operator", "numerical"])
@pytest.mark.parametrize("q", [2, 4])
def test_vacuum_normalization_uses_formal_z_before_partitions(engine, q):
    """Normalized Gaussian phi² action oracle, not raw-moment subtraction."""

    def gaussian_moment(degree):
        return math.prod(range(1, degree, 2)) if degree % 2 == 0 else 0

    z = [
        Fraction((-1) ** n * gaussian_moment(2 * n), math.factorial(n))
        for n in range(3)
    ]
    normalized = {}
    for degree in range(1, q + 1):
        raw = [
            Fraction((-1) ** n * gaussian_moment(degree + 2 * n), math.factorial(n))
            for n in range(3)
        ]
        normalized[degree] = []
        for n in range(3):
            normalized[degree].append(
                raw[n] - sum(z[k] * normalized[degree][n - k] for k in range(1, n + 1))
            )
    phi = sw.Field("phi", "physical")
    action = sw.Action([sw.Vertex([phi, phi], "V")])
    actual = _coefficients(q, action, {"V": np.asarray(1)}, engine, "connected")
    expected = _partition_cumulant(q, normalized)
    assert actual == pytest.approx([float(x) for x in expected], abs=1e-13)
    if q == 2:
        assert _coefficients(
            q, action, {"V": np.asarray(1)}, engine, "moment"
        ) == pytest.approx([1, -3, 7.5])
        assert actual == pytest.approx([1, -2, 4])


@pytest.mark.parametrize("n_components", [1, 2, 3])
@pytest.mark.parametrize("rank", [3, 4])
@pytest.mark.parametrize("engine", ["symbolic", "numerical"])
@pytest.mark.parametrize("mode", ["ordinary", "equal_time", "absorbed"])
def test_input_cumulant_hyperedge_keeps_asymmetric_tensor_and_aliases(
    n_components, rank, engine, mode
):
    phi, psi = sw.Field("phi", "physical", n_components), sw.Field(
        "psi", "response", n_components
    )
    flags = {
        "equal_time": mode == "equal_time",
        "already_R_contracted": mode == "absorbed",
    }
    vertex = sw.Vertex([psi] * rank, "K", local=False, **flags)
    bare = np.arange(n_components**rank).reshape((n_components,) * rank) / 10 + 0.3
    values = {"K": -(1j**rank) * bare / math.factorial(rank)}
    indices = tuple(k % n_components for k in range(rank))
    fixed = {chr(97 + k): v for k, v in enumerate(indices)}
    obs = [
        phi(f"x{k}") if n_components == 1 else phi(chr(97 + k), f"x{k}")
        for k in range(rank)
    ]
    if engine == "symbolic":
        result = sw.compute_moment(
            obs, sw.Action([vertex]), 1, iso_R=True, target="connected"
        )
        dts = result.diagram_terms(1)
        assert not result.diagram_terms(0)
    else:
        dts = sw.compute_moment_numerical(
            obs, sw.Action([vertex]), 1, values, fixed, iso_R=True, target="connected"
        ).get(1, [])
    value = sum(complex(np.sum(dt.evaluate_coupling(values, fixed))) for dt in dts)
    expected = sum(
        bare[tuple(indices[k] for k in perm)]
        for perm in itertools.permutations(range(rank))
    ) / math.factorial(rank)
    assert value == pytest.approx(expected, abs=1e-13)
    if mode == "equal_time":
        assert all(dt.equal_time_aliases for dt in dts)
    if mode == "absorbed":
        assert all(len(dt.r_absorbed_pairs) == rank for dt in dts)


@pytest.mark.parametrize("engine", ["symbolic", "operator", "numerical"])
def test_ownership_label_collision_filters_routings_not_representative(engine):
    phi, psi = sw.Field("phi", "physical", 2), sw.Field("psi", "response", 2)
    obs = [phi("a", "y_0"), phi("b", "x1"), phi("c", "x2")]
    vertex = sw.Vertex([psi, phi, phi], "F")
    vi = VertexInstance.instantiate(vertex, IndexContext(), 0)
    ops = obs + vi.field_operators
    plan = ConnectedIncidence.from_instances(obs, [vi])
    assert not plan.disjoint_labels
    topologies = wick_contract_spatial(
        ops, vertex_points=frozenset(vi.spatial_variables)
    )
    assert len(topologies) == 2
    for props, _mult, pairing in topologies.values():
        assert not plan.is_connected(pairing)
        routings = _enumerate_component_routings(
            props, pairing, ops, frozenset(vi.spatial_variables)
        )
        assert [plan.is_connected(p) for _, p in routings] == [False, True, True]
    # The engines name their vertex points around the observable's labels,
    # so the end-to-end count is that of any other spelling: 3 choices of
    # the R-connected external times 2 routings of the two phi legs.
    # Earlier versions also named the vertex point 'y_0' and counted 4.
    if engine == "numerical":
        result = sw.compute_moment_numerical(
            obs, sw.Action([vertex]), 1, {"F": np.ones((2,) * 3)}, target="connected"
        )
        assert len(result[1]) == 6
    else:
        result = sw.compute_moment(
            obs,
            sw.Action([vertex]),
            1,
            collect_topology=engine != "operator",
            target="connected",
        )
        assert len(result.diagrams_by_order[1]) == 6
        assert all(
            ConnectedIncidence.from_instances(
                d.observable_ops, d.vertex_instances
            ).is_connected(d.pairing)
            for d in result.diagrams_by_order[1]
        )


@pytest.mark.parametrize("engine", ["symbolic", "numerical"])
def test_zero_order_occurrences_ignore_location_and_uid(engine):
    phi = sw.Field("phi", "physical")
    same = phi("x")
    for q in (2, 4):
        if engine == "symbolic":
            result = sw.compute_moment([same] * q, sw.Action([]), 0, target="connected")
            dts = result.diagram_terms(0)
        else:
            dts = sw.compute_moment_numerical(
                [same] * q, sw.Action([]), 0, {}, target="connected"
            ).get(0, [])
        assert len(dts) == (1 if q == 2 else 0)


@pytest.mark.parametrize("engine", ["symbolic", "numerical"])
def test_impossible_edge_count_skips_wick_and_retains_prior_validation(
    monkeypatch, engine
):
    import sft_wick.perturbation as p

    phi, psi = sw.Field("phi", "physical"), sw.Field("psi", "response")
    action = sw.Action([sw.Vertex([psi, phi, phi], "F")])

    def forbidden(*a, **kw):
        raise AssertionError("Wick spatial enumeration should be skipped")

    monkeypatch.setattr(p, "wick_contract_spatial", forbidden)
    obs = [phi(f"x{k}") for k in range(6)]
    if engine == "symbolic":
        assert not sw.compute_moment(obs, action, 2, target="connected").diagram_terms(
            2
        )

        def call():
            return sw.compute_moment([phi("x")] * 6, action, 2, target="connected")

    else:
        assert not sw.compute_moment_numerical(
            obs, action, 2, {"F": np.asarray(1)}, target="connected"
        )

        def call():
            return sw.compute_moment_numerical(
                [phi("x")] * 6, action, 2, {}, target="connected"
            )

    with pytest.raises(ValueError, match="share the same spatial label"):
        call()


def _system():
    return sw.System(
        field=sw.FieldSpec("phi", n_components=1),
        linear=sw.DiagonalA(gamma=[1]),
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
                spatial=sw.ExponentialSpatial(sigma_x=1),
            )
        ),
    )


def test_public_target_cache_separation_and_old_metadata(tmp_path, monkeypatch):
    system = _system()
    obs = ("phi(x)", "phi(y)", "phi(z)", "phi(w)")
    moment = system.expand(obs, [0], cache_path=tmp_path, progress=False)
    connected = system.expand(
        obs, [0], cache_path=tmp_path, progress=False, target="connected"
    )
    assert moment.target == moment.raw_result.target == "moment"
    assert connected.target == connected.raw_result.target == "connected"
    assert len(moment.diagrams(0)) == 3 and connected.diagrams(0) == []
    assert len(list(tmp_path.glob("expansion_*.joblib"))) == 2
    import sft_wick.perturbation as p

    monkeypatch.setattr(
        p,
        "compute_moment",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    assert (
        system.expand(obs, [0], cache_path=tmp_path, progress=False).target == "moment"
    )
    assert (
        system.expand(
            obs, [0], cache_path=tmp_path, progress=False, target="connected"
        ).target
        == "connected"
    )
    del moment.raw_result.target
    object.__delattr__(moment, "target")
    restored = pickle.loads(pickle.dumps(moment))
    assert restored.target == restored.raw_result.target == "moment"


def test_target_validation_and_config_default():
    phi = sw.Field("phi", "physical")
    for target in ("cumulant", None, False):
        with pytest.raises(ValueError, match="target must"):
            sw.compute_moment([phi("x")], sw.Action([]), 0, target=target)
    with pytest.raises(ValueError, match="at least one"):
        sw.compute_moment([], sw.Action([]), 0, target="connected")
    assert sw.compute_moment([], sw.Action([]), 0).target == "moment"
    assert _parse_expand({"observable": ["phi(x)"], "orders": [0]}).target == "moment"
    assert (
        _parse_expand(
            {"observable": ["phi(x)"], "orders": [0], "target": "connected"}
        ).target
        == "connected"
    )
    with pytest.raises(ValueError, match="target must"):
        _parse_expand({"observable": ["phi(x)"], "orders": [0], "target": "cumulant"})


@pytest.mark.parametrize("target", ["moment", "connected"])
def test_scalar_numerical_symbols_match_vertex_instance_no_component_axes(target):
    phi, psi = sw.Field("phi", "physical"), sw.Field("psi", "response")
    vertex = sw.Vertex([psi] * 3, "K", local=False)
    instance = VertexInstance.instantiate(vertex, IndexContext(), 0)
    assert instance.coupling_symbol.indices == ()
    values = {"K": np.asarray(1j / 6)}
    dts = sw.compute_moment_numerical(
        [phi(f"x{k}") for k in range(3)], sw.Action([vertex]), 1, values, target=target
    )[1]
    assert all(
        isinstance(dt.coupling_sum, Symbol) and dt.coupling_sum.indices == ()
        for dt in dts
    )
    assert sum(complex(dt.evaluate_coupling(values)) for dt in dts) == pytest.approx(
        1, abs=1e-15
    )


@pytest.mark.parametrize("engine", ["symbolic", "numerical"])
def test_component_aliases_keep_external_response_phase_and_zero_leg_bound(
    engine, monkeypatch
):
    import sft_wick.perturbation as p

    phi, psi = sw.Field("phi", "physical", 3), sw.Field("psi", "response", 3)
    obs = [phi("a", "x"), psi("a", "y")]
    if engine == "symbolic":
        result = sw.compute_moment(
            obs, sw.Action([]), 0, iso_R=True, target="connected"
        )
        dts = result.diagram_terms(0)
        assert (
            "-\\mathrm{i}" in result.order(0).to_latex()
            or "\\mathrm{i}" in result.order(0).to_latex()
        )
    else:
        dts = sw.compute_moment_numerical(
            obs, sw.Action([]), 0, {}, iso_R=True, target="connected"
        )[0]
    assert len(dts) == 1 and dts[0].n_external_response == 1
    assert complex(dts[0].evaluate_coupling({}, {"a": 0})) == -1j
    assert dts[0].observable_phase_factor() == 1j

    def forbidden(*a, **kw):
        raise AssertionError("isolated zero-leg vertex should be pruned")

    monkeypatch.setattr(p, "wick_contract_spatial", forbidden)
    action = sw.Action([sw.Vertex([], "Z")])
    obs = [phi("a", "x"), phi("b", "y")]
    if engine == "symbolic":
        assert not sw.compute_moment(obs, action, 2, target="connected").diagram_terms(
            1
        )
    else:
        assert set(
            sw.compute_moment_numerical(obs, action, 2, {}, target="connected")
        ) == {0}


@pytest.mark.parametrize("engine", ["symbolic", "numerical"])
@pytest.mark.parametrize("q", [2, 3, 4])
@pytest.mark.parametrize("model", ["quadratic_drift", "multiplicative_noise"])
def test_local_response_vertices_against_independent_evolution_moment_partitions(
    engine, q, model
):
    """Exact formal coefficients from an ODE or geometric Brownian motion.

    Drift: phi=X/(1-lambda*t*X), used only as a formal series about zero.
    Noise: phi=X exp(beta*W-beta²*t/2), expanded in lambda=beta².
    X is centered unit Gaussian; the noise formula is an actual moment pin.
    """

    def gaussian(degree):
        return math.prod(range(1, degree, 2)) if degree % 2 == 0 else 0

    moments = {}
    for degree in range(1, q + 1):
        if model == "quadratic_drift":
            moments[degree] = [
                Fraction(math.comb(degree + n - 1, n) * gaussian(degree + n))
                for n in range(3)
            ]
        else:
            rate = Fraction(degree * (degree - 1), 2)
            moments[degree] = [
                Fraction(gaussian(degree)) * rate**n / math.factorial(n)
                for n in range(3)
            ]
    phi, psi = sw.Field("phi", "physical"), sw.Field("psi", "response")
    fields = [psi, phi, phi] if model == "quadratic_drift" else [psi, psi, phi, phi]
    action = sw.Action([sw.Vertex(fields, "V")])
    couplings = {"V": np.asarray(-1j if model == "quadratic_drift" else 0.5)}
    obs = [phi(f"x{k}") for k in range(q)]

    def time_volume(dt):
        internal = dt.integration_vars
        accepted = 0
        for ordering in itertools.permutations(internal):
            rank = {name: k for k, name in enumerate(ordering)}
            valid = True
            for prop in dt.propagators:
                if prop.kind != "R":
                    continue
                if prop.spatial_left in rank and prop.spatial_right not in rank:
                    valid = False
                if (
                    prop.spatial_left in rank
                    and prop.spatial_right in rank
                    and rank[prop.spatial_left] <= rank[prop.spatial_right]
                ):
                    valid = False
            accepted += valid
        return Fraction(accepted, math.factorial(len(internal)))

    for target in ("moment", "connected"):
        if engine == "symbolic":
            dts = sw.compute_moment(
                obs, action, 2, target=target
            ).diagram_terms_by_order
        else:
            dts = sw.compute_moment_numerical(obs, action, 2, couplings, target=target)
        actual = [
            sum(
                complex(dt.evaluate_coupling(couplings)) * float(time_volume(dt))
                for dt in dts.get(n, [])
            )
            for n in range(3)
        ]
        expected = moments[q] if target == "moment" else _partition_cumulant(q, moments)
        assert actual == pytest.approx([float(x) for x in expected], abs=1e-12)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_connected_zero_evaluate_sweep_export_plot_and_parallel(tmp_path, n_jobs):
    import matplotlib.pyplot as plt

    system = _system()
    expansion = system.expand(
        ("phi(x)", "phi(y)", "phi(z)", "phi(w)"),
        [0, 1],
        target="connected",
        progress=False,
    )
    props = system.propagators(t_max=1, n_grid_t=5, progress=False)
    result = expansion.evaluate(props, positions=dict.fromkeys("xyzw", 0), t_final=0.5)
    assert result.total == 0 and result.by_order == {0: 0, 1: 0}
    assert result.per_diagram == [] and result.by_vertex_type == {}
    assert result.to_dict()["n_diagrams_evaluated"] == 0
    sweep = expansion.sweep(
        props,
        positions_grid={k: [0] for k in "xyzw"},
        t_final_grid=[0.3, 0.5, 0.8],
        orders=[0, 1],
        n_jobs=n_jobs,
    )
    frame = sweep.to_dataframe()
    assert len(frame) == 6
    assert frame[["diagram_idx", "vertex_type", "n_cross_C"]].isna().all().all()
    assert len(sweep.totals()) == 6 and (sweep.totals().value == 0).all()
    assert sweep.by_vertex_type_totals().empty
    path = tmp_path / "zeros.csv"
    sweep.totals().to_csv(path, index=False)
    assert path.read_text().count("\n") == 7
    fig = sweep.plot(x="t_final", hue="order")
    assert all(
        np.array_equal(line.get_ydata(), [0, 0, 0]) for line in fig.axes[0].lines
    )
    plt.close(fig)


def test_connected_zero_yaml_cli_to_totals(tmp_path, capsys):
    from sft_wick.workflow.config import load_workflow_config, run_workflow
    from sft_wick.workflow.cli import _print_summary

    config = tmp_path / "zero.yaml"
    config.write_text("""system:
  field: {name: phi, n_components: 1}
  linear: {type: diagonal, gamma: [1.0]}
  vertices: []
  noise:
    kappa2:
      type: separable_translation
      temporal: {type: exponential, lam: 0.05, sigma_t: 0.3}
      spatial: {type: exponential, sigma_x: 1.0}
expand:
  observable: [phi(x), phi(y), phi(z), phi(w)]
  orders: [0, 1]
  target: connected
propagators:
  t_max: 1.0
  n_grid_t: 5
sweep:
  positions_grid: {x: [0.0], y: [0.0], z: [0.0], w: [0.0]}
  t_final_grid: [0.5, 0.8]
  component_tuples: [[0, 0, 0, 0]]
  orders: [0, 1]
  n_samples: 16
""")
    cfg = load_workflow_config(config)
    _print_summary(cfg)
    assert "target=connected" in capsys.readouterr().out
    sweep, totals = run_workflow(cfg)
    assert len(sweep.rows) == len(totals) == 4
    assert (totals.value == 0).all()
