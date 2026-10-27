"""Independent regressions for cuts on swept external times.

References are elementary integrals of Brownian covariance and exponentials,
with no use of the diagram integrator or its bound/split helpers.
"""

from dataclasses import dataclass, replace

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.workflow.specs import ConstantImpulse
from tests.test_gl_chunked_integration import (
    K_RAW,
    POS3,
    T_MIN,
    PairExponentialKappa,
    ZeroC,
    _cumulant,
    _noise,
)


@pytest.mark.parametrize(
    "method,chunk",
    [
        ("gauss_legendre", None),
        ("gauss_legendre", 3),
        ("nquad", None),
    ],
)
def test_lower_cut_resolves_max_on_earlier_swept_external(method, chunk):
    from sft_wick.expressions import Propagator, Rational
    from sft_wick.perturbation import DiagramTerm
    from sft_wick.evaluate import integrate_moment

    _, props, _ = _brownian(("phi(x)", "phi(y)"), [0])
    term = DiagramTerm(
        propagators=(
            Propagator("C", None, None, "x", "y"),
            Propagator("C", None, None, "y", "z"),
        ),
        coupling_sum=Rational(1),
        rational_prefactor=Rational(1),
        integration_vars=(),
        summation_indices=(),
        n_response=0,
    )
    c = 0.37
    # Integral_0^1 min(y,c) [Integral_0^1 min(x,y) dx] dy.
    ref = c / 3 - c**3 / 6 + c**4 / 24
    got, _ = integrate_moment(
        term.build_integrand({}),
        T_MIN + 1.0,
        props.cache,
        t_min=T_MIN,
        integrate_over={"x", "y"},
        external_times={"z": T_MIN + c},
        method=method,
        n_gauss=3,
        gl_chunk_size=chunk,
    )
    assert got == pytest.approx(ref, abs=2e-15, rel=0.0)


@pytest.mark.parametrize(
    "method,chunk",
    [
        ("gauss_legendre", None),
        ("gauss_legendre", 3),
        ("nquad", None),
    ],
)
def test_multiple_swept_lower_bounds_split_without_any_kernel_kink(method, chunk):
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=1),
        linear=sw.DiagonalA(gamma=[0.0]),
        noise=_noise(),
        t_min=T_MIN,
        vertices=[sw.LocalVertex("F", coupling=np.full((1, 1, 1), 0.5))],
    )
    exp = system.expand(("phi(y)", "psi(x)", "psi(w)"), orders=[1])
    props = system.propagators(
        t_max=2.0,
        c_closed_form=ZeroC(1),
        c_closed_form_only=True,
        c_closed_form_vectorized=True,
        progress=False,
    )
    a = 0.45
    # Unit R and 2F=1: volume of 0<=x,w<=v<=a is Integral_0^a v^2 dv.
    got = exp.evaluate(
        props,
        positions=dict.fromkeys("yxw", 0.0),
        t_final=T_MIN + 1.2,
        component_pair=(0, 0, 0),
        orders=[1],
        integrate_over={"x", "w"},
        external_times={"y": T_MIN + a},
        method=method,
        n_gauss=2,
        gl_chunk_size=chunk,
    )
    assert got.total == pytest.approx(a**3 / 3, abs=2e-15, rel=0.0)


def _pair_integral(c, L, beta):
    """Integral_0^L exp(-beta*abs(u-c)) du, including c outside the domain."""
    if c <= 0:
        return np.exp(beta * c) * (-np.expm1(-beta * L)) / beta
    if c >= L:
        return np.exp(-beta * (c - L)) * (-np.expm1(-beta * L)) / beta
    return (2 - np.exp(-beta * c) - np.exp(-beta * (L - c))) / beta


@pytest.mark.parametrize(
    "method,chunk",
    [
        ("gauss_legendre", None),
        ("gauss_legendre", 3),
        ("nquad", None),
    ],
)
@pytest.mark.parametrize(
    "a,b", [(0.25, 0.85), (0.85, 0.25), (0.45, 0.45), (0.0, 1.2), (1.5, 0.45)]
)
def test_multiple_fixed_cumulant_cuts_including_boundaries_and_beyond_sweep_limit(
    method, chunk, a, b
):
    calls = []
    kernel = PairExponentialKappa(calls)
    exp, props = _cumulant(kernel, vectorized=True, absorbed=True)
    L, beta, comps = 1.2, kernel.beta, (0, 1, 1)
    ref = (
        K_RAW[comps]
        * (
            _pair_integral(a, L, beta)
            + _pair_integral(b, L, beta)
            + L * np.exp(-beta * abs(a - b))
        )
        / 3
    )
    calls.clear()
    got = exp.evaluate(
        props,
        positions=POS3,
        t_final=T_MIN + L,
        component_pair=comps,
        orders=[1],
        integrate_over={"x"},
        external_times={"y": T_MIN + a, "z": T_MIN + b},
        method=method,
        n_gauss=8,
        gl_chunk_size=chunk,
    )
    assert got.total == pytest.approx(ref, rel=2e-12, abs=0.0)
    if chunk is not None:
        assert calls and max(calls) <= chunk


@dataclass
class BrownianC:
    calls: list
    sigma: float = 1.0
    has_diagonal_kink = True

    def __call__(self, x1, t1, x2, t2):
        left, right = np.broadcast_arrays(np.asarray(t1), np.asarray(t2))
        if left.ndim:
            self.calls.append(left.size)
        value = self.sigma * np.maximum(np.minimum(left, right) - T_MIN, 0.0)
        return value[..., None, None]


def _brownian(observable, orders, *, b=None):
    calls = []
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=1),
        linear=sw.DiagonalA(gamma=[0.0]),
        vertices=(
            [] if b is None else [sw.LocalVertex("F", coupling=np.full((1, 1, 1), b))]
        ),
        noise=replace(_noise(), sigma2=ConstantImpulse(1.0)),
        t_min=T_MIN,
    )
    exp = system.expand(observable, orders=orders)
    props = system.propagators(
        t_max=2.0,
        c_closed_form=BrownianC(calls),
        c_closed_form_only=True,
        c_closed_form_vectorized=True,
        progress=False,
    )
    calls.clear()
    return exp, props, calls


def _brownian_integral(a, L):
    """Integral_0^L min(u,a) du for a>=0."""
    return L**2 / 2 if a >= L else a * L - a**2 / 2


@pytest.mark.parametrize(
    "method,chunk",
    [
        ("gauss_legendre", None),
        ("gauss_legendre", 3),
        ("nquad", None),
    ],
)
@pytest.mark.parametrize("a", [0.45, 0.0, 1.2, 1.5])
def test_free_brownian_C_kink_on_a_swept_external_and_fixed_external(method, chunk, a):
    exp, props, calls = _brownian(("phi(x)", "phi(y)"), [0])
    L = 1.2
    ref = _brownian_integral(a, L)
    got = exp.evaluate(
        props,
        positions={"x": 0.0, "y": 0.0},
        t_final=T_MIN + L,
        component_pair=(0, 0),
        orders=[0],
        integrate_over={"x"},
        external_times={"y": T_MIN + a},
        method=method,
        n_gauss=3,
        gl_chunk_size=chunk,
    )
    assert got.total == pytest.approx(ref, rel=3e-14, abs=1e-15)
    if chunk is not None:
        assert calls and max(calls) <= chunk


@pytest.mark.parametrize(
    "method,chunk",
    [
        ("gauss_legendre", None),
        ("gauss_legendre", 3),
        ("nquad", None),
    ],
)
def test_four_point_gaussian_has_swept_diagonal_and_fixed_external_C_cuts(
    method, chunk
):
    exp, props, calls = _brownian(("phi(x)", "phi(y)", "phi(z)", "phi(w)"), [0])
    L, a, b = 1.2, 0.25, 0.85
    # Wick's Gaussian formula is independent of sft-wick's enumeration:
    # min(x,y)min(a,b)+min(x,a)min(y,b)+min(x,b)min(y,a).
    ref = min(a, b) * L**3 / 3 + 2 * _brownian_integral(a, L) * _brownian_integral(b, L)
    got = exp.evaluate(
        props,
        positions=dict.fromkeys("xyzw", 0.0),
        t_final=T_MIN + L,
        component_pair=(0,) * 4,
        orders=[0],
        integrate_over={"x", "y"},
        external_times={"z": T_MIN + a, "w": T_MIN + b},
        method=method,
        n_gauss=3,
        gl_chunk_size=chunk,
    )
    assert got.total == pytest.approx(ref, rel=3e-13, abs=0.0)
    if chunk is not None:
        assert calls and max(calls) <= chunk


@pytest.mark.parametrize(
    "method,chunk",
    [
        ("gauss_legendre", None),
        ("gauss_legendre", 7),
        ("nquad", None),
    ],
)
def test_cut_propagates_through_internal_vertex_to_swept_external_causal_chain(
    method, chunk
):
    b, L, a = 0.4, 1.2, 0.45
    exp, props, calls = _brownian(("phi(x)", "psi(w)", "phi(y)"), [1], b=b)
    # At O(b), expand the quadratic drift in the Gaussian process and its
    # response. The two response-chain and two mean terms integrate to:
    ref = b * (a * L**3 / 2 + a**2 * L**2 / 4 + L * a**3 / 3 - a**4 / 12)
    got = exp.evaluate(
        props,
        positions=dict.fromkeys("xwy", 0.0),
        t_final=T_MIN + L,
        component_pair=(0,) * 3,
        orders=[1],
        integrate_over={"x", "w"},
        external_times={"y": T_MIN + a},
        method=method,
        n_gauss=4,
        gl_chunk_size=chunk,
    )
    assert got.total == pytest.approx(ref, rel=3e-13, abs=0.0)
    if chunk is not None:
        assert calls and max(calls) <= chunk


@dataclass
class ScalarPairKappa:
    calls: list
    beta: float = 1.3
    has_coincident_time_kinks = True

    def __call__(self, n_array, t_array):
        t = np.asarray(t_array)
        self.calls.append(t.shape[1])
        env = (
            sum(
                np.exp(-self.beta * np.abs(t[i] - t[j]))
                for i, j in ((0, 1), (0, 2), (1, 2))
            )
            / 3
        )
        return env[:, None, None, None]


@pytest.mark.parametrize(
    "method,chunk",
    [
        ("gauss_legendre", None),
        ("gauss_legendre", 3),
        ("nquad", None),
    ],
)
def test_existing_internal_FK_cut_retains_its_exact_reference(method, chunk):
    calls = []
    kernel = ScalarPairKappa(calls)
    b, L, a, beta = 0.4, 1.2, 0.45, kernel.beta
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=1),
        linear=sw.DiagonalA(gamma=[0.0]),
        vertices=[sw.LocalVertex("F", coupling=np.full((1, 1, 1), b))],
        nonlocal_vertices=[
            sw.NonLocalVertex(
                "K",
                order=3,
                coupling=kernel,
                coupling_vectorized=True,
                already_R_contracted=True,
            )
        ],
        noise=_noise(),
        t_min=T_MIN,
    )
    exp = system.expand(("phi(x)", "phi(y)"), orders=[2])
    props = system.propagators(
        t_max=2.0,
        c_closed_form=ZeroC(1),
        c_closed_form_only=True,
        c_closed_form_vectorized=True,
        progress=False,
    )
    # Perturbative covariance = b int_0^L K(s,s,a) ds
    #                           + b int_0^a K(L,s,s) ds.
    pair_area = _pair_integral(a, L, beta)
    tail_area = (np.exp(-beta * (L - a)) - np.exp(-beta * L)) / beta
    ref = b * ((L + 2 * pair_area) + (a + 2 * tail_area)) / 3
    calls.clear()
    got = exp.evaluate(
        props,
        positions={"x": 0.0, "y": 0.3},
        t_final=T_MIN + L,
        external_times={"y": T_MIN + a},
        component_pair=(0, 0),
        orders=[2],
        vertex_types=["FK"],
        method=method,
        n_gauss=8,
        gl_chunk_size=chunk,
    )
    assert got.total == pytest.approx(ref, rel=2e-12, abs=0.0)
    if chunk is not None:
        assert calls and max(calls) <= chunk
