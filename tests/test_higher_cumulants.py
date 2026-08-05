"""Non-Gaussian driving at cumulant order m >= 4.

``NonLocalVertex`` documents an MSR factor table up to m = 4, and the ML-DMFT
programme needs cumulants above the cubic one.  Every existing test stopped at
m = 3, so the m = 4 row of that table had never been executed -- documented
but unverified.  These tests execute it, and cover the backend combination
that used to be computable by none of them: a callable (spacetime-dependent)
coupling together with a matrix-valued response propagator.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import sft_wick as sw

N = 2


def _kappa(order: int) -> np.ndarray:
    """A bare cumulant tensor with a few distinct non-zero entries."""
    K = np.zeros((N,) * order)
    K[(0,) * order] = 1.0
    K[(1,) * order] = 0.7
    K[(0, 0) + (1,) * (order - 2)] = 0.3
    return K


def _system(order, coupling, gammas=(1.0, 1.0)):
    return sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(gamma=list(gammas)),
        nonlocal_vertices=[
            sw.NonLocalVertex("K", order=order, coupling=coupling)
        ],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
            spatial=sw.ExponentialSpatial(sigma_x=1.0))),
    )


def _evaluate(system, order, method="qmc_vectorized", **kw):
    ext = ("phi_a(x)", "phi_b(y)", "phi_c(z)", "phi_d(w)")[:order]
    exp = system.expand(ext, orders=[1])
    props = system.propagators(t_max=4.0, n_grid_t=25)
    pos = {k: 0.0 for k in ("x", "y", "z", "w")[:order]}
    kw.setdefault("n_samples", 2 ** 12)
    return exp.evaluate(props, positions=pos, t_final=3.0,
                        component_pair=(0,) * order, orders=[1],
                        integrate_over="all", method=method, seed=7, **kw)


# --------------------------------------------------------------------- #
# HC1 — the MSR factor table, including the row nothing had ever run
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("order, expected", [
    (1, -1j),                  # mean drift
    (2, 0.5),                  # Gaussian kernel
    (3, 1j / 6),               # demo2's K
    (4, -1.0 / 24),            # documented but, until now, never executed
    (5, -1j / 120),
    (6, 1.0 / 720),
])
def test_HC1_msr_factor_matches_the_documented_table(order, expected):
    """``-(i^m)/m!`` against the table in the ``NonLocalVertex`` docstring.

    Computed independently here from the i^m cycle rather than by repeating
    the implementation's expression, so this is a check and not an echo.
    """
    v = sw.NonLocalVertex("K", order=order, coupling=np.zeros((N,) * order))
    cycle = [1j, -1.0, -1j, 1.0]                    # i^1, i^2, i^3, i^4
    want = -cycle[(order - 1) % 4] / math.factorial(order)
    assert v.msr_factor == pytest.approx(want)
    assert v.msr_factor == pytest.approx(expected)
    # and it is EXACT, not merely close: python's integer complex power is
    # repeated multiplication, so no spurious imaginary dust leaks into a
    # coupling that the reality check downstream would then have to absorb.
    assert v.msr_factor == want


def test_HC1_msr_factor_is_applied_to_the_coupling_tensor():
    K = _kappa(4)
    v = sw.NonLocalVertex("K", order=4, coupling=K)
    assert v.msr_coupling == pytest.approx((-1.0 / 24) * K)


# --------------------------------------------------------------------- #
# HC2 — a quartic cumulant end to end
# --------------------------------------------------------------------- #

def test_HC2_quartic_cumulant_expands_with_the_right_diagram_count():
    """A 4-point function of a Gaussian theory has exactly 3 Wick pairings,
    and one quartic vertex contributes exactly one first-order diagram."""
    exp = _system(4, _kappa(4)).expand(
        ("phi_a(x)", "phi_b(y)", "phi_c(z)", "phi_d(w)"), orders=[0, 1],
    )
    assert len(exp.diagrams(order=0)) == 3      # <phi phi><phi phi> pairings
    assert len(exp.diagrams(order=1)) == 1
    assert set(exp.by_vertex_type(order=1)) == {"K"}


def test_HC2_quartic_cumulant_evaluates_and_is_real():
    """The reality theorem: with E_psi = 0 and n_R = 4, the phase i^(-E_psi)
    is 1, so the diagram must come out real.  ``_real_or_raise`` enforces it,
    so a complex leak would raise rather than reach this assertion."""
    r = _evaluate(_system(4, _kappa(4)), 4)
    assert np.isfinite(r.total)
    assert abs(r.total) > 0.0, "the quartic vertex contributed nothing"
    assert np.imag(np.asarray(r.total)) == 0.0


def test_HC2_numeric_and_callable_quartic_couplings_agree():
    """Two independent routes to the same integrand: the static-tensor path
    and the per-sample callable path.  They exercise entirely different code
    in ``DiagramIntegrand``, so agreement is a real cross-check.
    """
    K = _kappa(4)
    static = _evaluate(_system(4, K), 4).total
    callable_ = _evaluate(_system(4, lambda n_list, t_list: K), 4).total
    assert static == pytest.approx(callable_, rel=1e-12)


def test_HC2_the_result_is_linear_in_the_quartic_amplitude():
    """First order in kappa^(4) must be exactly linear in it -- a check on
    the MSR factor being applied once and only once."""
    K = _kappa(4)
    one = _evaluate(_system(4, K), 4).total
    two = _evaluate(_system(4, 2.0 * K), 4).total
    assert two == pytest.approx(2.0 * one, rel=1e-12)


# --------------------------------------------------------------------- #
# HC3 — callable coupling + matrix-valued R.  This combination used to be
# computable by NO backend, and the error message sent the user to two that
# also refuse, for the other reason -- a loop of three NotImplementedErrors
# with no exit.  It is the natural ML-DMFT configuration: several components
# with distinct decay rates (matrix R) and a spacetime-dependent cumulant.
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("order", [3, 4])
def test_HC3_a_callable_coupling_works_with_matrix_valued_R(order):
    """The scalar loop is the natural home for the per-sample callable
    contract -- it already visits one sample at a time -- so it materialises
    the coupling per sample rather than refusing it.

    Checked against the STATIC tensor path at the same physical point: the
    callable returns that constant, so the two must agree exactly.  They run
    entirely different code to get there (per-sample materialisation vs the
    prebuilt ``coupling_array``), which is what makes the agreement a check.
    """
    K = _kappa(order)
    static = _evaluate(_system(order, K, gammas=(1.0, 1.3)), order,
                       method="qmc").total
    dynamic = _evaluate(_system(order, lambda n_list, t_list: K,
                                gammas=(1.0, 1.3)), order, method="qmc").total
    assert np.isfinite(static) and abs(static) > 0.0
    assert dynamic == pytest.approx(static, rel=1e-12)


@pytest.mark.parametrize("order", [3, 4])
def test_HC3_the_scalar_path_agrees_with_the_vectorised_one(order):
    """Where both backends are available (isotropic R), the new scalar
    materialisation and the pre-existing batched one must give the same
    number.  Two independently written integrators over the same integrand is
    the strongest check available without an analytic reference.
    """
    K = _kappa(order)
    scalar = _evaluate(_system(order, lambda n_list, t_list: K), order,
                       method="qmc").total
    batched = _evaluate(_system(order, lambda n_list, t_list: K), order,
                        method="qmc_vectorized").total
    assert scalar == pytest.approx(batched, rel=1e-12)


def test_HC3_a_spacetime_dependent_callable_is_actually_evaluated():
    """A path that quietly fell back to the static zeros placeholder, or that
    ignored the callable's arguments, would return the constant answer.  A
    genuinely t-dependent coupling must not."""
    K = _kappa(3)
    const = _evaluate(_system(3, lambda n_list, t_list: K, gammas=(1.0, 1.3)),
                      3, method="qmc").total
    varying = _evaluate(
        _system(3, lambda n_list, t_list: K * float(np.exp(-0.3 * np.sum(t_list))),
                gammas=(1.0, 1.3)), 3, method="qmc").total
    assert abs(varying - const) > 1e-6 * abs(const)
    assert np.isfinite(varying) and abs(varying) > 0.0


def test_HC3_evaluate_still_refuses_a_callable_with_no_coupling_array():
    """``DiagramIntegrand.evaluate`` reads a zeros placeholder on the dynamic
    path, so calling it without an explicit per-sample array would silently
    return 0.  The override is opt-in; the refusal stays for everyone else,
    and it names the way out."""
    from sft_wick.evaluate import DiagramIntegrand

    K = _kappa(3)
    system = _system(3, lambda n_list, t_list: K, gammas=(1.0, 1.3))
    exp = system.expand(("phi_a(x)", "phi_b(y)", "phi_c(z)"), orders=[1])
    assert len(exp.diagrams(order=1)) == 1        # the vertex is really there

    with pytest.raises(NotImplementedError, match="coupling_array"):
        # a bare integrand with a dynamic coupling and no override
        DiagramIntegrand.evaluate(
            _DummyDynamicIntegrand(), {}, {}, None,
        )


class _DummyDynamicIntegrand:
    """Minimum shape for the refusal branch: a non-None dynamic_coupling."""
    dynamic_coupling = object()


# --------------------------------------------------------------------- #
# HC4 — branches of the new scalar callable path that nothing else enters
# --------------------------------------------------------------------- #

def test_HC4_a_vectorized_callable_works_under_the_scalar_loop():
    """`coupling_vectorized=True` declares the BATCHED contract:
    ``(m_legs, n_samples) -> (n_samples,) + (N,)*m``.  ``evaluate_at`` is the
    per-sample entry point and used to ignore the flag, handing such a
    callable ``(m_legs,)`` arrays instead -- and some batched callables
    broadcast happily and return a plausible WRONG shape rather than raising.

    Matrix-valued R forces the scalar loop, so before this the combination
    "vectorized callable + matrix R" had no correct path at all.
    """
    K = _kappa(3)

    def per_sample(n_list, t_list):
        return K * float(np.exp(-0.3 * np.sum(t_list)))

    def batched(n_arr, t_arr):
        w = np.exp(-0.3 * np.sum(t_arr, axis=0))          # (n_samples,)
        return K[None, ...] * w[:, None, None, None]

    def system(coupling, vectorized, gammas):
        return sw.System(
            field=sw.FieldSpec("phi", n_components=N),
            linear=sw.DiagonalA(gamma=list(gammas)),
            nonlocal_vertices=[sw.NonLocalVertex(
                "K", order=3, coupling=coupling,
                coupling_vectorized=vectorized)],
            noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
                spatial=sw.ExponentialSpatial(sigma_x=1.0))),
        )

    # matrix R: only the scalar loop is legal, and both contracts must agree
    plain = _evaluate(system(per_sample, False, (1.0, 1.3)), 3,
                      method="qmc").total
    vec = _evaluate(system(batched, True, (1.0, 1.3)), 3, method="qmc").total
    assert np.isfinite(plain) and abs(plain) > 0.0
    assert vec == pytest.approx(plain, rel=1e-12)

    # isotropic R: cross-check the scalar loop against the batched backend,
    # which has its own independent vectorised materialisation
    scalar = _evaluate(system(batched, True, (1.0, 1.0)), 3,
                       method="qmc").total
    batch = _evaluate(system(batched, True, (1.0, 1.0)), 3,
                      method="qmc_vectorized").total
    assert scalar == pytest.approx(batch, rel=1e-12)


def test_HC4_an_equal_time_callable_gets_its_aliased_leg_times():
    """An ``equal_time=True`` non-local vertex shares ONE integration variable
    across its legs; the other legs are aliases resolved from a canonical
    representative.  ``dynamic_coupling_array`` fills those aliases itself, as
    ``evaluate`` does -- a leg left unfilled would raise, and a leg filled
    with the wrong time would silently change the answer.
    """
    K = _kappa(3)
    seen_times = []

    def recording(n_list, t_list):
        seen_times.append(tuple(np.ravel(np.asarray(t_list, dtype=float))))
        return K

    system = sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(gamma=[1.0, 1.3]),
        nonlocal_vertices=[sw.NonLocalVertex("K", order=3, coupling=recording,
                                             equal_time=True)],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
            spatial=sw.ExponentialSpatial(sigma_x=1.0))),
    )
    total = _evaluate(system, 3, method="qmc", n_samples=2 ** 9).total
    assert np.isfinite(total)
    assert seen_times, "the callable was never invoked"
    # equal_time means every leg of a given sample shares one time
    for ts in seen_times:
        assert len(ts) == 3
        assert ts[0] == pytest.approx(ts[1]) == pytest.approx(ts[2]), ts
