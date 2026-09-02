"""Repeated external spatial labels are refused, not silently mis-counted.

Writing two or more external operators with the SAME spatial label --
``("phi_a(x)", "phi_b(x))`` rather than ``("phi_a(x)", "phi_b(y)")`` --
used to return a silently wrong number.  The spatial Wick engine
enumerates topologies keyed by spatial LABEL and recovers the
operator-level count with a multiplicity
(``wick._compute_multiplicity``), which explicitly excludes observable
points on the assumption that each carries exactly one operator.  Two
externals sharing a label break that assumption.

Measured on ``main`` at d4df86c, demo2's system, t = 3.48, both
externals at position 0:

===============  ===================  ====================  =====
channel          distinct labels      coincident labels     ratio
===============  ===================  ====================  =====
order 0          1.15181265e-02       1.15181265e-02        1
order 2, F,  aa  3.67653928e-04       1.83826964e-04        2
order 2, FK, ab  7.60458863e-04       7.60458863e-04        1
===============  ===================  ====================  =====

-- inconsistent between channels, which is what made it dangerous.

**Why the obvious fix is wrong.**  It is tempting to carry a
multiplicity through the collapse.  That is only valid when the
coupling is symmetric in the affected indices.  The three-point case
shows what is actually lost -- not a factor but a SUM over the
assignments of external operators to legs, each with its own
component-index routing:

    ("phi_a(x)", "phi_b(y)", "phi_c(z)")  ->  K_abc + K_acb + K_bac
                                              + K_bca + K_cab + K_cba
    ("phi_a(x)", "phi_b(x)", "phi_c(x)")  ->  K_abc

``6 * K_abc`` equals that sum only for a K symmetric under all six
permutations.  demo2's kappa^3 happens to be (it is component-diagonal),
and in the two-point F channel with ``a = b`` the two routings coincide
-- which is why the observed ratios above are clean integers.  Neither
is a general rule.

Supporting coincident labels properly means keeping external operators
distinguishable through the label-keyed enumeration in
``wick.wick_contract_spatial``, so every routing is enumerated rather
than reconstructed from a count.  Until that lands, the spelling is
refused at both L1 and L0.  Nothing in the package, the demos or the
paper uses it: they all spell observables with distinct labels.
"""
from __future__ import annotations

import numpy as np
import pytest

import sft_wick as sw

N = 2


def _system():
    F = np.zeros((N, N, N))
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = F[1, 1, 0] = 0.5
    return sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
            spatial=sw.ExponentialSpatial(sigma_x=1.0))),
    )


@pytest.mark.parametrize("observable", [
    ("phi_a(x)", "phi_b(x)"),
    ("phi_a(x)", "phi_b(y)", "phi_c(y)"),
    ("phi_a(x)", "phi_b(x)", "phi_c(x)"),
])
def test_CE1_repeated_external_label_is_refused_at_L1(observable):
    """``System.expand`` refuses before producing any diagram."""
    sw.reset_uid_counter()
    with pytest.raises(ValueError, match="same spatial label"):
        _system().expand(observable, orders=[0, 2])


def test_CE1_distinct_labels_still_work():
    """The guard must not fire on the spelling everything actually uses."""
    sw.reset_uid_counter()
    expansion = _system().expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2])
    assert len(expansion.dts_by_order[2]) == 6


def test_CE1_repeated_external_label_is_refused_at_L0():
    """The raw API is guarded too -- ``compute_moment`` is a public
    entry point and an L1 check would not protect it."""
    from sft_wick import Action, Field, compute_moment
    sw.reset_uid_counter()
    phi = Field("phi", "physical", n_components=N)
    obs = [phi("a", "x"), phi("b", "x")]
    with pytest.raises(ValueError, match="same spatial label"):
        compute_moment(obs, Action(vertices=[]), order=1)


def test_CE1_error_names_the_label_and_the_fix():
    """A refusal is only useful if it says which label and what to do."""
    sw.reset_uid_counter()
    with pytest.raises(ValueError) as exc:
        _system().expand(("phi_a(x)", "phi_b(x)"), orders=[0, 2])
    msg = str(exc.value)
    assert "'x'" in msg
    assert "distinct" in msg
    assert "positions" in msg


def test_CE2_order_zero_is_exempt_and_correct():
    """Order 0 keeps every routing at coincident labels, so it is NOT
    refused -- and the exemption is checked, not assumed.

    The free-theory contraction has no vertices for the downstream
    diagram-isomorphism pass to merge, which is where the loss actually
    happens.  Two spellings of the same free four-point function must
    give the same set of terms:

        <phi_a(w) phi_b(x) phi_c(y) phi_d(z)>
            -> C_ab C_cd + C_ac C_bd + C_ad C_bc
        <phi_a(x) phi_b(x) phi_c(x) phi_d(x)>
            -> C_ab C_cd + C_ac C_bd + C_ad C_bc     (all three kept)

    and the scalar case must keep its multiplicity as a factor:

        <psi(w) phi(x) phi(y) phi(z)>
            -> R(x,w)C(y,z) + R(y,w)C(x,z) + R(z,w)C(x,y)
        <psi(x) phi(x) phi(x) phi(x)>  ->  3 R(x,x) C(x,x)

    The equal-point Itô tests in ``test_perturbation.py`` and
    ``test_deductive_numerics.py`` depend on this exemption.
    """
    from sft_wick import Action, Field, compute_moment
    A0 = Action(vertices=[])

    sw.reset_uid_counter()
    phi = Field("phi", "physical", n_components=2)
    coincident = compute_moment(
        [phi("a", "x"), phi("b", "x"), phi("c", "x"), phi("d", "x")],
        A0, order=0, response_phase=False,
    ).order(0).to_latex()
    for pair in ("C_{ab}", "C_{cd}", "C_{ac}", "C_{bd}", "C_{ad}", "C_{bc}"):
        assert pair in coincident, f"{pair} missing from {coincident}"

    sw.reset_uid_counter()
    phi_s = Field("phi", "physical")
    psi_s = Field("psi", "response")
    scalar = compute_moment(
        [psi_s("x"), phi_s("x"), phi_s("x"), phi_s("x")],
        A0, order=0, ito=False, response_phase=False,
    ).order(0).to_latex()
    assert "3" in scalar and "R(x, x)" in scalar and "C(x, x)" in scalar, scalar
