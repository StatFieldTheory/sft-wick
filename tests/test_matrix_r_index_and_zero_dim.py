"""Matrix-valued R: per-propagator component indices, and the
zero-dimensional dynamic-coupling branch.

Two defects of the matrix-R path, both invisible to the scalar-R path
(which never resolves component indices at all):

1. ``DiagramIntegrand._evaluate_r_product_general`` looked each R factor's
   indices up by its ``(spatial_left, spatial_right)`` endpoints.  Endpoints
   do not identify an R propagator --- a local vertex with two ψ legs puts
   *two* of them between the same two points --- so both factors of such a
   pair read the first match's indices and the diagram silently evaluated
   ``R[j,l] * R[j,l]`` instead of ``R[j,l] * R[k,m]``.

2. ``DiagramIntegrand._evaluate_zero_dimensional``'s dynamic-coupling branch
   multiplied ``cache.R_time_batch``, which is scalar-only, so a matrix R
   raised ``ValueError: setting an array element with a sequence``.  The
   batched backends then refused matrix R explicitly, but their refusals
   sat after their ``n_total == 0`` early return, so none of them fired.
   (The batched backends have since learned matrix R; see
   ``tests/test_matrix_r_batched.py``.)
"""

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.evaluate import (
    PropagatorCache,
    PropagatorModel,
    integrate_diagrams,
)

N = 2

#: A deliberately generic R: non-symmetric, no zero or repeated entry, so
#: R[j,l] * R[j,l] and R[j,l] * R[k,m] cannot coincide numerically.
R_GENERIC = np.array([[0.7, -0.4], [0.25, 1.3]])

#: Position-independent κ², so the reference contraction below needs no
#: direction bookkeeping of its own.
KAPPA_MAT = np.array([[1.0, 0.3], [0.3, 0.8]])


def _cache(R_mat: np.ndarray) -> PropagatorCache:
    """A bare matrix-R cache — no spatial table, no interpolation."""

    def R_time(t1, t2):  # noqa: ARG001
        return np.asarray(R_mat, dtype=float)

    def kappa2(n1, t1, n2, t2):  # noqa: ARG001
        return KAPPA_MAT * np.exp(-abs(t1 - t2))

    return PropagatorCache(
        PropagatorModel(
            R_time=R_time,
            kappa2=kappa2,
            n_components=N,
            iso_R=False,
            diag_C=False,
        )
    )


def _two_psi_vertex_terms(diag_R: bool = False) -> list:
    """Order-2 ⟨φ_a(x) φ_b(y)⟩ from a ψφφ and a ψψφ local vertex.

    The ψψφ vertex is the one the L1/L2 workflow cannot build; it is what
    puts two R propagators between the same two internal points.
    """
    sw.reset_uid_counter()
    phi = sw.Field("phi", "physical", n_components=N)
    psi = sw.Field("psi", "response", n_components=N)
    action = sw.Action(
        vertices=[
            sw.Vertex(fields=[psi, phi, phi], coupling="F", local=True),
            sw.Vertex(fields=[psi, psi, phi], coupling="G", local=True),
        ]
    )
    return sw.compute_moment(
        [phi("a", "x"), phi("b", "y")], action, order=2, diag_R=diag_R,
    ).diagram_terms(2)


def _coupling_values() -> dict:
    rng = np.random.default_rng(7)
    F = rng.normal(size=(N, N, N))
    G = rng.normal(size=(N, N, N))
    return {"F": -1j * F, "G": -1.0 * G}


#: Times with t_x > t_y > t_{y_0} > t_{y_1}, so every R factor of the
#: repeated-pair diagrams is causal and none of them is killed by Θ.
TIMES = {"x": 0.9, "y": 0.8, "y_0": 0.6, "y_1": 0.3}


def _repeated_pair_terms(terms: list) -> list[int]:
    """Indices of the diagrams carrying two R propagators on one pair."""
    out = []
    for i, dt in enumerate(terms):
        r_props = [p for p in dt.propagators if p.kind == "R"]
        pairs = [(p.spatial_left, p.spatial_right) for p in r_props]
        if len(pairs) != len(set(pairs)):
            out.append(i)
    return out


def _reference_value(dt, ig, times, cache, *, by_endpoint=False) -> complex:
    """Contract the diagram by hand from its symbolic content.

    Reads only ``dt.propagators``, ``dt.propagator_indices`` and the
    integrand's coupling tensor — none of the spatial-analysis machinery
    under test.  κ² is position independent here, so every C propagator is
    looked up at a single dummy direction.

    With ``by_endpoint=True`` this reproduces the **pre-fix** rule: resolve
    every R factor's indices through the first propagator sharing its
    endpoints.  The two differ exactly on a repeated pair.
    """
    idx_names = [name for name, _ in dt.propagator_indices]
    shape = tuple(dim for _, dim in dt.propagator_indices)
    coeff = np.asarray(ig.coupling_array)
    r_props = [p for p in dt.propagators if p.kind == "R"]
    c_props = [p for p in dt.propagators if p.kind == "C"]

    total = 0j
    for pidx in (np.ndindex(*shape) if shape else [()]):
        idx = dict(ig.fixed_indices)
        idx.update(dict(zip(idx_names, pidx)))
        val = complex(coeff[pidx] if shape else coeff)
        if val == 0:
            continue
        for p in r_props:
            t_l, t_r = times[p.spatial_left], times[p.spatial_right]
            if not t_l > t_r:            # retarded + Itô
                val = 0j
                break
            src = p
            if by_endpoint:
                src = next(q for q in r_props
                           if (q.spatial_left, q.spatial_right)
                           == (p.spatial_left, p.spatial_right))
            R_mat = np.asarray(cache.R_time(t_l, t_r))
            val *= R_mat[idx[src.index_left], idx[src.index_right]]
        else:
            for p in c_props:
                C_mat = cache.C_value(0.0, times[p.spatial_left],
                                      0.0, times[p.spatial_right])
                val *= C_mat[idx[p.index_left], idx[p.index_right]]
        total += val
    return total


def _directions(ig) -> dict:
    return {d: 0.0 for d in set(ig.spatial.direction_map.values())}


# --------------------------------------------------------------------------
# 1. Repeated R pair: each factor keeps its own component indices.
# --------------------------------------------------------------------------

def test_two_psi_leg_vertex_produces_a_repeated_r_pair():
    """The premise of the other tests: the ψψφ vertex really does put two
    R propagators between one pair of points, with *different* indices."""
    terms = _two_psi_vertex_terms()
    assert len(terms) == 11
    assert _repeated_pair_terms(terms) == [7, 9]

    for i in (7, 9):
        r_props = [p for p in terms[i].propagators if p.kind == "R"]
        on_internal = [p for p in r_props
                       if (p.spatial_left, p.spatial_right) == ("y_0", "y_1")]
        assert len(on_internal) == 2
        # Same endpoints, different component indices — which is exactly
        # what an endpoint lookup cannot distinguish.
        assert {(p.index_left, p.index_right) for p in on_internal} == {
            ("i_1", "i_3"), ("i_2", "i_4"),
        }


def test_kept_r_propagator_objects_returns_every_factor():
    """``_kept_r_propagator_objects`` must return one entry per R
    propagator, not one per distinct endpoint pair, and must line up
    element-for-element with ``spatial.r_propagators``."""
    terms = _two_psi_vertex_terms()
    cv = _coupling_values()
    for i in (7, 9):
        ig = terms[i].build_integrand(cv, {"a": 0, "b": 1})
        kept = ig._kept_r_propagator_objects()
        assert len(kept) == 3
        assert tuple((p.spatial_left, p.spatial_right) for p in kept) \
            == ig.spatial.r_propagators
        # The two factors on the repeated pair are distinct objects.
        assert len({(p.index_left, p.index_right) for p in kept}) == 3


@pytest.mark.parametrize("term_index", [7, 9])
@pytest.mark.parametrize("fixed", [{"a": 0, "b": 0}, {"a": 0, "b": 1}])
def test_repeated_pair_uses_each_factors_own_matrix_element(term_index, fixed):
    """Against a hand-written numpy contraction of the same diagram.

    The pre-fix endpoint lookup is evaluated alongside it and shown to
    differ, so this cannot pass on the old code.
    """
    terms = _two_psi_vertex_terms()
    cv = _coupling_values()
    cache = _cache(R_GENERIC)

    dt = terms[term_index]
    ig = dt.build_integrand(cv, fixed)
    got = ig.evaluate(TIMES, _directions(ig), cache)

    ref = _reference_value(dt, ig, TIMES, cache)
    old = _reference_value(dt, ig, TIMES, cache, by_endpoint=True)

    assert got == pytest.approx(ref, rel=1e-12, abs=1e-14)
    # Not a vacuous check: the diagram is non-zero and the old rule gave a
    # different answer for it.
    assert abs(ref) > 1e-3
    assert abs(ref - old) > 1e-3


def test_non_repeated_pairs_are_unchanged():
    """Every diagram *without* a repeated pair is untouched by the fix:
    there the endpoint lookup and the per-object read agree."""
    terms = _two_psi_vertex_terms()
    cv = _coupling_values()
    cache = _cache(R_GENERIC)
    checked = 0
    for i, dt in enumerate(terms):
        if i in (7, 9):
            continue
        ig = dt.build_integrand(cv, {"a": 0, "b": 1})
        got = ig.evaluate(TIMES, _directions(ig), cache)
        ref = _reference_value(dt, ig, TIMES, cache)
        old = _reference_value(dt, ig, TIMES, cache, by_endpoint=True)
        assert got == pytest.approx(ref, rel=1e-12, abs=1e-14)
        assert ref == pytest.approx(old, rel=1e-12, abs=1e-14)
        checked += 1
    assert checked == 9


@pytest.mark.parametrize("fixed", [{"a": 0, "b": 0}, {"a": 0, "b": 1}])
def test_identity_matrix_r_agrees_with_diag_r(fixed):
    """With R = 1, ``diag_R=True`` is exact, so the default-flag expansion
    (which sums over the off-diagonal R components as well) must reproduce
    it term by term.

    This is the invariant the endpoint lookup broke: before the fix the two
    sides parted company on diagrams 7 and 9 — the repeated-pair ones.
    """
    cache = _cache(np.eye(N))
    cv = _coupling_values()
    default_terms = _two_psi_vertex_terms(diag_R=False)
    diag_terms = _two_psi_vertex_terms(diag_R=True)
    assert len(default_terms) == len(diag_terms) == 11

    nonzero = 0
    for dt_d, dt_g in zip(default_terms, diag_terms):
        ig_d = dt_d.build_integrand(cv, fixed)
        ig_g = dt_g.build_integrand(cv, fixed)
        v_d = ig_d.evaluate(TIMES, _directions(ig_d), cache)
        v_g = ig_g.evaluate(TIMES, _directions(ig_g), cache)
        # A deliberate non-zero abs: diagram 10's time ordering is acausal
        # at these times, so both sides are a structural zero there and a
        # pure relative check would have nothing to compare.
        assert v_d == pytest.approx(v_g, rel=1e-12, abs=1e-14)
        nonzero += abs(v_d) > 1e-6
    assert nonzero >= 10


# --------------------------------------------------------------------------
# 2. Zero-dimensional integrand with a callable coupling and a matrix R.
# --------------------------------------------------------------------------

def _zero_dim_terms() -> list:
    """Three order-1 diagrams with no surviving time-integration variable.

    ``already_R_contracted=True`` absorbs both κ legs' R propagators, and
    the accompanying time aliases pin them to fixed external points, so
    nothing is left to integrate over.
    """
    sw.reset_uid_counter()
    phi = sw.Field("phi", "physical", n_components=N)
    psi = sw.Field("psi", "response", n_components=N)
    action = sw.Action(
        vertices=[
            sw.Vertex(fields=[psi, psi], coupling="K", local=False,
                      already_R_contracted=True),
        ]
    )
    observable = [phi("a", "x1"), phi("b", "x2"), phi("c", "x3"),
                  psi("d", "x4")]
    return sw.compute_moment(
        observable, action, order=1, diag_R=True,
    ).diagram_terms(1)


ZERO_DIM_FIXED = {"a": 0, "b": 1, "c": 0, "d": 0}
ZERO_DIM_EXTERNAL_TIMES = {"x4": 0.5}

#: Closed form for the setup below, with R = 1 and K ≡ 1/2 everywhere.
#:
#: Each diagram is ``-(δ_{·d}) (K_{··} + K_{··}) R_{··}(x_i, x4)`` times the
#: response phase, with the two absorbed R factors contributing 1.  The δ
#: kills diagram 1 (b = 1 ≠ d = 0) and leaves diagrams 0 and 2, each of
#: which is ``(0.5 + 0.5) × R[0,0](1.0, 0.5) = 1``.  Total: 2.
ZERO_DIM_EXPECTED = 2.0

ZERO_DIM_METHODS = ["qmc_scalar", "qmc_vectorized", "gauss_legendre", "nquad"]


def test_zero_dimensional_terms_have_no_integration_variable():
    """The premise: these diagrams really are zero-dimensional, which is
    what carries them past every backend's matrix-R refusal."""
    terms = _zero_dim_terms()
    assert len(terms) == 3
    for dt in terms:
        ig = dt.build_integrand({"K": 0.5 * np.ones((N, N))}, ZERO_DIM_FIXED)
        assert ig.spatial.time_integration_vars == ()
        assert ig.spatial.external_points == ("x1", "x2", "x3", "x4")
        assert len(ig.spatial.r_absorbed_pairs) == 2


@pytest.mark.parametrize("method", ZERO_DIM_METHODS)
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "callable"])
def test_zero_dimensional_matrix_r_coupling(method, dynamic):
    """A callable κ must give the same answer as the static tensor it
    returns, on every backend.

    Before the fix the callable raised ``ValueError: setting an array
    element with a sequence`` on ``qmc_vectorized``, ``gauss_legendre`` and
    ``nquad`` — the scalar-only ``R_time_batch`` being handed a 2×2 R.
    """
    static_K = 0.5 * np.ones((N, N))
    K = ((lambda n_list, t_list: 0.5 * np.ones((N, N)))  # noqa: ARG005
         if dynamic else static_K)

    total, details = integrate_diagrams(
        _zero_dim_terms(), {"K": K},
        lambda_f=1.0, cache=_cache(np.eye(N)), method=method,
        n_samples=2**8, seed=3, fixed_indices=ZERO_DIM_FIXED,
        external_times=ZERO_DIM_EXTERNAL_TIMES,
    )

    assert total == pytest.approx(ZERO_DIM_EXPECTED, rel=1e-12, abs=1e-14)
    # Per-diagram, so the total cannot be right by cancellation: the δ_{bd}
    # diagram vanishes and the other two contribute 1 each.
    assert [v for v, _ in details] == pytest.approx(
        [1.0, 0.0, 1.0], rel=1e-12, abs=1e-14,
    )


def test_zero_dimensional_callable_matches_static_exactly():
    """Same integrand, both coupling contracts, one comparison — the
    callable path must not merely be finite, it must be *right*."""
    cache = _cache(np.eye(N))
    kwargs = dict(
        lambda_f=1.0, cache=cache, method="gauss_legendre", n_samples=2**8,
        seed=3, fixed_indices=ZERO_DIM_FIXED,
        external_times=ZERO_DIM_EXTERNAL_TIMES,
    )
    static_total, _ = integrate_diagrams(
        _zero_dim_terms(), {"K": 0.5 * np.ones((N, N))}, **kwargs,
    )
    dynamic_total, _ = integrate_diagrams(
        _zero_dim_terms(),
        {"K": lambda n_list, t_list: 0.5 * np.ones((N, N))},  # noqa: ARG005
        **kwargs,
    )
    assert dynamic_total == pytest.approx(static_total, rel=1e-12, abs=1e-14)
    assert static_total == pytest.approx(ZERO_DIM_EXPECTED,
                                         rel=1e-12, abs=1e-14)


def _zero_dim_terms_summed_indices() -> list:
    """The same three diagrams without ``diag_R``.

    The surviving R keeps two different labels and is read off its
    diagonal.  The two absorbed κ legs carry summation indices (``i_0``,
    ``i_1``) in the contraction, and ``compute_moment`` pins them to their
    partners' labels: an absorbed R stands in for that Kronecker delta.
    """
    sw.reset_uid_counter()
    phi = sw.Field("phi", "physical", n_components=N)
    psi = sw.Field("psi", "response", n_components=N)
    action = sw.Action(
        vertices=[
            sw.Vertex(fields=[psi, psi], coupling="K", local=False,
                      already_R_contracted=True),
        ]
    )
    observable = [phi("a", "x1"), phi("b", "x2"), phi("c", "x3"),
                  psi("d", "x4")]
    return sw.compute_moment(
        observable, action, order=1, diag_R=False,
    ).diagram_terms(1)


#: Per-diagram closed form without ``diag_R``, R = R_GENERIC, K ≡ 1/2 at
#: (a,b,c,d) = (0,1,0,0): each diagram keeps one R, ``R_{ad}(x1,x4)``,
#: ``R_{bd}(x2,x4)`` or ``R_{cd}(x3,x4)``, and its two absorbed legs take
#: their partners' components, so ``(K + K) = 1`` multiplies R[0,0], R[1,0]
#: and R[0,0]: total 1.65.
SUMMED_EXPECTED = [R_GENERIC[0, 0], R_GENERIC[1, 0], R_GENERIC[0, 0]]


@pytest.mark.parametrize("method", ZERO_DIM_METHODS)
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "callable"])
def test_zero_dimensional_matrix_r_without_diag_r(method, dynamic):
    """Without ``diag_R`` the kept R is read off its diagonal (b = 1, d = 0)
    and the absorbed legs take their partners' components, on every backend.

    An absorbed R stands in for the Kronecker delta between its partner's
    component and its leg's; ``compute_moment`` now writes that delta into
    the terms whatever ``diag_R`` says.  Evaluating the legs' own summation
    indices instead sums them, ``Σ_{i₀i₁} (K + K) = 4`` times the same
    factors, total 6.6: the terms ``compute_moment`` wrote before, which the
    R-cache guard refused (see ``tests/test_r_cache_mismatch.py``).
    """
    static_K = 0.5 * np.ones((N, N))
    K = ((lambda n_list, t_list: 0.5 * np.ones((N, N)))  # noqa: ARG005
         if dynamic else static_K)

    total, details = integrate_diagrams(
        _zero_dim_terms_summed_indices(), {"K": K},
        lambda_f=1.0, cache=_cache(R_GENERIC), method=method,
        n_samples=2**8, seed=3, fixed_indices=ZERO_DIM_FIXED,
        external_times=ZERO_DIM_EXTERNAL_TIMES,
    )
    assert [v for v, _ in details] == pytest.approx(
        SUMMED_EXPECTED, rel=1e-12, abs=1e-14)
    assert total == pytest.approx(1.65, rel=1e-12, abs=0.0)


def test_zero_dimensional_summed_index_terms_pin_the_legs():
    """The premise of the test above: without ``diag_R`` the kept R carries
    two different labels, each absorbed R its partner's label twice, and no
    summation index is left."""
    terms = _zero_dim_terms_summed_indices()
    assert len(terms) == 3
    for dt, partner in zip(terms, "abc"):
        ig = dt.build_integrand({"K": 0.5 * np.ones((N, N))}, ZERO_DIM_FIXED)
        assert ig.spatial.time_integration_vars == ()
        assert dt.summation_indices == ()
        absorbed = set(dt.r_absorbed_pairs)
        kept = [p for p in dt.propagators
                if p.kind == "R"
                and (p.spatial_left, p.spatial_right) not in absorbed]
        assert [(p.index_left, p.index_right) for p in kept] == [
            (partner, "d")]
        for p in dt.propagators:
            if (p.spatial_left, p.spatial_right) in absorbed:
                assert p.index_left == p.index_right != "d"


def test_zero_dimensional_scalar_r_unchanged():
    """The scalar-R zero-dimensional path still goes through the batched
    ``R_time_batch`` branch and is untouched by the matrix-R detour."""

    def R_scalar(t1, t2):  # noqa: ARG001
        return 1.0

    def kappa2(n1, t1, n2, t2):  # noqa: ARG001
        return KAPPA_MAT * np.exp(-abs(t1 - t2))

    scalar_cache = PropagatorCache(
        PropagatorModel(R_time=R_scalar, kappa2=kappa2, n_components=N,
                        iso_R=True, diag_C=False)
    )
    kwargs = dict(
        lambda_f=1.0, cache=scalar_cache, method="gauss_legendre",
        n_samples=2**8, seed=3, fixed_indices=ZERO_DIM_FIXED,
        external_times=ZERO_DIM_EXTERNAL_TIMES,
    )
    static_total, _ = integrate_diagrams(
        _zero_dim_terms(), {"K": 0.5 * np.ones((N, N))}, **kwargs,
    )
    dynamic_total, _ = integrate_diagrams(
        _zero_dim_terms(),
        {"K": lambda n_list, t_list: 0.5 * np.ones((N, N))},  # noqa: ARG005
        **kwargs,
    )
    assert dynamic_total == pytest.approx(static_total, rel=1e-12, abs=1e-14)
    # Not vacuous: R = 1 puts the scalar control on the same closed form as
    # the matrix case, so a silent zero on either side would be caught.
    assert static_total == pytest.approx(ZERO_DIM_EXPECTED,
                                         rel=1e-12, abs=1e-14)
