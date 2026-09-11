"""Matrix-valued R: per-propagator component indices.

``DiagramIntegrand._evaluate_r_product_general`` looked each R factor's
indices up by its ``(spatial_left, spatial_right)`` endpoints.  Endpoints do
not identify an R propagator --- a local vertex with two ψ legs puts *two* of
them between the same two points --- so both factors of such a pair read the
first match's indices and the diagram silently evaluated
``R[j,l] * R[j,l]`` instead of ``R[j,l] * R[k,m]``.

The scalar-R path never resolves component indices at all and is unaffected.
"""

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.evaluate import PropagatorCache, PropagatorModel

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
