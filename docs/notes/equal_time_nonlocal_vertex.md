# Design note — `NonLocalVertex(equal_time=True)`

**Status**: landed 2026-05-14 (unreleased on `main`).

**Author of the patch**: applied during the STF_lensing path-integral
lensing project diagnosis (Zheng Zhang + AI assistant pair-coding); the
underlying physics observation — "canoes returns the equal-shell ζ,
sft-wick expects full cross-spacetime" — is Zheng's.

**Files modified**: `workflow/specs.py`, `workflow/system.py`,
`workflow/config.py`, `vertices.py`, `perturbation.py`, `evaluate.py`;
regression suite at `tests/test_equal_time_nonlocal.py`.

---

## Why the option exists

`NonLocalVertex(coupling=fn, order=m)` was originally specified as the
**full cross-spacetime** cumulant of the driving field, with
`fn(n_list, t_list)` receiving `m` independent `(position, time)`
pairs and the runtime integrating each independently per the MSR
action

```
W^(m)[ψ] = (1/m!) ∫ dz_1 ... dz_m  ψ(z_1) ψ(z_2) ... ψ(z_m) κ^(m)(z_1, ..., z_m).
```

Most cosmological pipelines that feed into sft-wick (CCL, canoes,
analogous "equal-time bispectrum" code) instead deliver an
**equal-shell** form
`ζ_eq(γ_12, γ_23, γ_31; λ)`. Mapping this onto the MSR action requires
the time-δ ansatz

```
κ^(m)(z_1, ..., z_m)  ≈  ∏_{i=1}^{m-1} δ(λ_i - λ_{i+1}) · ζ_eq(n_1, ..., n_m; λ_1).
```

Naively passing `ζ_eq` as `coupling` ignores the δ-structure: sft-wick
still integrates `m` independent times, contributing a spurious
`(t_final - t_min)^(m-1)` factor. For `m=3` and `t_final = 2360` Mpc
(STF_lensing fiducial), that's `~5.6×10^6` — six orders of magnitude
wrong, observed exactly in the Order-2 FK ⟨κκ⟩ diagnostic before this
patch landed.

`equal_time=True` declares the δ-structure: the runtime collapses the
`m` time integrations to one and keeps the `m` spatial integrations
independent. The user callable still receives length-`m` `n_list` /
`t_list`, but now all `m` entries of `t_list` are the same shared
sample of the single integration variable.

## Patch surface (alphabetical by file)

### `src/sft_wick/evaluate.py`

* `SpatialStructure` — added optional field
  `equal_time_aliases: tuple[tuple[str, str], ...] = ()` mapping each
  non-representative internal spatial label to its canonical time
  representative. Empty tuple ⇒ no aliasing (original cross-spacetime
  contract).
* `analyze_spatial(dt)` — when the input `DiagramTerm` carries
  `equal_time_aliases`, drop the non-representative labels from the
  surviving `time_integration_set` so the Jacobian gets one `width`
  factor per equal-time vertex instead of `m`. Time-ordering tuples
  that touch a filtered label are rewritten onto their canonical
  representative so causality between the equal-time vertex and the
  rest of the diagram is preserved on the surviving legs.
* `integrate_moment_qmc_vectorized`, `integrate_moment_gauss_legendre`
  — extended their inner `_times(var)` lookup to first consult the
  `equal_time_aliases` map, so R-propagator lookups, C-propagator
  lookups, and the dynamic-coupling `label_t` dict all see the
  collapsed time transparently. No change to how `times_arr` /
  `jacobians` are generated; the alias is purely a *lookup-time*
  redirection.
* `DiagramIntegrand.evaluate(times, ...)` — for the static-coupling
  per-sample path, expands the per-sample `times` dict to fill aliased
  keys from their representatives before passing through to
  `cache.R_product` / `times[sp]` lookups in C-propagator evaluation.

### `src/sft_wick/perturbation.py`

* `DiagramTerm` — added optional frozen-dataclass field
  `equal_time_aliases: tuple[tuple[str, str], ...] = ()`. Propagated
  through `apply_diagonal` (which constructs a new DiagramTerm) and
  through the two principal DiagramTerm-construction sites
  (collect-topology path at line ~1170 in `compute_moment` and the
  build-integrand path at line ~2235 in `compute_diagram_integrands`).
  Each site collects the alias map by iterating
  `vertex_instances[i].equal_time_aliases` and flattening to a sorted
  tuple of pairs.

### `src/sft_wick/vertices.py`

* `Vertex` — added field `equal_time: bool = False` (non-local
  vertices only; ignored when `local=True`). Constructor accepts it
  via keyword and stores it via `object.__setattr__` to play with the
  frozen-dataclass-style `__init__`.
* `VertexInstance` — added field
  `equal_time_aliases: dict = None` populated by `instantiate(...)`
  when `vertex.equal_time and not vertex.local and len(spatial_vars) > 1`.
  The first spatial variable is chosen as the canonical representative;
  the remaining `m-1` get mapped onto it.

### `src/sft_wick/workflow/specs.py`

* `NonLocalVertex` — added field `equal_time: bool = False` with a
  docstring explaining the equal-shell / single-shell motivation and
  citing the cosmology bispectrum API as the canonical use case.

### `src/sft_wick/workflow/system.py`

* `System.build_action()` — passes `nv.equal_time` through to the
  underlying `Vertex` when lowering each `NonLocalVertex` to its raw
  Action term. One-line change.

### `src/sft_wick/workflow/config.py`

* `_resolve_system_from_config()` — reads `equal_time` from the YAML
  `nonlocal_vertices` block (defaulting to `False`). One-line change
  in the `NonLocalVertex(...)` constructor call.

## What is NOT changed

* Diagram topology / Wick contractions / propagator routing — sft-wick
  still emits the same `r_propagators`, `c_propagators`,
  `direction_map`, and `coupling_sum`. Only the time-integration
  multiplicity for one vertex is reduced.
* MSR phase / response factor — unchanged. The
  `(-i^m)/m!` prefactor on `NonLocalVertex.msr_coupling` is applied
  identically.
* The coupling callable's signature — it still receives length-`m`
  `n_list` and `t_list`. The only observable difference is that all
  `m` entries of `t_list` are equal when `equal_time=True`.
* Caching key — `equal_time` is part of the
  `NonLocalVertex` dataclass and therefore enters the existing
  expansion / propagator cache hashes through the dataclass field
  ordering. **Important**: invalidate `expand` and `propagator`
  caches when flipping the flag (the rerun scripts in the
  STF_lensing project do this with `rm -rf` on the cache dirs).
* All existing tests — 403 pre-existing tests pass unchanged; the new
  `test_equal_time_nonlocal.py` adds 16 tests covering the new path,
  and `test_workflow_config.py::test_CF18_yaml_nonlocal_vertex_equal_time_round_trip`
  locks the L2 YAML contract.

## Limitations / open follow-ups

1. **`nquad` path explicitly refuses dynamic coupling** (existing
   behaviour, unrelated to this patch). The `equal_time` path is
   exercised through the GL and QMC vectorised paths.
2. **No partial-equal-time** — there is no way to declare "only the
   first 2 of m legs share a time, the third is independent". If a
   user needs that, they would declare a different vertex with
   `m=2` for the shared pair and add the third leg via a separate
   mechanism.
3. **Symbolic LaTeX rendering** of an equal-time vertex still emits
   `m` `∫` operators (since the symbolic side wraps each spatial
   variable in its own `IntegralOver` node at action-build time).
   This is purely cosmetic — the numerical integration uses the
   collapsed measure correctly. A future refactor could collapse the
   `IntegralOver` chain at LaTeX time by inspecting the alias map.
4. **Vectorised dynamic-coupling path** (`coupling_vectorized=True`)
   is exercised through `_times(lab)` aliasing in the same way as
   the per-sample path. Regression covered indirectly through the
   STF_lensing FK rerun; a unit test pinning the
   `coupling_vectorized=True, equal_time=True` combination would be
   a nice future addition.

## Validation reference

In the STF_lensing path-integral lensing project (which prompted this
patch), a single-γ FK Order-2 ⟨κκ⟩ at γ=1' gave:

* Pre-fix (`equal_time` absent): **9.0** (unphysical)
* Post-fix (`equal_time: true`): **1.77×10^-5** (matches the
  Order-2 FF magnitude at the same γ).

The ratio of 5×10^5 is exactly the Jensen's-inequality reduction
`[∫dt R(t)]^3 / ∫dt R(t)^3` for the actual Sachs Jacobi
response kernel R(λ_f, λ) = D(λ_f, λ) on `[0, λ_f]` with
`λ_f = 2360` Mpc.

For a uniform `R(t) = 1`, the predicted ratio is `(λ_f)^2 = 5.6×10^6`
exactly; the smaller observed factor reflects the (1 - λ/λ_f)
linear-ramp shape of R.

## Tests

`tests/test_equal_time_nonlocal.py` — 16 tests grouped into three layers:

**Layer 1 — spec / vertex / instance plumbing (7):**

1. spec accepts the flag with the expected default
2. `Vertex` carries `equal_time`
3. `VertexInstance.equal_time_aliases` is populated for non-local +
   equal_time vertices
4. ... empty for non-local + not-equal_time
5. ... empty for local vertices (default flag off)
6. `Vertex(local=True, equal_time=True)` raises `ValueError` — guard
   against the silent footgun where a flag on a local vertex would
   leave the spurious `t_max^(m-1)` factor in place undetected
7. `equal_time=True` with order 1 (m=1) yields an empty alias map
   (trivial collapse); order 2 (m=2) is the smallest non-trivial case

**Layer 2 — SpatialStructure and integration paths (6):**

8. `analyze_spatial` filters aliased legs from `time_integration_vars`
9. `analyze_spatial` is a no-op when no aliases are present
10. Static-coupling `DiagramIntegrand.evaluate` fills aliased labels
    without raising `KeyError` on a per-sample `times` dict
11. Dynamic-coupling callable receives `m` identical times when
    `equal_time=True`
12. Jacobian ratio: equal_time integrates to `span` and full integrates
    to `span^m` — exposing the `(t_max)^(m-1)` correction (GL path)
13. Same Jacobian ratio via `integrate_moment_qmc_vectorized` (the
    production-default integrator)

**Layer 3 — `_times` alias redirect inside QMC kernel (1) + diagram-term
end-to-end (1) + design fixtures (1):**

14. The QMC-vectorized `_times` resolver redirects aliased labels —
    a tracking cache asserts the three R-propagator left-time arrays
    are bit-identical after redirect
15. End-to-end: `System(equal_time NonLocalVertex).expand(...)`
    produces `DiagramTerm`s carrying the alias map at the diagram level
    (closes the loop from spec all the way to integration-ready
    DiagramTerm)
16. Reset-uid `autouse` fixture pins label ordering so the alias
    assertions are deterministic under arbitrary test ordering

Plus `tests/test_workflow_config.py::test_CF18_yaml_nonlocal_vertex_equal_time_round_trip`
which locks the L2 YAML contract.

Run with::

    pytest tests/test_equal_time_nonlocal.py tests/test_workflow_config.py::test_CF18_yaml_nonlocal_vertex_equal_time_round_trip -v
