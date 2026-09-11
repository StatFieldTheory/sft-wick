"""``System`` — the top-level user-facing spec for a stochastic-field
problem.  Lowers high-level specification objects to raw
``PropagatorModel`` / ``Field`` / ``Vertex`` / ``Action`` / etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any, Callable, Iterable

import numpy as np

from sft_wick.action import Action
from sft_wick.evaluate import PropagatorModel
from sft_wick.fields import Field, FieldOperator, reset_uid_counter
from sft_wick.perturbation import check_distinct_external_labels
from sft_wick.vertices import Vertex

from .specs import (
    ConstantImpulse,
    DiagonalA,
    FieldSpec,
    GaussianNoise,
    LinearOp,
    LocalVertex,
    NonLocalVertex,
    SeparableRotation,
    SeparableTranslation,
)


#: Offsets from ``t_min`` at which R, κ² and σ² are probed for off-diagonal
#: component entries, and the positions used for κ² and σ².  Irregular
#: values, so that a kernel is unlikely to vanish off the diagonal at every
#: probe point by coincidence.
_PROBE_TIME_PAIRS = ((0.37, 0.0), (1.3, 0.55), (2.9, 1.7), (0.8, 0.8),
                     (4.1, 0.2))
_PROBE_POSITION_PAIRS = ((0.0, 0.0), (0.0, 0.43), (1.1, -0.7))
_OFFDIAG_RTOL = 1e-12

_DIAG_R_REFUSAL = (
    "R(t, t') of this system has off-diagonal component entries (a dense "
    "drift, e.g. ExplicitR(iso_R=False)), and diag_R=True keeps only R_aa, "
    "which would give a wrong result without an error.  Pass diag_R=False "
    "to System.expand, and diag_C=False as well: C = ∫ R κ² R is then dense "
    "too."
)

_DIAG_C_REFUSAL = (
    "C of this system has off-diagonal component entries (from a dense R, "
    "a component-mixing κ² such as GeneralKappa2, or a matrix σ²), and "
    "diag_C=True keeps only C_aa, which would give a wrong result without "
    "an error.  Pass diag_C=False to System.expand, and build the "
    "propagators with diag_C=False, c_closed_form_only=True and a closed "
    "form returning the full (N, N) C (c_closed_form='auto' supplies one "
    "for DiagonalA + SeparableTranslation(ExponentialTemporal) + "
    "ConstantImpulse); the quadrature tables hold C_aa only."
)


def _offdiagonal(mat: Any) -> bool:
    """Whether a square matrix has an off-diagonal entry larger than
    ``_OFFDIAG_RTOL`` times its largest entry."""
    m = np.asarray(mat)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] < 2:
        return False
    scale = float(np.abs(m).max())
    off = float(np.abs(m - np.diag(np.diag(m))).max())
    return scale > 0.0 and off > _OFFDIAG_RTOL * scale


def _probe_offdiagonal(fn: Callable, arg_sets: Iterable[tuple]) -> bool | None:
    """``True`` if ``fn`` returns a matrix with an off-diagonal entry for
    any argument tuple, ``False`` if it never does, ``None`` if it raises.

    A kernel written for vector positions may reject the scalar probe
    positions.  That says nothing about its component structure, so the
    guards below read ``None`` as "cannot tell" and do not refuse.
    """
    try:
        return any(_offdiagonal(fn(*args)) for args in arg_sets)
    except Exception:  # noqa: BLE001 -- a user callable; see the docstring
        return None


_OBS_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<name>[a-zA-Z]+?)                     # field name (e.g. 'phi'),
                                             #   non-greedy so 'phi_a'
                                             #   parses as name='phi'
                                             #   + comp='a'
    (?:_(?P<comp>[a-zA-Z][a-zA-Z0-9]*))?     # optional component index
    \(\s*(?P<spatial>[a-zA-Z0-9_]+)\s*\)     # (spatial_arg)
    \s*$
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class System:
    """High-level specification of a stochastic field-theory problem.

    Combines all the physics inputs the user needs to describe in a
    single immutable object.  Use :meth:`expand` and
    :meth:`propagators` to get to the computational layer.

    Args:
        field: :class:`FieldSpec` — the physical field (φ) spec; the
            response field (ψ) is derived automatically.
        linear: :class:`LinearOp` variant — defines R.  Required
            unless ``explicit_R`` is supplied.
        noise: :class:`GaussianNoise` — κ² (+ optional σ²) defines C
            together with R.
        vertices: list of :class:`LocalVertex` — F^(n) local
            interactions.  May be empty (linear theory).
        nonlocal_vertices: list of :class:`NonLocalVertex` — κ^(m) for
            m ≥ 3 non-Gaussian driving contributions.  May be empty.
        t_min: lower time bound for propagator integrations.
        explicit_R: escape hatch — if set, overrides ``linear``.  The
            structured alternative is the :class:`ExplicitR` LinearOp
            variant; both work, but ``ExplicitR`` is preferred for
            serialisation and YAML configs.
    """

    field: FieldSpec
    linear: LinearOp
    noise: GaussianNoise
    vertices: tuple[LocalVertex, ...] = field(default_factory=tuple)
    nonlocal_vertices: tuple[NonLocalVertex, ...] = field(default_factory=tuple)
    t_min: float = 0.0
    explicit_R: Callable | None = None

    def __post_init__(self):
        # Normalise list → tuple for hashability.
        if isinstance(self.vertices, list):
            object.__setattr__(self, "vertices", tuple(self.vertices))
        if isinstance(self.nonlocal_vertices, list):
            object.__setattr__(self, "nonlocal_vertices", tuple(self.nonlocal_vertices))

    # --------------------------------------------------------------- #
    # Derived properties
    # --------------------------------------------------------------- #

    @property
    def n_components(self) -> int:
        return self.field.n_components

    @property
    def homogeneity(self) -> str:
        """Inferred from the noise κ² spec (``'translation'`` / …)."""
        return self.noise.kappa2.homogeneity

    @property
    def iso_R(self) -> bool:
        return self._effective_linear.is_iso_R

    @cached_property
    def _effective_linear(self) -> LinearOp:
        """``linear``, with a callable-rate :class:`DiagonalA` grid extended
        down to ``t_min`` (at its own node spacing) when it starts above
        it, so that no R below the grid comes from spline extrapolation."""
        lin = self.linear
        if (isinstance(lin, DiagonalA) and callable(lin.gamma)
                and self.t_min < lin.t_min_cache):
            spacing = ((lin.t_max_cache - lin.t_min_cache)
                       / max(lin.n_grid_cache - 1, 1))
            n_grid = int(np.ceil((lin.t_max_cache - self.t_min) / spacing)) + 1
            return replace(lin, t_min_cache=float(self.t_min),
                           n_grid_cache=max(n_grid, lin.n_grid_cache))
        return lin

    def _r_offdiagonal(self) -> bool | None:
        """Whether R(t, t') has off-diagonal component entries.

        ``False`` for a scalar R and for :class:`DiagonalA`; otherwise the
        R callable is probed at causal time pairs (``None`` if it cannot be
        evaluated there).
        """
        if self.iso_R:
            return False
        if self.explicit_R is None and isinstance(self.linear, DiagonalA):
            return False
        R = (self.explicit_R if self.explicit_R is not None
             else self.linear.build_R_callable())
        t0 = float(self.t_min)
        return _probe_offdiagonal(
            R, [(t0 + a, t0 + b) for a, b in _PROBE_TIME_PAIRS if a > b])

    def _kappa2_offdiagonal(self) -> bool | None:
        kappa2 = self.noise.kappa2
        if isinstance(kappa2, (SeparableTranslation, SeparableRotation)):
            return False                     # κ_t κ_x · I_N by construction
        fn = kappa2.build_callable(self.n_components)
        t0 = float(self.t_min)
        return _probe_offdiagonal(fn, [
            (x1, t0 + a, x2, t0 + b)
            for x1, x2 in _PROBE_POSITION_PAIRS
            for a, b in _PROBE_TIME_PAIRS
        ])

    def _sigma2_offdiagonal(self) -> bool | None:
        sigma2 = self.noise.sigma2
        if sigma2 is None:
            return False
        if isinstance(sigma2, ConstantImpulse):
            return _offdiagonal(np.asarray(sigma2.amplitude, dtype=float))
        fn = sigma2.build_callable(self.n_components)
        t0 = float(self.t_min)
        return _probe_offdiagonal(fn, [
            (x1, t0 + a, x2)
            for x1, x2 in _PROBE_POSITION_PAIRS
            for a, _ in _PROBE_TIME_PAIRS
        ])

    def _c_offdiagonal(self) -> bool | None:
        """Whether ``C = ∫R κ² R + ∫R σ² R`` has off-diagonal entries:
        ``True`` if R, κ² or σ² has one, ``None`` if a probe could not tell
        and none of the others found one."""
        parts = (self._r_offdiagonal(), self._kappa2_offdiagonal(),
                 self._sigma2_offdiagonal())
        if any(p is True for p in parts):
            return True
        return None if None in parts else False

    # --------------------------------------------------------------- #
    # Lowering to raw API objects
    # --------------------------------------------------------------- #

    def build_propagator_model(self, diag_C: bool = True) -> PropagatorModel:
        """Lower the spec to a :class:`PropagatorModel`.

        Args:
            diag_C: when ``True`` (default), the numerical C propagator is
                represented as a diagonal vector ``(n, N)``. When ``False``,
                the full ``(n, N, N)`` matrix is preserved -- required for
                observables that probe cross-component C entries
                (e.g. lensing kappa-gamma cross-correlation). Off-diagonal
                support is only meaningful in combination with
                ``Propagators.build(c_closed_form_only=True)`` because the
                spline-table paths build only diagonal entries.
        """
        R_time = (
            self.explicit_R
            if self.explicit_R is not None
            else self._effective_linear.build_R_callable()
        )
        kappa2 = self.noise.kappa2.build_callable(self.n_components)
        sigma2 = (
            None
            if self.noise.sigma2 is None
            else self.noise.sigma2.build_callable(self.n_components)
        )
        return PropagatorModel(
            R_time=R_time,
            kappa2=kappa2,
            n_components=self.n_components,
            iso_R=self.iso_R,
            diag_C=diag_C,
            t_min=self.t_min,
            sigma2=sigma2,
        )

    def build_fields(self) -> tuple[Field, Field]:
        """Return ``(phi, psi)`` :class:`Field` objects."""
        phi = Field(self.field.name, "physical",
                    n_components=self.n_components)
        psi = Field(self.field.response_name, "response",
                    n_components=self.n_components)
        return phi, psi

    def build_action(self) -> Action:
        """Lower vertices to a raw :class:`Action`."""
        phi, psi = self.build_fields()
        raw_vertices: list[Vertex] = []

        for lv in self.vertices:
            rank = np.asarray(lv.coupling).ndim
            # Convention: first axis = ψ, rest = φ
            fields = [psi] + [phi] * (rank - 1)
            raw_vertices.append(
                Vertex(fields=fields, coupling=lv.name, local=True)
            )

        for nv in self.nonlocal_vertices:
            raw_vertices.append(
                Vertex(
                    fields=[psi] * nv.order,
                    coupling=nv.name,
                    local=False,
                    equal_time=nv.equal_time,
                    already_R_contracted=nv.already_R_contracted,
                )
            )

        return Action(vertices=raw_vertices)

    def build_coupling_values(self) -> dict[str, Any]:
        """Dict passed to ``DiagramTerm.evaluate_coupling`` /
        ``build_integrand``.

        The MSR prefactors are applied here:

        - **Local** ``F^(n)``: multiplied by ``-i`` (``F_MSR = -i F``).
        - **Non-local** ``κ^(m)``: multiplied by ``-(i^m) / m!``
          (``K_MSR = -(i^m)/m! κ^(m)``).

        Users pass the **bare** physical tensors when constructing
        :class:`LocalVertex` / :class:`NonLocalVertex`; this method
        is the single point of truth for the MSR convention.  For a
        callable non-local coupling the returned dict value is a
        wrapped callable that applies the factor at evaluation time.
        """
        cv: dict[str, Any] = {}
        for lv in self.vertices:
            cv[lv.name] = lv.msr_coupling
        for nv in self.nonlocal_vertices:
            cv[nv.name] = nv.msr_coupling
        return cv

    # --------------------------------------------------------------- #
    # Public operations
    # --------------------------------------------------------------- #

    def expand(
        self,
        observable: Iterable,
        orders: Iterable[int],
        *,
        response_phase: bool = True,
        ito: bool = True,
        collect_topology: bool = True,
        iso_R: bool | None = None,
        diag_R: bool = True,
        diag_C: bool = True,
        iso_C: bool = False,
        cache_path: Any = None,
        progress: Any = None,
    ) -> "Expansion":
        """Run the perturbative expansion and return an
        :class:`Expansion` object.

        Args:
            observable: iterable of either ``FieldOperator`` objects
                or strings like ``"phi_a(x)"``.  Strings are parsed
                as ``field_compIndex(spatialArg)``.
            orders: iterable of perturbative orders to compute.
            response_phase, ito, collect_topology, iso_R, diag_R, diag_C, iso_C: forwarded to :func:`~sft_wick.perturbation.compute_moment`.  When ``iso_R`` is ``None`` the value is inferred from ``self.linear``.
            cache_path: directory (or file) for on-disk caching.
                ``None`` disables caching (prints a one-shot
                reminder).
            progress: progress-bar setting for the per-order expansion
                (``True`` / ``False`` / callable / ``None`` = inherit);
                see :mod:`sft_wick.progress`.
        """
        from sft_wick.perturbation import compute_moment
        from sft_wick.progress import progress as _progress_scope

        from .cache import load_or_compute
        from .expansion import Expansion

        orders_list = sorted(set(int(o) for o in orders))
        iso_R_val = self.iso_R if iso_R is None else iso_R

        # diag_R / diag_C collapse R and C onto their diagonals; a dense R
        # or a component-mixing noise has entries off it.
        if diag_R and not iso_R_val and self._r_offdiagonal():
            raise ValueError(_DIAG_R_REFUSAL)
        if diag_C and self._c_offdiagonal():
            raise ValueError(_DIAG_C_REFUSAL)

        obs_ops, obs_repr = _parse_observable(
            observable, self, max(orders_list),
        )

        spec_key = {
            "system_hash": _system_spec_key(self),
            "observable": obs_repr,
            "orders": tuple(orders_list),
            "flags": (
                response_phase, ito, collect_topology,
                iso_R_val, diag_R, diag_C, iso_C,
            ),
        }

        max_order = max(orders_list)

        def _compute():
            reset_uid_counter()
            action = self.build_action()
            result = compute_moment(
                obs_ops,
                action,
                order=max_order,
                response_phase=response_phase,
                ito=ito,
                collect_topology=collect_topology,
                diag_R=diag_R,
                diag_C=diag_C,
                iso_R=iso_R_val,
                iso_C=iso_C,
            )
            return {
                "dts_by_order": {
                    o: result.diagram_terms(o) for o in orders_list
                },
                "raw_result": result,
            }

        with _progress_scope(progress):
            payload = load_or_compute(
                cache_path, spec_key, _compute,
                operation_name="expansion",
            )

        return Expansion(
            system=self,
            dts_by_order=payload["dts_by_order"],
            orders=tuple(orders_list),
            observable_repr=obs_repr,
            raw_result=payload["raw_result"],
        )

    def propagators(
        self,
        t_max: float,
        n_grid_t: int = 60,
        *,
        homogeneity: str | None = None,
        r_max: float | None = None,
        n_grid_r: int | None = None,
        n_grid_cos: int | None = None,
        x_max: float | None = None,
        n_grid_x: int | None = None,
        n_jobs: int = 1,
        c_closed_form: Callable | str | None = "auto",
        cache_path: Any = None,
        interp_method: str = "linear",
        c_closed_form_only: bool = False,
        c_closed_form_vectorized: bool = False,
        c_method: str = "auto",
        c_n_gauss: int = 20,
        diag_C: bool = True,
        progress: Any = None,
    ) -> "Propagators":
        """Build a :class:`Propagators` object with a precomputed
        spatial table matching ``self.homogeneity``.

        By default (all grid args ``None``) the cache enters **lazy
        mode** for the inferred homogeneity — recommended for moment
        calculations at a small fixed set of external positions.

        How C is obtained is decided in two steps, both ``'auto'`` by
        default: ``c_closed_form`` (built-in closed form when the kernel
        family has one -- diagonal constant drift with a separable
        exponential-temporal noise, the demo1/demo2 family -- so no
        quadrature at all), then ``c_method`` (Gauss-Legendre with a
        converged node count for the package's own kernels, ``dblquad``
        for user callables).  The result is reported in
        :attr:`Propagators.c_source`.

        Args:
            t_max: upper time bound.
            n_grid_t: number of time grid points.
            homogeneity: override the inferred homogeneity
                (``'translation' | 'rotation' | 'general'``).
            r_max, n_grid_r: translation full-grid parameters
                (both given ⇒ full 3-D spline; either ``None`` ⇒
                lazy).
            n_grid_cos: rotation full-grid parameter (given ⇒ full
                3-D spline over ``cos θ``).
            x_max, n_grid_x: general full-grid parameters
                (both given ⇒ full 4-D spline).
            n_jobs: parallel workers for grid build (``-1`` = all
                cores, via joblib).
            c_closed_form: ``'auto'`` (default: the built-in closed form
                when one applies), ``None`` (force quadrature), or a
                callable ``(n1, t1, n2, t2) -> (N, N)`` returning the
                full C matrix directly (a user closed form; wins over
                the built-in one).  See :meth:`Propagators.build`.
            cache_path: directory (or file) for on-disk caching of
                the constructed :class:`PropagatorCache`.
            interp_method: ``RegularGridInterpolator`` method for the
                full-grid C tables. ``'linear'`` (default) is monotone
                and safe for steep tails; ``'cubic'`` gives O(h⁴) on
                smooth grids. Forwarded to :class:`PropagatorCache`.
            c_closed_form_only: when True (with ``c_closed_form`` set),
                skip every spline and route C lookups directly through
                the user's c_fn -- machine-precision agreement with
                the analytical form. Forwarded to
                :meth:`Propagators.build`.
            c_closed_form_vectorized: c_fn accepts batched arrays and
                returns ``(n, N, N)`` (only meaningful with
                ``c_closed_form_only=True``).
            c_method: Quadrature method for the inner C-propagator
                ``∫ R κ² R`` integral when no closed form applies.

                - ``'auto'`` (default) -- Gauss-Legendre with a node
                  count refined from ``c_n_gauss`` until the rule is
                  converged at the table's extreme cells (falling back
                  to ``dblquad`` if it never is) for the package's own
                  kernel families; ``dblquad`` for any user callable.
                - ``'gauss_legendre'`` -- tensor-product GL with a
                  diagonal-aware sub-region split and exactly
                  ``c_n_gauss`` nodes.  18-100× faster than dblquad on
                  piecewise-analytic κ²; accuracy at fixed node count
                  degrades with ``(γ + 1/σ_t) t_max``.  Not the right
                  tool for non-smooth κ² (discontinuous, oscillatory
                  at high frequency, integrable singularities like
                  ``1/√t``).
                - ``'dblquad'`` -- adaptive Gauss-Kronrod; robust on
                  any κ² but slow (10-250 ms / cell).

                Ignored when a closed form is in use.  See
                :doc:`/user_guide/workflow` "Choosing an integrator"
                for the full decision matrix.
            c_n_gauss: Per-dim GL node count for
                ``c_method='gauss_legendre'`` and the starting count
                for ``'auto'`` (default 20). Cost ``c_n_gauss²`` per
                sub-region.
            diag_C: when ``True`` (default), the numerical C propagator is
                represented as a diagonal vector ``(n, N)`` -- the
                bit-identical behaviour for every pre-existing caller.
                When ``False``, the full ``(n, N, N)`` matrix is preserved
                so observables can read off-diagonal entries
                ``C[a, b]`` with ``a != b`` (e.g. the lensing
                κ-γ₊ cross-correlation). Only meaningful with
                ``c_closed_form_only=True``: spline-table paths build
                diagonal entries only.
            progress: ``True`` / ``False`` / a ``(desc, done, total)``
                callable / ``None`` (inherit: the ``SFT_WICK_PROGRESS``
                environment variable, else on only when stderr is a
                terminal).  Controls the table-build progress bar; see
                :mod:`sft_wick.progress`.  Never affects results.
        """
        from sft_wick.progress import progress as _progress_scope

        from .propagators import Propagators

        lin = self._effective_linear
        if (self.explicit_R is None and isinstance(lin, DiagonalA)
                and callable(lin.gamma) and t_max > lin.t_max_cache):
            raise ValueError(
                f"t_max={t_max} exceeds DiagonalA.t_max_cache="
                f"{lin.t_max_cache}: the cumulative-rate spline behind R "
                f"would be extrapolated.  Raise t_max_cache (and "
                f"n_grid_cache with it)."
            )
        if diag_C and self._c_offdiagonal():
            raise ValueError(_DIAG_C_REFUSAL)

        with _progress_scope(progress):
            return Propagators.build(
                self,
                t_max=t_max,
                n_grid_t=n_grid_t,
                homogeneity=homogeneity,
                r_max=r_max,
                n_grid_r=n_grid_r,
                n_grid_cos=n_grid_cos,
                x_max=x_max,
                n_grid_x=n_grid_x,
                n_jobs=n_jobs,
                c_closed_form=c_closed_form,
                cache_path=cache_path,
                interp_method=interp_method,
                c_closed_form_only=c_closed_form_only,
                c_closed_form_vectorized=c_closed_form_vectorized,
                c_method=c_method,
                c_n_gauss=c_n_gauss,
                diag_C=diag_C,
            )


# =========================================================================
# Helpers
# =========================================================================


def _parse_observable(observable, system: System, _max_order: int = 1):
    """Normalise ``observable`` into a tuple of FieldOperators.

    Accepts:

    - ``("phi_a(x)", "phi_b(y)")`` — string form; each string is
      ``name_component(spatial)``.
    - an iterable of ``FieldOperator`` objects (advanced).

    Returns ``(obs_ops, obs_repr)`` where ``obs_repr`` is a hashable
    tuple describing the observable for caching.
    """
    ops = list(observable)
    if all(isinstance(o, str) for o in ops):
        phi, _psi = system.build_fields()
        out = []
        repr_list = []
        for spec in ops:
            m = _OBS_PATTERN.match(spec)
            if m is None:
                raise ValueError(
                    f"Cannot parse observable '{spec}'.  Expected form "
                    f"'name_component(spatial)' like 'phi_a(x)'."
                )
            name = m.group("name")
            comp = m.group("comp")
            spatial = m.group("spatial")
            # The RESPONSE field is nameable too.  Without it the declarative
            # API cannot express R(t, t') at all -- an observable needs a psi
            # leg, and only the physical field used to be accepted.
            if name == system.field.name:
                field = phi
            elif name == system.field.response_name:
                field = _psi
            else:
                raise ValueError(
                    f"Observable field '{name}' matches neither the system's "
                    f"physical field '{system.field.name}' nor its response "
                    f"field '{system.field.response_name}'."
                )
            if comp is None:
                if system.n_components != 1:
                    raise ValueError(
                        f"Observable '{spec}' omits a component index "
                        f"but the system has n_components="
                        f"{system.n_components}.  Use e.g. "
                        f"'{name}_a({spatial})'."
                    )
                out.append(field(spatial))
                repr_list.append((name, None, spatial))
            else:
                out.append(field(comp, spatial))
                repr_list.append((name, comp, spatial))
        check_distinct_external_labels(out, _max_order)
        return tuple(out), tuple(repr_list)

    if all(isinstance(o, FieldOperator) for o in ops):
        check_distinct_external_labels(ops, _max_order)
        repr_list = tuple(
            (o.field.name,
             tuple(o.component_indices) if o.component_indices else None,
             o.spatial_arg)
            for o in ops
        )
        return tuple(ops), repr_list

    raise TypeError(
        "observable must be either an iterable of 'name_comp(spatial)' "
        "strings or an iterable of FieldOperator objects; got mixed/"
        "unknown types."
    )


def _system_spec_key(system: System) -> Any:
    """Canonicalised representation of a ``System`` for caching.

    Callables inside kernel helpers are captured by their ``repr()``
    when the class is a frozen dataclass (joblib's content-hash
    handles them), but we include a light summary string in case the
    user passes raw ``lambda``s through escape hatches.
    """
    return {
        "field": (system.field.name, system.field.n_components),
        "linear": repr(system.linear),
        "noise_kappa2_type": type(system.noise.kappa2).__name__,
        "noise_kappa2_repr": repr(system.noise.kappa2),
        "noise_sigma2_repr": repr(system.noise.sigma2),
        "vertices": tuple(
            (v.name, np.asarray(v.coupling).shape,
             float(np.sum(np.abs(np.asarray(v.coupling)))))
            for v in system.vertices
        ),
        "nonlocal_vertices": tuple(
            (v.name, v.order, v.equal_time, v.already_R_contracted)
            for v in system.nonlocal_vertices
        ),
        "t_min": system.t_min,
    }
