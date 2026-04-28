"""``System`` — the top-level user-facing spec for a stochastic-field
problem.  Lowers high-level specification objects to raw
``PropagatorModel`` / ``Field`` / ``Vertex`` / ``Action`` / etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from sft_wick.action import Action
from sft_wick.evaluate import PropagatorModel
from sft_wick.fields import Field, FieldOperator, reset_uid_counter
from sft_wick.vertices import Vertex

from .specs import (
    FieldSpec,
    GaussianNoise,
    LinearOp,
    LocalVertex,
    NonLocalVertex,
)


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
        return self.linear.is_iso_R

    # --------------------------------------------------------------- #
    # Lowering to raw API objects
    # --------------------------------------------------------------- #

    def build_propagator_model(self) -> PropagatorModel:
        """Lower the spec to a :class:`PropagatorModel`."""
        R_time = (
            self.explicit_R
            if self.explicit_R is not None
            else self.linear.build_R_callable()
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
            diag_C=True,  # wrapper currently assumes diagonal C
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
    ) -> "Expansion":
        """Run the perturbative expansion and return an
        :class:`Expansion` object.

        Args:
            observable: iterable of either ``FieldOperator`` objects
                or strings like ``"phi_a(x)"``.  Strings are parsed
                as ``field_compIndex(spatialArg)``.
            orders: iterable of perturbative orders to compute.
            response_phase, ito, collect_topology, iso_R, diag_R,
            diag_C, iso_C: forwarded to
                :func:`~sft_wick.perturbation.compute_moment`.  When
                ``iso_R`` is ``None`` the value is inferred from
                ``self.linear``.
            cache_path: directory (or file) for on-disk caching.
                ``None`` disables caching (prints a one-shot
                reminder).
        """
        from sft_wick.perturbation import compute_moment

        from .cache import load_or_compute
        from .expansion import Expansion

        orders_list = sorted(set(int(o) for o in orders))
        iso_R_val = self.iso_R if iso_R is None else iso_R

        obs_ops, obs_repr = _parse_observable(observable, self)

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
        c_closed_form: Callable | None = None,
        cache_path: Any = None,
        interp_method: str = "linear",
    ) -> "Propagators":
        """Build a :class:`Propagators` object with a precomputed
        spatial table matching ``self.homogeneity``.

        By default (all grid args ``None``) the cache enters **lazy
        mode** for the inferred homogeneity — recommended for moment
        calculations at a small fixed set of external positions.

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
            c_closed_form: optional callable
                ``(n1, t1, n2, t2) -> (N, N)`` returning the full C
                matrix directly.  When provided, the spline-table
                builder bypasses ``scipy.integrate.dblquad`` — use
                this when the user knows C in closed form (e.g. OU
                kernel).
            cache_path: directory (or file) for on-disk caching of
                the constructed :class:`PropagatorCache`.
            interp_method: ``RegularGridInterpolator`` method for the
                full-grid C tables. ``'linear'`` (default) is monotone
                and safe for steep tails; ``'cubic'`` gives O(h⁴) on
                smooth grids. Forwarded to :class:`PropagatorCache`.
        """
        from .propagators import Propagators

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
        )


# =========================================================================
# Helpers
# =========================================================================


def _parse_observable(observable, system: System):
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
            if name != system.field.name:
                raise ValueError(
                    f"Observable field '{name}' does not match the "
                    f"system's field '{system.field.name}'."
                )
            if comp is None:
                if system.n_components != 1:
                    raise ValueError(
                        f"Observable '{spec}' omits a component index "
                        f"but the system has n_components="
                        f"{system.n_components}.  Use e.g. "
                        f"'{name}_a({spatial})'."
                    )
                out.append(phi(spatial))
                repr_list.append((name, None, spatial))
            else:
                out.append(phi(comp, spatial))
                repr_list.append((name, comp, spatial))
        return tuple(out), tuple(repr_list)

    if all(isinstance(o, FieldOperator) for o in ops):
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
            (v.name, v.order) for v in system.nonlocal_vertices
        ),
        "t_min": system.t_min,
    }
