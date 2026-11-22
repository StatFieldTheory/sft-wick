"""``Result`` and ``SweepResult``: structured output from
:meth:`Expansion.evaluate` / :meth:`Expansion.sweep`.

Accessors cover three common analyses:

1. **Total**: summed moment value.
2. **By order**: per-perturbative-order contribution.  Shows
   convergence.
3. **By vertex type**: demo2-style FF/FK/KK decomposition.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from math import fsum, isfinite, nan, nextafter
from typing import Any, Literal, NamedTuple

import numpy as np

from sft_wick._gl_error import _GLRefinement, _roundoff_budget


def _replica_standard_error(values) -> float:
    samples = np.asarray(values, dtype=float)
    # Scale by a power of two before subtraction and squaring. Shifting the
    # scaled values also gives exactly zero spread for identical estimates.
    _, exponent = np.frexp(np.max(np.abs(samples)))
    scaled = np.ldexp(samples, -exponent)
    error = np.std(scaled - scaled[0], ddof=1) / np.sqrt(samples.size)
    return float(np.ldexp(error, exponent))


def _replica_sum_error(rows, count: int) -> float:
    """Keep the correlations across diagrams that share a scramble."""
    samples = np.asarray(tuple(rows), dtype=float).reshape(-1, count)
    # Constants add no spread. Center each varying diagram so that a large
    # mean on one diagram cannot mask the fluctuations on another.
    varying = samples[np.any(samples != samples[:, :1], axis=1)]
    if not len(varying):
        return 0.0
    _, exponent = np.frexp(np.max(np.abs(varying)))
    scaled = np.ldexp(varying, -exponent)
    # Include the reference column in the same compensated sum. Forming
    # each diagram's difference first can round away a term that survives
    # cancellation between diagrams.
    reference = -scaled[:, 0]
    deviations = [fsum(chain(column, reference)) for column in scaled.T]
    return float(np.ldexp(_replica_standard_error(deviations), exponent))


def _replica_group_error(rows, count: int) -> tuple[float, str]:
    """Apply the most restrictive error status of the contributing rows."""
    rows = tuple(rows)
    kinds = {row["error_kind"] for row, _ in rows}
    if "heavy_tail" in kinds:
        return nan, "heavy_tail"
    if kinds <= {"exact_zero"}:
        return 0.0, "exact_zero"
    return _replica_sum_error((values for _, values in rows), count), "rqmc"


def _gl_difference(coarse: float, fine: float) -> float:
    delta = abs(coarse - fine)
    if not isfinite(delta):
        raise ValueError("Gauss-Legendre convergence difference is not finite.")
    return delta


def _refinement_group_error(rows) -> tuple[float, str]:
    """The difference of two signed sums in the evaluation's row order."""
    coarse, fine = 0.0, 0.0
    kind = "exact_zero"
    for row, refined in rows:
        coarse += row["value"]
        fine += refined
        if row["error_kind"] == "gl_delta":
            kind = "gl_delta"
    return _gl_difference(coarse, fine), kind


class _GLEstimate(NamedTuple):
    error: float
    delta: float
    residual: float
    roundoff: float


def _rule_sum_roundoff(operations: int, values) -> float:
    magnitudes = tuple(abs(value) for value in values)
    try:
        norm = fsum(magnitudes)
    except OverflowError:
        # With large cancelling diagrams the absolute sum can overflow
        # even though both the signed sum and its scaled budget fit.
        return fsum(_roundoff_budget(operations, value) for value in magnitudes)
    return _roundoff_budget(operations, norm)


def _enhanced_gl_error(rows, signed_sums=None) -> _GLEstimate:
    """Compare signed sums at three node counts and carry their budgets."""
    rows = tuple(rows)
    if signed_sums is None:
        coarse, middle, fine = 0.0, 0.0, 0.0
        for row, refinement in rows:
            coarse += row["value"]
            middle += refinement.middle
            fine += refinement.fine
    else:
        coarse, middle, fine = signed_sums
    delta = _gl_difference(coarse, fine)
    residual = _gl_difference(middle, fine)
    operations = max(0, len(rows) - 1)
    middle_budget = _rule_sum_roundoff(
        operations, (refinement.middle for _, refinement in rows),
    )
    fine_budget = _rule_sum_roundoff(
        operations, (refinement.fine for _, refinement in rows),
    )
    roundoff = fsum((
        middle_budget, 2.0 * fine_budget,
        *(refinement.middle_roundoff for _, refinement in rows),
        *(2.0 * refinement.fine_roundoff for _, refinement in rows),
    ))
    # The middle/fine comparison uses both rules, and the final fine
    # estimate uses its own evaluation a second time.
    error = fsum((delta, residual, roundoff))
    if error:
        error = nextafter(error, float("inf"))
    if not isfinite(error):
        raise ValueError("Gauss-Legendre enhanced error estimate is not finite.")
    return _GLEstimate(error, delta, residual, roundoff)


@dataclass(frozen=True)
class Result:
    """Single-point evaluation result.

    Attributes:
        total: summed moment value across all orders.
        by_order: ``{order: value}``.
        by_vertex_type: ``{label: value}`` (e.g. ``{"F": 0.4, "FK":
            0.01}``).
        per_diagram: list of dicts, one per diagram evaluated.  Keys:
            ``order, diagram_idx, vertex_type, n_cross_C, value,
            error``.
        positions: the ``{spatial_arg: x}`` at which the evaluation
            was performed.
        t_final: the external-time upper bound used.
        component_pair: observable's component indices (e.g. ``(1,
            1)``).
        n_samples, seed: integrator settings (for reproducibility).
        qmc_diagnostics: Sample concentration metrics indexed by
            ``(order, diagram_idx)`` when ``qmc_heavy_tail`` was requested.
    """

    total: float
    by_order: dict
    by_vertex_type: dict
    per_diagram: list
    positions: dict
    t_final: float
    component_pair: tuple
    n_samples: int
    seed: Any
    qmc_replicas: int | None = None
    qmc_diagnostics: dict = field(default_factory=dict, repr=False)
    _replica_rows: tuple[tuple[dict, tuple[float, ...]], ...] = field(default=(), repr=False)
    gl_error: bool | Literal["enhanced"] = False
    _refinement_rows: tuple[tuple[dict, float | _GLRefinement], ...] = field(
        default=(), repr=False,
    )

    def __repr__(self) -> str:
        return (
            f"Result(total={self.total:.6e}, "
            f"by_order={_fmt_dict(self.by_order)}, "
            f"by_vertex_type={_fmt_dict(self.by_vertex_type)}, "
            f"positions={self.positions}, t_final={self.t_final}, "
            f"component_pair={self.component_pair})"
        )

    def to_dict(self) -> dict:
        """Flat dict suitable for serialisation or tabulation."""
        result = {
            "total": self.total,
            "by_order": dict(self.by_order),
            "by_vertex_type": dict(self.by_vertex_type),
            "positions": dict(self.positions),
            "t_final": self.t_final,
            "component_pair": tuple(self.component_pair),
            "n_samples": self.n_samples,
            "seed": self.seed,
            "n_diagrams_evaluated": len(self.per_diagram),
        }
        if self.qmc_replicas is not None or self.gl_error:
            error, kind = self._group_error(self._error_rows)
            result.update(
                error=None if kind == "heavy_tail" else error,
                error_kind=kind,
                error_by_order=self._errors_by("order", self.by_order, json_safe=True),
                error_by_vertex_type=self._errors_by(
                    "vertex_type", self.by_vertex_type, json_safe=True,
                ),
            )
            if self.qmc_replicas is not None:
                result["qmc_replicas"] = self.qmc_replicas
            if self.gl_error == "enhanced":
                result["gl_delta"] = _enhanced_gl_error(self._refinement_rows).delta
                result["gl_delta_by_order"] = self._errors_by(
                    "order", self.by_order, raw_gl=True,
                )
                result["gl_delta_by_vertex_type"] = self._errors_by(
                    "vertex_type", self.by_vertex_type, raw_gl=True,
                )
        return result

    @property
    def _error_rows(self):
        return self._refinement_rows if self.gl_error else self._replica_rows

    def _group_error(self, rows):
        if self.gl_error == "enhanced":
            rows = tuple(rows)
            return _enhanced_gl_error(rows).error, "gl_estimate" if rows else "exact_zero"
        if self.gl_error:
            return _refinement_group_error(rows)
        return _replica_group_error(rows, self.qmc_replicas)

    @property
    def error(self) -> float | None:
        """RQMC standard error or requested GL error diagnostic."""
        if self.qmc_replicas is None and not self.gl_error:
            return None
        return self._group_error(self._error_rows)[0]

    def _errors_by(self, column, groups, *, json_safe=False, raw_gl=False):
        if self.qmc_replicas is None and not self.gl_error:
            return None
        details = defaultdict(list)
        for row, value in self._error_rows:
            details[row[column]].append((row, value))
        errors = {}
        for group in groups:
            if raw_gl:
                errors[group] = _enhanced_gl_error(details[group]).delta
            else:
                error, kind = self._group_error(details[group])
                errors[group] = None if json_safe and kind == "heavy_tail" else error
        return errors

    @property
    def error_by_order(self) -> dict | None:
        """Requested error estimates by perturbative order."""
        return self._errors_by("order", self.by_order)

    @property
    def error_by_vertex_type(self) -> dict | None:
        """Requested error estimates by vertex composition."""
        return self._errors_by("vertex_type", self.by_vertex_type)


@dataclass(frozen=True)
class SweepResult:
    """Grid-sweep result.  Internally a tidy row store.

    Each row is a dict with:

    - the sweep coordinate fields (``x``, ``y``, ..., ``t_final``, and one
      component column per observable operator: ``a``, ``b`` for a
      2-point observable, ``a``, ``b``, ``c`` for a 3-point one),
    - the diagram coordinate fields (``order``, ``diagram_idx``,
      ``vertex_type``, ``n_cross_C``),
    - the integrand output (``value``, ``error``).

    Typical usage::

        sweep.to_dataframe().groupby(['r', 't_final']).value.sum()
    """

    rows: list
    position_keys: tuple
    #: ``("t_x", "t_y", ...)`` when the sweep pinned externals at unequal
    #: times via ``external_times_grid``; empty otherwise.  Kept separate from
    #: ``position_keys`` because a point's spatial position and its time are
    #: independent sweep axes, and defaulted so existing construction sites
    #: and any pickled SweepResult keep working.
    external_time_keys: tuple = ()
    #: Component columns, one per observable operator, named by position:
    #: ``("a", "b")`` for a 2-point observable, ``("a", "b", "c")`` for a
    #: 3-point one.  Defaults to the 2-point pair so existing construction
    #: sites and pickled SweepResults keep working.
    component_keys: tuple = ("a", "b")
    qmc_replicas: int | None = None
    seed: int | None = None
    _replica_rows: tuple[tuple[dict, tuple[float, ...]], ...] = field(default=(), repr=False)
    gl_error: bool | Literal["enhanced"] = False
    _refinement_rows: tuple[tuple[dict, float | _GLRefinement], ...] = field(
        default=(), repr=False,
    )

    def to_dataframe(self):
        """Convert to a :class:`pandas.DataFrame`.  Requires pandas.

        Columns include position keys, ``t_final``, any ``t_<point>``
        columns, the component columns (:attr:`component_keys`),
        ``order``, ``diagram_idx``, ``vertex_type``, ``n_cross_C``,
        ``value``, ``error``.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "SweepResult.to_dataframe requires pandas.  Install "
                "with `pip install pandas`."
            ) from e
        return pd.DataFrame(self.rows)

    def totals(self) -> "pd.DataFrame":  # noqa: F821
        """Sum across diagrams at each sweep point, grouped by
        ``order`` and sweep coordinates.  Returns a DataFrame with
        one row per (positions, t_final, components, order) and a
        ``value`` column; the components are ``a, b`` for a 2-point
        observable and ``a, b, c, ...`` for longer ones.
        """
        group_cols = (
            list(self.position_keys) + ["t_final"]
            + list(self.external_time_keys) + list(self.component_keys)
            + ["order"]
        )
        return self._totals(group_cols)

    def by_vertex_type_totals(self) -> "pd.DataFrame":  # noqa: F821
        """Sum across diagrams grouped by (positions, t_final,
        components, ``vertex_type``).  Produces the demo2-style channel
        decomposition."""
        group_cols = (
            list(self.position_keys) + ["t_final"]
            + list(self.external_time_keys) + list(self.component_keys)
            + ["vertex_type"]
        )
        return self._totals(group_cols)

    def _totals(self, group_cols):
        frame = self.to_dataframe()
        groups = frame.groupby(group_cols, as_index=False)
        totals = groups["value"].sum()
        if self.qmc_replicas is not None:
            self._add_replica_errors(groups, totals)
        elif self.gl_error:
            self._add_refinement_errors(frame, group_cols, totals)
        return totals

    def _add_replica_errors(self, groups, totals):
        # Keep the matching observations if callers reorder or filter
        # the public rows list. Replaced rows have no replica history.
        values_by_row = {id(row): values for row, values in self._replica_rows}
        if any(id(row) not in values_by_row for row in self.rows):
            raise ValueError("RQMC totals require the original diagram rows.")
        errors, kinds = [], []
        for _, group in groups:
            diagram_rows = (self.rows[i] for i in group.index)
            error, kind = _replica_group_error(
                ((row, values_by_row[id(row)]) for row in diagram_rows),
                self.qmc_replicas,
            )
            errors.append(error)
            kinds.append(kind)
        totals["error"] = errors
        totals["error_kind"] = kinds

    def _add_refinement_errors(self, frame, group_cols, totals):
        import pandas as pd

        fine_by_row = {id(row): fine for row, fine in self._refinement_rows}
        if any(id(row) not in fine_by_row for row in self.rows):
            raise ValueError("GL convergence differences require the original diagram rows.")
        keys = [frame[col] for col in group_cols]
        if self.gl_error == "enhanced":
            refinements = [fine_by_row[id(row)] for row in self.rows]
            fine = pd.Series([refinement.fine for refinement in refinements], index=frame.index)
            middle = pd.Series([refinement.middle for refinement in refinements], index=frame.index)
            groups = frame["value"].groupby(keys)
            errors, deltas = [], []
            sums = zip(groups.sum(), middle.groupby(keys).sum(), fine.groupby(keys).sum(), strict=True)
            for (_, coarse), signed_sums in zip(groups, sums, strict=True):
                rows = ((self.rows[i], refinements[i]) for i in coarse.index)
                estimate = _enhanced_gl_error(rows, signed_sums)
                errors.append(estimate.error)
                deltas.append(estimate.delta)
            totals["error"] = errors
            totals["gl_delta"] = deltas
            active = frame["error_kind"].eq("gl_estimate").groupby(keys).any()
            totals["error_kind"] = np.where(active, "gl_estimate", "exact_zero")
            return
        fine = pd.Series(
            [fine_by_row[id(row)] for row in self.rows], index=frame.index, dtype=float,
        )
        # Use the same grouped summation as in the physical sweep totals.
        coarse_sums = frame["value"].groupby(keys).sum()
        fine_sums = fine.groupby(keys).sum()
        totals["error"] = [_gl_difference(a, b) for a, b in zip(coarse_sums, fine_sums)]
        has_delta = frame["error_kind"].eq("gl_delta").groupby(keys).any()
        totals["error_kind"] = np.where(has_delta, "gl_delta", "exact_zero")

    def plot(
        self,
        *,
        x: str,
        y: str = "value",
        hue: str | None = "order",
        facet_col: str | None = None,
        aggregate: str = "sum",
        **kwargs,
    ):
        """Quick line plot via matplotlib.

        Args:
            x: column to put on the x-axis (typically a position
                key).
            y: value column (default ``'value'``).
            hue: column whose unique values become separate lines
                (default ``'order'``).
            facet_col: column whose unique values become subplot
                panels.  ``None`` ⇒ single panel.
            aggregate: ``'sum'`` or ``'none'``.  ``'sum'`` aggregates
                ``y`` across all other (non-``x``/``hue``/``facet``)
                coordinates before plotting.
            kwargs: forwarded to ``matplotlib.axes.Axes.plot``.
        """
        import matplotlib.pyplot as plt

        df = self.to_dataframe()

        if aggregate == "sum":
            group_cols = [x]
            if hue is not None:
                group_cols.append(hue)
            if facet_col is not None:
                group_cols.append(facet_col)
            # x, hue and facet_col may name one column; pandas 2 refuses a
            # repeated grouping key.
            group_cols = list(dict.fromkeys(group_cols))
            groups = df.groupby(group_cols, as_index=False)
            if y == "error" and self.qmc_replicas is not None:
                df = groups.first()[group_cols].copy()
                self._add_replica_errors(groups, df)
            elif (y == "error" and self.gl_error
                  or y == "gl_delta" and self.gl_error == "enhanced"):
                totals = groups.first()[group_cols].copy()
                self._add_refinement_errors(df, group_cols, totals)
                df = totals
            else:
                df = groups[y].sum()

        if facet_col is None:
            fig, ax = plt.subplots()
            _plot_into(ax, df, x, y, hue, **kwargs)
            return fig

        facet_values = sorted(df[facet_col].unique())
        fig, axes = plt.subplots(
            1, len(facet_values), sharey=True,
            figsize=(4 * len(facet_values), 4),
        )
        if len(facet_values) == 1:
            axes = [axes]
        for ax, fv in zip(axes, facet_values):
            sub = df[df[facet_col] == fv]
            _plot_into(ax, sub, x, y, hue, **kwargs)
            ax.set_title(f"{facet_col} = {fv}")
        return fig


def _plot_into(ax, df, x, y, hue, **kwargs):
    if hue is None:
        ax.plot(df[x], df[y], **kwargs)
    else:
        for h_val in sorted(df[hue].unique()):
            sub = df[df[hue] == h_val].sort_values(x)
            ax.plot(sub[x], sub[y], label=f"{hue}={h_val}", **kwargs)
        ax.legend()
    ax.set_xlabel(x)
    ax.set_ylabel(y)


def _fmt_dict(d: dict) -> str:
    return "{" + ", ".join(f"{k}: {v:.3e}" for k, v in sorted(d.items())) + "}"
