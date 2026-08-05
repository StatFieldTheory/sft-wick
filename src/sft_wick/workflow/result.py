"""``Result`` and ``SweepResult`` — structured output from
:meth:`Expansion.evaluate` / :meth:`Expansion.sweep`.

These are what users actually look at.  Accessors are designed around
three common analyses:

1. **Total**: summed moment value.
2. **By order**: per-perturbative-order contribution.  Reveals
   convergence.
3. **By vertex type**: demo2-style FF/FK/KK decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
        return {
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


@dataclass(frozen=True)
class SweepResult:
    """Grid-sweep result.  Internally a tidy row store.

    Each row is a dict with:

    - the sweep coordinate fields (``x``, ``y``, ..., ``t_final``,
      ``a``, ``b``),
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

    def to_dataframe(self):
        """Convert to a :class:`pandas.DataFrame`.  Requires pandas.

        Columns include position keys, ``t_final``, ``a``, ``b``,
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
        one row per (positions, t_final, a, b, order) and a
        ``value`` column.
        """
        df = self.to_dataframe()
        group_cols = (
            list(self.position_keys) + ["t_final"]
            + list(self.external_time_keys) + ["a", "b", "order"]
        )
        return (
            df.groupby(group_cols, as_index=False)["value"]
            .sum()
        )

    def by_vertex_type_totals(self) -> "pd.DataFrame":  # noqa: F821
        """Sum across diagrams grouped by (positions, t_final, a, b,
        ``vertex_type``).  Produces the demo2-style channel
        decomposition."""
        df = self.to_dataframe()
        group_cols = (
            list(self.position_keys) + ["t_final"]
            + list(self.external_time_keys) + ["a", "b", "vertex_type"]
        )
        return (
            df.groupby(group_cols, as_index=False)["value"]
            .sum()
        )

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
            df = df.groupby(group_cols, as_index=False)[y].sum()

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
