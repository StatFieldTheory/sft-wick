"""Matplotlib rendering for :class:`FeynmanDiagram` objects.

Visual conventions are now controlled by a
:class:`~sft_wick.render_style.RenderStyle` instance: colours,
linestyles, marker sizes, fonts, layout parameters, and label
formatting are all configurable.  See :mod:`sft_wick.render_style`
for the available presets (default / publication / grayscale /
minimal) and :mod:`sft_wick.render_labels` for the label
override hooks.

The constructor remains backward-compatible — calling
``DiagramRenderer(figsize=(8, 6))`` keeps working with the
``default_style`` preset.

For a LaTeX-native alternative, see
:class:`sft_wick.drawing_tikz.TikzRenderer`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .diagrams import FeynmanDiagram
from .render_labels import (
    LabelCallable,
    default_external_label,
    default_vertex_label,
    resolve_label,
)
from .render_layout import compute_layout, label_offset, neighbor_center
from .render_style import (
    LayoutParams,
    NodeStyle,
    PropagatorStyle,
    RenderStyle,
    default_style,
)


# Map our backend-agnostic linestyle names to matplotlib values.
_MPL_LINESTYLE: dict[str, str] = {
    "solid": "-",
    "dashed": "--",
    "dotted": ":",
    "dashdot": "-.",
}

_MPL_MARKER: dict[str, str] = {
    "circle": "o",
    "square": "s",
    "diamond": "D",
    "triangle": "^",
}


class DiagramRenderer:
    """Render :class:`FeynmanDiagram` objects with matplotlib.

    Args:
        figsize: Figure size in inches for the single-diagram entry
            point and the per-cell size for ``draw_all``.
        style: Visual style.  ``None`` → :func:`default_style`.
        external_label_fn: Optional callable
            ``fn(node_id, node_attrs) -> str | None`` invoked per
            external node.  Returning ``None`` falls through to the
            built-in formatter.
        vertex_label_fn: Same shape, for interaction vertices.
        label_format: Default external-label format
            (:data:`~sft_wick.render_style.LABEL_COMPACT` /
            ``LABEL_FULL`` / ``LABEL_TIME_F``).  Overrides
            ``style.label_format`` when supplied explicitly.
    """

    # Backwards-compatible class attribute.  Users that previously
    # mutated ``DiagramRenderer.PROP_STYLES["C"]["color"]`` to retheme
    # globally still see their override take effect via
    # :meth:`_resolve_style`.
    PROP_STYLES: dict[str, dict[str, Any]] = {
        "C": {"linestyle": "-", "color": "#2166ac", "linewidth": 2.0},
        "R": {"linestyle": "--", "color": "#d6604d", "linewidth": 2.0},
    }

    def __init__(
        self,
        figsize: tuple[float, float] = (6, 5),
        style: RenderStyle | None = None,
        external_label_fn: LabelCallable | None = None,
        vertex_label_fn: LabelCallable | None = None,
        label_format: str | None = None,
    ) -> None:
        self.figsize = figsize
        # Track whether the user supplied a custom style.  When they
        # did not, we honour any monkey-patches to the legacy class
        # attribute ``PROP_STYLES`` (for backward compatibility);
        # when they did, we respect their style verbatim.
        self._user_supplied_style: bool = style is not None
        self._raw_style: RenderStyle = style if style is not None else default_style()
        self.external_label_fn = external_label_fn
        self.vertex_label_fn = vertex_label_fn
        self._label_format_override = label_format

    # ------------------------------------------------------------------
    # Style helpers
    # ------------------------------------------------------------------

    @property
    def style(self) -> RenderStyle:
        """The active :class:`RenderStyle`.

        When the user did *not* pass a ``style`` to the constructor,
        any monkey-patches to the legacy class attribute
        :attr:`PROP_STYLES` are folded in for backward compatibility.
        When the user did pass a ``style``, it is respected verbatim.
        """
        return self._resolve_style()

    def _resolve_style(self) -> RenderStyle:
        base = self._raw_style
        if self._user_supplied_style:
            return base
        # Backwards-compat: pick up any tweaks the user made to
        # ``DiagramRenderer.PROP_STYLES`` at the class level.
        legacy_props: dict[str, PropagatorStyle] = {}
        for kind, defaults in type(self).PROP_STYLES.items():
            current = base.propagators.get(kind)
            if current is None:
                continue
            ls_name = _matplotlib_linestyle_to_name(defaults.get("linestyle"))
            legacy_props[kind] = PropagatorStyle(
                color=defaults.get("color", current.color),
                linestyle=ls_name or current.linestyle,
                linewidth=defaults.get("linewidth", current.linewidth),
                arrow=current.arrow,
                legend_label=current.legend_label,
            )
        if legacy_props:
            new_props = dict(base.propagators)
            new_props.update(legacy_props)
            return base.with_overrides(propagators=new_props)
        return base

    @property
    def label_format(self) -> str:
        return (
            self._label_format_override
            if self._label_format_override is not None
            else self._raw_style.label_format
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(
        self,
        diagram: FeynmanDiagram,
        ax: plt.Axes | None = None,
        title: str = "",
        title_kwargs: Mapping[str, Any] | None = None,
        external_labels: Mapping[str, str] | None = None,
        vertex_labels: Mapping[str, str] | None = None,
        positions: Mapping[str, tuple[float, float]] | None = None,
        show_legend: bool | None = None,
    ) -> plt.Axes:
        """Draw a single Feynman diagram on a matplotlib axes.

        Args:
            diagram:         The diagram to render.
            ax:              Existing axes to draw on; ``None`` →
                             a new figure of size ``self.figsize``.
            title:           Per-axes title.  Empty string → use
                             ``diagram.summary()``.
            title_kwargs:    Extra kwargs passed to ``ax.set_title``
                             (overrides the style's
                             ``title_fontsize``).
            external_labels: Map ``{node_id: label}`` overriding
                             specific external-vertex labels.
            vertex_labels:   Same shape, for interaction vertices.
            positions:       Map ``{node_id: (x, y)}`` pinning
                             specific node coordinates and skipping
                             spring layout for them.
            show_legend:     Override ``style.show_legend``.
        """
        style = self._resolve_style()
        with mpl.rc_context(style.effective_rcparams()):
            return self._draw_within_context(
                diagram=diagram,
                ax=ax,
                title=title,
                title_kwargs=title_kwargs,
                external_labels=external_labels,
                vertex_labels=vertex_labels,
                positions=positions,
                show_legend=show_legend,
                style=style,
            )

    def draw_all(
        self,
        diagrams: list[FeynmanDiagram],
        ncols: int = 3,
        suptitle: str = "Feynman Diagrams",
        suptitle_kwargs: Mapping[str, Any] | None = None,
        subtitle_fn: Callable[[int, FeynmanDiagram, int], str] | None = None,
        multiplicities: Sequence[int] | None = None,
        show: bool = False,
        external_labels: Mapping[int, Mapping[str, str]] | None = None,
    ) -> plt.Figure:
        """Draw a grid of Feynman diagrams.

        Args:
            diagrams:        Diagrams to lay out, one per subplot.
            ncols:           Maximum columns (clipped to ``len(diagrams)``).
            suptitle:        Figure-level title; pass ``""`` to skip.
            suptitle_kwargs: Extra kwargs forwarded to ``fig.suptitle``.
            subtitle_fn:     Per-subplot title formatter
                             ``fn(index, diagram, multiplicity) -> str``.
                             ``None`` → ``"#i: <summary>  [xN]"``.
            multiplicities:  Optional per-diagram multiplicities (for
                             default subtitles).
            show:            If ``True``, call ``plt.show()`` after
                             building the figure.  Default ``False``
                             — callers that ``savefig`` should leave
                             this off.
            external_labels: Optional map keyed by *subplot index*
                             whose values are
                             ``{node_id: label}`` overrides for that
                             subplot.

        Returns:
            The matplotlib :class:`Figure` (always returned, even when
            ``show=True``).
        """
        n = len(diagrams)
        style = self._resolve_style()
        if n == 0:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.text(0.5, 0.5, "No diagrams", ha="center", va="center")
            ax.axis("off")
            if show:
                plt.show()
            return fig

        if multiplicities is None:
            multiplicities = [1] * n

        ncols = min(ncols, n)
        nrows = (n + ncols - 1) // ncols

        with mpl.rc_context(style.effective_rcparams()):
            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(self.figsize[0] * ncols, self.figsize[1] * nrows),
            )
            if nrows == 1 and ncols == 1:
                axes = np.array([axes])
            axes_flat = np.array(axes).flatten()

            for i, (diagram, ax) in enumerate(zip(diagrams, axes_flat)):
                mult = multiplicities[i]
                if subtitle_fn is not None:
                    label = subtitle_fn(i, diagram, mult)
                else:
                    summary = diagram.summary(short=True)
                    label = f"#{i + 1}: {summary}"
                    if mult > 1:
                        label += f"  [x{mult}]"
                ext_overrides = (
                    external_labels.get(i)
                    if external_labels is not None else None
                )
                self._draw_within_context(
                    diagram=diagram,
                    ax=ax,
                    title=label,
                    title_kwargs=None,
                    external_labels=ext_overrides,
                    vertex_labels=None,
                    positions=None,
                    show_legend=None,
                    style=style,
                )

            for j in range(n, len(axes_flat)):
                axes_flat[j].axis("off")

            if suptitle:
                kw: dict[str, Any] = {"fontsize": style.suptitle_fontsize}
                if suptitle_kwargs:
                    kw.update(dict(suptitle_kwargs))
                fig.suptitle(suptitle, **kw)
                fig.tight_layout(rect=(0, 0, 1, 0.95))
            else:
                fig.tight_layout()

        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Internal: do the actual drawing inside the rc_context block
    # ------------------------------------------------------------------

    def _draw_within_context(
        self,
        diagram: FeynmanDiagram,
        ax: plt.Axes | None,
        title: str,
        title_kwargs: Mapping[str, Any] | None,
        external_labels: Mapping[str, str] | None,
        vertex_labels: Mapping[str, str] | None,
        positions: Mapping[str, tuple[float, float]] | None,
        show_legend: bool | None,
        style: RenderStyle,
    ) -> plt.Axes:
        if ax is None:
            _fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        g = diagram.graph
        if g.number_of_nodes() == 0:
            shown_title = title or "Empty diagram"
            self._set_title(ax, shown_title, title_kwargs, style)
            ax.axis("off")
            return ax

        pos = compute_layout(diagram, style.layout, positions)

        # Pre-count parallel edges and self-loops
        edge_pair_count: Counter = Counter()
        self_loop_count: Counter = Counter()
        for u, v, _key in g.edges(keys=True):
            if u == v:
                self_loop_count[u] += 1
            else:
                pair = (min(u, v), max(u, v))
                edge_pair_count[pair] += 1

        edge_pair_idx: Counter = Counter()
        self_loop_idx: Counter = Counter()

        # Edges
        for u, v, _key, data in g.edges(keys=True, data=True):
            kind = data.get("kind", "C")
            prop_style = style.propagators.get(kind)
            if prop_style is None:
                prop_style = next(iter(style.propagators.values()))
            p1 = np.array(pos[u])
            p2 = np.array(pos[v])

            if u == v:
                idx = self_loop_idx[u]
                total = self_loop_count[u]
                self_loop_idx[u] += 1
                ncenter = neighbor_center(g, u, pos)
                self._draw_self_loop(
                    ax, p1, prop_style,
                    loop_index=idx, n_loops=total,
                    away_from=ncenter,
                )
            else:
                pair = (min(u, v), max(u, v))
                idx = edge_pair_idx[pair]
                total = edge_pair_count[pair]
                edge_pair_idx[pair] += 1

                phi_end = data.get("phi_end")
                if prop_style.arrow and phi_end is not None and phi_end == u:
                    self._draw_edge(ax, p1, p2, prop_style,
                                    key=idx, n_parallel=total)
                elif prop_style.arrow and phi_end is not None:
                    self._draw_edge(ax, p2, p1, prop_style,
                                    key=idx, n_parallel=total)
                else:
                    self._draw_edge(ax, p1, p2, prop_style,
                                    key=idx, n_parallel=total)

        # Nodes
        ext_nodes = diagram.external_nodes
        vert_nodes = diagram.vertex_nodes
        all_pts = np.array([pos[n] for n in g.nodes()])
        center = np.mean(all_pts, axis=0)

        for n in ext_nodes:
            self._draw_node(
                ax, pos[n], style.external_node, zorder=5,
            )
            label = resolve_label(
                node_id=n,
                node_attrs=g.nodes[n],
                overrides=external_labels,
                callable_fn=self.external_label_fn,
                default_fn=lambda attrs: default_external_label(
                    attrs, fmt=self.label_format,
                ),
            )
            self._draw_label(
                ax, pos[n], label, style.external_label,
                center=center,
            )

        for n in vert_nodes:
            self._draw_node(
                ax, pos[n], style.vertex_node, zorder=5,
            )
            label = resolve_label(
                node_id=n,
                node_attrs=g.nodes[n],
                overrides=vertex_labels,
                callable_fn=self.vertex_label_fn,
                default_fn=default_vertex_label,
            )
            if label:
                self._draw_label(
                    ax, pos[n], label, style.vertex_label,
                    center=center,
                )

        # Legend
        legend_visible = (
            style.show_legend if show_legend is None else show_legend
        )
        if legend_visible:
            self._draw_legend(ax, g, style)

        # Title
        shown_title = title if title else diagram.summary()
        self._set_title(ax, shown_title, title_kwargs, style)

        ax.axis("off")
        ax.set_aspect("equal")

        margin = style.layout.margin
        # The layout is normalised iff bbox-normalisation is on AND the
        # caller did not pin any nodes (manual positions skip the
        # normalisation pass — see render_layout.compute_layout).
        layout_was_normalised = (
            style.layout.normalize_bbox and not positions
        )
        if layout_was_normalised:
            # Use the standardised target extent so every panel in a
            # grid has identical coordinate limits — this is what
            # makes loop radii, edge thicknesses, and marker sizes
            # look consistent across diagrams.
            half_w = style.layout.target_extent[0] / 2.0
            half_h = style.layout.target_extent[1] / 2.0
            ax.set_xlim(-half_w - margin, half_w + margin)
            ax.set_ylim(-half_h - margin, half_h + margin)
        else:
            ax.set_xlim(
                all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin,
            )
            ax.set_ylim(
                all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin,
            )

        return ax

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def _draw_edge(
        self,
        ax: plt.Axes,
        p1: np.ndarray,
        p2: np.ndarray,
        style: PropagatorStyle,
        key: int = 0,
        n_parallel: int = 1,
    ) -> None:
        from matplotlib.patches import FancyArrowPatch

        if n_parallel <= 1:
            rad = 0.0
        else:
            idx = key - (n_parallel - 1) / 2.0
            rad = idx * 0.35

        arrow = FancyArrowPatch(
            posA=tuple(p1),
            posB=tuple(p2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>" if style.arrow else "-",
            color=style.color,
            linestyle=_MPL_LINESTYLE.get(style.linestyle, style.linestyle),
            linewidth=style.linewidth,
            mutation_scale=12,
            shrinkA=4,
            shrinkB=4,
            zorder=3 if style.arrow else 2,
        )
        ax.add_patch(arrow)

    def _draw_self_loop(
        self,
        ax: plt.Axes,
        center: np.ndarray,
        style: PropagatorStyle,
        loop_index: int = 0,
        n_loops: int = 1,
        away_from: np.ndarray | None = None,
    ) -> None:
        if away_from is not None:
            base_dir = center - away_from
            norm = np.linalg.norm(base_dir)
            if norm > 1e-8:
                base_dir = base_dir / norm
            else:
                base_dir = np.array([0.0, 1.0])
        else:
            base_dir = np.array([0.0, 1.0])

        if n_loops > 1:
            spread = np.pi / 3
            angle_offset = (loop_index - (n_loops - 1) / 2) * spread
            cos_a, sin_a = np.cos(angle_offset), np.sin(angle_offset)
            direction = np.array([
                base_dir[0] * cos_a - base_dir[1] * sin_a,
                base_dir[0] * sin_a + base_dir[1] * cos_a,
            ])
        else:
            direction = base_dir

        loop_radius = 0.5
        loop_center = center + (loop_radius + 0.05) * direction

        circle = plt.Circle(
            tuple(loop_center),
            loop_radius,
            fill=False,
            color=style.color,
            linestyle=_MPL_LINESTYLE.get(style.linestyle, style.linestyle),
            linewidth=style.linewidth,
            zorder=2,
        )
        ax.add_patch(circle)

        if style.arrow:
            angle = np.arctan2(direction[1], direction[0])
            arrow_angle = angle + np.pi / 3
            tip = loop_center + loop_radius * np.array(
                [np.cos(arrow_angle), np.sin(arrow_angle)])
            tail = loop_center + loop_radius * np.array(
                [np.cos(arrow_angle + 0.2), np.sin(arrow_angle + 0.2)])
            ax.annotate(
                "", xy=tuple(tip), xytext=tuple(tail),
                arrowprops=dict(arrowstyle="-|>", color=style.color,
                                lw=style.linewidth),
            )

    def _draw_node(
        self,
        ax: plt.Axes,
        point: np.ndarray,
        style: NodeStyle,
        zorder: int = 5,
    ) -> None:
        marker = _MPL_MARKER.get(style.shape, "o")
        face = style.color if style.fill else "none"
        edge = style.edge_color if style.edge_color != "none" else style.color
        ax.plot(
            point[0], point[1],
            marker=marker,
            markersize=style.size,
            markerfacecolor=face,
            markeredgecolor=edge,
            linestyle="None",
            zorder=zorder,
        )

    def _draw_label(
        self,
        ax: plt.Axes,
        point: np.ndarray,
        text: str,
        style: "Any",  # LabelStyle, but loosely typed to avoid extra import here
        center: np.ndarray,
    ) -> None:
        if not text:
            return
        offset = label_offset(point, center, distance=style.offset_pt)
        bbox = (
            dict(
                boxstyle="round,pad=0.15", facecolor="white",
                edgecolor="none", alpha=style.bbox_alpha,
            )
            if style.bbox else None
        )
        ax.annotate(
            text, point,
            textcoords="offset points", xytext=offset,
            ha="center", va="center",
            fontsize=style.fontsize,
            fontweight="bold" if style.bold else "normal",
            bbox=bbox,
            zorder=6,
        )

    def _draw_legend(
        self,
        ax: plt.Axes,
        g: nx.MultiGraph,
        style: RenderStyle,
    ) -> None:
        edge_kinds = {data.get("kind") for _, _, data in g.edges(data=True)}
        handles = []
        for kind, prop in style.propagators.items():
            if kind not in edge_kinds:
                continue
            label = prop.legend_label
            if not label:
                continue
            handles.append(plt.Line2D(
                [0], [0],
                color=prop.color,
                linestyle=_MPL_LINESTYLE.get(prop.linestyle, prop.linestyle),
                lw=prop.linewidth,
                label=label,
            ))
        if handles:
            ax.legend(
                handles=handles,
                loc=style.legend_loc,
                fontsize=style.legend_fontsize,
                framealpha=0.8,
                edgecolor="none",
            )

    def _set_title(
        self,
        ax: plt.Axes,
        title: str,
        title_kwargs: Mapping[str, Any] | None,
        style: RenderStyle,
    ) -> None:
        if not title or style.title_fontsize <= 0:
            return
        kw: dict[str, Any] = {"fontsize": style.title_fontsize}
        if title_kwargs:
            kw.update(dict(title_kwargs))
        ax.set_title(title, **kw)

    # ------------------------------------------------------------------
    # Backwards-compat helpers (kept in case external code uses them)
    # ------------------------------------------------------------------

    @staticmethod
    def _neighbor_center(
        g: nx.MultiGraph, node: str, pos: dict[str, np.ndarray],
    ) -> np.ndarray:
        return neighbor_center(g, node, pos)

    @staticmethod
    def _label_offset(
        point: np.ndarray, center: np.ndarray, distance: float = 18,
    ) -> tuple[float, float]:
        return label_offset(point, center, distance=distance)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _matplotlib_linestyle_to_name(ls: str | None) -> str | None:
    """Reverse of ``_MPL_LINESTYLE``.

    The legacy ``PROP_STYLES`` class attribute uses ``"-"`` / ``"--"``;
    the new style API uses ``"solid"`` / ``"dashed"``.  This helper
    bridges the two so ``DiagramRenderer.PROP_STYLES["C"]["linestyle"]
    = "-."`` still works.
    """
    if ls is None:
        return None
    inverse = {
        "-": "solid",
        "--": "dashed",
        ":": "dotted",
        "-.": "dashdot",
    }
    return inverse.get(ls, ls)
