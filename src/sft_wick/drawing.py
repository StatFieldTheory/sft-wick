"""Feynman diagram rendering using matplotlib.

C propagators: solid blue line (correlation)
R propagators: dashed red line with arrow (response)
External points: filled circles with labels
Vertices: filled squares with coupling labels
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .diagrams import FeynmanDiagram


class DiagramRenderer:
    """Renders FeynmanDiagram objects using matplotlib."""

    PROP_STYLES: dict[str, dict[str, Any]] = {
        "C": {"linestyle": "-", "color": "#2166ac", "linewidth": 2.0},
        "R": {"linestyle": "--", "color": "#d6604d", "linewidth": 2.0},
    }

    def __init__(self, figsize: tuple[float, float] = (6, 5)) -> None:
        self.figsize = figsize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(
        self,
        diagram: FeynmanDiagram,
        ax: plt.Axes | None = None,
        title: str = "",
    ) -> plt.Axes:
        """Draw a single Feynman diagram."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        g = diagram.graph
        if g.number_of_nodes() == 0:
            ax.set_title(title or "Empty diagram")
            ax.axis("off")
            return ax

        pos = self._compute_layout(diagram)

        # --- Pre-count parallel edges and self-loops ---
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

        # --- Draw edges ---
        for u, v, _key, data in g.edges(keys=True, data=True):
            kind = data.get("kind", "C")
            style = self.PROP_STYLES.get(kind, self.PROP_STYLES["C"])
            p1 = np.array(pos[u])
            p2 = np.array(pos[v])

            if u == v:
                idx = self_loop_idx[u]
                total = self_loop_count[u]
                self_loop_idx[u] += 1
                # Choose loop direction away from neighbors
                neighbor_center = self._neighbor_center(g, u, pos)
                self._draw_self_loop(
                    ax, p1, style, kind,
                    loop_index=idx, n_loops=total,
                    away_from=neighbor_center,
                )
            else:
                pair = (min(u, v), max(u, v))
                idx = edge_pair_idx[pair]
                total = edge_pair_count[pair]
                edge_pair_idx[pair] += 1

                # Determine direction for R propagators
                phi_end = data.get("phi_end")
                if kind == "R" and phi_end is not None:
                    if phi_end == u:
                        self._draw_edge(ax, p1, p2, style, kind,
                                        key=idx, n_parallel=total)
                    else:
                        self._draw_edge(ax, p2, p1, style, kind,
                                        key=idx, n_parallel=total)
                else:
                    self._draw_edge(ax, p1, p2, style, kind,
                                    key=idx, n_parallel=total)

        # --- Draw nodes ---
        ext_nodes = diagram.external_nodes
        vert_nodes = diagram.vertex_nodes

        # Compute label directions: away from center of mass
        all_pts = np.array([pos[n] for n in g.nodes()])
        center = np.mean(all_pts, axis=0)

        _label_bbox = dict(
            boxstyle="round,pad=0.15", facecolor="white",
            edgecolor="none", alpha=0.85,
        )

        for n in ext_nodes:
            p = pos[n]
            label = g.nodes[n].get("label", n)
            ax.plot(*p, "ko", markersize=8, zorder=5)
            offset = self._label_offset(p, center, distance=26)
            ax.annotate(
                label, p,
                textcoords="offset points", xytext=offset,
                ha="center", va="center", fontsize=9,
                bbox=_label_bbox, zorder=6,
            )

        for n in vert_nodes:
            p = pos[n]
            coupling = g.nodes[n].get("coupling", "")
            ax.plot(*p, "ks", markersize=10, zorder=5)
            # Place coupling label to the side, away from edges
            offset = self._label_offset(p, center, distance=20)
            ax.annotate(
                f"${coupling}$", p,
                textcoords="offset points", xytext=offset,
                ha="center", va="center", fontsize=9, fontweight="bold",
                bbox=_label_bbox, zorder=6,
            )

        # --- Legend ---
        legend_handles = []
        edge_kinds = {data.get("kind") for _, _, data in g.edges(data=True)}
        if "C" in edge_kinds:
            legend_handles.append(
                plt.Line2D([0], [0], color=self.PROP_STYLES["C"]["color"],
                           linestyle="-", lw=2, label="C (correlation)")
            )
        if "R" in edge_kinds:
            legend_handles.append(
                plt.Line2D([0], [0], color=self.PROP_STYLES["R"]["color"],
                           linestyle="--", lw=2, label="R (response)")
            )
        if legend_handles:
            ax.legend(
                handles=legend_handles, loc="lower right",
                fontsize=7, framealpha=0.8, edgecolor="none",
            )

        ax.set_title(title or diagram.summary(), fontsize=10)
        ax.axis("off")
        ax.set_aspect("equal")

        # Padding so labels don't clip
        margin = 1.3
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)

        return ax

    def draw_all(
        self,
        diagrams: list[FeynmanDiagram],
        ncols: int = 3,
        suptitle: str = "Feynman Diagrams",
        multiplicities: Sequence[int] | None = None,
    ) -> plt.Figure:
        """Draw multiple Feynman diagrams in a subplot grid."""
        n = len(diagrams)
        if n == 0:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.text(0.5, 0.5, "No diagrams", ha="center", va="center")
            ax.axis("off")
            return fig

        if multiplicities is None:
            multiplicities = [1] * n

        ncols = min(ncols, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(self.figsize[0] * ncols, self.figsize[1] * nrows),
        )
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes_flat = np.array(axes).flatten()

        for i, (diagram, ax) in enumerate(zip(diagrams, axes_flat)):
            mult = multiplicities[i]
            summary = diagram.summary(short=True)
            label = f"#{i + 1}: {summary}"
            if mult > 1:
                label += f"  [x{mult}]"
            self.draw(diagram, ax=ax, title=label)

        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        return fig

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _compute_layout(self, diagram: FeynmanDiagram) -> dict[str, np.ndarray]:
        """Compute node positions for the diagram."""
        g = diagram.graph
        ext = diagram.external_nodes
        verts = diagram.vertex_nodes
        n_nodes = g.number_of_nodes()

        if n_nodes == 0:
            return {}
        if n_nodes == 1:
            return {list(g.nodes())[0]: np.array([0.0, 0.0])}

        pos: dict[str, np.ndarray] = {}

        n_ext = len(ext)
        n_vert = len(verts)

        # External nodes on a circle
        ext_radius = 2.5
        if n_ext == 2:
            # Special case: place externals on far left / far right
            # so they are always the outermost points.
            pos[ext[0]] = np.array([-ext_radius, 0.0])
            pos[ext[1]] = np.array([ext_radius, 0.0])
        elif n_ext > 0:
            for i, node in enumerate(ext):
                angle = 2 * np.pi * i / n_ext - np.pi / 2
                pos[node] = np.array([ext_radius * np.cos(angle),
                                      ext_radius * np.sin(angle)])

        # Vertex nodes
        if n_vert == 0:
            pass
        elif n_vert == 1:
            pos[verts[0]] = np.array([0.0, 0.0])
        elif n_ext == 0:
            # Only vertices, no externals: arrange in a circle
            for i, v in enumerate(verts):
                angle = 2 * np.pi * i / n_vert - np.pi / 2
                pos[v] = np.array([np.cos(angle), np.sin(angle)])
        else:
            # Multiple vertices with externals: spring layout with
            # external positions pinned
            init_pos = dict(pos)
            # Spread initial vertex positions to avoid collinear degeneration
            for i, v in enumerate(verts):
                angle = 2 * np.pi * i / n_vert + np.pi / 4
                init_pos[v] = np.array([0.5 * np.cos(angle),
                                        0.5 * np.sin(angle)])
            spring_pos = nx.spring_layout(
                g, pos=init_pos, fixed=ext,
                k=2.0, iterations=200, seed=42,
            )
            for v in verts:
                pos[v] = spring_pos[v]

            # Ensure minimum distance between all vertex pairs
            min_dist = 0.8
            for _ in range(10):  # iterate to convergence
                changed = False
                for iv, v1 in enumerate(verts):
                    for v2 in verts[iv + 1:]:
                        d = np.linalg.norm(pos[v1] - pos[v2])
                        if 0 < d < min_dist:
                            direction = (pos[v2] - pos[v1]) / d
                            push = (min_dist - d) / 2 + 0.05
                            pos[v1] = pos[v1] - push * direction
                            pos[v2] = pos[v2] + push * direction
                            changed = True
                        elif d == 0:
                            # Coincident: push apart randomly
                            pos[v2] = pos[v2] + np.array([min_dist, 0.0])
                            changed = True
                if not changed:
                    break

            # For 2-external layouts, clamp vertices to stay strictly
            # between the externals so externals are always outermost.
            if n_ext == 2:
                x_lo = min(pos[ext[0]][0], pos[ext[1]][0])
                x_hi = max(pos[ext[0]][0], pos[ext[1]][0])
                inset = 0.35 * (x_hi - x_lo)  # keep 35% margin
                for v in verts:
                    pos[v][0] = np.clip(pos[v][0],
                                        x_lo + inset, x_hi - inset)

        return pos

    # ------------------------------------------------------------------
    # Edge drawing
    # ------------------------------------------------------------------

    def _draw_edge(
        self,
        ax: plt.Axes,
        p1: np.ndarray,
        p2: np.ndarray,
        style: dict,
        kind: str,
        key: int = 0,
        n_parallel: int = 1,
    ) -> None:
        """Draw a propagator line between two points.

        Parallel edges alternate curvature direction so they don't
        overlap.
        """
        from matplotlib.patches import FancyArrowPatch

        # Curvature: center around zero, alternate sides
        if n_parallel <= 1:
            rad = 0.0
        else:
            idx = key - (n_parallel - 1) / 2.0
            rad = idx * 0.35

        arrow = FancyArrowPatch(
            posA=tuple(p1),
            posB=tuple(p2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>" if kind == "R" else "-",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            mutation_scale=12,
            shrinkA=4,
            shrinkB=4,
            zorder=3 if kind == "R" else 2,
        )
        ax.add_patch(arrow)

    def _draw_self_loop(
        self,
        ax: plt.Axes,
        center: np.ndarray,
        style: dict,
        kind: str,
        loop_index: int = 0,
        n_loops: int = 1,
        away_from: np.ndarray | None = None,
    ) -> None:
        """Draw a self-loop (tadpole) at a node.

        The loop is drawn as a circle tangent to the node, pointing
        away from neighboring nodes so it doesn't overlap edges.
        """
        # Determine the direction for this loop
        if away_from is not None:
            # Point away from the center of neighbors
            base_dir = center - away_from
            norm = np.linalg.norm(base_dir)
            if norm > 1e-8:
                base_dir = base_dir / norm
            else:
                base_dir = np.array([0.0, 1.0])
        else:
            base_dir = np.array([0.0, 1.0])

        # For multiple loops, rotate them apart
        if n_loops > 1:
            spread = np.pi / 3  # 60 degrees spread
            angle_offset = (loop_index - (n_loops - 1) / 2) * spread
            cos_a, sin_a = np.cos(angle_offset), np.sin(angle_offset)
            rotated = np.array([
                base_dir[0] * cos_a - base_dir[1] * sin_a,
                base_dir[0] * sin_a + base_dir[1] * cos_a,
            ])
            direction = rotated
        else:
            direction = base_dir

        loop_radius = 0.5
        loop_center = center + (loop_radius + 0.05) * direction

        circle = plt.Circle(
            tuple(loop_center),
            loop_radius,
            fill=False,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            zorder=2,
        )
        ax.add_patch(circle)

        # Arrow indicator for R self-loops
        if kind == "R":
            angle = np.arctan2(direction[1], direction[0])
            arrow_angle = angle + np.pi / 3
            tip = loop_center + loop_radius * np.array(
                [np.cos(arrow_angle), np.sin(arrow_angle)])
            tail = loop_center + loop_radius * np.array(
                [np.cos(arrow_angle + 0.2), np.sin(arrow_angle + 0.2)])
            ax.annotate(
                "", xy=tuple(tip), xytext=tuple(tail),
                arrowprops=dict(arrowstyle="-|>", color=style["color"],
                                lw=style["linewidth"]),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _neighbor_center(
        g: nx.MultiGraph, node: str, pos: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute the centroid of a node's neighbors (excluding self)."""
        neighbors = [n for n in g.neighbors(node) if n != node]
        if not neighbors:
            # No neighbors other than self: return position below
            return pos[node] + np.array([0.0, -1.0])
        return np.mean([pos[n] for n in neighbors], axis=0)

    @staticmethod
    def _label_offset(
        point: np.ndarray, center: np.ndarray, distance: float = 18,
    ) -> tuple[float, float]:
        """Compute a label offset (in points) pointing away from center."""
        direction = point - center
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return (0.0, distance)  # default: above
        direction = direction / norm
        return (direction[0] * distance, direction[1] * distance)
