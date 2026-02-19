"""Feynman diagram rendering using matplotlib.

C propagators: solid blue line (correlation)
R propagators: dashed red line with arrow (response)
External points: filled circles with labels
Vertices: filled dots with coupling labels
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from .diagrams import FeynmanDiagram


class DiagramRenderer:
    """Renders FeynmanDiagram objects using matplotlib."""

    PROP_STYLES: dict[str, dict[str, Any]] = {
        "C": {"linestyle": "-", "color": "blue", "linewidth": 2.0},
        "R": {"linestyle": "--", "color": "red", "linewidth": 2.0},
    }

    def __init__(self, figsize: tuple[float, float] = (6, 5)) -> None:
        self.figsize = figsize

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

        # Draw edges (propagators)
        for u, v, key, data in g.edges(keys=True, data=True):
            kind = data.get("kind", "C")
            style = self.PROP_STYLES.get(kind, self.PROP_STYLES["C"])
            p1 = np.array(pos[u])
            p2 = np.array(pos[v])

            if u == v:
                # Self-loop (tadpole)
                self._draw_self_loop(ax, p1, style, kind)
            else:
                # Offset multiple edges between same nodes
                self._draw_edge(ax, p1, p2, style, kind, key)

        # Draw nodes
        ext_nodes = diagram.external_nodes
        vert_nodes = diagram.vertex_nodes

        for n in ext_nodes:
            p = pos[n]
            label = g.nodes[n].get("label", n)
            ax.plot(*p, "ko", markersize=8, zorder=5)
            ax.annotate(
                label,
                p,
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=9,
            )

        for n in vert_nodes:
            p = pos[n]
            label = g.nodes[n].get("label", n)
            ax.plot(*p, "ks", markersize=10, zorder=5)
            ax.annotate(
                label,
                p,
                textcoords="offset points",
                xytext=(0, -16),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

        # Legend
        legend_handles = []
        edge_kinds = {data.get("kind") for _, _, data in g.edges(data=True)}
        if "C" in edge_kinds:
            legend_handles.append(
                mpatches.Patch(color="blue", label="C (correlation)")
            )
        if "R" in edge_kinds:
            legend_handles.append(
                mpatches.Patch(color="red", label="R (response)")
            )
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

        ax.set_title(title or diagram.summary())
        ax.axis("off")
        ax.set_aspect("equal")
        return ax

    def draw_all(
        self,
        diagrams: list[FeynmanDiagram],
        ncols: int = 3,
        suptitle: str = "Feynman Diagrams",
    ) -> plt.Figure:
        """Draw multiple Feynman diagrams in a grid."""
        n = len(diagrams)
        if n == 0:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.text(0.5, 0.5, "No diagrams", ha="center", va="center")
            ax.axis("off")
            return fig

        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(self.figsize[0] * ncols, self.figsize[1] * nrows),
        )
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes_flat = np.array(axes).flatten()

        for i, (diagram, ax) in enumerate(zip(diagrams, axes_flat)):
            self.draw(diagram, ax=ax, title=f"Diagram {i + 1}")

        # Hide unused axes
        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout()
        plt.show()
        return fig

    def _compute_layout(self, diagram: FeynmanDiagram) -> dict[str, np.ndarray]:
        """Compute node positions for the diagram."""
        g = diagram.graph
        ext = diagram.external_nodes
        verts = diagram.vertex_nodes

        if g.number_of_nodes() <= 1:
            return {n: np.array([0.0, 0.0]) for n in g.nodes()}

        # Use spring layout as starting point
        pos = nx.spring_layout(g, seed=42)

        # Place external nodes on a circle, vertices inside
        if ext and verts:
            n_ext = len(ext)
            for i, node in enumerate(ext):
                angle = 2 * np.pi * i / n_ext - np.pi / 2
                pos[node] = np.array([1.5 * np.cos(angle), 1.5 * np.sin(angle)])

            # Keep vertex positions from spring layout but scale down
            for node in verts:
                pos[node] = pos[node] * 0.5

        return pos

    def _draw_edge(
        self,
        ax: plt.Axes,
        p1: np.ndarray,
        p2: np.ndarray,
        style: dict,
        kind: str,
        key: int = 0,
    ) -> None:
        """Draw a propagator line between two points."""
        # Offset for multiple edges
        offset = 0.05 * key
        mid = (p1 + p2) / 2
        perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
        norm = np.linalg.norm(perp)
        if norm > 0:
            perp = perp / norm
        ctrl = mid + offset * perp

        ax.annotate(
            "",
            xy=p2,
            xytext=p1,
            arrowprops=dict(
                arrowstyle="->" if kind == "R" else "-",
                color=style["color"],
                linestyle=style["linestyle"],
                lw=style["linewidth"],
                connectionstyle=f"arc3,rad={0.1 * key}" if key > 0 else "arc3,rad=0",
            ),
        )

    def _draw_self_loop(
        self,
        ax: plt.Axes,
        center: np.ndarray,
        style: dict,
        kind: str,
    ) -> None:
        """Draw a self-loop (tadpole) at a node."""
        loop_radius = 0.15
        circle = plt.Circle(
            (center[0], center[1] + loop_radius),
            loop_radius,
            fill=False,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
        ax.add_patch(circle)
