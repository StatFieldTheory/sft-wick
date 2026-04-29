"""Position computation for Feynman-diagram nodes (backend-agnostic).

This module owns the geometric layout of a :class:`FeynmanDiagram`:

1. external nodes are placed on a circle (or, for the common
   two-external case, mirrored on the x-axis),
2. interaction vertices are seeded around the origin and refined by
   ``networkx.spring_layout`` with externals pinned,
3. a post-pass enforces a minimum vertex separation,
4. for two-external layouts, vertices are clamped horizontally so
   the externals stay outermost.

Both rendering backends (:mod:`sft_wick.drawing` and
:mod:`sft_wick.drawing_tikz`) call :func:`compute_layout` and then
translate the resulting positions to their own coordinate
conventions.

The function accepts a ``manual_positions`` map so users can pin
arbitrary nodes — particularly useful when the spring layout
produces an aesthetically poor result and the user wants full
control.
"""

from __future__ import annotations

from collections.abc import Mapping

import networkx as nx
import numpy as np

from .diagrams import FeynmanDiagram
from .render_style import LayoutParams


def compute_layout(
    diagram: FeynmanDiagram,
    params: LayoutParams = LayoutParams(),
    manual_positions: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, np.ndarray]:
    """Return ``{node_id: np.array([x, y])}`` for every node.

    Args:
        diagram: The diagram to lay out.
        params:  Layout parameters (radius, spring-layout knobs,
                 separation thresholds, …).
        manual_positions: Optional pin map.  Any node listed here is
                 placed at the given coordinate and excluded from
                 spring-layout updates.  Externals not listed fall
                 back to the circular placement; vertices not listed
                 follow the standard spring + min-distance pipeline.

    Returns:
        A dict mapping node-id to a length-2 numpy array.  Empty
        diagrams produce an empty dict; single-node diagrams return
        one entry at the origin.
    """
    g = diagram.graph
    ext = diagram.external_nodes
    verts = diagram.vertex_nodes
    n_nodes = g.number_of_nodes()

    if n_nodes == 0:
        return {}
    if n_nodes == 1:
        only = list(g.nodes())[0]
        if manual_positions and only in manual_positions:
            return {only: np.array(manual_positions[only], dtype=float)}
        return {only: np.array([0.0, 0.0])}

    pos: dict[str, np.ndarray] = {}
    pinned: set[str] = set()

    if manual_positions:
        for node_id, xy in manual_positions.items():
            if node_id in g.nodes:
                pos[node_id] = np.array(xy, dtype=float)
                pinned.add(node_id)

    # ------------------------------------------------------------------
    # External nodes — circle / mirrored pair
    # ------------------------------------------------------------------
    n_ext = len(ext)
    free_ext = [n for n in ext if n not in pinned]
    if n_ext == 2 and len(free_ext) == 2:
        pos[ext[0]] = np.array([-params.ext_radius, 0.0])
        pos[ext[1]] = np.array([params.ext_radius, 0.0])
    elif free_ext:
        # Place free externals at the angular slots they would occupy
        # in the full-circle scheme, so adding pins doesn't reshuffle
        # the others.
        for i, node in enumerate(ext):
            if node in pinned:
                continue
            angle = 2 * np.pi * i / max(n_ext, 1) - np.pi / 2
            pos[node] = np.array(
                [params.ext_radius * np.cos(angle),
                 params.ext_radius * np.sin(angle)]
            )

    # ------------------------------------------------------------------
    # Vertex nodes
    # ------------------------------------------------------------------
    n_vert = len(verts)
    free_verts = [v for v in verts if v not in pinned]

    if n_vert == 0:
        pass
    elif len(free_verts) == 0:
        pass  # all vertices are pinned
    elif n_vert == 1 and len(free_verts) == 1:
        pos[verts[0]] = np.array([0.0, 0.0])
    elif n_ext == 0:
        for i, v in enumerate(verts):
            if v in pinned:
                continue
            angle = 2 * np.pi * i / n_vert - np.pi / 2
            pos[v] = np.array([np.cos(angle), np.sin(angle)])
    else:
        # Spring layout with externals + pins fixed.
        init_pos = dict(pos)
        for i, v in enumerate(free_verts):
            angle = 2 * np.pi * i / max(len(free_verts), 1) + np.pi / 4
            init_pos.setdefault(
                v,
                np.array([0.5 * np.cos(angle), 0.5 * np.sin(angle)]),
            )
        fixed_nodes = list(pinned | set(ext))
        spring_pos = nx.spring_layout(
            g,
            pos=init_pos,
            fixed=fixed_nodes if fixed_nodes else None,
            k=params.spring_k,
            iterations=params.spring_iterations,
            seed=params.spring_seed,
        )
        for v in free_verts:
            pos[v] = spring_pos[v]

        # Min-distance enforcement between every pair of vertices that
        # are *not* both pinned (we never push pinned nodes).
        _enforce_min_distance(pos, verts, pinned, params.min_vertex_dist)

    # ------------------------------------------------------------------
    # Two-external clamp: keep vertices strictly between the externals
    # ------------------------------------------------------------------
    if n_ext == 2 and verts:
        x_lo = min(pos[ext[0]][0], pos[ext[1]][0])
        x_hi = max(pos[ext[0]][0], pos[ext[1]][0])
        if x_hi > x_lo:
            inset = 0.35 * (x_hi - x_lo)
            for v in verts:
                if v in pinned:
                    continue
                pos[v] = np.array([
                    np.clip(pos[v][0], x_lo + inset, x_hi - inset),
                    pos[v][1],
                ])

    # ------------------------------------------------------------------
    # Normalise the bounding box so every diagram has a comparable
    # visual extent.  Skipped whenever the caller pinned any node —
    # rescaling would move pins away from their intended coordinates.
    # ------------------------------------------------------------------
    if params.normalize_bbox and pos and not pinned:
        _normalize_bbox(pos, params.target_extent)

    return pos


def _normalize_bbox(
    pos: dict[str, np.ndarray],
    target_extent: tuple[float, float],
) -> None:
    """Re-centre and uniformly scale ``pos`` in-place so its bounding
    box fits within ``target_extent`` without distortion.

    Aspect ratio is preserved: the scale factor is the smaller of
    ``target_w / current_w`` and ``target_h / current_h``, so a
    short-and-wide diagram gets stretched horizontally to the target
    width while a tall-and-narrow one is bounded by the target height.
    """
    pts = np.array(list(pos.values()), dtype=float)
    centroid = pts.mean(axis=0)
    pts_centred = pts - centroid

    half_w = float(np.abs(pts_centred[:, 0]).max())
    half_h = float(np.abs(pts_centred[:, 1]).max())
    target_half_w = target_extent[0] / 2.0
    target_half_h = target_extent[1] / 2.0

    if half_w < 1e-8 and half_h < 1e-8:
        # All nodes coincident — nothing meaningful to scale.
        return
    if half_w < 1e-8:
        scale = target_half_h / half_h
    elif half_h < 1e-8:
        scale = target_half_w / half_w
    else:
        scale = min(target_half_w / half_w, target_half_h / half_h)

    for k in pos:
        pos[k] = (pos[k] - centroid) * scale


def _enforce_min_distance(
    pos: dict[str, np.ndarray],
    verts: list[str],
    pinned: set[str],
    min_dist: float,
    max_iter: int = 10,
) -> None:
    """In-place: push every pair of free vertices at least ``min_dist`` apart."""
    if min_dist <= 0:
        return
    for _ in range(max_iter):
        changed = False
        for iv, v1 in enumerate(verts):
            for v2 in verts[iv + 1:]:
                d = float(np.linalg.norm(pos[v1] - pos[v2]))
                if d == 0:
                    if v2 not in pinned:
                        pos[v2] = pos[v2] + np.array([min_dist, 0.0])
                        changed = True
                    elif v1 not in pinned:
                        pos[v1] = pos[v1] + np.array([-min_dist, 0.0])
                        changed = True
                    continue
                if d >= min_dist:
                    continue
                direction = (pos[v2] - pos[v1]) / d
                push = (min_dist - d) / 2 + 0.05
                if v1 not in pinned:
                    pos[v1] = pos[v1] - push * direction
                    changed = True
                if v2 not in pinned:
                    pos[v2] = pos[v2] + push * direction
                    changed = True
        if not changed:
            break


def neighbor_center(
    g: nx.MultiGraph,
    node: str,
    pos: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Centroid of a node's non-self neighbours, used for self-loop
    direction.

    If the node has no other neighbours, returns a point one unit
    below it so self-loops on isolated tadpoles point straight up.
    """
    neighbors = [n for n in g.neighbors(node) if n != node]
    if not neighbors:
        return pos[node] + np.array([0.0, -1.0])
    return np.mean(np.array([pos[n] for n in neighbors]), axis=0)


def label_offset(
    point: np.ndarray,
    center: np.ndarray,
    distance: float = 18.0,
) -> tuple[float, float]:
    """Return a label-offset vector (in points) pointing away from
    ``center``.

    Used by both backends to place labels on the outside of the
    bounding box where they are least likely to overlap edges.
    """
    direction = point - center
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return (0.0, distance)
    direction = direction / norm
    return (float(direction[0] * distance), float(direction[1] * distance))
