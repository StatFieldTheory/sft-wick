"""TikZ/PGF rendering for :class:`FeynmanDiagram` objects.

This backend produces LaTeX source — a ``tikzpicture`` environment —
that you can drop straight into a paper with ``\\input{fig.tex}``.
It uses no external Python dependencies (pure string templates) and
shares the layout, label, and style abstractions with the
matplotlib backend in :mod:`sft_wick.drawing`.

Required LaTeX packages on the user's side
==========================================

The generated code uses standard TikZ features:

.. code-block:: latex

    \\usepackage{tikz}
    \\usetikzlibrary{arrows.meta}

For a self-contained PDF (``standalone=True``) the document also
needs the ``standalone`` document class, which is part of any TeX
Live / MiKTeX install.

Quick start
===========

.. code-block:: python

    from sft_wick import TikzRenderer, publication_style

    renderer = TikzRenderer(style=publication_style())
    tikz_code = renderer.to_string(fd)
    renderer.save(fd, "fig/diag1.tex")
    renderer.save(
        fd, "fig/diag1_standalone.tex",
        standalone=True,         # produce a complete document
    )

External-vertex labels follow the same override hierarchy as the
matplotlib backend (per-call dict > callable > default formatter);
see :mod:`sft_wick.render_labels`.

Limitations (v1)
================

- ``rcparams`` is ignored — TikZ output uses LaTeX-side typography.
- ``bbox`` on labels is ignored — TikZ labels render without a box
  by default; users can post-process by editing the generated
  ``every label/.style``.
- No ``pdflatex`` invocation: the user runs LaTeX themselves.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .diagrams import FeynmanDiagram
from .render_labels import (
    LabelCallable,
    default_external_label,
    default_vertex_label,
    resolve_label,
)
from .render_layout import compute_layout, neighbor_center
from .render_style import (
    NodeStyle,
    PropagatorStyle,
    RenderStyle,
    default_style,
)

# ----------------------------------------------------------------------
# Style → TikZ option mappers
# ----------------------------------------------------------------------

_TIKZ_LINESTYLE: dict[str, str] = {
    "solid": "solid",
    "dashed": "dashed",
    "dotted": "dotted",
    "dashdot": "dashdotted",
}


def _tikz_color(color: str) -> str:
    """Convert a Python-side colour spec to a TikZ ``\\definecolor``-friendly form.

    Hex strings ``"#rrggbb"`` are converted to the
    ``{rgb,255:red,r;green,g;blue,b}`` form which TikZ understands
    inline; named colours are passed through verbatim (TikZ's xcolor
    package recognises ``black``, ``red``, ``blue``, etc.).
    """
    if not color:
        return "black"
    if color.startswith("#") and len(color) == 7:
        try:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
        except ValueError:
            return "black"
        return f"{{rgb,255:red,{r};green,{g};blue,{b}}}"
    return color


def _prop_style_options(prop: PropagatorStyle) -> str:
    """Return the comma-separated TikZ options for a propagator style."""
    parts = [
        f"draw={_tikz_color(prop.color)}",
        _TIKZ_LINESTYLE.get(prop.linestyle, prop.linestyle),
        f"line width={prop.linewidth}pt",
    ]
    if prop.arrow:
        parts.append("-{Latex[length=2mm]}")
    return ", ".join(parts)


def _node_shape(shape: str) -> str:
    return {
        "circle": "circle",
        "square": "rectangle",
        "diamond": "diamond",
        "triangle": "regular polygon, regular polygon sides=3",
    }.get(shape, "circle")


def _node_style_options(node: NodeStyle) -> str:
    """TikZ option string for a node-style (used in ``\\tikzset``)."""
    parts = [
        _node_shape(node.shape),
        "inner sep=0pt",
        f"minimum size={node.size}pt",
    ]
    if node.fill:
        parts.append(f"fill={_tikz_color(node.color)}")
    if node.edge_color != "none":
        parts.append(f"draw={_tikz_color(node.edge_color)}")
    elif not node.fill:
        parts.append(f"draw={_tikz_color(node.color)}")
    return ", ".join(parts)


# ----------------------------------------------------------------------
# Renderer
# ----------------------------------------------------------------------

class TikzRenderer:
    """Render a :class:`FeynmanDiagram` as a tikzpicture LaTeX string.

    Args:
        style: A :class:`~sft_wick.render_style.RenderStyle`.
            Defaults to :func:`~sft_wick.render_style.default_style`.
        external_label_fn: Optional callable
            ``fn(node_id, node_attrs) -> str | None`` for systematic
            external-label overrides.
        vertex_label_fn:   Same shape, for interaction vertices.
        label_format: Default external-label format flag
            (overrides ``style.label_format`` when supplied).
        scale: TikZ ``scale=…`` factor — multiplies every position.
        standalone: When ``True``, :meth:`to_string` and :meth:`save`
            wrap the figure in a complete ``\\documentclass{standalone}``
            document instead of a bare ``tikzpicture`` environment.
    """

    def __init__(
        self,
        style: RenderStyle | None = None,
        external_label_fn: LabelCallable | None = None,
        vertex_label_fn: LabelCallable | None = None,
        label_format: str | None = None,
        scale: float = 1.0,
        standalone: bool = False,
    ) -> None:
        self.style: RenderStyle = style if style is not None else default_style()
        self.external_label_fn = external_label_fn
        self.vertex_label_fn = vertex_label_fn
        self._label_format_override = label_format
        self.scale = scale
        self.standalone = standalone

    @property
    def label_format(self) -> str:
        return (
            self._label_format_override
            if self._label_format_override is not None
            else self.style.label_format
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_string(
        self,
        diagram: FeynmanDiagram,
        external_labels: Mapping[str, str] | None = None,
        vertex_labels: Mapping[str, str] | None = None,
        positions: Mapping[str, tuple[float, float]] | None = None,
        standalone: bool | None = None,
    ) -> str:
        """Build the TikZ source for a single diagram.

        Args:
            diagram:         The diagram to render.
            external_labels: Optional ``{node_id: label}`` overrides.
            vertex_labels:   Same, for interaction vertices.
            positions:       Optional ``{node_id: (x, y)}`` pin map.
            standalone:      If given, override the constructor's
                             ``standalone`` flag for this call only.

        Returns:
            A LaTeX source string.  Always ends with a trailing
            newline.
        """
        body = self._tikzpicture(
            diagram, external_labels, vertex_labels, positions,
        )
        wrap_standalone = (
            self.standalone if standalone is None else standalone
        )
        if wrap_standalone:
            return _wrap_standalone(body)
        return body

    def save(
        self,
        diagram: FeynmanDiagram,
        path: str | Path,
        external_labels: Mapping[str, str] | None = None,
        vertex_labels: Mapping[str, str] | None = None,
        positions: Mapping[str, tuple[float, float]] | None = None,
        standalone: bool | None = None,
    ) -> Path:
        """Write the TikZ source to disk and return the path.

        Creates parent directories as needed.
        """
        text = self.to_string(
            diagram,
            external_labels=external_labels,
            vertex_labels=vertex_labels,
            positions=positions,
            standalone=standalone,
        )
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        return out

    def save_all(
        self,
        diagrams: Sequence[FeynmanDiagram],
        path_pattern: str | Path,
        external_labels: Mapping[int, Mapping[str, str]] | None = None,
        vertex_labels: Mapping[int, Mapping[str, str]] | None = None,
        positions: Mapping[int, Mapping[str, tuple[float, float]]] | None = None,
        standalone: bool | None = None,
    ) -> list[Path]:
        """Save many diagrams using a ``str.format``-style pattern.

        Example: ``path_pattern="fig/order2_diag_{i:02d}.tex"`` writes
        ``fig/order2_diag_00.tex``, ``fig/order2_diag_01.tex``, …

        Per-diagram overrides are keyed by *subplot index*.
        """
        pattern = str(path_pattern)
        written: list[Path] = []
        for i, fd in enumerate(diagrams):
            ext = external_labels.get(i) if external_labels else None
            vrt = vertex_labels.get(i) if vertex_labels else None
            pos = positions.get(i) if positions else None
            target = Path(pattern.format(i=i))
            written.append(self.save(
                fd, target,
                external_labels=ext,
                vertex_labels=vrt,
                positions=pos,
                standalone=standalone,
            ))
        return written

    # ------------------------------------------------------------------
    # Body generation
    # ------------------------------------------------------------------

    def _tikzpicture(
        self,
        diagram: FeynmanDiagram,
        external_labels: Mapping[str, str] | None,
        vertex_labels: Mapping[str, str] | None,
        positions: Mapping[str, tuple[float, float]] | None,
    ) -> str:
        style = self.style
        g = diagram.graph
        if g.number_of_nodes() == 0:
            return self._empty_picture()

        pos = compute_layout(diagram, style.layout, positions)

        lines: list[str] = []
        lines.append(self._begin_picture(style))

        # Node-style definitions (so users can post-edit easily)
        lines.append(
            f"  \\tikzset{{ext/.style={{{_node_style_options(style.external_node)}}}}}"
        )
        lines.append(
            f"  \\tikzset{{vert/.style={{{_node_style_options(style.vertex_node)}}}}}"
        )

        # Propagator-kind styles
        for kind, prop in style.propagators.items():
            opts = _prop_style_options(prop)
            lines.append(f"  \\tikzset{{{kind}/.style={{{opts}}}}}")

        # External nodes
        for n in diagram.external_nodes:
            x, y = pos[n]
            label = resolve_label(
                node_id=n,
                node_attrs=g.nodes[n],
                overrides=external_labels,
                callable_fn=self.external_label_fn,
                default_fn=lambda attrs: default_external_label(
                    attrs, fmt=self.label_format,
                ),
            )
            label_pos = _label_anchor(pos[n], _diagram_center(pos))
            tikz_label = (
                f"label={{[label distance={style.external_label.offset_pt / 6:.1f}pt,"
                f"font=\\fontsize{{{style.external_label.fontsize}}}"
                f"{{{style.external_label.fontsize * 1.2}}}\\selectfont]"
                f"{label_pos}:{label}}}"
            ) if label else ""
            opts = "ext"
            if tikz_label:
                opts += f", {tikz_label}"
            lines.append(
                f"  \\node[{opts}] ({_safe_node_id(n)}) at ({x:.4f}, {y:.4f}) {{}};"
            )

        # Vertex nodes
        for n in diagram.vertex_nodes:
            x, y = pos[n]
            label = resolve_label(
                node_id=n,
                node_attrs=g.nodes[n],
                overrides=vertex_labels,
                callable_fn=self.vertex_label_fn,
                default_fn=default_vertex_label,
            )
            label_pos = _label_anchor(pos[n], _diagram_center(pos))
            opts = "vert"
            if label:
                tikz_label = (
                    f"label={{[label distance={style.vertex_label.offset_pt / 6:.1f}pt,"
                    f"font=\\fontsize{{{style.vertex_label.fontsize}}}"
                    f"{{{style.vertex_label.fontsize * 1.2}}}\\selectfont]"
                    f"{label_pos}:{label}}}"
                )
                opts += f", {tikz_label}"
            lines.append(
                f"  \\node[{opts}] ({_safe_node_id(n)}) at ({x:.4f}, {y:.4f}) {{}};"
            )

        # Edges
        lines.append("  % propagators")
        for u, v, _key, data in g.edges(keys=True, data=True):
            kind = data.get("kind", "C")
            line = self._edge_line(u, v, kind, data, g, pos)
            lines.append(line)

        lines.append("\\end{tikzpicture}")
        return "\n".join(lines) + "\n"

    def _begin_picture(self, style: RenderStyle) -> str:
        return (
            f"\\begin{{tikzpicture}}[scale={self.scale}, "
            f"every node/.style={{inner sep=0pt}}]"
        )

    def _edge_line(
        self,
        u: str,
        v: str,
        kind: str,
        data: Mapping[str, Any],
        g: Any,
        pos: Any,
    ) -> str:
        prop = self.style.propagators.get(kind)
        u_id = _safe_node_id(u)
        v_id = _safe_node_id(v)

        if u == v:
            # Self-loop direction informed by neighbour centroid.
            ncenter = neighbor_center(g, u, pos)
            direction = pos[u] - ncenter
            angle = _atan2_deg(direction)
            anchor = _angle_to_loop_anchor(angle)
            return (
                f"  \\draw[{kind}] ({u_id}) "
                f"edge[loop {anchor}, looseness=10] ({u_id});"
            )

        # Reverse for arrow direction (R: arrow points psi → phi).
        if prop is not None and prop.arrow:
            phi_end = data.get("phi_end")
            if phi_end is not None and phi_end == u:
                # arrow goes psi (=v) -> phi (=u)
                return f"  \\draw[{kind}] ({v_id}) -- ({u_id});"
            return f"  \\draw[{kind}] ({u_id}) -- ({v_id});"

        return f"  \\draw[{kind}] ({u_id}) -- ({v_id});"

    def _empty_picture(self) -> str:
        return (
            "\\begin{tikzpicture}\n"
            "  \\node {Empty diagram};\n"
            "\\end{tikzpicture}\n"
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _safe_node_id(node_id: str) -> str:
    """TikZ node names cannot contain underscores cleanly in some
    documents; replace with a hyphen-free form to be safe.
    """
    return node_id.replace("_", "")


def _diagram_center(pos: Mapping[str, Any]) -> Any:
    import numpy as np
    arr = np.array(list(pos.values()), dtype=float)
    return arr.mean(axis=0)


def _label_anchor(point: Any, center: Any) -> str:
    """Pick a TikZ label anchor (``above``/``below``/``left``/``right``)
    based on the direction from the diagram centre to the node."""
    import numpy as np
    direction = np.asarray(point, dtype=float) - np.asarray(center, dtype=float)
    if np.linalg.norm(direction) < 1e-8:
        return "above"
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    return _angle_to_anchor(angle)


def _angle_to_anchor(angle_deg: float) -> str:
    a = ((angle_deg + 360.0) % 360.0)
    if a < 22.5 or a >= 337.5:
        return "right"
    if a < 67.5:
        return "above right"
    if a < 112.5:
        return "above"
    if a < 157.5:
        return "above left"
    if a < 202.5:
        return "left"
    if a < 247.5:
        return "below left"
    if a < 292.5:
        return "below"
    return "below right"


def _angle_to_loop_anchor(angle_deg: float) -> str:
    a = ((angle_deg + 360.0) % 360.0)
    if 45.0 <= a < 135.0:
        return "above"
    if 135.0 <= a < 225.0:
        return "left"
    if 225.0 <= a < 315.0:
        return "below"
    return "right"


def _atan2_deg(direction: Any) -> float:
    import numpy as np
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


_STANDALONE_TEMPLATE = r"""\documentclass[border=4pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\begin{document}
%s
\end{document}
"""


def _wrap_standalone(body: str) -> str:
    return _STANDALONE_TEMPLATE % body.rstrip()
