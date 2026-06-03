"""Backend-agnostic visual styling for Feynman-diagram rendering.

This module defines a hierarchy of frozen dataclasses that capture
*what* a diagram should look like (colors, line styles, marker
sizes, fonts, layout parameters) without committing to *how* it
gets drawn.  The matplotlib renderer in
:mod:`sft_wick.drawing` and the TikZ/PGF renderer in
:mod:`sft_wick.drawing_tikz` both consume the same
:class:`RenderStyle` and translate it to backend-native settings.

Three named presets are provided:

* :func:`default_style`     — the colourful on-screen look (blue C,
  red R) used for quick inspection in notebooks.
* :func:`publication_style` — a cleaner, smaller-marker look with
  publication-friendly fonts.  Honours ``usetex`` if the user has a
  LaTeX install.
* :func:`grayscale_style`   — black/grey palette for printed papers
  and B&W reproduction.
* :func:`minimal_style`     — strips legend / labels / boxes for
  inset-style use inside a larger figure.

All four call into the same primitives, so users may freely mix
parts (e.g. ``publication_style().with_overrides(show_legend=False)``).

Three label-format flags control the default text of external
vertices (the ``$\\phi_a(\\cdot)$`` operators):

* :data:`LABEL_COMPACT` — ``$\\phi_a$`` (no spatial argument).  Default.
* :data:`LABEL_FULL`    — ``$\\phi_a(x_1)$`` (spatial argument
  retained — the pre-2026-04 default).
* :data:`LABEL_TIME_F`  — ``$\\phi_a(t_f)$`` (substitute ``t_f`` for
  the spatial argument).

Override these on a per-diagram or per-renderer basis with the
``label_format=…`` constructor argument or the ``external_labels``
keyword on :meth:`DiagramRenderer.draw`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any


# ----------------------------------------------------------------------
# Label format flags
# ----------------------------------------------------------------------

LABEL_COMPACT: str = "compact"
LABEL_FULL: str = "full"
LABEL_TIME_F: str = "time_f"

LABEL_FORMATS: tuple[str, ...] = (LABEL_COMPACT, LABEL_FULL, LABEL_TIME_F)


# ----------------------------------------------------------------------
# Sub-dataclasses
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class PropagatorStyle:
    """Visual style for a single propagator kind (``C`` or ``R``).

    Attributes:
        color:        Stroke colour as a hex string (``"#1f3a5f"``).
        linestyle:    One of ``"solid" | "dashed" | "dotted"``.
        linewidth:    Line thickness (matplotlib points / TikZ ``line width``).
        arrow:        Whether to draw a directed arrowhead (used for R).
        legend_label: Text shown in the legend; empty string → no entry.
    """
    color: str
    linestyle: str
    linewidth: float
    arrow: bool = False
    legend_label: str = ""


@dataclass(frozen=True)
class NodeStyle:
    """Visual style for an external point or interaction vertex.

    Attributes:
        shape:      ``"circle"`` or ``"square"``.
        size:       Marker size (matplotlib ``markersize`` /
                    TikZ ``minimum size``).
        fill:       Whether the marker is filled.
        color:      Fill colour (or stroke if ``fill=False``).
        edge_color: Outline colour; ``"none"`` for no outline.
    """
    shape: str
    size: float
    fill: bool = True
    color: str = "black"
    edge_color: str = "none"


@dataclass(frozen=True)
class LabelStyle:
    """Visual style for a label rendered next to a node.

    Attributes:
        fontsize:   Label font size in points.
        bold:       If ``True``, the label is bold.
        bbox:       If ``True``, draw a white rounded box behind the label
                    (matplotlib only; TikZ ignores this and lets the user
                    style via ``every label/.style`` if desired).
        bbox_alpha: Opacity of that box.
        offset_pt:  Distance from the node to the label centre, in
                    typographic points.
    """
    fontsize: float
    bold: bool = False
    bbox: bool = True
    bbox_alpha: float = 0.85
    offset_pt: float = 26.0


@dataclass(frozen=True)
class LayoutParams:
    """Numerical parameters controlling the position-computation step.

    These are honoured identically by both backends.

    Attributes:
        ext_radius:        Radius of the circle on which external
                           nodes are placed.
        spring_k:          Optimal edge length for ``spring_layout``.
        spring_iterations: Iteration cap for ``spring_layout``.
        spring_seed:       Seed for ``spring_layout`` (None → non-deterministic).
        min_vertex_dist:   Minimum allowed Euclidean distance between
                           interaction vertices; vertices closer than
                           this are pushed apart in a post-pass.
        margin:            Padding added to axis limits around the
                           diagram bounding box (matplotlib only).
        component_gap:     Minimum horizontal gap between disconnected
                           connected components before bbox
                           normalisation.  Helps factorised diagrams
                           remain visually separated.
        normalize_bbox:    If ``True`` (default), the final layout is
                           re-centred on the origin and uniformly
                           scaled so it fits within
                           :attr:`target_extent` without distortion.
                           Gives every diagram in a grid roughly the
                           same visual extent — recommended whenever
                           drawing multiple diagrams side by side.
        target_extent:     Target ``(width, height)`` for the
                           normalised bounding box, in matplotlib
                           units.  Default ``(6.0, 4.0)`` matches a
                           4:3 ish aspect-ratio panel.
        parallel_edge_curvature:
                           Bow strength for parallel edges between the
                           same node pair (matplotlib backend).  Each
                           edge in a bundle is drawn as an ``arc3`` with
                           ``rad = (key - (n-1)/2) * parallel_edge_curvature``,
                           so a 2-edge bundle bows by ``±0.5 ×`` this
                           value to opposite sides.  Larger → fuller
                           arcs.  Default ``0.6`` (apex offset ≈ 30 % of
                           the chord for a 2-edge bundle).
    """
    ext_radius: float = 2.5
    spring_k: float = 2.0
    spring_iterations: int = 200
    spring_seed: int | None = 42
    min_vertex_dist: float = 0.8
    margin: float = 0.75
    component_gap: float = 1.0
    normalize_bbox: bool = True
    target_extent: tuple[float, float] = (5.6, 3.6)
    parallel_edge_curvature: float = 0.6


# ----------------------------------------------------------------------
# Top-level style
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class RenderStyle:
    """Complete visual specification for a Feynman-diagram render.

    Both the matplotlib (:class:`sft_wick.drawing.DiagramRenderer`)
    and the TikZ (:class:`sft_wick.drawing_tikz.TikzRenderer`)
    backends consume an instance of this class and translate it to
    their native style options.

    Use :func:`default_style`, :func:`publication_style`,
    :func:`grayscale_style`, or :func:`minimal_style` as a starting
    point and call :meth:`with_overrides` to customise.

    Attributes:
        propagators:        Mapping ``{"C": PropagatorStyle,
                            "R": PropagatorStyle}``.
        external_node:      Style for external observable nodes.
        vertex_node:        Style for interaction vertices.
        external_label:     Style for labels next to external nodes.
        vertex_label:       Style for labels next to interaction
                            vertices (typically the coupling name).
        title_fontsize:     Font size of an axes title.
        suptitle_fontsize:  Font size of the figure-level suptitle.
        legend_fontsize:    Font size of the propagator-kind legend.
        legend_loc:         Matplotlib legend location string.
        show_legend:        Master toggle for the legend.
        layout:             Numerical layout parameters.
        rcparams:           Optional mapping fed into a
                            ``matplotlib.rc_context`` around the draw
                            call.  Useful for serif fonts, math
                            rendering, etc.  Ignored by TikZ.
        mathtext:           If ``True``, labels are wrapped in ``$…$``
                            so matplotlib renders them as math.
        usetex:             Convenience flag.  When ``True``, sets
                            ``"text.usetex": True`` in ``rcparams``
                            (overriding any prior value) — requires a
                            working LaTeX install on the user's system.
        label_format:       Default external-label format
                            (``LABEL_COMPACT`` etc.).  May be
                            overridden per-call.
    """
    propagators: Mapping[str, PropagatorStyle]
    external_node: NodeStyle
    vertex_node: NodeStyle
    external_label: LabelStyle
    vertex_label: LabelStyle
    title_fontsize: float = 12.0
    suptitle_fontsize: float = 14.0
    legend_fontsize: float = 12.0
    legend_loc: str = "lower right"
    show_legend: bool = True
    layout: LayoutParams = field(default_factory=LayoutParams)
    rcparams: Mapping[str, Any] | None = None
    mathtext: bool = True
    usetex: bool = False
    label_format: str = LABEL_COMPACT

    def __post_init__(self) -> None:
        if self.label_format not in LABEL_FORMATS:
            raise ValueError(
                f"label_format must be one of {LABEL_FORMATS}, got "
                f"{self.label_format!r}."
            )
        # Freeze the propagator mapping so callers can rely on
        # immutability of the whole object.
        if not isinstance(self.propagators, MappingProxyType):
            object.__setattr__(
                self, "propagators",
                MappingProxyType(dict(self.propagators)),
            )
        if self.rcparams is not None and not isinstance(
                self.rcparams, MappingProxyType):
            object.__setattr__(
                self, "rcparams",
                MappingProxyType(dict(self.rcparams)),
            )

    # ------------------------------------------------------------------
    # Convenience: produce a modified copy
    # ------------------------------------------------------------------

    def with_overrides(self, **kwargs: Any) -> "RenderStyle":
        """Return a new ``RenderStyle`` with the given top-level fields
        replaced.  Nested fields (``propagators["C"].color``) need
        :meth:`with_propagator` or manual construction.
        """
        return replace(self, **kwargs)

    def with_propagator(self, kind: str, **prop_kwargs: Any) -> "RenderStyle":
        """Return a new ``RenderStyle`` with one propagator's style
        partially replaced.

        Example:

        .. code-block:: python

            style = publication_style().with_propagator("C", color="black")
        """
        if kind not in self.propagators:
            raise KeyError(
                f"No propagator kind {kind!r} in style "
                f"(known: {sorted(self.propagators)})."
            )
        new_prop = replace(self.propagators[kind], **prop_kwargs)
        new_props = dict(self.propagators)
        new_props[kind] = new_prop
        return replace(self, propagators=new_props)

    def effective_rcparams(self) -> dict[str, Any]:
        """Resolve ``rcparams`` + ``usetex`` into a single dict.

        Used by the matplotlib backend to feed ``rc_context``.
        """
        rc: dict[str, Any] = dict(self.rcparams) if self.rcparams else {}
        if self.usetex:
            rc["text.usetex"] = True
        return rc


# ----------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------

def _default_propagators() -> dict[str, PropagatorStyle]:
    return {
        "C": PropagatorStyle(
            color="#2166ac", linestyle="solid", linewidth=2.0,
            arrow=False, legend_label="C (correlation)",
        ),
        "R": PropagatorStyle(
            color="#d6604d", linestyle="dashed", linewidth=2.0,
            arrow=True, legend_label="R (response)",
        ),
    }


def default_style() -> RenderStyle:
    """The colourful on-screen preset (blue C, red R, square vertices).

    Matches the pre-2026-04 visual defaults — use this as the
    backward-compatible baseline.
    """
    return RenderStyle(
        propagators=_default_propagators(),
        external_node=NodeStyle(shape="circle", size=8.0),
        vertex_node=NodeStyle(shape="square", size=10.0),
        external_label=LabelStyle(fontsize=12.0, bold=False, offset_pt=24.0),
        vertex_label=LabelStyle(fontsize=12.0, bold=True, offset_pt=18.0),
        title_fontsize=12.0,
        suptitle_fontsize=14.0,
        legend_fontsize=12.0,
        show_legend=True,
        layout=LayoutParams(),
        label_format=LABEL_COMPACT,
    )


def publication_style(*, usetex: bool = False) -> RenderStyle:
    """Cleaner preset for paper-quality figures.

    Smaller, thinner markers, refined palette (slate-blue C, warm-grey
    R), serif labels via mathtext or LaTeX, compact (no boxed) labels,
    and a discreet legend.  Pass ``usetex=True`` if you have a working
    LaTeX install — otherwise mathtext renders the math.
    """
    rc = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "axes.titleweight": "regular",
    }
    return RenderStyle(
        propagators={
            "C": PropagatorStyle(
                color="#1f3a5f", linestyle="solid", linewidth=1.2,
                arrow=False, legend_label=r"$C$",
            ),
            "R": PropagatorStyle(
                color="#666666", linestyle="dashed", linewidth=1.2,
                arrow=True, legend_label=r"$R$",
            ),
        },
        external_node=NodeStyle(shape="circle", size=6.0),
        vertex_node=NodeStyle(shape="square", size=7.0),
        external_label=LabelStyle(
            fontsize=12.0, bold=False, bbox=False, offset_pt=22.0,
        ),
        vertex_label=LabelStyle(
            fontsize=12.0, bold=False, bbox=False, offset_pt=18.0,
        ),
        title_fontsize=12.0,
        suptitle_fontsize=13.0,
        legend_fontsize=11.0,
        legend_loc="lower right",
        show_legend=True,
        layout=LayoutParams(),
        rcparams=rc,
        usetex=usetex,
        label_format=LABEL_COMPACT,
    )


def grayscale_style() -> RenderStyle:
    """Black-and-white preset.

    Distinguishes propagator kinds by line style and arrow-head only;
    no colour information.  Suitable for printed papers and B&W
    reproduction.
    """
    return RenderStyle(
        propagators={
            "C": PropagatorStyle(
                color="black", linestyle="solid", linewidth=1.4,
                arrow=False, legend_label=r"$C$",
            ),
            "R": PropagatorStyle(
                color="black", linestyle="dashed", linewidth=1.4,
                arrow=True, legend_label=r"$R$",
            ),
        },
        external_node=NodeStyle(shape="circle", size=6.5),
        vertex_node=NodeStyle(shape="square", size=7.5),
        external_label=LabelStyle(
            fontsize=11.0, bold=False, bbox=False, offset_pt=22.0,
        ),
        vertex_label=LabelStyle(
            fontsize=11.0, bold=False, bbox=False, offset_pt=18.0,
        ),
        title_fontsize=11.0,
        suptitle_fontsize=13.0,
        legend_fontsize=10.5,
        show_legend=True,
        layout=LayoutParams(),
        label_format=LABEL_COMPACT,
    )


def minimal_style() -> RenderStyle:
    """Stripped-down preset for inset use.

    No legend, no boxes behind labels, no per-axes title.  Good when
    embedding diagrams as small panels inside a larger composite figure.
    """
    style = publication_style()
    return style.with_overrides(
        show_legend=False,
        external_label=replace(style.external_label, bbox=False, fontsize=9.0),
        vertex_label=replace(style.vertex_label, bbox=False, fontsize=8.5),
        title_fontsize=0.0,  # backend may treat 0 as "skip"
    )
