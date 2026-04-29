"""Regenerate the example diagram images bundled with the user guide.

Run from anywhere::

    python docs/_static/diagrams/_generate.py

Outputs all images and the standalone TikZ source file alongside this
script.  Re-run after changes to the renderer that affect default
appearance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sft_wick import (
    Action,
    DiagramRenderer,
    Field,
    TikzRenderer,
    Vertex,
    compute_moment,
    default_style,
    grayscale_style,
    minimal_style,
    publication_style,
    reset_uid_counter,
)

OUT = Path(__file__).resolve().parent


# ----------------------------------------------------------------------
# Build a representative diagram (order-1 tadpole)
# ----------------------------------------------------------------------

def _build_order1_action_and_obs():
    """Quartic ``F = [phi, phi, psi, psi]`` — gives a non-trivial
    order-1 tadpole, used for the single-diagram demos
    (presets / custom labels / manual positions).
    """
    reset_uid_counter()
    phi = Field("phi", "physical", n_components=2)
    psi = Field("psi", "response", n_components=2)
    F = Vertex(fields=[phi, phi, psi, psi], coupling="F")
    action = Action(vertices=[F])
    obs = [phi("a", "x_1"), phi("b", "x_2")]
    return action, obs


def _build_order2_action_and_obs():
    """Cubic ``F = [phi, phi, psi]`` — yields a richer set of
    distinct order-2 topologies, all using the same vertex type
    (no F/G mix in the showcase grid).
    """
    reset_uid_counter()
    phi = Field("phi", "physical", n_components=2)
    psi = Field("psi", "response", n_components=2)
    F = Vertex(fields=[phi, phi, psi], coupling="F")
    action = Action(vertices=[F])
    obs = [phi("a", "x_1"), phi("b", "x_2")]
    return action, obs


def build_diagram():
    """Single representative diagram (order-1 tadpole)."""
    action, obs = _build_order1_action_and_obs()
    result = compute_moment(obs, action, order=1)
    return result.diagrams_by_order[1][-1].to_feynman_diagram()


def build_order2_diagrams():
    """All Feynman diagrams of the 2-point correlator at order 2.

    Returns
    -------
    diagrams : list[FeynmanDiagram]
    multiplicities : list[int]
        Equivalence-class size for each diagram (used as ``[xN]``
        annotations in the grid title).
    """
    action, obs = _build_order2_action_and_obs()
    result = compute_moment(obs, action, order=2)
    records = result.diagrams_by_order.get(2, [])
    diagrams = [r.to_feynman_diagram() for r in records]
    # Collapse topologically identical entries to one slot per
    # equivalence class for a cleaner figure.
    seen: dict[tuple, tuple[int, int]] = {}
    unique: list = []
    multiplicities: list[int] = []
    for fd in diagrams:
        key = fd.canonical_form()
        if key in seen:
            slot, _ = seen[key]
            multiplicities[slot] += 1
        else:
            seen[key] = (len(unique), 1)
            unique.append(fd)
            multiplicities.append(1)
    return unique, multiplicities


# ----------------------------------------------------------------------
# Figure 1: 2x2 preset comparison
# ----------------------------------------------------------------------

def render_presets(fd) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    presets = [
        ("default_style()", default_style()),
        ("publication_style()", publication_style()),
        ("grayscale_style()", grayscale_style()),
        ("minimal_style()", minimal_style()),
    ]
    for ax, (label, style) in zip(axes.flat, presets):
        DiagramRenderer(figsize=(5, 4), style=style).draw(
            fd, ax=ax, title=label,
        )

    fig.tight_layout()
    out = OUT / "presets.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Figure 2: custom external labels
# ----------------------------------------------------------------------

def render_custom_labels(fd) -> Path:
    e0, e1 = fd.external_nodes
    style = publication_style()
    # Push labels further out so they don't overlap the markers.
    style = style.with_overrides(
        external_label=type(style.external_label)(
            fontsize=12.0, bold=False, bbox=False, offset_pt=34.0,
        ),
    )
    renderer = DiagramRenderer(figsize=(7, 4), style=style)
    ax = renderer.draw(
        fd,
        title="external_labels override",
        external_labels={
            e0: r"$\varphi_a(t_f, \mathbf{x}_a)$",
            e1: r"$\varphi_b(t_f, \mathbf{x}_b)$",
        },
    )
    out = OUT / "custom_labels.png"
    ax.figure.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(ax.figure)
    return out


# ----------------------------------------------------------------------
# Figure 3: pinned vertex (manual positions)
# ----------------------------------------------------------------------

def render_manual_positions(fd) -> Path:
    v = fd.vertex_nodes[0]
    renderer = DiagramRenderer(
        figsize=(6, 4),
        style=publication_style(),
    )
    ax = renderer.draw(
        fd,
        title="positions={vert: (0, 1.5)}",
        positions={v: (0.0, 1.5)},
    )
    out = OUT / "manual_positions.png"
    ax.figure.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(ax.figure)
    return out


# ----------------------------------------------------------------------
# Figure 4: TikZ standalone source
# ----------------------------------------------------------------------

def render_tikz(fd) -> Path:
    out = OUT / "example.tex"
    TikzRenderer(style=publication_style(), standalone=True).save(fd, out)
    return out


# ----------------------------------------------------------------------
# Figures 5 & 6: full order-2 expansion of the 2-point function
# ----------------------------------------------------------------------

def render_order2_grid(diagrams, multiplicities) -> tuple[Path, Path]:
    """Two side-by-side renderings (publication + grayscale) of the
    order-2 expansion.

    Showcases ``draw_all`` with multiplicities and a tighter grid.
    """
    pub_renderer = DiagramRenderer(figsize=(4.0, 3.0), style=publication_style())
    fig_pub = pub_renderer.draw_all(
        diagrams, ncols=3,
        suptitle=r"$\langle \phi_a \phi_b \rangle$  -  order 2 (publication_style)",
        multiplicities=multiplicities,
    )
    out_pub = OUT / "order2_publication.png"
    fig_pub.savefig(out_pub, dpi=140, bbox_inches="tight")
    plt.close(fig_pub)

    gray_renderer = DiagramRenderer(figsize=(4.0, 3.0), style=grayscale_style())
    fig_gray = gray_renderer.draw_all(
        diagrams, ncols=3,
        suptitle=r"$\langle \phi_a \phi_b \rangle$  -  order 2 (grayscale_style)",
        multiplicities=multiplicities,
    )
    out_gray = OUT / "order2_grayscale.png"
    fig_gray.savefig(out_gray, dpi=140, bbox_inches="tight")
    plt.close(fig_gray)

    return out_pub, out_gray


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main() -> int:
    fd = build_diagram()
    diagrams_o2, mults_o2 = build_order2_diagrams()
    print(f"order-2 expansion: {len(diagrams_o2)} unique topologies "
          f"(multiplicities: {mults_o2})")
    pub2, gray2 = render_order2_grid(diagrams_o2, mults_o2)
    paths = [
        render_presets(fd),
        render_custom_labels(fd),
        render_manual_positions(fd),
        render_tikz(fd),
        pub2,
        gray2,
    ]
    print("wrote:")
    for p in paths:
        print(f"  {p.relative_to(OUT)}  ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
