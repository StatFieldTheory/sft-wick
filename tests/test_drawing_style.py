"""Tests for the matplotlib renderer + style abstractions.

Locks in:

* presets behave (default / publication / grayscale / minimal),
* ``RenderStyle`` is immutable and ``with_*`` helpers return copies,
* ``DiagramRenderer`` honours per-call label overrides, callable
  hooks, and manual position pinning,
* ``draw_all`` no longer calls ``plt.show()`` implicitly,
* the legacy class attribute ``DiagramRenderer.PROP_STYLES`` still
  works for users that monkey-patched it.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from sft_wick import (  # noqa: E402
    DiagramRenderer,
    FeynmanDiagram,
    LABEL_COMPACT,
    LABEL_FULL,
    LABEL_TIME_F,
    LayoutParams,
    PropagatorStyle,
    RenderStyle,
    default_style,
    grayscale_style,
    minimal_style,
    publication_style,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def two_external_diagram() -> FeynmanDiagram:
    fd = FeynmanDiagram()
    e0 = fd.add_external_point(
        label=r"$\phi_a$",
        field_type="physical",
        component="a",
        spatial="x_1",
        full_label=r"$\phi_a(x_1)$",
    )
    e1 = fd.add_external_point(
        label=r"$\phi_b$",
        field_type="physical",
        component="b",
        spatial="x_2",
        full_label=r"$\phi_b(x_2)$",
    )
    v = fd.add_vertex("F")
    fd.add_propagator(e0, v, kind="C")
    fd.add_propagator(v, e1, kind="C")
    return fd


@pytest.fixture
def diagram_with_R(two_external_diagram: FeynmanDiagram) -> FeynmanDiagram:
    fd = two_external_diagram
    v = fd.vertex_nodes[0]
    e0 = fd.external_nodes[0]
    fd.add_propagator(v, e0, kind="R", phi_end=e0, psi_end=v)
    return fd


# ----------------------------------------------------------------------
# RenderStyle / presets
# ----------------------------------------------------------------------

class TestRenderStyle:
    def test_default_style_returns_renderstyle(self):
        s = default_style()
        assert isinstance(s, RenderStyle)
        assert "C" in s.propagators
        assert "R" in s.propagators
        assert s.label_format == LABEL_COMPACT

    def test_publication_style_uses_serif(self):
        s = publication_style()
        rc = s.effective_rcparams()
        assert rc["font.family"] == "serif"
        # usetex defaults off
        assert "text.usetex" not in rc

    def test_publication_style_usetex_flag(self):
        s = publication_style(usetex=True)
        assert s.effective_rcparams()["text.usetex"] is True

    def test_grayscale_style_no_color(self):
        s = grayscale_style()
        for prop in s.propagators.values():
            assert prop.color in {"black", "#000000"}

    def test_minimal_style_hides_legend(self):
        s = minimal_style()
        assert s.show_legend is False

    def test_with_overrides_returns_new_object(self):
        s = default_style()
        s2 = s.with_overrides(show_legend=False)
        assert s is not s2
        assert s.show_legend is True
        assert s2.show_legend is False

    def test_with_propagator_returns_new_object(self):
        s = default_style()
        s2 = s.with_propagator("C", color="black")
        assert s.propagators["C"].color != "black"
        assert s2.propagators["C"].color == "black"
        # other kinds untouched
        assert s2.propagators["R"].color == s.propagators["R"].color

    def test_propagators_mapping_immutable(self):
        s = default_style()
        with pytest.raises(TypeError):
            s.propagators["C"] = PropagatorStyle(  # type: ignore[index]
                color="black", linestyle="solid", linewidth=1.0,
            )

    def test_invalid_label_format_raises(self):
        s = default_style()
        with pytest.raises(ValueError):
            s.with_overrides(label_format="bogus")

    def test_layout_params_frozen(self):
        p = LayoutParams()
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.ext_radius = 5.0  # type: ignore[misc]


# ----------------------------------------------------------------------
# DiagramRenderer
# ----------------------------------------------------------------------

class TestRenderer:
    def test_constructor_backward_compatible(self, two_external_diagram):
        renderer = DiagramRenderer(figsize=(4, 3))
        assert renderer.figsize == (4, 3)
        ax = renderer.draw(two_external_diagram, title="t")
        assert ax.get_title() == "t"
        plt.close("all")

    def test_uses_supplied_style(self, two_external_diagram):
        style = default_style().with_propagator("C", color="#123456")
        renderer = DiagramRenderer(style=style)
        ax = renderer.draw(two_external_diagram)
        # The C edge must use the overridden colour.  FancyArrowPatch
        # stores it via get_edgecolor()/get_facecolor().
        from matplotlib.patches import FancyArrowPatch
        edges = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
        assert edges, "expected at least one edge patch"
        colors = {tuple(round(c, 4) for c in p.get_edgecolor()) for p in edges}
        # 0x12 / 0x34 / 0x56 → 18 / 52 / 86 in /255
        target = (0.0706, 0.2039, 0.3373, 1.0)
        match = any(
            all(abs(a - b) < 0.01 for a, b in zip(c, target))
            for c in colors
        )
        assert match, f"expected #123456 in edge colours, got {colors}"
        plt.close("all")

    def test_external_label_override_dict(self, two_external_diagram):
        renderer = DiagramRenderer()
        e0 = two_external_diagram.external_nodes[0]
        ax = renderer.draw(
            two_external_diagram,
            external_labels={e0: r"$\Phi_{\rm new}$"},
        )
        texts = [t.get_text() for t in ax.texts]
        assert r"$\Phi_{\rm new}$" in texts
        plt.close("all")

    def test_external_label_callable(self, two_external_diagram):
        captured: list[str] = []

        def fn(node_id, attrs):
            captured.append(node_id)
            return f"NODE-{node_id}"

        renderer = DiagramRenderer(external_label_fn=fn)
        ax = renderer.draw(two_external_diagram)
        texts = [t.get_text() for t in ax.texts]
        assert any(t.startswith("NODE-ext_") for t in texts)
        # Both externals visited
        assert len(captured) == 2
        plt.close("all")

    def test_label_format_compact_omits_spatial(self, two_external_diagram):
        renderer = DiagramRenderer(label_format=LABEL_COMPACT)
        ax = renderer.draw(two_external_diagram)
        texts = [t.get_text() for t in ax.texts]
        # Compact form: $\phi_{a}$ — no parentheses
        assert any(t == r"$\phi_{a}$" for t in texts)
        assert not any("(" in t and "x_1" in t for t in texts)
        plt.close("all")

    def test_label_format_full_uses_full_label(self, two_external_diagram):
        renderer = DiagramRenderer(label_format=LABEL_FULL)
        ax = renderer.draw(two_external_diagram)
        texts = [t.get_text() for t in ax.texts]
        assert any("x_1" in t for t in texts)
        plt.close("all")

    def test_label_format_time_f(self, two_external_diagram):
        renderer = DiagramRenderer(label_format=LABEL_TIME_F)
        ax = renderer.draw(two_external_diagram)
        texts = [t.get_text() for t in ax.texts]
        assert any("t_f" in t for t in texts)
        plt.close("all")

    def test_manual_positions_pin_node(self, two_external_diagram):
        renderer = DiagramRenderer()
        e0 = two_external_diagram.external_nodes[0]
        ax = renderer.draw(
            two_external_diagram,
            positions={e0: (-7.5, 1.25)},
        )
        # axis bounds reflect the pinned-far-out node
        assert ax.get_xlim()[0] <= -7.5
        plt.close("all")

    def test_show_legend_false_hides_legend(self, diagram_with_R):
        renderer = DiagramRenderer(
            style=default_style().with_overrides(show_legend=False),
        )
        ax = renderer.draw(diagram_with_R)
        assert ax.get_legend() is None
        plt.close("all")

    def test_show_legend_per_call_override(self, diagram_with_R):
        renderer = DiagramRenderer()  # default style: show_legend=True
        ax = renderer.draw(diagram_with_R, show_legend=False)
        assert ax.get_legend() is None
        plt.close("all")

    def test_grayscale_preset_yields_black_edges(self, diagram_with_R):
        from matplotlib.patches import FancyArrowPatch
        renderer = DiagramRenderer(style=grayscale_style())
        ax = renderer.draw(diagram_with_R)
        for p in ax.patches:
            if isinstance(p, FancyArrowPatch):
                rgba = tuple(round(c, 3) for c in p.get_edgecolor())
                # Pure black RGBA = (0, 0, 0, 1)
                assert rgba[:3] == (0.0, 0.0, 0.0)
        plt.close("all")

    def test_draw_all_does_not_call_show(self, two_external_diagram):
        renderer = DiagramRenderer()
        with patch.object(plt, "show") as mock_show:
            fig = renderer.draw_all(
                [two_external_diagram, two_external_diagram], ncols=2,
            )
        assert mock_show.call_count == 0
        # Returned figure has expected number of axes
        assert len(fig.axes) >= 2
        plt.close("all")

    def test_draw_all_show_true_calls_show(self, two_external_diagram):
        renderer = DiagramRenderer()
        with patch.object(plt, "show") as mock_show:
            renderer.draw_all([two_external_diagram], show=True)
        assert mock_show.call_count == 1
        plt.close("all")

    def test_legacy_PROP_STYLES_override(self, two_external_diagram):
        from matplotlib.patches import FancyArrowPatch
        original = dict(DiagramRenderer.PROP_STYLES["C"])
        try:
            DiagramRenderer.PROP_STYLES["C"] = {
                "linestyle": "-", "color": "#00ff00", "linewidth": 3.0,
            }
            ax = DiagramRenderer().draw(two_external_diagram)
            target = (0.0, 1.0, 0.0, 1.0)
            edges = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
            colors = {tuple(round(c, 3) for c in e.get_edgecolor()) for e in edges}
            assert any(
                all(abs(a - b) < 0.01 for a, b in zip(c, target)) for c in colors
            ), f"expected lime green via legacy PROP_STYLES, got {colors}"
        finally:
            DiagramRenderer.PROP_STYLES["C"] = original
            plt.close("all")

    def test_publication_rcparams_do_not_leak(self, two_external_diagram):
        before = matplotlib.rcParams["font.family"]
        renderer = DiagramRenderer(style=publication_style())
        renderer.draw(two_external_diagram)
        after = matplotlib.rcParams["font.family"]
        assert before == after
        plt.close("all")

    def test_empty_diagram_produces_axes(self):
        fd = FeynmanDiagram()
        ax = DiagramRenderer().draw(fd, title="empty")
        assert ax.get_title() == "empty"
        plt.close("all")
