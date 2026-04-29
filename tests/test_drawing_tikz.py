"""Tests for the TikZ rendering backend.

These tests assert the *shape* of the generated LaTeX (presence of
key constructs, correct node IDs, proper handling of self-loops, R
arrows, label overrides) rather than byte-exact output, so the
backend can evolve without rewriting tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sft_wick import (
    FeynmanDiagram,
    LABEL_COMPACT,
    LABEL_FULL,
    LABEL_TIME_F,
    TikzRenderer,
    default_style,
    grayscale_style,
    publication_style,
)


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


@pytest.fixture
def diagram_with_self_loop(two_external_diagram: FeynmanDiagram) -> FeynmanDiagram:
    fd = two_external_diagram
    v = fd.vertex_nodes[0]
    fd.add_propagator(v, v, kind="C")
    return fd


# ----------------------------------------------------------------------
# Output shape
# ----------------------------------------------------------------------

class TestTikzShape:
    def test_contains_tikzpicture_envelope(self, two_external_diagram):
        s = TikzRenderer().to_string(two_external_diagram)
        assert s.startswith(r"\begin{tikzpicture}")
        assert r"\end{tikzpicture}" in s
        assert s.endswith("\n")

    def test_contains_node_styles(self, two_external_diagram):
        s = TikzRenderer().to_string(two_external_diagram)
        assert r"\tikzset{ext/.style=" in s
        assert r"\tikzset{vert/.style=" in s

    def test_contains_external_node_definitions(self, two_external_diagram):
        s = TikzRenderer().to_string(two_external_diagram)
        # External node names are derived as ext0, ext1, ... (underscores stripped)
        assert "(ext0)" in s
        assert "(ext1)" in s

    def test_contains_C_edge(self, two_external_diagram):
        s = TikzRenderer().to_string(two_external_diagram)
        assert r"\draw[C]" in s

    def test_R_edge_uses_arrow_style(self, diagram_with_R):
        s = TikzRenderer().to_string(diagram_with_R)
        assert r"\draw[R]" in s
        # The R style declaration must include a Latex arrow tip
        assert "Latex" in s

    def test_self_loop_uses_loop_construct(self, diagram_with_self_loop):
        s = TikzRenderer().to_string(diagram_with_self_loop)
        assert "edge[loop" in s

    def test_label_compact_default(self, two_external_diagram):
        s = TikzRenderer().to_string(two_external_diagram)
        assert r"\phi_{a}" in s
        # No spatial argument in compact mode
        assert "(x_1)" not in s

    def test_label_full_includes_spatial(self, two_external_diagram):
        s = TikzRenderer(label_format=LABEL_FULL).to_string(two_external_diagram)
        assert "(x_1)" in s

    def test_label_time_f(self, two_external_diagram):
        s = TikzRenderer(label_format=LABEL_TIME_F).to_string(two_external_diagram)
        assert "t_f" in s


# ----------------------------------------------------------------------
# Override hooks
# ----------------------------------------------------------------------

class TestTikzOverrides:
    def test_external_label_dict_override(self, two_external_diagram):
        e0 = two_external_diagram.external_nodes[0]
        s = TikzRenderer().to_string(
            two_external_diagram,
            external_labels={e0: r"$\Phi_{\rm new}$"},
        )
        assert r"\Phi_{\rm new}" in s

    def test_vertex_label_dict_override(self, two_external_diagram):
        v = two_external_diagram.vertex_nodes[0]
        s = TikzRenderer().to_string(
            two_external_diagram,
            vertex_labels={v: r"$g_{\rm eff}$"},
        )
        assert r"g_{\rm eff}" in s

    def test_external_label_callable(self, two_external_diagram):
        def fn(node_id, attrs):
            return rf"$X_{{{node_id}}}$"
        s = TikzRenderer(external_label_fn=fn).to_string(two_external_diagram)
        assert r"X_{ext_0}" in s

    def test_manual_positions_used_in_coordinates(self, two_external_diagram):
        e0 = two_external_diagram.external_nodes[0]
        s = TikzRenderer().to_string(
            two_external_diagram,
            positions={e0: (-7.5, 1.25)},
        )
        assert "-7.5000, 1.2500" in s


# ----------------------------------------------------------------------
# Standalone wrapping
# ----------------------------------------------------------------------

class TestTikzStandalone:
    def test_standalone_wraps_in_documentclass(self, two_external_diagram):
        s = TikzRenderer(standalone=True).to_string(two_external_diagram)
        assert r"\documentclass" in s
        assert "standalone" in s
        assert r"\begin{document}" in s
        assert r"\end{document}" in s
        assert r"\usepackage{tikz}" in s

    def test_standalone_per_call_override(self, two_external_diagram):
        renderer = TikzRenderer(standalone=False)
        s_bare = renderer.to_string(two_external_diagram)
        s_doc = renderer.to_string(two_external_diagram, standalone=True)
        assert r"\documentclass" not in s_bare
        assert r"\documentclass" in s_doc


# ----------------------------------------------------------------------
# File I/O
# ----------------------------------------------------------------------

class TestTikzSave:
    def test_save_writes_file(self, two_external_diagram, tmp_path: Path):
        out = tmp_path / "fig.tex"
        path = TikzRenderer().save(two_external_diagram, out)
        assert path == out
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert r"\begin{tikzpicture}" in text

    def test_save_creates_parent_dirs(self, two_external_diagram, tmp_path: Path):
        out = tmp_path / "deep" / "tree" / "fig.tex"
        TikzRenderer().save(two_external_diagram, out)
        assert out.exists()

    def test_save_all_uses_pattern(self, two_external_diagram, tmp_path: Path):
        diagrams = [two_external_diagram, two_external_diagram, two_external_diagram]
        pattern = str(tmp_path / "diag_{i:02d}.tex")
        paths = TikzRenderer().save_all(diagrams, pattern)
        assert [p.name for p in paths] == [
            "diag_00.tex", "diag_01.tex", "diag_02.tex",
        ]
        for p in paths:
            assert p.exists()

    def test_save_all_per_diagram_external_labels(
        self, two_external_diagram, tmp_path: Path,
    ):
        diagrams = [two_external_diagram, two_external_diagram]
        e0 = two_external_diagram.external_nodes[0]
        per_diagram_overrides = {1: {e0: r"$\Psi$"}}
        TikzRenderer().save_all(
            diagrams,
            str(tmp_path / "d_{i}.tex"),
            external_labels=per_diagram_overrides,
        )
        first = (tmp_path / "d_0.tex").read_text(encoding="utf-8")
        second = (tmp_path / "d_1.tex").read_text(encoding="utf-8")
        assert r"\Psi" not in first
        assert r"\Psi" in second


# ----------------------------------------------------------------------
# Style integration
# ----------------------------------------------------------------------

class TestTikzStyles:
    def test_publication_style_uses_serif_label_distance(self, two_external_diagram):
        s = TikzRenderer(style=publication_style()).to_string(two_external_diagram)
        assert "label distance=" in s

    def test_grayscale_style_uses_black(self, two_external_diagram):
        s = TikzRenderer(style=grayscale_style()).to_string(two_external_diagram)
        # Both propagator kinds should declare black explicitly.
        assert "draw=black" in s

    def test_default_style_uses_rgb_color(self, two_external_diagram):
        s = TikzRenderer(style=default_style()).to_string(two_external_diagram)
        # default has hex color → translated to rgb,255:red,...
        assert "rgb,255" in s


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------

class TestTikzEdgeCases:
    def test_empty_diagram(self):
        s = TikzRenderer().to_string(FeynmanDiagram())
        assert r"\begin{tikzpicture}" in s
        assert r"\end{tikzpicture}" in s

    def test_default_label_format_compact(self, two_external_diagram):
        # Renderer default mirrors style default (LABEL_COMPACT)
        s = TikzRenderer().to_string(two_external_diagram)
        assert r"\phi_{a}" in s
        assert "x_1" not in s

    def test_scale_appears_in_tikzpicture_options(self, two_external_diagram):
        s = TikzRenderer(scale=2.5).to_string(two_external_diagram)
        assert "scale=2.5" in s
