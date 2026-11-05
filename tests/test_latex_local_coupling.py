"""``DiagramTerm.to_latex`` must not print a local coupling's spacetime point.

A local vertex's coupling symbol carries its (single) point in
``spatial_args`` so that two copies of the vertex stay distinguishable and
a callable coupling can be evaluated at the right place, but the point
is suppressed when rendering (``Symbol.local``).  The simplification
passes that symmetrise the coupling sum rebuilt the symbols WITHOUT the
``local`` flag, so the second term of ``F_{i0 i1 i2} + F_{i1 i0 i2}``
came out as ``F_{i1 i0 i2}(y_0)`` -- the rendering the CPC referee
quoted from Table 1 of the paper.
"""

from __future__ import annotations

import re

from sft_wick import Action, Field, Vertex, compute_moment, reset_uid_counter
from sft_wick.expressions import Symbol


def _walk_symbols(expr):
    if isinstance(expr, Symbol):
        yield expr
        return
    for attr in ("factors", "terms"):
        for child in getattr(expr, attr, ()) or ():
            yield from _walk_symbols(child)
    for attr in ("expr", "body", "integrand"):
        child = getattr(expr, attr, None)
        if child is not None:
            yield from _walk_symbols(child)


def test_local_coupling_symbols_keep_the_local_flag_through_simplification():
    reset_uid_counter()
    phi = Field("phi", "physical", n_components=3)
    psi = Field("psi", "response", n_components=3)
    action = Action(vertices=[Vertex(fields=[phi, phi, psi], coupling="F")])
    # Distinct spatial labels per external operator: sharing one is
    # refused at interacting orders (see
    # tests/test_coincident_external_labels.py).  This test is about
    # symbol flags and rendering, not about the labels.
    obs = [phi("a", "x"), phi("b", "y"), phi("c", "z")]
    result = compute_moment(obs, action, order=1)

    seen = 0
    for dt in result.diagram_terms(1):
        for sym in _walk_symbols(dt.coupling_sum):
            if sym.name == "F":
                seen += 1
                assert sym.local, (
                    f"F symbol lost its local flag: {sym!r} in "
                    f"{dt.coupling_sum.to_latex()}"
                )
        latex = dt.to_latex()
        # No F_{...}(point) anywhere: a local coupling has no rendered argument.
        assert not re.search(r"F_\{[^}]*\}\s*\(", latex), latex
    assert seen >= 4, "expected several F symbols across the order-1 diagrams"


def test_display_names_reach_propagators_and_nothing_else():
    """The names were replaced in the rendered text, so a coupling named KC
    became KG, and swapped names ran into each other."""
    from sft_wick.expressions import Product, Propagator
    from sft_wick.latex import LaTeXFormatter

    c = Propagator("C", "a", "b", "x", "y")
    r = Propagator("R", "a", "b", "x", "y")
    assert (
        LaTeXFormatter({"C": "G", "R": "R"}).format(
            Product((Symbol("KC", (), ("y_0", "y_1")), c))
        )
        == "KC(y_{0}, y_{1}) G_{ab}(x, y)"
    )
    assert LaTeXFormatter({"C": "R", "R": "C"}).format(Product((c, r))) == (
        "R_{ab}(x, y) C_{ab}(x, y)"
    )


def test_display_names_refuse_what_is_not_an_expression():
    """A ``DiagramTerm`` has its own ``to_latex``, which would ignore the
    names with no error."""
    import pytest

    from sft_wick.latex import LaTeXFormatter

    reset_uid_counter()
    phi = Field("phi", "physical", n_components=2)
    psi = Field("psi", "response", n_components=2)
    action = Action(vertices=[Vertex(fields=[phi, phi, psi], coupling="F")])
    result = compute_moment([phi("a", "x"), phi("b", "y")], action, order=2)
    with pytest.raises(TypeError, match="takes an Expr"):
        LaTeXFormatter({"C": "G"}).format(result.diagram_terms(2)[0])


def test_evaluated_latex_renders_a_propagator_as_to_latex_does():
    import numpy as np

    phi = Field("phi", "physical", 2)
    action = Action([Vertex([phi, phi], "Q")])
    observable = [phi("a_1", "x"), phi("b", "y")]
    term = compute_moment(observable, action, 1).diagram_terms(1)[0]
    evaluated = term.to_latex_evaluated(
        {"Q": np.array([[1.0, 2.0], [3.0, 4.0]])}, {"a_1": 0, "b": 1}
    )
    assert "C_{a_{1}b}(x, y)" in term.to_latex()
    assert "C_{a_{1}b}(x, y)" in evaluated
