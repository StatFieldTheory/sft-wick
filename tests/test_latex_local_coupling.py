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
    obs = [phi("a", "x"), phi("b", "x"), phi("c", "y")]
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
