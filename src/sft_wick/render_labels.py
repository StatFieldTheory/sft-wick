"""Label-text generation for Feynman-diagram nodes.

Both rendering backends (matplotlib and TikZ) call into this module
to decide what string sits next to each external point and each
interaction vertex.  Centralising the logic keeps the two backends
visually consistent.

The default external label varies with the
:data:`~sft_wick.render_style.LABEL_COMPACT` /
:data:`~sft_wick.render_style.LABEL_FULL` /
:data:`~sft_wick.render_style.LABEL_TIME_F` flag.  Users can:

1. Override individual node labels with the ``external_labels`` /
   ``vertex_labels`` keyword on
   :meth:`sft_wick.drawing.DiagramRenderer.draw`.
2. Supply a callable
   ``fn(node_id: str, node_attrs: Mapping) -> str``
   on the renderer constructor for systematic overrides.
3. Keep the default and just switch the format flag globally.

Resolution order, highest priority first:
``overrides[node_id] > callable_fn(...) > default_fn(...)``.

Mathtext / dollar wrapping is the caller's responsibility for
overrides; the default formatters return strings already wrapped in
``$…$`` so they render as math in matplotlib.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .render_style import LABEL_COMPACT, LABEL_FORMATS, LABEL_FULL, LABEL_TIME_F


# ----------------------------------------------------------------------
# Default formatters
# ----------------------------------------------------------------------

def _field_symbol(node_attrs: Mapping[str, Any]) -> str:
    """Return ``r'\\phi'`` or ``r'\\psi'`` depending on the field type.

    Handles both the string form (``"physical"`` / ``"response"``)
    and the older :class:`sft_wick.fields.FieldType` enum (which has a
    ``.value`` attribute).
    """
    ft = node_attrs.get("field_type", "physical")
    ft_str = ft.value if hasattr(ft, "value") else ft
    return r"\phi" if ft_str == "physical" else r"\psi"


def default_external_label(
    node_attrs: Mapping[str, Any],
    fmt: str = LABEL_COMPACT,
) -> str:
    """Format the default label for an external node.

    Args:
        node_attrs: Networkx node-data dict for the external point.
            Expected keys: ``field_type``, ``component``, ``spatial``,
            and optionally ``full_label`` (for ``LABEL_FULL``).
        fmt: One of :data:`LABEL_COMPACT`, :data:`LABEL_FULL`,
            :data:`LABEL_TIME_F`.

    Returns:
        A string already wrapped in ``$…$`` for math-mode rendering.

    Raises:
        ValueError: If ``fmt`` is not one of the recognised formats.
    """
    if fmt not in LABEL_FORMATS:
        raise ValueError(
            f"fmt must be one of {LABEL_FORMATS}, got {fmt!r}."
        )

    sym = _field_symbol(node_attrs)
    comp = node_attrs.get("component")

    if fmt == LABEL_FULL:
        # Honour a pre-computed full label if the diagram builder
        # stashed one (current convention in
        # FeynmanDiagram.from_pairing); otherwise reconstruct.
        full = node_attrs.get("full_label")
        if full:
            return full
        spatial = node_attrs.get("spatial", "")
        if comp is None:
            return rf"${sym}({spatial})$" if spatial else f"${sym}$"
        return rf"${sym}_{{{comp}}}({spatial})$" if spatial \
            else rf"${sym}_{{{comp}}}$"

    if fmt == LABEL_TIME_F:
        if comp is None:
            return rf"${sym}(t_f)$"
        return rf"${sym}_{{{comp}}}(t_f)$"

    # LABEL_COMPACT
    if comp is None:
        return f"${sym}$"
    return rf"${sym}_{{{comp}}}$"


def default_vertex_label(node_attrs: Mapping[str, Any]) -> str:
    """Format the default label for an interaction vertex.

    Wraps the coupling name (e.g. ``"F"``) in ``$…$``.
    Empty coupling → empty string (no label drawn).
    """
    coupling = node_attrs.get("coupling", "")
    if not coupling:
        return ""
    # If the user already wrapped it in $...$, leave it alone.
    if coupling.startswith("$") and coupling.endswith("$"):
        return coupling
    return f"${coupling}$"


# ----------------------------------------------------------------------
# Resolution
# ----------------------------------------------------------------------

LabelCallable = Callable[[str, Mapping[str, Any]], str]


def resolve_label(
    node_id: str,
    node_attrs: Mapping[str, Any],
    overrides: Mapping[str, str] | None,
    callable_fn: LabelCallable | None,
    default_fn: Callable[[Mapping[str, Any]], str],
) -> str:
    """Pick the best label for a node by checking sources in order.

    Priority: ``overrides[node_id]`` > ``callable_fn(node_id, attrs)``
    > ``default_fn(attrs)``.

    A ``None`` or missing entry in ``overrides`` falls through.  A
    callable returning ``None`` also falls through (so the callable
    can opt out for specific nodes).
    """
    if overrides is not None and node_id in overrides:
        val = overrides[node_id]
        if val is not None:
            return val

    if callable_fn is not None:
        val = callable_fn(node_id, node_attrs)
        if val is not None:
            return val

    return default_fn(node_attrs)


# Re-export format flags so callers can do
#     from sft_wick.render_labels import LABEL_COMPACT
# without reaching into render_style.
__all__ = [
    "LABEL_COMPACT",
    "LABEL_FULL",
    "LABEL_TIME_F",
    "LabelCallable",
    "default_external_label",
    "default_vertex_label",
    "resolve_label",
]
