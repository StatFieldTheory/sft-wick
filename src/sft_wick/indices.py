"""Index management for Wick contractions.

Generates fresh component indices and spatial variables for vertex instantiation.
"""

from __future__ import annotations

from typing import Iterable


class IndexContext:
    """Generates fresh, unique index names for vertex instantiation.

    When multiple copies of vertices are created during perturbative expansion,
    each copy needs non-overlapping internal indices.

    Args:
        reserved: names the generated names must not take, typically every
            spatial label and component label of the observable.  The
            Wick engines key operators by these strings, so an internal name
            equal to an external label merges the two into one point (or
            one index).  A reserved name is skipped and the counter moves
            on.  With nothing reserved the names are ``i_0, i_1, ...`` and
            ``y_0, y_1, ...`` in order.

    The two counters also skip every name either of them has already
    issued, so a caller that passes the same ``prefix`` to both still gets
    distinct names.
    """

    def __init__(self, reserved: Iterable[str] = ()) -> None:
        self._component_counter = 0
        self._spatial_counter = 0
        self._taken: set[str] = set(reserved)

    def fresh_component_index(self, prefix: str = "i") -> str:
        """Generate a fresh component index name like i_0, i_1, ..."""
        name, self._component_counter = self._next_free(
            prefix, self._component_counter
        )
        return name

    def fresh_spatial_variable(self, prefix: str = "y") -> str:
        """Generate a fresh spatial variable name like y_0, y_1, ..."""
        name, self._spatial_counter = self._next_free(prefix, self._spatial_counter)
        return name

    def _next_free(self, prefix: str, counter: int) -> tuple[str, int]:
        """The first ``f"{prefix}_{k}"`` with ``k >= counter`` not yet taken,
        and the counter value after it."""
        name = f"{prefix}_{counter}"
        while name in self._taken:
            counter += 1
            name = f"{prefix}_{counter}"
        self._taken.add(name)
        return name, counter + 1


#: The prefixes the package's generators use: ``i`` for component indices
#: (:class:`IndexContext` and the canonical renames of the simplifier) and
#: ``y`` for vertex points.
GENERATED_PREFIXES = ("i", "y")


def is_generated_name(label: str) -> bool:
    """True when ``label`` is spelled like a name the package generates
    (``i_0``, ``y_12``, ...): a generated prefix, an underscore and a
    non-negative integer without leading zeros."""
    prefix, sep, number = str(label).partition("_")
    return (
        sep == "_"
        and prefix in GENERATED_PREFIXES
        and number.isdigit()
        and number.isascii()
        and str(int(number)) == number
    )


def fresh_names(prefix: str, count: int, reserved: Iterable[str]) -> list[str]:
    """``count`` names ``f"{prefix}_{k}"`` for ``k = 0, 1, ...``, skipping
    every name in ``reserved``.

    With no reserved name of that form the result is
    ``[f"{prefix}_0", ..., f"{prefix}_{count - 1}"]``.
    """
    ctx = IndexContext(reserved)
    return [ctx.fresh_component_index(prefix) for _ in range(count)]


def observable_labels(observable: Iterable) -> frozenset[str]:
    """Every spatial label and component label carried by ``observable``
    (an iterable of :class:`~sft_wick.fields.FieldOperator`).

    These are the names the expansion's generated names must avoid: see
    :class:`IndexContext`.  Both kinds are reserved for both generators,
    since a vertex-copy coupling symbol carries spatial and component
    labels side by side and some substitutions rename both at once.
    """
    labels: set[str] = set()
    for op in observable:
        labels.add(op.spatial_arg)
        if op.component_index is not None:
            labels.add(op.component_index)
    return frozenset(labels)
