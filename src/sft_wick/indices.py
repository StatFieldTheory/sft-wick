"""Index management for Wick contractions.

Generates fresh component indices and spatial variables for vertex instantiation.
"""

from __future__ import annotations


class IndexContext:
    """Generates fresh, unique index names for vertex instantiation.

    When multiple copies of vertices are created during perturbative expansion,
    each copy needs non-overlapping internal indices.
    """

    def __init__(self) -> None:
        self._component_counter = 0
        self._spatial_counter = 0

    def fresh_component_index(self, prefix: str = "i") -> str:
        """Generate a fresh component index name like i_0, i_1, ..."""
        name = f"{prefix}_{self._component_counter}"
        self._component_counter += 1
        return name

    def fresh_spatial_variable(self, prefix: str = "y") -> str:
        """Generate a fresh spatial variable name like y_0, y_1, ..."""
        name = f"{prefix}_{self._spatial_counter}"
        self._spatial_counter += 1
        return name
