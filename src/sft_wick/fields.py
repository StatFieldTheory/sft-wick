"""Field declarations and field operators.

A Field is a declaration (phi, psi) with metadata.
A FieldOperator is a specific instance with bound component index and spatial argument.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FieldType(Enum):
    PHYSICAL = "physical"  # phi
    RESPONSE = "response"  # psi


# Global counter for unique FieldOperator IDs
_uid_counter = itertools.count()


def reset_uid_counter() -> None:
    """Reset the UID counter. Useful for reproducible tests."""
    global _uid_counter
    _uid_counter = itertools.count()


@dataclass(frozen=True)
class Field:
    """Declaration of a field species.

    Examples:
        phi = Field('phi', FieldType.PHYSICAL, n_components=3)
        psi = Field('psi', FieldType.RESPONSE)  # scalar (n_components=1)
    """

    name: str
    field_type: FieldType
    n_components: int = 1

    def __init__(self, name: str, field_type: str | FieldType, n_components: int = 1):
        object.__setattr__(self, "name", name)
        if isinstance(field_type, str):
            field_type = FieldType(field_type)
        object.__setattr__(self, "field_type", field_type)
        object.__setattr__(self, "n_components", n_components)

    def __call__(self, *args: str) -> FieldOperator:
        """Create a field operator with bound indices.

        For multi-component (n_components > 1):
            phi('a', 'x') -> phi_a(x)

        For scalar (n_components == 1):
            phi('x') -> phi(x)
        """
        if self.is_scalar:
            if len(args) != 1:
                raise ValueError(
                    f"Scalar field '{self.name}' expects 1 argument (spatial), "
                    f"got {len(args)}"
                )
            return FieldOperator(
                field=self,
                component_index=None,
                spatial_arg=args[0],
                uid=next(_uid_counter),
            )
        else:
            if len(args) != 2:
                raise ValueError(
                    f"Multi-component field '{self.name}' expects 2 arguments "
                    f"(component, spatial), got {len(args)}"
                )
            return FieldOperator(
                field=self,
                component_index=args[0],
                spatial_arg=args[1],
                uid=next(_uid_counter),
            )

    @property
    def is_scalar(self) -> bool:
        return self.n_components == 1

    @property
    def is_physical(self) -> bool:
        return self.field_type == FieldType.PHYSICAL

    @property
    def is_response(self) -> bool:
        return self.field_type == FieldType.RESPONSE


@dataclass(frozen=True)
class FieldOperator:
    """A field at a specific point with a specific component index.

    Each instance has a unique uid to distinguish identical-looking operators
    (e.g., two phi_a(x) in the same product).
    """

    field: Field
    component_index: Optional[str]
    spatial_arg: str
    uid: int

    @property
    def field_type(self) -> FieldType:
        return self.field.field_type

    @property
    def is_physical(self) -> bool:
        return self.field.is_physical

    @property
    def is_response(self) -> bool:
        return self.field.is_response

    def __repr__(self) -> str:
        if self.component_index is not None:
            return f"{self.field.name}_{self.component_index}({self.spatial_arg})"
        return f"{self.field.name}({self.spatial_arg})"
