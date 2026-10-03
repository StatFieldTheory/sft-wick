"""Internal utility functions."""

from __future__ import annotations

from math import prod

import numpy as np


def require_bool(value, name: str, where: str) -> bool:
    """Return ``value`` as a ``bool`` if it is a Python or NumPy bool;
    raise ``TypeError`` otherwise.

    ``bool(value)`` accepts anything: a tuple of leg groups passed as
    ``equal_time`` became ``True`` and put every leg at one time, which
    returned a wrong value without an error.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(
        f"{where}: {name} must be a bool; got {type(value).__name__} "
        f"{value!r}."
    )


def double_factorial(n: int) -> int:
    """Compute n!! = n * (n-2) * (n-4) * ... * 1.

    For odd n: n!! = 1 * 3 * 5 * ... * n
    For even n: n!! = 2 * 4 * 6 * ... * n
    0!! = 1, (-1)!! = 1
    """
    if n <= 0:
        return 1
    return prod(range(n, 0, -2))
