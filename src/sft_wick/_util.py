"""Internal utility functions."""

from __future__ import annotations

from math import prod


def double_factorial(n: int) -> int:
    """Compute n!! = n * (n-2) * (n-4) * ... * 1.

    For odd n: n!! = 1 * 3 * 5 * ... * n
    For even n: n!! = 2 * 4 * 6 * ... * n
    0!! = 1, (-1)!! = 1
    """
    if n <= 0:
        return 1
    return prod(range(n, 0, -2))
