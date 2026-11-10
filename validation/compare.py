"""Package values against the oracle, tag by tag, with declared tolerances.

The oracle's values come from :func:`validation.hierarchy.moments`; the
package's from ``validation/lower.py`` (run in a subprocess, so this
process never loads sft_wick), as ``{tag: (value, error)}``.  A tag no
diagram carries has the package value 0.

A value passes when ``|package − oracle| ≤ rel |oracle| + k_err error +
floor scale``, where ``error`` is the package's own estimate and ``scale``
the largest finite ``|value|`` of the comparison on either side, so that an
exact zero is compared against the size of the observable.  A joint
cumulant is a difference of products of raw moments that can cancel to a
few ulp of its largest product, so pass the scale that
:func:`validation.hierarchy.moments_with_scale` returns.  A row with a
value, error or scale that is not finite fails.  The tolerances depend on
the integrator and on where C comes from (:data:`TOLERANCES`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

__all__ = ["Tolerance", "TOLERANCES", "tolerance", "Row", "compare_values"]


@dataclass(frozen=True)
class Tolerance:
    rel: float
    k_err: float
    floor: float


#: ``(method, C source) → Tolerance``; ``'closed'`` stands for the oracle's
#: and the package's closed forms.  Gauss-Legendre with an exact C is
#: spectral once the kinks are split; ``nquad`` stops at scipy's default
#: 1.49e-8 relative per nested level; the QMC error column is a
#: single-run estimate that measured up to 9 times too small at 2^16
#: samples, hence the floor; quadrature tables carry their interpolation
#: error (1e-8 to 1e-5 in the demos).
TOLERANCES = {
    ("gauss_legendre", "closed"): Tolerance(rel=1e-10, k_err=0.0, floor=1e-12),
    ("nquad", "closed"): Tolerance(rel=1e-7, k_err=10.0, floor=1e-12),
    ("qmc_vectorized", "closed"): Tolerance(rel=1e-9, k_err=10.0, floor=1e-8),
    ("qmc_scalar", "closed"): Tolerance(rel=1e-9, k_err=10.0, floor=1e-8),
    ("qmc", "closed"): Tolerance(rel=1e-9, k_err=10.0, floor=1e-8),
}
TABLE = Tolerance(rel=1e-4, k_err=10.0, floor=1e-6)


def tolerance(method: str, c_source: str) -> Tolerance:
    """The tolerance of an integrator with C from ``c_source`` (``'oracle'``
    or ``'auto'``: a closed form; ``'table'``: quadrature tables)."""
    if c_source == "table":
        return TABLE
    key = (method, "closed")
    if key not in TOLERANCES:
        raise ValueError(f"no tolerance for method {method!r}")
    return TOLERANCES[key]


@dataclass(frozen=True)
class Row:
    tag: tuple
    oracle: float
    package: float
    error: float
    diff: float
    allowed: float
    finite: bool = True

    @property
    def ok(self) -> bool:
        return self.finite and abs(self.diff) <= self.allowed

    @property
    def ratio(self) -> float:
        """``|diff| / allowed``: below 1 passes."""
        return abs(self.diff) / self.allowed if self.allowed else float(
            "inf") if self.diff else 0.0


def compare_values(oracle: Mapping, package: Mapping, tol: Tolerance,
                   scale: float | None = None) -> list:
    """One :class:`Row` per oracle tag.  ``package`` maps a tag to
    ``(value, error)``; a package tag the oracle was not asked for is an
    error (the comparison would be incomplete)."""
    extra = set(map(tuple, package)) - set(map(tuple, oracle))
    if extra:
        raise ValueError(f"package tags {sorted(extra)} have no oracle value")
    if scale is None:
        values = [float(v) for v in oracle.values()] + [
            float(v) for v, _ in package.values()]
        scale = max((abs(v) for v in values if math.isfinite(v)), default=0.0)
    rows = []
    for tag in sorted(oracle):
        ref = float(oracle[tag])
        value, error = (float(x) for x in package.get(tag, (0.0, 0.0)))
        finite = all(map(math.isfinite, (ref, value, error, scale)))
        allowed = (tol.rel * abs(ref) + tol.k_err * error + tol.floor * scale
                   if finite else 0.0)
        rows.append(Row(tuple(tag), ref, value, error, value - ref, allowed,
                        finite))
    return rows
