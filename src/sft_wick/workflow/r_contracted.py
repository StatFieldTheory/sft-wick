"""Brute-force reference utilities for ``already_R_contracted=True``.

The R-contracted form of a non-local vertex is the identity

.. math::

    \\kappa^{(m)}_R(\\gamma; z_1', \\ldots, z_m')
        \\;:=\\; \\int dz_1 \\cdots dz_m \\;
                \\prod_{i=1}^m R(z_i', z_i) \\,
                \\kappa^{(m)}(\\gamma; z_1, \\ldots, z_m).

This module provides :func:`build_R_contracted_callable`, which wraps a
raw :math:`\\kappa^{(m)}` callable into the R-contracted form by
brute-force numerical quadrature on a user-supplied :math:`\\chi`-grid.
Use it as the **reference comparand** when validating the
``NonLocalVertex(already_R_contracted=True)`` dispatch (see
``docs/notes/R_contracted_nonlocal_vertex.md`` §4) — for production work,
the upstream `canoes` library publishes analytical FFTlog-of-W
contractions that avoid the narrow-kernel cost entirely.

The utility is intentionally simple (a tensor-product trapezoid rule
over scalar leg times); it is not optimised for high accuracy. For
production validation pass a sufficiently fine ``chi_grid`` to resolve
the narrow-kernel diagonal of the raw :math:`\\kappa^{(m)}`.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def build_R_contracted_callable(
    raw_coupling_fn: Callable[..., np.ndarray],
    R_time: Callable[[float, float], float],
    chi_grid: Sequence[float] | np.ndarray,
    *,
    order: int = 3,
    n_components: int | None = None,
    causal: bool = True,
) -> Callable[..., np.ndarray]:
    """Wrap a raw ``κ^(m)`` callable as an R-contracted reference.

    The returned callable matches the ``NonLocalVertex`` callable
    contract: ``fn(n_list, t_list) -> (N,)*m`` with ``t_list[i]``
    interpreted as the **partner (outer)** time ``λ_i'`` rather than
    the leg-internal time ``χ_i``. Spatial positions are passed through
    unchanged — full spatial R-contraction is left to the user kernel
    (the analytical FFTlog-of-W chain on the canoes side).

    Wrap the result with ``NonLocalVertex(already_R_contracted=True,
    coupling=fn)`` to consume it via the Phase-2 dispatch (once it
    lands).

    Args:
        raw_coupling_fn: The raw ``κ^(m)(γ; χ_1, …, χ_m)`` callable.
            Must accept ``(n_list, t_list)`` with length-``order``
            sequences and return a ``(N,)*order`` numpy array.
        R_time: Causal scalar response propagator
            ``R_time(t_outer, t_inner) -> float``. The reference does
            **not** apply ``δ(n − n')`` — n_list is passed unchanged to
            ``raw_coupling_fn`` for every χ-sample.
        chi_grid: 1-D grid of inner times ``χ`` used for the
            tensor-product trapezoid quadrature on each leg.
        order: ``m`` — the rank of the κ tensor. Defaults to 3 (the
            squeezed bispectrum case). The implementation is general
            in ``order``.
        n_components: Optional ``N`` — used only for an early shape
            assertion on the raw callable's first output. If ``None``,
            the shape is inferred from the first evaluation.
        causal: If ``True`` (default), the R kernel is treated as
            causal (``R(t_outer, t_inner) = 0`` when
            ``t_inner > t_outer``). The brute-force loop short-circuits
            those χ-samples to zero; pass ``False`` to evaluate over
            the full grid (useful for testing non-causal kernels).

    Returns:
        A callable ``fn(n_list, t_list_outer) -> (N,)*order``
        suitable for use as a ``NonLocalVertex.coupling`` under
        ``already_R_contracted=True``.

    Notes:
        Computational cost per call is
        ``O(|chi_grid|^order)`` raw-callable evaluations. For ``m=3``
        and a 200-node χ-grid that's 8 × 10⁶ raw evaluations per
        outer-time tuple — slow but exact (to trapezoid order) and
        easy to debug. Vectorise over outer-time tuples upstream when
        a large `(λ_1', λ_2', λ_3')` table is needed.
    """
    chi = np.asarray(chi_grid, dtype=float)
    if chi.ndim != 1 or chi.size < 2:
        raise ValueError(
            "chi_grid must be a 1-D array of at least 2 points; "
            f"got shape {chi.shape}."
        )
    if int(order) < 1:
        raise ValueError(f"order must be ≥ 1, got {order}.")
    order_int = int(order)

    # Trapezoid weights on the chi grid: dx_i = (chi[i+1] - chi[i-1]) / 2
    # at interior nodes, half that at the endpoints.
    dchi = np.empty_like(chi)
    dchi[1:-1] = 0.5 * (chi[2:] - chi[:-2])
    dchi[0] = 0.5 * (chi[1] - chi[0])
    dchi[-1] = 0.5 * (chi[-1] - chi[-2])

    def fn(n_list, t_list_outer):
        # The R-contracted callable is per-sample; vectorising over
        # multiple outer tuples is the caller's job (see e.g. a
        # tabulator that builds an (Nλ, Nλ, Nλ, N, N, N) array offline).
        n_arr = np.asarray(n_list)
        t_outer = np.asarray(t_list_outer, dtype=float)
        if t_outer.shape != (order_int,):
            raise ValueError(
                f"t_list_outer must have shape ({order_int},); "
                f"got {t_outer.shape}."
            )

        # Precompute R(t_outer[i], chi[k]) for every (i, k).
        R_table = np.empty((order_int, chi.size), dtype=float)
        for i in range(order_int):
            for k, c in enumerate(chi):
                if causal and c > t_outer[i]:
                    R_table[i, k] = 0.0
                else:
                    R_table[i, k] = float(R_time(t_outer[i], c))

        result = None  # shape inferred from first call

        # Tensor-product trapezoid over (k_1, k_2, …, k_m).
        for multi_idx in np.ndindex(*([chi.size] * order_int)):
            # Build the χ-tuple and the joint weight ∏ R(t', χ) dχ
            w = 1.0
            short_circuit = False
            for axis, k in enumerate(multi_idx):
                rval = R_table[axis, k]
                if rval == 0.0:
                    short_circuit = True
                    break
                w *= rval * dchi[k]
            if short_circuit:
                continue
            chi_tuple = tuple(chi[k] for k in multi_idx)

            raw_val = raw_coupling_fn(n_arr, chi_tuple)
            raw_arr = np.asarray(raw_val)
            if result is None:
                if n_components is not None:
                    expected = (n_components,) * order_int
                    if raw_arr.shape != expected:
                        raise ValueError(
                            f"raw_coupling_fn returned shape "
                            f"{raw_arr.shape}, expected {expected}."
                        )
                result = np.zeros(raw_arr.shape, dtype=raw_arr.dtype)
            result += w * raw_arr

        if result is None:
            # All χ-samples short-circuited (e.g. causal R + outer
            # time below the χ-grid). Return zeros with whatever shape
            # one bare-callable evaluation suggests.
            probe = np.asarray(raw_coupling_fn(n_arr, tuple(chi[0] for _ in range(order_int))))
            result = np.zeros(probe.shape, dtype=probe.dtype)

        return result

    fn.__doc__ = (
        "Brute-force R-contracted κ^({m}) reference callable. "
        "Returns the R-weighted m-leg integral of the wrapped raw "
        "κ^({m}) on the supplied χ-grid."
    ).format(m=order_int)
    fn.order = order_int
    return fn


__all__ = ["build_R_contracted_callable"]
