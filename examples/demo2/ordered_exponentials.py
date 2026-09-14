"""Elementary exponential integrals on an ordered, separately capped simplex.

Integrate exp(offset + sum(e_i*u_i)) over 0 <= u_0 <= ... <= u_{m-1},
with u_i <= t_i.  Let J_k be the first k nested integrals.  The states
z_k = exp(sum(e[k:])*s)*J_k(s) obey a lower-bidiagonal constant ODE,
with diagonal p_k=sum(e[k:]) and subdiagonal 1.  The k-th gate closes at
min(t[k:]).  Exponentiate each interval, then close its gate.

For distinct rates a matrix entry is the divided difference of exp:
E_ij = sum(exp(p_k*h)/prod(p_k-p_l), k=j..i, l=j..i excluding k).
Short intervals use the convergent matrix Taylor series to avoid
cancellation.  Coincident rates use scipy's small matrix exponential.
All exponentials are shifted by the largest active rate to avoid
overflow; the physical prefactor is restored only at the end.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.linalg import expm
from joblib.externals import cloudpickle

# L2 serializes external hook modules by value.  Include this sibling
# helper too, so an already running worker need not import its directory.
cloudpickle.register_pickle_by_value(sys.modules[__name__])
_COEFFICIENTS = {}


def _coefficients(rates, first):
    key = (rates, first)
    if key in _COEFFICIENTS:
        return _COEFFICIENTS[key]
    p = np.r_[np.cumsum(np.asarray(rates)[::-1])[::-1], 0.0][first:]
    shift = max(p)
    q = p - shift
    n = len(q)
    differences = np.abs(q[:, None]-q[None, :])
    np.fill_diagonal(differences, np.inf)
    if differences.min() < 1e-4*max(np.max(np.abs(q)), 1):
        value = q, shift, None
    else:
        terms = []
        for i in range(n):
            for j in range(i+1):
                terms.append((i, j, tuple(
                    (k, 1/np.prod([q[k]-q[l] for l in range(j, i+1) if l != k]))
                    for k in range(j, i+1))))
        value = q, shift, tuple(terms)
    if len(_COEFFICIENTS) >= 256:
        _COEFFICIENTS.clear()
    _COEFFICIENTS[key] = value
    return value


def ordered_integral(times, rates, offset):
    """Batched times (m,n), fixed rates (m,), and log prefactor (n,)."""
    times = np.asarray(times, dtype=float)
    rates = tuple(float(rate) for rate in rates)
    m, n = times.shape
    if len(rates) != m:
        raise ValueError("one rate per leg required")
    bounds = np.minimum.accumulate(times[::-1], axis=0)[::-1]
    state = np.zeros((m+1, n))
    state[0] = 1
    start = np.zeros(n)
    log_scale = np.broadcast_to(offset, (n,)).copy()
    for first, end in enumerate(bounds):
        width = np.maximum(end-start, 0)
        q, shift, terms = _coefficients(rates, first)
        current = state[first:]
        following = np.zeros_like(current)
        small = width*max(np.max(np.abs(q)), 1) < 2
        big = ~small
        if np.any(big):
            if terms is None:
                matrix = np.diag(q) + np.diag(np.ones(len(q)-1), -1)
                for column in np.flatnonzero(big):
                    following[:, column] = expm(matrix*width[column]) @ current[:, column]
            else:
                exponentials = np.exp(q[:, None]*width[big])
                for i, j, coefficients in terms:
                    following[i, big] += sum(c*exponentials[k] for k, c in coefficients)*current[j, big]
        if np.any(small):
            term = current[:, small].copy()
            value = term.copy()
            for degree in range(1, 29):
                derivative = q[:, None]*term
                derivative[1:] += term[:-1]
                term = derivative*(width[small]/degree)
                value += term
            following[:, small] = value
        state[first:] = following
        state[first] = 0  # closed gate: this state can no longer feed the next
        log_scale += shift*width
        start = end
    return state[-1]*np.exp(log_scale)
