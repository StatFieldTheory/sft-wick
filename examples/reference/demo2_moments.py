"""Independent moment equations for demo2, including separate cumulant channels.

For M independent replicas of the stationary OU noise eta, define
U = sum(eta)/sqrt(M), V = sum(eta**2-lambda)/sqrt(M), h = 1/sqrt(M).
The force U+alpha*V has the original exact covariance for every M;
its m-th cumulant scales as h**(m-2).  Thus the F**3*h coefficient of
<phi phi> is precisely FFFK (kappa3 times C), while F**3*h**3 is kappa5.
The h**0 coefficients give the Gaussian channels with exact C_eff.

At two spatial sites V is the centered sample Gram matrix, including
the off-diagonal product eta(x)*eta(y).  This closes the polynomial
Markov generator under spatial correlations.  Each physical component
has its own independent copy.  We solve coefficient equations directly,
without diagrams, Wick contractions, response phases, R/C tables or
numerical quadrature.  Initial noise moments follow from stationarity.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ito_moments import PolySDE, solve, unit

CHANNELS = {
    "o0": (0, 0), "ff": (2, 0), "fk": (1, 1), "ffk4": (2, 2),
    "ffff": (4, 0), "fffk": (3, 1), "fk5": (3, 3),
}
PAIRS = ((0, 0), (0, 1), (1, 1))


def moments(times, r=0.0, *, channels=("fffk",), lam=0.05, alpha=0.6,
            sigma_t=0.3, sigma_x=1.0, gamma=1.0):
    """Return {channel: (n_times, 3 pairs)} for fixed external times.

    phi starts at zero; eta starts in its stationary OU distribution.
    r=0 uses one site to reduce the size of the moment system.
    """
    if min(lam, sigma_t, sigma_x, gamma) <= 0 or r < 0:
        raise ValueError("positive scales and nonnegative r required")
    sites = 1 if r == 0 else 2
    gram = [(i, j) for i in range(sites) for j in range(i, sites)]
    noise_size = sites + len(gram)
    n_phi = 2 * sites
    dimension = n_phi + 2 * noise_size
    sde = PolySDE(dimension, 2)
    zero = (0, 0)
    h_tag = (0, 1)
    rho = 1 / sigma_t
    spatial = np.full((sites, sites), np.exp(-r / sigma_x))
    np.fill_diagonal(spatial, 1.0)
    drift = np.zeros((dimension, dimension))

    def phi(a, site):
        return 2 * site + a

    for a in (0, 1):
        start = n_phi + a * noise_size
        u = list(range(start, start + sites))
        v = {ij: start + sites + k for k, ij in enumerate(gram)}

        def gram_index(i, j):
            return v[tuple(sorted((i, j)))]

        def diffusion(i, j, coefficient, monomial=None, tag=zero):
            if coefficient:
                sde.diffusion.append((i, j, coefficient,
                                      (0,)*dimension if monomial is None else monomial, tag))

        for i in range(sites):
            p = phi(a, i)
            drift[p, p] = -gamma
            drift[p, u[i]] = 1
            drift[p, gram_index(i, i)] = alpha
            drift[u[i], u[i]] = -rho
            for j in range(sites):
                diffusion(u[i], u[j], 2*rho*lam*spatial[i, j])
            for j, k in gram:
                for left, right in ((u[i], v[j, k]), (v[j, k], u[i])):
                    diffusion(left, right, 2*rho*lam*spatial[i, j], unit(dimension, u[k]), h_tag)
                    diffusion(left, right, 2*rho*lam*spatial[i, k], unit(dimension, u[j]), h_tag)
        for i, j in gram:
            drift[v[i, j], v[i, j]] = -2*rho
            for k, l in gram:
                # Ito product rule for two entries of the sample Gram matrix.
                for p, q, s, t in ((i, k, j, l), (i, l, j, k),
                                    (j, k, i, l), (j, l, i, k)):
                    coefficient = 2*rho*lam*spatial[p, q]
                    diffusion(v[i, j], v[k, l], coefficient*lam*spatial[s, t])
                    diffusion(v[i, j], v[k, l], coefficient,
                              unit(dimension, gram_index(s, t)), h_tag)
    sde.add_linear_drift(drift)
    for site in range(sites):
        sde.drift.extend([
            (phi(0, site), 1.0, unit(dimension, phi(1, site), phi(1, site)), (1, 0)),
            (phi(1, site), 1.0, unit(dimension, phi(0, site), phi(1, site)), (1, 0)),
        ])

    @lru_cache(None)
    def stationary(monomial, h_order):
        if h_order < 0 or any(monomial[:n_phi]):
            return 0.0
        if not any(monomial):
            return float(h_order == 0)
        rate = -sum(drift[i, i]*power for i, power in enumerate(monomial))
        return sum(c*stationary(beta, h_order-tag[1])
                   for beta, c, tag in sde.generator(monomial)
                   if beta != monomial) / rate

    def initial(monomial, tag):
        return 0.0 if tag[0] else stationary(monomial, tag[1])

    targets = {(channel, pair): (unit(dimension, phi(pair[0], 0), phi(pair[1], sites-1)),
                                CHANNELS[channel])
               for channel in channels for pair in PAIRS}
    result = solve(sde, list(targets.values()), times, initial=initial)
    return {channel: np.stack([result[targets[channel, pair]] for pair in PAIRS], axis=1)
            for channel in channels}
