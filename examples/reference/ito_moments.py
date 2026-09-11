r"""Exact perturbative moments of a finite-dimensional polynomial Itô SDE.

An independent reference for sft-wick.  It imports nothing from the package
and uses no diagrams, no Wick contractions, no MSR response field and no R
or C propagator: only the generator of a Markov process acting on
polynomials.

The process ``X ∈ R^D`` starts from a given distribution at ``t = 0`` and
obeys

.. math::

    dX_i = f_i(X)\,dt + (\text{Gaussian white noise})_i + dJ_i ,

* ``f`` is a polynomial vector field, given as a list of monomial terms;
* the Gaussian part has a covariance rate ``S(X)`` that is itself a
  polynomial in ``X`` (constant for additive noise, ``(g_0 + g_1 X)^2`` for
  multiplicative noise under the Itô convention);
* ``J`` is a compensated compound-Poisson process: events at rate ``ν``
  carry a random jump vector, and the caller supplies its rate-weighted
  moments ``M_γ = ν E[J^γ]`` for multi-indices ``|γ| ≥ 2``.  Compensation
  removes the ``|γ| = 1`` terms, so the jumps have zero mean.

For a monomial ``X^α`` the generator is

.. math::

    L X^α = Σ_i α_i f_i(X) X^{α−e_i}
          + ½ Σ_{ij} S_{ij}(X) ∂_i ∂_j X^α
          + Σ_{γ ≤ α,\,|γ| ≥ 2} \binom{α}{γ} M_γ X^{α−γ},

so the moments ``m_α(t) = E[X^α]`` obey linear ODEs, ``ṁ_α = Σ_β
L_{αβ} m_β``.  They do not close: a quadratic drift raises the degree.  Each
term of the generator therefore carries a *tag*, a vector of non-negative
integers counting powers of bookkeeping parameters (for example ``ε`` for
the quadratic drift and ``μ`` for jump cumulants of order ``m ≥ 3``,
weighted ``μ^{m−2}``).  Writing ``m_α = Σ_K p^K m_α^{(K)}`` and matching
powers gives

.. math::

    ṁ_α^{(K)} = Σ_{\text{terms}} c\, m_β^{(K − \text{tag})},

a finite, block-triangular linear system for the coefficients a target
needs (every term either keeps the tag and does not raise the degree, or
lowers the tag).  :func:`solve` collects that system by breadth-first search
from the targets and integrates it exactly with
``scipy.sparse.linalg.expm_multiply``.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from math import comb
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

Monomial = tuple[int, ...]
Tag = tuple[int, ...]

__all__ = ["Monomial", "Tag", "PolySDE", "solve", "unit", "monomial_of"]


def unit(D: int, *indices: int) -> Monomial:
    """The monomial ``X_{i_1} X_{i_2} …`` as an exponent tuple."""
    out = [0] * D
    for i in indices:
        out[i] += 1
    return tuple(out)


def monomial_of(D: int, indices: Iterable[int]) -> Monomial:
    return unit(D, *indices)


def _add(a: Monomial, b: Monomial) -> Monomial:
    return tuple(x + y for x, y in zip(a, b))


def _sub(a: Monomial, b: Monomial) -> Monomial | None:
    out = tuple(x - y for x, y in zip(a, b))
    return out if min(out) >= 0 else None


def _tag_sub(a: Tag, b: Tag) -> Tag | None:
    out = tuple(x - y for x, y in zip(a, b))
    return out if min(out, default=0) >= 0 else None


@dataclass
class PolySDE:
    """A polynomial Itô SDE with jumps, in terms of its generator.

    Args:
        D: dimension of ``X``.
        n_tags: number of bookkeeping parameters.
        drift: terms ``(i, c, p, tag)``: ``f_i(X) ⊃ c X^p``.
        diffusion: terms ``(i, j, c, p, tag)``: ``S_ij(X) ⊃ c X^p``.  Give
            both ``(i, j)`` and ``(j, i)`` for an off-diagonal entry.
        jump_moment: ``γ ↦ M_γ = ν E[J^γ]`` for ``|γ| ≥ 2``, or ``None``.
        max_jump_order: largest ``|γ|`` kept.
        jump_tag: ``|γ| ↦ tag`` of that jump term.
    """

    D: int
    n_tags: int
    drift: list = field(default_factory=list)
    diffusion: list = field(default_factory=list)
    jump_moment: Callable[[Monomial], float] | None = None
    max_jump_order: int = 0
    jump_tag: Callable[[int], Tag] | None = None

    def __post_init__(self) -> None:
        self._jump_cache: dict[Monomial, float] = {}

    def zero_tag(self) -> Tag:
        return (0,) * self.n_tags

    # -- construction helpers -------------------------------------------

    def add_linear_drift(self, A: np.ndarray, tag: Tag | None = None) -> None:
        """``f(X) ⊃ A X``."""
        tag = self.zero_tag() if tag is None else tag
        for i, j in zip(*np.nonzero(A)):
            self.drift.append((int(i), float(A[i, j]), unit(self.D, j), tag))

    def add_quadratic_drift(self, Q: np.ndarray, tag: Tag) -> None:
        """``f_i(X) ⊃ Σ_jk Q_ijk X_j X_k``."""
        for i, j, k in zip(*np.nonzero(Q)):
            self.drift.append((int(i), float(Q[i, j, k]),
                               unit(self.D, j, k), tag))

    def add_constant_diffusion(self, S: np.ndarray,
                               tag: Tag | None = None) -> None:
        """Additive white noise with covariance rate ``S``."""
        tag = self.zero_tag() if tag is None else tag
        for i, j in zip(*np.nonzero(S)):
            self.diffusion.append((int(i), int(j), float(S[i, j]),
                                   (0,) * self.D, tag))

    # -- the generator ----------------------------------------------------

    def _jump(self, gamma: Monomial) -> float:
        val = self._jump_cache.get(gamma)
        if val is None:
            val = float(self.jump_moment(gamma))
            self._jump_cache[gamma] = val
        return val

    def generator(self, alpha: Monomial) -> list[tuple[Monomial, float, Tag]]:
        """``L X^α`` as ``(β, c, tag)`` terms: ``Σ c X^β`` at that tag."""
        out: list[tuple[Monomial, float, Tag]] = []
        D = self.D
        for i, c, p, tag in self.drift:
            if alpha[i] == 0:
                continue
            beta = _add(_sub(alpha, unit(D, i)), p)
            out.append((beta, c * alpha[i], tag))
        for i, j, c, p, tag in self.diffusion:
            if i == j:
                k = alpha[i] * (alpha[i] - 1)
            else:
                k = alpha[i] * alpha[j]
            if k == 0:
                continue
            beta = _add(_sub(alpha, unit(D, i, j)), p)
            out.append((beta, 0.5 * c * k, tag))
        if self.jump_moment is not None and self.max_jump_order >= 2:
            ranges = [range(a + 1) for a in alpha]
            for gamma in itertools.product(*ranges):
                order = sum(gamma)
                if order < 2 or order > self.max_jump_order:
                    continue
                coeff = 1.0
                for a, g in zip(alpha, gamma):
                    coeff *= comb(a, g)
                m = self._jump(tuple(gamma))
                if m == 0.0:
                    continue
                out.append((_sub(alpha, tuple(gamma)), coeff * m,
                            self.jump_tag(order)))
        return out


def solve(
    sde: PolySDE,
    targets: Sequence[tuple[Monomial, Tag]],
    times: Sequence[float],
    initial: Callable[[Monomial, Tag], float] | None = None,
) -> dict[tuple[Monomial, Tag], np.ndarray]:
    """Moment coefficients ``m_α^{(K)}(t)`` for every target ``(α, K)``.

    Args:
        sde: the process.
        targets: ``(α, K)`` pairs.
        times: output times ``t ≥ 0`` (increasing).
        initial: ``(α, K) ↦ m_α^{(K)}(0)``.  Default: ``X(0) = 0``, i.e. 1 for
            the constant monomial at the zero tag and 0 otherwise.

    Returns:
        ``{(α, K): array over times}``.
    """
    zero_tag = sde.zero_tag()
    zero_mono = (0,) * sde.D
    if initial is None:
        def initial(alpha, tag):  # noqa: ANN001
            return 1.0 if (alpha == zero_mono and tag == zero_tag) else 0.0

    index: dict[tuple[Monomial, Tag], int] = {}
    rows: list[list[tuple[tuple[Monomial, Tag], float]]] = []
    queue = list(dict.fromkeys(targets))
    while queue:
        node = queue.pop()
        if node in index:
            continue
        index[node] = len(rows)
        alpha, tag = node
        row = []
        for beta, c, dtag in sde.generator(alpha):
            src_tag = _tag_sub(tag, dtag)
            if src_tag is None:
                continue
            src = (beta, src_tag)
            row.append((src, c))
            if src not in index:
                queue.append(src)
        rows.append(row)

    n = len(rows)
    data, ri, ci = [], [], []
    for r, row in enumerate(rows):
        for src, c in row:
            data.append(c)
            ri.append(r)
            ci.append(index[src])
    M = csr_matrix((data, (ri, ci)), shape=(n, n))
    y0 = np.zeros(n)
    for node, k in index.items():
        y0[k] = initial(*node)

    times = np.asarray(times, dtype=float)
    out = np.empty((len(times), n))
    t_prev, y = 0.0, y0
    for m, t in enumerate(times):
        if t < t_prev:
            raise ValueError("times must be increasing")
        y = expm_multiply(M * (t - t_prev), y) if t > t_prev else y
        out[m] = y
        t_prev = t
    return {node: out[:, index[node]].copy() for node in targets}


def moments_from_cumulants(cumulant: Callable[[tuple[int, ...]], float],
                           indices: Sequence[int]) -> float:
    """``E[Π_k Y_{i_k}]`` from joint cumulants by the set-partition formula.

    ``cumulant`` receives a sorted tuple of variable indices (one entry per
    factor) and returns their joint cumulant.
    """
    idx = list(indices)
    total = 0.0
    for partition in _set_partitions(list(range(len(idx)))):
        term = 1.0
        for block in partition:
            term *= cumulant(tuple(sorted(idx[b] for b in block)))
            if term == 0.0:
                break
        total += term
    return total


def _set_partitions(items: list[int]):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _set_partitions(rest):
        for k in range(len(part)):
            yield part[:k] + [[first] + part[k]] + part[k + 1:]
        yield [[first]] + part


def as_mapping(result: Mapping) -> dict:
    return dict(result)
