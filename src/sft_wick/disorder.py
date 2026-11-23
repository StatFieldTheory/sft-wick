"""Gaussian couplings fixed over a trajectory and shared between replicas."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from itertools import product
from math import isfinite
from numbers import Integral, Real
from typing import Iterable

import numpy as np

from ._contraction_admission import DEFAULT_CONTRACTION_MAX_BYTES
from .expressions import KroneckerDelta, Product, Rational, Sum, Symbol
from .workflow import (
    ConstantImpulse,
    DiagonalA,
    ExponentialSpatial,
    ExponentialTemporal,
    FieldSpec,
    GaussianNoise,
    GroupedVertex,
    LocalVertex,
    SeparableTranslation,
    System,
)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class RandomCoupling:
    """A static Gaussian tensor acting as a random polynomial drift.

    The output is the first index. For an ordered rank-r tensor the iid
    entries have variance variance / n_sites**(r-1). The GOE matrix adds
    the transpose pairing with the same weight. Copies see the same tensor.
    """

    n_sites: int
    variance: float = 1.0
    ensemble: str = "iid"
    rank: int = 2

    def __post_init__(self):
        object.__setattr__(self, "n_sites", _integer(self.n_sites, "n_sites", 1))
        object.__setattr__(self, "rank", _integer(self.rank, "rank", 2))
        object.__setattr__(self, "variance", _real(self.variance, "variance"))
        if self.variance < 0:
            raise ValueError("variance must be nonnegative")
        if self.rank > 32:
            raise ValueError("a grouped coupling supports at most 64 slots")
        if self.ensemble not in ("iid", "goe") or (
            self.ensemble == "goe" and self.rank != 2
        ):
            raise ValueError("use iid for an ordered tensor, or goe for a matrix")

    @property
    def entry_variance(self) -> float:
        """Weight of the direct pairing of two disorder tensors."""
        return self.variance / self.n_sites ** (self.rank - 1)

    def vertex(self, *, copies: int = 1, name: str = "J") -> GroupedVertex:
        """The averaged two-time vertex for use in a general Wick System.

        Components are ordered as copy*n_sites + site. Supply independent
        thermal noise across the copies in the free system.
        """
        copies = _integer(copies, "copies", 1)
        half = ("psi",) + ("phi",) * (self.rank - 1)
        return _RandomVertex(
            name,
            half * 2,
            _EntryCovariance(self, copies),
            (0,) * self.rank + (1,) * self.rank,
        )

    def coefficient(
        self,
        observations: Iterable[tuple[int, int, float]],
        *,
        pairs: int,
        cubic: int = 0,
        copies: int = 1,
        decay: float = 1.0,
        noise: float = 1.0,
        t_min: float = 0.0,
        connected: bool = False,
        n_gauss: int = 8,
    ) -> float:
        """Coefficient of g**(2*pairs) u**cubic for isotropic white-noise sites.

        Each observation is (copy, site, time). The drift consists of
        -decay*x + g*J[x,...,x] - u*x**3, with empty initial data and
        independent thermal white noises of covariance noise per copy and
        site. All copies share J. With connected=True return the cumulant.
        """
        pairs = _integer(pairs, "pairs")
        cubic = _integer(cubic, "cubic")
        copies = _integer(copies, "copies", 1)
        n_gauss = _integer(n_gauss, "n_gauss", 1)
        decay, noise, t_min = (
            _real(decay, "decay"),
            _real(noise, "noise"),
            _real(t_min, "t_min"),
        )
        if decay <= 0 or noise < 0:
            raise ValueError("decay must be positive and noise nonnegative")
        if not isinstance(connected, (bool, np.bool_)):
            raise ValueError("connected must be boolean")
        sites, clocks = [], {}
        for i, (copy, site, time) in enumerate(observations):
            copy, site = _integer(copy, "copy"), _integer(site, "site")
            time = _real(time, "time")
            if copy >= copies or site >= self.n_sites or time < t_min:
                raise ValueError("observation outside its copy, site or time domain")
            sites.append((copy, site))
            clocks[f"x{i}"] = time
        if not sites:
            raise ValueError("supply at least one observation")
        if not noise or (pairs and not self.variance) or t_min in clocks.values():
            return 0.0
        order = pairs + cubic
        target = "connected" if connected else "moment"
        expansion = _indexed(self.rank, len(sites), pairs, cubic, target)
        fixed = {f"a{i}": value for i, value in enumerate(sites)}
        scalar = _system(1, self.rank, bool(pairs), bool(cubic), decay, noise, t_min)
        terms = []
        for term in expansion.diagrams(order):
            names = [template[0] for _, template in term.vertex_types]
            if names.count("J") != pairs or names.count("U") != cubic:
                continue
            reduced = _contract(term, self, copies, fixed)
            if reduced is not None:
                terms.append(reduced)
        if not terms:
            return 0.0
        end = max(clocks.values())
        scalar_expansion = replace(
            expansion, system=scalar, dts_by_order={order: terms}
        )
        result = scalar_expansion.evaluate(
            scalar.propagators(t_max=end, progress=False),
            positions=dict.fromkeys(clocks, 0.0),
            external_times=clocks,
            t_final=end,
            method="gauss_legendre",
            n_gauss=n_gauss,
        )
        return result.total


@dataclass(frozen=True)
class _EntryCovariance:
    law: RandomCoupling
    copies: int

    def array(self, phase=1.0):
        law, copies = self.law, self.copies
        n, rank = law.n_sites, law.rank
        d = n * copies
        if 8 * d ** (2 * rank) + 8 * rank * n**rank > DEFAULT_CONTRACTION_MAX_BYTES:
            raise ValueError(
                "dense disorder covariance exceeds workspace; use coefficient for isotropic sites"
            )
        output = np.zeros((d,) * (2 * rank))
        indices = np.indices((n,) * rank, sparse=True)
        for c, h in product(range(copies), repeat=2):
            left = tuple(c * n + i for i in indices)
            right = tuple(h * n + i for i in indices)
            output[left + right] = phase * law.entry_variance
            if law.ensemble == "goe":
                output[left + right[::-1]] += phase * law.entry_variance
        return output

    def __call__(self, positions, times):
        return self.array()


class _RandomVertex(GroupedVertex):
    def shape_problem(self, n_components):
        required = self.coupling.law.n_sites * self.coupling.copies
        if n_components != required:
            return f"the shared disorder requires {required} components; the field has {n_components}"
        return None

    def build_msr_coupling(self, n_components):
        return self.coupling.array(self.msr_factor.real)


def _system(n, rank, disorder, cubic, decay, noise, t_min):
    # Enumeration and time integration use separate small templates. Both
    # retain the two clocks, action phase, and the ordered tensor slots.
    noise_spec = GaussianNoise(
        SeparableTranslation(ExponentialTemporal(0, 1), ExponentialSpatial(1)),
        ConstantImpulse(noise),
    )
    return System(
        FieldSpec("phi", n),
        DiagonalA([decay]),
        noise_spec,
        t_min=t_min,
        grouped_vertices=(
            (RandomCoupling(n, rank=rank, variance=float(n ** (rank - 1))).vertex(),)
            if disorder
            else ()
        ),
        vertices=(LocalVertex("U", -np.ones((n,) * 4)),) if cubic else (),
    )


@lru_cache(maxsize=32)
def _indexed(rank, count, pairs, cubic, target):
    # Two template components keep symbolic indices alive; every free
    # index is counted below at the physical dimension before integration.
    model = _system(2, rank, bool(pairs), bool(cubic), 1.0, 1.0, 0.0)
    return model.expand(
        [f"phi_a{i}(x{i})" for i in range(count)],
        orders=[pairs + cubic],
        routing="multiplicity",
        target=target,
        progress=False,
    )


def _alternatives(expr, law):
    """Expand the coupling polynomial into site and copy index equalities."""
    if isinstance(expr, Rational):
        yield Fraction(expr.numerator, expr.denominator), (), ()
    elif isinstance(expr, KroneckerDelta):
        edge = ((expr.index1, expr.index2),)
        yield 1, edge, edge
    elif isinstance(expr, Symbol):
        ix = expr.indices
        if expr.name == "U":
            equal = tuple(zip(ix, ix[1:]))
            yield 1, equal, equal
        else:
            left, right = ix[: law.rank], ix[law.rank :]
            within = tuple(zip(left, left[1:])) + tuple(zip(right, right[1:]))
            yield 1, tuple(zip(left, right)), within
            if law.ensemble == "goe":
                yield 1, tuple(zip(left, reversed(right))), within
    elif isinstance(expr, Sum):
        for part in expr.terms:
            yield from _alternatives(part, law)
    elif isinstance(expr, Product):
        for terms in product(*(_alternatives(part, law) for part in expr.factors)):
            scale = Fraction(1)
            sites, copies = [], []
            for value, s, c in terms:
                scale *= value
                sites.extend(s)
                copies.extend(c)
            yield scale, sites, copies
    else:
        raise ValueError(
            f"unsupported disorder coupling expression {type(expr).__name__}"
        )


def _assignments(labels, edges, fixed, dimension):
    parent = {k: k for k in labels}

    def root(k):
        while k != parent[k]:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for a, b in edges:
        parent[root(a)] = root(b)
    pinned = {}
    for k, value in fixed.items():
        rep = root(k)
        if pinned.setdefault(rep, value) != value:
            return 0
    return dimension ** (len({root(k) for k in labels}) - len(pinned))


def _contract(term, law, copies, fixed):
    labels = set(fixed).union(k for k, _ in term.summation_indices)
    sites = {k: value[1] for k, value in fixed.items()}
    replica = {k: value[0] for k, value in fixed.items()}
    count = sum(
        scale
        * _assignments(labels, s, sites, law.n_sites)
        * _assignments(labels, c, replica, copies)
        for scale, s, c in _alternatives(term.coupling_sum, law)
    )
    if not count:
        return None
    # The phase and factorial are still supplied by the scalar System's
    # vertices. The colour factor retains only sums over component labels.
    coeff = (
        Fraction(term.rational_prefactor.numerator, term.rational_prefactor.denominator)
        * count
    )
    # Normalize the free index loops before converting diagram weights to
    # floating point. In particular a large N can cancel 1/N exactly.
    covariance = Fraction.from_float(law.variance) / law.n_sites ** (law.rank - 1)
    coeff *= covariance ** sum(tpl[0] == "J" for _, tpl in term.vertex_types)
    factors = [
        Symbol(template[0], (), points, local=template[1])
        for points, template in term.vertex_types
    ]
    return replace(
        term,
        propagators=tuple(
            replace(p, index_left=None, index_right=None) for p in term.propagators
        ),
        summation_indices=(),
        coupling_sum=Product(factors) if factors else Rational(1),
        rational_prefactor=Rational(coeff.numerator, coeff.denominator),
    )
