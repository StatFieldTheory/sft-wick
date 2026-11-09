r"""Exact moments of a :class:`~validation.modelspec.ModelSpec`, channel by
channel, from the moment hierarchy of its Markov embedding.

R is local in space and every vertex acts at one point, so the fields at a
finite set of observation points, together with the states of the noise,
form a finite-dimensional Markov process.  Per observation point ``i`` (legs
at equal positions share one point) the states are

* ``φ_{a,i}``, from 0;
* ``ζ_{k,i}`` for each OU block: ``dζ = −ζ/σ_k dt + dB`` with
  ``⟨dB_{k,i} dB_{k,j}⟩ = (2λ_k/σ_k) K_k(x_i, x_j) dt``, started in its
  stationary law;
* ``η_{a,i}`` for exponential Poisson pulses: ``dη = −η/τ_a dt`` plus the
  jumps, started in its stationary law (Campbell's cumulants);
* ``I_{a,i} = ∫ φ_{a,i} dt`` for each integrated leg, from 0;

and, shared by every point, ``Z_a`` with ``dZ = 0`` when the model has
static cumulants.  The generator is the sum of independent parts: drift and
constant diffusion (``ito_moments.PolySDE``), the multiplicative columns
(``hormander_moments.VectorFieldSDE``, Itô or Stratonovich), and one
jumps-only ``PolySDE`` each for the Poisson events and the white cumulants.
Every term carries a tag over :meth:`ModelSpec.channels`, so the
coefficient at a tag is the sum of the diagrams of that channel content.

The time axis starts at 0 here and at ``t_min`` in the model.  The initial
law of the noise states is evaluated by memoised recursion over the block
that holds the first factor (Isserlis' theorem for the Gaussian states), not
by enumerating set partitions.  Imports no sft-wick code.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

_REFERENCE = Path(__file__).resolve().parents[1] / "examples" / "reference"
if str(_REFERENCE) not in sys.path:
    sys.path.insert(0, str(_REFERENCE))
import hormander_moments as hm  # noqa: E402
import ito_moments as im  # noqa: E402

from .modelspec import ModelSpec, Observable, as_array  # noqa: E402

__all__ = ["Embedding", "moments", "tags_up_to", "check_terminates"]

Tag = tuple


def tags_up_to(model: ModelSpec, weight: int) -> list:
    """Every tag over ``model.channels()`` whose entries sum to ``≤ weight``."""
    n = len(model.channels())
    return [t for t in itertools.product(range(weight + 1), repeat=n)
            if sum(t) <= weight]


def _tag_sub(a: Tag, b: Tag) -> Tag | None:
    out = tuple(x - y for x, y in zip(a, b))
    return out if min(out, default=0) >= 0 else None


def _symmetrised(tensor: np.ndarray, comps: tuple) -> float:
    """The tensor symmetrised over its legs, at the index tuple ``comps``."""
    perms = set(itertools.permutations(comps))
    return float(sum(tensor[p] for p in perms) / len(perms))


def _overlap(widths: Sequence[float], xs: np.ndarray,
             powers: Sequence[int]) -> float:
    r"""``∫ d^dz Π_j e^{−k_j |x_j − z|²/(2 s_j²)}`` in closed form."""
    w = np.array([k / s ** 2 for k, s in zip(powers, widths)])
    total = w.sum()
    mean = (w[:, None] * xs).sum(axis=0) / total
    spread = float((w * ((xs - mean) ** 2).sum(axis=1)).sum())
    d = xs.shape[1]
    return (2.0 * math.pi / total) ** (d / 2.0) * math.exp(-0.5 * spread)


class _Family:
    """Independent initial-law block: joint moments of its states from
    their joint cumulants, by the recursion on the block holding the first
    factor, ``E[Y_1 Y_S] = Σ_{B ∋ 1} κ(B) E[Y_{S∖B}]`` with tags."""

    def __init__(self, states: Iterable[int], sizes: Iterable[int],
                 cumulant: Callable[[tuple], float],
                 tag: Callable[[int], Tag], channels: Iterable[int]):
        self.states = frozenset(states)
        self.sizes = tuple(sorted(set(sizes)))
        self.cumulant = cumulant
        self.tag = tag
        self.channels = frozenset(channels)
        self._memo: dict = {}

    def project(self, tag: Tag) -> Tag:
        return tuple(x if k in self.channels else 0 for k, x in enumerate(tag))

    def moment(self, legs: tuple, tag: Tag) -> float:
        if not legs:
            return 0.0 if any(tag) else 1.0
        key = (legs, tag)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        first, rest = legs[0], legs[1:]
        total = 0.0
        for m in self.sizes:
            if m - 1 > len(rest):
                break
            sub = _tag_sub(tag, self.tag(m))
            if sub is None:
                continue
            for pick in itertools.combinations(range(len(rest)), m - 1):
                c = self.cumulant((first,) + tuple(rest[j] for j in pick))
                if c != 0.0:
                    left = tuple(x for j, x in enumerate(rest)
                                 if j not in pick)
                    total += c * self.moment(left, sub)
        self._memo[key] = total
        return total


class Embedding:
    """The Markov embedding of ``model`` at ``points``.

    Args:
        model: the model.
        points: observation positions (tuples of floats).
        integrated: ``(a, i)`` pairs whose time integral ``I_{a,i}`` is a
            state.

    The object is what ``ito_moments.solve`` and ``hormander_moments.two_time``
    take: it has ``D``, ``n_tags``, ``zero_tag()`` and ``generator()``.
    """

    def __init__(self, model: ModelSpec, points: Sequence[tuple],
                 integrated: Iterable[tuple] = ()):
        self.model = model
        self.points = tuple(tuple(float(v) for v in p) for p in points)
        self.channels = model.channels()
        self.n_tags = len(self.channels)
        N, P = model.n_components, len(self.points)
        self.N, self.P = N, P
        count = itertools.count()
        self.phi = {(a, i): next(count) for a in range(N) for i in range(P)}
        self.ou = {(b, i): next(count) for b in range(len(model.ou))
                   for i in range(P)}
        coloured = model.poisson is not None and model.poisson.pulse == \
            "exponential"
        self.eta = ({(a, i): next(count) for a in range(N) for i in range(P)}
                    if coloured else {})
        static = any(c.time == "static" for c in model.cumulants)
        self.z = {a: next(count) for a in range(N)} if static else {}
        self.integ = {key: next(count) for key in sorted(set(integrated))}
        self.D = next(count)
        self._kernels = [np.array([[blk.spatial(x, y) for y in self.points]
                                   for x in self.points])
                         for blk in model.ou]
        self._parts = [p for p in (self._core(), self._multiplicative(),
                                   self._poisson(), self._white_cumulants())
                       if p is not None]
        self._families = self._initial_law()
        self._cache: dict = {}
        check_terminates(self)

    # -- tags ----------------------------------------------------------------

    def zero_tag(self) -> Tag:
        return (0,) * self.n_tags

    def unit(self, channel: str) -> Tag:
        tag = [0] * self.n_tags
        tag[self.channels.index(channel)] = 1
        return tuple(tag)

    # -- the generator ---------------------------------------------------------

    def generator(self, alpha: tuple) -> list:
        hit = self._cache.get(alpha)
        if hit is None:
            hit = [term for part in self._parts
                   for term in part.generator(alpha)]
            self._cache[alpha] = hit
        return hit

    def _local_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """A drift tensor ``T_{a b …}`` acting at every point separately."""
        rank = tensor.ndim
        out = np.zeros((self.D,) * rank)
        for i in range(self.P):
            for idx in itertools.product(range(self.N), repeat=rank):
                if tensor[idx] != 0.0:
                    out[tuple(self.phi[(a, i)] for a in idx)] += tensor[idx]
        return out

    def _core(self) -> im.PolySDE:
        m, D = self.model, self.D
        sde = im.PolySDE(D=D, n_tags=self.n_tags)
        gamma = as_array(m.gamma)
        A = np.zeros((D, D))
        for (a, i), k in self.phi.items():
            A[k, k] = -gamma[a]
            for b, blk in enumerate(m.ou):
                A[k, self.ou[(b, i)]] += blk.loading[a]
            if self.eta:
                A[k, self.eta[(a, i)]] += 1.0
            if self.z:
                A[k, self.z[a]] += 1.0
        for (b, _), k in self.ou.items():
            A[k, k] = -1.0 / m.ou[b].sigma_t
        for (a, _), k in self.eta.items():
            A[k, k] = -1.0 / m.poisson.tau[a]
        for (a, i), k in self.integ.items():
            A[k, self.phi[(a, i)]] += 1.0
        sde.add_linear_drift(A)
        if m.source is not None:
            tag = self.unit("h")
            for (a, _), k in self.phi.items():
                if m.source[a] != 0.0:
                    sde.drift.append((k, float(m.source[a]), (0,) * D, tag))
        if m.quadratic is not None:
            sde.add_quadratic_drift(self._local_tensor(as_array(m.quadratic)),
                                    tag=self.unit("F"))
        if m.cubic is not None:
            sde.add_cubic_drift(self._local_tensor(as_array(m.cubic)),
                                tag=self.unit("G"))
        S = np.zeros((D, D))
        for (b, i), k in self.ou.items():
            blk = m.ou[b]
            for j in range(self.P):
                S[k, self.ou[(b, j)]] = (2.0 * blk.lam / blk.sigma_t
                                         * self._kernels[b][i, j])
        if m.white is not None:
            W = as_array(m.white)
            for (a, _), k in self.phi.items():
                for (b, _), q in self.phi.items():
                    S[k, q] += W[a, b]
        sde.add_constant_diffusion(S)
        return sde

    def _multiplicative(self) -> hm.VectorFieldSDE | None:
        mult = self.model.multiplicative
        if mult is None:
            return None
        g0, g1 = as_array(mult.g0), as_array(mult.g1)
        N, M = g0.shape
        sde = hm.VectorFieldSDE(D=self.D, n_tags=self.n_tags,
                                interpretation=mult.interpretation)
        coords = [self.phi[(a, i)] for i in range(self.P) for a in range(N)]
        g1_tag = self.unit("g") if mult.has_g1 else self.zero_tag()
        for k in range(M):
            col0 = np.tile(g0[:, k], self.P)
            col1 = np.kron(np.eye(self.P), g1[:, k, :])
            sde.columns.append(hm.affine_column(self.D, self.n_tags, coords,
                                                col0, col1, g1_tag))
        return sde

    def _poisson(self) -> im.PolySDE | None:
        p = self.model.poisson
        if p is None:
            return None
        jumped = {k: key for key, k in (self.eta or self.phi).items()}
        tags = {2: self.zero_tag(), **{m: self.unit(f"P{m}")
                                       for m in p.orders}}

        def moment(gamma: tuple) -> float:
            if sum(gamma) not in tags:
                return 0.0
            comps, xs, powers = [], [], []
            for k, g in enumerate(gamma):
                if g == 0:
                    continue
                key = jumped.get(k)
                if key is None:
                    return 0.0
                comps.append(key[0])
                xs.append(self.points[key[1]])
                powers.append(g)
            amp = p.nu * math.prod(p.amplitude[a] ** g
                                   for a, g in zip(comps, powers))
            return amp * _overlap([p.width[a] for a in comps],
                                  np.array(xs, dtype=float), powers)

        return im.PolySDE(D=self.D, n_tags=self.n_tags, jump_moment=moment,
                          max_jump_order=max(tags), jump_tag=tags.__getitem__)

    def _white_cumulants(self) -> im.PolySDE | None:
        white = {c.order: as_array(c.tensor) for c in self.model.cumulants
                 if c.time == "white"}
        if not white:
            return None
        comp_of = {k: a for (a, _), k in self.phi.items()}
        tags = {m: self.unit(f"W{m}") for m in white}

        def moment(gamma: tuple) -> float:
            K = white.get(sum(gamma))
            if K is None:
                return 0.0
            comps = []
            for k, g in enumerate(gamma):
                if g == 0:
                    continue
                if k not in comp_of:
                    return 0.0
                comps += [comp_of[k]] * g
            return _symmetrised(K, tuple(comps))

        return im.PolySDE(D=self.D, n_tags=self.n_tags, jump_moment=moment,
                          max_jump_order=max(white), jump_tag=tags.__getitem__)

    # -- the initial law ---------------------------------------------------------

    def _channel_index(self, prefix: str) -> list:
        return [k for k, name in enumerate(self.channels)
                if name[0] == prefix and name[1:].isdigit()]

    def _initial_law(self) -> list:
        m, out = self.model, []
        if self.ou:
            block_of = {k: key for key, k in self.ou.items()}

            def gaussian(block: tuple) -> float:
                (b1, i1), (b2, i2) = block_of[block[0]], block_of[block[1]]
                if b1 != b2:
                    return 0.0
                return m.ou[b1].lam * self._kernels[b1][i1, i2]

            out.append(_Family(block_of, (2,), gaussian,
                               lambda size: self.zero_tag(), ()))
        if self.eta:
            p = m.poisson
            key_of = {k: key for key, k in self.eta.items()}

            def campbell(block: tuple) -> float:
                comps = [key_of[s][0] for s in block]
                xs = np.array([self.points[key_of[s][1]] for s in block])
                amp = p.nu * math.prod(p.amplitude[a] for a in comps)
                return (amp * _overlap([p.width[a] for a in comps], xs,
                                       [1] * len(comps))
                        / sum(1.0 / p.tau[a] for a in comps))

            sizes = (2, *p.orders)
            out.append(_Family(key_of, sizes, campbell,
                               lambda size: (self.zero_tag() if size == 2
                                             else self.unit(f"P{size}")),
                               self._channel_index("P")))
        if self.z:
            static = {c.order: as_array(c.tensor) for c in m.cumulants
                      if c.time == "static"}
            comp_of = {k: a for a, k in self.z.items()}

            def cumulant(block: tuple) -> float:
                return _symmetrised(static[len(block)],
                                    tuple(comp_of[s] for s in block))

            out.append(_Family(comp_of, static, cumulant,
                               lambda size: self.unit(f"S{size}"),
                               self._channel_index("S")))
        self._family_of = {s: f for f, fam in enumerate(out)
                           for s in fam.states}
        return out

    def initial(self, alpha: tuple, tag: Tag) -> float:
        """``E[X^α]`` at ``t = 0`` and at tag ``tag``."""
        legs: list = [[] for _ in self._families]
        for k, n in enumerate(alpha):
            if n:
                f = self._family_of.get(k)
                if f is None:
                    return 0.0
                legs[f] += [k] * n
        left = tag
        value = 1.0
        for fam, fam_legs in zip(self._families, legs):
            part = fam.project(tag)
            value *= fam.moment(tuple(fam_legs), part)
            if value == 0.0:
                return 0.0
            left = _tag_sub(left, part)
        return value if not any(left) else 0.0

    # -- queries ---------------------------------------------------------------

    def raw_moment(self, legs: Sequence[tuple], tags: Sequence[Tag]) -> dict:
        """``{tag: E[Π X_s(t)]}`` for legs ``(state, t)``, ``t ≥ 0``, at one
        or two distinct times."""
        times = sorted({t for _, t in legs})
        if len(times) == 1:
            mono = im.monomial_of(self.D, [s for s, _ in legs])
            out = im.solve(self, [(mono, t) for t in tags], [times[0]],
                           initial=self.initial)
            return {t: float(out[(mono, t)][0]) for t in tags}
        if len(times) == 2:
            early, late = times
            first = im.monomial_of(self.D, [s for s, t in legs if t == early])
            second = im.monomial_of(self.D, [s for s, t in legs if t == late])
            return hm.two_time(self, first, second, list(tags), early, late,
                               initial=self.initial)
        raise NotImplementedError("three or more distinct external times")

    def joint_cumulant(self, legs: Sequence[tuple],
                       tags: Sequence[Tag]) -> dict:
        """The joint cumulant of the legs, from the raw moments of every
        subset: ``m(S) = Σ_{B ∋ first} κ(B) m(S∖B)``, with the tags of the
        two factors adding.

        The subtraction leaves a residue of a few ulp of the raw moment at
        the same tag where the cumulant is exactly 0, so compare a zero
        against the raw moment's scale, not against 0.
        """
        n = len(legs)
        grid = sorted({low for t in tags
                       for low in itertools.product(*(range(k + 1)
                                                      for k in t))})
        raw = {mask: self.raw_moment([legs[j] for j in range(n)
                                      if mask >> j & 1], grid)
               for mask in range(1, 1 << n)}
        cum: dict = {}
        for mask in sorted(raw, key=lambda x: bin(x).count("1")):
            first = mask & -mask
            value = dict(raw[mask])
            rest = mask ^ first
            sub = rest
            while True:
                block = sub | first
                if block != mask:
                    _subtract_product(value, cum[block], raw[mask ^ block],
                                      grid)
                if sub == 0:
                    break
                sub = (sub - 1) & rest
            cum[mask] = value
        return {tuple(t): cum[(1 << n) - 1][tuple(t)] for t in tags}


def _subtract_product(out: dict, p: dict, q: dict, grid: list) -> None:
    """``out −= p · q`` as polynomials in the tags, truncated to ``grid``."""
    for K in grid:
        total = 0.0
        for K1 in itertools.product(*(range(k + 1) for k in K)):
            total += p[K1] * q[tuple(a - b for a, b in zip(K, K1))]
        out[K] -= total


def check_terminates(emb: Embedding) -> None:
    """Refuse a generator that ``ito_moments.solve`` could not finish.

    ``solve`` follows a term at the zero tag that raises the degree for
    ever, and ``zip`` silently truncates a tag of the wrong length.
    """
    n = emb.n_tags

    def check(tag: Tag, degree_raise: int, what: str) -> None:
        if len(tag) != n or min(tag, default=0) < 0:
            raise ValueError(f"{what}: tag {tag} does not have {n} entries")
        if degree_raise > 0 and not any(tag):
            raise ValueError(f"{what} raises the degree at the zero tag; "
                             "the hierarchy would not close")

    for part in emb._parts:
        for i, _, p, tag in getattr(part, "drift", ()):
            check(tag, sum(p) - 1, f"drift term on state {i}")
        for i, _, _, p, tag in getattr(part, "diffusion", ()):
            check(tag, sum(p) - 2, f"diffusion term on state {i}")
        for column in getattr(part, "columns", ()):
            for i, poly in column.items():
                for (mono, tag), _ in poly.items():
                    check(tag, sum(mono) - 1, f"noise column entry {i}")


def _points(obs: Observable) -> tuple:
    """Distinct positions of the legs, and the point index of each label."""
    points: list = []
    point_of = {}
    for leg in obs.legs:
        pos = obs.position(leg.label)
        if pos not in points:
            points.append(pos)
        point_of[leg.label] = points.index(pos)
    return points, point_of


def moments(model: ModelSpec, obs: Observable,
            tags: Sequence[Tag]) -> dict:
    """The coefficients of ``obs`` at each tag over ``model.channels()``."""
    tags = [tuple(int(x) for x in t) for t in tags]
    n_tags = len(model.channels())
    if any(len(t) != n_tags or min(t, default=0) < 0 for t in tags):
        raise ValueError(f"tags have {n_tags} non-negative entries "
                         f"{model.channels()}")
    if obs.t_final < model.t_min:
        raise ValueError("t_final is before t_min")
    for leg in obs.legs:
        if leg.comp >= model.n_components:
            raise ValueError(f"component {leg.comp} of a model with "
                             f"N = {model.n_components}")
        if leg.op != "phi":
            raise NotImplementedError("response (psi) legs are not "
                                      "compiled yet")
        if obs.time(leg.label) < model.t_min:
            raise ValueError(f"time of {leg.label!r} is before t_min")
    points, point_of = _points(obs)
    over = set(obs.integrate_over)
    emb = Embedding(model, points,
                    {(leg.comp, point_of[leg.label]) for leg in obs.legs
                     if leg.label in over})
    legs = []
    for leg in obs.legs:
        i = point_of[leg.label]
        if leg.label in over:
            legs.append((emb.integ[(leg.comp, i)], obs.t_final - model.t_min))
        else:
            legs.append((emb.phi[(leg.comp, i)],
                         obs.time(leg.label) - model.t_min))
    if obs.target == "connected":
        return emb.joint_cumulant(legs, tags)
    return emb.raw_moment(legs, tags)
