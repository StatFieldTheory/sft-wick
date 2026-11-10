r"""Lowering of a :class:`~validation.modelspec.ModelSpec` onto the current
sft-wick release (the L1 API: ``System``, ``Expansion``, ``Propagators``).

This module, and ``lower_v042.py`` for the 0.4.2 release, are the only
modules of ``validation/`` that import sft_wick;
``validation/worker.py`` runs either in its own process.  A model becomes

* ``DiagonalA(gamma)`` for the response;
* ``LocalVertex`` ``h`` (rank 1), ``F`` and ``G`` for the drift;
* ``NonLocalVertex`` ``S{m}`` (static) and ``W{m}`` (``equal_time``) for the
  cumulants, and ``P{m}`` for each kept Poisson order, whose coupling is the
  raw Campbell cumulant (``equal_time`` for white pulses, a callable kinked
  where two leg times cross for exponential pulses);
* a ``MultiplicativeImpulse`` whose vertices are renamed ``D``, ``E``
  (``ψψφ``, ``ψψφφ``) and ``B``, ``L`` (the Stratonovich drift), so that no
  name clashes with the cubic drift ``G``; white noise beside it is folded
  into extra ``g0`` columns;
* the noise covariance for the record (the package probes it for the
  diagonal flags), and C from one of three sources: ``'oracle'``, the
  closed form of :mod:`validation.lowering_parts` for every model;
  ``'auto'``, the package's
  built-in closed form, for the models it covers; ``'table'``, the package's
  quadrature tables.

The closed form of C and the noise callables are in
:mod:`validation.lowering_parts`, shared with ``lower_v042.py``.

:func:`package_moments` evaluates an observable and sums the package's
diagrams by their tag over ``model.channels()``, read from the vertex
multiset of each diagram.

Not lowered: response (``psi``) legs, and white Poisson pulses beside
multiplicative noise (both raise ``NotImplementedError``).  The package
refuses multiplicative noise with legs at different positions.
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

import sft_wick as sw  # noqa: E402
from sft_wick.workflow.specs import (ConstantImpulse,  # noqa: E402
                                     CustomImpulse, CustomKernel,
                                     GeneralKappa2, MultiplicativeImpulse)

from .lowering_parts import (Kappa2Of,  # noqa: E402
                             PoissonCumulant, Sigma2Of, _DampedCosine, _Unit,
                             check_times, closed_form_c,
                             diagonal_c, observable_strings,
                             package_position, vertex_counts)
from .modelspec import ModelSpec, Observable, as_array  # noqa: E402
from .stages import stage as _stage  # noqa: E402

__all__ = ["to_system", "closed_form_c", "diagonal_c", "propagators",
           "Lowered", "package_moments", "MULTIPLICATIVE_NAMES"]

#: ``(vertex_names, drift_names)`` given to the MultiplicativeImpulse.
MULTIPLICATIVE_NAMES = (("D", "E"), ("B", "L"))
C_SOURCES = ("oracle", "auto", "table")


# ---------------------------------------------------------------------------
# The System
# ---------------------------------------------------------------------------

def _structured_kappa2(model: ModelSpec):
    """``SeparableTranslation``/``SeparableRotation`` when the coloured
    noise is ``δ_ab λ e^{−|Δt|/σ} K``, else ``None``."""
    N = model.n_components
    if len(model.ou) != N or model.poisson is not None:
        return None
    first = model.ou[0]
    for a, blk in enumerate(model.ou):
        if (blk.lam, blk.sigma_t, blk.spatial) != (first.lam, first.sigma_t,
                                                   first.spatial):
            return None
        if tuple(blk.loading) != tuple(np.eye(N)[a]):
            return None
    temporal = sw.ExponentialTemporal(lam=first.lam, sigma_t=first.sigma_t)
    kind, params = first.spatial.kind, first.spatial.params
    if kind == "legendre":
        return sw.SeparableRotation(temporal=temporal,
                                    angular=sw.LegendreAngular(
                                        coeffs=list(params)))
    spatial = {"exponential": lambda: sw.ExponentialSpatial(params[0]),
               "gaussian": lambda: sw.GaussianSpatial(params[0]),
               "damped_cosine": lambda: CustomKernel(
                   fn=_DampedCosine(*params)),
               "constant": lambda: CustomKernel(fn=_Unit())}.get(kind)
    if spatial is None:
        return None
    return sw.SeparableTranslation(temporal=temporal, spatial=spatial())


def _noise(model: ModelSpec, structured: bool) -> sw.GaussianNoise:
    kappa2 = _structured_kappa2(model) if structured else None
    if kappa2 is None:
        coloured = model.ou or (model.poisson is not None
                                and model.poisson.pulse == "exponential")
        kappa2 = (GeneralKappa2(fn=Kappa2Of(model)) if coloured else
                  sw.SeparableTranslation(
                      temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=1.0),
                      spatial=sw.GaussianSpatial(sigma_x=1.0)))
    mult, p = model.multiplicative, model.poisson
    white_poisson = p is not None and p.pulse == "white"
    if mult is not None:
        if white_poisson:
            raise NotImplementedError(
                "white Poisson pulses beside multiplicative noise do not fit "
                "the one sigma2 of a System")
        g0, g1 = as_array(mult.g0), as_array(mult.g1)
        if model.white is not None:
            vals, vecs = np.linalg.eigh(as_array(model.white))
            keep = vals > 1e-14 * max(1.0, float(vals.max()))
            extra = vecs[:, keep] * np.sqrt(vals[keep])
            g0 = np.concatenate([g0, extra], axis=1)
            g1 = np.concatenate([g1, np.zeros((g1.shape[0], extra.shape[1],
                                               g1.shape[2]))], axis=1)
        sigma2 = MultiplicativeImpulse(
            g0=g0, g1=g1, interpretation=mult.interpretation,
            vertex_names=MULTIPLICATIVE_NAMES[0],
            drift_names=MULTIPLICATIVE_NAMES[1])
    elif white_poisson:
        sigma2 = CustomImpulse(fn=Sigma2Of(model))
    elif model.white is not None:
        sigma2 = ConstantImpulse(as_array(model.white))
    else:
        sigma2 = None
    return sw.GaussianNoise(kappa2=kappa2, sigma2=sigma2)


def to_system(model: ModelSpec, structured: bool = False) -> sw.System:
    """The ``System`` of ``model``.  ``structured`` declares the coloured
    noise with the package's separable specs where it can, which the
    built-in closed form and the tables need."""
    vertices = []
    for name, value in (("h", model.source), ("F", model.quadratic),
                        ("G", model.cubic)):
        if value is not None:
            vertices.append(sw.LocalVertex(name, coupling=as_array(value)))
    nonlocal_vertices = [
        sw.NonLocalVertex(c.channel, order=c.order,
                          coupling=as_array(c.tensor),
                          equal_time=(c.time == "white"))
        for c in model.cumulants]
    p = model.poisson
    if p is not None:
        nonlocal_vertices += [
            sw.NonLocalVertex(f"P{m}", order=m, coupling=PoissonCumulant(
                model, m), coupling_vectorized=True,
                equal_time=(p.pulse == "white"))
            for m in p.orders]
    return sw.System(field=sw.FieldSpec("phi", model.n_components),
                     linear=sw.DiagonalA(gamma=list(model.gamma)),
                     vertices=vertices, nonlocal_vertices=nonlocal_vertices,
                     noise=_noise(model, structured), t_min=model.t_min)


def propagators(system: sw.System, model: ModelSpec, t_max: float,
                c_source: str = "oracle", diag_C: bool = False,
                n_grid_t: int = 41):
    """The propagators of ``system`` with C from ``c_source``."""
    if c_source == "oracle":
        return system.propagators(
            t_max=t_max, c_closed_form=closed_form_c(model),
            c_closed_form_only=True, c_closed_form_vectorized=True,
            diag_C=diag_C, progress=False)
    if c_source == "auto":
        return system.propagators(
            t_max=t_max, c_closed_form="auto", c_closed_form_only=True,
            c_closed_form_vectorized=True, diag_C=diag_C, progress=False)
    if c_source == "table":
        return system.propagators(
            t_max=t_max, n_grid_t=n_grid_t, c_closed_form=None,
            c_method="gauss_legendre", diag_C=diag_C, progress=False)
    raise ValueError(f"c_source must be one of {C_SOURCES}, got {c_source!r}")


# ---------------------------------------------------------------------------
# Evaluation by tag
# ---------------------------------------------------------------------------

def channel_weights(model: ModelSpec, system: sw.System) -> dict:
    """Vertex name → its tag over ``model.channels()``."""
    channels = model.channels()

    def unit(name: str, power: int = 1) -> tuple:
        return tuple(power if c == name else 0 for c in channels)

    weights = {name: unit(name) for name in channels if name != "g"}
    for v in system.multiplicative_vertices:
        weights[v.name] = unit("g", v.g1_power)
    return weights


class Lowered:
    """A model and an observable on the package: the expansion is built
    once and evaluated by any integrator.

    Args:
        model, obs: the pair to evaluate.
        max_weight: largest tag weight compared; every order up to it is
            expanded (a vertex carries weight ≥ 1).
        flags: ``System.expand`` flags; ``diag_C`` defaults to
            :func:`diagonal_c`, ``iso_C`` to False.
        c_source: ``'oracle'``, ``'auto'`` or ``'table'``.
    """

    def __init__(self, model: ModelSpec, obs: Observable, max_weight: int,
                 flags: dict | None = None, c_source: str = "oracle",
                 n_grid_t: int = 41):
        with _stage("lower"):
            if any(leg.op != "phi" for leg in obs.legs):
                raise NotImplementedError("response legs are not compared yet")
            check_times(model, obs)
            self.model, self.obs = model, obs
            self.max_weight = int(max_weight)
            self.flags = dict(diag_C=diagonal_c(model), iso_C=False)
            self.flags.update(flags or {})
            self.system = to_system(model, structured=(c_source != "oracle"))
        with _stage("expand"):
            self.expansion = self.system.expand(
                observable_strings(obs, model.n_components),
                orders=list(range(self.max_weight + 1)), target=obs.target,
                progress=False, **self.flags)
        with _stage("propagators"):
            self.props = propagators(self.system, model, obs.t_final + 0.25,
                                     c_source, self.flags["diag_C"], n_grid_t)
        self.weights = channel_weights(model, self.system)
        self.c_source = c_source

    def tag_of(self, order: int, idx: int) -> tuple:
        dt = self.expansion.diagrams(order)[idx]
        tag = [0] * len(self.model.channels())
        counts = vertex_counts(dt.coupling_sum)
        if sum(counts.values()) != order:
            raise RuntimeError(f"diagram {order}/{idx} has vertices {counts}")
        for name, k in counts.items():
            for c, w in enumerate(self.weights[name]):
                tag[c] += k * w
        return tuple(tag)

    def rows(self, method: str, orders=None, **settings) -> list:
        """``(tag, value, error)`` of every diagram of ``orders`` (default
        every order up to ``max_weight``) whose tag has weight ``≤
        max_weight``.

        A tag of weight ``k`` is complete in the diagrams of order ``k`` when
        every vertex carries weight 1.  A multiplicative vertex carries the
        power of ``g1`` (``E`` and ``L`` carry 2), so one tag collects
        diagrams of several orders, and ``orders`` must then be every order.
        """
        obs = self.obs
        over = set(obs.integrate_over) or None
        every = list(range(self.max_weight + 1))
        orders = every if orders is None else sorted(set(orders))
        if orders != every and "g" in self.model.channels():
            raise ValueError("a subset of orders gives incomplete tags of the "
                             "g channel; evaluate every order")
        with _stage("evaluate"):
            res = self.expansion.evaluate(
                self.props,
                positions={lab: package_position(p)
                           for lab, p in obs.positions},
                t_final=obs.t_final,
                component_pair=tuple(leg.comp for leg in obs.legs),
                orders=orders,
                external_times=dict(obs.times) or None, integrate_over=over,
                method=method, **settings)
        out = []
        for row in res.per_diagram:
            tag = self.tag_of(row["order"], row["diagram_idx"])
            if sum(tag) <= self.max_weight:
                out.append((tag, float(np.real(row["value"])),
                            float(row.get("error") or 0.0)))
        return out

    def evaluate(self, method: str, orders=None, **settings) -> dict:
        """``{tag: (value, error)}``: :meth:`rows` summed by tag, the errors
        in quadrature."""
        value: dict = defaultdict(float)
        var: dict = defaultdict(float)
        for tag, v, e in self.rows(method, orders, **settings):
            value[tag] += v
            var[tag] += e ** 2
        return {t: (value[t], float(np.sqrt(var[t]))) for t in value}


def package_moments(model: ModelSpec, obs: Observable, max_weight: int,
                    method: str = "gauss_legendre", flags: dict | None = None,
                    c_source: str = "oracle", **settings) -> dict:
    """``{tag: (value, error)}``: the package's diagrams summed by tag."""
    return Lowered(model, obs, max_weight, flags, c_source).evaluate(
        method, **settings)
