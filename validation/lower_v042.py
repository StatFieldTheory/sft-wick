r"""Lowering of a :class:`~validation.modelspec.ModelSpec` onto sft-wick 0.4.2.

The 0.4.2 release and the current one share the package name, so this
module runs only in a ``python -m validation.worker --lowering
validation.lower_v042`` process with the unpacked 0.4.2 wheel first on the
path.  It never imports ``validation/lower.py``, which needs the current
API.  The lowering is that of ``lower.py`` restricted to what 0.4.2 has:

* no ``MultiplicativeImpulse``, so a model with multiplicative noise is
  refused;
* no ``target`` in ``System.expand``, so a joint cumulant is refused;
* the rest (``DiagonalA``, static local vertices, static and ``equal_time``
  non-local vertices, callable Poisson cumulants, ``GeneralKappa2``,
  ``CustomImpulse``, ``ConstantImpulse``, a user closed form of C) as in
  ``lower.py``, with the parts that do not depend on the release taken
  from :mod:`validation.lowering_parts`.

A refusal raises ``NotImplementedError`` in the ``lower`` stage.  What 0.4.2
does with what it accepts is what the replay measures.
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

import sft_wick as sw  # noqa: E402
from sft_wick.workflow.specs import (ConstantImpulse,  # noqa: E402
                                     CustomImpulse, CustomKernel,
                                     GeneralKappa2)

from .lowering_parts import (Kappa2Of, PoissonCumulant,  # noqa: E402
                             Sigma2Of, _DampedCosine, _Unit, check_times,
                             closed_form_c, diagonal_c, observable_strings,
                             package_position, vertex_counts)
from .modelspec import ModelSpec, Observable, as_array  # noqa: E402
from .stages import stage as _stage  # noqa: E402

__all__ = ["to_system", "propagators", "Lowered"]

C_SOURCES = ("oracle", "auto", "table")


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


def _noise(model: ModelSpec, structured: bool):
    kappa2 = _structured_kappa2(model) if structured else None
    if kappa2 is None:
        coloured = model.ou or (model.poisson is not None
                                and model.poisson.pulse == "exponential")
        kappa2 = (GeneralKappa2(fn=Kappa2Of(model)) if coloured else
                  sw.SeparableTranslation(
                      temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=1.0),
                      spatial=sw.GaussianSpatial(sigma_x=1.0)))
    p = model.poisson
    if p is not None and p.pulse == "white":
        sigma2 = CustomImpulse(fn=Sigma2Of(model))
    elif model.white is not None:
        sigma2 = ConstantImpulse(as_array(model.white))
    else:
        sigma2 = None
    return sw.GaussianNoise(kappa2=kappa2, sigma2=sigma2)


def to_system(model: ModelSpec, structured: bool = False):
    """The 0.4.2 ``System`` of ``model``."""
    if model.multiplicative is not None:
        raise NotImplementedError("0.4.2 has no multiplicative noise")
    vertices = [sw.LocalVertex(name, coupling=as_array(value))
                for name, value in (("h", model.source),
                                    ("F", model.quadratic),
                                    ("G", model.cubic)) if value is not None]
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


def propagators(system, model: ModelSpec, t_max: float,
                c_source: str = "oracle", diag_C: bool = False,
                n_grid_t: int = 41):
    """The 0.4.2 propagators of ``system`` with C from ``c_source``."""
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


class Lowered:
    """A model and an observable on sft-wick 0.4.2, with the interface of
    :class:`validation.lower.Lowered` that ``validation/worker.py`` uses."""

    def __init__(self, model: ModelSpec, obs: Observable, max_weight: int,
                 flags: dict | None = None, c_source: str = "oracle",
                 n_grid_t: int = 41):
        with _stage("lower"):
            if any(leg.op != "phi" for leg in obs.legs):
                raise NotImplementedError("response legs are not compared yet")
            if obs.target != "moment":
                raise NotImplementedError("0.4.2 has no connected target")
            check_times(model, obs)
            self.model, self.obs = model, obs
            self.max_weight = int(max_weight)
            self.flags = dict(diag_C=diagonal_c(model), iso_C=False)
            self.flags.update(flags or {})
            self.system = to_system(model, structured=(c_source != "oracle"))
        with _stage("expand"):
            self.expansion = self.system.expand(
                observable_strings(obs, model.n_components),
                orders=list(range(self.max_weight + 1)), progress=False,
                **self.flags)
        with _stage("propagators"):
            self.props = propagators(self.system, model, obs.t_final + 0.25,
                                     c_source, self.flags["diag_C"], n_grid_t)
        channels = model.channels()
        self.weights = {name: tuple(int(c == name) for c in channels)
                        for name in channels}
        self.c_source = c_source

    def tag_of(self, order: int, idx: int) -> tuple:
        dt = self.expansion.diagrams(order)[idx]
        counts = vertex_counts(dt.coupling_sum)
        if sum(counts.values()) != order:
            raise RuntimeError(f"diagram {order}/{idx} has vertices {counts}")
        tag = [0] * len(self.model.channels())
        for name, k in counts.items():
            for c, w in enumerate(self.weights[name]):
                tag[c] += k * w
        return tuple(tag)

    def rows(self, method: str, orders=None, **settings) -> list:
        """``(tag, value, error)`` of every diagram of ``orders`` (default
        every order up to ``max_weight``)."""
        obs = self.obs
        every = list(range(self.max_weight + 1))
        orders = every if orders is None else sorted(set(orders))
        with _stage("evaluate"):
            res = self.expansion.evaluate(
                self.props,
                positions={lab: package_position(p)
                           for lab, p in obs.positions},
                t_final=obs.t_final,
                component_pair=tuple(leg.comp for leg in obs.legs),
                orders=orders, external_times=dict(obs.times) or None,
                integrate_over=set(obs.integrate_over) or None,
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
