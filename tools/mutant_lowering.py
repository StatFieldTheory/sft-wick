r"""The lowering of ``validation/lower.py`` with two choices it does not offer,
for ``tools/mutant_fuzz.py``.

``python -m validation.worker --lowering mutant_lowering`` imports this
module, with ``tools/`` on the module search path, in place of
``validation.lower``.  Two environment variables, read at import, choose:

* ``MUTANT_FUZZ_SPELLING``: ``distinct`` (default) spells the observable as
  ``validation.lower`` does, one label per leg; ``shared`` gives the legs
  that sit at one position and one time a single label, the spelling that
  ``System.expand`` refuses at interacting orders.  The oracle's observable
  keeps its distinct labels, so the oracle's value is the one a correct
  package returns for the distinct spelling.
* ``MUTANT_FUZZ_ROUTE``: ``l1`` (default) evaluates through
  ``Expansion.evaluate``; ``l0`` evaluates the same diagrams through
  ``sft_wick.evaluate.integrate_diagrams``, the L0 entry point, with the
  propagator cache of ``System.propagators``.  The L0 route has the guards
  of ``integrate_diagrams`` and not the extra ones of ``Expansion.evaluate``
  (one spatial site under multiplicative noise, the diagonal-C reading at
  every position).

Like ``validation/lower.py`` this module imports sft_wick, so it runs only
in the worker process; it lives in ``tools/`` so that the import guard of
``validation/`` (``tests/test_validation_imports.py``) is unchanged.
"""
from __future__ import annotations

import os

import numpy as np

from validation import lower as base
from validation.lowering_parts import (check_times, diagonal_c,
                                       observable_strings, package_position)
from validation.stages import stage as _stage

sw = base.sw

SPELLINGS = ("distinct", "shared")
ROUTES = ("l1", "l0")
SPELLING = os.environ.get("MUTANT_FUZZ_SPELLING", "distinct")
ROUTE = os.environ.get("MUTANT_FUZZ_ROUTE", "l1")
if SPELLING not in SPELLINGS or ROUTE not in ROUTES:
    raise RuntimeError(f"MUTANT_FUZZ_SPELLING={SPELLING!r} (one of "
                       f"{SPELLINGS}) and MUTANT_FUZZ_ROUTE={ROUTE!r} (one of "
                       f"{ROUTES})")


def shared_labels(obs) -> dict:
    """Leg label to the label it is spelled with: the first leg at the same
    position and time, neither integrated, keeps its own."""
    first: dict = {}
    out = {}
    for leg in obs.legs:
        key = (obs.position(leg.label), obs.time(leg.label),
               leg.label in obs.integrate_over)
        out[leg.label] = first.setdefault(key, leg.label)
    return out


def spelled(obs, n_components: int) -> tuple:
    """The operator strings of ``obs`` under :data:`SPELLING`."""
    strings = observable_strings(obs, n_components)
    if SPELLING == "distinct":
        return strings
    names = shared_labels(obs)
    return tuple(s.replace(f"({leg.label})", f"({names[leg.label]})")
                 for s, leg in zip(strings, obs.legs))


class Lowered(base.Lowered):
    """:class:`validation.lower.Lowered` under :data:`SPELLING` and
    :data:`ROUTE`."""

    def __init__(self, model, obs, max_weight, flags=None, c_source="oracle",
                 n_grid_t=41):
        with _stage("lower"):
            if any(leg.op != "phi" for leg in obs.legs):
                raise NotImplementedError("response legs are not compared yet")
            check_times(model, obs)
            self.model, self.obs = model, obs
            self.max_weight = int(max_weight)
            self.flags = dict(diag_C=diagonal_c(model), iso_C=False)
            self.flags.update(flags or {})
            self.system = base.to_system(model,
                                         structured=(c_source != "oracle"))
            self.labels = (shared_labels(obs) if SPELLING == "shared"
                           else {leg.label: leg.label for leg in obs.legs})
        with _stage("expand"):
            self.expansion = self.system.expand(
                spelled(obs, model.n_components),
                orders=list(range(self.max_weight + 1)), target=obs.target,
                progress=False, **self.flags)
        with _stage("propagators"):
            self.props = base.propagators(
                self.system, model, obs.t_final + 0.25, c_source,
                self.flags["diag_C"], n_grid_t)
        self.weights = base.channel_weights(model, self.system)
        self.c_source = c_source

    def _positions(self) -> dict:
        used = set(self.labels.values())
        return {lab: package_position(p) for lab, p in self.obs.positions
                if lab in used}

    def rows(self, method: str, orders=None, **settings) -> list:
        if ROUTE == "l1" and SPELLING == "distinct":
            return super().rows(method, orders, **settings)
        every = list(range(self.max_weight + 1))
        orders = every if orders is None else sorted(set(orders))
        if orders != every and "g" in self.model.channels():
            raise ValueError("a subset of orders gives incomplete tags of the "
                             "g channel; evaluate every order")
        obs = self.obs
        used = set(self.labels.values())
        over = {lab for lab in obs.integrate_over if lab in used} or None
        times = {lab: t for lab, t in obs.times if lab in used} or None
        if ROUTE == "l1":
            return self._l1_rows(method, orders, over, times, settings)
        return self._l0_rows(method, orders, over, times, settings)

    def _keep(self, order: int, idx: int, value, error) -> tuple | None:
        tag = self.tag_of(order, idx)
        if sum(tag) > self.max_weight:
            return None
        return (tag, float(np.real(value)), float(error or 0.0))

    def _l1_rows(self, method, orders, over, times, settings) -> list:
        with _stage("evaluate"):
            res = self.expansion.evaluate(
                self.props, positions=self._positions(),
                t_final=self.obs.t_final,
                component_pair=tuple(leg.comp for leg in self.obs.legs),
                orders=orders, external_times=times, integrate_over=over,
                method=method, **settings)
        rows = (self._keep(r["order"], r["diagram_idx"], r["value"],
                           r.get("error")) for r in res.per_diagram)
        return [r for r in rows if r is not None]

    def _fixed_indices(self) -> dict:
        """The component of each observable index letter, as
        ``Expansion.evaluate`` passes it."""
        out = {}
        for (_name, comp, _point), leg in zip(self.expansion.observable_repr,
                                               self.obs.legs):
            if comp is not None:
                out[comp] = int(leg.comp)
        return out

    def _l0_rows(self, method, orders, over, times, settings) -> list:
        from sft_wick.evaluate import integrate_diagrams

        out = []
        coupling_values = self.system.build_coupling_values()
        with _stage("evaluate"):
            for order in orders:
                terms = list(self.expansion.diagrams(order))
                if not terms:
                    continue
                _total, details = integrate_diagrams(
                    terms, coupling_values=coupling_values,
                    lambda_f=self.obs.t_final, cache=self.props.cache,
                    t_min=float(self.system.t_min), method=method,
                    fixed_indices=self._fixed_indices(),
                    positions=self._positions(), integrate_over=over,
                    external_times=times, **settings)
                for idx, (value, error) in enumerate(details):
                    row = self._keep(order, idx, value, error)
                    if row is not None:
                        out.append(row)
        return out
