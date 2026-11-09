"""Reduce a failing fuzz case to a smaller one that still fails.

:func:`shrink` removes one feature at a time: a drift term, a noise block,
a cumulant, a leg, a component, the second time, the integrated leg, the
connected target, ``t_min``.  It keeps a candidate when the comparison
still fails with the same integrator, and stops when no candidate does or
the budget of evaluations is spent.  A candidate keeps the flags of the
case it came from where it still admits them and never switches one on; a
flag it can no longer admit is switched off and named in the step.  Run it
as ``python -m validation.fuzz shrink DIR --seed S --method M``.  Imports no
sft-wick code: the package runs in the worker subprocess.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from .draw import Case, admissible_flags
from .fuzz import Worker, job_of, oracle_of, records_of, verdict
from .modelspec import (Leg, ModelSpec, Observable, model_from_dict,
                        model_to_dict, observable_from_dict,
                        observable_to_dict)
from .stages import JobTimeout

__all__ = ["without_component", "candidates", "shrink", "main"]


def without_component(model: ModelSpec, obs: Observable, k: int):
    """The case with component ``k`` removed, or ``None`` if a leg reads it
    or the model has one component."""
    if any(leg.comp == k for leg in obs.legs) or model.n_components < 2:
        return None
    keep = [a for a in range(model.n_components) if a != k]

    def cut(x, m):
        if x is None:
            return None
        return np.asarray(x, dtype=float)[np.ix_(*[keep] * m)]

    mult = model.multiplicative
    if mult is not None:
        g0 = np.asarray(mult.g0)[keep]
        g1 = np.asarray(mult.g1)[np.ix_(keep, range(g0.shape[1]), keep)]
        mult = dataclasses.replace(mult, g0=g0, g1=g1)
    p = model.poisson
    if p is not None:
        p = dataclasses.replace(
            p, amplitude=tuple(p.amplitude[a] for a in keep),
            width=tuple(p.width[a] for a in keep),
            tau=None if p.tau is None else tuple(p.tau[a] for a in keep))
    ou = tuple(dataclasses.replace(b, loading=tuple(b.loading[a] for a in keep))
               for b in model.ou)
    model = dataclasses.replace(
        model, gamma=tuple(model.gamma[a] for a in keep),
        source=None if model.source is None else tuple(
            model.source[a] for a in keep),
        quadratic=cut(model.quadratic, 3), cubic=cut(model.cubic, 4),
        white=cut(model.white, 2), ou=ou, multiplicative=mult, poisson=p,
        cumulants=tuple(dataclasses.replace(c, tensor=cut(c.tensor, c.order))
                        for c in model.cumulants))
    legs = [Leg(leg.comp - (leg.comp > k), leg.label, leg.op)
            for leg in obs.legs]
    return model, dataclasses.replace(obs, legs=legs)


def _ou_groups(model: ModelSpec) -> list:
    """Indices of OU blocks that share a rate, amplitude and kernel (the
    per-component copies of an isotropic model), for groups of two or
    more."""
    groups: dict = {}
    for i, blk in enumerate(model.ou):
        key = (blk.lam, blk.sigma_t, blk.spatial)
        groups.setdefault(key, []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def candidates(case: Case):
    """``(step, model, observable)`` with one feature removed."""
    m, obs = case.model, case.observable
    rep = dataclasses.replace
    for name in ("cubic", "quadratic", "source", "white", "multiplicative",
                 "poisson"):
        if getattr(m, name) is not None:
            yield f"drop {name}", rep(m, **{name: None}), obs
    if m.poisson is not None and m.poisson.orders:
        yield "Poisson orders ()", rep(m, poisson=rep(m.poisson,
                                                      orders=())), obs
    for group in _ou_groups(m):
        yield (f"drop OU {group}",
               rep(m, ou=tuple(b for i, b in enumerate(m.ou)
                               if i not in group)), obs)
    for i in range(len(m.ou)):
        yield f"drop OU {i}", rep(m, ou=m.ou[:i] + m.ou[i + 1:]), obs
    for i, c in enumerate(m.cumulants):
        yield (f"drop {c.channel}",
               rep(m, cumulants=m.cumulants[:i] + m.cumulants[i + 1:]), obs)
    if m.t_min != 0.0:
        shift = -m.t_min
        yield "t_min 0", rep(m, t_min=0.0), rep(
            obs, t_final=obs.t_final + shift,
            times={k: v + shift for k, v in obs.times})
    if obs.target == "connected":
        yield "moment target", m, rep(obs, target="moment")
    if obs.times:
        yield "one time", m, rep(obs, times=())
    if obs.integrate_over:
        yield "no integrated leg", m, rep(obs, integrate_over=())
    if len(obs.legs) > 1:
        for i, leg in enumerate(obs.legs):
            legs = obs.legs[:i] + obs.legs[i + 1:]
            labels = {lg.label for lg in legs}
            yield f"drop leg {leg.label}", m, Observable(
                legs, {k: v for k, v in obs.positions if k in labels},
                obs.t_final, times={k: v for k, v in obs.times if k in labels},
                integrate_over=tuple(x for x in obs.integrate_over
                                     if x in labels), target=obs.target)
    for k in range(m.n_components):
        cut = without_component(m, obs, k)
        if cut is not None:
            yield f"drop component {k}", cut[0], cut[1]


def _still_fails(case: Case, method: str, worker: Worker, timeout: float,
                 oracle_timeout: float | None) -> dict | None:
    try:
        ref, scale = oracle_of(case, oracle_timeout)
    except (JobTimeout, Exception):  # noqa: BLE001  (not a usable candidate)
        return None
    job = job_of(case, method, timeout)
    got = verdict(worker.run(job), ref, scale, method, job=job)
    return got if got["verdict"] == "fail" else None


def shrink(case: Case, method: str, package: str, lowering: str,
           timeout: float, budget: int = 200, extra_path: tuple = (),
           oracle_timeout: float | None = 600.0) -> tuple:
    """The smallest case reached, the steps taken, and its verdict."""
    worker = Worker(Path(package), lowering, extra_path)
    steps: list = []
    try:
        result = _still_fails(case, method, worker, timeout, oracle_timeout)
        if result is None:
            raise ValueError("the case does not fail")
        spent, changed = 1, True
        while changed and spent < budget:
            changed = False
            for step, model, obs in candidates(case):
                try:
                    allowed = admissible_flags(model)
                except ValueError:
                    continue
                flags = {k: bool(case.flags.get(k)) and v
                         for k, v in allowed.items()}
                lost = sorted(k for k, v in flags.items()
                              if case.flags.get(k) and not v)
                cand = Case(case.seed, case.kind, model, obs, case.max_weight,
                            flags)
                spent += 1
                got = _still_fails(cand, method, worker, timeout,
                                   oracle_timeout)
                if got is not None:
                    case, result, changed = cand, got, True
                    steps.append(step + (f" ({', '.join(lost)} off)"
                                         if lost else ""))
                    break
                if spent >= budget:
                    break
    finally:
        worker.close()
    return case, steps, result


def main(args) -> None:
    """``python -m validation.fuzz shrink``: write the reduced case beside
    the records."""
    records = [r for r in records_of(Path(args.out) / "records.jsonl")
               if r["seed"] == args.seed and r["method"] == args.method]
    if not records:
        raise SystemExit(f"no record of seed {args.seed} with {args.method}")
    r = records[0]
    case = Case(r["seed"], r["kind"], model_from_dict(r["model"]),
                observable_from_dict(r["observable"]), r["max_weight"],
                r["flags"])
    small, steps, result = shrink(case, args.method, args.package,
                                  args.lowering, args.timeout, args.budget,
                                  tuple(args.extra_path), args.oracle_timeout)
    out = dict(seed=args.seed, method=args.method, steps=steps,
               model=model_to_dict(small.model),
               observable=observable_to_dict(small.observable),
               max_weight=small.max_weight, flags=small.flags,
               channels=list(small.model.channels()), **result)
    target = Path(args.out) / f"shrunk_{args.seed}_{args.method}.json"
    target.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("steps", "channels", "worst_tag",
                                           "worst_ratio")}))
