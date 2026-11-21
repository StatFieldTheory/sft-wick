#!/usr/bin/env python
r"""Kill corpus mutants with drawn cases judged by the independent oracle.

``python tools/mutant_fuzz.py run OUT --oracle-cache DIR`` takes the
historical corpus entries of :data:`TARGETS` (or those named by
``--labels``), copies the project once per entry with the entry applied
(``tools/mutation_check.py``'s copy mode, ``OUT/trees/<label>``), draws
cases with :func:`validation.draw.draw`, computes the oracle of every case
once, runs each case under the unmutated ``src/`` and under the copy of its
entry with ``python -m validation.worker`` (each case once under the
unmutated tree, :func:`unique_items`), and joins the two records of a
(seed, mode, route, integrator).  ``summary OUT`` prints the result again.

The kill rule (:func:`kill_of`):

* the unmutated tree passes against the oracle and the mutant fails (a
  value outside the tolerance of :mod:`validation.compare`), or
* the entry restores a refusal (its :class:`Target` has ``expect``), the
  unmutated tree refuses the case with an error matching ``expect``, and
  the mutant returns a value the oracle rejects.

The mutant's failure must include a finite value outside its tolerance: a
fail whose only failing rows are not finite (a NaN or an infinity) is not a
kill.  A mutant that refuses for another reason, raises, times out or
returns a value the oracle accepts is not killed.  An entry with no kill
makes the run exit 1.

Modes (:func:`force`) change a drawn case where the default draw never
reaches a refusal; ``default`` leaves the draw unchanged:

* ``shared-label``: every leg at the first leg's position and at
  ``t_final``, spelled with one label by ``tools/mutant_lowering.py``;
* ``diag-c``: ``diag_C=True`` on a model whose C mixes components;
* ``iso-r-on``: ``iso_R=True`` with distinct rates (a scalar R against a
  matrix-R cache); ``iso-r-off``: ``iso_R=False, diag_R=False`` with equal
  rates (two-index R against a scalar cache);
* ``multiplicative-apart``: the legs of a multiplicative-noise case moved
  to distinct positions, drawn from the seed.

Each forced case has a control (:func:`control_of`), the same case with
admissible flags and the distinct spelling, which both trees must pass for
a kill to count.  ``--route l0`` evaluates through
``sft_wick.evaluate.integrate_diagrams`` instead of ``Expansion.evaluate``
(see ``tools/mutant_lowering.py``).

``OUT/kills.json`` keeps, for each row, the compared rows of both trees at
the mutant's worst tag (:func:`row_at`).  The cases of an entry's first
kill and their controls are saved in ``OUT/kill_cases/<label>.json``, and
``run --case-file FILE`` runs the cases of such a file again instead of
drawing them.  ``tools/generated_kills.py`` collects the kills of several
runs into ``docs/verification/generated_kills.json``.

The oracle row of a case is computed once and cached outside the project
in ``--oracle-cache`` (``tools/mutant_oracle.py``), in at most ``--jobs``
processes of at most ``--oracle-max-rss-gb`` each; then every package runs
its jobs in at most ``--jobs`` worker processes.  The two phases do not
overlap.  This process imports no sft_wick code.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Iterable, Sequence

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation import fuzz  # noqa: E402
from validation.draw import Case, admissible_flags, draw  # noqa: E402
from validation.modelspec import (Observable, as_array,  # noqa: E402
                                  model_from_dict, model_to_dict,
                                  observable_from_dict, observable_to_dict)


def _load_sibling(name: str):
    path = TOOLS / f"{name}.py"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mutation_check = _load_sibling("mutation_check")
load_corpus = mutation_check.load_corpus
#: The oracle rows and their cache (``tools/mutant_oracle.py``).
mutant_oracle = _load_sibling("mutant_oracle")
oracle_key = mutant_oracle.oracle_key
oracle_rows = mutant_oracle.oracle_rows
check_cache_dir = mutant_oracle.check_cache_dir
_ORACLE_SHA = mutant_oracle.ORACLE_SHA

MODES = ("default", "shared-label", "diag-c", "iso-r-on", "iso-r-off",
         "multiplicative-apart")
ROUTES = ("l1", "l0")
SPELLING_OF = {"shared-label": "shared"}
METHODS = ("gauss_legendre", "qmc_vectorized")
#: Jobs, worker processes and oracle processes at a time, at most.
MAX_JOBS = 3
DEFAULT_TIMEOUT = 240.0
DEFAULT_ORACLE_TIMEOUT = 600.0
DEFAULT_SEARCH = (0, 3000)
DEFAULT_PER_LABEL = 6
CONTROL = "/control"


@dataclasses.dataclass(frozen=True)
class Target:
    """How a corpus entry is reached: the draw mode, the evaluation route,
    and, for an entry that removes a refusal, the text the unmutated tree
    refuses with."""

    mode: str
    route: str = "l1"
    expect: str | None = None
    methods: tuple = METHODS
    note: str | None = None


#: Why an entry is run on the L0 route.
_L0_ONLY = (
    "Equivalent on L1 and L2: Expansion.evaluate refuses the forced cases "
    "for the unmutated tree and the mutant alike, by {guard}, which the "
    "entry does not change. Killed on L0 (integrate_diagrams); with that "
    "guard also removed, L1 returns the value the mutant returns on L0.")

TARGETS = {
    "historical-shared-external-label-accepted": Target(
        "shared-label", expect=r"share the same spatial label"),
    "historical-callable-coupling-at-one-leg-order": Target("default"),
    "historical-iso-c-trace": Target("default"),
    "historical-t-min-not-reaching-integrators": Target("default"),
    # The refusals of System.expand and integrate_diagrams that the entries
    # remove, not the guards of Expansion.evaluate ("κ² of the noise has
    # off-diagonal ...", "the external points ... at different positions").
    "historical-diagonal-flags-on-a-non-diagonal-system": Target(
        "diag-c", route="l0",
        expect=r"of this system has off-diagonal component entries",
        note=_L0_ONLY.format(guard="the diagonal-C reading at every position "
                             "(_guard_diagonal_c)")),
    "historical-r-group-takes-any-position": Target(
        "multiplicative-apart", route="l0",
        expect=r"the spatial points '[^']+' and '[^']+' are at different"
        r"\s+positions",
        note=_L0_ONLY.format(guard="the single-site guard under "
                             "multiplicative noise (_guard_single_site)")),
    "historical-r-cache-mismatch-accepted": Target(
        "iso-r-on", expect=r"propagator cache's R is a (scalar|matrix)"),
    "historical-no-cut-at-a-pinned-external-time": Target(
        "default", methods=("gauss_legendre",)),
}


class ForceError(ValueError):
    """The case does not meet the precondition of the mode."""


# ---------------------------------------------------------------------------
# Forced cases
# ---------------------------------------------------------------------------

def _leg_labels(obs: Observable) -> list:
    return [leg.label for leg in obs.legs]


def _with_positions(obs: Observable, positions: dict, *, times=None,
                    integrate_over=None) -> Observable:
    return Observable(list(obs.legs), positions, obs.t_final,
                      times=dict(obs.times) if times is None else times,
                      integrate_over=(obs.integrate_over if integrate_over
                                      is None else integrate_over),
                      target=obs.target)


def _apart(case: Case) -> dict:
    """Distinct positions for the legs, drawn from the seed: each at least
    0.2 from the origin and from every other."""
    obs = case.observable
    rng = np.random.default_rng([case.seed, 7919])
    dim = len(obs.position(obs.legs[0].label))
    out: dict = {}
    for lab in _leg_labels(obs):
        while True:
            v = rng.uniform(-1.0, 1.0, size=dim)
            if np.linalg.norm(v) > 0.2 and all(
                    np.linalg.norm(v - np.asarray(p)) > 0.2
                    for p in out.values()):
                break
        out[lab] = tuple(float(c) for c in v)
    return out


def force(case: Case, mode: str) -> Case:
    """``case`` under ``mode`` (see the module docstring); raises
    :class:`ForceError` when the case does not meet the mode's
    precondition."""
    if mode == "default":
        return case
    flags, obs, model = dict(case.flags), case.observable, case.model
    rates = set(model.gamma)
    if mode == "shared-label":
        if len(obs.legs) < 2:
            raise ForceError("shared-label needs two legs or more")
        first = obs.position(obs.legs[0].label)
        obs = _with_positions(obs, {lab: first for lab in _leg_labels(obs)},
                              times={}, integrate_over=())
        return dataclasses.replace(case, observable=obs,
                                   max_weight=max(1, case.max_weight))
    if mode == "diag-c":
        if flags["diag_C"]:
            raise ForceError("diag-c needs a C that mixes components")
        flags["diag_C"] = True
    elif mode == "iso-r-on":
        if len(rates) == 1:
            raise ForceError("iso-r-on needs distinct rates")
        flags["iso_R"] = True
    elif mode == "iso-r-off":
        if len(rates) != 1 or model.n_components == 1:
            raise ForceError("iso-r-off needs N >= 2 and equal rates")
        flags.update(iso_R=False, diag_R=False)
    elif mode == "multiplicative-apart":
        if model.multiplicative is None or len(obs.legs) < 2:
            raise ForceError("multiplicative-apart needs multiplicative noise "
                             "and two legs or more")
        obs = _with_positions(obs, _apart(case))
        return dataclasses.replace(case, observable=obs)
    else:
        raise ForceError(f"unknown mode {mode!r}; the modes are {MODES}")
    return dataclasses.replace(case, flags=flags)


def control_of(forced: Case, mode: str) -> tuple | None:
    """The admissible twin of a forced case and its spelling, or None for
    the default mode: the same observable spelled with distinct labels, the
    model's admissible flags, and, for ``multiplicative-apart``, every leg
    back at the first leg's position."""
    if mode == "default":
        return None
    control = dataclasses.replace(forced,
                                  flags=admissible_flags(forced.model))
    if mode == "multiplicative-apart":
        obs = forced.observable
        first = obs.position(obs.legs[0].label)
        control = dataclasses.replace(control, observable=_with_positions(
            obs, {lab: first for lab in _leg_labels(obs)}))
    return control, "distinct"


# ---------------------------------------------------------------------------
# Seeds that reach each entry
# ---------------------------------------------------------------------------

def _white_offdiagonal(model) -> bool:
    if model.white is None:
        return False
    return bool(np.any(np.triu(as_array(model.white), 1) != 0.0))


def _costly(case: Case) -> bool:
    """The shape of every oracle that ran out of memory in earlier runs."""
    m = case.model
    return (m.n_components == 3 and len(case.observable.legs) == 3
            and m.cubic is not None and case.max_weight >= 2)


def _reaches(label: str, case: Case) -> bool:
    m, obs = case.model, case.observable
    comps = {leg.comp for leg in obs.legs}
    position_dependent = m.poisson is not None or any(
        blk.spatial.kind != "constant" for blk in m.ou)
    white = (m.white is not None or any(c.time == "white"
                                        for c in m.cumulants)
             or (m.poisson is not None and m.poisson.pulse == "white"))
    return {
        "historical-shared-external-label-accepted": len(obs.legs) >= 2,
        "historical-callable-coupling-at-one-leg-order": (
            m.poisson is not None and m.n_components >= 2
            and any(k in m.poisson.orders for k in (3, 4))),
        "historical-iso-c-trace": (case.flags.get("iso_C", False)
                                   and m.n_components >= 2),
        "historical-t-min-not-reaching-integrators": m.t_min != 0.0,
        "historical-diagonal-flags-on-a-non-diagonal-system": (
            not case.flags["diag_C"] and not _white_offdiagonal(m)
            and m.multiplicative is None and len(comps) >= 2),
        "historical-r-group-takes-any-position": (
            m.multiplicative is not None and len(obs.legs) >= 2
            and position_dependent),
        "historical-r-cache-mismatch-accepted": len(set(m.gamma)) > 1,
        "historical-no-cut-at-a-pinned-external-time": (
            bool(obs.times) and white),
    }[label]


def candidates(label: str, start: int, stop: int, limit: int) -> list:
    """The first ``limit`` seeds in ``[start, stop)`` whose draw reaches the
    code of ``label`` and meets its mode's precondition."""
    mode = TARGETS[label].mode
    out = []
    for seed in range(start, stop):
        case = draw(seed)
        if _costly(case) or not _reaches(label, case):
            continue
        try:
            force(case, mode)
        except ForceError:
            continue
        out.append(seed)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# The kill rule
# ---------------------------------------------------------------------------

def rejected_value(record: dict) -> bool:
    """Whether ``record`` has a compared row whose oracle value, value and
    error are finite and whose value lies outside its tolerance."""
    for _tag, oracle, value, error, allowed, _ratio in record.get("rows") or ():
        finite = all(math.isfinite(float(x)) for x in (oracle, value, error))
        if finite and abs(float(value) - float(oracle)) > float(allowed):
            return True
    return False


def kill_of(head: dict, mutant: dict, expect: str | None = None) -> tuple:
    """``(killed, reason)`` for the records of one case and integrator under
    the unmutated tree and under the mutant (see the module docstring)."""
    hv, mv = head.get("verdict"), mutant.get("verdict")
    if hv == "refusal" and expect is None:
        return False, "baseline refusal"
    if hv == "refusal" and not re.search(expect, head.get("error") or ""):
        return False, "baseline refused otherwise"
    if hv not in ("pass", "refusal"):
        return False, f"baseline {hv}"
    if mv != "fail":
        return False, f"mutant {mv}"
    if not rejected_value(mutant):
        return False, "mutant non-finite"
    return True, "value" if hv == "pass" else "refusal"


def _key(record: dict) -> tuple:
    return (record["seed"], record.get("mode", "default"), record.get("route"),
            record["method"])


def row_at(record: dict, tag) -> dict | None:
    """The compared row of ``record`` at ``tag`` as ``{tag, oracle, value,
    error, allowed, ratio}``, or None (no values, or no such tag)."""
    if tag is None:
        return None
    for row in record.get("rows") or ():
        if list(row[0]) == list(tag):
            return dict(zip(("tag", "oracle", "value", "error", "allowed",
                             "ratio"), row))
    return None


def join(head_records: Iterable[dict], mutant_records: Iterable[dict],
         expect: str | None = None) -> list:
    """One row per (seed, mode, route, integrator) of ``head_records``: both
    verdicts, ``killed`` and the reason of :func:`kill_of`, and the
    compared rows of both trees at the mutant's worst tag."""
    mutants = {_key(r): r for r in mutant_records}
    rows = []
    for head in head_records:
        key = _key(head)
        row = dict(seed=key[0], mode=key[1], method=key[3],
                   route=key[2], head=head.get("verdict"),
                   head_error=head.get("error"))
        mutant = mutants.get(key)
        if mutant is None:
            row.update(mutant=None, killed=False, reason="no mutant record")
        else:
            killed, reason = kill_of(head, mutant, expect)
            tag = mutant.get("worst_tag")
            row.update(mutant=mutant.get("verdict"),
                       mutant_error=mutant.get("error"),
                       worst_ratio=mutant.get("worst_ratio"),
                       worst_tag=tag, mutant_row=row_at(mutant, tag),
                       head_row=row_at(head, tag),
                       killed=killed, reason=reason)
        rows.append(row)
    return rows


def attach_controls(rows: list, head_records: Iterable[dict],
                    mutant_records: Iterable[dict]) -> list:
    """Each row with the verdicts of its control under both trees, and
    ``counted``: a kill whose control passed under both, or that has no
    control."""
    heads = {_key(r): r for r in head_records}
    mutants = {_key(r): r for r in mutant_records}
    out = []
    for row in rows:
        if row["mode"].endswith(CONTROL):
            continue
        key = (row["seed"], row["mode"] + CONTROL, row["route"],
               row["method"])
        control = None
        if row["mode"] != "default":
            control = [heads.get(key, {}).get("verdict"),
                       mutants.get(key, {}).get("verdict")]
        counted = row["killed"] and (control is None
                                     or control == ["pass", "pass"])
        out.append(dict(row, control=control, counted=counted))
    return out


def label_summary(label: str, rows: list) -> dict:
    kills = [r for r in rows if r["counted"]]
    reasons = collections.Counter(
        r["reason"] if not r["killed"] else "control not passing"
        for r in rows if not r["counted"])
    return dict(label=label, target=dataclasses.asdict(TARGETS[label])
                if label in TARGETS else None,
                cases=len({(r["seed"], r["mode"]) for r in rows}),
                rows=len(rows), killed=bool(kills),
                kills=[dict(seed=r["seed"], mode=r["mode"],
                            method=r["method"], rule=r["reason"],
                            worst_ratio=r.get("worst_ratio"),
                            mutant_row=r.get("mutant_row"),
                            head_row=r.get("head_row"),
                            head_error=r.get("head_error"))
                       for r in kills],
                not_killed=dict(reasons))


# ---------------------------------------------------------------------------
# The package side
# ---------------------------------------------------------------------------

class ModeWorker(fuzz.Worker):
    """:class:`validation.fuzz.Worker` running ``tools/mutant_lowering.py``
    with a spelling and a route, ``PYTHONHASHSEED=0`` (the mutated R-group
    resolver depends on ``frozenset`` order) and no bytecode written into
    the trees."""

    def __init__(self, package: pathlib.Path, spelling: str, route: str,
                 max_rss: float | None):
        super().__init__(package, "mutant_lowering", (TOOLS,), max_rss)
        self.mode_env = dict(MUTANT_FUZZ_SPELLING=spelling,
                             MUTANT_FUZZ_ROUTE=route)

    def _start(self) -> None:
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [str(self.package), str(fuzz.ROOT), *self.extra_path]),
            OMP_NUM_THREADS="1", VALIDATION_EXPECT_PACKAGE=str(self.package),
            VALIDATION_PARENT_PID=str(os.getpid()), PYTHONHASHSEED="0",
            PYTHONDONTWRITEBYTECODE="1", **self.mode_env)
        self.stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "validation.worker", "--lowering",
             self.lowering], cwd=fuzz.ROOT, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.stderr)


@dataclasses.dataclass(frozen=True)
class Item:
    """One case to run: its mode (with :data:`CONTROL` for a control), the
    spelling and route of the lowering, and the integrators."""

    case: Case
    mode: str
    spelling: str
    route: str
    methods: tuple


def item_to_dict(item: Item) -> dict:
    """A case and how it is run, in the form :func:`items_from_file`
    reads."""
    case = item.case
    return dict(seed=case.seed, kind=case.kind, mode=item.mode,
                spelling=item.spelling, route=item.route,
                methods=list(item.methods), model=model_to_dict(case.model),
                observable=observable_to_dict(case.observable),
                max_weight=case.max_weight, flags=dict(case.flags))


def items_from_file(path: pathlib.Path) -> tuple:
    """``(label, items)`` of a case file written by :func:`run` (``OUT/
    kill_cases/<label>.json``): the cases as saved, not drawn again."""
    data = json.loads(pathlib.Path(path).read_text())
    items = [Item(Case(d["seed"], d["kind"], model_from_dict(d["model"]),
                       observable_from_dict(d["observable"]),
                       d["max_weight"], dict(d["flags"])),
                  d["mode"], d["spelling"], d["route"], tuple(d["methods"]))
             for d in data["items"]]
    return data["label"], items


def items_for(label: str, seeds: Sequence[int], mode: str | None = None,
              route: str | None = None, methods: tuple | None = None) -> tuple:
    """``(items, skipped)``: the forced cases of ``seeds`` and their
    controls, and the seeds whose draw the mode refuses, with the reason."""
    target = TARGETS[label]
    mode = mode or target.mode
    route = route or target.route
    methods = methods or target.methods
    items, skipped = [], []
    for seed in seeds:
        try:
            forced = force(draw(seed), mode)
        except ForceError as exc:
            skipped.append(dict(seed=seed, mode=mode, reason=str(exc)))
            continue
        items.append(Item(forced, mode, SPELLING_OF.get(mode, "distinct"),
                          route, methods))
        control = control_of(forced, mode)
        if control is not None:
            items.append(Item(control[0], mode + CONTROL, control[1], route,
                              methods))
    return items, skipped


def unique_items(items: Iterable[Item]) -> list:
    """``items`` with one item per seed, mode, route and spelling, running
    the integrators any of them asks for, so that the unmutated tree runs
    each case once when entries share seeds."""
    merged: dict = {}
    for item in items:
        key = (item.case.seed, item.mode, item.route, item.spelling)
        prev = merged.get(key)
        if prev is None:
            merged[key] = item
            continue
        if prev.case != item.case:
            raise ValueError(f"two different cases for {key}")
        asked = set(prev.methods) | set(item.methods)
        merged[key] = dataclasses.replace(prev, methods=tuple(
            m for m in fuzz.METHODS if m in asked))
    return list(merged.values())


def records_for(records: Iterable[dict], items: Iterable[Item]) -> list:
    """The records of ``records`` that ``items`` ask for: the same seed,
    mode, route and integrator."""
    wanted = {(i.case.seed, i.mode, i.route, m) for i in items
              for m in i.methods}
    return [r for r in records if _key(r) in wanted]


def _record(item: Item, method: str, oracle: dict, worker, timeout: float,
            package_label: str) -> dict:
    case = item.case
    record = dict(seed=case.seed, kind=case.kind, mode=item.mode,
                  spelling=item.spelling, route=item.route,
                  package_label=package_label, features=fuzz.features(case),
                  model=model_to_dict(case.model),
                  observable=observable_to_dict(case.observable),
                  max_weight=case.max_weight, flags=case.flags,
                  oracle_seconds=oracle.get("seconds"), method=method,
                  settings=fuzz.METHODS[method])
    if "lost" in oracle:
        return dict(record, verdict=oracle["lost"], error=oracle.get("error"))
    ref = {tuple(t): v for t, v in oracle["rows"]}
    try:
        job = fuzz.job_of(case, method, timeout)
        result = worker.run(job)
        record.update({k: result[k] for k in fuzz._KEPT if k in result})
        record.update(fuzz.verdict(result, ref, oracle["scale"], method,
                                   job=job))
    except Exception as exc:  # noqa: BLE001  (one record, not the run)
        worker.close()
        record.update(verdict="harness-error",
                      error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc(limit=-8))
    return record


def run_package(package: pathlib.Path, package_label: str,
                items: Sequence[Item], oracles: dict, jobs: int,
                timeout: float, max_rss: float, log=print) -> list:
    """The records of ``items`` under the package at ``package`` (a ``src``
    directory), in at most ``jobs`` worker processes."""
    groups: dict = collections.defaultdict(list)
    for item in items:
        for method in item.methods:
            groups[(item.spelling, item.route)].append((item, method))
    records: list = []
    for (spelling, route), work in sorted(groups.items()):
        chunks = [work[k::jobs] for k in range(min(jobs, len(work)))]

        def run_chunk(chunk, spelling=spelling, route=route):
            worker = ModeWorker(package, spelling, route, max_rss)
            try:
                return [_record(item, method, oracles[oracle_key(item.case)],
                                worker, timeout, package_label)
                        for item, method in chunk]
            finally:
                worker.close()

        with cf.ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            for out in pool.map(run_chunk, chunks):
                records += out
        log(f"{package_label}: {spelling}/{route} {len(work)} jobs done")
    return records


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_tree(mut, out: pathlib.Path) -> pathlib.Path:
    """A fresh copy of the project at ``out`` with ``mut`` applied."""
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mutation_check.mutated_copy(mut, out)
    return out


def _write_jsonl(path: pathlib.Path, rows: Iterable[dict]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _save_kill_cases(directory: pathlib.Path, label: str,
                     items: Sequence[Item], entry: dict) -> None:
    """``directory/<label>.json``: the cases of the first counted kill of
    ``entry`` and their controls, for ``run --case-file``."""
    if not entry["kills"]:
        return
    first = entry["kills"][0]
    keep = [item_to_dict(i) for i in items if i.case.seed == first["seed"]
            and i.mode in (first["mode"], first["mode"] + CONTROL)]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{label}.json").write_text(json.dumps(
        dict(label=label, kill=first, items=keep), indent=1))


def _seeds_for(label: str, args) -> list:
    if args.seeds:
        return list(args.seeds)
    start, stop = args.search
    return candidates(label, start, stop, args.per_label)


def run(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = check_cache_dir(pathlib.Path(args.oracle_cache))
    corpus = {m.label: m for m in load_corpus()}
    labels = args.labels or list(TARGETS)
    unknown = [lab for lab in labels if lab not in TARGETS]
    if unknown:
        raise SystemExit(f"no target for {unknown}; targets: {list(TARGETS)}")
    methods = tuple(args.methods) if args.methods else None
    plan, skipped = {}, []
    if args.case_file:
        label, items = items_from_file(args.case_file)
        if label not in TARGETS:
            raise SystemExit(f"{args.case_file}: no target for {label!r}")
        labels, plan = [label], {label: items}
    for label in [] if args.case_file else labels:
        items, skip = items_for(label, _seeds_for(label, args), args.mode,
                                args.route, methods)
        plan[label] = items
        skipped += [dict(s, label=label) for s in skip]
    every = [item for items in plan.values() for item in items]
    _write_jsonl(out / "cases.jsonl", (
        dict(label=label, seed=i.case.seed, mode=i.mode, spelling=i.spelling,
             route=i.route, methods=list(i.methods),
             oracle_key=oracle_key(i.case))
        for label, items in plan.items() for i in items))
    oracles = oracle_rows([i.case for i in every], cache, args.jobs,
                          args.oracle_timeout,
                          args.oracle_max_rss_gb * fuzz.GB)
    records_dir = out / "records"
    records_dir.mkdir(exist_ok=True)
    head = run_package(ROOT / "src", "head", unique_items(every), oracles,
                       args.jobs, args.timeout, args.max_rss_gb * fuzz.GB)
    _write_jsonl(records_dir / "head.jsonl", head)
    summary = dict(oracle_sha=_ORACLE_SHA, skipped=skipped, labels={})
    for label in labels:
        mut = corpus[label]
        tree = build_tree(mut, out / "trees" / label)
        mutated = mut.target(tree)
        records = run_package(tree / "src", label, plan[label], oracles,
                              args.jobs, args.timeout,
                              args.max_rss_gb * fuzz.GB)
        _write_jsonl(records_dir / f"{label}.jsonl", records)
        heads = records_for(head, plan[label])
        rows = attach_controls(join(heads, records, TARGETS[label].expect),
                               heads, records)
        entry = label_summary(label, rows)
        entry.update(path=mut.relpath, mutated_sha256=_sha256(mutated),
                     corpus_old_sha256=hashlib.sha256(
                         mut.old.encode()).hexdigest(),
                     rows_detail=rows)
        summary["labels"][label] = entry
        _save_kill_cases(out / "kill_cases", label, plan[label], entry)
        if not args.keep_trees:
            shutil.rmtree(tree)
    (out / "kills.json").write_text(json.dumps(summary, indent=1))
    print_summary(summary)
    return 0 if all(e["killed"] for e in summary["labels"].values()) else 1


def print_summary(summary: dict) -> None:
    for label, entry in summary["labels"].items():
        state = "KILLED" if entry["killed"] else "SURVIVED"
        kills = ", ".join(f"{k['seed']}/{k['method']}/{k['rule']}"
                          for k in entry["kills"][:6])
        print(f"{state:9s} {label}: {len(entry['kills'])} of "
              f"{entry['rows']} rows ({entry['cases']} cases) {kills}")
        if entry["not_killed"]:
            print(f"          not killed: {entry['not_killed']}")
    if summary.get("skipped"):
        print(f"skipped by the mode's precondition: {len(summary['skipped'])}")


def _jobs(text: str) -> int:
    value = int(text)
    if not 1 <= value <= MAX_JOBS:
        raise argparse.ArgumentTypeError(f"--jobs is between 1 and {MAX_JOBS}")
    return value


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python tools/mutant_fuzz.py")
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the targets and write OUT/kills.json")
    r.add_argument("out")
    r.add_argument("--oracle-cache", required=True,
                   help="directory of cached oracle rows, outside the project")
    r.add_argument("--labels", nargs="+")
    r.add_argument("--seeds", nargs="+", type=int)
    r.add_argument("--case-file",
                   help="run the saved cases of OUT/kill_cases/<label>.json "
                   "instead of drawn ones")
    r.add_argument("--search", nargs=2, type=int, default=DEFAULT_SEARCH,
                   metavar=("START", "STOP"))
    r.add_argument("--per-label", type=int, default=DEFAULT_PER_LABEL)
    r.add_argument("--mode", choices=MODES)
    r.add_argument("--route", choices=ROUTES)
    r.add_argument("--methods", nargs="+", choices=tuple(fuzz.METHODS))
    r.add_argument("--jobs", type=_jobs, default=1)
    r.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    r.add_argument("--oracle-timeout", type=float,
                   default=DEFAULT_ORACLE_TIMEOUT)
    r.add_argument("--max-rss-gb", type=float, default=fuzz.DEFAULT_MAX_RSS_GB)
    r.add_argument("--oracle-max-rss-gb", type=float,
                   default=fuzz.DEFAULT_ORACLE_MAX_RSS_GB)
    r.add_argument("--alone", action="store_true",
                   help="allow --oracle-max-rss-gb above 8 with --jobs 1")
    r.add_argument("--keep-trees", action="store_true")
    s = sub.add_parser("summary", help="print OUT/kills.json")
    s.add_argument("out")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "summary":
        print_summary(json.loads(
            (pathlib.Path(args.out) / "kills.json").read_text()))
        return 0
    limit = fuzz.DEFAULT_ORACLE_MAX_RSS_GB
    if not 0 < args.oracle_max_rss_gb <= limit and not (
            args.alone and args.jobs == 1 and args.oracle_max_rss_gb > 0):
        raise SystemExit(f"--oracle-max-rss-gb must be in (0, {limit:g}]; "
                         "above that run one case alone (--alone --jobs 1)")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
