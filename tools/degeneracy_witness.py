#!/usr/bin/env python
r"""Which demo configurations can see which historical mutants.

``python tools/degeneracy_witness.py run --scratch DIR --heavy`` evaluates
every configuration of :data:`CONFIGS` at one grid point under the
unmutated ``src/`` and under a copy of the project with one corpus entry of
:data:`SITES` applied, and writes ``docs/verification/blind_spots.json``.
``python tools/degeneracy_witness.py check`` checks that file against the
current corpus, source, configurations and its own rule.

Each entry has a *condition*, read by a probe at the line it changes
(``Site.probe``, ``tools/witness_sentinel.py``): the state in which the
entry changes what its function does, such as ``t_min != 0`` for the entry
that drops ``t_min``.  Each (configuration, mutant) cell gets one of three
states:

* ``not exercised``: the lines the entry changes never run for the
  configuration, or they run and the condition never holds there;
* ``exercised, blind``: the condition holds on some call, and every value
  of the configuration moves by no more than its tolerance;
* ``sees``: the condition holds and some value moves by more than its
  tolerance, or the mutant raises where the unmutated tree returns.

A cell that fits none of them is ``error``: the unmutated tree raises, a
value moves although the condition never holds for it, a value is missing
on one side, or the function's result changes on a call where the probe
found the condition false (the probe would then be incomplete).

Each configuration runs in a fresh ``python -I -B`` process with the tree's
``src`` first on ``sys.path`` and the project root after it (for
``validation/``), and the process checks that the package it loaded comes
from that ``src``.  Under a mutant it installs the sentinel of
``tools/witness_sentinel.py``.  For the leg-order entry each call is tagged
with the vertex multiset of its coupling sum and the package-case group it
belongs to, so a value of the demo 4 package cases is judged only by the
calls of its own diagrams; a YAML configuration's values are judged by
every call.

The tolerance of a value ``v`` is ``rel * |v| + floor * scale``, with
``rel`` and ``floor`` the accuracy the configuration's notes state
(:data:`CONFIGS`, ``source``) and ``scale`` the largest unmutated value of
the configuration (``Config.scale == "config"``) or of the value's
perturbative order (``"order"``).
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EXAMPLES = ROOT / "examples"
RECORD = ROOT / "docs" / "verification" / "blind_spots.json"

NOT_EXERCISED = "not exercised"
BLIND = "exercised, blind"
SEES = "sees"
STATES = (NOT_EXERCISED, BLIND, SEES)
ERROR = "error"
MAX_JOBS = 3
RULE = (
    "not exercised: the changed lines never run, or the entry's condition, "
    "read by a probe at the changed line, never holds; exercised, blind: "
    "the condition holds on some call and every value moves by at most "
    "rel*|value| + floor*scale; sees: some value moves by more, or the "
    "mutant raises where the unmutated tree returns")
#: The per-call counts a cell keeps.
COUNTS = ("line_hits", "calls", "condition_calls", "diverged_calls",
          "diverged_without_condition", "probe_errors")


def _load_sibling(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sentinel_mod = _load_sibling("witness_sentinel")


@dataclasses.dataclass(frozen=True)
class Site:
    """Where a mutant is watched: the function the entry edits, the probe
    of its condition (``tools/witness_sentinel.py``) and the condition in
    words.

    ``equivalent`` lists pairs of results the function's callers treat
    alike; ``decision='r_cache'`` watches the edited function only at the
    kind of R cache the evaluation reads; ``attribution`` tags each call
    (``vertices``: the vertex multiset of its first argument, a coupling
    sum; or ``module:function``)."""

    module: str
    function: str
    probe: tuple
    condition: str
    equivalent: tuple = ()
    decision: str | None = None
    attribution: str | None = None


@dataclasses.dataclass(frozen=True)
class MutantTree:
    """A tree with one mutant applied: ``src`` holds the package, and
    ``head_file`` is the unmutated file the entry edits."""

    label: str
    src: pathlib.Path
    site: Site
    head_file: pathlib.Path


@dataclasses.dataclass(frozen=True)
class Config:
    """A configuration evaluated at one point.

    ``kind`` is ``yaml`` (``target`` a path under ``examples/``, ``args``
    its overrides), ``package_cases`` (``target`` a group of
    :mod:`validation.demo_package_cases`) or ``callable`` (``target``
    ``module:function``, called with ``args``, returning ``{row: value}``
    or ``({row: value}, {row: [tag, ...]})``).
    """

    id: str
    kind: str
    target: str
    args: dict
    rel: float
    floor: float
    demo: str
    source: str
    heavy: bool = False
    scale: str = "config"


SITES = {
    "historical-shared-external-label-accepted": Site(
        "sft_wick.perturbation", "check_distinct_external_labels",
        ("test", "if not repeated:", True),
        "two or more external operators share a spatial label"),
    "historical-callable-coupling-at-one-leg-order": Site(
        "sft_wick.perturbation", "_split_callable_occurrences",
        ("before", "key = key_for(e.name, tuple(e.spatial_args))",
         "next((legs for n, legs in legs_by_key.values() if n == e.name), "
         "tuple(e.spatial_args)) != tuple(e.spatial_args)"),
        "a callable coupling occurs at a leg order other than the first one "
        "seen for it", attribution="vertices"),
    "historical-iso-c-trace": Site(
        "sft_wick.evaluate", "_isotropic_C",
        ("before", "return c if batched else c[0]", "diag.shape[-1] > 1"),
        "an index-free C propagator of two or more components, whose trace "
        "is not c"),
    "historical-t-min-not-reaching-integrators": Site(
        "sft_wick.evaluate", "_integration_t_min",
        ("before", "return t_min", "float(t_min) != 0.0"),
        "an explicit t_min other than 0"),
    "historical-diagonal-flags-on-a-non-diagonal-system": Site(
        "sft_wick.workflow.system", "_probe_offdiagonal",
        ("test", "if _offdiagonal(fn(*args)):", False),
        "the probed callable returns a matrix with an off-diagonal entry",
        # The guards read None ("cannot tell") as they read False.
        equivalent=((None, False),)),
    "historical-r-group-takes-any-position": Site(
        "sft_wick.evaluate", "_require_one_position_per_group",
        ("test", "if pair is None:", True),
        "two points of one R-connected direction group are at different "
        "positions"),
    "historical-r-cache-mismatch-accepted": Site(
        "sft_wick.evaluate", "_r_index_mismatch",
        ("test", "if kind:", False),
        "the R indices of a diagram do not fit the kind of R of the cache "
        "it is evaluated against", decision="r_cache"),
    "historical-no-cut-at-a-pinned-external-time": Site(
        "sft_wick.evaluate", "_kink_constant_cuts",
        ("before", "cut_times.append((var, float(fixed_times[other])))",
         "lower[var] < float(fixed_times[other]) < upper[var]"),
        "a kink against an external pinned at a fixed time lies strictly "
        "inside an integration variable's range"),
}
#: The functions read besides the edited one, per decision.
DECISION_FUNCTIONS = {"r_cache": ("DiagramIntegrand._require_r_indices_match",
                                  "_cache_r_type")}
DECISION_TEXT = {"r_cache": (
    "watched at the kind of R cache the evaluation reads: in "
    "DiagramIntegrand._require_r_indices_match and in the check of "
    "integrate_diagrams")}

_ONE_POINT = {"sweep.positions_grid": {"x": [0.0], "y": [0.5]},
              "sweep.t_final_grid": [1.0]}
_DEMO1 = ("examples/demo1/L2/INTEGRATION_ERROR.md: at t = 1 QMC is exact to "
          "5 digits and GL is converged; floor on each order's largest value")
_DEMO2 = ("examples/demo2/INTERPRETATION.md: relative acceptance target 1e-4 "
          "of the refined channels, the tightest figure the notes give, "
          "applied to these order-0 and order-2 channels; floor on each "
          "order's largest value")
_DEMO3_FK = ("examples/demo3/INTERPRETATION.md: order-2 Gauss-Legendre at "
             "machine precision, and the closed form of T_m at 1e-12 or "
             "better by construction; floor on the largest value")
_DEMO3_F3K = ("examples/demo3/INTERPRETATION.md: order-4 Gauss-Legendre "
              "1.1e-5 relative at n_gauss 12; floor on the largest value")
_DEMO4 = ("validation/compare.py: Gauss-Legendre with an exact C, rel 1e-10 "
          "and floor 1e-12 of the largest value; "
          "tests/test_demo4_asymmetric_noise.py asserts 1e-12")

CONFIGS = (
    Config("demo1_config.yaml", "yaml", "demo1_config.yaml",
           {"overrides": _ONE_POINT}, 1e-5, 1e-5, "demo1", _DEMO1,
           scale="order"),
    Config("demo1/L2/config.yaml", "yaml", "demo1/L2/config.yaml",
           {"overrides": _ONE_POINT}, 1e-5, 1e-5, "demo1", _DEMO1,
           scale="order"),
    Config("demo2_config.yaml", "yaml", "demo2_config.yaml",
           {"overrides": {"sweep.positions_grid": {"x": [0.0], "y": [0.5]}}},
           1e-4, 1e-4, "demo2", _DEMO2, scale="order"),
    Config("demo2/L2/config_FF.yaml", "yaml", "demo2/L2/config_FF.yaml",
           {"overrides": _ONE_POINT}, 1e-4, 1e-4, "demo2", _DEMO2,
           scale="order"),
    Config("demo2/L2/config_FK.yaml", "yaml", "demo2/L2/config_FK.yaml",
           {"overrides": _ONE_POINT}, 1e-4, 1e-4, "demo2", _DEMO2,
           scale="order"),
    Config("demo3/config_FK.yaml", "yaml", "demo3/config_FK.yaml",
           {"overrides": {**_ONE_POINT, "expand.cache_path": None}},
           1e-12, 1e-12, "demo3", _DEMO3_FK),
    Config("demo3/config_F3K.yaml", "yaml", "demo3/config_F3K.yaml",
           {"overrides": {"sweep.t_final_grid": [1.0],
                          "expand.cache_path": None}},
           1e-5, 1e-5, "demo3", _DEMO3_F3K),
    Config("demo4/config_level_a.yaml", "yaml", "demo4/config_level_a.yaml",
           {"overrides": {}}, 1e-10, 1e-12, "demo4", _DEMO4),
    Config("demo4/config_level_a_4pt.yaml", "yaml",
           "demo4/config_level_a_4pt.yaml", {"overrides": {}}, 1e-10, 1e-12,
           "demo4", _DEMO4),
    Config("validation/demo_package_cases.py:demo4", "package_cases",
           "demo4", {}, 1e-10, 1e-12, "demo4", _DEMO4),
    Config("validation/demo_package_cases.py:demo4 order 4", "package_cases",
           "demo4 order 4", {}, 1e-10, 1e-12, "demo4", _DEMO4, heavy=True),
)


# ---------------------------------------------------------------------------
# The corpus and the digests that make a record stale
# ---------------------------------------------------------------------------

def target_mutations() -> list:
    """The corpus entries of :data:`SITES`, in that order."""
    corpus = {m.label: m for m in _load_sibling("mutation_check").load_corpus()}
    return [corpus[label] for label in SITES]


def corpus_digest(mutations: Iterable) -> str:
    """sha256 of the label, path, old and new text of each entry."""
    text = json.dumps([[m.label, m.relpath, m.old, m.new] for m in mutations])
    return hashlib.sha256(text.encode()).hexdigest()


def _digest(paths: Iterable[pathlib.Path], base: pathlib.Path = ROOT) -> str:
    h = hashlib.sha256()
    for path in sorted(set(paths)):
        h.update(path.relative_to(base).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def src_digest(src: pathlib.Path) -> str:
    return _digest(pathlib.Path(src).rglob("*.py"), pathlib.Path(src))


def config_inputs(config: Config) -> list:
    """The files a configuration reads besides ``src/``: its YAML file, the
    Python and YAML files of its demo directory, and ``validation/`` for
    the package cases."""
    files = []
    if config.kind == "yaml":
        files.append(EXAMPLES / config.target)
    demo = EXAMPLES / config.demo
    if demo.is_dir():
        files += [p for p in demo.rglob("*")
                  if p.suffix in (".py", ".yaml", ".yml") and p.is_file()
                  and "__pycache__" not in p.parts]
    if config.kind == "package_cases":
        files += list((ROOT / "validation").glob("*.py"))
    return sorted(set(files))


def inputs_digest(config: Config) -> str:
    return _digest(config_inputs(config))


def _function_source(path: pathlib.Path, qualname: str) -> str:
    text = path.read_text(encoding="utf-8")
    scope = ast.parse(text).body
    *owners, name = qualname.split(".")
    for owner in owners:
        scope = next(n for n in scope if isinstance(n, ast.ClassDef)
                     and n.name == owner).body
    node = next(n for n in scope
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(text, node)


def watched_digest(src: pathlib.Path = ROOT / "src") -> str:
    """sha256 of the unmutated text of every watched function."""
    h = hashlib.sha256()
    for label, site in SITES.items():
        path = pathlib.Path(src, *site.module.split(".")).with_suffix(".py")
        for name in (site.function, *DECISION_FUNCTIONS.get(site.decision,
                                                             ())):
            h.update(f"{label}:{name}".encode())
            h.update(_function_source(path, name).encode())
    return h.hexdigest()


def _tool_digest() -> str:
    return _digest([TOOLS / "degeneracy_witness.py",
                    TOOLS / "witness_sentinel.py"])


# ---------------------------------------------------------------------------
# The child
# ---------------------------------------------------------------------------

def child(spec: dict, sink) -> None:
    """Run the configurations of ``spec``, one JSON line each."""
    sys.path[:0] = [spec["src"], *spec.get("extra_path", []), spec["root"]]
    os.environ["SFT_WICK_QUIET_CACHE"] = "1"
    package = spec.get("package", "sft_wick")
    sentinel = None
    if spec.get("site"):
        raw = dict(spec["site"])
        raw["probe"] = tuple(raw["probe"])
        sentinel = sentinel_mod.install(Site(**raw),
                                        pathlib.Path(spec["head_file"]))
    for raw in spec["configs"]:
        config = Config(**raw)
        if sentinel is not None:
            sentinel.reset()
        out = dict(config=config.id, rows={}, row_tags={}, error=None)
        try:
            rows, tags = sentinel_mod.RUNNERS[config.kind](config)
            out.update(rows=rows, row_tags=tags)
        except Exception as exc:  # noqa: BLE001  (recorded per config)
            out["error"] = f"{type(exc).__name__}: {exc}"[:500]
        sentinel_mod.check_origin(package, pathlib.Path(spec["src"]))
        if sentinel is not None:
            out.update(sentinel.report())
        sink.write(json.dumps(out) + "\n")
        sink.flush()


# ---------------------------------------------------------------------------
# The parent: one child per tree, and the states
# ---------------------------------------------------------------------------

def evaluate(src: pathlib.Path, configs: Sequence[Config], *,
             target: MutantTree | None = None, extra_path: Iterable = (),
             package: str = "sft_wick", env: dict | None = None,
             timeout: float | None = None) -> list:
    """The child's output for ``configs`` on the package in ``src``, with
    the sentinel of ``target`` installed when it is given."""
    spec = dict(src=str(src), root=str(ROOT), package=package,
                extra_path=[str(p) for p in extra_path],
                configs=[dataclasses.asdict(c) for c in configs])
    if target is not None:
        spec.update(site=dataclasses.asdict(target.site),
                    head_file=str(target.head_file))
    child_env = dict(os.environ if env is None else env, OMP_NUM_THREADS="2",
                     PYTHONHASHSEED="0")
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(
            [sys.executable, "-I", "-B", str(pathlib.Path(__file__).resolve()),
             "child"], input=json.dumps(spec), capture_output=True, text=True,
            cwd=cwd, env=child_env, timeout=timeout)
    rows = [json.loads(line) for line in proc.stdout.splitlines()
            if line.startswith("{")]
    if proc.returncode != 0 or len(rows) != len(configs):
        raise RuntimeError(f"the witness child exited {proc.returncode} with "
                           f"{len(rows)} of {len(configs)} rows: "
                           f"{proc.stderr[-2000:]}")
    return rows


def freeze(obj):
    """``obj`` with every list turned into a tuple, recursively."""
    if isinstance(obj, (list, tuple)):
        return tuple(freeze(x) for x in obj)
    return obj


def _scale_key(name: str, how: str) -> str:
    if how == "order":
        for part in name.split("|"):
            if part.startswith("order="):
                return part
    return ""


def row_state(abs_diff: float, tolerance: float, condition: bool,
              line_hits: int) -> str:
    """The state of one value: ``error`` when it moved while the condition
    never held for it (see the module docstring)."""
    if line_hits == 0 or not condition:
        return NOT_EXERCISED if abs_diff == 0.0 else ERROR
    return BLIND if abs_diff <= tolerance else SEES


def classify_rows(head: dict, mutant: dict, row_tags: dict,
                  condition_tags: Sequence, line_hits: int, rel: float,
                  floor: float, scale: str = "config") -> dict:
    """The state of every row (see the module docstring).  ``row_tags``
    gives each row its call tags, or None; a call that held the condition
    with the tag None applies to every row."""
    tags = [freeze(t) for t in condition_tags]
    scales: dict = {}
    for name, h in head.items():
        key = _scale_key(name, scale)
        scales[key] = max(scales.get(key, 0.0), abs(h))
    out = {}
    for name in sorted(set(head) | set(mutant)):
        if name not in head or name not in mutant:
            out[name] = dict(state=ERROR, head=head.get(name),
                             mutant=mutant.get(name), reason=(
                                 "missing under the mutant" if name in head
                                 else "missing under the unmutated tree"))
            continue
        h, m = head[name], mutant[name]
        own = row_tags.get(name)
        own = None if own is None else {freeze(t) for t in own}
        condition = bool(tags) and (own is None or any(
            t is None or t in own for t in tags))
        diff = abs(m - h)
        row_scale = scales[_scale_key(name, scale)]
        tol = rel * abs(h) + floor * row_scale
        state = row_state(diff, tol, condition, line_hits)
        out[name] = dict(state=state, head=h, mutant=m, abs_diff=diff,
                         tolerance=tol, scale=row_scale, condition=condition)
        if state == ERROR:
            out[name]["reason"] = "moved while the condition did not hold"
    return out


def aggregate(states: Iterable[str]) -> str:
    states = set(states)
    for state in (ERROR, SEES, BLIND):
        if state in states:
            return state
    return NOT_EXERCISED


def _rel_diffs(rows: dict) -> list:
    return [r["abs_diff"] / abs(r["head"]) for r in rows.values()
            if r.get("head") and r.get("abs_diff") is not None]


def cell_of(config: Config, label: str, head: dict, mutant: dict) -> dict:
    """The cell of one configuration under one mutant."""
    cell = dict(config=config.id, mutant=label,
                **{k: mutant.get(k, 0) for k in COUNTS},
                first_condition=mutant.get("first_condition"),
                first_divergence=mutant.get("first_divergence"),
                tolerance=dict(rel=config.rel, floor=config.floor,
                               scale=config.scale))
    empty = dict(max_abs_diff=None, max_rel_diff=None, rows={})
    if head["error"]:
        return dict(cell, **empty, state=ERROR, head_error=head["error"],
                    reason="the unmutated tree raises")
    if cell["diverged_without_condition"]:
        return dict(cell, **empty, state=ERROR, reason=(
            "the function's result changed on a call where the probe found "
            "the condition false"))
    if mutant["error"]:
        if cell["line_hits"] and cell["condition_calls"]:
            return dict(cell, **empty, state=SEES, seen_by="mutant raises",
                        mutant_error=mutant["error"])
        return dict(cell, **empty, state=ERROR, mutant_error=mutant["error"],
                    reason="the mutant raises and the condition never held")
    tags = [json.loads(t) for t in mutant.get("condition_tags", [])]
    rows = classify_rows(head["rows"], mutant["rows"], head["row_tags"], tags,
                         cell["line_hits"], config.rel, config.floor,
                         config.scale)
    diffs = [r["abs_diff"] for r in rows.values() if "abs_diff" in r]
    return dict(cell, state=aggregate(r["state"] for r in rows.values()),
                max_abs_diff=max(diffs, default=0.0),
                max_rel_diff=max(_rel_diffs(rows), default=0.0), rows=rows)


def witness(head_src: pathlib.Path, targets: Sequence[MutantTree],
            configs: Sequence[Config], *, extra_path: Iterable = (),
            jobs: int = 1, head_cache: pathlib.Path | None = None,
            log=lambda _msg: None) -> list:
    """Every (configuration, mutant) cell; at most ``jobs`` children run at
    a time.  ``head_cache`` keeps the unmutated rows of each
    configuration."""
    extra_path = tuple(extra_path)
    jobs = max(1, min(jobs, MAX_JOBS))
    package = (targets[0].site.module.split(".")[0] if targets
               else "sft_wick")
    head_rows = _cached_head(head_src, configs, extra_path, head_cache, log,
                             package)

    def one(target):
        log(f"{target.label}: start")
        rows = evaluate(target.src, configs, target=target,
                        extra_path=extra_path, package=package)
        log(f"{target.label}: done")
        return target.label, rows

    cells = []
    with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
        for label, rows in pool.map(one, targets):
            for config, out in zip(configs, rows):
                cells.append(cell_of(config, label, head_rows[config.id],
                                     out))
    return cells


def _cached_head(head_src, configs, extra_path, cache, log, package) -> dict:
    digest = [src_digest(pathlib.Path(head_src)), _tool_digest()]
    out, todo = {}, []
    for config in configs:
        key = hashlib.sha256(json.dumps(
            [dataclasses.asdict(config), *digest,
             inputs_digest(config)]).encode()).hexdigest()
        path = None if cache is None else cache / f"{key}.json"
        if path is not None and path.exists():
            out[config.id] = json.loads(path.read_text())
        else:
            todo.append((config, path))
    if todo:
        log(f"head: {len(todo)} configurations")
        for (config, path), row in zip(todo, evaluate(
                head_src, [c for c, _ in todo], extra_path=extra_path,
                package=package)):
            out[config.id] = row
            if path is not None and not row["error"]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row))
    return out


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def summary_of(cells: list, configs: Sequence[dict], labels: list) -> dict:
    """What each configuration does not see and is blind to, the demo 4
    blind cells, the error cells, and how far the verdicts are from their
    tolerances: the largest relative move of a blind value and the
    smallest largest-relative-move of a cell that sees by value."""
    states = {(c["config"], c["mutant"]): c["state"] for c in cells}
    ids = [c["id"] for c in configs]
    blind_rel = [x for c in cells for x in _rel_diffs(
        {k: r for k, r in c["rows"].items() if r["state"] == BLIND})]
    sees_rel = [c["max_rel_diff"] for c in cells
                if c["state"] == SEES and c.get("max_rel_diff") is not None]
    return dict(
        not_seen={i: [m for m in labels if states[(i, m)] != SEES]
                  for i in ids},
        blind={i: [m for m in labels if states[(i, m)] == BLIND]
               for i in ids},
        demo4_blind=sorted(f"{c['id']}: {m}" for c in configs
                           if c["demo"] == "demo4" for m in labels
                           if states[(c["id"], m)] == BLIND),
        errors=sorted(f"{c['config']}: {c['mutant']}" for c in cells
                      if c["state"] == ERROR),
        margins=dict(blind_max_rel=max(blind_rel, default=None),
                     sees_min_rel=min(sees_rel, default=None)))


def report(cells: list, configs: Sequence[Config], mutations: list,
           head_src: pathlib.Path) -> dict:
    """The JSON of ``docs/verification/blind_spots.json``."""
    labels = [m.label for m in mutations]
    config_rows = [dict(id=c.id, kind=c.kind, target=c.target,
                        overrides=c.args.get("overrides", {}), rel=c.rel,
                        floor=c.floor, scale=c.scale, demo=c.demo,
                        heavy=c.heavy, source=c.source,
                        inputs_sha256=inputs_digest(c))
                   for c in configs]
    return dict(
        rule=RULE, corpus_sha256=corpus_digest(mutations),
        watched_sha256=watched_digest(head_src),
        head_src_sha256=src_digest(head_src),
        configs=config_rows,
        mutants=[dict(label=m.label, path=m.relpath,
                      function=SITES[m.label].function,
                      condition=SITES[m.label].condition,
                      probe=list(SITES[m.label].probe),
                      decision=DECISION_TEXT.get(SITES[m.label].decision))
                 for m in mutations],
        cells=cells, summary=summary_of(cells, config_rows, labels))


def build_trees(mutations: list, scratch: pathlib.Path) -> list:
    """One mutated copy of the project per entry, under ``scratch``."""
    mc = _load_sibling("mutation_check")
    out = []
    for mut in mutations:
        dest = scratch / "trees" / mut.label
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        mc.mutated_copy(mut, dest)
        out.append(MutantTree(mut.label, dest / "src", SITES[mut.label],
                              mut.target(ROOT)))
    return out


def _cell_problems(cell: dict, config: Config | None) -> list:
    """Where a cell's stored state, tolerances or differences do not follow
    from its stored values and its configuration."""
    where = f"{cell['config']}: {cell['mutant']}"
    problems = []
    if config is not None and cell["tolerance"] != dict(
            rel=config.rel, floor=config.floor, scale=config.scale):
        problems.append(f"{where}: the tolerance is not the config's")
    if not cell["rows"]:
        if cell["state"] == ERROR or (cell["state"] == SEES
                                      and cell.get("seen_by")):
            return problems
        return problems + [f"{where}: no rows"]
    rel, floor = cell["tolerance"]["rel"], cell["tolerance"]["floor"]
    how = cell["tolerance"].get("scale", "config")
    scales: dict = {}
    for name, row in cell["rows"].items():
        if row.get("head") is not None:
            key = _scale_key(name, how)
            scales[key] = max(scales.get(key, 0.0), abs(row["head"]))
    for name, row in cell["rows"].items():
        if "abs_diff" not in row:
            if row["state"] != ERROR:
                problems.append(f"{where}: row {name} lacks a value")
            continue
        scale = scales[_scale_key(name, how)]
        if (row["abs_diff"] != abs(row["mutant"] - row["head"])
                or row.get("scale") != scale
                or row["tolerance"] != rel * abs(row["head"]) + floor * scale):
            problems.append(f"{where}: row {name} has a difference or a "
                            "tolerance its values do not give")
        want = row_state(row["abs_diff"], row["tolerance"],
                         row.get("condition", False), cell["line_hits"])
        if row["state"] != want:
            problems.append(f"{where}: row {name} is {row['state']}, its "
                            f"values give {want}")
    if cell["state"] != aggregate(r["state"] for r in cell["rows"].values()):
        problems.append(f"{where}: the state is not the worst of its rows")
    return problems


def check(path: pathlib.Path, configs: Sequence[Config] = CONFIGS) -> list:
    """Problems of the record at ``path`` against the current corpus, the
    watched functions, the configurations and the files they read, and its
    own rule."""
    record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    problems = []
    if record["corpus_sha256"] != corpus_digest(target_mutations()):
        problems.append("the corpus entries changed since the record")
    if record.get("watched_sha256") != watched_digest():
        problems.append("a watched function changed since the record")
    stored = {c["id"]: c for c in record["configs"]}
    if list(stored) != [c.id for c in configs]:
        problems.append("the record's configurations are not those of "
                        "CONFIGS, heavy ones included")
    for config in configs:
        row = stored.get(config.id)
        if row and (row["rel"], row["floor"], row.get("scale")) != (
                config.rel, config.floor, config.scale):
            problems.append(f"{config.id}: tolerance differs from CONFIGS")
        if row and row.get("inputs_sha256") != inputs_digest(config):
            problems.append(f"{config.id}: its files changed since the "
                            "record")
    cells = {(c["config"], c["mutant"]): c for c in record["cells"]}
    if set(cells) != {(i, m) for i in stored for m in SITES}:
        problems.append("the cells do not cover every config and mutant")
    by_id = {c.id: c for c in configs}
    for cell in record["cells"]:
        problems += _cell_problems(cell, by_id.get(cell["config"]))
    summary = summary_of(record["cells"], record["configs"],
                         [m["label"] for m in record["mutants"]])
    if summary != record["summary"]:
        problems.append("the summary does not follow from the cells")
    if summary["errors"]:
        problems.append(f"error cells: {summary['errors']}")
    if summary["demo4_blind"]:
        problems.append(f"demo 4 is blind: {summary['demo4_blind']}")
    return problems


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python tools/degeneracy_witness.py")
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="write the record")
    r.add_argument("--out", default=str(RECORD))
    r.add_argument("--scratch", required=True,
                   help="directory for the mutated trees and the cache")
    r.add_argument("--configs", nargs="+")
    r.add_argument("--labels", nargs="+")
    r.add_argument("--heavy", action="store_true",
                   help="include the configurations marked heavy")
    r.add_argument("--jobs", type=int, default=1, choices=range(1, MAX_JOBS + 1))
    c = sub.add_parser("check", help="check the record")
    c.add_argument("path", nargs="?", default=str(RECORD))
    sub.add_parser("child", help=argparse.SUPPRESS)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "child":
        sink = os.fdopen(os.dup(1), "w", buffering=1)
        os.dup2(2, 1)
        child(json.loads(sys.stdin.read()), sink)
        sink.close()
        return 0
    if args.command == "check":
        problems = check(pathlib.Path(args.path))
        for p in problems:
            print(p)
        return 1 if problems else 0
    scratch = pathlib.Path(args.scratch).resolve()
    if scratch == ROOT or ROOT in scratch.parents:
        raise SystemExit(f"--scratch {scratch} lies inside {ROOT}")
    configs = [c for c in CONFIGS if (args.heavy or not c.heavy)
               and (not args.configs or c.id in args.configs)]
    mutations = [m for m in target_mutations()
                 if not args.labels or m.label in args.labels]
    trees = build_trees(mutations, scratch)
    cells = witness(ROOT / "src", trees, configs, jobs=args.jobs,
                    head_cache=scratch / "head_cache",
                    log=lambda msg: print(msg, flush=True))
    record = report(cells, configs, mutations, ROOT / "src")
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(record, indent=1, sort_keys=False) + "\n",
                   encoding="utf-8")
    for config in configs:
        print(f"{config.id}: not seen {len(record['summary']['not_seen'][config.id])}"
              f", blind {record['summary']['blind'][config.id]}")
    print(f"errors: {record['summary']['errors']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
