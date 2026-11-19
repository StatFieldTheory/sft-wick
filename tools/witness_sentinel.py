r"""The part of ``tools/degeneracy_witness.py`` that runs in the child
process, next to the package: the runners of the configurations and the
sentinel of a mutant.

:func:`install` replaces the mutated function, in every module of the
package that holds it, by a wrapper.  On each call the wrapper runs the
unmutated function, compiled from the unmutated file in the mutated
module's namespace with a probe at the line the entry changes
(:func:`head_function`), and then the mutated function, on the same
arguments.  It records, per call:

* whether the probe found the entry's *condition* true: the state, read at
  the changed line, in which the entry changes what the function does;
* whether the two functions returned different results or raised
  differently, which cross-checks the probe;
* a tag of the call, when the site attributes calls (for the leg-order
  entry, the vertex multiset of the coupling sum).

A probe has one of two kinds.  ``("test", anchor, value)`` wraps the test of
the ``if`` statement whose first line is ``anchor``, where the entry puts
the constant ``value``: the condition holds when the unmutated test differs
from it.  ``("before", anchor, expression)`` evaluates ``expression`` in the
unmutated function's frame just before the statement whose first line is
``anchor``: the condition holds when it is true, or when it raises.

``sys.monitoring`` counts the line events on the lines of the mutated file
that differ from the unmutated one.  This module does not import the
package at import time; :func:`install` and the runners import it.
"""
from __future__ import annotations

import __future__
import ast
import collections.abc
import difflib
import importlib
import inspect
import json
import pathlib
import reprlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
#: Overrides every YAML run takes: no output files, no worker pool (the
#: sentinel lives in this process).
YAML_FIXED = {"output": [], "sweep.n_jobs": 1}
TEST_PROBE = "__degeneracy_witness_test__"
BEFORE_PROBE = "__degeneracy_witness_before__"
#: The package-case group being evaluated, set by the runner; it prefixes
#: the tag of an attributed call.
GROUP: list = [None]
_REPR = reprlib.Repr(maxstring=120, maxother=120, maxlist=6, maxtuple=6,
                     maxdict=6)


# ---------------------------------------------------------------------------
# Comparing two outcomes
# ---------------------------------------------------------------------------

def _equal(a, b) -> bool:
    if a is b:
        return True
    try:
        import numpy as np
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            x, y = np.asarray(a), np.asarray(b)
            try:
                return bool(np.array_equal(x, y, equal_nan=True))
            except TypeError:                            # object arrays
                return bool(np.array_equal(x, y))
    except ImportError:                                  # pragma: no cover
        pass
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(_equal(x, y) for x, y in zip(a, b)))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001  (an object that refuses ==)
        return repr(a) == repr(b)


def outcome(fn, args, kwargs) -> tuple:
    """``("return", value)`` or ``("raise", exception)`` of one call."""
    try:
        return ("return", fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001  (compared, then re-raised)
        return ("raise", exc)


def same(first: tuple, second: tuple, equivalent=()) -> bool:
    """Whether two outcomes agree: the same exception type and message, or
    equal results.  ``equivalent`` holds pairs of results that the site's
    callers treat alike (compared by identity), such as ``(None, False)``
    for a probe whose callers read "cannot tell" as "no"."""
    if first[0] != second[0]:
        return False
    if first[0] == "raise":
        return (type(first[1]) is type(second[1])
                and str(first[1]) == str(second[1]))
    x, y = first[1], second[1]
    for a, b in equivalent:
        if (x is a and y is b) or (x is b and y is a):
            return True
    return _equal(x, y)


def _materialised(args: tuple, kwargs: dict) -> tuple:
    """Arguments that are iterators, as lists, so both functions see them."""
    def fix(a):
        return list(a) if isinstance(a, collections.abc.Iterator) else a
    return tuple(fix(a) for a in args), {k: fix(v) for k, v in kwargs.items()}


# ---------------------------------------------------------------------------
# The unmutated function with its probe
# ---------------------------------------------------------------------------

def _anchored(func: ast.AST, text: str, anchor: str) -> list:
    """``(block, index, statement)`` of every statement in ``func`` whose
    first source line is ``anchor``."""
    found = []
    for node in ast.walk(func):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for k, stmt in enumerate(block):
                segment = ast.get_source_segment(text, stmt) or ""
                if segment.split("\n", 1)[0].rstrip() == anchor:
                    found.append((block, k, stmt))
    return found


def _call(name: str, args: list) -> ast.Call:
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=args,
                    keywords=[])


def instrument(func: ast.AST, text: str, probe) -> None:
    """Put ``probe`` (see the module docstring) into the function ``func``
    parsed from ``text``.  Raises ``ValueError`` unless the anchor names
    exactly one statement of the right kind."""
    kind, anchor, arg = probe
    found = _anchored(func, text, anchor)
    if len(found) != 1:
        raise ValueError(f"the probe anchor {anchor!r} names {len(found)} "
                         "statements, not one")
    block, k, stmt = found[0]
    if kind == "test":
        if not isinstance(stmt, ast.If):
            raise ValueError(f"the probe anchor {anchor!r} is not an if")
        stmt.test = _call(TEST_PROBE, [stmt.test, ast.Constant(arg)])
    elif kind == "before":
        expr = ast.parse(arg, mode="eval").body
        thunk = ast.Lambda(args=ast.arguments(
            posonlyargs=[], args=[], vararg=None, kwonlyargs=[],
            kw_defaults=[], kwarg=None, defaults=[]), body=expr)
        block.insert(k, ast.Expr(_call(BEFORE_PROBE, [thunk])))
    else:
        raise ValueError(f"unknown probe kind {kind!r}")


def _future_flags(tree: ast.Module) -> int:
    flags = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            for alias in node.names:
                feature = getattr(__future__, alias.name, None)
                if feature is not None:
                    flags |= feature.compiler_flag
    return flags


def head_function(head_text: str, name: str, namespace: dict, probe=None):
    """The function ``name`` of ``head_text``, with ``probe`` put in, compiled
    in ``namespace`` under the file's ``__future__`` imports."""
    tree = ast.parse(head_text)
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == name), None) or next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == name)
    if probe:
        instrument(node, head_text, probe)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    code = compile(module, f"<unmutated {name}>", "exec",
                   flags=_future_flags(tree), dont_inherit=True)
    local: dict = {}
    exec(code, namespace, local)
    return local[name]


def changed_lines(head_text: str, mutant_text: str) -> set:
    """Line numbers of the mutant file that differ from the unmutated one."""
    matcher = difflib.SequenceMatcher(None, head_text.splitlines(),
                                      mutant_text.splitlines(),
                                      autojunk=False)
    lines: set = set()
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            lines.update(range(j1 + 1, max(j2, j1 + 1) + 1))
    return lines


def _code_objects(code: types.CodeType):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _code_objects(const)


# ---------------------------------------------------------------------------
# The record of one watch
# ---------------------------------------------------------------------------

class Sentinel:
    """Counts of one mutant's watch (see the module docstring)."""

    def __init__(self) -> None:
        self._open: list = []
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.line_hits = 0
        self.condition_calls = 0
        self.diverged_calls = 0
        self.diverged_without_condition = 0
        self.probe_errors = 0
        self.condition_tags: list = []
        self.first_condition = None
        self.first_divergence = None

    def test_probe(self, value, mutant_value):
        try:
            on = bool(value) != bool(mutant_value)
        except Exception:  # noqa: BLE001  (counted, and read as holding)
            self.probe_errors += 1
            on = True
        self._mark(on)
        return value

    def before_probe(self, thunk) -> None:
        try:
            on = bool(thunk())
        except Exception:  # noqa: BLE001  (counted, and read as holding)
            self.probe_errors += 1
            on = True
        self._mark(on)

    def _mark(self, on: bool) -> None:
        if on and self._open:
            self._open[-1] = True

    def begin(self) -> None:
        self._open.append(False)

    def end(self) -> bool:
        return self._open.pop()

    def note(self, condition: bool, diverged: bool, tag, detail: str) -> None:
        self.calls += 1
        if condition:
            self.condition_calls += 1
            self.condition_tags.append(tag)
            if self.first_condition is None:
                self.first_condition = detail
        if diverged:
            self.diverged_calls += 1
            if not condition:
                self.diverged_without_condition += 1
            if self.first_divergence is None:
                self.first_divergence = detail

    def report(self) -> dict:
        return dict(calls=self.calls, line_hits=self.line_hits,
                    condition_calls=self.condition_calls,
                    diverged_calls=self.diverged_calls,
                    diverged_without_condition=self.diverged_without_condition,
                    probe_errors=self.probe_errors,
                    condition_tags=sorted({json.dumps(t) for t in
                                           self.condition_tags}),
                    first_condition=self.first_condition,
                    first_divergence=self.first_divergence)


def _watch_lines(sentinel: Sentinel, function, lines: set) -> None:
    mon = sys.monitoring
    for tool in (3, 4):
        try:
            mon.use_tool_id(tool, "degeneracy-witness")
            break
        except ValueError:
            continue
    else:                                                # pragma: no cover
        raise RuntimeError("no free sys.monitoring tool id")

    def on_line(code, line):
        if line in lines:
            sentinel.line_hits += 1
            return None
        return mon.DISABLE

    mon.register_callback(tool, mon.events.LINE, on_line)
    for code in _code_objects(function.__code__):
        mon.set_local_events(tool, code, mon.events.LINE)


def _replace_everywhere(package: str, old, new) -> int:
    count = 0
    for name, module in list(sys.modules.items()):
        if module is None or not (name == package
                                  or name.startswith(package + ".")):
            continue
        for attr, value in list(vars(module).items()):
            if value is old:
                setattr(module, attr, new)
                count += 1
    return count


# ---------------------------------------------------------------------------
# Installing the watch
# ---------------------------------------------------------------------------

def _vertex_tag(args: tuple) -> tuple:
    from validation.lowering_parts import vertex_counts
    return tuple(sorted(vertex_counts(args[0]).items()))


def attribution_of(name: str | None):
    """The function that tags a call from its arguments: ``vertices`` (the
    vertex multiset of a coupling sum) or ``module:function``."""
    if not name:
        return None
    if name == "vertices":
        return _vertex_tag
    module, _, function = name.partition(":")
    return getattr(importlib.import_module(module), function)


def _watch(sentinel: Sentinel, site, unmutated, mutated):
    equivalent = [tuple(pair) for pair in (site.equivalent or ())]
    attribute = attribution_of(site.attribution)

    def call(args: tuple, kwargs: dict):
        args, kwargs = _materialised(args, kwargs)
        sentinel.begin()
        try:
            first = outcome(unmutated, args, kwargs)
        finally:
            condition = sentinel.end()
        second = outcome(mutated, args, kwargs)
        diverged = not same(first, second, equivalent)
        tag = None
        if attribute is not None and (condition or diverged):
            tag = [GROUP[0], attribute(args)]
        sentinel.note(condition, diverged, tag,
                      f"{_REPR.repr(first)} / {_REPR.repr(second)}")
        if second[0] == "raise":
            raise second[1]
        return second[1]

    return call


def _r_cache_decision(module, call, mutated):
    """``call`` for the cache kind an evaluation reads.  The edited function
    is evaluated for both kinds of cache by
    ``DiagramIntegrand._r_index_mismatches``; inside
    ``_require_r_indices_match`` the call for the other kind is passed to
    the mutant unwatched.  The check of ``integrate_diagrams``, made at the
    cache's own kind, is watched."""
    used: list = []
    cls = module.DiagramIntegrand
    original = cls._require_r_indices_match

    def require(self, cache):
        used.append(module._cache_r_type(cache))
        try:
            return original(self, cache)
        finally:
            used.pop()

    cls._require_r_indices_match = require

    def decided(args: tuple, kwargs: dict):
        iso_R = args[1] if len(args) > 1 else kwargs.get("iso_R")
        if used and used[-1] is not None and bool(iso_R) != used[-1]:
            return mutated(*args, **kwargs)
        return call(args, kwargs)

    return decided


def install(site, head_file: pathlib.Path) -> Sentinel:
    """Watch ``site`` (a ``Site`` of ``tools/degeneracy_witness.py``) in the
    loaded package, the unmutated text being that of ``head_file``."""
    module = importlib.import_module(site.module)
    mutated = getattr(module, site.function)
    head_text = pathlib.Path(head_file).read_text(encoding="utf-8")
    mutant_text = pathlib.Path(inspect.getsourcefile(mutated)).read_text(
        encoding="utf-8")
    lines = changed_lines(head_text, mutant_text)
    if not lines:
        raise RuntimeError(f"{site.module} has no line that differs from "
                           f"{head_file}: the entry is not applied")
    sentinel = Sentinel()
    namespace = vars(module)
    namespace[TEST_PROBE] = sentinel.test_probe
    namespace[BEFORE_PROBE] = sentinel.before_probe
    unmutated = head_function(head_text, site.function, namespace,
                              site.probe)
    _watch_lines(sentinel, mutated, lines)
    call = _watch(sentinel, site, unmutated, mutated)
    if site.decision == "r_cache":
        call = _r_cache_decision(module, call, mutated)

    def wrapper(*args, **kwargs):
        return call(args, kwargs)

    wrapper.__wrapped__ = mutated
    if not _replace_everywhere(site.module.split(".")[0], mutated, wrapper):
        raise RuntimeError(f"{site.module}.{site.function} was not replaced")
    return sentinel


def check_origin(package: str, src: pathlib.Path) -> None:
    """Raise ``RuntimeError`` if a loaded module of ``package`` comes from
    outside ``src``."""
    src = pathlib.Path(src).resolve()
    for name, module in list(sys.modules.items()):
        if module is None or not (name == package
                                  or name.startswith(package + ".")):
            continue
        path = getattr(module, "__file__", None)
        if path and src not in pathlib.Path(path).resolve().parents:
            raise RuntimeError(f"{name} was loaded from {path}, not from "
                               f"{src}")


# ---------------------------------------------------------------------------
# The configurations (``Config`` of tools/degeneracy_witness.py)
# ---------------------------------------------------------------------------

def _row_name(row: dict) -> str:
    return "|".join(f"{k}={row[k]}" for k in sorted(row)
                    if k not in ("value", "error", "n_diagrams"))


def run_yaml(config) -> tuple:
    from sft_wick.workflow.config import load_workflow_config, run_workflow
    overrides = {**config.args.get("overrides", {}), **YAML_FIXED}
    cfg = load_workflow_config(EXAMPLES / config.target, overrides=overrides)
    _sweep, totals = run_workflow(cfg, progress=False)
    rows = {_row_name(r): float(r["value"])
            for r in totals.to_dict(orient="records")}
    return rows, {name: None for name in rows}


def run_package_cases(config) -> tuple:
    from validation.demo_package_cases import package_cases
    from validation.lower import Lowered

    groups: dict = {}
    for case in package_cases():
        if case.group == config.target:
            key = (case.model, case.observable, case.max_weight, case.orders,
                   case.settings, case.c_source, case.flags)
            groups.setdefault(key, []).append(case)
    rows, tags = {}, {}
    for k, ((model, obs, weight, orders, settings, c_source, flags),
            cases) in enumerate(groups.items()):
        GROUP[0] = k
        try:
            got = Lowered(model, obs, weight, dict(flags), c_source).evaluate(
                cases[0].method, orders=orders, **dict(settings))
        finally:
            GROUP[0] = None
        channels = model.channels()
        for case in cases:
            rows[case.source] = float(sum(got.get(t, (0.0, 0.0))[0]
                                          for t in case.tags))
            tags[case.source] = [
                [k, sorted([ch, n] for ch, n in zip(channels, t) if n)]
                for t in case.tags]
    return rows, tags


def run_callable(config) -> tuple:
    module, _, name = config.target.partition(":")
    out = getattr(importlib.import_module(module), name)(**config.args)
    rows, tags = out if isinstance(out, tuple) else (out, None)
    tags = tags or {}
    return ({k: float(v) for k, v in rows.items()},
            {k: (None if tags.get(k) is None
                 else [[None, t] for t in tags[k]]) for k in rows})


RUNNERS = {"yaml": run_yaml, "package_cases": run_package_cases,
           "callable": run_callable}
