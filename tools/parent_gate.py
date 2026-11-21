#!/usr/bin/env python
"""Run a change's tests on its parent's source: the parent-commit gate.

For a change from a parent P to a head C, the gate runs C's added, modified
and renamed ``tests/**/test_*.py`` on three trees: C itself ("head"), P with
P's own versions of the files that exist there ("parent_own"), and an
overlay that is C everywhere except the source root (``src/``), which is
P's ("overlay").  It passes the change when at least one test that passes
on C fails on the overlay with an assertion.  It rejects a change whose
tests pass there, or fail there only through names the change adds, by
raising, at a time or memory limit, or in a run that wrote no report.  It
calls the change undetermined when C's or P's own run of the touched files
fails in the current environment.  ``docs/verification/parent_gate.rst``
describes the verdicts and records a replay over the fix commits of the
history.

Run::

    python tools/parent_gate.py judge --head REV [--base REV] [--json FILE]
    python tools/parent_gate.py replay --range RANGE --out DIR [--record FILE]
    python tools/parent_gate.py render --record FILE [--out FILE] [--check]

``judge`` exits 0 when the change passes, 1 when it is rejected, 3 when it
is undetermined and 2 on a tool error; a gate stopped by SIGTERM, SIGHUP or
SIGINT kills its runs, removes its trees and exits 128 plus the signal.
Standard library only.  The trees, the pytest command and the import-origin
check come from ``tools/mutation_tree.py``; the data types and the reading of
a JUnit report from ``tools/parent_gate_report.py``; a run under its limits
from ``tools/parent_gate_run.py``; the JSON, the job summary, the replay, the
record and its page from ``tools/parent_gate_record.py``.  The record carries
:func:`rules_digest`, a digest of the source of every module-level import,
definition and assignment here, in ``tools/parent_gate_report.py`` and in
``tools/parent_gate_run.py`` that decides a verdict, and of the ``__main__``
guard.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import traceback
import sys
import tarfile
import tempfile
import types
from collections.abc import Sequence
from pathlib import Path


class GateError(Exception):
    """A revision, tree or tool file the gate cannot work with."""


def _load_sibling(name: str) -> types.ModuleType:
    """
    ``tools/<name>.py``, loaded by path under a name of its own.
    """
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_parent_gate_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: Trees, processes and the pytest command: ``tools/mutation_tree.py``.
tree = _load_sibling("mutation_tree")

FORMAT = 1
TOOLS = Path(__file__).resolve().parent
DEFAULT_REPO = TOOLS.parent
RUN_NAMES = ("head", "parent_own", "overlay")
#: A test file the gate runs: ``tests/**/test_*.py``.
TEST_PATH = re.compile(r"^tests/(?:.+/)?test_[^/]*\.py$")
#: Removed from the environment of every git command and every test run.
GIT_ENV_REMOVED = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
                   "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                   "GIT_COMMON_DIR", "GIT_NAMESPACE")
#: Removed from the environment of every test run, with ``PYTEST_XDIST_*``:
#: pytest options and plugins, and Python settings that change what a test
#: sees (``PYTHONOPTIMIZE`` removes assert statements).
PYTEST_ENV_REMOVED = ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                      "PYTEST_DEBUG", "PYTHONWARNINGS", "PYTHONOPTIMIZE", "PYTHONDEVMODE")
PYTEST_ENV_PREFIX = "PYTEST_XDIST_"
#: Added to every pytest command after the flags of ``tools/mutation_tree.py``.
RUN_FLAGS = ("--continue-on-collection-errors", "-q", "-rfEs", "--tb=short")
#: Exit codes of a head or parent's own run that are not a fault: tests
#: passed, tests failed (judged by the report) and no test collected.
OWN_RUN_EXITS = (0, 1, 5)
#: Exit codes of an overlay run that wrote a report the gate can judge.
OVERLAY_EXITS = (0, 1)
TAIL_LINES = 8
#: The exit code of a pytest run that was interrupted: a SIGINT, a
#: KeyboardInterrupt or ``pytest.exit`` without a code.
INTERRUPTED_EXIT = 2
#: The line pytest-xdist writes when a worker process ends without
#: reporting back; it does not say which signal ended the worker.
WORKER_CRASH = re.compile(r"^\[gw\d+\] node down: ", re.M)
ENVIRONMENT_DISTRIBUTIONS = ("numpy", "scipy", "pytest", "pyyaml", "joblib")


#: Data types, the names a source defines and the reading of a JUnit report:
#: ``tools/parent_gate_report.py``.
report_tools = _load_sibling("parent_gate_report")
Settings, Touched, TestResult, Run = (report_tools.Settings, report_tools.Touched,
                                      report_tools.TestResult, report_tools.Run)
Verdict, Judgement, Names = report_tools.Verdict, report_tools.Judgement, report_tools.Names
MESSAGE_LIMIT, ASSERTION_CLASSES = report_tools.MESSAGE_LIMIT, report_tools.ASSERTION_CLASSES
MISSING_NAME_CLASSES = report_tools.MISSING_NAME_CLASSES
FAILING, COUNT_KEYS = report_tools.FAILING, report_tools.COUNT_KEYS
defined_names, read_report = report_tools.defined_names, report_tools.read_report
exception_of, failure_kind = report_tools.exception_of, report_tools.failure_kind
missing_name, source_frame = report_tools.missing_name, report_tools.source_frame
_target_names = report_tools._target_names

#: A run under the time and memory limits: ``tools/parent_gate_run.py``.
run_tools = _load_sibling("parent_gate_run")
Outcome, run_limited, Stopped = run_tools.Outcome, run_tools.run_limited, run_tools.Stopped
MEMORY_MEASURE, EXTERNAL_SIGNALS = run_tools.MEMORY_MEASURE, run_tools.EXTERNAL_SIGNALS


#: The decision of each verdict.  Not part of :func:`rules_digest`: a change
#: of a decision re-aggregates a replay record without re-running it.
DECISION: dict[str, str] = {
    "no_code_change": "pass",
    "no_touched_tests": "reject",
    "environment": "undetermined",
    "no_evidence": "reject",
    "overlay_not_run": "reject",
    "accepted": "pass",
    "crash_only": "reject",
    "resource_only": "reject",
    "new_api_only": "reject",
    "passes_on_parent": "reject",
}
EXIT_CODES = {"pass": 0, "reject": 1, "undetermined": 3}


# ---- git ----

def git_env() -> dict[str, str]:
    """The environment without the variables that point git elsewhere."""
    return {key: value for key, value in os.environ.items() if key not in GIT_ENV_REMOVED}


def git(repo: Path, *args: str) -> str:
    """The standard output of ``git -C repo args``; GateError on failure."""
    proc = subprocess.run(["git", "-C", str(repo), *args], env=git_env(),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        raise GateError(f"git {' '.join(args)}: {lines[-1] if lines else 'failed'}")
    return proc.stdout


def resolve(repo: Path, head: str, base: str | None) -> tuple[str, str, str | None, str]:
    """Head sha, parent sha, base sha and the rule that chose the parent."""
    head_sha = git(repo, "rev-parse", "--verify", f"{head}^{{commit}}").strip()
    if base is None:
        parents = git(repo, "rev-list", "--parents", "-n", "1", head_sha).split()[1:]
        if not parents:
            raise GateError(f"{head_sha} is a root commit: it has no parent to judge against")
        return head_sha, parents[0], None, "first_parent"
    base_sha = git(repo, "rev-parse", "--verify", f"{base}^{{commit}}").strip()
    parent = git(repo, "merge-base", base_sha, head_sha).strip()
    if not parent:
        raise GateError(f"{base_sha} and {head_sha} have no merge base")
    return head_sha, parent, base_sha, "merge_base"


def touched_tests(repo: Path, parent: str, head: str) -> tuple[Touched, ...]:
    """The test files the change adds, modifies or renames, in path order."""
    fields = git(repo, "diff", "--name-status", "-M", "-z", parent, head).split("\0")
    touched, index = [], 0
    while index < len(fields) and fields[index]:
        status = fields[index]
        if status.startswith(("R", "C")):
            old, new = fields[index + 1], fields[index + 2]
            index += 3
        else:
            old = new = fields[index + 1]
            index += 2
        if status[0] not in "AMR" or not TEST_PATH.match(new):
            continue
        touched.append(Touched(new, status[0], None if status[0] == "A" else old))
    return tuple(sorted(touched, key=lambda t: t.path))


def source_changed(repo: Path, parent: str, head: str, source_root: str) -> bool:
    return bool(git(repo, "diff", "--name-only", parent, head, "--", source_root).strip())


# ---- Trees ----

def _refuse_export_attributes(repo: Path, rev: str) -> None:
    """GateError if an attribute could make ``git archive`` differ from ``rev``."""
    names = [n for n in git(repo, "ls-tree", "-r", "--name-only", "-z", rev).split("\0")
             if n == ".gitattributes" or n.endswith("/.gitattributes")]
    texts = [(f"{rev}:{n}", git(repo, "cat-file", "-p", f"{rev}:{n}")) for n in names]
    info = Path(git(repo, "rev-parse", "--git-path", "info/attributes").strip())
    info = info if info.is_absolute() else Path(repo) / info
    if info.is_file():
        texts.append((str(info), info.read_text(encoding="utf-8", errors="replace")))
    for where, text in texts:
        if "export-ignore" in text or "export-subst" in text:
            raise GateError(f"{where} sets export-ignore or export-subst, so "
                            "`git archive` would not reproduce the commit")


def _extract(repo: Path, args: Sequence[str], dest: Path) -> None:
    proc = subprocess.Popen(["git", "-C", str(repo), *args], env=git_env(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
            if hasattr(tarfile, "data_filter"):
                archive.extractall(dest, filter="data")
            else:
                archive.extractall(dest)
    except tarfile.TarError as exc:
        proc.kill()
        proc.wait()
        raise GateError(f"git {' '.join(args)}: {exc}") from None
    stderr = proc.stderr.read().decode("utf-8", "replace")
    if proc.wait() != 0:
        raise GateError(f"git {' '.join(args)}: {stderr.strip() or 'failed'}")


def build_tree(repo: Path, rev: str, dest: Path) -> Path:
    """The tree of ``rev``, from ``git archive``, at ``dest`` (which must not exist)."""
    _refuse_export_attributes(repo, rev)
    dest.mkdir(parents=True)
    _extract(repo, ["archive", "--format=tar", rev], dest)
    return dest


def build_overlay(repo: Path, head: str, parent: str, dest: Path,
                  source_root: str) -> Path:
    """The head's tree with the source root replaced by the parent's."""
    build_tree(repo, head, dest)
    refuse_linked_source(dest, source_root, head)
    if (dest / source_root).exists():
        shutil.rmtree(dest / source_root)
    _refuse_export_attributes(repo, parent)
    _extract(repo, ["archive", "--format=tar", parent, source_root], dest)
    refuse_linked_source(dest, source_root, parent)
    return dest


def refuse_linked_source(tree_dir: Path, source_root: str, rev: str) -> None:
    """GateError if the source root of ``rev`` in ``tree_dir`` is a symbolic link."""
    if (tree_dir / source_root).is_symlink():
        raise GateError(f"{source_root} of {rev[:12]} is a symbolic link, so the overlay "
                        "could not replace the head's source with the parent's")


def detect_package(tree_dir: Path, source_root: str) -> str:
    """The one directory under the source root that holds ``__init__.py``."""
    root = tree_dir / source_root
    found = sorted(p.name for p in root.iterdir()
                   if p.is_dir() and (p / "__init__.py").is_file()) if root.is_dir() else []
    if len(found) != 1:
        raise GateError(f"{source_root}/ of the head holds {len(found)} packages "
                        f"({', '.join(found) or 'none'}); name one with --package")
    return found[0]


def _layout(tree_dir: Path, package: str, settings: Settings):
    layout = tree.Layout(root=tree_dir, package=package,
                         source_root=Path(settings.source_root), corpus=tree_dir)
    try:
        tree.check_layout(layout, corpus=False)
    except tree.HarnessError as exc:
        raise GateError(str(exc)) from None
    problem = tree.origin_problem(tree_dir, layout=layout)
    if problem is not None:
        raise GateError(problem)
    return layout


# ---- Running pytest ----

def run_env(tree_dir: Path, layout, settings: Settings) -> dict[str, str]:
    """:func:`tree_env` with a fixed hash seed and no inherited pytest or git settings.

    ``PYTHONPATH`` holds the tree's own entries only, and git looks for a
    repository no higher than the tree: a tree inside another work tree is
    not taken for part of it.
    """
    env = {key: value for key, value in tree.tree_env(tree_dir, layout=layout).items()
           if key not in PYTEST_ENV_REMOVED and key not in GIT_ENV_REMOVED
           and not key.startswith(PYTEST_ENV_PREFIX)}
    env["PYTHONPATH"] = os.pathsep.join(str(entry) for entry in layout.path_entries(tree_dir))
    env["GIT_CEILING_DIRECTORIES"] = str(Path(tree_dir).parent)
    env["PYTHONHASHSEED"] = settings.hash_seed
    env["LOKY_MAX_CPU_COUNT"] = str(settings.loky_max_cpu)
    return env


def pytest_args(tree_dir: Path, files: Sequence[str], *, layout, settings: Settings,
                report: Path, basetemp: Path) -> list[str]:
    command = [*tree.pytest_command(tree_dir, files, fail_fast=False, layout=layout),
               "-m", settings.markers, *RUN_FLAGS,
               "--rootdir", str(tree_dir), "--confcutdir", str(tree_dir),
               "--basetemp", str(basetemp), "--junit-xml", str(report)]
    if settings.pytest_workers >= 2:
        command += ["-n", str(settings.pytest_workers), "--dist", "worksteal"]
    return command


def run_tests(tree_dir: Path, files: Sequence[str], *, name: str, layout,
              settings: Settings, limit: float, work: Path, head_names: Names,
              parent_names: Names, expected: Sequence[TestResult] = ()) -> Run:
    """Run ``files`` on ``tree_dir`` and read the report."""
    report, log = work / f"{name}.xml", work / f"{name}.log"
    report.unlink(missing_ok=True)
    command = pytest_args(tree_dir, files, layout=layout, settings=settings,
                          report=report, basetemp=work / f"{name}-tmp")
    outcome = run_limited(command, cwd=tree_dir, env=run_env(tree_dir, layout, settings),
                          timeout=limit, max_rss_bytes=int(settings.max_rss_gb * 2**30),
                          log=log)
    _refuse_external_kill(outcome, name)
    killed = "timeout" if outcome.timed_out else "memory" if outcome.memory_killed else None
    has_report = report.is_file() and tree._testcases(report) is not None
    cases = tree._testcases(report) if report.is_file() else None
    results = read_report(cases, tree_dir, expected=expected, head_names=head_names,
                          parent_names=parent_names, source_root=settings.source_root,
                          killed=killed)
    lines = [line for line in outcome.output.splitlines() if line.strip()]
    shown = [part.replace(str(tree_dir), "<tree>").replace(str(work), "<work>")
             for part in command]
    return Run(name, tuple(files), outcome.exit, has_report, outcome.seconds,
               outcome.peak_mb, outcome.timed_out, outcome.memory_killed,
               results, tuple(lines[-TAIL_LINES:]), tuple(shown), outcome.maxrss_mb)


def _refuse_external_kill(outcome: Outcome, name: str) -> None:
    """GateError for a run that something other than the gate may have stopped.

    That is a pytest process ended by a signal of :data:`EXTERNAL_SIGNALS`,
    a run reported as interrupted (:data:`INTERRUPTED_EXIT`), and a run in
    which a pytest-xdist worker crashed (:data:`WORKER_CRASH`): the gate
    cannot tell which signal ended the worker.
    """
    if outcome.exit is None or outcome.timed_out or outcome.memory_killed:
        return
    if outcome.exit < 0 and -outcome.exit in EXTERNAL_SIGNALS:
        raise GateError(f"the {name} run was killed by {signal.Signals(-outcome.exit).name} "
                        "from outside the gate, so it gives no verdict")
    if outcome.exit == INTERRUPTED_EXIT:
        raise GateError(f"the {name} run exited {INTERRUPTED_EXIT}: pytest was interrupted, "
                        "so it gives no verdict")
    if WORKER_CRASH.search(outcome.output):
        raise GateError(f"a pytest-xdist worker of the {name} run crashed, and the gate "
                        "cannot tell whether a signal from outside ended it, so the run "
                        "gives no verdict")


# ---- Deciding ----

def run_fault(run: Run) -> str | None:
    """Why an own run (head or parent's own) is not a clean run, or None."""
    if run.timed_out:
        return "timeout"
    if run.memory_killed:
        return "memory"
    if not run.has_report or run.exit not in OWN_RUN_EXITS:
        return "failure"
    if any(r.outcome in ("failed", "error") for r in run.results):
        return "failure"
    return None


def passed_ids(run: Run) -> list[str]:
    return [r.id for r in run.results if r.outcome == "passed"]


def _named(ids: Sequence[str]) -> str:
    shown = ", ".join(ids[:3])
    return shown + (f" and {len(ids) - 3} more" if len(ids) > 3 else "")


def _own_fault(run: Run, side: str, cause: str) -> Verdict:
    failing = tuple(r for r in run.results if r.outcome in FAILING)
    whose = "the change's own run" if side == "head" else "the parent's own run"
    if cause in ("timeout", "memory"):
        what = f"was killed at its {cause} limit"
    elif failing:
        what = f"has {len(failing)} failing tests: {_named([r.id for r in failing])}"
    else:
        what = f"exited {run.exit} with no report that the gate can read"
    return Verdict("environment", side, cause, f"{whose} of the touched files {what}",
                   failing)


def _overlay_verdict(head: Run, parent_own: Run | None, overlay: Run) -> Verdict:
    if (not overlay.has_report or overlay.exit not in OVERLAY_EXITS) \
            and not (overlay.timed_out or overlay.memory_killed):
        last = overlay.tail[-1] if overlay.tail else "no output"
        report = "" if overlay.has_report else " and wrote no report"
        return Verdict("overlay_not_run", None, None,
                       f"the run on the parent's source exited {overlay.exit}{report}: {last}")
    passed = set(passed_ids(head))
    existing = {r.id for r in parent_own.results} if parent_own else set()
    failures = tuple(dataclasses.replace(r, existing=r.id in existing)
                     for r in overlay.results if r.id in passed and r.outcome in FAILING)
    kinds = {r.kind for r in failures}
    if "assertion" in kinds:
        witnesses = tuple(r for r in failures if r.kind == "assertion")
        return Verdict("accepted", None, None,
                       f"{len(witnesses)} tests that pass on the head fail on the parent's "
                       f"source with an assertion: {_named([r.id for r in witnesses])}",
                       witnesses)
    if "crash" in kinds:
        verdict, how = "crash_only", "only by raising"
    elif kinds & {"timeout", "memory"}:
        verdict, how = "resource_only", "only at a time or memory limit"
    elif kinds:
        verdict, how = "new_api_only", "only through names the change adds"
    else:
        return Verdict("passes_on_parent", None, None,
                       f"the {len(passed)} tests that pass on the head do not fail on "
                       "the parent's source")
    return Verdict(verdict, None, None,
                   f"{len(failures)} tests that pass on the head fail on the parent's "
                   f"source {how}: {_named([r.id for r in failures])}", failures)


def decide(*, src_changed: bool, touched: Sequence[Touched], head: Run | None,
           parent_own: Run | None, overlay: Run | None) -> Verdict:
    """The verdict of a change, the first rule that applies.

    A run that the earlier rules make unnecessary is None.
    """
    if not src_changed:
        return Verdict("no_code_change", None, None,
                       "the change makes no difference under the source root, so the "
                       "gate has nothing to judge")
    if not touched:
        return Verdict("no_touched_tests", None, None,
                       "the change adds, modifies or renames no tests/**/test_*.py")
    fault = run_fault(head)
    if fault is not None:
        return _own_fault(head, "head", fault)
    if not passed_ids(head):
        return Verdict("no_evidence", None, None,
                       "no test of the touched files passes on the head (all skipped, "
                       "deselected or none collected)")
    fault = run_fault(parent_own) if parent_own is not None else None
    if fault is not None:
        return _own_fault(parent_own, "parent", fault)
    if overlay is None:
        raise GateError("no overlay run although no earlier rule decided")
    return _overlay_verdict(head, parent_own, overlay)


def _later_limit(settings: Settings, head: Run) -> float:
    return min(settings.time_limit,
               max(settings.overlay_floor, settings.overlay_factor * head.seconds))


def _run_trees(repo: Path, head: str, parent: str, touched: Sequence[Touched],
               settings: Settings, work: Path) -> tuple[str, dict[str, Run | None]]:
    """Build the trees and make the runs that :func:`decide` needs."""
    runs: dict[str, Run | None] = dict.fromkeys(RUN_NAMES)
    root = settings.source_root
    c_tree = build_tree(repo, head, work / "C")
    refuse_linked_source(c_tree, root, head)
    package = settings.package or detect_package(c_tree, root)
    names = defined_names(c_tree / root)
    common = dict(settings=settings, work=work, head_names=names)
    runs["head"] = run_tests(c_tree, [t.path for t in touched], name="head",
                             layout=_layout(c_tree, package, settings),
                             limit=settings.time_limit, parent_names=names, **common)
    if run_fault(runs["head"]) is not None or not passed_ids(runs["head"]):
        return package, runs
    limit = _later_limit(settings, runs["head"])
    old = [t.old_path for t in touched if t.old_path]
    if old:
        p_tree = build_tree(repo, parent, work / "P")
        runs["parent_own"] = run_tests(p_tree, old, name="parent_own",
                                       layout=_layout(p_tree, package, settings),
                                       limit=limit, parent_names=names, **common)
        if run_fault(runs["parent_own"]) is not None:
            return package, runs
    o_tree = build_overlay(repo, head, parent, work / "O", root)
    runs["overlay"] = run_tests(o_tree, [t.path for t in touched], name="overlay",
                                layout=_layout(o_tree, package, settings), limit=limit,
                                parent_names=defined_names(o_tree / root),
                                expected=runs["head"].results, **common)
    return package, runs


def _clean(work: Path, created: bool) -> None:
    if created:
        shutil.rmtree(work, ignore_errors=True)
        return
    for name in ("C", "P", "O", *(f"{run}-tmp" for run in RUN_NAMES)):
        shutil.rmtree(work / name, ignore_errors=True)


def judge(repo: Path, head: str, base: str | None = None, *,
          settings: Settings = Settings(), work_dir: Path | None = None,
          keep_trees: bool = False) -> Judgement:
    """Judge the change from the parent of ``head`` (or its merge base with ``base``)."""
    repo = Path(repo).resolve()
    head_sha, parent, base_sha, rule = resolve(repo, head, base)
    subject = git(repo, "log", "-1", "--format=%s", head_sha).strip()
    src_changed = source_changed(repo, parent, head_sha, settings.source_root)
    touched = touched_tests(repo, parent, head_sha)
    runs: dict[str, Run | None] = dict.fromkeys(RUN_NAMES)
    package, scrubbed = settings.package, [repo]
    if src_changed and touched:
        created = work_dir is None
        work = Path(tempfile.mkdtemp(prefix="parent-gate-")) if created else Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        work = work.resolve()
        scrubbed.insert(0, work)
        try:
            package, runs = _run_trees(repo, head_sha, parent, touched, settings, work)
        finally:
            if not keep_trees:
                _clean(work, created)
    verdict = decide(src_changed=src_changed, touched=touched, **runs)
    reason = _scrub(verdict.reason, scrubbed)
    return Judgement(head_sha, parent, base_sha, rule, subject, src_changed, tuple(touched),
                     runs, verdict.verdict, DECISION[verdict.verdict], verdict.side,
                     verdict.cause, reason, verdict.witnesses, settings,
                     environment_versions(), package)


# ---- The digest of the rules ----

#: Module-level names of this file that do not decide a verdict.
NOT_RULES = ("DECISION", "EXIT_CODES", "NOT_RULES", "REUSED_RULES", "RULE_FILES",
             "main", "summary_markdown", "to_json", "normalised_source",
             "rules_digest", "environment_versions", "record_tools", "_scrub",
             "_parser", "_add_run_options", "_settings_from", "_judge_main",
             "_definitions", "_kept_lines", "COUNT_KEYS", "_limit", "_command",
             "HANDLED_SIGNALS", "install_signal_handlers", "restore_signal_handlers",
             "_on_signal", "_kill_live", "_is_main_guard")
#: The files whose every definition outside :data:`NOT_RULES` is a rule.
RULE_FILES = ("parent_gate.py", "parent_gate_report.py", "parent_gate_run.py")
#: Definitions of tools/mutation_tree.py whose behaviour the verdicts depend on.
REUSED_RULES = ("HARNESS_ENV", "PYTEST_FLAGS", "Layout", "tree_env",
                "pytest_command", "import_origin", "origin_problem", "_testcases")


def _is_main_guard(node: ast.stmt) -> bool:
    return isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"


def _definitions(module: ast.Module) -> dict[str, ast.stmt]:
    """The module-level statement under each name it binds, as :func:`normalised_source` reads it.

    Definitions and assignments are found under the names they bind,
    imports under the bound or full dotted name, and the ``__main__`` guard
    under ``__main__``.  A later statement under the same name replaces an
    earlier one, and other statements are not found.
    """
    found: dict[str, ast.stmt] = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found[node.name] = node
        elif isinstance(node, ast.Assign):
            found.update((n, node) for t in node.targets for n in _target_names(t))
        elif isinstance(node, ast.AnnAssign):
            found.update((n, node) for n in _target_names(node.target))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            found.update((alias.asname or alias.name, node) for alias in node.names)
        elif _is_main_guard(node):
            found["__main__"] = node
    return found


def _kept_lines(node: ast.stmt, lines: list[bytes]) -> list[str]:
    """The lines of ``node`` that :func:`normalised_source` keeps, stripped on the right.

    ``lines`` are the file's lines in UTF-8.  Docstrings are cut out by their
    :mod:`ast` offsets, which count UTF-8 bytes, so code beside one is kept.
    Blank and comment-only lines are dropped, except inside a string literal
    that spans several lines.
    """
    out, interior, docs = list(lines), set(), []
    for inner in ast.walk(node):
        body = getattr(inner, "body", None)
        if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and body \
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            docs.append(body[0].value)
        if (isinstance(inner, ast.JoinedStr) or (isinstance(inner, ast.Constant) and isinstance(
                inner.value, (str, bytes)))) and inner.end_lineno > inner.lineno:
            interior.update(range(inner.lineno + 1, inner.end_lineno + 1))
    for doc in sorted(docs, key=lambda d: (d.lineno, d.col_offset), reverse=True):
        first, last = doc.lineno - 1, doc.end_lineno - 1
        out[first:last + 1] = ([out[first][:doc.col_offset] + out[last][doc.end_col_offset:]]
                               + [b""] * (last - first))
        interior.difference_update(range(doc.lineno + 1, doc.end_lineno + 1))
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    kept = [out[i - 1].decode("utf-8").rstrip() for i in range(start, node.end_lineno + 1)]
    return [line for i, line in enumerate(kept, start)
            if i in interior or (line.strip() and not line.lstrip().startswith("#"))]


def normalised_source(path: Path, names: Sequence[str] | None) -> str:
    """The source text of the named module-level statements of ``path`` (:func:`_definitions`).

    With ``names`` None, every definition not in :data:`NOT_RULES`, in file
    order.  Each is written as ``name NUL text NUL``, where the text is
    the statement from its first decorator to its last line as
    :func:`_kept_lines` keeps it.
    """
    source = path.read_text(encoding="utf-8")
    lines = source.encode("utf-8").splitlines()
    found = _definitions(ast.parse(source))
    if names is None:
        names = [name for name in found if name not in NOT_RULES]
    missing = [name for name in names if name not in found]
    if missing:
        raise GateError(f"{path.name} no longer defines {', '.join(missing)}, which the "
                        "gate's verdicts depend on")
    return "".join(f"{name}\0" + "\n".join(_kept_lines(found[name], lines)) + "\0"
                   for name in names)


def rules_digest(tools_dir: Path | None = None) -> str:
    """SHA-256 of the rules in :data:`RULE_FILES` and ``mutation_tree.py`` of ``tools_dir``."""
    tools_dir = Path(tools_dir) if tools_dir is not None else TOOLS
    text = "".join(normalised_source(tools_dir / name, None) for name in RULE_FILES)
    text += normalised_source(tools_dir / "mutation_tree.py", REUSED_RULES)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---- Presentation ----

def environment_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(),
                "platform": f"{platform.system()}-{platform.machine()}",
                "memory": MEMORY_MEASURE}
    for dist in ENVIRONMENT_DISTRIBUTIONS:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = "absent"
    return versions


def _scrub(text: str, paths: Sequence[Path]) -> str:
    """``text`` with the given directories, the interpreter's prefix and home shortened."""
    replacements = []
    for path in paths:
        replacements += [(str(Path(path).resolve()), "<work>"), (str(path), "<work>")]
    for prefix in {sys.prefix, sys.base_prefix, sys.exec_prefix}:
        replacements.append((prefix, "<python>"))
    replacements.append((str(Path.home()), "~"))
    for old, new in sorted(replacements, key=lambda pair: -len(pair[0])):
        if old and old != "/":
            text = text.replace(old, new)
    return text


def to_json(j: Judgement, *, public: bool = False) -> dict:
    """The judgement as JSON; ``public`` drops messages, output and commands."""
    return record_tools().judgement_json(j, sys.modules[__name__], public=public)


def summary_markdown(j: Judgement) -> str:
    """The judgement as Markdown for a job summary."""
    return record_tools().summary_markdown(j, sys.modules[__name__])


@functools.lru_cache(maxsize=None)
def record_tools() -> types.ModuleType:
    """``tools/parent_gate_record.py``: population, replay, record and page."""
    return _load_sibling("parent_gate_record")


# ---- Command line ----

def _limit(text: str) -> float:
    """A time or memory limit: a finite number above zero."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not a finite number above zero")
    return value


def _add_run_options(ap: argparse.ArgumentParser) -> None:
    default = Settings()
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO,
                    help="the repository; default: the one that holds this tool")
    ap.add_argument("--keep-trees", action="store_true")
    ap.add_argument("--markers", default=default.markers)
    ap.add_argument("--time-limit", type=_limit, default=default.time_limit, metavar="SECONDS")
    ap.add_argument("--max-rss-gb", type=_limit, default=default.max_rss_gb, metavar="GB",
                    help=f"limit on the summed {MEMORY_MEASURE} of a run's processes")
    ap.add_argument("--pytest-workers", type=int, default=default.pytest_workers, metavar="N")
    ap.add_argument("--source-root", default=default.source_root, metavar="DIR")
    ap.add_argument("--package", default=None, metavar="NAME")


def _settings_from(args: argparse.Namespace) -> Settings:
    return Settings(markers=args.markers, time_limit=args.time_limit,
                    max_rss_gb=args.max_rss_gb, pytest_workers=args.pytest_workers,
                    source_root=args.source_root, package=args.package)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="command", required=True)
    judge_ap = sub.add_parser("judge", help="judge one change")
    judge_ap.add_argument("--head", required=True, metavar="REV")
    judge_ap.add_argument("--base", default=None, metavar="REV",
                          help="judge against the merge base of BASE and HEAD")
    judge_ap.add_argument("--json", type=Path, default=None, metavar="FILE")
    judge_ap.add_argument("--summary", type=Path, default=None, metavar="FILE",
                          help="append the verdict as Markdown")
    judge_ap.add_argument("--work-dir", type=Path, default=None, metavar="DIR")
    _add_run_options(judge_ap)
    replay_ap = sub.add_parser("replay", help="judge the fix commits of a range")
    replay_ap.add_argument("--range", required=True, dest="rev_range", metavar="RANGE")
    replay_ap.add_argument("--earlier-range", default=None, metavar="RANGE")
    replay_ap.add_argument("--out", required=True, type=Path, metavar="DIR")
    replay_ap.add_argument("--jobs", type=int, default=2, metavar="N")
    replay_ap.add_argument("--first", action="append", default=[], metavar="REV")
    replay_ap.add_argument("--record", type=Path, default=None, metavar="FILE")
    _add_run_options(replay_ap)
    render_ap = sub.add_parser("render", help="write the page of a replay record")
    render_ap.add_argument("--record", required=True, type=Path, metavar="FILE")
    render_ap.add_argument("--out", type=Path, default=None, metavar="FILE")
    render_ap.add_argument("--check", action="store_true")
    return ap


def _judge_main(args: argparse.Namespace) -> int:
    j = judge(args.repo, args.head, args.base, settings=_settings_from(args),
              work_dir=args.work_dir, keep_trees=args.keep_trees)
    if args.json is not None:
        args.json.write_text(json.dumps(to_json(j), indent=1) + "\n", encoding="utf-8")
    if args.summary is not None:
        with open(args.summary, "a", encoding="utf-8") as summary:
            summary.write(summary_markdown(j))
    print(f"{j.head[:12]} {j.verdict} ({j.decision}): {j.reason}")
    return EXIT_CODES[j.decision]


def _command(args: argparse.Namespace) -> int:
    if args.command == "judge":
        return _judge_main(args)
    if args.command == "replay":
        return record_tools().replay_main(args, sys.modules[__name__])
    return record_tools().render_main(args, sys.modules[__name__])


def main(argv: Sequence[str] | None = None) -> int:
    """The command line; SIGTERM, SIGHUP and SIGINT kill the live runs and exit."""
    args = _parser().parse_args(argv)
    previous = run_tools.install_signal_handlers()
    try:
        return _command(args)
    except (GateError, tree.HarnessError, OSError, RuntimeError) as exc:
        print(f"parent_gate: error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(f"parent_gate: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        run_tools.restore_signal_handlers(previous)


if __name__ == "__main__":
    sys.exit(main())
