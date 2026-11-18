#!/usr/bin/env python
"""Mutation-test the test suite.

Each entry of the corpus is a textual change to one source file and a pytest
selector naming the tests expected to notice it.  The harness applies the
change, runs the selector, restores the file and reports a verdict.  The
entries are kept in a corpus directory, one module per source area, each
with a list ``ENTRIES`` of ``(label, path, old, new, tests, why)``; ``path``
is relative to the directory of the package under test.  :func:`load_corpus`
reads and checks them.

A :class:`Layout` names the project root, the package, the directory put
on the import path and the corpus.  :data:`DEFAULT_LAYOUT` is that of the
repository that holds this file.  ``--project-root``, ``--package``,
``--source-root``, ``--corpus`` and ``--extra-path`` replace its parts;
``--root`` is the destination of a copy, not the project.

Before a mutation is applied, ``old`` must occur once in the target file
(otherwise ``STALE`` or ``AMBIGUOUS``) and the selector must exit 0 with at
least one passing test on unmutated source (otherwise ``SELECTOR BROKEN``).
Under ``--baseline-only`` an entry that passes both checks is
``BASELINE OK``.  With the mutation applied:

=======  ==============  ===================================================
code     verdict         meaning
=======  ==============  ===================================================
1        ``KILLED``      tests ran and at least one failed
0        ``SURVIVED``    the selected tests pass on the mutated source
other    ``ERROR rc=N``  no test verdict: 2 interrupted (for example by a
                         collection error), 3 internal error, 4 usage error,
                         5 no tests collected, negative if killed by a signal
none     ``TIMEOUT``     the run exceeded its time limit and was killed
=======  ==============  ===================================================

Exit status: 0 if every selected mutation is ``KILLED`` (``BASELINE OK``
under ``--baseline-only``) and every target file has the bytes it had at the
start, or if the requested shard is empty; 1 otherwise; 2 if no label
matches, which includes an empty corpus; 3 if the layout options, the
corpus, the ``--json`` target or the tree cannot be used.

``docs/verification/mutation.rst`` describes the options, which tree a run
imports and how the source is restored.  ``tools/mutation_tree.py`` holds
the code that starts processes on a tree, copies it and keeps the journal;
its names are re-exported here.  A function that needs the layout takes
it as the keyword ``layout``, :data:`DEFAULT_LAYOUT` by default.
:func:`mutated_copy`, :func:`run_in_tree` and :func:`run_mutant_in_copy`
serve other tools that need a tree with one mutant applied.
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import importlib.util
import json
import os
import pathlib
import runpy
import shlex
import signal
import subprocess
import sys
import types
from collections.abc import Callable, Iterator, Sequence


def _load_sibling(name: str) -> types.ModuleType:
    """``tools/<name>.py``, loaded by path.

    ``tools/`` is not a package, and it is not on ``sys.path`` when this file
    is itself loaded by path, as the tests load it.
    """
    path = pathlib.Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


#: Processes, copies and the journal of a tree: ``tools/mutation_tree.py``.
tree = _load_sibling("mutation_tree")
HarnessError = tree.HarnessError
PytestRun = tree.PytestRun
Layout, DEFAULT_LAYOUT = tree.Layout, tree.DEFAULT_LAYOUT
add_layout_arguments, requested_layout = (tree.add_layout_arguments,
                                          tree.requested_layout)
check_layout, default_runner = tree.check_layout, tree.default_runner
PACKAGE, SRC_REL, JOURNAL_NAME = tree.PACKAGE, tree.SRC_REL, tree.JOURNAL_NAME
TAIL_LINES = tree.TAIL_LINES
tree_env, run_in_tree, run_group = tree.tree_env, tree.run_in_tree, tree.run_group
import_origin, origin_problem = tree.import_origin, tree.origin_problem
pytest_command, run_pytest = tree.pytest_command, tree.run_pytest
run_pytest_matrix = tree.run_pytest_matrix
_count_passed, _failed_files = tree._count_passed, tree._failed_files
copy_tree, write_journal, recover = tree.copy_tree, tree.write_journal, tree.recover

#: The project root, package directory and corpus of :data:`DEFAULT_LAYOUT`.
ROOT = DEFAULT_LAYOUT.root
SRC = ROOT / SRC_REL
MUTATION_DIR = DEFAULT_LAYOUT.corpus
#: The fields of a corpus entry, in order.
ENTRY_FIELDS = ("label", "path", "old", "new", "tests", "why")

EXIT_NOTHING_SELECTED = 2
EXIT_SETUP = 3

#: Without ``--timeout``, a mutated run may take this many times the
#: duration of its selector's baseline run, plus :data:`TIMEOUT_PAD` seconds.
TIMEOUT_FACTOR = 10.0
TIMEOUT_PAD = 30.0


@dataclasses.dataclass(frozen=True)
class Mutation:
    """One source edit, plus the tests that must notice it."""

    label: str
    relpath: str
    old: str
    new: str
    tests: str
    why: str

    @property
    def path(self) -> pathlib.Path:
        """The file this mutation edits in the project of the default layout."""
        return self.target(DEFAULT_LAYOUT.root)

    def target(self, root: pathlib.Path, *,
               layout: Layout = DEFAULT_LAYOUT) -> pathlib.Path:
        """The file this mutation edits in the project at ``root``."""
        return root / layout.package_rel / self.relpath


# ---- The corpus ----

def mutation_modules(directory: pathlib.Path | None = None) -> tuple[str, ...]:
    """The names of the corpus modules in ``directory``, sorted.

    ``--shard`` selects by position in the concatenated list, so adding or
    renaming a module changes every shard.
    """
    directory = MUTATION_DIR if directory is None else directory
    return tuple(sorted(path.stem for path in directory.glob("*.py")))


def load_entries(name: str, directory: pathlib.Path | None = None) -> list:
    """``ENTRIES`` of the module ``name`` in ``directory``.

    The file is run by its path, so the corpus directory need not be a
    package or lie on the import path.  Raises :class:`HarnessError` if the
    module cannot be run or has no ``ENTRIES`` list.
    """
    path = (MUTATION_DIR if directory is None else directory) / f"{name}.py"
    try:
        namespace = runpy.run_path(str(path))
    except Exception as exc:
        raise HarnessError(f"{path} cannot be run: {type(exc).__name__}: "
                           f"{exc}") from None
    entries = namespace.get("ENTRIES")
    if not isinstance(entries, (list, tuple)):
        raise HarnessError(f"{path} defines no ENTRIES list")
    return list(entries)


def entry_problem(fields: object, *,
                  layout: Layout = DEFAULT_LAYOUT) -> str | None:
    """Why ``fields`` is not a valid corpus entry, or None."""
    if not isinstance(fields, (tuple, list)) or len(fields) != len(ENTRY_FIELDS):
        return f"is not a tuple of the {len(ENTRY_FIELDS)} fields {ENTRY_FIELDS}"
    if not all(isinstance(field, str) for field in fields):
        return "has a field that is not a string"
    label, relpath, old, new, tests, _ = fields
    rel = pathlib.Path(relpath)
    if not relpath or rel.is_absolute() or ".." in rel.parts:
        return (f"has the path {relpath!r}, which is not relative to "
                f"{layout.package_rel}")
    if not label.strip() or not tests.strip():
        return "has an empty label or selector"
    if not old:
        return "has an empty `old`"
    if old == new:
        return "has `new` equal to `old`"
    return None


def load_corpus(directory: pathlib.Path | None = None, *,
                layout: Layout = DEFAULT_LAYOUT) -> list[Mutation]:
    """Every entry of every module in ``directory``, checked.

    ``directory`` defaults to the corpus of ``layout``.  Raises
    :class:`HarnessError`, naming the module and the position, for an entry
    that :func:`entry_problem` rejects or whose label is taken.
    """
    directory = layout.corpus if directory is None else directory
    corpus: list[Mutation] = []
    where_label: dict[str, str] = {}
    for name in mutation_modules(directory):
        for position, fields in enumerate(load_entries(name, directory)):
            where = f"{directory / name}.py, entry {position},"
            problem = entry_problem(fields, layout=layout)
            if problem is not None:
                raise HarnessError(f"{where} {problem}")
            mut = Mutation(*fields)
            if mut.label in where_label:
                raise HarnessError(f"{where} has the label {mut.label!r} of "
                                   f"{where_label[mut.label].rstrip(',')}")
            where_label[mut.label] = where
            corpus.append(mut)
    return corpus


# ---- Restoring the source ----

#: Files currently holding injected source, and their original bytes.
#:
#: A ``finally`` does not run when Python's default SIGTERM handling ends
#: the process.  A file is registered here before it is written and is
#: restored from a signal handler and from ``atexit`` as well as from the
#: ``finally``.  Restoring twice is harmless.
_IN_FLIGHT: dict[pathlib.Path, bytes] = {}


def _restore_all() -> None:
    for path, original in list(_IN_FLIGHT.items()):
        try:
            path.write_bytes(original)
        except OSError:                                  # pragma: no cover
            print(f"!! COULD NOT RESTORE {path}; its original bytes are in "
                  f"{JOURNAL_NAME} at the root of the tree", file=sys.stderr)
        _IN_FLIGHT.pop(path, None)


def _on_signal(signum, _frame):                          # pragma: no cover
    _restore_all()
    print(f"\n!! interrupted (signal {signum}); source restored",
          file=sys.stderr)
    raise SystemExit(128 + signum)


def install_restore_handlers() -> None:
    """Restore the source when the process is signalled or exits.

    Registering again replaces the earlier ``atexit`` registration.
    """
    atexit.unregister(_restore_all)
    atexit.register(_restore_all)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError) as exc:
            print(f"!! no restore handler for {sig.name} ({exc}); after that "
                  f"signal the next run restores the source from "
                  f"{JOURNAL_NAME}", file=sys.stderr)


# ---- Verdicts and runs ----

KILLED = "KILLED"
SURVIVED = "SURVIVED"
BROKEN = "SELECTOR BROKEN"
ERROR = "ERROR"
TIMEOUT = "TIMEOUT"
STALE = "STALE"
AMBIGUOUS = "AMBIGUOUS"
BASELINE_OK = "BASELINE OK"
VERDICTS = (KILLED, SURVIVED, BROKEN, ERROR, TIMEOUT, STALE, AMBIGUOUS,
            BASELINE_OK)
#: The verdicts that leave the exit status at 0.
PASSING = (KILLED, BASELINE_OK)

EXIT_CODE_MEANING = {
    1: "tests failed",
    2: "interrupted, for example by a collection error",
    3: "pytest internal error",
    4: "command-line usage error",
    5: "no tests collected",
}
NO_TESTS_COLLECTED = 5
#: Appended to the reason when a selector runs no test on the default layout.
DESELECTED_HINT = tree.deselected_hint(DEFAULT_LAYOUT)


@dataclasses.dataclass(frozen=True)
class Baseline:
    """Whether a selector can show a kill, judged on unmutated source."""

    ok: bool
    reason: str
    tail: tuple[str, ...] = ()
    seconds: float = 0.0


@dataclasses.dataclass(frozen=True)
class Result:
    """The verdict on one mutation."""

    verdict: str
    reason: str
    tail: tuple[str, ...] = ()
    returncode: int | None = None      # of the mutated run, if there was one
    failed_files: tuple[str, ...] = ()  # of the mutated run

    @property
    def mark(self) -> str:
        if self.verdict == ERROR:
            return f"{ERROR} rc={self.returncode}"
        return self.verdict


#: Called as ``runner(root, args, timeout=seconds_or_None)``.
Runner = Callable[..., PytestRun]


def _describe(code: int) -> str:
    fallback = "killed by a signal" if code < 0 else "not a pytest exit code"
    return f"exit code {code} ({EXIT_CODE_MEANING.get(code, fallback)})"


# ---- Checks on unmutated source ----

def _judge_baseline(run: PytestRun, hint: str = DESELECTED_HINT) -> Baseline:
    if run.timed_out:
        return Baseline(False, "on unmutated source pytest exceeded the "
                        "--timeout limit", run.tail, run.seconds)
    if run.returncode != 0:
        reason = ("on unmutated source pytest gave "
                  f"{_describe(run.returncode)}, so there is no passing test "
                  "for the mutation to break; exit code 0 is required")
        if run.returncode == NO_TESTS_COLLECTED:
            reason += f". {hint}"
        return Baseline(False, reason, run.tail, run.seconds)
    if not run.passed:
        return Baseline(False, "on unmutated source pytest gave exit code 0 "
                        "but reported no passing test", run.tail, run.seconds)
    return Baseline(True, f"{run.passed} passed on unmutated source",
                    seconds=run.seconds)


class Baselines:
    """Selector results on unmutated source, one pytest run per selector.

    ``timeout`` limits every run, the baselines and the mutated runs.
    Without it a baseline has no limit and a mutated run gets
    :data:`TIMEOUT_FACTOR` times its baseline's duration plus
    :data:`TIMEOUT_PAD` seconds.  ``runner`` defaults to
    :func:`default_runner` of ``layout``.
    """

    def __init__(self, root: pathlib.Path, runner: Runner | None = None,
                 timeout: float | None = None, *,
                 layout: Layout = DEFAULT_LAYOUT) -> None:
        self._root = root
        self._runner = default_runner(layout) if runner is None else runner
        self._hint = tree.deselected_hint(layout)
        self._timeout = timeout
        self._cache: dict[tuple[str, ...], Baseline] = {}

    def check(self, tests: str) -> Baseline:
        try:
            args = tuple(shlex.split(tests))
        except ValueError as exc:
            return Baseline(False, f"the selector cannot be parsed ({exc})")
        if args not in self._cache:
            run = self._runner(self._root, args, timeout=self._timeout)
            self._cache[args] = _judge_baseline(run, self._hint)
        return self._cache[args]

    def mutant_timeout(self, tests: str) -> float:
        """The time limit of a mutated run of the selector ``tests``."""
        if self._timeout is not None:
            return self._timeout
        return TIMEOUT_FACTOR * self.check(tests).seconds + TIMEOUT_PAD


def _occurrences(text: str, old: str) -> int:
    """Occurrences of ``old`` in ``text``, overlapping ones included."""
    count, start = 0, text.find(old)
    while start != -1:
        count += 1
        start = text.find(old, start + 1)
    return count


def _match_problem(mut: Mutation, root: pathlib.Path,
                   layout: Layout) -> tuple[str, str] | None:
    """``(verdict, reason)`` unless ``old`` occurs once in the target file."""
    path = mut.target(root, layout=layout)
    package_rel = layout.package_rel
    where = package_rel / mut.relpath
    if not path.is_file():
        return STALE, f"{where} is not a file"
    resolved = path.resolve()
    if (root / package_rel).resolve() not in resolved.parents:
        return STALE, f"{where} resolves to {resolved}, outside {package_rel}"
    if root.resolve() not in resolved.parents:
        return STALE, f"{where} resolves to {resolved}, outside the tree {root}"
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        return STALE, (f"{where} is not UTF-8 ({exc.reason} at byte "
                       f"{exc.start}); `old` cannot be matched")
    count = _occurrences(text, mut.old)
    if count == 0:
        return STALE, (f"`old` has 0 matches in {where}; the entry tests "
                       "nothing until its `old` string is updated")
    if count > 1:
        return AMBIGUOUS, (f"`old` has {count} matches in {where}, overlapping "
                           "ones included; exactly 1 is required")
    return None


def preflight(mut: Mutation, root: pathlib.Path, baselines: Baselines, *,
              layout: Layout = DEFAULT_LAYOUT) -> Result | None:
    """The reason ``mut`` cannot be run, or None.  Writes nothing."""
    match = _match_problem(mut, root, layout)
    baseline = baselines.check(mut.tests)
    if match is None and baseline.ok:
        return None
    if match is None:
        return Result(BROKEN, baseline.reason, baseline.tail)
    verdict, reason = match
    if baseline.ok:
        return Result(verdict, reason)
    return Result(verdict, f"{reason}; also {BROKEN}: {baseline.reason}",
                  baseline.tail)


def check_mutation(mut: Mutation, root: pathlib.Path, baselines: Baselines, *,
                   layout: Layout = DEFAULT_LAYOUT) -> Result:
    """Verdict of the checks on unmutated source.  Writes nothing."""
    refused = preflight(mut, root, baselines, layout=layout)
    if refused is not None:
        return refused
    return Result(BASELINE_OK, baselines.check(mut.tests).reason)


# ---- Applying a mutation ----

def _mutated_bytes(mut: Mutation, original: bytes) -> bytes:
    return original.decode("utf-8").replace(mut.old, mut.new).encode("utf-8")


def _run_mutated(mut: Mutation, root: pathlib.Path, runner: Runner,
                 timeout: float, layout: Layout) -> PytestRun:
    path = mut.target(root, layout=layout)
    original = path.read_bytes()
    mutated = _mutated_bytes(mut, original)
    _IN_FLIGHT[path] = original
    journal = write_journal(root, path, original, mutated)
    try:
        path.write_bytes(mutated)
        return runner(root, shlex.split(mut.tests), timeout=timeout)
    finally:
        path.write_bytes(original)
        _IN_FLIGHT.pop(path, None)
        journal.unlink(missing_ok=True)


def run_mutation(mut: Mutation, root: pathlib.Path,
                 baselines: Baselines | None = None,
                 runner: Runner | None = None, *,
                 layout: Layout = DEFAULT_LAYOUT) -> Result:
    """Apply ``mut`` in the project at ``root``, run its tests, restore.

    ``runner`` defaults to :func:`default_runner` of ``layout``.
    """
    runner = default_runner(layout) if runner is None else runner
    if baselines is None:
        baselines = Baselines(root, runner, layout=layout)
    refused = preflight(mut, root, baselines, layout=layout)
    if refused is not None:
        return refused
    limit = baselines.mutant_timeout(mut.tests)
    run = _run_mutated(mut, root, runner, limit, layout)
    code = run.returncode
    if run.timed_out:
        return Result(TIMEOUT, f"the mutated run exceeded its limit of "
                      f"{limit:.0f} s and was killed; a hang is not a test "
                      "failure", run.tail, code)
    if code == 1:
        return Result(KILLED, "a selected test failed", run.tail, code,
                      run.failed_files)
    if code == 0:
        return Result(SURVIVED, "the selected tests pass with the mutation "
                      "applied", run.tail, code)
    return Result(ERROR, f"with the mutation applied pytest gave "
                  f"{_describe(code)}; no test verdict", run.tail, code)


def _require_match(mut: Mutation, root: pathlib.Path, layout: Layout) -> None:
    problem = _match_problem(mut, root, layout)
    if problem is not None:
        raise HarnessError(f"{mut.label}: {problem[0]}: {problem[1]}")


def mutated_copy(mut: Mutation, dest: pathlib.Path, *,
                 source: pathlib.Path | None = None,
                 layout: Layout = DEFAULT_LAYOUT) -> pathlib.Path:
    """Copy the tree at ``source`` to ``dest`` with ``mut`` applied there.

    ``source`` defaults to the root of ``layout``.  The mutant stays
    applied: the copy is meant to be discarded.  Raises
    :class:`HarnessError` unless ``old`` occurs once in the target file of
    ``source``, which is checked before anything is copied, and of the copy,
    whose target must not lead out of the copy through a symbolic link.
    """
    source = layout.root if source is None else source
    _require_match(mut, source, layout)
    copy_tree(source, dest)
    _require_match(mut, dest, layout)
    path = mut.target(dest, layout=layout)
    path.write_bytes(_mutated_bytes(mut, path.read_bytes()))
    return dest


def run_mutant_in_copy(mut: Mutation, dest: pathlib.Path,
                       command: Sequence[str], *,
                       source: pathlib.Path | None = None,
                       layout: Layout = DEFAULT_LAYOUT,
                       ) -> subprocess.CompletedProcess[str]:
    """Run ``command`` on a copy of ``source`` at ``dest`` with ``mut`` applied.

    The copy is made by :func:`mutated_copy` and the command is started by
    :func:`run_in_tree`, so the command and its children import the mutated
    package.  Raises :class:`HarnessError` if they would not.
    """
    mutated_copy(mut, dest, source=source, layout=layout)
    problem = origin_problem(dest, layout=layout)
    if problem is not None:
        raise HarnessError(problem)
    return run_in_tree(dest, command, layout=layout)


# ---- The sweep ----

def snapshot(mutations: Sequence[Mutation], root: pathlib.Path, *,
             layout: Layout = DEFAULT_LAYOUT) -> dict[pathlib.Path, bytes]:
    """The bytes of every existing file the mutations target."""
    paths = sorted({mut.target(root, layout=layout) for mut in mutations})
    return {path: path.read_bytes() for path in paths if path.is_file()}


def changed_files(before: dict[pathlib.Path, bytes]) -> tuple[pathlib.Path, ...]:
    """The snapshot files whose bytes now differ from ``before``."""
    return tuple(path for path, data in before.items()
                 if not path.is_file() or path.read_bytes() != data)


@dataclasses.dataclass(frozen=True)
class Report:
    """The outcome of a sweep."""

    results: tuple[tuple[Mutation, Result], ...]
    changed: tuple[pathlib.Path, ...]      # target files that differ at the end
    matrix: bool = False                   # every selected test was run

    def count(self, verdict: str) -> int:
        return sum(result.verdict == verdict for _, result in self.results)

    @property
    def exit_code(self) -> int:
        clean = not self.changed and all(
            result.verdict in PASSING for _, result in self.results)
        return 0 if clean else 1

    def as_json(self) -> dict:
        """The verdicts as plain data, one record per mutation, in order."""
        return {
            "matrix": self.matrix,
            "changed": [str(path) for path in self.changed],
            "mutants": [
                {"label": mut.label, "path": mut.relpath, "tests": mut.tests,
                 "verdict": result.verdict, "returncode": result.returncode,
                 "failed_files": list(result.failed_files)}
                for mut, result in self.results],
        }


def _result_lines(mut: Mutation, result: Result, matrix: bool) -> Iterator[str]:
    yield f"[{result.mark}] {mut.label}"
    yield f"    tests: {mut.tests}"
    if result.verdict == KILLED:
        if matrix:
            yield f"    failed in: {', '.join(result.failed_files)}"
        return
    yield f"    {result.reason}"
    if result.verdict == BASELINE_OK:
        return
    yield from (f"    | {line}" for line in result.tail)
    yield f"    why it matters: {mut.why}"


def _summary_lines(report: Report, baseline_only: bool) -> Iterator[str]:
    shown = ((BASELINE_OK, BROKEN, STALE, AMBIGUOUS) if baseline_only
             else (KILLED, SURVIVED, BROKEN, ERROR, TIMEOUT, STALE, AMBIGUOUS))
    yield " | ".join(f"{verdict.lower()} {report.count(verdict)}"
                     for verdict in shown)
    if report.changed:
        yield "!! SOURCE CHANGED during the run; inspect with `git diff`:"
        yield from (f"    {path}" for path in report.changed)
    else:
        yield "target files have the bytes they had at the start"


def sweep(mutations: Sequence[Mutation], root: pathlib.Path, *,
          baseline_only: bool = False, runner: Runner | None = None,
          matrix: bool = False, timeout: float | None = None,
          layout: Layout = DEFAULT_LAYOUT) -> Report:
    """Run every mutation in turn, print the verdicts and a summary.

    ``runner`` defaults to :func:`default_runner` of ``layout``.  ``matrix``
    says that ``runner`` runs every selected test, so the files it reports
    as failed are the whole list.  ``timeout`` is described in
    :class:`Baselines`.
    """
    runner = default_runner(layout) if runner is None else runner
    before = snapshot(mutations, root, layout=layout)
    baselines = Baselines(root, runner, timeout, layout=layout)
    print(f"{len(mutations)} mutation(s) on {root}\n" + "=" * 72, flush=True)
    results = []
    for mut in mutations:
        result = (check_mutation(mut, root, baselines, layout=layout)
                  if baseline_only
                  else run_mutation(mut, root, baselines, runner, layout=layout))
        print("\n".join(_result_lines(mut, result, matrix)), flush=True)
        results.append((mut, result))
    report = Report(tuple(results), changed_files(before), matrix)
    print("=" * 72)
    print("\n".join(_summary_lines(report, baseline_only)), flush=True)
    return report


# ---- Command line ----

def parse_shard(text: str) -> tuple[int, int]:
    """``"I/N"`` as ``(I, N)`` with ``1 <= I <= N``."""
    try:
        index, count = (int(part) for part in text.split("/"))
    except ValueError:
        raise ValueError(f"expected I/N, got {text!r}") from None
    if not 1 <= index <= count:
        raise ValueError(f"need 1 <= I <= N, got {text!r}")
    return index, count


def _shard_argument(text: str) -> tuple[int, int]:
    try:
        return parse_shard(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _seconds_argument(text: str) -> float:
    try:
        seconds = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected seconds, got {text!r}") from None
    if not seconds > 0:
        raise argparse.ArgumentTypeError(f"need a positive time, got {text!r}")
    return seconds


def select(mutations: Sequence[Mutation], k: str = "",
           shard: tuple[int, int] | None = None) -> list[Mutation]:
    """Mutations whose label contains ``k``, then shard ``I`` of ``N``."""
    matching = [mut for mut in mutations if k.lower() in mut.label.lower()]
    if shard is None:
        return matching
    index, count = shard
    return matching[index - 1::count]


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-k", default="", help="only mutations whose label matches")
    ap.add_argument(
        "--shard", default=None, metavar="I/N", type=_shard_argument,
        help="run only shard I of N (1-based); an empty shard exits 0.")
    ap.add_argument(
        "--baseline-only", action="store_true",
        help="check each entry on unmutated source (`old` occurs once, the "
             "selector passes and selects a test) and write no source.")
    ap.add_argument(
        "--matrix", action="store_true",
        help="run every selected test (no -x) and list the test files that "
             "fail under each mutant.")
    ap.add_argument(
        "--tests", default=None, metavar="SELECTOR",
        help="use this pytest selector for every selected entry in place of "
             "its own, for example 'tests/ -n 4'.")
    ap.add_argument(
        "--timeout", default=None, metavar="SECONDS", type=_seconds_argument,
        help=f"kill any pytest run after SECONDS.  Default: no limit for a "
             f"baseline, and {TIMEOUT_FACTOR:g} times the baseline's duration "
             f"plus {TIMEOUT_PAD:g} s for a mutated run.")
    ap.add_argument(
        "--root", default=None, metavar="DIR", type=pathlib.Path,
        help="copy the project to DIR (absent or empty, outside the project) "
             "and run there; the project is not written.  DIR is the "
             "destination of the copy; the project is --project-root.")
    add_layout_arguments(ap)
    ap.add_argument(
        "--json", default=None, metavar="FILE", type=pathlib.Path,
        help="write the verdicts, and the failed test files of each mutant, "
             "to FILE.")
    return ap


def _nothing_selected(total: int, k: str, corpus: pathlib.Path) -> str:
    if total == 0:
        return (f"no mutation to run: the corpus in {corpus} has no "
                "entry")
    return (f"no mutation selected: no label of the {total} "
            f"{'entry' if total == 1 else 'entries'} contains {k!r}")


def _check_json_target(path: pathlib.Path | None) -> None:
    if path is None:
        return
    parent = path.parent
    if not (parent.is_dir() and os.access(parent, os.W_OK)):
        raise HarnessError(f"--json: {parent} is not a writable directory")


def _start(args: argparse.Namespace, layout: Layout,
           mutations: Sequence[Mutation] | None) -> list[Mutation]:
    """The corpus, after the checks that precede any selection.

    Prints the notice of a journal recovery.  Raises :class:`HarnessError`.
    """
    problem = tree.corpus_problem(args) if mutations is None else None
    if problem is not None:
        raise HarnessError(problem)
    check_layout(layout, corpus=mutations is None)
    corpus = (load_corpus(layout=layout) if mutations is None
              else list(mutations))
    _check_json_target(args.json)
    notice = recover(layout.root, layout=layout)
    if notice is not None:
        print(notice)
    return corpus


def main(argv: Sequence[str] | None = None, *, root: pathlib.Path = ROOT,
         mutations: Sequence[Mutation] | None = None,
         runner: Runner | None = None) -> int:
    """Run the command line; ``root`` is the project without --project-root."""
    install_restore_handlers()
    args = _parser().parse_args(argv)
    layout = requested_layout(args, root=root, corpus=MUTATION_DIR)
    try:
        corpus = _start(args, layout, mutations)
    except HarnessError as exc:
        print(f"cannot start: {exc}")
        return EXIT_SETUP
    matching = select(corpus, args.k)
    if not matching:
        print(_nothing_selected(len(corpus), args.k, layout.corpus))
        return EXIT_NOTHING_SELECTED
    selected = select(matching, "", args.shard)
    if not selected:
        print(f"shard {args.shard[0]}/{args.shard[1]} of the {len(matching)} "
              "selected mutation(s) is empty; nothing to run")
        return 0
    if args.tests is not None:
        selected = [dataclasses.replace(mut, tests=args.tests)
                    for mut in selected]
    try:
        tree_root = tree.prepare_tree(layout, args.root)
    except HarnessError as exc:
        print(f"cannot start: {exc}")
        return EXIT_SETUP
    if runner is None:
        runner = default_runner(layout, matrix=args.matrix)
    report = sweep(selected, tree_root, baseline_only=args.baseline_only,
                   runner=runner, matrix=args.matrix, timeout=args.timeout,
                   layout=layout)
    if args.json is not None:
        args.json.write_text(json.dumps(report.as_json(), indent=2) + "\n",
                             encoding="utf-8")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
