"""The layout, processes, copies and journal of a tree under mutation test.

Used by ``tools/mutation_check.py``, which loads this file by path and
re-exports its names.  Nothing here knows about a corpus entry: a tree is a
project root, and a :class:`Layout` says where in it the package lies.
Every function that needs the layout takes it as the keyword ``layout``,
which defaults to :data:`DEFAULT_LAYOUT`, the layout of the repository that
holds this file.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import functools
import hashlib
import json
import keyword
import os
import pathlib
import shlex
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence


class HarnessError(Exception):
    """The corpus or a tree is not in a state the harness can work on."""


# ---- The layout of a project ----

@dataclasses.dataclass(frozen=True)
class Layout:
    """Where a project keeps the package under test and the corpus.

    ``root`` is the project root, the tree a sweep runs in unless it is
    copied.  ``package`` is the import name of the package, dotted for a
    subpackage.  ``source_root`` is the directory, relative to a project
    root, that is put on the import path.  ``corpus`` is the directory of
    the corpus modules; it may lie outside the project.  ``extra_paths``
    are further directories, relative to a project root, put on the import
    path after the source root.  ``deselected_markers`` are the markers
    the project's pytest ``addopts`` deselects, named in the reason when a
    selector runs no test.  The functions that take a tree as ``root`` use
    only the relative parts, so one layout serves the project and a copy
    of it.
    """

    root: pathlib.Path
    package: str
    source_root: pathlib.Path
    corpus: pathlib.Path
    extra_paths: tuple[pathlib.Path, ...] = ()
    deselected_markers: tuple[str, ...] = ()

    @property
    def package_rel(self) -> pathlib.Path:
        """The package directory relative to a project root.

        The ``path`` of a corpus entry is relative to it.
        """
        return self.source_root.joinpath(*self.package.split("."))

    def path_entry(self, root: pathlib.Path) -> pathlib.Path:
        """The absolute directory put first on the import path of ``root``."""
        return root.resolve() / self.source_root

    def path_entries(self, root: pathlib.Path) -> tuple[pathlib.Path, ...]:
        """The absolute directories put on the import path of ``root``, in order."""
        return (self.path_entry(root),
                *(root.resolve() / extra for extra in self.extra_paths))


_TOOLS = pathlib.Path(__file__).resolve().parent

#: The layout of the repository that holds this file.
DEFAULT_LAYOUT = Layout(
    root=_TOOLS.parent,
    package="sft_wick",
    source_root=pathlib.Path("src"),
    corpus=_TOOLS / "mutations",
    deselected_markers=("slow", "heavy"),
)
#: The parts of :data:`DEFAULT_LAYOUT` under their earlier names.
PACKAGE = DEFAULT_LAYOUT.package
PYTHONPATH_REL = DEFAULT_LAYOUT.source_root
SRC_REL = DEFAULT_LAYOUT.package_rel



def deselected_hint(layout: Layout) -> str:
    """Appended to the reason when a selector runs no test on ``layout``."""
    if not layout.deselected_markers:
        return ("check the selector, and whether the project's addopts "
                "deselects its tests by marker; a -m in the selector "
                "replaces the one in addopts")
    markers = " or ".join(layout.deselected_markers)
    return (f"the project's addopts deselects tests marked {markers}; a "
            "selector for one of them needs its own -m")


# ---- The options that name a layout ----

def add_layout_arguments(ap: argparse.ArgumentParser) -> None:
    """Add ``--project-root``, ``--package``, ``--source-root``, ``--corpus``."""
    default = DEFAULT_LAYOUT
    ap.add_argument(
        "--project-root", default=None, metavar="DIR", type=pathlib.Path,
        help="the project to test, swept in place unless --root is given.  "
             "Default: the repository that holds this tool.")
    ap.add_argument(
        "--package", default=None, metavar="NAME",
        help=f"import name of the package under test, dotted for a "
             f"subpackage.  Default: {default.package}.")
    ap.add_argument(
        "--source-root", default=None, metavar="DIR", type=pathlib.Path,
        help=f"the directory, relative to the project root, that is put on "
             f"the import path; the package lies under it.  Default: "
             f"{default.source_root}.")
    ap.add_argument(
        "--corpus", default=None, metavar="DIR", type=pathlib.Path,
        help="the directory of the corpus modules; it may lie outside the "
             "project.  Required with --project-root, --package or "
             "--source-root.  Default: mutations/ beside this tool.")
    ap.add_argument(
        "--extra-path", default=None, metavar="DIR", type=pathlib.Path,
        action="append",
        help="a further directory, relative to the project root, put on the "
             "import path after the source root; repeatable.  Default: none.")


def requested_layout(args: argparse.Namespace, *,
                     root: pathlib.Path | None = None,
                     corpus: pathlib.Path | None = None) -> Layout:
    """The layout the options ask for, the parts of the default elsewhere.

    ``root`` stands in for ``--project-root`` when that is not given, and
    the corpus of the default is ``corpus``.
    """
    def given(value, fallback):
        return fallback if value is None else value

    default = DEFAULT_LAYOUT
    extra = args.extra_path
    return Layout(root=given(args.project_root, given(root, default.root)),
                  package=given(args.package, default.package),
                  source_root=given(args.source_root, default.source_root),
                  corpus=given(args.corpus, given(corpus, default.corpus)),
                  extra_paths=default.extra_paths if extra is None
                  else tuple(extra),
                  deselected_markers=default.deselected_markers
                  if args.project_root is None else ())


def corpus_problem(args: argparse.Namespace) -> str | None:
    """Why the options name another project but not its corpus, or None."""
    named = [option for option, value in (
        ("--project-root", args.project_root), ("--package", args.package),
        ("--source-root", args.source_root)) if value is not None]
    if not named or args.corpus is not None:
        return None
    return (f"--corpus: required with {', '.join(named)}; the default corpus "
            "belongs to the default project")


def _is_import_name(name: str) -> bool:
    return all(part.isidentifier() and not keyword.iskeyword(part)
               for part in name.split("."))


def _check_relative_dir(option: str, root: pathlib.Path,
                        rel: pathlib.Path) -> None:
    if rel.is_absolute() or ".." in rel.parts:
        raise HarnessError(f"{option}: {rel} is not a path relative to the "
                           "project root without ..")
    if not (root / rel).is_dir():
        raise HarnessError(f"{option}: {root / rel} is not a directory")


def check_layout(layout: Layout, *, corpus: bool = True) -> None:
    """Raise :class:`HarnessError`, naming the option, unless ``layout`` works.

    The package name must be a dotted import name, the project root a
    directory, the source root and each extra path a relative path without
    ``..`` to a directory, and the package a directory under the source
    root that does not resolve, through a symbolic link, outside the
    project root.  With ``corpus``, the corpus must be a directory.  The
    corpus may lie outside the project.
    """
    if not _is_import_name(layout.package):
        raise HarnessError(f"--package: {layout.package!r} is not a dotted "
                           "import name such as pkg or pkg.sub")
    root = layout.root
    if not root.is_dir():
        raise HarnessError(f"--project-root: {root} is not a directory")
    _check_relative_dir("--source-root", root, layout.source_root)
    for extra in layout.extra_paths:
        _check_relative_dir("--extra-path", root, extra)
    package_dir = root / layout.package_rel
    if not package_dir.is_dir():
        raise HarnessError(f"--package: {package_dir} is not a directory")
    if not _is_within(package_dir.resolve(), root.resolve()):
        raise HarnessError(f"--package: {package_dir} resolves to "
                           f"{package_dir.resolve()}, outside the project "
                           f"root {root.resolve()}")
    if corpus and not layout.corpus.is_dir():
        raise HarnessError(f"--corpus: {layout.corpus} is not a directory")


#: Written in the tree before a mutant is and removed after the restore.  A
#: run killed by SIGKILL leaves it behind, and the next run reads it.
JOURNAL_NAME = ".mutation-inflight.json"

#: Names left out of a copy of the tree, at any depth: repository data,
#: agent state, caches and the journal.
COPY_IGNORED = (".git", ".claude", ".agent-state", "__pycache__",
                ".pytest_cache", ".hypothesis", ".ruff_cache", ".mypy_cache",
                "_build", "*.egg-info", JOURNAL_NAME)

#: Set to ``1`` in the environment of every process the harness starts, so
#: a test that compares the corpus with the source can skip while a sweep
#: has the source under test.
HARNESS_ENV = "MUTATION_HARNESS"

#: Seconds to wait for the output of a process group after killing it.
KILL_GRACE = 10.0


#: Added to every pytest invocation.  ``xunit1`` makes the JUnit report name
#: the file of each test case.
PYTEST_FLAGS = ("--no-header", "-p", "no:cacheprovider",
                "-o", "junit_family=xunit1")
#: Stops a run at the first failure.  Left out for the kill matrix.
FAIL_FAST_FLAG = "-x"
#: Lines of pytest output kept for the report.
TAIL_LINES = 8


@dataclasses.dataclass(frozen=True)
class PytestRun:
    """What one pytest invocation reported."""

    returncode: int
    passed: int | None                 # None if pytest wrote no JUnit report
    tail: tuple[str, ...]
    failed_files: tuple[str, ...] = ()  # files with a failing or erroring test
    seconds: float = 0.0               # wall-clock duration
    timed_out: bool = False            # killed at its time limit


# ---- The journal ----

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_journal(root: pathlib.Path, path: pathlib.Path, original: bytes,
                   mutated: bytes) -> pathlib.Path:
    journal = root / JOURNAL_NAME
    record = {"path": path.relative_to(root).as_posix(),
              "original": base64.b64encode(original).decode("ascii"),
              "mutant_sha256": _sha256(mutated)}
    journal.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return journal


def _read_journal(journal: pathlib.Path, layout: Layout,
                  ) -> tuple[str, bytes, str]:
    try:
        record = json.loads(journal.read_text(encoding="utf-8"))
        return (record["path"], base64.b64decode(record["original"]),
                record["mutant_sha256"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HarnessError(f"{journal} cannot be read ({exc}); check "
                           f"{layout.package_rel} with `git diff`, then "
                           "remove it") from None


def recover(root: pathlib.Path, *,
            layout: Layout = DEFAULT_LAYOUT) -> str | None:
    """Undo the mutant a killed run left in ``root``; a message, or None.

    The journal of the killed run names the file, its original bytes and the
    digest of the mutant.  The file is restored only if it still holds that
    mutant.  Raises :class:`HarnessError` if it holds anything else.
    """
    journal = root / JOURNAL_NAME
    if not journal.exists():
        return None
    relpath, original, mutant_sha256 = _read_journal(journal, layout)
    path = root / relpath
    current = path.read_bytes() if path.is_file() else None
    if current is None or (current != original
                           and _sha256(current) != mutant_sha256):
        raise HarnessError(
            f"a killed run left {journal}: {path} held a mutant then and is "
            "missing or has changed since.  Check the file; its original "
            "bytes are in the journal, base64-encoded.  Then remove the journal")
    if current != original:
        path.write_bytes(original)
    journal.unlink()
    return f"{journal}: {path} holds its original bytes after a killed run"


# ---- Running a command on a tree ----

def tree_env(root: pathlib.Path, *,
             layout: Layout = DEFAULT_LAYOUT) -> dict[str, str]:
    """The environment of every process started on the tree at ``root``.

    The source root of ``layout`` in ``root``, made absolute, is first in
    ``PYTHONPATH``, then its extra paths, ahead of the entries that ``site``
    adds for installed packages, so a child process of a test imports the
    package of this tree.  ``PYTHONDONTWRITEBYTECODE`` is
    set so that no run leaves a ``.pyc`` behind.  A ``.pyc`` is validated by
    the source's size and its modification time in whole seconds, so one
    written less than a second before a same-length mutation would be loaded
    in place of the mutated source.  :data:`HARNESS_ENV` is set to ``1``.
    """
    inherited = os.environ.get("PYTHONPATH")
    entries = ([str(entry) for entry in layout.path_entries(root)]
               + ([inherited] if inherited else []))
    return {**os.environ, "PYTHONPATH": os.pathsep.join(entries),
            "PYTHONDONTWRITEBYTECODE": "1", HARNESS_ENV: "1"}


def run_in_tree(root: pathlib.Path, command: Sequence[str], *,
                layout: Layout = DEFAULT_LAYOUT,
                ) -> subprocess.CompletedProcess[str]:
    """Run ``command`` in ``root`` with the environment of :func:`tree_env`.

    Standard output and standard error are captured as text.
    """
    return subprocess.run(
        list(command), cwd=root, capture_output=True, encoding="utf-8",
        errors="replace", env=tree_env(root, layout=layout))


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # The group has exited.  macOS answers EPERM for a group whose only
        # members are zombies.
        return


def _drain(proc: subprocess.Popen) -> str:
    """The rest of the output of a killed process, then reap it."""
    try:
        return proc.communicate(timeout=KILL_GRACE)[0]
    except subprocess.TimeoutExpired:
        # A process that left the group still holds the pipe.
        proc.wait()
        return ""


def run_group(command: Sequence[str], *, cwd: pathlib.Path,
              env: dict[str, str], timeout: float | None,
              ) -> tuple[int, str, bool]:
    """Run ``command`` in a session of its own.

    Returns the exit code, standard output and standard error together, and
    whether the time limit was reached.  At the limit, and on any exception
    (a signal handler raising ``SystemExit`` included), the whole process
    group is killed, so pytest-xdist workers and the tests' own children end
    with it.
    """
    proc = subprocess.Popen(
        list(command), cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
        start_new_session=True)
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        output = _drain(proc)
        return proc.returncode, output, True
    except BaseException:
        _kill_group(proc)
        proc.wait()
        raise
    return proc.returncode, output, False


def import_origin(root: pathlib.Path, *,
                  layout: Layout = DEFAULT_LAYOUT) -> pathlib.Path:
    """The directory a process started on ``root`` imports the package from.

    For a namespace package, the first directory of its ``__path__``, which
    is the one searched first for its modules.
    """
    code = (f"import importlib, os, sys; "
            f"m = importlib.import_module({layout.package!r}); "
            f"p = getattr(m, '__path__', None) or [os.path.dirname(m.__file__)]; "
            f"sys.stdout.write(list(p)[0])")
    proc = run_in_tree(root, [sys.executable, "-c", code], layout=layout)
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        raise HarnessError(f"`import {layout.package}` failed in {root}: "
                           f"{lines[-1] if lines else 'no output'}")
    return pathlib.Path(proc.stdout).resolve()


def origin_problem(root: pathlib.Path, *,
                   layout: Layout = DEFAULT_LAYOUT) -> str | None:
    """Why runs on ``root`` would not test its source, or None."""
    expected = (root / layout.package_rel).resolve()
    try:
        found = import_origin(root, layout=layout)
    except HarnessError as exc:
        return str(exc)
    if found != expected:
        return (f"a process started in {root} imports {layout.package} from "
                f"{found}, not from {expected}; a mutation there would not "
                "be tested")
    return None


def pytest_command(root: pathlib.Path, args: Sequence[str], *,
                   fail_fast: bool,
                   layout: Layout = DEFAULT_LAYOUT) -> list[str]:
    """The pytest command line for ``args`` on the tree at ``root``.

    Without the junit report option, which names a temporary file.  The
    paths are absolute: pytest reads a relative ``pythonpath`` against its
    root directory, which is ``root`` itself.  They are quoted, because
    pytest splits the value with :func:`shlex.split`.
    """
    paths = " ".join(shlex.quote(str(entry))
                     for entry in layout.path_entries(root))
    flags = [*PYTEST_FLAGS, "-o", f"pythonpath={paths}"]
    if fail_fast:
        flags.append(FAIL_FAST_FLAG)
    return [sys.executable, "-m", "pytest", *args, *flags]


def _testcases(report: pathlib.Path) -> list[ET.Element] | None:
    try:
        return list(ET.parse(report).getroot().iter("testcase"))
    except (OSError, ET.ParseError):
        return None


def _count_passed(report: pathlib.Path) -> int | None:
    """Passing test cases in a JUnit report; None if there is no report.

    The count comes from the XML because the console summary depends on the
    verbosity flags in effect.
    """
    cases = _testcases(report)
    if cases is None:
        return None
    not_passed = ("skipped", "failure", "error")
    return sum(all(case.find(tag) is None for tag in not_passed)
               for case in cases)


def _failed_files(report: pathlib.Path) -> tuple[str, ...]:
    """The test files with a ``failure`` or ``error`` case in a JUnit report.

    A case without a ``file`` attribute is listed by its ``classname``.
    """
    cases = _testcases(report) or []
    failed = {case.get("file") or case.get("classname") or "?"
              for case in cases
              if case.find("failure") is not None or case.find("error") is not None}
    return tuple(sorted(failed))


def _run_pytest(root: pathlib.Path, args: Sequence[str], *, fail_fast: bool,
                timeout: float | None, layout: Layout) -> PytestRun:
    with tempfile.TemporaryDirectory() as tmp:
        report = pathlib.Path(tmp) / "junit.xml"
        start = time.monotonic()
        code, output, timed_out = run_group(
            [*pytest_command(root, args, fail_fast=fail_fast, layout=layout),
             f"--junit-xml={report}"],
            cwd=root, env=tree_env(root, layout=layout), timeout=timeout)
        seconds = time.monotonic() - start
        passed = _count_passed(report)
        failed = _failed_files(report)
    lines = [line for line in output.splitlines() if line.strip()]
    return PytestRun(code, passed, tuple(lines[-TAIL_LINES:]), failed,
                     seconds, timed_out)


def run_pytest(root: pathlib.Path, args: Sequence[str], *,
               timeout: float | None = None,
               layout: Layout = DEFAULT_LAYOUT) -> PytestRun:
    """Run pytest with ``args`` in ``root``, stopping at the first failure."""
    return _run_pytest(root, args, fail_fast=True, timeout=timeout,
                       layout=layout)


def run_pytest_matrix(root: pathlib.Path, args: Sequence[str], *,
                      timeout: float | None = None,
                      layout: Layout = DEFAULT_LAYOUT) -> PytestRun:
    """Run pytest with ``args`` in ``root`` to the end of the selection."""
    return _run_pytest(root, args, fail_fast=False, timeout=timeout,
                       layout=layout)


def default_runner(layout: Layout = DEFAULT_LAYOUT, *,
                   matrix: bool = False) -> Callable[..., PytestRun]:
    """:func:`run_pytest`, or :func:`run_pytest_matrix`, on ``layout``."""
    return functools.partial(run_pytest_matrix if matrix else run_pytest,
                             layout=layout)


# ---- A copy of the tree ----

def _is_within(path: pathlib.Path, parent: pathlib.Path) -> bool:
    return path == parent or parent in path.parents


def _discard_copy(dest: pathlib.Path, created: bool) -> bool:
    """Remove what a failed copy wrote to ``dest``; True on success."""
    try:
        if created:
            shutil.rmtree(dest)
            return True
        for child in dest.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    except OSError:
        return False
    return True


def copy_tree(source: pathlib.Path, dest: pathlib.Path) -> pathlib.Path:
    """Copy the project at ``source`` to ``dest`` and return ``dest``.

    ``dest`` must be absent or an empty directory, and outside ``source``.
    The names in :data:`COPY_IGNORED` are left out and symbolic links are
    copied as links.  ``source`` is only read.  Raises :class:`HarnessError`
    if the copy fails, after removing what it wrote.
    """
    source_abs, dest_abs = source.resolve(), dest.resolve()
    if _is_within(dest_abs, source_abs):
        raise HarnessError(f"{dest} is inside the tree it would copy, {source}")
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise HarnessError(f"{dest} exists and is not an empty directory")
    created = not dest.exists()
    try:
        shutil.copytree(source, dest, symlinks=True, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(*COPY_IGNORED))
    except OSError as exc:                               # shutil.Error included
        cleaned = _discard_copy(dest, created)
        raise HarnessError(
            f"copying {source} to {dest} failed: {exc}; "
            + ("the partial copy was removed" if cleaned
               else f"remove the partial copy in {dest}")) from None
    return dest


def prepare_tree(layout: Layout, copy_to: pathlib.Path | None) -> pathlib.Path:
    """The tree to sweep: the project root, or a new copy of it at ``copy_to``.

    Raises :class:`HarnessError` if the copy cannot be made or if the tree
    does not import its own package.
    """
    root = layout.root
    if copy_to is not None:
        try:
            root = copy_tree(root, copy_to)
        except HarnessError as exc:
            raise HarnessError(f"--root: {exc}") from None
    problem = origin_problem(root, layout=layout)
    if problem is not None:
        raise HarnessError(problem)
    return root
