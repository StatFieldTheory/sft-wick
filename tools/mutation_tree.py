"""Processes, copies and the journal of a tree under mutation test.

Used by ``tools/mutation_check.py``, which loads this file by path and
re-exports its names.  Nothing here knows about a corpus entry: a tree is a
project root with the package under ``src/``.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import pathlib
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence

#: The package whose source is mutated.
PACKAGE = "sft_wick"
#: The directory, inside a project root, that is put on the import path.
PYTHONPATH_REL = pathlib.Path("src")
#: The source directory inside a project root.  Mutation paths are relative
#: to it.
SRC_REL = PYTHONPATH_REL / PACKAGE

#: Written in the tree before a mutant is and removed after the restore.  A
#: run killed by SIGKILL leaves it behind, and the next run reads it.
JOURNAL_NAME = ".mutation-inflight.json"

#: Names left out of a copy of the tree, at any depth: repository data,
#: agent state, caches and the journal.
COPY_IGNORED = (".git", ".claude", ".agent-state", "__pycache__",
                ".pytest_cache", ".hypothesis", ".ruff_cache", ".mypy_cache",
                "_build", "*.egg-info", JOURNAL_NAME)

#: Seconds to wait for the output of a process group after killing it.
KILL_GRACE = 10.0


class HarnessError(Exception):
    """The corpus or a tree is not in a state the harness can work on."""


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


def _read_journal(journal: pathlib.Path) -> tuple[str, bytes, str]:
    try:
        record = json.loads(journal.read_text(encoding="utf-8"))
        return (record["path"], base64.b64decode(record["original"]),
                record["mutant_sha256"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HarnessError(f"{journal} cannot be read ({exc}); check "
                           f"{SRC_REL} with `git diff`, then remove it") from None


def recover(root: pathlib.Path) -> str | None:
    """Undo the mutant a killed run left in ``root``; a message, or None.

    The journal of the killed run names the file, its original bytes and the
    digest of the mutant.  The file is restored only if it still holds that
    mutant.  Raises :class:`HarnessError` if it holds anything else.
    """
    journal = root / JOURNAL_NAME
    if not journal.exists():
        return None
    relpath, original, mutant_sha256 = _read_journal(journal)
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

def tree_env(root: pathlib.Path) -> dict[str, str]:
    """The environment of every process started on the tree at ``root``.

    ``root/src``, made absolute, is first in ``PYTHONPATH``, ahead of the
    entries that ``site`` adds for installed packages, so a child process of
    a test imports the package of this tree.  ``PYTHONDONTWRITEBYTECODE`` is
    set so that no run leaves a ``.pyc`` behind.  A ``.pyc`` is validated by
    the source's size and its modification time in whole seconds, so one
    written less than a second before a same-length mutation would be loaded
    in place of the mutated source.
    """
    inherited = os.environ.get("PYTHONPATH")
    entries = [str(root.resolve() / PYTHONPATH_REL)] + ([inherited] if inherited else [])
    return {**os.environ, "PYTHONPATH": os.pathsep.join(entries),
            "PYTHONDONTWRITEBYTECODE": "1"}


def run_in_tree(root: pathlib.Path, command: Sequence[str],
                ) -> subprocess.CompletedProcess[str]:
    """Run ``command`` in ``root`` with the environment of :func:`tree_env`.

    Standard output and standard error are captured as text.
    """
    return subprocess.run(
        list(command), cwd=root, capture_output=True, encoding="utf-8",
        errors="replace", env=tree_env(root))


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


def import_origin(root: pathlib.Path) -> pathlib.Path:
    """The directory a process started on ``root`` imports the package from."""
    code = (f"import {PACKAGE}, os, sys; "
            f"sys.stdout.write(os.path.dirname({PACKAGE}.__file__))")
    proc = run_in_tree(root, [sys.executable, "-c", code])
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        raise HarnessError(f"`import {PACKAGE}` failed in {root}: "
                           f"{lines[-1] if lines else 'no output'}")
    return pathlib.Path(proc.stdout).resolve()


def origin_problem(root: pathlib.Path) -> str | None:
    """Why runs on ``root`` would not test its source, or None."""
    expected = (root / SRC_REL).resolve()
    try:
        found = import_origin(root)
    except HarnessError as exc:
        return str(exc)
    if found != expected:
        return (f"a process started in {root} imports {PACKAGE} from {found}, "
                f"not from {expected}; a mutation there would not be tested")
    return None


def pytest_command(root: pathlib.Path, args: Sequence[str], *,
                   fail_fast: bool) -> list[str]:
    """The pytest command line for ``args`` on the tree at ``root``.

    Without the junit report option, which names a temporary file.  The
    path is absolute: pytest reads a relative ``pythonpath`` against its
    root directory, which is ``root`` itself.
    """
    flags = [*PYTEST_FLAGS, "-o", f"pythonpath={root.resolve() / PYTHONPATH_REL}"]
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
                timeout: float | None) -> PytestRun:
    with tempfile.TemporaryDirectory() as tmp:
        report = pathlib.Path(tmp) / "junit.xml"
        start = time.monotonic()
        code, output, timed_out = run_group(
            [*pytest_command(root, args, fail_fast=fail_fast),
             f"--junit-xml={report}"],
            cwd=root, env=tree_env(root), timeout=timeout)
        seconds = time.monotonic() - start
        passed = _count_passed(report)
        failed = _failed_files(report)
    lines = [line for line in output.splitlines() if line.strip()]
    return PytestRun(code, passed, tuple(lines[-TAIL_LINES:]), failed,
                     seconds, timed_out)


def run_pytest(root: pathlib.Path, args: Sequence[str], *,
               timeout: float | None = None) -> PytestRun:
    """Run pytest with ``args`` in ``root``, stopping at the first failure."""
    return _run_pytest(root, args, fail_fast=True, timeout=timeout)


def run_pytest_matrix(root: pathlib.Path, args: Sequence[str], *,
                      timeout: float | None = None) -> PytestRun:
    """Run pytest with ``args`` in ``root`` to the end of the selection."""
    return _run_pytest(root, args, fail_fast=False, timeout=timeout)


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

