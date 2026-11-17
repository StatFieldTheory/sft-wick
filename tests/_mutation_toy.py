"""A toy project for the tests of ``tools/mutation_check.py``.

Used by ``test_mutation_harness.py``, ``test_mutation_harness_sweep.py`` and
``test_mutation_harness_trees.py``.
The project has the layout of this repository: a package ``sft_wick`` under
``src/`` holding one module, ``harness_toy.py``, a ``pytest.ini`` whose
``addopts`` deselects the ``slow`` and ``heavy`` markers, and a few test
files.  :data:`CASES` lists the mutations of that module the tests apply.

The toy package has the name of the real one so that the harness, which
looks for ``src/sft_wick``, runs on it unchanged.  It is imported only by the
``pytest`` subprocesses the harness starts in the toy project.  The test
process itself imports neither the toy package nor the real one.

``tools/`` is not a package, so the harness is loaded by path.  pytest does
not collect this file, because its name does not start with ``test_``.
"""

from __future__ import annotations

import atexit
import contextlib
import dataclasses
import hashlib
import importlib.util
import pathlib
import signal
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "mutation_check.py"
PACKAGE_REL = pathlib.Path("src") / "sft_wick"

TOY_SOURCE = (
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "\n"
    "def scale(a):\n"
    "    return 2 * a\n"
)

TOY_TESTS = '''\
import pytest
from sft_wick import harness_toy


def test_add_small():
    assert harness_toy.add(2, 3) == 5


def test_scale_double():
    assert harness_toy.scale(4) == 8


@pytest.mark.slow
def test_add_large():
    assert harness_toy.add(20, 30) == 50


@pytest.mark.heavy
def test_add_heavy():
    assert harness_toy.add(200, 300) == 500


@pytest.mark.skip(reason="selected but never run")
def test_skipped():
    assert harness_toy.add(1, 1) == 3
'''

#: A second test file that sees the same function, for the kill matrix.
TOY_MORE_TESTS = '''\
from sft_wick import harness_toy


def test_add_negative():
    assert harness_toy.add(-2, -3) == -5


def test_add_zero():
    assert harness_toy.add(0, 0) == 0
'''

#: Fails on the unmutated source.
TOY_RED_TESTS = '''\
from sft_wick import harness_toy


def test_red_on_clean_source():
    assert harness_toy.add(1, 1) == 3
'''

#: Passes at once on clean source.  With ``add`` mutated it announces itself
#: and then waits, so the outer test can signal the harness mid-run.
TOY_HANG_TESTS = '''\
import pathlib
import time

from sft_wick import harness_toy

HERE = pathlib.Path(__file__).parent


def test_waits_while_mutated():
    if harness_toy.add(2, 3) == 5:
        return
    (HERE / "mutated.flag").write_text("x")
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline and not (HERE / "stop.flag").exists():
        time.sleep(0.02)
    raise AssertionError("mutated")
'''

#: Passes at once on clean source.  With ``add`` mutated it starts a child
#: process that sleeps, writes the child's pid to ``grandchild.pid`` and loops
#: for ever, so only a time limit ends the run.
TOY_LOOP_TESTS = '''\
import pathlib
import subprocess
import sys

from sft_wick import harness_toy

HERE = pathlib.Path(__file__).parent


def test_loop_until_add_is_right():
    if harness_toy.add(2, 3) == 5:
        return
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    (HERE / "grandchild.pid").write_text(str(child.pid))
    n = 0
    while harness_toy.add(n, 1) != n + 1:
        n += 1
'''

#: Reads the package in the test process and in a child process, and appends
#: the file each one imported to ``origin.log`` beside the test file.  The
#: child gets the environment of the test process unchanged, the way the
#: subprocess tests of this repository start their children.
TOY_ORIGIN_TESTS = '''\
import pathlib
import subprocess
import sys

from sft_wick import harness_toy

HERE = pathlib.Path(__file__).parent
CHILD = "from sft_wick import harness_toy as m; print(m.__file__, m.add(2, 3))"


def _record(kind, path):
    with open(HERE / "origin.log", "a", encoding="utf-8") as log:
        log.write(f"{kind} {path}\\n")


def test_in_process_value():
    _record("in-process", harness_toy.__file__)
    assert harness_toy.add(2, 3) == 5


def test_child_process_value():
    out = subprocess.run([sys.executable, "-c", CHILD], capture_output=True,
                         text=True, check=True).stdout.split()
    _record("child", out[0])
    assert out[1] == "5"
'''

TOY_INI = (
    "[pytest]\n"
    "addopts = -m 'not slow and not heavy' --strict-markers\n"
    "pythonpath = src\n"
    "markers =\n"
    "    slow: deselected by addopts\n"
    "    heavy: deselected by addopts\n"
)

ADD, SUB = "return a + b", "return a - b"
SCALE2, SCALE3 = "return 2 * a", "return 3 * a"
#: Wrong only for 10 <= a < 100, which only the ``slow`` test reaches.
ADD_LARGE_ONLY = "return a + b if not 10 <= a < 100 else 0"
NOOP = "return a + b  # same behaviour"
ONE_TEST = "tests/test_toy.py -k add_small"
TWO_FILES = "tests/test_toy.py tests/test_more.py"


@dataclasses.dataclass(frozen=True)
class Case:
    old: str
    new: str
    tests: str
    path: str = "harness_toy.py"


CASES = {
    # ``old`` and ``new`` have equal length: a .pyc written by the baseline
    # run in the same second would still validate against the mutated file.
    "killed": Case(ADD, SUB, ONE_TEST),
    "noop": Case(ADD, NOOP, ONE_TEST),
    "syntax_error": Case(ADD, "return a +", ONE_TEST),
    # Only ``scale_double`` sees this, so the whole ``or`` has to arrive.
    "quoted_k_kills": Case(
        SCALE2, SCALE3, "tests/test_toy.py -k 'add_small or scale_double'"),
    "quoted_k_misses": Case(
        SCALE2, SCALE3, "tests/test_toy.py -k 'add_small or add_large'"),
    # No ``-m``: the project's addopts leaves the slow test out.
    "default_m_misses": Case(ADD, ADD_LARGE_ONLY, "tests/test_toy.py"),
    "quoted_m_misses": Case(
        ADD, ADD_LARGE_ONLY, "tests/test_toy.py -m 'not slow and not heavy'"),
    # A ``-m`` in the selector replaces the one in addopts.
    "plain_m_kills": Case(ADD, ADD_LARGE_ONLY, "tests/test_toy.py -m slow"),
    # The only match is slow, and the selector has no ``-m`` of its own.
    "slow_without_m": Case(ADD, ADD_LARGE_ONLY, "tests/test_toy.py -k add_large"),
    "heavy_without_m": Case(ADD, SUB, "tests/test_toy.py -k add_heavy"),
    "selects_nothing": Case(ADD, SUB, "tests/test_toy.py -k no_such_test"),
    "unquoted_or": Case(SCALE2, SCALE3, "tests/test_toy.py -k add_small or scale"),
    "missing_test_file": Case(ADD, SUB, "tests/test_absent.py"),
    "red_baseline": Case(ADD, NOOP, "tests/test_red.py"),
    "only_skipped": Case(ADD, SUB, "tests/test_toy.py -k skipped"),
    "unbalanced_quote": Case(ADD, SUB, "tests/test_toy.py -k 'add_small"),
    "stale": Case("return a * b", SUB, ONE_TEST),
    "ambiguous": Case("    return ", "    return 1 + ", ONE_TEST),
    "missing_source": Case(ADD, SUB, ONE_TEST, path="absent.py"),
}

#: Scenarios that must be refused before any source is written.
REFUSED = {
    **dict.fromkeys(("selects_nothing", "unquoted_or", "missing_test_file",
                     "red_baseline", "only_skipped", "unbalanced_quote",
                     "slow_without_m", "heavy_without_m"),
                    "SELECTOR BROKEN"),
    "stale": "STALE", "ambiguous": "AMBIGUOUS", "missing_source": "STALE",
}


@contextlib.contextmanager
def loaded_harness():
    """Load ``tools/mutation_check.py`` by path and unload it on exit."""
    name = "mutation_check_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def make_project(root: pathlib.Path,
                 source: bytes = TOY_SOURCE.encode()) -> pathlib.Path:
    """Write the toy project under ``root``; return the path of its module."""
    package = root / PACKAGE_REL
    package.mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pytest.ini").write_text(TOY_INI)
    (package / "__init__.py").write_text("")
    (root / "tests" / "test_toy.py").write_text(TOY_TESTS)
    (root / "tests" / "test_more.py").write_text(TOY_MORE_TESTS)
    (root / "tests" / "test_red.py").write_text(TOY_RED_TESTS)
    module = package / "harness_toy.py"
    module.write_bytes(source)
    return module


def tree_digest(root: pathlib.Path) -> dict[str, str]:
    """``{relative path: sha256}`` of every file under ``root``.

    Directories are listed with an empty digest, so a directory that appears
    or disappears changes the result.
    """
    digest = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        digest[rel] = (hashlib.sha256(path.read_bytes()).hexdigest()
                       if path.is_file() else "")
    return digest


def mutation(harness, case: Case, label: str = "toy"):
    return harness.Mutation(label, case.path, case.old, case.new, case.tests,
                            "why it matters")


def fake_runner(harness, *codes, passed=1, log=None, failed_files=(),
                timeouts=None):
    """A runner returning ``codes`` in order, without starting pytest.

    ``log`` collects the arguments of each call and ``timeouts`` its limit.
    """
    remaining = list(codes)

    def runner(root, args, timeout=None):
        code = remaining.pop(0)
        if log is not None:
            log.append(tuple(args))
        if timeouts is not None:
            timeouts.append(timeout)
        return harness.PytestRun(code, passed, (f"fake tail rc={code}",),
                                 tuple(failed_files) if code == 1 else ())

    return runner


def stand_in(module, **replaced):
    """A namespace with the names of ``module``, some of them replaced."""
    return types.SimpleNamespace(**{**vars(module), **replaced})


def record_restore_handlers(harness, monkeypatch) -> dict[str, list]:
    """Record the handlers the harness registers instead of installing them.

    Only the harness's references to ``atexit`` and ``signal`` are replaced.
    The standard-library modules are left as they are: ``tempfile`` registers
    an exit hook of its own the first time a process creates a
    ``TemporaryDirectory``, which can happen inside a test that uses this.
    """
    record = {"signals": [], "atexit": []}
    monkeypatch.setattr(harness, "signal", stand_in(
        signal, signal=lambda sig, handler: record["signals"].append(sig)))
    monkeypatch.setattr(harness, "atexit", stand_in(
        atexit, register=record["atexit"].append))
    return record
