"""The mutation harness on a project laid out unlike this repository.

``tools/mutation_check.py`` takes the layout of the project it tests from
``--project-root``, ``--package``, ``--source-root`` and ``--corpus``.  The
tests here build the toy project of ``tests/_mutation_toy.py`` with the
package ``toypkg`` under ``lib/`` and keep the corpus in a directory outside
the toy tree.  Most of them start the harness as a script in a subprocess,
with the layout given only on its command line.

A ``sitecustomize`` module in the toy's ``lib/`` logs the command line and
the environment of every Python process started with ``lib/`` on its path:
the import probe, each pytest run and the children of the toy tests.  The
tests check from that log, from the journal and from the output that the
name of this repository's package occurs nowhere in those runs.  The last
test checks that the two tool files name it only in their default layout.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

import pytest

from ._mutation_toy import (
    ADD,
    NOOP,
    ONE_TEST,
    ROOT,
    SCALE2,
    SCALE3,
    SCRIPT,
    SUB,
    TOY_HANG_TESTS,
    TOY_ORIGIN_TESTS,
    TWO_FILES,
    fake_runner,
    for_package,
    loaded_harness,
    make_project,
    record_restore_handlers,
    tree_digest,
)

PACKAGE = "toypkg"
SOURCE_ROOT = "lib"
PACKAGE_REL = pathlib.Path(SOURCE_ROOT) / PACKAGE
#: The import name of this repository's package, which no run may use.
FORBIDDEN = "sft_" + "wick"
TOOL_FILES = (ROOT / "tools" / "mutation_check.py",
              ROOT / "tools" / "mutation_tree.py")

#: Appends ``{"argv": ..., "env": ...}`` of the starting process to
#: ``process.log`` at the root of the project that holds it.
PROCESS_LOG_HOOK = '''\
import json
import os
import pathlib
import sys

with open(pathlib.Path(__file__).resolve().parents[1] / "process.log", "a",
          encoding="utf-8") as _log:
    _log.write(json.dumps({"argv": sys.orig_argv, "env": dict(os.environ)})
               + "\\n")
'''


def entry(label: str, old: str, new: str, tests: str) -> tuple:
    return (label, "harness_toy.py", old, new, tests, "why it matters")


KILLED_ENTRY = entry("killed", ADD, SUB, ONE_TEST)
SURVIVOR_ENTRY = entry("survivor", ADD, NOOP, ONE_TEST)
MATRIX_ENTRY = entry("sign of add", ADD, SUB, TWO_FILES)
SCALE_ENTRY = entry("factor of scale", SCALE2, SCALE3, TWO_FILES)
HANG_ENTRY = entry("hang", ADD, SUB, "tests/test_hang.py")
CHILD_ENTRY = entry("child sees it", ADD, SUB, "tests/test_origin.py -k child")


def make_toy(root: pathlib.Path, *, package: str = PACKAGE,
             source_root: str = SOURCE_ROOT) -> pathlib.Path:
    """The toy project, by default ``toypkg`` under ``lib/``; its module."""
    root.mkdir()
    module = make_project(root, package=package, source_root=source_root)
    (root / source_root / "sitecustomize.py").write_text(PROCESS_LOG_HOOK)
    (root / "tests" / "test_hang.py").write_text(
        for_package(TOY_HANG_TESTS, package))
    (root / "tests" / "test_origin.py").write_text(
        for_package(TOY_ORIGIN_TESTS, package))
    return module


def make_corpus(directory: pathlib.Path, *entries: tuple) -> pathlib.Path:
    directory.mkdir()
    (directory / "area.py").write_text(f"ENTRIES = {list(entries)!r}\n")
    return directory


def layout_options(root: pathlib.Path, corpus: pathlib.Path, *,
                   package: str = PACKAGE,
                   source_root: str = SOURCE_ROOT) -> list[str]:
    return ["--project-root", str(root), "--package", package,
            "--source-root", source_root, "--corpus", str(corpus)]


def toy_layout(harness, root: pathlib.Path, corpus: pathlib.Path):
    return harness.Layout(root=root, package=PACKAGE,
                          source_root=pathlib.Path(SOURCE_ROOT), corpus=corpus)


def clean_env(**extra: str) -> dict[str, str]:
    """This environment without pytest's variables, ``PYTHONPATH``, the name
    of this repository's package or its path.

    Whatever the runs then contain that names either comes from the harness.
    """
    env = {key: value for key, value in os.environ.items()
           if FORBIDDEN not in key + value and str(ROOT) not in value
           and not key.startswith("PYTEST") and key != "PYTHONPATH"}
    return {**env, "PYTHONDONTWRITEBYTECODE": "1", **extra}


def harness_command(*args: str) -> list[str]:
    return [sys.executable, str(SCRIPT), *args]


def run_harness(*args: str, env: dict[str, str] | None = None,
                cwd: pathlib.Path | None = None,
                ) -> subprocess.CompletedProcess[str]:
    command = harness_command(*args)
    assert FORBIDDEN not in " ".join(command)
    return subprocess.run(command, capture_output=True, text=True, cwd=cwd,
                          env=clean_env() if env is None else env, timeout=300)


def logged_processes(root: pathlib.Path) -> list[dict]:
    log = root / "process.log"
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return [json.loads(line) for line in lines]


def assert_layout_only(root: pathlib.Path, *texts: str,
                       source_root: str = SOURCE_ROOT) -> list[dict]:
    """No logged process, and none of ``texts``, names the forbidden package
    or the path of this repository.

    Every logged process has the source root of ``root`` first on its path.
    Returns the logged processes.
    """
    processes = logged_processes(root)
    assert processes
    for process in processes:
        assert FORBIDDEN not in json.dumps(process)
        assert str(ROOT) not in json.dumps(process)
        first = process["env"]["PYTHONPATH"].split(os.pathsep)[0]
        assert pathlib.Path(first) == (root / source_root).resolve()
    for text in texts:
        assert FORBIDDEN not in text and str(ROOT) not in text
    return processes


def is_pytest(process: dict) -> bool:
    return process["argv"][1:3] == ["-m", "pytest"]


@pytest.fixture
def toy(tmp_path):
    root = tmp_path / "project"
    return root, make_toy(root)


@pytest.fixture(scope="module")
def harness():
    with loaded_harness() as module:
        yield module


# ---- Sweeps driven by the layout options ----

def test_baseline_only_on_another_layout(toy, tmp_path):
    root, module = toy
    corpus = make_corpus(tmp_path / "corpus", KILLED_ENTRY)
    before = tree_digest(root)
    proc = run_harness(*layout_options(root, corpus), "--baseline-only")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[BASELINE OK] killed" in proc.stdout
    processes = assert_layout_only(root, proc.stdout, proc.stderr)
    # The import probe, then one baseline run of pytest.
    assert [is_pytest(p) for p in processes] == [False, True]
    assert f"import_module('{PACKAGE}')" in processes[0]["argv"][-1]
    assert {path: digest for path, digest in tree_digest(root).items()
            if path != "process.log"} == before


def test_killed_and_surviving_mutants_on_another_layout(toy, tmp_path):
    root, module = toy
    original = module.read_bytes()
    corpus = make_corpus(tmp_path / "corpus", KILLED_ENTRY, SURVIVOR_ENTRY)
    report = tmp_path / "report.json"
    proc = run_harness(*layout_options(root, corpus), "--json", str(report))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "[KILLED] killed" in proc.stdout
    assert "[SURVIVED] survivor" in proc.stdout
    assert "killed 1 | survived 1" in proc.stdout
    assert module.read_bytes() == original
    record = report.read_text()
    assert [(m["label"], m["verdict"]) for m in json.loads(record)["mutants"]] == [
        ("killed", "KILLED"), ("survivor", "SURVIVED")]
    processes = assert_layout_only(root, proc.stdout, proc.stderr, record)
    for process in filter(is_pytest, processes):
        option = process["argv"].index(
            f"pythonpath={(root / SOURCE_ROOT).resolve()}")
        assert process["argv"][option - 1] == "-o"


def test_matrix_on_another_layout(toy, tmp_path):
    root, _ = toy
    corpus = make_corpus(tmp_path / "corpus", MATRIX_ENTRY, SCALE_ENTRY)
    proc = run_harness(*layout_options(root, corpus), "--matrix")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "    failed in: tests/test_more.py, tests/test_toy.py" in proc.stdout
    assert "    failed in: tests/test_toy.py" in proc.stdout
    processes = assert_layout_only(root, proc.stdout, proc.stderr)
    assert all("-x" not in p["argv"] for p in filter(is_pytest, processes))


def test_copy_mode_on_another_layout(toy, tmp_path):
    root, module = toy
    corpus = make_corpus(tmp_path / "corpus", CHILD_ENTRY)
    before = tree_digest(root)
    dest = tmp_path / "copy"
    proc = run_harness(*layout_options(root, corpus), "--root", str(dest))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[KILLED] child sees it" in proc.stdout
    assert tree_digest(root) == before
    processes = assert_layout_only(dest, proc.stdout, proc.stderr)
    # The test's own child process ran on the copy as well.
    children = [p for p in processes if p["argv"][1] == "-c"
                and "harness_toy as m" in p["argv"][2]]
    assert children
    copied = (dest / PACKAGE_REL / "harness_toy.py").resolve()
    assert {pathlib.Path(line.split(" ", 1)[1]).resolve() for line in
            (dest / "tests" / "origin.log").read_text().splitlines()} == {copied}
    assert copied.read_bytes() == module.read_bytes()


def test_journal_recovery_after_sigkill_on_another_layout(toy, tmp_path):
    root, module = toy
    original = module.read_bytes()
    corpus = make_corpus(tmp_path / "corpus", HANG_ENTRY)
    flag = root / "tests" / "mutated.flag"
    command = harness_command(*layout_options(root, corpus))
    proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=clean_env())
    try:
        deadline = time.monotonic() + 120.0
        while not flag.exists() and proc.poll() is None:
            assert time.monotonic() < deadline, "the mutated run never started"
            time.sleep(0.02)
        assert proc.poll() is None, proc.communicate()[0]
        proc.send_signal(signal.SIGKILL)
        killed_output, _ = proc.communicate(timeout=60.0)
    finally:
        (root / "tests" / "stop.flag").write_text("x")
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    journal = root / ".mutation-inflight.json"
    journal_text = journal.read_text(encoding="utf-8")
    assert json.loads(journal_text)["path"] == f"{SOURCE_ROOT}/{PACKAGE}/harness_toy.py"
    assert module.read_bytes() == original.replace(ADD.encode(), SUB.encode())
    after = run_harness(*layout_options(root, corpus), "-k", "no such label")
    assert after.returncode == 2, after.stdout + after.stderr
    assert "after a killed run" in after.stdout
    assert module.read_bytes() == original
    assert not journal.exists()
    assert_layout_only(root, killed_output, journal_text, after.stdout,
                       after.stderr)


# ---- The import probe ----

#: Runs the harness as a script would, except that processes it starts get
#: this environment unchanged, so ``PYTHONPATH`` decides what they import.
DRIVER_WITHOUT_PATH_ENTRY = '''\
import importlib.util
import os
import sys

spec = importlib.util.spec_from_file_location("mutation_check", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.tree.tree_env = lambda *args, **kwargs: dict(os.environ)
raise SystemExit(module.main(sys.argv[2:]))
'''


def test_probe_finds_a_second_copy_earlier_on_pythonpath(toy, tmp_path):
    root, module = toy
    elsewhere = tmp_path / "elsewhere"
    make_toy(elsewhere)
    corpus = make_corpus(tmp_path / "corpus", KILLED_ENTRY)
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER_WITHOUT_PATH_ENTRY)
    before = tree_digest(root)
    command = [sys.executable, str(driver), str(SCRIPT),
               *layout_options(root, corpus)]
    proc = subprocess.run(
        command, capture_output=True, text=True, timeout=300,
        env=clean_env(PYTHONPATH=str(elsewhere / SOURCE_ROOT)))
    assert proc.returncode == 3, proc.stdout + proc.stderr
    found = (elsewhere / PACKAGE_REL).resolve()
    assert f"imports {PACKAGE} from {found}" in proc.stdout
    assert str((root / PACKAGE_REL).resolve()) in proc.stdout
    assert tree_digest(root) == before
    # The driver itself and the probe import from ``elsewhere``; no pytest
    # run starts.
    processes = logged_processes(elsewhere)
    assert [p["argv"][1] for p in processes] == [str(driver), "-c"]
    assert FORBIDDEN not in json.dumps(processes) + proc.stdout + proc.stderr
    assert str(ROOT) not in json.dumps(processes[1]) + proc.stdout


def test_probe_finds_a_second_copy_at_the_project_root(toy, tmp_path):
    # ``python -c`` and ``python -m pytest`` put the working directory, the
    # project root, ahead of PYTHONPATH.  A leftover ``toypkg/`` there wins.
    root, _ = toy
    leftover = root / PACKAGE
    leftover.mkdir()
    (leftover / "__init__.py").write_text("")
    corpus = make_corpus(tmp_path / "corpus", KILLED_ENTRY)
    proc = run_harness(*layout_options(root, corpus))
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert f"imports {PACKAGE} from {leftover.resolve()}" in proc.stdout
    processes = assert_layout_only(root, proc.stdout, proc.stderr)
    assert not any(map(is_pytest, processes))


# ---- Checking the options ----

@pytest.mark.parametrize("option, value, message", [
    ("--package", "toy-pkg", "is not a dotted import name"),
    ("--package", "1toy", "is not a dotted import name"),
    ("--package", "toypkg..sub", "is not a dotted import name"),
    ("--package", "class", "is not a dotted import name"),
    ("--package", "other", "is not a directory"),
    ("--source-root", "missing", "is not a directory"),
    ("--source-root", "../lib", "relative to the project root"),
    ("--source-root", "/lib", "relative to the project root"),
    ("--corpus", "missing", "is not a directory"),
    ("--corpus", "file", "is not a directory"),
    ("--project-root", "missing", "is not a directory"),
    ("--extra-path", "missing", "is not a directory"),
    ("--extra-path", "../lib", "relative to the project root"),
])
def test_bad_layout_options_exit_three_before_any_run(
        harness, toy, tmp_path, monkeypatch, capsys, option, value, message):
    root, _ = toy
    record_restore_handlers(harness, monkeypatch)
    corpus = make_corpus(tmp_path / "corpus", KILLED_ENTRY)
    (tmp_path / "file").write_text("x")
    if option in ("--corpus", "--project-root") and value != "/lib":
        value = str(tmp_path / value)
    options = layout_options(root, corpus)
    if option in options:
        options[options.index(option) + 1] = value
    else:
        options += [option, value]
    log = []
    code = harness.main(options, runner=fake_runner(harness, 0, 1, log=log))
    printed = capsys.readouterr().out
    assert code == harness.EXIT_SETUP == 3
    assert f"cannot start: {option}" in printed and message in printed
    assert log == []
    assert not (root / "process.log").exists()


def test_corpus_entries_are_checked_against_the_given_layout(harness, tmp_path):
    layout = harness.Layout(root=tmp_path, package="toypkg.sub",
                            source_root=pathlib.Path(SOURCE_ROOT),
                            corpus=tmp_path)
    assert layout.package_rel == pathlib.Path("lib/toypkg/sub")
    bad = ("label", "../outside.py") + KILLED_ENTRY[2:]
    (tmp_path / "area.py").write_text(f"ENTRIES = [{bad!r}]\n")
    with pytest.raises(harness.HarnessError, match="not relative to lib/toypkg/sub"):
        harness.load_corpus(layout=layout)


def test_default_layout_is_this_repository(harness):
    layout = harness.DEFAULT_LAYOUT
    assert layout.root == ROOT
    assert layout.corpus == ROOT / "tools" / "mutations" == harness.MUTATION_DIR
    assert harness.SRC == ROOT / layout.package_rel == ROOT / harness.SRC_REL
    assert layout.package == harness.PACKAGE == FORBIDDEN


# ---- Paths of other shapes ----

def test_a_source_root_with_a_space_is_one_path_entry(tmp_path):
    # pytest splits an ini path list with shlex.  Unquoted, ``<root>/my lib``
    # would put ``<root>/my`` and ``<root>/lib`` on the path, and pytest
    # would import the unmutated copy in ``lib/``.
    root = tmp_path / "project"
    module = make_toy(root, source_root="my lib")
    decoy = root / "lib" / PACKAGE
    decoy.mkdir(parents=True)
    (decoy / "__init__.py").write_text("")
    (decoy / "harness_toy.py").write_bytes(module.read_bytes())
    corpus = make_corpus(tmp_path / "corpus", KILLED_ENTRY)
    proc = run_harness(*layout_options(root, corpus, source_root="my lib"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[KILLED] killed" in proc.stdout
    assert_layout_only(root, proc.stdout, proc.stderr, source_root="my lib")


def test_relative_project_root_and_corpus_with_a_dotted_package(tmp_path):
    root = tmp_path / "project"
    module = make_toy(root, package="toypkg.sub")
    original = module.read_bytes()
    make_corpus(tmp_path / "corpus", KILLED_ENTRY)
    options = layout_options(pathlib.Path("project"), pathlib.Path("corpus"),
                             package="toypkg.sub")
    proc = run_harness(*options, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[KILLED] killed" in proc.stdout
    assert module.read_bytes() == original
    assert_layout_only(root, proc.stdout, proc.stderr)


def test_extra_paths_follow_the_source_root(toy, tmp_path):
    root, _ = toy
    (root / "helpers").mkdir()
    (root / "helpers" / "thelper.py").write_text("TWO = 2\n")
    ini = root / "pytest.ini"
    ini.write_text(ini.read_text().replace(
        f"pythonpath = {SOURCE_ROOT}", f"pythonpath = {SOURCE_ROOT} helpers"))
    (root / "tests" / "test_helper.py").write_text(
        f"import thelper\nfrom {PACKAGE} import harness_toy\n\n\n"
        "def test_add_two():\n    assert harness_toy.add(thelper.TWO, 3) == 5\n")
    corpus = make_corpus(tmp_path / "corpus",
                         entry("helper", ADD, SUB, "tests/test_helper.py"))
    proc = run_harness(*layout_options(root, corpus), "--extra-path", "helpers")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[KILLED] helper" in proc.stdout
    processes = assert_layout_only(root, proc.stdout, proc.stderr)
    entries = [str((root / name).resolve()) for name in (SOURCE_ROOT, "helpers")]
    for process in processes:
        assert process["env"]["PYTHONPATH"].split(os.pathsep)[:2] == entries
    for process in filter(is_pytest, processes):
        assert f"pythonpath={' '.join(entries)}" in process["argv"]


def test_probe_accepts_a_namespace_package(harness, toy, tmp_path):
    root, _ = toy
    (root / PACKAGE_REL / "__init__.py").unlink()
    assert harness.origin_problem(
        root, layout=toy_layout(harness, root, tmp_path)) is None


def test_a_package_that_resolves_outside_the_project_is_refused(
        harness, toy, tmp_path, monkeypatch, capsys):
    root, _ = toy
    record_restore_handlers(harness, monkeypatch)
    (root / SOURCE_ROOT).rename(tmp_path / "shared")
    (root / SOURCE_ROOT).symlink_to(pathlib.Path("..") / "shared")
    corpus = make_corpus(tmp_path / "corpus", KILLED_ENTRY)
    log = []
    code = harness.main(layout_options(root, corpus),
                        runner=fake_runner(harness, 0, 1, log=log))
    printed = capsys.readouterr().out
    assert code == 3, printed
    assert "cannot start: --package" in printed
    assert "outside the project root" in printed
    assert log == []


def test_mutated_copy_refuses_a_package_linked_back_into_the_source(
        harness, toy, tmp_path):
    # The link is absolute, so the copy's ``lib`` leads into the project.
    root, _ = toy
    (root / SOURCE_ROOT).rename(root / "real")
    (root / SOURCE_ROOT).symlink_to(root / "real")
    before = tree_digest(root)
    with pytest.raises(harness.HarnessError, match="outside the tree"):
        harness.mutated_copy(harness.Mutation(*KILLED_ENTRY), tmp_path / "copy",
                             layout=toy_layout(harness, root, tmp_path))
    assert tree_digest(root) == before


def test_mutated_copy_copies_the_project_of_the_layout(harness, toy, tmp_path):
    root, module = toy
    dest = harness.mutated_copy(harness.Mutation(*KILLED_ENTRY),
                                tmp_path / "copy",
                                layout=toy_layout(harness, root, tmp_path))
    text = module.read_text()
    assert (dest / PACKAGE_REL / "harness_toy.py").read_text() == text.replace(
        ADD, SUB)
    assert ADD in text


def test_baselines_run_on_the_given_layout(harness, toy, tmp_path):
    root, _ = toy
    baselines = harness.Baselines(root,
                                  layout=toy_layout(harness, root, tmp_path))
    assert baselines.check(ONE_TEST).ok


# ---- Messages that name the layout ----

def test_layout_options_without_a_corpus_exit_three(
        harness, toy, tmp_path, monkeypatch, capsys):
    root, _ = toy
    record_restore_handlers(harness, monkeypatch)
    options = layout_options(root, tmp_path)[:-2]
    log = []
    code = harness.main(options, runner=fake_runner(harness, 0, 1, log=log))
    printed = capsys.readouterr().out
    assert code == 3, printed
    assert "cannot start: --corpus" in printed
    assert log == []


def test_an_empty_corpus_is_named_in_the_message(
        harness, toy, tmp_path, monkeypatch, capsys):
    root, _ = toy
    record_restore_handlers(harness, monkeypatch)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    code = harness.main(layout_options(root, corpus),
                        runner=fake_runner(harness))
    printed = capsys.readouterr().out
    assert code == harness.EXIT_NOTHING_SELECTED, printed
    assert f"the corpus in {corpus} has no entry" in printed


def test_a_selector_that_runs_no_test_gets_no_marker_of_this_repository(
        harness, toy, tmp_path, monkeypatch, capsys):
    root, _ = toy
    record_restore_handlers(harness, monkeypatch)
    corpus = make_corpus(tmp_path / "corpus", KILLED_ENTRY)
    code = harness.main(layout_options(root, corpus),
                        runner=fake_runner(harness, 5))
    printed = capsys.readouterr().out
    assert code == 1, printed
    assert "[SELECTOR BROKEN] killed" in printed
    assert "deselects its tests by marker" in printed
    assert "slow" not in printed and "heavy" not in printed


def test_an_unreadable_journal_names_the_package_of_the_layout(
        harness, toy, tmp_path, monkeypatch, capsys):
    root, _ = toy
    record_restore_handlers(harness, monkeypatch)
    (root / harness.JOURNAL_NAME).write_text("not json")
    corpus = make_corpus(tmp_path / "corpus", KILLED_ENTRY)
    code = harness.main(layout_options(root, corpus),
                        runner=fake_runner(harness))
    printed = capsys.readouterr().out
    assert code == 3, printed
    assert f"check {PACKAGE_REL} with `git diff`" in printed
    assert FORBIDDEN not in printed


# ---- The tool files name the package only in their default layout ----

def _default_layout_lines(path: pathlib.Path) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name)
                        and target.id == "DEFAULT_LAYOUT"
                        for target in node.targets)):
            lines.update(range(node.lineno, node.end_lineno + 1))
    return lines


def test_tool_files_name_the_layout_only_in_the_default():
    pattern = re.compile(rf"{FORBIDDEN}|\bsrc\b")
    found = {}
    for path in TOOL_FILES:
        allowed = _default_layout_lines(path)
        found[path.name] = [
            number for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1)
            if pattern.search(line) and number not in allowed]
    assert found == {path.name: [] for path in TOOL_FILES}
    assert any(_default_layout_lines(path) for path in TOOL_FILES)
