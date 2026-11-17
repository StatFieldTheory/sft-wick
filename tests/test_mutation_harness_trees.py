"""The mutation harness: the kill matrix, copied trees, and which tree is run.

Tests for ``tools/mutation_check.py`` on the toy project of
``tests/_mutation_toy.py``, with real ``pytest`` subprocesses unless a test
says otherwise.

* ``--matrix`` runs every selected test and lists the test files that fail
  under each mutant.
* ``--root DIR`` copies the tree to ``DIR`` and runs the sweep there.  The
  tests compare a digest of the original tree before and after, and read
  from a log the toy tests write which file each process imported.
* A toy project elsewhere on ``PYTHONPATH`` stands for an editable install
  that points at another checkout.  The harness has to make the test process
  and its child processes import the tree it mutates.

The verdict rules are tested in ``tests/test_mutation_harness.py``, the sweep
and the command line in ``tests/test_mutation_harness_sweep.py``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import sys

import pytest

from ._mutation_toy import (
    ADD,
    CASES,
    ONE_TEST,
    PACKAGE_REL,
    SCALE2,
    SCALE3,
    SUB,
    TOY_ORIGIN_TESTS,
    TWO_FILES,
    Case,
    fake_runner,
    loaded_harness,
    make_project,
    mutation,
    record_restore_handlers,
    tree_digest,
)

CHILD_ONLY = "tests/test_origin.py -k child"
IN_PROCESS_ONLY = "tests/test_origin.py -k in_process"
#: Prints the file a child process imports and what ``add`` returns there.
PROBE = "from sft_wick import harness_toy as m; print(m.__file__, m.add(2, 3))"


@pytest.fixture(scope="module")
def harness():
    with loaded_harness() as module:
        yield module


@pytest.fixture
def recorded_handlers(harness, monkeypatch):
    return record_restore_handlers(harness, monkeypatch)


@pytest.fixture
def toy(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    return root, make_project(root)


@pytest.fixture
def elsewhere(tmp_path, monkeypatch):
    """A second toy project, first on ``PYTHONPATH`` of this process.

    It plays the editable install of another checkout: a Python process
    started without further measures imports ``sft_wick`` from here.
    """
    root = tmp_path / "elsewhere"
    root.mkdir()
    make_project(root)
    monkeypatch.setenv("PYTHONPATH", str(root / "src"))
    return root


def add_origin_tests(root: pathlib.Path) -> None:
    (root / "tests" / "test_origin.py").write_text(TOY_ORIGIN_TESTS)


def origins(root: pathlib.Path) -> dict[str, set[pathlib.Path]]:
    """``{kind: files imported}`` from the log of ``tests/test_origin.py``."""
    found: dict[str, set[pathlib.Path]] = {}
    for line in (root / "tests" / "origin.log").read_text().splitlines():
        kind, path = line.split(" ", 1)
        found.setdefault(kind, set()).add(pathlib.Path(path).resolve())
    return found


# ---- The kill matrix ----

def test_pytest_command_fails_fast_unless_asked_for_the_matrix(harness, tmp_path):
    fast = harness.pytest_command(tmp_path, ["tests/a.py"], fail_fast=True)
    full = harness.pytest_command(tmp_path, ["tests/a.py"], fail_fast=False)
    assert "-x" in fast and "-x" not in full
    assert [arg for arg in fast if arg != "-x"] == full
    for command in (fast, full):
        assert command[:4] == [sys.executable, "-m", "pytest", "tests/a.py"]
        option = command.index(f"pythonpath={tmp_path / 'src'}")
        assert command[option - 1] == "-o"


def test_matrix_lists_every_test_file_that_fails(harness, toy, recorded_handlers,
                                                 tmp_path, capsys):
    root, module = toy
    original = module.read_bytes()
    out = tmp_path / "matrix.json"
    muts = [mutation(harness, Case(ADD, SUB, TWO_FILES), "sign of add"),
            mutation(harness, Case(SCALE2, SCALE3, TWO_FILES), "factor of scale")]
    code = harness.main(["--matrix", "--json", str(out)], root=root, mutations=muts)
    assert code == 0
    assert module.read_bytes() == original
    printed = capsys.readouterr().out
    assert "    failed in: tests/test_more.py, tests/test_toy.py" in printed
    record = json.loads(out.read_text())
    assert record["matrix"] is True
    assert [(m["label"], m["verdict"], m["failed_files"])
            for m in record["mutants"]] == [
        ("sign of add", "KILLED", ["tests/test_more.py", "tests/test_toy.py"]),
        ("factor of scale", "KILLED", ["tests/test_toy.py"]),
    ]
    assert {m["returncode"] for m in record["mutants"]} == {1}
    assert {m["tests"] for m in record["mutants"]} == {TWO_FILES}


def test_without_matrix_the_run_stops_at_the_first_failure(harness, toy):
    root, _ = toy
    mut = mutation(harness, Case(ADD, SUB, TWO_FILES))
    stopped = harness.run_mutation(mut, root)
    full = harness.run_mutation(mut, root, runner=harness.run_pytest_matrix)
    assert stopped.verdict == full.verdict == harness.KILLED
    assert stopped.failed_files == ("tests/test_toy.py",)
    assert full.failed_files == ("tests/test_more.py", "tests/test_toy.py")


def test_matrix_keeps_the_exit_code_rule(harness, toy, recorded_handlers, capsys):
    # A survivor and a collection error have no failing file to list.
    root, _ = toy
    muts = [mutation(harness, dataclasses.replace(CASES["noop"], tests=TWO_FILES),
                     "survivor"),
            mutation(harness, dataclasses.replace(CASES["syntax_error"],
                                                  tests=TWO_FILES), "broken")]
    report_code = harness.main(["--matrix"], root=root, mutations=muts)
    assert report_code == 1
    printed = capsys.readouterr().out
    assert "[SURVIVED] survivor" in printed and "[ERROR rc=2] broken" in printed
    assert "failed in:" not in printed


def test_failed_files_come_from_failures_and_errors_in_the_report(harness,
                                                                  tmp_path):
    report = tmp_path / "junit.xml"
    report.write_text(
        "<testsuites><testsuite>"
        "<testcase classname='tests.test_a' name='t1' file='tests/test_a.py'/>"
        "<testcase classname='tests.test_b' name='t2' file='tests/test_b.py'>"
        "<failure message='x'/></testcase>"
        "<testcase classname='tests.test_b' name='t3' file='tests/test_b.py'>"
        "<failure message='x'/></testcase>"
        "<testcase classname='tests.test_c' name='t4' file='tests/test_c.py'>"
        "<error message='fixture'/></testcase>"
        "<testcase classname='tests.test_d' name='t5' file='tests/test_d.py'>"
        "<skipped message='s'/></testcase>"
        "<testcase classname='tests.test_e.TestX' name='t6'>"
        "<failure message='no file attribute'/></testcase>"
        "</testsuite></testsuites>")
    assert harness._failed_files(report) == (
        "tests.test_e.TestX", "tests/test_b.py", "tests/test_c.py")
    assert harness._count_passed(report) == 1
    assert harness._failed_files(tmp_path / "absent.xml") == ()


def test_tests_option_replaces_every_selector(harness, toy, recorded_handlers):
    root, _ = toy
    log = []
    muts = [mutation(harness, CASES["killed"], "a"),
            mutation(harness, CASES["quoted_k_kills"], "b")]
    code = harness.main(["--tests", "tests/test_more.py -k 'add_zero or x'"],
                        root=root, mutations=muts,
                        runner=fake_runner(harness, 0, 1, 1, log=log))
    assert code == 0
    # One baseline for the shared selector, then one run per mutant.
    assert log == [("tests/test_more.py", "-k", "add_zero or x")] * 3


def test_json_report_of_a_baseline_only_run(harness, toy, recorded_handlers,
                                            tmp_path):
    root, _ = toy
    out = tmp_path / "report.json"
    muts = [mutation(harness, CASES["killed"], "fine"),
            mutation(harness, CASES["stale"], "old text gone")]
    code = harness.main(["--baseline-only", "--json", str(out)], root=root,
                        mutations=muts, runner=fake_runner(harness, 0))
    assert code == 1
    record = json.loads(out.read_text())
    assert record["matrix"] is False
    assert [(m["label"], m["verdict"], m["returncode"], m["failed_files"])
            for m in record["mutants"]] == [
        ("fine", "BASELINE OK", None, []), ("old text gone", "STALE", None, [])]


# ---- Which tree a run imports ----

def test_tree_env_puts_the_source_of_the_tree_first(harness, tmp_path,
                                                    monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/one", "/two"]))
    env = harness.tree_env(tmp_path)
    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(tmp_path / "src"), "/one", "/two"]
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    monkeypatch.delenv("PYTHONPATH")
    assert harness.tree_env(tmp_path)["PYTHONPATH"] == str(tmp_path / "src")
    assert "PYTHONPATH" not in os.environ


def test_import_origin_is_the_tree_despite_another_one_on_the_path(
        harness, toy, elsewhere):
    root, _ = toy
    assert harness.import_origin(root) == (root / PACKAGE_REL).resolve()
    assert harness.origin_problem(root) is None


def test_origin_check_refuses_a_tree_that_imports_another_one(
        harness, toy, elsewhere, recorded_handlers, monkeypatch, capsys):
    # Without the path entry of ``tree_env`` a process started in the tree
    # imports the package of ``elsewhere``.  The sweep must not start.
    root, module = toy
    before = tree_digest(root)
    monkeypatch.setattr(harness.tree, "tree_env", lambda root_: dict(os.environ))
    assert harness.import_origin(root) == (elsewhere / PACKAGE_REL).resolve()
    log = []
    code = harness.main([], root=root, mutations=[mutation(harness, CASES["killed"])],
                        runner=fake_runner(harness, 0, 1, log=log))
    assert code == harness.EXIT_SETUP == 3
    assert log == []
    printed = capsys.readouterr().out
    assert str((elsewhere / PACKAGE_REL).resolve()) in printed
    assert tree_digest(root) == before


def test_origin_check_reports_a_package_that_does_not_import(harness, toy):
    root, _ = toy
    (root / PACKAGE_REL / "__init__.py").write_text(
        "raise RuntimeError('planted import failure')\n")
    problem = harness.origin_problem(root)
    assert problem is not None and "planted import failure" in problem


@pytest.mark.parametrize("selector, kinds", [
    (CHILD_ONLY, {"child"}), (IN_PROCESS_ONLY, {"in-process"})])
def test_in_place_sweep_runs_the_mutated_tree_in_every_process(
        harness, toy, elsewhere, recorded_handlers, selector, kinds):
    # The mutant is killed by a test that reads ``add`` in a child process,
    # so the child imported the mutated file and not the one of ``elsewhere``.
    root, module = toy
    add_origin_tests(root)
    original = module.read_bytes()
    code = harness.main([], root=root,
                        mutations=[mutation(harness, Case(ADD, SUB, selector))])
    assert code == 0
    assert module.read_bytes() == original
    assert origins(root) == dict.fromkeys(kinds, {module.resolve()})
    assert not (elsewhere / "tests" / "origin.log").exists()


# ---- A copy of the tree ----

def test_copy_tree_leaves_out_caches_and_repository_data(harness, toy, tmp_path):
    root, _ = toy
    for name in (".git", ".claude", ".agent-state", ".pytest_cache", ".hypothesis"):
        (root / name).mkdir()
        (root / name / "data").write_text("x")
    (root / PACKAGE_REL / "__pycache__").mkdir()
    (root / PACKAGE_REL / "__pycache__" / "harness_toy.cpython-312.pyc").write_bytes(b"x")
    (root / "docs" / "_build").mkdir(parents=True)
    (root / "docs" / "_build" / "index.html").write_text("x")
    (root / "docs" / "page.rst").write_text("kept\n")
    before = tree_digest(root)
    dest = tmp_path / "copy"
    assert harness.copy_tree(root, dest) == dest
    copied = tree_digest(dest)
    left_out = sorted(set(before) - set(copied))
    assert {name.split("/")[0] for name in left_out} == {
        ".git", ".claude", ".agent-state", ".pytest_cache", ".hypothesis",
        "src", "docs"}
    assert all("__pycache__" in name or "_build" in name
               for name in left_out if name.split("/")[0] in ("src", "docs"))
    assert copied == {name: digest for name, digest in before.items()
                      if name not in left_out}
    assert copied["docs/page.rst"] and copied["src/sft_wick/harness_toy.py"]
    assert tree_digest(root) == before


@pytest.mark.parametrize("where", [
    "inside", "nonempty", "only_a_subdirectory", "file", "same",
    "relative_inside", "through_a_link"])
def test_copy_tree_refuses_an_unsafe_destination(harness, toy, tmp_path,
                                                 monkeypatch, where):
    root, _ = toy
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep")
    (tmp_path / "holds_a_dir" / "empty").mkdir(parents=True)
    a_file = tmp_path / "a_file"
    a_file.write_text("x")
    (tmp_path / "link").symlink_to(root)
    monkeypatch.chdir(root)
    dest = {"inside": root / "copy", "nonempty": occupied,
            "only_a_subdirectory": tmp_path / "holds_a_dir", "file": a_file,
            "same": root, "relative_inside": pathlib.Path("copy"),
            "through_a_link": tmp_path / "link" / "copy"}[where]
    before = (tree_digest(root), tree_digest(tmp_path / "holds_a_dir"),
              tree_digest(occupied), a_file.read_text())
    with pytest.raises(harness.HarnessError):
        harness.copy_tree(root, dest)
    assert (tree_digest(root), tree_digest(tmp_path / "holds_a_dir"),
            tree_digest(occupied), a_file.read_text()) == before


@pytest.mark.parametrize("dest_exists", [False, True])
def test_a_copy_that_fails_is_removed_and_reported(harness, toy, tmp_path,
                                                   dest_exists):
    root, _ = toy
    unreadable = root / "tests" / "unreadable.txt"
    unreadable.write_text("x")
    unreadable.chmod(0)
    dest = tmp_path / "copy"
    if dest_exists:
        dest.mkdir()
    try:
        with pytest.raises(harness.HarnessError, match="partial copy was removed"):
            harness.copy_tree(root, dest)
    finally:
        unreadable.chmod(0o644)
    assert dest.exists() == dest_exists
    assert not dest_exists or list(dest.iterdir()) == []


def test_copy_tree_accepts_an_empty_directory(harness, toy, tmp_path):
    root, _ = toy
    dest = tmp_path / "empty"
    dest.mkdir()
    harness.copy_tree(root, dest)
    assert tree_digest(dest) == tree_digest(root)


def test_root_runs_the_sweep_in_a_copy_and_leaves_the_tree_identical(
        harness, toy, elsewhere, recorded_handlers, tmp_path, capsys):
    root, module = toy
    add_origin_tests(root)
    before = tree_digest(root)
    elsewhere_before = tree_digest(elsewhere)
    dest = tmp_path / "copy"
    muts = [mutation(harness, Case(ADD, SUB, CHILD_ONLY), "child sees it"),
            mutation(harness, Case(ADD, SUB, IN_PROCESS_ONLY), "pytest sees it"),
            mutation(harness, dataclasses.replace(CASES["noop"],
                                                  tests="tests/test_origin.py"),
                     "no-op")]
    code = harness.main(["--root", str(dest)], root=root, mutations=muts)
    printed = capsys.readouterr().out
    assert "[KILLED] child sees it" in printed
    assert "[KILLED] pytest sees it" in printed
    assert "[SURVIVED] no-op" in printed
    assert code == 1
    # The original tree, and the one a plain child process would import.
    assert tree_digest(root) == before
    assert tree_digest(elsewhere) == elsewhere_before
    # Every process of every run imported the copy.
    copied_module = (dest / PACKAGE_REL / "harness_toy.py").resolve()
    assert origins(dest) == {"in-process": {copied_module},
                             "child": {copied_module}}
    # The copy is restored after each mutant, like a tree mutated in place.
    assert copied_module.read_bytes() == module.read_bytes()
    assert str(dest) in printed


def test_root_given_as_a_relative_path(harness, toy, elsewhere,
                                       recorded_handlers, tmp_path, monkeypatch,
                                       capsys):
    # pytest reads a relative ``pythonpath`` against the copy itself and a
    # child process reads a relative ``PYTHONPATH`` against its own working
    # directory, so the harness has to pass both as absolute paths.
    root, _ = toy
    add_origin_tests(root)
    before = tree_digest(root)
    monkeypatch.chdir(tmp_path)
    code = harness.main(["--root", "relative-copy"], root=root,
                        mutations=[mutation(harness, Case(ADD, SUB, CHILD_ONLY),
                                            "child sees it")])
    assert "[KILLED] child sees it" in capsys.readouterr().out
    assert code == 0
    copied_module = (tmp_path / "relative-copy" / PACKAGE_REL
                     / "harness_toy.py").resolve()
    assert origins(tmp_path / "relative-copy") == {"child": {copied_module}}
    assert tree_digest(root) == before


def test_root_refused_before_any_run(harness, toy, recorded_handlers, tmp_path,
                                     capsys):
    root, _ = toy
    before = tree_digest(root)
    log = []
    code = harness.main(["--root", str(root / "tests")], root=root,
                        mutations=[mutation(harness, CASES["killed"])],
                        runner=fake_runner(harness, 0, 1, log=log))
    assert code == harness.EXIT_SETUP
    assert log == []
    assert "--root" in capsys.readouterr().out
    assert tree_digest(root) == before


def test_run_mutant_in_copy_runs_a_command_on_the_mutated_copy(
        harness, toy, elsewhere, tmp_path):
    root, module = toy
    before = tree_digest(root)
    dest = tmp_path / "mutant"
    proc = harness.run_mutant_in_copy(
        mutation(harness, CASES["killed"]), dest, [sys.executable, "-c", PROBE],
        source=root)
    assert proc.returncode == 0, proc.stderr
    imported, value = proc.stdout.split()
    copied_module = dest / PACKAGE_REL / "harness_toy.py"
    assert pathlib.Path(imported).resolve() == copied_module.resolve()
    assert value == "-1"
    # The mutant stays applied in the copy; the original is untouched.
    assert copied_module.read_bytes() == module.read_bytes().replace(
        ADD.encode(), SUB.encode())
    assert tree_digest(root) == before
    assert harness._IN_FLIGHT == {}


def test_run_in_tree_without_a_mutant_imports_that_tree(harness, toy, elsewhere):
    root, module = toy
    proc = harness.run_in_tree(root, [sys.executable, "-c", PROBE])
    imported, value = proc.stdout.split()
    assert pathlib.Path(imported).resolve() == module.resolve()
    assert value == "5"


@pytest.mark.parametrize("name", ["stale", "ambiguous", "missing_source"])
def test_mutated_copy_refuses_an_entry_that_does_not_match_once(
        harness, toy, tmp_path, name):
    root, _ = toy
    before = tree_digest(root)
    dest = tmp_path / "mutant"
    with pytest.raises(harness.HarnessError) as caught:
        harness.mutated_copy(mutation(harness, CASES[name]), dest, source=root)
    assert name == "missing_source" or "match" in str(caught.value)
    assert not dest.exists()
    assert tree_digest(root) == before


def test_mutated_copy_refuses_a_target_linked_back_into_the_source(
        harness, toy, tmp_path):
    # An absolute link is copied as it is, so in the copy it leads back to
    # the original tree.  The mutant must not be written through it.
    root, module = toy
    (root / PACKAGE_REL / "alias.py").symlink_to(module.resolve())
    before = tree_digest(root)
    with pytest.raises(harness.HarnessError, match="outside"):
        harness.mutated_copy(
            mutation(harness, dataclasses.replace(CASES["killed"], path="alias.py")),
            tmp_path / "mutant", source=root)
    assert tree_digest(root) == before


def test_mutated_copy_of_one_test_selector_is_killed_by_plain_pytest(
        harness, toy, elsewhere, tmp_path):
    # The copy is an ordinary project: its own pytest run sees the mutant.
    root, _ = toy
    dest = harness.mutated_copy(mutation(harness, CASES["killed"]),
                                tmp_path / "mutant", source=root)
    run = harness.run_pytest(dest, ONE_TEST.split())
    assert run.returncode == 1
    assert run.failed_files == ("tests/test_toy.py",)
