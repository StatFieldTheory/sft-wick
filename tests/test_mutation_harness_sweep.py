"""Running the mutation harness: restoring source, the sweep, the command line.

Unit tests for ``tools/mutation_check.py`` on the toy project of
``tests/_mutation_toy.py``.  Most of them inject a fake runner, which makes
failure paths (an exception during the mutated run, a target file changed or
deleted during the sweep) reachable without a subprocess.  Some tests of
``main`` run real ``pytest`` subprocesses on the toy project, one of them on a
mutant that loops for ever; one runs the script as a subprocess; two send
SIGTERM or SIGKILL to a harness whose mutated run is waiting.

The kill matrix, the sweep in a copied tree and the check of which tree a
run imports are tested in ``tests/test_mutation_harness_trees.py``.  The
verdict rules are tested in ``tests/test_mutation_harness.py``.
"""

from __future__ import annotations

import atexit
import dataclasses
import os
import signal
import subprocess
import sys
import time

import pytest
from ._mutation_toy import (
    ADD,
    CASES,
    PACKAGE_REL,
    REFUSED,
    ROOT,
    SCRIPT,
    SUB,
    TOY_HANG_TESTS,
    TOY_LOOP_TESTS,
    TOY_SOURCE,
    Case,
    fake_runner,
    loaded_harness,
    make_project,
    mutation,
    record_restore_handlers,
    stand_in,
)


@pytest.fixture(scope="module")
def harness():
    with loaded_harness() as module:
        yield module


@pytest.fixture
def toy(tmp_path):
    return tmp_path, make_project(tmp_path)


# ---- Where a mutation is applied, and restoring the source ----

def test_source_is_restored_when_the_mutated_run_raises(harness, toy):
    root, module = toy
    original = module.read_bytes()
    state = {"calls": 0, "seen_mutated": False}

    def runner(root_, args, timeout=None):
        state["calls"] += 1
        if state["calls"] == 1:
            return harness.PytestRun(0, 1, ())
        state["seen_mutated"] = module.read_bytes() != original
        assert harness._IN_FLIGHT == {module: original}
        state["journal"] = (root / harness.JOURNAL_NAME).exists()
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        harness.run_mutation(mutation(harness, CASES["killed"]), root, runner=runner)
    assert state == {"calls": 2, "seen_mutated": True, "journal": True}
    assert module.read_bytes() == original
    assert harness._IN_FLIGHT == {}
    assert not (root / harness.JOURNAL_NAME).exists()


def test_restore_is_bytewise_for_crlf_and_non_ascii(harness, tmp_path):
    source = ("# § 4.4 café\r\n" + TOY_SOURCE).encode("utf-8")
    module = make_project(tmp_path, source)
    during = []

    def runner(root_, args, timeout=None):
        during.append(module.read_bytes())
        return harness.PytestRun(len(during) - 1, 1, ())      # baseline 0, then 1

    result = harness.run_mutation(mutation(harness, CASES["killed"]), tmp_path,
                                  runner=runner)
    assert result.verdict == harness.KILLED
    assert during == [source, source.replace(ADD.encode(), SUB.encode())]
    assert module.read_bytes() == source


def test_restore_all_puts_back_in_flight_files(harness, toy):
    # What the signal handler and atexit hook call.
    _, module = toy
    original = module.read_bytes()
    harness._IN_FLIGHT[module] = original
    module.write_bytes(b"mutated\n")
    harness._restore_all()
    assert module.read_bytes() == original
    assert harness._IN_FLIGHT == {}


def test_mutation_paths(harness, tmp_path):
    mut = mutation(harness, CASES["killed"])
    assert mut.target(tmp_path) == tmp_path / PACKAGE_REL / "harness_toy.py"
    assert mut.path == harness.SRC / "harness_toy.py" == mut.target(harness.ROOT)
    assert harness.ROOT == ROOT


# ---- The sweep: report, summary, exit status, integrity check ----

def test_sweep_counts_and_exit_status(harness, toy, capsys):
    root, module = toy
    original = module.read_bytes()
    muts = [
        mutation(harness, CASES["killed"], "m-killed"),
        mutation(harness, dataclasses.replace(CASES["noop"], tests="t -k s"),
                 "m-survived"),
        mutation(harness, dataclasses.replace(CASES["killed"], tests="t -k e"),
                 "m-error"),
        mutation(harness, dataclasses.replace(CASES["killed"], tests="t -k b"),
                 "m-broken"),
        mutation(harness, CASES["stale"], "m-stale"),
        mutation(harness, CASES["ambiguous"], "m-ambiguous"),
    ]
    # baseline, mutated | baseline, mutated | baseline, mutated | baseline
    runner = fake_runner(harness, 0, 1, 0, 0, 0, 3, 5)
    report = harness.sweep(muts, root, runner=runner)
    assert [r.mark for _, r in report.results] == [
        "KILLED", "SURVIVED", "ERROR rc=3", "SELECTOR BROKEN", "STALE", "AMBIGUOUS"]
    assert [m.label for m, _ in report.results] == [m.label for m in muts]
    assert {v: report.count(v) for v in harness.VERDICTS} == {
        "KILLED": 1, "SURVIVED": 1, "SELECTOR BROKEN": 1, "ERROR": 1,
        "TIMEOUT": 0, "STALE": 1, "AMBIGUOUS": 1, "BASELINE OK": 0}
    assert report.changed == ()
    assert report.exit_code == 1
    assert module.read_bytes() == original
    out = capsys.readouterr().out
    assert ("killed 1 | survived 1 | selector broken 1 | error 1 | timeout 0 | "
            "stale 1 | ambiguous 1") in out.splitlines()
    # pytest's tail is attached to every verdict except KILLED.
    assert "fake tail rc=1" not in out
    assert "fake tail rc=0" in out and "fake tail rc=3" in out
    assert "fake tail rc=5" in out


@pytest.mark.parametrize("mutated_rc, exit_code", [
    (1, 0), (0, 1), (2, 1), (3, 1), (4, 1), (5, 1)])
def test_sweep_exit_status_is_zero_only_if_everything_is_killed(
        harness, toy, mutated_rc, exit_code):
    root, _ = toy
    muts = [mutation(harness, CASES["killed"], "a"),
            mutation(harness, CASES["killed"], "b")]
    report = harness.sweep(muts, root,
                           runner=fake_runner(harness, 0, 1, mutated_rc))
    assert report.exit_code == exit_code


@pytest.mark.parametrize("name", ["selects_nothing", "stale", "ambiguous",
                                  "missing_source"])
def test_sweep_exit_status_is_nonzero_for_a_refused_mutation(harness, toy, name):
    root, _ = toy
    baseline_rc = 5 if name == "selects_nothing" else 0
    report = harness.sweep([mutation(harness, CASES[name])], root,
                           runner=fake_runner(harness, baseline_rc))
    assert [r.verdict for _, r in report.results] == [REFUSED[name]]
    assert report.changed == ()
    assert report.exit_code == 1


def test_baseline_only_never_writes_source(harness, toy, capsys):
    root, module = toy
    before = (module.read_bytes(), module.stat().st_mtime_ns)
    muts = [
        mutation(harness, CASES["killed"], "ok-1"),
        mutation(harness, CASES["noop"], "ok-2"),
        mutation(harness, dataclasses.replace(CASES["killed"], tests="t -k b"),
                 "broken"),
        mutation(harness, CASES["stale"], "stale"),
        mutation(harness, CASES["ambiguous"], "ambiguous"),
    ]
    log = []
    report = harness.sweep(muts, root, baseline_only=True,
                           runner=fake_runner(harness, 0, 5, log=log))
    # One run per distinct selector and no mutated run.
    assert log == [("tests/test_toy.py", "-k", "add_small"), ("t", "-k", "b")]
    assert [r.verdict for _, r in report.results] == [
        "BASELINE OK", "BASELINE OK", "SELECTOR BROKEN", "STALE", "AMBIGUOUS"]
    assert report.exit_code == 1
    assert (module.read_bytes(), module.stat().st_mtime_ns) == before
    assert ("baseline ok 2 | selector broken 1 | stale 1 | ambiguous 1"
            in capsys.readouterr().out.splitlines())


def test_baseline_only_exit_status_is_zero_when_every_entry_is_runnable(
        harness, toy):
    root, _ = toy
    report = harness.sweep([mutation(harness, CASES["killed"])], root,
                           baseline_only=True, runner=fake_runner(harness, 0))
    assert report.exit_code == 0


def test_sweep_reports_source_that_differs_from_the_start(harness, toy, capsys):
    root, module = toy
    original = module.read_bytes()
    codes = iter([0, 1])

    def runner(root_, args, timeout=None):
        code = next(codes)
        if code == 0:                    # the baseline run edits the source
            module.write_bytes(module.read_bytes() + b"# edited during the run\n")
        return harness.PytestRun(code, 1, ())

    report = harness.sweep([mutation(harness, CASES["killed"])], root, runner=runner)
    assert [r.verdict for _, r in report.results] == ["KILLED"]
    assert module.read_bytes() != original
    assert report.changed == (module,)
    assert report.exit_code == 1
    assert str(module) in capsys.readouterr().out


def test_sweep_reports_a_target_file_deleted_during_the_run(harness, toy, capsys):
    root, module = toy

    def runner(root_, args, timeout=None):
        module.unlink()
        return harness.PytestRun(0, 1, ())

    report = harness.sweep([mutation(harness, CASES["killed"])], root,
                           baseline_only=True, runner=runner)
    assert [r.verdict for _, r in report.results] == ["BASELINE OK"]
    assert not module.exists()
    assert report.changed == (module,)
    assert report.exit_code == 1
    assert str(module) in capsys.readouterr().out


def test_target_file_that_is_not_utf8_is_stale(harness, tmp_path):
    source = b"# caf\xe9\n" + TOY_SOURCE.encode()           # Latin-1 e-acute
    with pytest.raises(UnicodeDecodeError):
        source.decode("utf-8")
    module = make_project(tmp_path, source)
    log = []
    report = harness.sweep([mutation(harness, CASES["killed"])], tmp_path,
                           runner=fake_runner(harness, 0, 1, log=log))
    (_, result), = report.results
    assert result.verdict == harness.STALE
    assert "is not UTF-8" in result.reason
    assert len(log) == 1                              # the baseline; no mutated run
    assert module.read_bytes() == source
    assert harness._IN_FLIGHT == {}
    assert report.exit_code == 1


# ---- Command line ----

def labelled(harness, *labels):
    return [mutation(harness, CASES["killed"], label) for label in labels]


def test_select_by_label_substring(harness):
    muts = labelled(harness, "free_R diagonal", "Free_C factor", "sunset sign")
    assert [m.label for m in harness.select(muts, "FREE")] == [
        "free_R diagonal", "Free_C factor"]
    assert harness.select(muts, "") == muts
    assert harness.select(muts, "no such label") == []


@pytest.mark.parametrize("total", [1, 2, 7])
@pytest.mark.parametrize("n", [1, 2, 3, 7, 9])
def test_shards_partition_the_selection(harness, total, n):
    muts = labelled(harness, *[f"m{i}" for i in range(total)])
    shards = [harness.select(muts, "", (i, n)) for i in range(1, n + 1)]
    labels = [m.label for shard in shards for m in shard]
    assert sorted(labels) == sorted(m.label for m in muts)
    assert shards[0][0] is muts[0]


@pytest.mark.parametrize("text, expected", [
    ("1/1", (1, 1)), ("2/3", (2, 3)), ("3/3", (3, 3))])
def test_parse_shard(harness, text, expected):
    assert harness.parse_shard(text) == expected


@pytest.mark.parametrize("text", ["0/3", "4/3", "1/0", "-1/3", "2", "a/b", "1/2/3", ""])
def test_parse_shard_rejects_out_of_range_and_malformed(harness, text):
    with pytest.raises(ValueError):
        harness.parse_shard(text)


@pytest.fixture
def recorded_handlers(harness, monkeypatch):
    return record_restore_handlers(harness, monkeypatch)


def test_recording_the_handlers_leaves_the_standard_library_alone(
        harness, recorded_handlers):
    def hook():
        pass

    atexit.register(hook)                    # what tempfile does on first use
    atexit.unregister(hook)
    signal.signal(signal.SIGHUP, signal.getsignal(signal.SIGHUP))
    assert recorded_handlers == {"signals": [], "atexit": []}
    harness.install_restore_handlers()
    assert recorded_handlers["atexit"] == [harness._restore_all]
    assert len(recorded_handlers["signals"]) == 3


def test_main_runs_the_selection_and_installs_the_restore_handlers(
        harness, toy, recorded_handlers, capsys):
    root, module = toy
    original = module.read_bytes()
    muts = [mutation(harness, CASES["killed"], "real kill"),
            mutation(harness, CASES["noop"], "no-op")]
    assert harness.main(["-k", "REAL"], root=root, mutations=muts) == 0
    assert set(recorded_handlers["signals"]) == {
        signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    assert recorded_handlers["atexit"] == [harness._restore_all]
    assert module.read_bytes() == original
    assert "[KILLED] real kill" in capsys.readouterr().out


def test_main_passes_k_and_shard_to_the_selection(harness, toy, recorded_handlers,
                                                  capsys):
    root, _ = toy
    muts = [mutation(harness, CASES["killed"], "first"),
            mutation(harness, CASES["killed"], "second")]

    def run(argv, *codes):
        code = harness.main(argv, root=root, mutations=muts,
                            runner=fake_runner(harness, *codes))
        return code, capsys.readouterr().out

    code, out = run([], 0, 1, 0)                 # baseline, first, second
    assert code == 1
    assert "[KILLED] first" in out and "[SURVIVED] second" in out
    code, out = run(["--shard", "2/2"], 0, 1)
    assert code == 0
    assert "first" not in out and "[KILLED] second" in out
    code, out = run(["-k", "FIR"], 0, 0)
    assert code == 1
    assert "[SURVIVED] first" in out and "second" not in out


def test_main_baseline_only(harness, toy, recorded_handlers, capsys):
    root, module = toy
    before = (module.read_bytes(), module.stat().st_mtime_ns)
    muts = [mutation(harness, CASES["noop"], "no-op"),
            mutation(harness, CASES["selects_nothing"], "nothing selected")]
    assert harness.main(["--baseline-only", "-k", "no-op"],
                        root=root, mutations=muts) == 0
    assert harness.main(["--baseline-only"], root=root, mutations=muts,
                        runner=fake_runner(harness, 0, 5)) == 1
    assert (module.read_bytes(), module.stat().st_mtime_ns) == before
    out = capsys.readouterr().out
    assert "[BASELINE OK] no-op" in out
    assert "[SELECTOR BROKEN] nothing selected" in out


def test_main_with_no_matching_mutation_exits_two(harness, toy, recorded_handlers,
                                                  capsys):
    root, _ = toy
    muts = labelled(harness, "only one")
    assert harness.main(["-k", "absent"], root=root, mutations=muts) == 2
    out = capsys.readouterr().out
    assert "no mutation selected" in out and "1 entr" in out


def test_main_with_an_empty_corpus_exits_two_and_says_so(harness, toy,
                                                         recorded_handlers, capsys):
    root, module = toy
    before = (module.read_bytes(), module.stat().st_mtime_ns)
    log = []
    code = harness.main([], root=root, mutations=[],
                        runner=fake_runner(harness, log=log))
    assert code == harness.EXIT_NOTHING_SELECTED == 2
    assert "has no entry" in capsys.readouterr().out
    assert log == []
    assert (module.read_bytes(), module.stat().st_mtime_ns) == before


@pytest.mark.parametrize("shard", ["0/2", "3/2", "x"])
def test_main_rejects_a_bad_shard(harness, toy, recorded_handlers, shard, capsys):
    root, _ = toy
    with pytest.raises(SystemExit) as exc:
        harness.main(["--shard", shard], root=root, mutations=labelled(harness, "a"))
    assert exc.value.code == 2
    assert "--shard" in capsys.readouterr().err


def test_script_entry_point_exits_two_when_nothing_is_selected():
    # No label matches, so no test is run and no source is written.
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "-k", "no mutation has this label"],
        capture_output=True, text=True)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "no mutation" in proc.stdout


DRIVER = '''\
import importlib.util
import pathlib
import sys

spec = importlib.util.spec_from_file_location("mutation_check", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
mut = module.Mutation("hang", "harness_toy.py", "return a + b", "return a - b",
                      "tests/test_hang.py", "why")
raise SystemExit(module.main([], root=pathlib.Path(sys.argv[2]), mutations=[mut]))
'''


def test_sigterm_during_a_mutated_run_restores_the_source(tmp_path):
    module = make_project(tmp_path)
    original = module.read_bytes()
    (tmp_path / "tests" / "test_hang.py").write_text(TOY_HANG_TESTS)
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    flag = tmp_path / "tests" / "mutated.flag"
    proc = subprocess.Popen(
        [sys.executable, str(driver), str(SCRIPT), str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    try:
        deadline = time.monotonic() + 60.0
        while not flag.exists() and proc.poll() is None:
            assert time.monotonic() < deadline, "the mutated run never started"
            time.sleep(0.02)
        assert proc.poll() is None, proc.communicate()[0]
        assert module.read_bytes() == original.replace(ADD.encode(), SUB.encode())
        proc.send_signal(signal.SIGTERM)
        output, _ = proc.communicate(timeout=60.0)
    finally:
        (tmp_path / "tests" / "stop.flag").write_text("x")
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == 128 + signal.SIGTERM, output
    assert module.read_bytes() == original
    assert "source restored" in output



def test_sigkill_leaves_a_journal_the_next_run_restores_from(harness, tmp_path,
                                                             recorded_handlers,
                                                             capsys):
    # SIGKILL cannot be caught; the journal written before the mutant is how
    # the next run finds and undoes it, without touching other files.
    module = make_project(tmp_path)
    original = module.read_bytes()
    (tmp_path / "tests" / "test_hang.py").write_text(TOY_HANG_TESTS)
    other = tmp_path / PACKAGE_REL / "__init__.py"
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    flag = tmp_path / "tests" / "mutated.flag"
    proc = subprocess.Popen(
        [sys.executable, str(driver), str(SCRIPT), str(tmp_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    try:
        deadline = time.monotonic() + 60.0
        while not flag.exists() and proc.poll() is None:
            assert time.monotonic() < deadline, "the mutated run never started"
            time.sleep(0.02)
        proc.kill()
        proc.wait()
    finally:
        (tmp_path / "tests" / "stop.flag").write_text("x")
    assert module.read_bytes() == original.replace(ADD.encode(), SUB.encode())
    assert (tmp_path / harness.JOURNAL_NAME).exists()
    other.write_text("# an edit of the user's, not the harness's\n")
    code = harness.main(["-k", "nothing has this label"], root=tmp_path,
                        mutations=[mutation(harness, CASES["killed"])])
    assert code == 2
    assert module.read_bytes() == original
    assert other.read_text() == "# an edit of the user's, not the harness's\n"
    assert not (tmp_path / harness.JOURNAL_NAME).exists()
    assert "after a killed run" in capsys.readouterr().out


def write_killed_run_journal(harness, root, module, current: bytes | None):
    """Leave the journal of a run killed with the ``killed`` mutant applied."""
    original = module.read_bytes()
    mutant = original.replace(ADD.encode(), SUB.encode())
    harness.write_journal(root, module, original, mutant)
    module.write_bytes(current if current is not None else mutant)


@pytest.mark.parametrize("state", ["mutant", "original"])
def test_recover_restores_only_the_recorded_mutant(harness, toy, state):
    root, module = toy
    original = module.read_bytes()
    write_killed_run_journal(harness, root, module,
                             None if state == "mutant" else original)
    assert "holds its original bytes" in harness.recover(root)
    assert module.read_bytes() == original
    assert not (root / harness.JOURNAL_NAME).exists()
    assert harness.recover(root) is None


def test_recover_refuses_a_file_changed_since_the_kill(harness, toy,
                                                       recorded_handlers, capsys):
    root, module = toy
    edited = b"def add(a, b):\n    return 42\n"
    write_killed_run_journal(harness, root, module, edited)
    log = []
    code = harness.main([], root=root, mutations=[mutation(harness, CASES["killed"])],
                        runner=fake_runner(harness, 0, 1, log=log))
    assert code == harness.EXIT_SETUP
    assert log == []
    assert module.read_bytes() == edited
    assert (root / harness.JOURNAL_NAME).exists()
    assert "has changed since" in capsys.readouterr().out


# ---- Time limits ----

def test_mutated_run_limit_is_a_multiple_of_the_baseline_or_the_option(harness,
                                                                       toy):
    root, _ = toy
    timeouts = []

    def runner(root_, args, timeout=None):
        timeouts.append(timeout)
        return harness.PytestRun(0 if len(timeouts) % 2 else 1, 1, (), seconds=2.0)

    mut = mutation(harness, CASES["killed"])
    assert harness.run_mutation(mut, root, runner=runner).verdict == harness.KILLED
    assert timeouts == [None, 10 * 2.0 + 30]
    timeouts.clear()
    baselines = harness.Baselines(root, runner, timeout=7.0)
    assert harness.run_mutation(mut, root, baselines, runner).verdict == harness.KILLED
    assert timeouts == [7.0, 7.0]


def test_a_baseline_that_times_out_is_broken(harness, toy):
    root, module = toy
    original = module.read_bytes()

    def runner(root_, args, timeout=None):
        return harness.PytestRun(-9, None, ("cut",), timed_out=True)

    result = harness.run_mutation(mutation(harness, CASES["killed"]), root,
                                  harness.Baselines(root, runner, 1.0), runner)
    assert result.verdict == harness.BROKEN
    assert "--timeout" in result.reason
    assert module.read_bytes() == original


def pid_is_gone(pid: int, wait: float = 10.0) -> bool:
    """Whether no process ``pid`` exists, polled for up to ``wait`` seconds."""
    deadline = time.monotonic() + wait
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def test_a_mutant_that_loops_is_killed_with_its_process_group(
        harness, toy, recorded_handlers, capsys):
    root, module = toy
    original = module.read_bytes()
    (root / "tests" / "test_loop.py").write_text(TOY_LOOP_TESTS)
    started = time.monotonic()
    code = harness.main(["--timeout", "6"], root=root,
                        mutations=[mutation(harness, Case(ADD, SUB, "tests/test_loop.py"),
                                            "loops")])
    assert time.monotonic() - started < 60
    out = capsys.readouterr().out
    assert code == 1
    assert "[TIMEOUT] loops" in out and "timeout 1" in out
    assert module.read_bytes() == original
    # A child of the test process ended with the group.
    pid = int((root / "tests" / "grandchild.pid").read_text())
    try:
        assert pid_is_gone(pid)
    finally:
        if not pid_is_gone(pid, wait=0):
            os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize("text", ["0", "-1", "soon"])
def test_main_rejects_a_bad_timeout(harness, toy, recorded_handlers, text, capsys):
    root, _ = toy
    with pytest.raises(SystemExit) as exc:
        harness.main(["--timeout", text], root=root, mutations=labelled(harness, "a"))
    assert exc.value.code == 2
    assert "--timeout" in capsys.readouterr().err


# ---- Command line: setup checks ----

def test_an_empty_shard_of_a_non_empty_selection_exits_zero(harness, toy,
                                                           recorded_handlers,
                                                           capsys):
    root, _ = toy
    log = []
    code = harness.main(["--shard", "3/3"], root=root,
                        mutations=labelled(harness, "a", "b"),
                        runner=fake_runner(harness, log=log))
    assert code == 0
    assert log == []
    assert "shard 3/3 of the 2 selected mutation(s) is empty" in capsys.readouterr().out


def test_main_refuses_an_invalid_corpus(harness, toy, recorded_handlers,
                                        monkeypatch, tmp_path, capsys):
    root, _ = toy
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "area.py").write_text("ENTRIES = [('a', 'x.py', 'o', 'n', 't')]\n")
    monkeypatch.setattr(harness, "MUTATION_DIR", corpus)
    assert harness.main([], root=root) == harness.EXIT_SETUP
    assert "area.py, entry 0" in capsys.readouterr().out


def test_main_refuses_a_json_file_in_a_missing_directory(harness, toy,
                                                         recorded_handlers,
                                                         tmp_path, capsys):
    root, _ = toy
    log = []
    code = harness.main(["--json", str(tmp_path / "absent" / "out.json")],
                        root=root, mutations=labelled(harness, "a"),
                        runner=fake_runner(harness, 0, 1, log=log))
    assert code == harness.EXIT_SETUP
    assert log == []
    assert "--json" in capsys.readouterr().out


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0,
                    reason="needs POSIX permissions and a non-root user")
def test_main_refuses_a_json_file_in_a_read_only_directory(harness, toy,
                                                          recorded_handlers,
                                                          tmp_path, capsys):
    root, _ = toy
    read_only = tmp_path / "read-only"
    read_only.mkdir()
    read_only.chmod(0o555)
    log = []
    try:
        code = harness.main(["--json", str(read_only / "out.json")],
                            root=root, mutations=labelled(harness, "a"),
                            runner=fake_runner(harness, 0, 1, log=log))
    finally:
        read_only.chmod(0o755)
    assert code == harness.EXIT_SETUP
    assert log == []
    assert "not a writable directory" in capsys.readouterr().out


def test_restore_handlers_are_registered_once_however_often_main_runs(
        harness, monkeypatch):
    registered = []
    monkeypatch.setattr(harness, "signal", stand_in(
        signal, signal=lambda sig, handler: None))
    monkeypatch.setattr(harness, "atexit", stand_in(
        atexit, register=registered.append,
        unregister=lambda fn: registered.remove(fn) if fn in registered else None))
    harness.install_restore_handlers()
    harness.install_restore_handlers()
    assert registered == [harness._restore_all]


def test_a_handler_that_cannot_be_installed_is_reported(harness, monkeypatch,
                                                       capsys):
    def refuse(sig, handler):
        raise ValueError("signal only works in main thread")

    monkeypatch.setattr(harness, "signal", stand_in(signal, signal=refuse))
    monkeypatch.setattr(harness, "atexit", stand_in(
        atexit, register=lambda fn: None, unregister=lambda fn: None))
    harness.install_restore_handlers()
    err = capsys.readouterr().err
    assert err.count("no restore handler") == 3
    assert harness.JOURNAL_NAME in err
