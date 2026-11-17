"""The verdicts of the mutation harness, ``tools/mutation_check.py``.

The harness decides whether a test caught a defect.  These tests run it on
the toy project of ``tests/_mutation_toy.py``, built under ``tmp_path``: one
module, ``src/sft_wick/harness_toy.py``, and a few test files.  The test
process imports no ``sft_wick``, toy or real, and the repository's own tests
are not run.

Three kinds of test:

* ``outcomes`` runs the harness with real ``pytest`` subprocesses, once per
  module, over the scenarios in ``CASES``.  The tests below it read the
  recorded verdicts.  A spy around the runner records the source bytes on disk
  at every ``pytest`` invocation, so "the mutation was applied" and "the
  mutation was never applied" are observed.
* the tests after those inject a fake runner, which makes exit codes that are
  hard to provoke (3, a signal death) reachable without a subprocess.
* the last group checks how the corpus is read and validated, and reads the
  corpus in ``tools/mutations/`` against the current source.  It runs no test.

Restoring the source, the sweep, time limits and the command line are tested
in ``tests/test_mutation_harness_sweep.py``; the kill matrix and copied trees
in ``tests/test_mutation_harness_trees.py``.
"""

from __future__ import annotations

import collections
import dataclasses
import shlex

import pytest
from ._mutation_toy import (
    CASES,
    ONE_TEST,
    PACKAGE_REL,
    REFUSED,
    fake_runner,
    loaded_harness,
    make_project,
    mutation,
)


@pytest.fixture(scope="module")
def harness():
    with loaded_harness() as module:
        yield module


# ---- Real pytest subprocesses, one sweep over CASES ----

#: One pytest invocation: its arguments, the toy module's bytes on disk when it
#: returned, and the ``PytestRun``.
Call = collections.namedtuple("Call", "args source run")
#: One scenario: the ``Result``, the calls made for it, the module afterwards.
Outcome = collections.namedtuple("Outcome", "result calls source_after")
Outcomes = collections.namedtuple("Outcomes", "root original by_name")


@pytest.fixture(scope="module")
def outcomes(harness, tmp_path_factory):
    root = tmp_path_factory.mktemp("toy")
    module = make_project(root)
    original = module.read_bytes()
    seen: list[Call] = []

    def spy(root_, args, timeout=None):
        run = harness.run_pytest(root_, args, timeout=timeout)
        seen.append(Call(tuple(args), module.read_bytes(), run))
        return run

    baselines = harness.Baselines(root, runner=spy)
    by_name = {}
    for name, case in CASES.items():
        start = len(seen)
        result = harness.run_mutation(
            mutation(harness, case, name), root, baselines, runner=spy)
        by_name[name] = Outcome(result, tuple(seen[start:]), module.read_bytes())
    return Outcomes(root, original, by_name)


def mutated_sources(outcomes, name):
    """Distinct non-original source texts pytest saw while ``name`` ran."""
    return {c.source for c in outcomes.by_name[name].calls
            if c.source != outcomes.original}


def expected_mutant(outcomes, name):
    case = CASES[name]
    assert outcomes.original.count(case.old.encode()) == 1
    return outcomes.original.replace(case.old.encode(), case.new.encode())


def test_real_mutation_is_killed(harness, outcomes):
    result = outcomes.by_name["killed"].result
    assert len(CASES["killed"].old) == len(CASES["killed"].new)
    assert result.verdict == harness.KILLED
    assert result.returncode == 1
    assert mutated_sources(outcomes, "killed") == {expected_mutant(outcomes, "killed")}


def test_noop_mutation_survives(harness, outcomes):
    result = outcomes.by_name["noop"].result
    assert result.verdict == harness.SURVIVED
    assert result.returncode == 0
    # It ran against the changed text; it did not survive by not being applied.
    assert mutated_sources(outcomes, "noop") == {expected_mutant(outcomes, "noop")}
    assert any("passed" in line for line in result.tail)


def test_quoted_k_expression_arrives_as_one_argument(harness, outcomes):
    case = CASES["quoted_k_kills"]
    assert case.tests.split() != shlex.split(case.tests)
    out = outcomes.by_name["quoted_k_kills"]
    assert out.result.verdict == harness.KILLED
    assert {c.args for c in out.calls} == {
        ("tests/test_toy.py", "-k", "add_small or scale_double")}


def test_quoted_k_expression_limits_the_selection(harness, outcomes):
    # Same mutation as above; this expression leaves out the one test that
    # sees it.  Dropping -k and running the whole file would kill it.
    assert CASES["quoted_k_misses"].new == CASES["quoted_k_kills"].new
    assert outcomes.by_name["quoted_k_misses"].result.verdict == harness.SURVIVED
    assert mutated_sources(outcomes, "quoted_k_misses")


def test_quoted_m_expression_is_applied(harness, outcomes):
    assert outcomes.by_name["plain_m_kills"].result.verdict == harness.KILLED
    assert outcomes.by_name["quoted_m_misses"].result.verdict == harness.SURVIVED
    assert mutated_sources(outcomes, "quoted_m_misses")


def test_addopts_deselects_slow_unless_the_selector_has_its_own_m(harness,
                                                                 outcomes):
    # One mutation, three selectors over the same file.  The project's
    # addopts hides the slow test that sees it; ``-m slow`` in the selector
    # replaces the expression in addopts.
    assert (CASES["default_m_misses"].new == CASES["plain_m_kills"].new
            == CASES["slow_without_m"].new)
    assert outcomes.by_name["default_m_misses"].result.verdict == harness.SURVIVED
    assert mutated_sources(outcomes, "default_m_misses")
    assert outcomes.by_name["plain_m_kills"].calls[0].run.passed == 1


@pytest.mark.parametrize("name", ["slow_without_m", "heavy_without_m"])
def test_selector_for_a_deselected_test_says_no_passing_test(harness, outcomes,
                                                             name):
    out = outcomes.by_name[name]
    assert [c.run.returncode for c in out.calls] == [5]
    assert out.result.verdict == harness.BROKEN
    assert "no passing test" in out.result.reason
    assert "slow" in out.result.reason and "heavy" in out.result.reason
    assert "-m" in out.result.reason


@pytest.mark.parametrize("name, baseline_rc", [
    ("selects_nothing", 5),
    ("unquoted_or", 4),
    ("missing_test_file", 4),
    ("red_baseline", 1),
])
def test_selector_failing_on_clean_source_is_broken(harness, outcomes, name,
                                                    baseline_rc):
    out = outcomes.by_name[name]
    assert [c.run.returncode for c in out.calls] == [baseline_rc]
    assert out.result.verdict == harness.BROKEN
    assert out.result.returncode is None
    assert f"exit code {baseline_rc}" in out.result.reason
    assert "no passing test" in out.result.reason
    # The hint about deselected markers belongs to "no tests collected" only.
    assert (harness.DESELECTED_HINT in out.result.reason) == (baseline_rc == 5)
    assert out.result.tail == out.calls[0].run.tail != ()


def test_tail_is_the_end_of_stdout_and_stderr(harness, outcomes):
    # A failure report is longer than TAIL_LINES; the summary is its last line.
    failing = outcomes.by_name["red_baseline"].calls[0].run.tail
    assert len(failing) == harness.TAIL_LINES == 8
    assert all(line.strip() for line in failing)
    assert "1 failed" in failing[-1]
    # pytest writes a usage error to stderr.
    usage = outcomes.by_name["unquoted_or"].calls[0].run.tail
    assert any("file or directory not found: or" in line for line in usage)


def test_selector_that_only_skips_is_broken(harness, outcomes):
    out = outcomes.by_name["only_skipped"]
    assert [(c.run.returncode, c.run.passed) for c in out.calls] == [(0, 0)]
    assert out.result.verdict == harness.BROKEN
    assert "no passing test" in out.result.reason


def test_unparseable_selector_is_broken_without_running_pytest(harness, outcomes):
    out = outcomes.by_name["unbalanced_quote"]
    assert out.calls == ()
    assert out.result.verdict == harness.BROKEN


def test_passed_count_excludes_skipped_and_deselected(outcomes):
    # The expression leaves add_small, scale_double and the skipped test; two
    # of them pass.
    baseline = outcomes.by_name["quoted_m_misses"].calls[0].run
    assert (baseline.returncode, baseline.passed) == (0, 2)


def test_syntax_error_is_an_error_not_a_kill(harness, outcomes):
    result = outcomes.by_name["syntax_error"].result
    assert mutated_sources(outcomes, "syntax_error")
    assert result.verdict == harness.ERROR
    assert result.returncode == 2
    assert result.mark == "ERROR rc=2"
    assert any("SyntaxError" in line for line in result.tail)


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_refused_mutation_is_never_written(outcomes, name):
    out = outcomes.by_name[name]
    assert out.result.verdict == REFUSED[name]
    assert out.result.returncode is None
    assert mutated_sources(outcomes, name) == set()


@pytest.mark.parametrize("name, count", [("stale", 0), ("ambiguous", 2)])
def test_old_must_occur_exactly_once(outcomes, name, count):
    assert outcomes.original.count(CASES[name].old.encode()) == count
    assert f"{count} match" in outcomes.by_name[name].result.reason


@pytest.mark.parametrize("name", sorted(CASES))
def test_source_is_restored_after_every_case(harness, outcomes, name):
    assert outcomes.by_name[name].source_after == outcomes.original
    assert harness._IN_FLIGHT == {}


def test_runs_leave_no_cache_files(outcomes):
    assert sum(len(o.calls) for o in outcomes.by_name.values()) >= 10
    leftovers = [p for p in outcomes.root.rglob("*")
                 if p.name in ("__pycache__", ".pytest_cache")]
    assert leftovers == []


# ---- Verdict rules, with a fake runner ----

@pytest.fixture
def toy(tmp_path):
    return tmp_path, make_project(tmp_path)


@pytest.mark.parametrize("code, verdict, mark", [
    (0, "SURVIVED", "SURVIVED"), (1, "KILLED", "KILLED"),
    *[(code, "ERROR", f"ERROR rc={code}") for code in (2, 3, 4, 5, 6, 137, -9, -1)],
])
def test_only_exit_code_one_is_a_kill(harness, toy, code, verdict, mark):
    root, _ = toy
    result = harness.run_mutation(
        mutation(harness, CASES["killed"]), root,
        runner=fake_runner(harness, 0, code))
    assert (result.verdict, result.mark, result.returncode) == (verdict, mark, code)
    assert result.tail == (f"fake tail rc={code}",)


@pytest.mark.parametrize("code, passed, ok", [
    (0, 1, True), (0, 250, True), (0, 0, False), (0, None, False),
    (1, 1, False), (2, 0, False), (3, 0, False), (4, None, False),
    (5, 0, False), (-9, None, False),
])
def test_baseline_needs_exit_zero_and_a_passing_test(harness, toy, code, passed, ok):
    root, module = toy
    original = module.read_bytes()
    log = []
    result = harness.run_mutation(
        mutation(harness, CASES["killed"]), root,
        runner=fake_runner(harness, code, 1, passed=passed, log=log))
    assert (result.verdict == harness.KILLED) == ok
    assert (result.verdict == harness.BROKEN) == (not ok)
    assert len(log) == (2 if ok else 1)
    assert module.read_bytes() == original


def test_each_distinct_selector_is_baselined_once(harness, toy):
    root, _ = toy
    log = []
    runner = fake_runner(harness, *[0] * 5, log=log)
    baselines = harness.Baselines(root, runner=runner)
    checks = [baselines.check(tests) for tests in (
        "tests/test_toy.py -k add_small",
        "tests/test_toy.py   -k   add_small",
        "tests/test_toy.py -k 'add_small'",
        "tests/test_toy.py -k scale",
        "tests/test_toy.py -k add_small",
    )]
    assert all(check.ok for check in checks)
    assert log == [("tests/test_toy.py", "-k", "add_small"),
                   ("tests/test_toy.py", "-k", "scale")]


def test_a_broken_baseline_is_cached_too(harness, toy):
    root, _ = toy
    log = []
    baselines = harness.Baselines(root, runner=fake_runner(harness, 5, log=log))
    first = baselines.check(ONE_TEST)
    assert baselines.check(ONE_TEST) == first
    assert not first.ok
    assert len(log) == 1


def test_stale_entry_with_a_broken_selector_reports_both(harness, toy):
    root, module = toy
    original = module.read_bytes()
    case = dataclasses.replace(CASES["stale"], tests="tests/test_toy.py -k none")
    log = []
    result = harness.run_mutation(mutation(harness, case), root,
                                  runner=fake_runner(harness, 5, log=log))
    assert result.verdict == harness.STALE
    assert "0 matches" in result.reason and "exit code 5" in result.reason
    assert result.tail == ("fake tail rc=5",)
    assert len(log) == 1
    assert module.read_bytes() == original


@pytest.mark.parametrize("where", ["directory", "link_out"])
def test_target_that_is_not_a_file_inside_the_package_is_stale(harness, toy,
                                                              where):
    # A directory, or a link that leads out of src/sft_wick, is never written.
    root, module = toy
    outside = root / "outside.py"
    outside.write_bytes(module.read_bytes())
    (root / PACKAGE_REL / "sub").mkdir()
    (root / PACKAGE_REL / "linked.py").symlink_to(outside)
    relpath = {"directory": "sub", "link_out": "linked.py"}[where]
    case = dataclasses.replace(CASES["killed"], path=relpath)
    before = outside.read_bytes()
    result = harness.run_mutation(mutation(harness, case), root,
                                  runner=fake_runner(harness, 0, 1))
    assert result.verdict == harness.STALE
    assert ("not a file" if where == "directory" else "outside") in result.reason
    assert outside.read_bytes() == before


def test_snapshot_covers_an_empty_target_file(harness, tmp_path):
    module = make_project(tmp_path, b"")
    before = harness.snapshot([mutation(harness, CASES["killed"])], tmp_path)
    assert before == {module: b""}
    module.write_bytes(b"x = 1\n")
    assert harness.changed_files(before) == (module,)


def test_passed_count_is_none_without_a_readable_report(harness, tmp_path):
    assert harness._count_passed(tmp_path / "absent.xml") is None
    truncated = tmp_path / "truncated.xml"
    truncated.write_text("<testsuites><testsuite>")
    assert harness._count_passed(truncated) is None
    empty = tmp_path / "empty.xml"
    empty.write_text("<testsuites><testsuite tests='0'/></testsuites>")
    assert harness._count_passed(empty) == 0


def test_module_docstring_names_every_verdict(harness):
    missing = [v for v in harness.VERDICTS if f"``{v}" not in harness.__doc__]
    assert missing == []


# ---- The corpus in tools/mutations/: static checks only, no test is run ----

def test_corpus_modules_are_discovered_from_the_directory(harness):
    on_disk = sorted(p.stem for p in harness.MUTATION_DIR.glob("*.py"))
    assert on_disk
    assert list(harness.mutation_modules()) == on_disk
    entries = [tuple(fields) for name in on_disk
               for fields in harness.load_entries(name)]
    assert [(m.label, m.relpath, m.old, m.new, m.tests, m.why)
            for m in harness.load_corpus()] == entries


def test_discovery_reads_any_directory_in_name_order(harness, tmp_path):
    entry = ("label", "harness_toy.py", "old", "new", "tests/test_toy.py", "why")
    (tmp_path / "b_second.py").write_text(f"ENTRIES = [{entry!r}]\n")
    (tmp_path / "a_first.py").write_text("ENTRIES = []\n")
    (tmp_path / "notes.txt").write_text("not a module\n")
    assert harness.mutation_modules(tmp_path) == ("a_first", "b_second")
    (mut,) = harness.load_corpus(tmp_path)
    assert (mut.label, mut.relpath, mut.old, mut.new) == entry[:4]


@pytest.mark.parametrize("text", [
    "ENTRY = []\n", "ENTRIES = None\n", "ENTRIES = [\n", "raise KeyError('x')\n"])
def test_corpus_module_that_cannot_be_read_is_an_error(harness, tmp_path, text):
    (tmp_path / "broken.py").write_text(text)
    with pytest.raises(harness.HarnessError, match="broken.py"):
        harness.load_corpus(tmp_path)


GOOD_ENTRY = ("label", "harness_toy.py", "old", "new", "tests/test_toy.py", "why")


@pytest.mark.parametrize("entry, problem", [
    (GOOD_ENTRY[:5], "6 fields"),
    ("not a tuple", "6 fields"),
    (GOOD_ENTRY[:4] + (3, "why"), "not a string"),
    (("label", "/abs/harness_toy.py") + GOOD_ENTRY[2:], "not relative"),
    (("label", "../outside.py") + GOOD_ENTRY[2:], "not relative"),
    (("label", "") + GOOD_ENTRY[2:], "not relative"),
    (("label", "harness_toy.py", "", "new") + GOOD_ENTRY[4:], "empty `old`"),
    (("label", "harness_toy.py", "same", "same") + GOOD_ENTRY[4:], "equal"),
    (("  ",) + GOOD_ENTRY[1:], "empty label"),
    (GOOD_ENTRY[:4] + ("", "why"), "empty label or selector"),
])
def test_corpus_entry_is_validated(harness, tmp_path, entry, problem):
    (tmp_path / "area.py").write_text(f"ENTRIES = [{GOOD_ENTRY!r}, {entry!r}]\n")
    with pytest.raises(harness.HarnessError, match="area.py, entry 1") as caught:
        harness.load_corpus(tmp_path)
    assert problem in str(caught.value)


def test_corpus_labels_are_unique_across_modules(harness, tmp_path):
    (tmp_path / "a.py").write_text(f"ENTRIES = [{GOOD_ENTRY!r}]\n")
    (tmp_path / "b.py").write_text(f"ENTRIES = [{GOOD_ENTRY!r}]\n")
    with pytest.raises(harness.HarnessError, match="b.py, entry 0.*a.py, entry 0"):
        harness.load_corpus(tmp_path)


def test_a_mutant_whose_old_overlaps_itself_is_ambiguous(harness, toy):
    # ``str.count`` finds one "\n\n" in the three newlines after ``add``.
    root, module = toy
    original = module.read_bytes()
    assert original.count(b"\n\n") == 1
    case = dataclasses.replace(CASES["killed"], old="\n\n", new="\n#\n")
    result = harness.run_mutation(mutation(harness, case), root,
                                  runner=fake_runner(harness, 0))
    assert result.verdict == harness.AMBIGUOUS
    assert "2 matches" in result.reason
    assert module.read_bytes() == original


def test_every_listed_mutation_applies_to_the_current_source(harness):
    # One row per entry, so two entries with one label are both checked.
    corpus = harness.load_corpus()
    matches = [(m.label, harness._occurrences(m.path.read_text(encoding="utf-8"),
                                              m.old))
               for m in corpus]
    assert [(label, n) for label, n in matches if n != 1] == []
