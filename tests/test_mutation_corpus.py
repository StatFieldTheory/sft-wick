"""The mutation corpus in ``tools/mutations/`` and its kill matrix.

The corpus holds 15 ``historical`` entries, each reintroducing a defect that
a commit of this repository fixed, and 15 ``operator`` entries, each a small
edit at a site of the numerical or symbolic pipeline.  These tests read the
corpus and the current source; they run no mutant.

``docs/verification/kill_matrix.json`` records the sweep of every entry
against the local default suite (``-m 'not slow and not heavy'``).  It is
written by ``tools/kill_matrix.py`` from the ``--matrix --json`` report of
``tools/mutation_check.py``.  The tests below require it to exist, to name
the current corpus, to name only test files the fast suite still runs, and
to record at least 29 kills; they fail until the sweep has been run and the
file committed.  A test added to or re-marked in the fast suite after the
sweep does not make them fail.

Under the harness (``MUTATION_HARNESS`` is set in the environment of every
process it starts) the tests that read the source or the kill matrix skip.
With a mutant applied the mutated file no longer holds the entry's ``old``
text, and a failure there would be counted as a kill of every mutant; and
the sweep that writes the matrix must not depend on the matrix it replaces.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import os
import pathlib
import re
import sys

import pytest

from ._mutation_toy import loaded_harness

ROOT = pathlib.Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs" / "verification" / "kill_matrix.json"
CHANGELOG = ROOT / "CHANGELOG.md"
N_PER_KIND = 15
MIN_KILLED = 29
CLASSIFICATIONS = ("coverage gap", "equivalent")


def _load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    module_name = f"{name}_under_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    with loaded_harness() as module:
        yield module


@pytest.fixture(scope="module")
def kill_matrix_tool():
    return _load_tool("kill_matrix")


@pytest.fixture(scope="module")
def corpus(harness):
    return harness.load_corpus()


@pytest.fixture
def clean_source(harness):
    if os.environ.get(harness.HARNESS_ENV):
        pytest.skip("run by the mutation harness")
    if (harness.ROOT / harness.JOURNAL_NAME).exists():
        pytest.skip("a mutant is applied to this tree")


# ---- The corpus ----

def test_the_corpus_has_15_historical_and_15_operator_entries(corpus):
    kinds = collections.Counter(mut.kind for mut in corpus)
    assert kinds == {"historical": N_PER_KIND, "operator": N_PER_KIND}
    assert len(corpus) == 2 * N_PER_KIND


def test_every_old_occurs_once_in_its_target(harness, corpus, clean_source):
    counts = [(mut.label, harness._occurrences(
        mut.path.read_text(encoding="utf-8"), mut.old)) for mut in corpus]
    assert [(label, n) for label, n in counts if n != 1] == []


def test_no_two_entries_write_the_same_file(harness, corpus, clean_source):
    seen: dict[tuple[str, str], str] = {}
    same_as_original, duplicates = [], []
    for mut in corpus:
        original = mut.path.read_text(encoding="utf-8")
        mutated = original.replace(mut.old, mut.new)
        if mutated == original:
            same_as_original.append(mut.label)
        key = (mut.relpath, mutated)
        if key in seen:
            duplicates.append((seen[key], mut.label))
        seen.setdefault(key, mut.label)
    assert same_as_original == []
    assert duplicates == []


def test_a_historical_entry_names_its_fix_and_its_changelog_heading(corpus):
    headings = {line[4:].strip() for line in
                CHANGELOG.read_text(encoding="utf-8").splitlines()
                if line.startswith("### ")}
    missing = []
    for mut in corpus:
        if mut.kind != "historical":
            continue
        # A fix older than the changelog's first entries has no heading.
        commits = re.findall(r"\b[0-9a-f]{7,40}\b", mut.why)
        quoted = re.findall(r'"([^"]+)"', mut.why)
        if not commits or not all(q in headings for q in quoted):
            missing.append(mut.label)
    assert missing == []


def test_a_kind_outside_the_two_is_a_setup_error(harness, tmp_path):
    entry = ("label", "harness_toy.py", "old", "new", "tests/test_toy.py", "why")
    untyped = ("label2",) + entry[1:]
    (tmp_path / "a.py").write_text(
        f"ENTRIES = [{entry + ('historical',)!r}, {untyped!r}]\n")
    first, second = harness.load_corpus(tmp_path)
    assert (first.kind, second.kind) == ("historical", None)
    (tmp_path / "a.py").write_text(f"ENTRIES = [{entry + ('mutant',)!r}]\n")
    with pytest.raises(harness.HarnessError, match="a.py, entry 0.*kind 'mutant'"):
        harness.load_corpus(tmp_path)


# ---- The kill matrix ----

@pytest.fixture(scope="module")
def matrix(harness):
    if os.environ.get(harness.HARNESS_ENV):
        pytest.skip("run by the mutation harness")
    if not MATRIX.exists():
        pytest.fail(f"{MATRIX.relative_to(ROOT)} is missing: run the sweep "
                    "described in docs/verification/mutation.rst")
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def test_the_kill_matrix_names_the_current_corpus(matrix, corpus,
                                                  kill_matrix_tool):
    assert [(m["label"], m["kind"]) for m in matrix["mutants"]] == [
        (mut.label, mut.kind) for mut in corpus]
    assert matrix["corpus_sha256"] == kill_matrix_tool.corpus_digest(corpus)


def test_every_killing_test_file_is_still_in_the_fast_suite(
        matrix, kill_matrix_tool, clean_source):
    files = {node.split("::", 1)[0]
             for node in kill_matrix_tool.fast_suite_node_ids(ROOT)}
    named = {f for m in matrix["mutants"] for f in m["failed_files"]}
    assert sorted(named - files) == []


def test_every_test_file_of_the_kill_matrix_exists(matrix):
    named = {f for m in matrix["mutants"] for f in m["failed_files"]}
    assert sorted(f for f in named if not (ROOT / f).is_file()) == []


def test_every_mutant_not_killed_is_classified_with_a_reason(matrix):
    unclassified = [
        m["label"] for m in matrix["mutants"]
        if m["verdict"] != "KILLED"
        and (m.get("classification") not in CLASSIFICATIONS
             or not str(m.get("reason") or "").strip())]
    classified_kills = [m["label"] for m in matrix["mutants"]
                        if m["verdict"] == "KILLED" and m.get("classification")]
    assert unclassified == []
    assert classified_kills == []


def test_at_least_29_of_the_30_mutants_are_killed(matrix):
    killed = sum(m["verdict"] == "KILLED" for m in matrix["mutants"])
    assert matrix["killed"] == killed
    assert killed >= MIN_KILLED


# ---- tools/kill_matrix.py on a small corpus ----

def _toy_corpus(harness):
    return [harness.Mutation("a", "x.py", "1", "2", "tests/", "why", "operator"),
            harness.Mutation("b", "x.py", "3", "4", "tests/", "why", "historical")]


def _toy_report(**verdicts):
    return {"matrix": True, "changed": [], "mutants": [
        {"label": label, "path": "x.py", "tests": "tests/", "verdict": verdict,
         "returncode": 1 if verdict == "KILLED" else 0,
         "failed_files": ["tests/test_a.py"] if verdict == "KILLED" else []}
        for label, verdict in verdicts.items()]}


def test_the_corpus_digest_follows_every_field_but_the_selector_and_why(
        harness, kill_matrix_tool):
    import dataclasses
    toy = _toy_corpus(harness)
    digest = kill_matrix_tool.corpus_digest(toy)
    for field, value in (("label", "c"), ("relpath", "y.py"), ("old", "5"),
                         ("new", "6"), ("kind", "historical")):
        edited = [dataclasses.replace(toy[0], **{field: value}), toy[1]]
        assert kill_matrix_tool.corpus_digest(edited) != digest, field
    same = [dataclasses.replace(toy[0], tests="tests/x.py", why="other"), toy[1]]
    assert kill_matrix_tool.corpus_digest(same) == digest


def test_a_rebuilt_matrix_keeps_the_classification_of_a_survivor(
        harness, kill_matrix_tool):
    toy = _toy_corpus(harness)
    first = kill_matrix_tool.build(_toy_report(a="KILLED", b="SURVIVED"), toy,
                                   ["tests/test_a.py::t"], previous=None)
    assert first["killed"] == 1
    assert first["totals"] == {
        "entries": 2, "by_verdict": {"KILLED": 1, "SURVIVED": 1},
        "by_kind": {"operator": {"KILLED": 1}, "historical": {"SURVIVED": 1}}}
    assert first["pytest_command"].startswith("python -m pytest tests/ ")
    assert kill_matrix_tool.unclassified(first) == ["b"]
    first["mutants"][1].update(classification="equivalent", reason="r")
    again = kill_matrix_tool.build(_toy_report(a="KILLED", b="SURVIVED"), toy,
                                   ["tests/test_a.py::t"], previous=first)
    assert again["mutants"][1]["classification"] == "equivalent"
    assert kill_matrix_tool.unclassified(again) == []
    killed_now = kill_matrix_tool.build(_toy_report(a="KILLED", b="KILLED"),
                                        toy, ["tests/test_a.py::t"],
                                        previous=first)
    assert killed_now["mutants"][1]["classification"] is None
    import dataclasses
    edited = [toy[0], dataclasses.replace(toy[1], new="5")]
    rewritten = kill_matrix_tool.build(_toy_report(a="KILLED", b="SURVIVED"),
                                       edited, ["tests/test_a.py::t"],
                                       previous=first)
    assert rewritten["mutants"][1]["classification"] is None
    assert kill_matrix_tool.unclassified(rewritten) == ["b"]


def test_a_report_that_is_not_a_matrix_or_misses_an_entry_is_refused(
        harness, kill_matrix_tool):
    toy = _toy_corpus(harness)
    report = _toy_report(a="KILLED", b="KILLED")
    with pytest.raises(ValueError, match="--matrix"):
        kill_matrix_tool.build({**report, "matrix": False}, toy, [], None)
    with pytest.raises(ValueError, match="'b'"):
        kill_matrix_tool.build(_toy_report(a="KILLED"), toy, [], None)
    with pytest.raises(ValueError, match="changed"):
        kill_matrix_tool.build({**report, "changed": ["src/x.py"]}, toy, [], None)


def test_reports_are_joined_and_a_label_in_two_is_refused(
        harness, kill_matrix_tool):
    toy = _toy_corpus(harness)
    both = _toy_report(a="KILLED", b="SURVIVED")
    shard_a = {**both, "mutants": both["mutants"][:1]}
    shard_b = {**both, "mutants": both["mutants"][1:], "changed": ["x.py"]}
    joined = kill_matrix_tool.join_reports([shard_a, shard_b])
    assert [m["label"] for m in joined["mutants"]] == ["a", "b"]
    assert joined["changed"] == ["x.py"] and joined["matrix"] is True
    assert kill_matrix_tool.join_reports(
        [shard_a, {**shard_b, "matrix": False}])["matrix"] is False
    with pytest.raises(ValueError, match="'a' is in two reports"):
        kill_matrix_tool.join_reports([shard_a, both])
    assert kill_matrix_tool.build(
        kill_matrix_tool.join_reports([shard_a, {**shard_b, "changed": []}]),
        toy, [], None)["killed"] == 1


def test_rows_of_the_matrix_are_reused_only_for_an_unchanged_entry(
        harness, kill_matrix_tool):
    import dataclasses
    toy = _toy_corpus(harness)
    previous = kill_matrix_tool.build(_toy_report(a="KILLED", b="SURVIVED"),
                                      toy, [], None)
    previous["mutants"][1].update(classification="equivalent", reason="r")
    only_a = _toy_report(a="SURVIVED")
    reused = kill_matrix_tool.reuse_rows(only_a, toy, previous)
    rebuilt = kill_matrix_tool.build(reused, toy, [], previous)
    assert [m["verdict"] for m in rebuilt["mutants"]] == ["SURVIVED", "SURVIVED"]
    assert rebuilt["mutants"][1]["classification"] == "equivalent"
    assert rebuilt["selector"] == "tests/"
    edited = [toy[0], dataclasses.replace(toy[1], new="5")]
    with pytest.raises(ValueError, match="no record of 'b'"):
        kill_matrix_tool.build(
            kill_matrix_tool.reuse_rows(only_a, edited, previous),
            edited, [], previous)
    assert kill_matrix_tool.reuse_rows(only_a, toy, None) == only_a
    with pytest.raises(ValueError, match="more than one selector"):
        kill_matrix_tool.reuse_rows(only_a, toy,
                                    {**previous, "selector": ["x", "y"]})


def test_a_reused_row_keeps_the_test_count_it_was_swept_against(
        harness, kill_matrix_tool):
    toy = _toy_corpus(harness)
    previous = kill_matrix_tool.build(_toy_report(a="KILLED", b="KILLED"),
                                      toy, ["t1", "t2"], None)
    for row in previous["mutants"]:
        del row["fast_suite_tests"]
    reused = kill_matrix_tool.reuse_rows(_toy_report(a="KILLED"), toy, previous)
    rebuilt = kill_matrix_tool.build(reused, toy, ["t1", "t2", "t3"], previous)
    assert [m["fast_suite_tests"] for m in rebuilt["mutants"]] == [3, 2]
    assert rebuilt["fast_suite_tests"] == 3
    again = kill_matrix_tool.build(
        kill_matrix_tool.reuse_rows(_toy_report(b="KILLED"), toy, rebuilt),
        toy, ["t1", "t2", "t3", "t4"], rebuilt)
    assert [m["fast_suite_tests"] for m in again["mutants"]] == [3, 4]
    catalog = _load_tool("gen_test_catalog")
    assert "swept against the 3 to 4 tests" in catalog._swept_against(again)
    assert "ran the 3 tests" in catalog._swept_against(rebuilt | {
        "mutants": rebuilt["mutants"][:1]})


def test_a_kill_by_a_flaky_test_file_alone_is_refused(harness, kill_matrix_tool):
    toy = _toy_corpus(harness)
    flaky = sorted(kill_matrix_tool.FLAKY_FILES)[0]
    report = _toy_report(a="KILLED", b="KILLED")
    report["mutants"][1]["failed_files"] = [flaky]
    with pytest.raises(ValueError, match="'b'.*run them again"):
        kill_matrix_tool.build(report, toy, [], None)
    report["mutants"][1]["failed_files"] = [flaky, "tests/test_a.py"]
    assert kill_matrix_tool.build(report, toy, [], None)["killed"] == 2


# ---- The catalogue ----

def test_the_catalogue_shows_kills_and_survivors_only_with_a_matrix(
        harness, kill_matrix_tool):
    catalog = _load_tool("gen_test_catalog")
    counts = collections.Counter(dict.fromkeys(catalog.test_files(), 1))
    plain = catalog.render(counts, kill_matrix=None)
    assert "Killed mutants" not in plain
    assert "Surviving mutants" not in plain
    toy = _toy_corpus(harness)
    fixture = kill_matrix_tool.build(_toy_report(a="KILLED", b="SURVIVED"),
                                     toy, ["tests/test_a.py::t"], None)
    fixture["mutants"][0]["failed_files"] = ["tests/test_mutation_corpus.py"]
    fixture["mutants"][1].update(classification="coverage gap",
                                 reason="no test reads it")
    text = catalog.render(counts, kill_matrix=fixture)
    assert text.count("     - Killed mutants") == len(
        [s for s in catalog.SUBJECTS if any(
            catalog.FILE_META[f][0] == s for f in catalog.test_files())])
    row = text.split("``test_mutation_corpus.py``")[1].split("   * - ")[0]
    assert row.rstrip().endswith("- 1")
    survivors = text.split("Surviving mutants")[1]
    assert "``b``" in survivors and "historical" in survivors
    assert "coverage gap" in survivors and "no test reads it" in survivors
