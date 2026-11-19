"""The record of the drawn cases that kill the historical mutants.

``docs/verification/generated_kills.json`` is written by
``tools/generated_kills.py`` from runs of ``tools/mutant_fuzz.py``, which
judge drawn cases under the unmutated tree and under each historical corpus
entry against the oracle of ``validation/``.  The tests here read the
committed record: every entry of the main line has a kill, the record names
each entry by the digest of its path, ``old`` and ``new``, and each kill
satisfies the rule it is recorded under.  They start no worker and import no
sft_wick code.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "verification" / "generated_kills.json"

under_harness = pytest.mark.skipif(
    bool(os.environ.get("MUTATION_HARNESS")),
    reason="the mutation harness has the source under test")


def _load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    module_name = f"{name}_under_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gk():
    return _load_tool("generated_kills")


@pytest.fixture(scope="module")
def record():
    if not RECORD.exists():
        pytest.fail(f"{RECORD.relative_to(ROOT)} is missing; run "
                    "python tools/generated_kills.py write OUT ...")
    return json.loads(RECORD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def targets(gk):
    mf, _ = gk._tools()
    return mf.TARGETS


@under_harness
def test_record_names_the_eight_target_entries_by_digest(gk, record,
                                                         targets):
    _, km = gk._tools()
    entries = gk.target_entries()
    assert [e["label"] for e in record["entries"]] == list(targets)
    assert len(entries) == 8
    for row, mut in zip(record["entries"], entries):
        assert row["kind"] == "historical", row["label"]
        assert row["path"] == mut.relpath, row["label"]
        assert row["sha256"] == km.entry_digest(mut), row["label"]
    assert record["corpus_sha256"] == km.corpus_digest(entries)


@under_harness
def test_every_target_entry_has_a_kill(record, targets):
    for row in record["entries"]:
        kill = row["kill"]
        assert row["killed"] and kill is not None, row["label"]
        assert kill["rule"] in ("value", "refusal"), row["label"]
        assert kill["method"] in row["methods_killed"], row["label"]
        assert kill["seed"] in row["searched"]["seeds"], row["label"]
        if kill["rule"] == "refusal":
            assert targets[row["label"]].expect is not None, row["label"]


@under_harness
def test_each_kill_meets_the_rule_it_is_recorded_under(record, targets):
    for row in record["entries"]:
        kill = row["kill"]
        target = targets[row["label"]]
        assert abs(kill["mutant"] - kill["oracle"]) > kill["allowed"], (
            row["label"])
        if kill["rule"] == "value":
            assert kill["head_error"] is None, row["label"]
            assert abs(kill["head"] - kill["oracle"]) <= kill["head_allowed"], (
                row["label"])
        else:
            assert kill["head"] is None, row["label"]
            assert re.search(target.expect, kill["head_error"]), row["label"]
        if kill["mode"] != "default":
            assert kill["control"] == ["pass", "pass"], row["label"]
        assert kill["route"] == target.route, row["label"]
        if kill["route"] == "l0":
            assert row["target"]["note"].startswith("Equivalent on L1 and L2")


@under_harness
def test_check_accepts_the_record_and_reports_a_changed_entry(gk, record,
                                                              tmp_path):
    assert gk.check(RECORD) == []
    stale = json.loads(json.dumps(record))
    stale["entries"][0]["sha256"] = "0" * 64
    stale["entries"][1]["killed"] = False
    stale["oracle_sha256"] = "0" * 64
    path = tmp_path / "generated_kills.json"
    path.write_text(json.dumps(stale), encoding="utf-8")
    problems = gk.check(path)
    assert any("changed since its kill" in p for p in problems)
    assert any("not killed" in p for p in problems)
    assert any("oracle's code changed" in p for p in problems)


def test_record_holds_no_local_path():
    text = RECORD.read_text(encoding="utf-8")
    assert "/Users/" not in text and "/private/" not in text
    assert "/tmp/" not in text and "/home/" not in text
