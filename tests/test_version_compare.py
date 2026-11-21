"""Exercise the release comparison using planted outcomes in two installs."""

import hashlib
import json
import math
from zipfile import ZipFile

import pytest

from sft_wick.workflow.cli import main


def _install(path, values, version):
    pkg = path / "sft_wick"
    workflow = pkg / "workflow"
    workflow.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f"__version__ = {version!r}\n")
    (workflow / "__init__.py").touch()
    (workflow / "config.py").write_text(
        f"""from types import SimpleNamespace

VALUES = {values!r}


def _parse_workflow(raw, base_dir):
    assert raw["output"] == []
    assert raw["expand"]["cache_path"] is None
    assert raw["propagators"]["cache_path"] is None
    return raw


class Totals(list):
    def to_dict(self, orient):
        assert orient == "records"
        return [{{"order": r["order"], "x": r["x"], "value": r["value"]}} for r in self]


def run_workflow(config, progress):
    assert progress is False
    if config["system"].get("refuse") and "refuse" in VALUES:
        raise ValueError("planted workflow refusal")
    rows = [
        dict(order=2, x=x, error=e, value=v)
        for x, (v, e) in VALUES.items() if x != "refuse"
    ]
    return SimpleNamespace(rows=rows), Totals(rows)
"""
    )
    return path


def _corpus(path, *case_lines):
    corpus = path / "corpus.yaml"
    corpus.write_text("cases:\n" + "\n".join(f"  - {case}" for case in case_lines))
    return corpus


def _config(path, *, method="qmc_vectorized", refuse=False, seed=42):
    path.write_text(
        f"""system: {{refuse: {str(refuse).lower()}}}
expand: {{cache_path: retained.joblib}}
propagators: {{cache_path: retained.joblib}}
sweep: {{method: {method}, seed: {seed if seed is not None else 'null'}}}
output: [{{type: table, path: retained.txt}}]
"""
    )
    return path


def _compare(corpus, baseline, candidate, path, *extra):
    code = main([
        "compare-versions", str(corpus),
        "--baseline", str(baseline),
        "--candidate", str(candidate),
        "--json", str(path),
        *map(str, extra),
    ])
    return code, json.loads(path.read_text()) if path.exists() else None


def _rows(report, case_id="probe"):
    selected = next(case for case in report["cases"] if case["id"] == case_id)
    return {row["coordinates"]["x"]: row for row in selected["rows"]}


def test_version_gate_uses_planted_values_and_requires_qmc_declarations(tmp_path):
    baseline = _install(tmp_path / "old" / "src", {
        0: (1.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1e-8), 3: (1.0, 1e-8),
    }, "0.5.0")
    candidate_dir = _install(tmp_path / "new", {
        0: (1.0, 0.0), 1: (math.nextafter(1.0, math.inf), 0.0),
        2: (1.0 + 1e-9, 1e-8), 3: (1.001, 1e-8),
    }, "0.6.0")
    candidate = tmp_path / "candidate.whl"
    with ZipFile(candidate, "w") as archive:
        for file in candidate_dir.rglob("*.py"):
            archive.write(file, file.relative_to(candidate_dir))

    _config(tmp_path / "qmc.yaml")
    corpus = _corpus(tmp_path, "{id: probe, config: qmc.yaml}")
    code, report = _compare(corpus, baseline.parent, candidate, tmp_path / "report.json")
    assert code == 1
    assert report["baseline"]["version"] == "0.5.0"
    assert report["candidate"]["version"] == "0.6.0"
    rows = _rows(report)
    assert {x: row["classification"] for x, row in rows.items()} == {
        0: "bitwise", 1: "round-off", 2: "within uncertainty", 3: "moved",
    }
    assert not rows[2]["declared"]
    assert not rows[3]["declared"]

    declared = tmp_path / "declared.yaml"
    declared.write_text(
        """changes:
  - {case: probe, where: {x: 2}, kind: moved, reason: Declared QMC movement}
  - {case: probe, where: {x: 3}, kind: redrawn, reason: Changed sampler}
"""
    )
    code, report = _compare(
        corpus, baseline.parent, candidate, tmp_path / "declared.json",
        "--declared", declared,
    )
    assert code == 0
    assert _rows(report)[2]["classification"] == "within uncertainty"
    assert _rows(report)[2]["declared"]
    assert _rows(report)[3]["classification"] == "redrawn"
    assert len(report["candidate"]["sha256"]) == 64
    assert not (tmp_path / "retained.joblib").exists()
    assert not (tmp_path / "retained.txt").exists()


def test_version_gate_aligns_by_physical_axes_and_reports_absent_rows(tmp_path):
    baseline = _install(tmp_path / "old", {0: (2.0, 0.0), 1: (3.0, 0.0)}, "0.5")
    candidate = _install(tmp_path / "new", {2: (4.0, 0.0), 0: (2.0, 0.0)}, "0.6")
    _config(tmp_path / "gl.yaml", method="gauss_legendre")
    corpus = _corpus(tmp_path, "{id: probe, config: gl.yaml}")
    code, report = _compare(corpus, baseline, candidate, tmp_path / "report.json")
    assert code == 1
    assert {x: row["classification"] for x, row in _rows(report).items()} == {
        0: "bitwise", 1: "row only on one side", 2: "row only on one side",
    }


def test_version_gate_distinguishes_refusal_and_allows_other_cases_to_run(tmp_path):
    baseline = _install(tmp_path / "old", {0: (2.0, 0.0), "refuse": (0.0, 0.0)}, "0.5")
    candidate = _install(tmp_path / "new", {0: (2.0, 0.0)}, "0.6")
    _config(tmp_path / "refuse.yaml", refuse=True)
    for name, extra in (("date", "2026-01-02"), ("infinity", ".inf")):
        config = _config(tmp_path / f"{name}.yaml")
        with config.open("a") as stream:
            stream.write(f"extra: {extra}\n")
    _config(tmp_path / "normal.yaml")
    corpus = _corpus(
        tmp_path,
        "{id: refusal, config: refuse.yaml}",
        "{id: date, config: date.yaml}",
        "{id: infinity, config: infinity.yaml}",
        "{id: normal, config: normal.yaml}",
    )
    code, report = _compare(corpus, baseline, candidate, tmp_path / "report.json")
    assert code == 1
    assert report["cases"][0]["classification"] == "refused on one side"
    assert report["cases"][0]["candidate"]["rows"][0]["value"] == 2.0
    for case in report["cases"][1:3]:
        assert case["classification"] == "refused on both sides"
        assert "could not encode worker request" in case["baseline"]["failure"]
        assert "could not encode worker request" in case["candidate"]["failure"]
    assert _rows(report, "normal")[0]["classification"] == "bitwise"


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e300])
def test_version_gate_roundoff_boundary_is_relative_at_every_scale(scale):
    from sft_wick.workflow.version_compare import _classify

    far = scale
    while abs(far - scale) / far <= 4.5e-16:
        near = far
        far = math.nextafter(far, math.inf)
    baseline = {"value": scale, "error": None}
    assert _classify(baseline, {"value": near, "error": None}, False) in {"bitwise", "round-off"}
    assert _classify(baseline, {"value": far, "error": None}, False) == "moved"
    assert _classify({"value": 0.0, "error": None}, baseline, False) == "moved"


def test_version_driver_runs_against_selected_package(tmp_path):
    baseline = _install(tmp_path / "old", {}, "0.5")
    candidate = _install(tmp_path / "new", {}, "0.7")
    (tmp_path / "driver.py").write_text(
        """from sft_wick import __version__

def run():
    return [dict(coordinates=dict(x=0, order=2), value=float(__version__), error=None)]
"""
    )
    corpus = _corpus(tmp_path, "{id: probe, driver: driver.py, method: gauss_legendre}")
    code, report = _compare(corpus, baseline, candidate, tmp_path / "report.json")
    assert code == 1
    assert _rows(report)[0]["baseline"]["value"] == 0.5
    assert _rows(report)[0]["candidate"]["value"] == 0.7


def test_version_gate_does_not_overwrite_input(tmp_path):
    baseline = _install(tmp_path / "old", {0: (2.0, 0.0)}, "0.5")
    candidate = _install(tmp_path / "new", {0: (2.0, 0.0)}, "0.6")
    config = _config(tmp_path / "probe.yaml")
    text = config.read_text()
    corpus = _corpus(tmp_path, "{id: probe, config: probe.yaml}")
    assert main([
        "compare-versions", str(corpus),
        "--baseline", str(baseline), "--candidate", str(candidate),
        "--json", str(config), "--force",
    ]) == 1
    assert config.read_text() == text


def test_version_gate_refuses_unseeded_qmc(tmp_path):
    values = {0: (2.0, 0.0)}
    baseline = _install(tmp_path / "old", values, "0.5")
    candidate = _install(tmp_path / "new", values, "0.6")
    _config(tmp_path / "probe.yaml", seed=None)
    corpus = _corpus(tmp_path, "{id: probe, config: probe.yaml}")
    code, report = _compare(corpus, baseline, candidate, tmp_path / "report.json")
    assert code == 1
    assert report["cases"][0]["classification"] == "refused on both sides"
    assert "fixed sweep.seed" in report["cases"][0]["baseline"]["failure"]


def test_version_gate_hashes_the_input_that_was_parsed(tmp_path, monkeypatch):
    from sft_wick.workflow import version_compare

    values = {0: (2.0, 0.0)}
    baseline = _install(tmp_path / "old", values, "0.5")
    candidate = _install(tmp_path / "new", values, "0.6")
    config = _config(tmp_path / "probe.yaml")
    original = config.read_bytes()
    corpus = _corpus(tmp_path, "{id: probe, config: probe.yaml}")
    load_cases = version_compare._load_cases

    def replace_after_load(*args):
        loaded = load_cases(*args)
        config.write_bytes(original + b"\n# Replaced after loading\n")
        return loaded

    monkeypatch.setattr(version_compare, "_load_cases", replace_after_load)
    code, report = _compare(corpus, baseline, candidate, tmp_path / "report.json")
    assert code == 1
    assert report["input_sha256"][str(config.resolve())] == hashlib.sha256(original).hexdigest()
    assert report["input_sha256_after"][str(config.resolve())] != hashlib.sha256(original).hexdigest()
