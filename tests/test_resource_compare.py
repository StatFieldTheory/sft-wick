"""Ordered numerical gates, protected inputs, and true fresh-process probes."""

import copy
import json
import os
import py_compile
from pathlib import Path

import pytest
import yaml

from sft_wick.workflow import resource_compare as rc


def _raw(method="qmc_vectorized"):
    return {
        "system": {
            "field": {"n_components": 1},
            "linear": {"type": "diagonal", "gamma": [1.0]},
            "vertices": [{"name": "F", "coupling": [[[0.1]]]}],
            "noise": {
                "kappa2": {
                    "type": "separable_translation",
                    "temporal": {"type": "exponential", "lam": 0.05, "sigma_t": 0.3},
                    "spatial": {"type": "exponential", "sigma_x": 1.0},
                }
            },
        },
        "expand": {"observable": ["phi(x)", "phi(y)"], "orders": [0, 2]},
        "propagators": {"t_max": 1.0, "n_grid_t": 5},
        "sweep": {
            "positions_grid": {"x": [0.0], "y": [0.0]},
            "t_final_grid": [0.5, 0.8],
            "component_pairs": [[0, 0]],
            "method": method,
            "n_samples": 16,
            "n_gauss": 3,
            "seed": 7,
        },
    }


def _files(tmp_path, raw=None, choices=None):
    config, variants = tmp_path / "config.yaml", tmp_path / "variants.yaml"
    config.write_text(yaml.safe_dump(_raw() if raw is None else raw, sort_keys=False))
    variants.write_text(
        yaml.safe_dump(
            {"reuse": {"reuse_integrands": True}} if choices is None else choices,
            sort_keys=False,
        )
    )
    return config, variants, tmp_path / "report.json"


def _payload():
    return dict(
        rows=[
            dict(
                x=0.0,
                t_final=0.5,
                a=0,
                b=0,
                order=0,
                diagram_idx=0,
                vertex_type="",
                n_cross_C=1,
                value=0.2,
                error=0.01,
            )
        ],
        position_keys=["x"],
        external_time_keys=[],
        component_keys=["a", "b"],
        totals_columns=["x", "t_final", "a", "b", "order", "value"],
        totals_records=[dict(x=0.0, t_final=0.5, a=0, b=0, order=0, value=0.2)],
    )


def _child(request, timeout, source_root):
    return dict(
        ok=True,
        result=_payload(),
        workflow_seconds=0.2,
        config_load_seconds=0.01,
        traced_peak_bytes=1000,
        rss_lifetime_peak_bytes=10000,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"n_samples": 8},
        {"seed": 8},
        {"method": "qmc_scalar"},
        {"gl_chunk_size": True},
        {"qmc_coupling_chunk_size": 0},
        {"reuse_integrands": 1},
        {},
    ],
)
def test_only_strict_resource_variants_allowed(tmp_path, changes):
    paths = _files(tmp_path, choices={"bad": changes})
    with pytest.raises(ValueError):
        rc.compare_resources(*paths)
    assert not paths[-1].exists()


@pytest.mark.parametrize("choices", [{}, {"baseline": {"reuse_integrands": True}}])
def test_named_nonempty_variants(tmp_path, choices):
    with pytest.raises(ValueError):
        rc.compare_resources(*_files(tmp_path, choices=choices))


def test_duplicate_keys_and_yaml_merge_semantics():
    with pytest.raises(ValueError, match="duplicate"):
        rc._safe_mapping(
            "reuse: {reuse_integrands: true}\nreuse: {gl_chunk_size: 4}", "variants"
        )
    assert rc._safe_mapping(
        "defaults: &d {a: 1, b: 2}\ncase: {<<: *d, a: 3}", "config"
    )["case"] == {"a": 3, "b": 2}


@pytest.mark.parametrize("path", [123, False, "", None])
def test_declared_input_paths_must_be_strings(tmp_path, path):
    raw = _raw()
    raw["system"]["vertices"] = [{"name": "F", "coupling_path": path}]
    with pytest.raises(ValueError, match="path string"):
        rc.compare_resources(*_files(tmp_path, raw))


@pytest.mark.parametrize("section", ["expand", "propagators", "sweep"])
def test_serial_only(tmp_path, section):
    raw = _raw()
    raw[section]["n_jobs"] = 2
    with pytest.raises(ValueError, match="n_jobs"):
        rc.compare_resources(*_files(tmp_path, raw))


def test_unseeded_qmc_rejected_but_gl_seed_is_preserved(tmp_path, monkeypatch):
    raw = _raw()
    raw["sweep"]["seed"] = None
    paths = _files(tmp_path, raw)
    with pytest.raises(ValueError, match="fixed"):
        rc.compare_resources(*paths)
    raw["sweep"]["method"] = "gauss_legendre"
    paths = _files(tmp_path, raw)
    requests = []

    def child(request, *args):
        requests.append(request)
        assert yaml.safe_load(request["config_text"])["sweep"]["seed"] is None
        return _child(request, *args)

    monkeypatch.setattr(rc, "_run_child", child)
    result = rc.compare_resources(*paths)
    assert result["ok"] and len(requests) == 8


@pytest.mark.parametrize("field", ["error", "value", "diagram_idx", "vertex_type"])
def test_full_rows_not_only_totals_are_checked(field):
    base, changed = _payload(), _payload()
    changed["rows"][0][field] = 0.03 if field in {"error", "value"} else "changed"
    assert not rc.compare_results(base, changed)["within_tolerance"]


@pytest.mark.parametrize(
    "field", ["position_keys", "external_time_keys", "component_keys", "totals_columns"]
)
def test_all_axis_and_total_column_metadata(field):
    base, changed = _payload(), _payload()
    changed[field] = list(reversed(changed[field])) + ["different"]
    assert not rc.compare_results(base, changed, atol=100)["metadata_equal"]


def test_order_cancellation_tolerance_and_nonfinite():
    base = _payload()
    base["rows"].append(dict(base["rows"][0], diagram_idx=1, value=-0.1))
    changed = copy.deepcopy(base)
    changed["rows"].reverse()
    assert not rc.compare_results(base, changed)["metadata_equal"]
    changed = copy.deepcopy(base)
    changed["rows"][0]["value"] += 1.0
    changed["rows"][1]["value"] -= 1.0
    assert not rc.compare_results(base, changed)["within_tolerance"]
    changed = copy.deepcopy(base)
    changed["rows"][0]["value"] += 1e-10
    checked = rc.compare_results(base, changed, atol=1e-9)
    assert checked["within_tolerance"] and not checked["exact"]
    changed["rows"][0]["error"] = float("nan")
    assert not rc.compare_results(base, changed, atol=100)["within_tolerance"]
    changed["rows"] = []
    assert not rc.compare_results(base, changed)["within_tolerance"]


def test_extreme_tolerance_does_not_accept_overflow():
    base, changed = _payload(), _payload()
    base["rows"][0]["value"] = 1e308
    changed["rows"][0]["value"] = -1.6e308
    check = rc.compare_results(base, changed, rtol=2.0)
    assert not check["within_tolerance"] and check["difference_overflow"]
    assert rc.compare_results(base, changed, rtol=3.0)["within_tolerance"]


def test_encoder_native_and_wide_longdouble_boundary():
    import numpy as np
    from sft_wick.workflow.result import SweepResult

    value = np.longdouble(0.25)
    sweep = SweepResult(
        rows=[
            dict(
                x=np.array([0.0, 0.5], dtype=np.longdouble),
                value=value,
                error=np.float64(0.01),
            )
        ],
        position_keys=("x",),
    )

    class Totals:
        columns = ["x", "value"]

        def to_dict(self, orient):
            assert orient == "records"
            return [dict(x=(0.0, 0.5), value=value)]

    if value.dtype.itemsize > 8:
        with pytest.raises(TypeError, match="unsupported"):
            rc.encode_results(sweep, Totals())
    else:
        payload = rc.encode_results(sweep, Totals())
        assert payload["rows"][0] == dict(x=[0.0, 0.5], value=0.25, error=0.01)
        assert isinstance(payload["rows"][0]["value"], float)


def test_unchanged_case_round_order_and_report_protection(tmp_path, monkeypatch):
    paths = _files(tmp_path)
    monkeypatch.setattr(rc, "_run_child", _child)
    result = rc.compare_resources(*paths)
    assert result["ok"] and result["execution_count"] == 8
    assert [(r["case"], r["mode"]) for r in result["execution_order"]] == [
        ("baseline", "timing"),
        ("reuse", "timing"),
        ("baseline", "memory"),
        ("reuse", "memory"),
        ("reuse", "timing"),
        ("baseline", "timing"),
        ("reuse", "memory"),
        ("baseline", "memory"),
    ]
    with pytest.raises(FileExistsError):
        rc.compare_resources(*paths)
    rc.compare_resources(*paths, force=True)
    with pytest.raises(ValueError, match="replace"):
        rc.compare_resources(paths[0], paths[1], paths[0], force=True)
    hardlink = tmp_path / "alias.yaml"
    os.link(paths[0], hardlink)
    with pytest.raises(ValueError, match="replace"):
        rc.compare_resources(paths[0], paths[1], hardlink, force=True)


def test_repeat_and_signed_zero_gates_withhold_ratios(tmp_path, monkeypatch):
    paths = _files(tmp_path)
    count = 0

    def child(request, *args):
        nonlocal count
        count += 1
        result = _child(request, *args)
        if count == 3:
            result["result"]["rows"][0]["error"] += 0.1
        return result

    monkeypatch.setattr(rc, "_run_child", child)
    report = rc.compare_resources(*paths, atol=1.0)
    assert not report["ok"]
    assert all("workflow_time_ratio" not in c for c in report["cases"].values())

    def zeros(request, *args):
        result = _child(request, *args)
        result["result"]["rows"][0]["value"] = (
            -0.0 if request["resource_overrides"] else 0.0
        )
        return result

    monkeypatch.setattr(rc, "_run_child", zeros)
    report = rc.compare_resources(*paths, force=True)
    assert not report["ok"] and report["cases"]["reuse"]["stable"]


def test_declared_module_change_and_deletion_are_failures(tmp_path, monkeypatch):
    module = tmp_path / "kernel.data"
    module.write_text("old module")
    raw = _raw()
    raw["system"]["noise"]["kappa2"] = {
        "type": "callable_module",
        "module": "kernel.data",
    }
    paths = _files(tmp_path, raw)
    assert module.resolve() in rc._known_files(raw, tmp_path)

    def child(request, *args):
        module.unlink()
        return _child(request, *args)

    monkeypatch.setattr(rc, "_run_child", child)
    report = rc.compare_resources(*paths)
    assert not report["ok"] and not report["known_inputs_unchanged"]
    assert report["input_fingerprints_after"][str(module.resolve())] is None
    assert json.loads(paths[-1].read_text())["failures"]


def test_timeout_is_a_written_failure_report(tmp_path):
    paths = _files(tmp_path)
    report = rc.compare_resources(*paths, timeout=0.001)
    assert not report["ok"] and report["runs"][0]["failure"] == "timeout"
    assert paths[-1].is_file()


def test_declared_symlink_retarget_with_identical_bytes_is_detected(
    tmp_path, monkeypatch
):
    target_a, target_b = tmp_path / "a.py", tmp_path / "b.py"
    target_a.write_text("raise AssertionError('parent must not load callbacks')")
    target_b.write_bytes(target_a.read_bytes())
    link = tmp_path / "callback.py"
    link.symlink_to(target_a)
    raw = _raw()
    raw["system"]["vertices"] = [
        {"name": "F", "rank": 3, "coupling_module": "callback.py"}
    ]
    paths = _files(tmp_path, raw)

    def child(request, *args):
        link.unlink()
        link.symlink_to(target_b)
        return _child(request, *args)

    monkeypatch.setattr(rc, "_run_child", child)
    result = rc.compare_resources(*paths)
    assert not result["ok"] and not result["known_inputs_unchanged"]
    assert result["input_links_before"][str(link)] == str(target_a)
    assert result["input_links_after"][str(link)] == str(target_b)


def test_symlink_config_preserves_lexical_base_dir(tmp_path, monkeypatch):
    source, modules = tmp_path / "source", tmp_path / "modules"
    source.mkdir()
    modules.mkdir()
    callback = modules / "callback.py"
    callback.write_text("raise AssertionError('parent must not load callbacks')")
    raw = _raw()
    raw["system"]["vertices"] = [
        {"name": "F", "rank": 3, "coupling_module": "callback.py"}
    ]
    config, variants, report = _files(source, raw)
    link = modules / "config.yaml"
    link.symlink_to(config)

    def child(request, *args):
        assert request["base_dir"] == str(modules)
        assert str(callback) in request["input_fingerprints"]
        return _child(request, *args)

    monkeypatch.setattr(rc, "_run_child", child)
    assert rc.compare_resources(link, variants, report)["ok"]


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
@pytest.mark.parametrize("force", [False, True])
def test_late_report_alias_does_not_modify_inputs(tmp_path, monkeypatch, kind, force):
    config, variants, report = _files(tmp_path)
    original = config.read_bytes()

    def child(request, *args):
        if not report.exists():
            if kind == "symlink":
                report.symlink_to(config)
            else:
                os.link(config, report)
        return _child(request, *args)

    monkeypatch.setattr(rc, "_run_child", child)
    if force:
        result = rc.compare_resources(config, variants, report, force=True)
        assert result["ok"] and json.loads(report.read_text())["ok"]
        assert not report.is_symlink() and not report.samefile(config)
    else:
        with pytest.raises(FileExistsError):
            rc.compare_resources(config, variants, report)
    assert config.read_bytes() == original
    assert not list(tmp_path.glob(".report.json.*.tmp"))


def test_late_report_directory_retarget_fails_closed(tmp_path, monkeypatch):
    config, variants, _ = _files(tmp_path)
    directory = tmp_path / "reports"
    directory.mkdir()
    report = directory / "report.json"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    def child(request, *args):
        if not directory.is_symlink():
            directory.rename(tmp_path / "original-directory")
            directory.symlink_to(elsewhere, target_is_directory=True)
        return _child(request, *args)

    monkeypatch.setattr(rc, "_run_child", child)
    with pytest.raises(ValueError, match="directory changed"):
        rc.compare_resources(config, variants, report, force=True)
    assert not list(elsewhere.iterdir())


def test_atomic_report_preserves_existing_regular_permissions(tmp_path, monkeypatch):
    config, variants, report = _files(tmp_path)
    report.write_text("old")
    report.chmod(0o640)
    monkeypatch.setattr(rc, "_run_child", _child)
    rc.compare_resources(config, variants, report, force=True)
    assert report.stat().st_mode & 0o777 == 0o640


def test_cli_failure_and_complete_report_table(tmp_path, monkeypatch, capsys):
    from sft_wick.workflow.cli import main

    config, variants, report = _files(tmp_path)
    monkeypatch.setattr(rc, "_run_child", _child)
    assert (
        main(
            [
                "compare-resources",
                str(config),
                "--variants",
                str(variants),
                "--json",
                str(report),
            ]
        )
        == 0
    )
    assert "traced(load+workflow)" in capsys.readouterr().out
    assert (
        main(
            [
                "compare-resources",
                str(config),
                "--variants",
                str(variants),
                "--json",
                str(report),
            ]
        )
        == 1
    )


def test_real_fresh_static_gl_reuse_chunk_and_no_outputs(tmp_path):
    raw = _raw("gauss_legendre")
    output, expansion, propagators = [
        tmp_path / name for name in ("output.csv", "expand.cache", "props.cache")
    ]
    for path in (output, expansion, propagators):
        path.write_bytes(b"protected sentinel")
    raw["output"] = [{"type": "table", "format": "csv", "path": str(output)}]
    raw["expand"]["cache_path"] = str(expansion)
    raw["propagators"]["cache_path"] = str(propagators)
    paths = _files(
        tmp_path,
        raw,
        {"reuse": {"reuse_integrands": True}, "chunk": {"gl_chunk_size": 2}},
    )
    report = rc.compare_resources(*paths, rtol=1e-12, atol=1e-14)
    assert report["ok"] and report["execution_count"] == 12
    assert all(
        run["package_path"].startswith(str(Path(rc.__file__).parents[1]))
        for run in report["runs"]
    )
    assert all(run["source_fingerprints"] for run in report["runs"])
    assert all(
        path.read_bytes() == b"protected sentinel"
        for path in (output, expansion, propagators)
    )
    assert all(case["stable"] for case in report["cases"].values())


def test_real_dynamic_qmc_relative_module_ignores_stale_bytecode(tmp_path):
    # Compile an equal-length old callback, then change source while preserving
    # timestamp and file size. -B alone would still load the stale bytecode.
    module = tmp_path / "coupling.py"
    old = "import numpy as np\ndef coupling_fn(n, t):\n    return np.full((np.asarray(t).shape[-1], 1, 1, 1), 0.0)\n"
    new = old.replace("0.0)", "0.1)")
    module.write_text(old)
    py_compile.compile(str(module), doraise=True)
    stat = module.stat()
    module.write_text(new)
    os.utime(module, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    raw = _raw()
    raw["system"]["vertices"] = [
        {
            "name": "F",
            "rank": 3,
            "coupling_module": "./coupling.py",
            "coupling_vectorized": True,
        }
    ]
    paths = _files(tmp_path, raw, {"chunks": {"qmc_coupling_chunk_size": 4}})
    report = rc.compare_resources(*paths)
    assert report["ok"] and report["execution_count"] == 8
    assert report["input_fingerprints_before"][str(module.resolve())] == rc._hash(
        module
    )
    assert any(
        row["order"] == 2 and row["value"] != 0
        for row in report["runs"][0]["result"]["rows"]
    )
