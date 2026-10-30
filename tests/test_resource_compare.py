"""Ordered numerical gates, protected inputs, and true fresh-process probes."""

import copy
import itertools
import json
import os
import py_compile
import time
import tracemalloc
from pathlib import Path

import pytest
import yaml

from sft_wick.workflow import resource_compare as rc
from sft_wick.workflow.config import (
    ConfigError,
    _load_yaml,
)


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


@pytest.mark.parametrize("text", [
    "case: {<<: {a: 1, a: 2}}",
    "case: {<<: [{a: 1, a: 2}]}",
    "case: {<<: [{a: 1}, {b: 2, b: 3}]}",
    "case: {<<: &d {a: 1, a: 2}}",
    "case: {<<: {<<: {a: 1, a: 2}}}",
], ids=["inline-mapping", "inline-list", "second-source", "anchor-in-merge",
        "nested-merge"])
def test_duplicate_keys_inside_a_merge_source(text):
    """A mapping written in place as the value of ``<<:`` was flattened
    without the duplicate check, so ``{a: 1, a: 2}`` gave ``a: 2``."""
    with pytest.raises(ValueError, match=r"duplicate mapping key 'case\.[ab]'"):
        rc._safe_mapping(text, "variants")


def test_merge_overrides_still_apply():
    case = rc._safe_mapping("case: {<<: [{a: 1}, {a: 2, b: 3}], c: 4}",
                            "config")
    assert case["case"] == {"a": 1, "b": 3, "c": 4}
    nested = rc._safe_mapping("case: {<<: {<<: {a: 1}, a: 5}}", "config")
    assert nested["case"] == {"a": 5}


#: Legal uses of ``<<``, each read by plain ``yaml.safe_load`` as the
#: reference: an anchor merged after it was read (its own merge is flattened
#: by then), chains, diamonds, an alias used again as a plain value, an
#: anchor defined in a merge, a node that merges itself, ``=`` as a key.
LEGAL_MERGES = {
    "anchor-with-override": "small: &s {<<: {c: 2}, c: 4}\n"
                            "large: {<<: *s, c: 8}\n",
    "anchor-with-list": "small: &s {<<: [{c: 2}, {c: 4}]}\n"
                        "large: {<<: *s, n: 3}\n",
    "chain": "a: &a {<<: {c: 1}, c: 2}\nb: &b {<<: *a, c: 3}\n"
             "d: {<<: *b, e: 4}\n",
    "diamond": "a: &a {c: 1}\nb: &b {<<: *a, d: 2}\n"
               "e: {<<: [*a, *b], f: 3}\ng: {<<: [*b, *a]}\n",
    "merged-twice": "a: &a {<<: {c: 1}, c: 2}\nb: {<<: *a}\nd: {<<: *a}\n",
    "merged-then-plain-value": "a: &a {<<: {c: 1}, c: 2}\nb: {<<: *a}\nd: *a\n",
    "anchor-in-a-merge": "a: {<<: &s {c: 1}, d: 2}\nb: {<<: *s, c: 3}\n",
    "self-merge": "case: &b {<<: *b, c: 4}\n",
    "self-merge-in-a-list": "case: &b {<<: [*b], c: 4}\n",
    "nested-value": "a: &a {n: {<<: {x: 1}, x: 2}}\nb: {<<: *a}\n",
    "value-key": "=: 1\nb: 2\n",
    "quoted-merge-word": "y: {<<: {a: 1}, '<<': 2}\n",
    "mutual-merge": "p:\n  r: &a\n    <<:\n      - {k: 1}\n"
                    "      - &b {<<: *a}\nz: *b\n",
    "value-tag-keeps-its-text": "!!value a: 1\n!!value b: 2\n",
    "value-tag-beside-equals": "!!value a: 1\n=: 2\n",
    "merge-tag-text-as-a-key": "<<: {a: 1}\ntag:yaml.org,2002:merge: 2\n",
    "string-from-a-mapping-key": "? !!str {=: a}\n: 1\nb: 2\n",
    "merge-under-a-complex-key": "? !!merge [x]\n: {a: 1}\nb: 2\n",
    "null-beside-a-merge": "~: 1\n<<: {a: 1}\n",
    "override-under-a-key-in-pairs": "a: !!pairs [? {<<: {c: 1}, c: 2} : 1]\n",
    "override-beside-a-key-reaching-its-mapping":
        "&A\nx: !!str {=: z, ? [*A] : 1}\nlater: {<<: {a: 1}, a: 2}\n",
    "key-of-pairs-merging-a-scalar-read-mapping":
        "a: !!pairs\n  - ? {<<: &m !!str {=: x}}\n    : 1\nb: *m\n",
    "unread-key-of-an-undefined-tag": "a: !!str {=: z, ? !!foo x : 1}\n",
    "unread-key-that-is-no-int": "a: !!str {=: z, !!int zz : 1}\n",
    "unread-key-that-is-no-date": "a: !!null {? = : z, ? !!timestamp a : b}\n",
    "unread-key-that-is-no-bool": "a: !!str {=: z, !!bool x : 1}\n",
    "unread-key-with-an-empty-int": "a: !!str {=: z, !!int '' : 1}\n",
    "unread-timestamp-of-a-mapping": "a: !!str {=: z, ? !!timestamp {=: x} : 1}\n",
    "unread-key-too-large": "a: !!str {=: z, 1" + ":0" * 180 + ".0 : 1}\n",
    "unread-key-holding-itself": "x: !!str {=: z, ? &a !!str {=: *a} : 1}\n",
    "two-unread-keys-of-no-type": "a: !!str {=: z, !!foo b: 1, !!foo c: 2}\n",
    "two-unread-keys-as-lists": "a: !!str {=: z, ? [a] : 1, ? [b] : 2}\n",
    "repeat-in-a-mapping-read-as-a-scalar": "? !!str {=: k, b: 1, b: 2}\n: 3\n",
    "two-merge-keys-in-a-mapping-read-as-a-scalar":
        "a: !!str {=: z, <<: {b: 1}, <<: {c: 2}}\n",
    "shared-key-twice-in-a-mapping-read-as-a-scalar":
        "{&k a: 1, ? !!str {=: z, *k : 1, *k : 2} : 3}\n",
    "set-key-in-pairs": "p: !!pairs\n- ? !!set {a: null}\n  : 1\n",
    "value-key-retagged-before-a-scalar-read":
        "{&k !!value z: a, ? !!str {*k : b, =: c} : 1, b: 2}\n",
    "value-key-retagged-after-a-scalar-read":
        "a: [{&k !!value z: a}]\nb: {? !!str {*k : b, =: c} : 1, c: 2}\n",
    "not-a-number-written-twice": "{!!float nan: 1, !!float nan: 2}\n",
    "key-at-the-end-of-a-long-chain":
        "a0: &a0 [1]\n"
        + "".join(f"a{i}: &a{i} [*a{i - 1}]\n" for i in range(1, 400))
        + "p: !!pairs\n- ? *a399\n  : 1\n",
}

#: Keys written twice; plain YAML keeps the last of each without a word.
REPEATED_KEYS = {
    "in-an-anchor-merged-later": "a: &a {c: 1, c: 2}\nb: {<<: *a}\n",
    "in-a-value-of-a-source": "case: {<<: {n: {x: 1, x: 2}}}\n",
    "merge-key": "y: {<<: {a: 1}, <<: {a: 2}}\n",
    "null": "~: 1\nnull: 2\n",
    "value-key": "=: 1\n=: 2\n",
    "equals-and-quoted-equals": "=: 1\n\"=\": 2\n",
    "value-tag-repeats-its-text": "!!value a: 1\na: 2\n",
    "merge-key-under-a-complex-key": "? !!merge [x]\n: {a: 1}\n<<: {b: 2}\n",
    "in-a-merge-under-a-complex-key": "? !!merge [x]\n: {a: 1, a: 2}\n",
    "string-from-a-mapping-key": "? !!str {=: a}\n: 1\na: 2\n",
    "int-from-a-mapping-key": "? !!int {=: 1}\n: 1\n1: 2\n",
    "in-the-value-of-a-complex-key": "? !!str {=: k}\n: {b: 1, b: 2}\n",
    "value-key-retagged-before-a-scalar-read":
        "{&k !!value z: a, ? !!str {*k : b, =: c} : 1, c: 2}\n",
    "value-key-retagged-after-a-scalar-read":
        "a: [{&k !!value z: a}]\nb: {? !!str {*k : b, =: c} : 1, b: 2}\n",
    "deep-in-a-key-read-as-a-scalar":
        "? !!str {=: a, z: " + "[" * 200 + "]" * 200 + "}\n: 1\na: 2\n",
    "at-the-end-of-a-chain-in-a-key-read-as-a-scalar":
        "c0: &c0 [x]\n"
        + "".join(f"c{i}: &c{i} [*c{i - 1}]\n" for i in range(1, 200))
        + "m:\n  ? !!str {=: a, z: *c199}\n  : 1\n  a: 2\n",
    "an-integer-and-a-float": "1: a\n1.0: b\n",
    "a-flag-and-an-integer": "true: a\n1: b\n",
    "not-a-number-by-alias": "{&k !!float nan: 1, *k : 2}\n",
    "dot-nan-by-alias": "{&k .nan: 1, *k : 2}\n",
    "dot-nan-written-twice": "{.nan: 1, .nan: 2}\n",
    "value-tag-on-a-mapping-key": "? !!value {=: a}\n: 1\na: 2\n",
    "after-a-key-reaching-its-mapping":
        "&A\n<<: {m: 1}\nx: !!str {=: z, ? [*A] : 1}\nb: 1\nb: 2\n",
    "merged-in-after-a-key-reaching-its-mapping":
        "&A\nx: !!str {=: z, ? [*A] : 1}\n<<: {m: 1, m: 2}\n",
    "merge-key-after-a-key-reaching-its-mapping":
        "&m\nn: !!str {? [*m] : 1, =: v}\n<<: {a: 1}\n<<: {b: 2}\n",
    "after-a-key-of-omap-reaching-its-mapping":
        "&A\n<<: {m: 1}\nx: !!omap [{? [*A] : 1}]\nb: 1\nb: 2\n",
    "past-the-digit-limit-of-str":
        "? 0x" + "f" * 4000 + "\n: 1\n? 0x" + "f" * 4000 + "\n: 2\n",
}

#: Mappings in which a key is kept unconstructed, or kept as a list.
_KEY_HOSTS = ("x: !!str {{=: z, ? {key} : 1}}\n",
              "x: !!int {{=: 7, ? {key} : 1}}\n",
              "x: !!omap [{{? {key} : 1}}]\n",
              "x: !!pairs [{{? {key} : 1}}]\n")
#: Keys that reach the mapping ``A`` around them, or merge inside themselves.
_REACHING_KEYS = ("[*A]", "{k: *A}", "!!pairs [{k: *A}]",
                  "!!omap [{k: *A}]", "!!set {? *A}", "{<<: {c: 1}, c: 2}")
#: What the walk still has to read after such a key, and whether it repeats.
_AFTER_THE_KEY = (("b: 1\nb: 2\n", True),
                  ("<<: {m: 1, m: 2}\n", True),
                  ("<<: {a: 1}\n<<: {b: 2}\n", True),
                  ("later: {<<: {a: 1}, a: 2}\n", False))


@pytest.mark.parametrize("text", LEGAL_MERGES.values(), ids=LEGAL_MERGES)
def test_legal_merges_read_as_plain_yaml_reads_them(text):
    """The duplicate check used to run on nodes ``flatten_mapping`` had
    rewritten, so a merge of an anchor with its own override was refused,
    and a node merging itself raised ``RecursionError``."""
    expected = repr(yaml.safe_load(text))
    assert repr(rc._safe_mapping(text, "variants")) == expected
    assert repr(_load_yaml(text)) == expected


@pytest.mark.parametrize("text", REPEATED_KEYS.values(), ids=REPEATED_KEYS)
def test_keys_written_twice_are_refused_by_both_loaders(text):
    with pytest.raises(ValueError, match="duplicate mapping key"):
        rc._safe_mapping(text, "variants")
    with pytest.raises(ConfigError) as info:
        _load_yaml(text)
    assert info.value.code == "duplicate_key"


def _reading(load, text):
    try:
        return repr(load(text))
    except (yaml.YAMLError, LookupError, AttributeError, TypeError, ValueError,
            ArithmeticError, RecursionError):
        return "refused"


@pytest.mark.parametrize("head", ["&A\n", "&A\n<<: {m: 1}\n"],
                         ids=["no-merge", "merge-first"])
@pytest.mark.parametrize("tail,repeats", _AFTER_THE_KEY,
                         ids=["key", "in-a-source", "merge-key", "override"])
def test_a_document_reads_as_plain_yaml_reads_it(head, tail, repeats):
    """Constructing such a key before the read flattened ``A`` under the
    loop over it, which skipped an entry and rewrote others."""
    for host, key in itertools.product(_KEY_HOSTS, _REACHING_KEYS):
        text = head + host.format(key=key) + tail
        plain = _reading(yaml.safe_load, text)
        if repeats and plain != "refused":
            with pytest.raises(ConfigError) as info:
                _load_yaml(text)
            assert info.value.code == "duplicate_key", text
        else:
            assert _reading(_load_yaml, text) == plain, text


def test_a_file_safe_load_refuses_stays_refused():
    """The walk read the alias key first, and the constructor then used
    that value where it would have failed."""
    text = "{? << : &n1 !!str {? = : z}, ? *n1 : a}\n"
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(text)
    with pytest.raises(yaml.YAMLError):
        _load_yaml(text)


def _alias_lattice(depth):
    """Lists of nine aliases of the list before, ``9 ** depth`` when printed."""
    lines = ["a0: &a0 [" + ", ".join("x" * 9) + "]\n"]
    for i in range(1, depth + 1):
        lines.append(f"a{i}: &a{i} [" + ", ".join([f"*a{i - 1}"] * 9) + "]\n")
    return "".join(lines)


def test_a_list_of_aliases_as_a_key_is_not_printed():
    """``str`` of this key has ``9 ** 8`` entries."""
    start = time.perf_counter()
    with pytest.raises(yaml.YAMLError, match="unhashable"):
        _load_yaml(_alias_lattice(8) + "? *a8\n: 1\n")
    pairs = _load_yaml(_alias_lattice(8) + "p: !!pairs\n- ? *a8\n  : 1\n")["p"]
    assert time.perf_counter() - start < 5.0
    assert len(pairs) == 1 and pairs[0][1] == 1


def test_a_key_past_the_digit_limit_of_str_reads():
    """``int`` reads these bases without the limit ``str`` has."""
    text = "? 0x" + "f" * 4000 + "\n: 1\n"
    assert _load_yaml(text) == yaml.safe_load(text)
    assert rc._safe_mapping(text, "config") == yaml.safe_load(text)


def test_a_key_that_holds_itself_reads_but_is_no_config():
    text = "a: !!pairs [? &x [*x] : 1]\n"
    assert repr(_load_yaml(text)) == repr(yaml.safe_load(text))
    with pytest.raises(ValueError, match="holds itself"):
        rc._safe_mapping(text, "config")


def _pairs_of_a_shared_list(entries, keys):
    return (f"b: &b [{', '.join('v' * entries)}]\np: !!pairs\n"
            + "".join(f"- {{? *b : {i}}}\n" for i in range(keys)))


def _scalar_keys_reaching_a_list(entries, keys):
    return (f"b: &b [{', '.join('v' * entries)}]\n"
            + "".join(f"k{i}: !!str {{=: {i}, q: *b}}\n" for i in range(keys)))


def _a_shared_binary_key(size, aliases):
    return (f"- ? &k !!binary {'A' * size}\n  : 1\n"
            + "".join("- {? *k : 1}\n" for _ in range(aliases)))


@pytest.mark.parametrize("text", [
    _pairs_of_a_shared_list(2000, 1000),
    _scalar_keys_reaching_a_list(2000, 1000),
    _a_shared_binary_key(400_000, 1000),
], ids=["list-key", "scalar-read-mapping-key", "binary-key"])
def test_each_key_is_read_once(text):
    """The comparison uses the keys the read gave, at about its cost."""
    start = time.perf_counter()
    plain = yaml.safe_load(text)
    plain_time = time.perf_counter() - start
    start = time.perf_counter()
    assert _load_yaml(text) == plain
    assert time.perf_counter() - start < 10 * plain_time + 1.0


def test_a_key_expanding_a_merge_chain_is_refused_at_once():
    """Reading the key apart from the document expanded the chain, 11 s and
    1 GB, where YAML refuses the key without resuming it."""
    links = "".join(f"a{i}: &a{i} {{<<: [*a{i - 1}, *a{i - 1}]}}\n"
                    for i in range(1, 25))
    start = time.perf_counter()
    with pytest.raises(yaml.YAMLError, match="unhashable"):
        _load_yaml("a0: &a0 {x: 1, y: 2}\n" + links + "? [*a24]\n: 1\n")
    assert time.perf_counter() - start < 2.0


def _nested_alias_keys(length, depth, inner):
    return (f"? &k {'x' * length}\n: " + "{? *k : " * depth + inner
            + "}" * depth + "\n")


def test_a_path_is_built_only_for_a_repeat():
    """A path at every key took 1.7 GB on this 23 KB file."""
    text = _nested_alias_keys(20_000, 400, "1")
    tracemalloc.start()
    try:
        _load_yaml(text)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 100 * 2**20


def test_a_key_is_cut_short_in_a_path():
    """The line of one refusal was 8 MB on the same file."""
    with pytest.raises(ConfigError) as info:
        _load_yaml(_nested_alias_keys(20_000, 400, "{a: 1, a: 2}"))
    assert info.value.path.endswith(".a")
    assert len(str(info.value)) < 40_000


def test_a_repeat_at_the_end_of_a_long_chain_gets_its_path():
    links = "".join(f"- &a{i} {{n: *a{i - 1}}}\n" for i in range(1, 3000))
    text = f"s:\n- &a0 {{n: 1, n: 2}}\n{links}"
    with pytest.raises(ConfigError) as info:
        _load_yaml(text)
    assert info.value.path == "s[0].n"


def test_the_first_repeat_in_the_file_is_reported():
    """The constructor reads a mapping before the mappings in it."""
    with pytest.raises(ConfigError) as info:
        _load_yaml("a: {b: 1, b: 2}\nc: 3\nc: 4\n")
    assert info.value.path == "a.b"


_ALIAS_KEY_TWICE_IN_TWO_MAPPINGS = """\
key: &k a
defs:
  %s: &one {*k : 1, *k : 2}
  %s: &two {*k : 3, *k : 4}
use: {<<: [%s]}
"""


@pytest.mark.parametrize("text,path", [
    (_ALIAS_KEY_TWICE_IN_TWO_MAPPINGS % ("first", "second", "*one, *two"),
     "defs.first.a"),
    (_ALIAS_KEY_TWICE_IN_TWO_MAPPINGS % ("first", "second", "*two, *one"),
     "defs.first.a"),
    (_ALIAS_KEY_TWICE_IN_TWO_MAPPINGS % ("z", "b", "*one, *two"), "defs.z.a"),
    (_ALIAS_KEY_TWICE_IN_TWO_MAPPINGS % ("z", "b", "*two, *one"), "defs.z.a"),
    (_ALIAS_KEY_TWICE_IN_TWO_MAPPINGS % ("a_longer_name", "b", "*two, *one"),
     "defs.a_longer_name.a"),
    ("key: &k a\ndefs: [&one {*k : 1, *k : 2}, &two {*k : 3, *k : 4}]\n"
     "use: {<<: [*one, *two]}\n", "defs[0].a"),
    ("key: &k a\no:\n  n: {*k : 1, *k : 2}\n  *k : 3\n  *k : 4\n", "o.a"),
], ids=["in-file-order", "reversed", "names-against-file-order",
        "reversed-names-against-file-order", "later-column-first", "one-line",
        "nested"])
def test_a_tie_on_an_alias_key_names_the_earlier_mapping(text, path):
    """Aliases of one anchor all sit at its place.  The report does not
    follow the order the reader met the mappings in."""
    with pytest.raises(ConfigError) as info:
        _load_yaml(text)
    assert info.value.path == path


@pytest.mark.parametrize("text,path", [
    ("- {b: 1, b: 2}\n", "[0].b"),
    ("a:\n- {b: 1, b: 2}\n", "a[0].b"),
    ("a: {<<: {b: 1, b: 2}}\n", "a.b"),
    ("a: {<<: [{c: 1}, {b: 1, b: 2}]}\n", "a.b"),
    ("? !!str {=: k}\n: {b: 1, b: 2}\n", "<key>.b"),
    ("a: 1\n? !!str {=: a}\n: 2\n", "<key>"),
    ("a: !!pairs\n- ? {b: 1, b: 2}\n  : 1\n", "a[0].b"),
    ("!!merge a: {x: 1}\n!!merge b: {y: 2}\n", "<<"),
    ("? !!merge [x]\n: {a: 1}\n? !!merge [y]\n: {b: 2}\n", "<<"),
    ("a: &x {b: 1, b: 2}\nc: *x\n", "a.b"),
    ("? !!str {=: k, z: &x {c: 1, c: 2}}\n: 1\ny: *x\n", "z.c"),
], ids=["in-a-list", "in-a-list-under-a-key", "merged-in", "merged-in-by-a-list",
        "under-a-non-scalar-key", "a-non-scalar-key", "inside-a-non-scalar-key",
        "tagged-merge-keys", "non-scalar-merge-keys", "first-route",
        "first-route-through-a-mapping-no-one-reads"])
def test_the_path_of_a_repeat(text, path):
    with pytest.raises(ConfigError) as info:
        _load_yaml(text)
    assert info.value.path == path


def test_keys_are_read_with_the_constructors_of_safe_loader(monkeypatch):
    """A second constructor did not see what is added to ``SafeLoader``,
    which is the loader the document is read with."""
    constructors = dict(yaml.SafeLoader.yaml_constructors)
    constructors["!env"] = lambda loader, node: loader.construct_scalar(node).upper()
    constructors["tag:yaml.org,2002:float"] = (
        lambda loader, node: loader.construct_scalar(node))
    monkeypatch.setattr(yaml.SafeLoader, "yaml_constructors", constructors)
    with pytest.raises(ConfigError):
        _load_yaml("!env a: 1\n!env a: 2\n")
    assert _load_yaml("1.0: a\n1.00: b\n") == {"1.0": "a", "1.00": "b"}


def test_a_repeated_key_is_reported_with_its_path():
    with pytest.raises(ValueError, match=r"duplicate mapping key 'case\.n\.x'"):
        rc._safe_mapping("case: {<<: {n: {x: 1, x: 2}}}\n", "variants")


def test_the_message_names_the_file_the_key_and_both_lines():
    with pytest.raises(ValueError, match=r"^variants: duplicate mapping key "
                                         r"'a': .* on lines 1 and 2"):
        rc._safe_mapping("a: 1\na: 2\n", "variants")


def test_a_repeated_merge_key_is_reported_at_its_own_path():
    with pytest.raises(ConfigError) as info:
        _load_yaml("y: {<<: {a: 1}, <<: {b: 2}}\n")
    assert info.value.path == "y.<<"


_TWO_SOURCES = "a: &a {x: 1, y: 1}\nb: &b {x: 2, z: 2}\n"


def test_two_merge_keys_read_as_one_list_with_the_later_first():
    """Of two ``<<`` keys the later wins a shared key; in one list the
    earlier does.  The list the refusal and the guide advise must keep the
    value the file had."""
    twice = yaml.safe_load(_TWO_SOURCES + "c: {<<: *a, <<: *b}\n")
    advised = _load_yaml(_TWO_SOURCES + "c: {<<: [*b, *a]}\n")
    assert advised["c"] == twice["c"] == {"x": 2, "y": 1, "z": 2}
    assert _load_yaml(_TWO_SOURCES + "c: {<<: [*a, *b]}\n")["c"]["x"] == 1


def test_the_refusal_of_two_merge_keys_says_what_yaml_does():
    with pytest.raises(ConfigError) as info:
        _load_yaml(_TWO_SOURCES + "c: {<<: *a, <<: *b}\n")
    detail = info.value.detail
    assert "only the last" not in detail
    assert "merges both and the later wins" in detail
    assert "in one << list the earlier wins" in detail


def test_the_refusal_of_a_plain_key_says_what_yaml_does():
    with pytest.raises(ConfigError) as info:
        _load_yaml("a: 1\na: 2\n")
    assert "keep only the last" in info.value.detail
    assert "merges" not in info.value.detail


@pytest.mark.parametrize("text", ["a: &a {b: *a}\n", "a: &a {b: [*a]}\n",
                                  "a: &a [*a]\n", "a: &a !!pairs [b: *a]\n"],
                         ids=["mapping", "through-a-list", "list", "pairs"])
def test_a_value_holding_itself_is_refused(text):
    """``_known_files`` and the JSON report walk the config, and a value
    holding itself reached that walk as a ``RecursionError``."""
    with pytest.raises(ValueError, match="^config: invalid YAML"):
        rc._safe_mapping(text, "config")


def test_a_node_used_twice_does_not_hold_itself():
    text = "a: &a {b: 1}\nc: [*a, *a]\nd: {e: *a, f: [*a]}\n"
    assert rc._safe_mapping(text, "config") == yaml.safe_load(text)


class _CountedList(list):
    """A list that counts how often it is read."""

    reads = 0

    def __iter__(self):
        _CountedList.reads += 1
        return super().__iter__()


def test_a_node_used_twice_is_walked_once():
    """Each level of aliases doubles the paths through the file."""
    node = _CountedList([1])
    for _ in range(16):
        node = _CountedList([node, node])
    _CountedList.reads = 0
    assert not rc._holds_itself({"a": node})
    assert _CountedList.reads == 17


def test_a_long_alias_chain_stays_readable():
    links = "".join(f"- &a{i} {{n: *a{i - 1}}}\n" for i in range(1, 700))
    text = f"s:\n- &a0 {{n: 1}}\n{links}z: *a699\n"
    assert rc._safe_mapping(text, "config")["z"]["n"]["n"]


def test_a_long_alias_chain_reached_from_its_far_end_stays_readable():
    """A merged key comes first, so the walk starts at the last link."""
    links = "".join(f"- &a{i} {{n: *a{i - 1}}}\n" for i in range(1, 700))
    text = f"<<: {{z: 1}}\ns:\n- &a0 {{n: 1}}\n{links}z: *a699\n"
    assert rc._safe_mapping(text, "config")["z"]["n"]["n"]


def test_a_merge_chain_reads_as_far_as_plain_yaml_reads_it():
    """The read goes down the chain from its far end, a frame per link."""
    links = "".join(f"  l{i}: &l{i} {{<<: *l{i - 1}, k{i}: 1}}\n"
                    for i in range(1, 700))
    text = f"defs:\n  l0: &l0 {{x: 1}}\n{links}use: *l699\n"
    assert _load_yaml(text) == yaml.safe_load(text)
    assert rc._safe_mapping(text, "config") == yaml.safe_load(text)


def test_an_empty_file_is_not_a_mapping():
    with pytest.raises(ValueError, match="^variants must be a mapping"):
        rc._safe_mapping("", "variants")
    assert _load_yaml("") is None


@pytest.mark.parametrize("text", [
    "a: \x01\n",                     # the reader refuses it on creation
    "? [a, b]\n: 1\n",                # unhashable key
    "a: !!bool abc\n",                # KeyError in the constructor
    "a: !!int ''\n",                  # IndexError
    "a: !!float ''\n",                # IndexError
    "a: !!timestamp abc\n",           # AttributeError
    "a: 2001-13-45\n",                # ValueError
    "a: !!int abc\n",                 # ValueError
    "a: !!float abc\n",               # ValueError
    "a: !!timestamp {=: x}\n",        # TypeError
    "a: 1" + ":0" * 180 + ".0\n",     # OverflowError
    "a: " + "[" * 600 + "1" + "]" * 600 + "\n",  # RecursionError
], ids=["control-character", "complex-key", "bool-tag", "int-tag", "float-tag",
        "timestamp-tag", "bad-date", "int-word", "float-word",
        "timestamp-mapping", "sexagesimal-overflow", "too-deep"])
def test_unreadable_yaml_is_one_error(text):
    """The file is named, and the CLI's ``except`` clauses catch it."""
    with pytest.raises(ValueError, match="^config: invalid YAML"):
        rc._safe_mapping(text, "config")


@pytest.mark.parametrize("text", [
    "? [a, b]\n: 1\n", "? !!set {a: null}\n: 1\n",
    "? {<<: {c: 1}, c: 2}\n: 1\n", "a: !!bool abc\n", "a: !!int ''\n",
    "a: !!timestamp abc\n", "a: 2001-13-45\n", "a: !!int abc\n",
    "a: !!float abc\n", "a: !!timestamp {=: x}\n", "a: 1" + ":0" * 180 + ".0\n",
    "a: " + "[" * 600 + "1" + "]" * 600 + "\n",
], ids=["complex-key", "set-key", "merge-override-in-a-key", "bool-tag",
        "int-tag", "timestamp-tag", "bad-date", "int-word", "float-word",
        "timestamp-mapping", "sexagesimal-overflow", "too-deep"])
def test_the_workflow_loader_gives_a_yaml_error(text):
    with pytest.raises(yaml.YAMLError):
        _load_yaml(text)


@pytest.mark.parametrize("text", [
    "base: {a: 1}\nuse:\n  <<: base\n  b: 2\n", "use:\n  <<: [base]\n",
    "use:\n  <<: 1\n", "use:\n  <<: [[a, b]]\n", "x: !!map 1\n",
    "x: !!map [1, 2]\n", "x: !!set 1\n",
], ids=["a-word", "a-list-of-a-word", "a-number", "a-nested-list",
        "map-tag-on-a-number", "map-tag-on-a-list", "set-tag-on-a-number"])
def test_a_merge_of_no_mapping_is_refused_as_plain_yaml_refuses_it(text):
    """``<<: base`` with the ``*`` left out.  PyYAML's message keeps the line."""
    with pytest.raises(yaml.YAMLError) as plain:
        yaml.safe_load(text)
    with pytest.raises(yaml.YAMLError) as read:
        _load_yaml(text)
    assert str(read.value) == str(plain.value)


def test_a_file_of_another_encoding_gives_a_yaml_error(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_bytes(b"a: caf\xe9\n")
    with path.open(encoding="utf-8") as f, pytest.raises(yaml.YAMLError):
        _load_yaml(f)


@pytest.mark.parametrize("name,text,label", [
    ("config", "sweep: {n_samples: 1, n_samples: 2}\n", "config"),
    ("variants", "case: {gl_chunk_size: 1, gl_chunk_size: 2}\n", "variants"),
])
def test_compare_resources_refuses_a_repeated_key_in_either_file(
        tmp_path, name, text, label):
    paths = {"config": tmp_path / "c.yaml", "variants": tmp_path / "v.yaml"}
    paths["config"].write_text("sweep: {n_samples: 1}\n")
    paths["variants"].write_text("case: {gl_chunk_size: 1}\n")
    paths[name].write_text(text)
    with pytest.raises(ValueError, match=f"^{label}: duplicate mapping key"):
        rc.compare_resources(paths["config"], paths["variants"],
                             tmp_path / "r.json")


@pytest.mark.parametrize("text", ["system: \x01\n", "a: 1" + ":0" * 180 + ".0\n"],
                         ids=["control-character", "sexagesimal-overflow"])
def test_compare_resources_cli_reports_unreadable_yaml_without_a_traceback(
        tmp_path, capsys, text):
    from sft_wick.workflow.cli import main

    config = tmp_path / "c.yaml"
    config.write_text(text)
    variants = tmp_path / "v.yaml"
    variants.write_text("case: {gl_chunk_size: 1}\n")
    status = main(["compare-resources", str(config), "--variants", str(variants),
                   "--json", str(tmp_path / "r.json")])
    err = capsys.readouterr().err
    assert status == 1
    assert "config: invalid YAML" in err and "Traceback" not in err


def test_variants_with_alias_overrides_pass_validation():
    raw = rc._safe_mapping(
        "small: &s {<<: {gl_chunk_size: 2}, gl_chunk_size: 4}\n"
        "large: {<<: *s, gl_chunk_size: 8}\n", "variants")
    rc._validate_variants(raw)
    assert raw == {"small": {"gl_chunk_size": 4}, "large": {"gl_chunk_size": 8}}


def test_known_files_walks_a_long_chain(tmp_path):
    node = {"coupling_module": "m.py"}
    for _ in range(5000):
        node = {"a": [node]}
    assert rc._known_files(node, tmp_path) == {(tmp_path / "m.py").absolute()}


def test_known_files_walks_a_shared_node_once(tmp_path):
    """Aliases used twice per level: 2**60 paths to one node."""
    node = {"module": "m.py"}
    for _ in range(60):
        node = {"a": node, "b": [node]}
    assert rc._known_files(node, tmp_path) == {(tmp_path / "m.py").absolute()}
