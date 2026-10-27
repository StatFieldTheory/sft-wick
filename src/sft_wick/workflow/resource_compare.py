"""Explicit, serial, fresh-process comparison of workflow resource choices.

Only six resource fields may vary. Calculations are repeated (including a
separate memory pass) and their complete ordered results are checked before
any resource ratios are reported. This is a diagnostic experiment, not an
automatic resource policy, a sandbox, or a total-memory budget.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import statistics
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pandas import DataFrame

    from .result import SweepResult

_CONTROLS = {
    "reuse_integrands",
    "gl_chunk_size",
    "qmc_coupling_chunk_size",
    "propagator_cache_bytes",
    "contraction_backend",
    "contraction_max_bytes",
}
_THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_files(source_root):
    paths, visited = set(), set()

    def unreadable(exc):
        # This missing sentinel makes an incomplete inventory fail closed.
        paths.add(Path(exc.filename or source_root) / ".unreadable-source-inventory.py")

    for directory, subdirs, names in os.walk(
        source_root, followlinks=True, onerror=unreadable
    ):
        resolved = Path(directory).resolve()
        if resolved in visited:
            subdirs[:] = []
            continue
        visited.add(resolved)
        paths.update(Path(directory) / name for name in names if name.endswith(".py"))
    return paths


def _safe_mapping(text, label):
    import yaml

    # PyYAML otherwise silently replaces repeated keys, including variant
    # names. Reject them before a potentially expensive experiment starts.
    class UniqueLoader(yaml.SafeLoader):
        pass

    def construct(loader, node, deep=False):
        explicit = set()
        for key_node, value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = loader.construct_object(key_node, deep=deep)
            if key in explicit:
                raise ValueError(f"{label}: duplicate mapping key {key!r}")
            explicit.add(key)
        loader.flatten_mapping(node)
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct
    )
    try:
        raw = yaml.load(text, Loader=UniqueLoader)
    except (yaml.YAMLError, TypeError) as exc:
        raise ValueError(f"{label}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping")
    return raw


def _validate_variants(raw):
    if not raw:
        raise ValueError("variants must contain at least one named resource variant")
    for name, values in raw.items():
        if not isinstance(name, str) or not name.strip() or name == "baseline":
            raise ValueError(
                "variant names must be nonempty strings other than 'baseline'"
            )
        if not isinstance(values, dict) or not values:
            raise ValueError(f"variant {name!r} must contain resource fields")
        if set(values) - _CONTROLS:
            raise ValueError(f"variant {name!r}: only {sorted(_CONTROLS)} may change")
        for key, value in values.items():
            if key == "reuse_integrands":
                if not isinstance(value, bool):
                    raise ValueError(f"{name}.{key} must be a boolean")
            elif key == "contraction_backend":
                if type(value) is not str or value not in (
                    "ordered",
                    "einsum",
                    "factorized",
                ):
                    raise ValueError(
                        f"{name}.{key} must be ordered, einsum or factorized"
                    )
            elif key == "contraction_max_bytes":
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{name}.{key} must be a positive integer")
            elif key == "propagator_cache_bytes":
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError(
                        f"{name}.{key} must be null or a nonnegative integer"
                    )
            elif value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name}.{key} must be null or a positive integer")


def _validate_raw(raw):
    for section in ("system", "expand", "propagators", "sweep"):
        if not isinstance(raw.get(section), dict):
            raise ValueError(f"config.{section} must be a mapping")
    for section in ("expand", "propagators", "sweep"):
        jobs = raw[section].get("n_jobs", 1)
        if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs != 1:
            raise ValueError("compare-resources initially requires all three n_jobs=1")
    sw = raw["sweep"]
    method = sw.get("method", "qmc_vectorized")
    if not isinstance(method, str):
        raise ValueError("sweep.method must be a string")
    if method.startswith("qmc") and sw.get("seed", 42) is None:
        raise ValueError("QMC comparisons require a fixed sweep.seed")
    _validate_lookup_contract(sw)
    original = {k: sw[k] for k in _CONTROLS if k in sw}
    if original:
        _validate_variants({"original": original})


def _validate_lookup_contract(sw):
    from .._propagator_lookup import validate_cache_options

    try:
        validate_cache_options(
            sw.get("propagator_cache_bytes"), sw.get("propagator_cache_contract")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


def _known_files(raw, base_dir):
    files = set()

    def visit(item):
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(key, str) and (
                    key.endswith("_module") or key == "module" or key == "coupling_path"
                ):
                    if value is None and key != "coupling_path":
                        continue
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(
                            f"declared {key} must be a nonempty path string for comparison"
                        )
                    files.add((base_dir / value).absolute())
                visit(value)
        elif isinstance(item, list):
            for value in item:
                visit(value)

    visit(raw)
    return files


def input_fingerprints(
    config: str | Path, variants: str | Path, known_files: set[Path], source_root: Path
) -> dict[str, str | None]:
    """Hash declared local inputs and all Python files in the active source tree.

    Unknown imported dependencies and external state are outside this scope.
    File absence is recorded so deletion is an input change, not a lost check.
    """
    paths = {Path(config), Path(variants), *known_files}
    paths.update(_source_files(source_root))
    paths.update(path.resolve() for path in list(paths))
    fingerprints = {}
    for path in sorted(paths):
        try:
            value = _hash(path) if path.is_file() else None
        except OSError:
            value = None
        fingerprints[str(path.absolute())] = value
    return fingerprints


def _input_links(config, variants, known_files, source_root):
    paths = {Path(config), Path(variants), *known_files, *_source_files(source_root)}
    return {
        str(path.absolute()): str(path.resolve())
        for path in sorted(paths)
        if path.absolute() != path.resolve()
    }


def encode_results(sweep: SweepResult, totals: DataFrame) -> dict:
    """Encode all ordered sweep rows, axis metadata and total columns/records.

    Unsupported values and nonfinite numbers raise rather than being truncated
    or silently coerced. Encoding happens outside the measured workflow.
    """

    def normal(value):
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biuf" or (
                value.dtype.kind == "f" and value.dtype.itemsize > 8
            ):
                raise TypeError(f"unsupported result array dtype {value.dtype}")
            return normal(value.tolist())
        if isinstance(value, np.generic):
            if value.dtype.kind not in "biuf" or (
                value.dtype.kind == "f" and value.dtype.itemsize > 8
            ):
                raise TypeError(f"unsupported result scalar dtype {value.dtype}")
            # Some NumPy longdouble scalars have an 8-byte dtype but item()
            # still returns np.generic. Convert approved native kinds directly
            # so that normalization cannot recurse indefinitely.
            if value.dtype.kind == "b":
                return bool(value)
            if value.dtype.kind in "iu":
                return int(value)
            return normal(float(value))
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("result contains a nonfinite number")
            return value
        if isinstance(value, (list, tuple)):
            return [normal(v) for v in value]
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise TypeError("result column names must be strings")
            return {k: normal(v) for k, v in value.items()}
        raise TypeError(f"unsupported result value type {type(value).__name__}")

    payload = normal(
        dict(
            rows=sweep.rows,
            position_keys=sweep.position_keys,
            external_time_keys=sweep.external_time_keys,
            component_keys=sweep.component_keys,
            totals_columns=list(totals.columns),
            totals_records=totals.to_dict(orient="records"),
        )
    )
    _validate_payload(payload)
    return payload


def _validate_payload(payload):
    if not isinstance(payload, dict) or set(payload) != {
        "rows",
        "position_keys",
        "external_time_keys",
        "component_keys",
        "totals_columns",
        "totals_records",
    }:
        raise ValueError("invalid complete result payload")
    if not payload["rows"] or not payload["totals_records"]:
        raise ValueError("empty result rows cannot establish equivalence")
    for rows, numeric in (
        (payload["rows"], ("value", "error")),
        (payload["totals_records"], ("value",)),
    ):
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("result rows must be mappings")
            for key in numeric:
                value = row.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"result {key} must be a finite real number")


def compare_results(
    reference: dict | None,
    candidate: dict | None,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> dict:
    """Check ordered metadata exactly and every value/error against the baseline.

    Tolerances are explicit; ``abs(candidate-reference) <= atol +
    rtol*abs(reference)``. ``exact`` also distinguishes signed floating zeros.
    """
    _validate_tolerances(rtol, atol)
    result: dict[str, Any] = dict(
        exact=False,
        within_tolerance=False,
        metadata_equal=False,
        max_abs_difference=0.0,
        difference_overflow=False,
        reason=None,
    )
    try:
        _validate_payload(reference)
        _validate_payload(candidate)
    except (TypeError, ValueError) as exc:
        result["reason"] = str(exc)
        return result
    assert isinstance(reference, dict) and isinstance(candidate, dict)
    for key in (
        "position_keys",
        "external_time_keys",
        "component_keys",
        "totals_columns",
    ):
        if reference[key] != candidate[key]:
            result["reason"] = f"metadata differs: {key}"
            return result
    exact = within = True
    for key, numeric in (("rows", {"value", "error"}), ("totals_records", {"value"})):
        if len(reference[key]) != len(candidate[key]):
            result["reason"] = f"row count differs: {key}"
            return result
        for a, b in zip(reference[key], candidate[key]):
            if list(a) != list(b) or {
                k: v for k, v in a.items() if k not in numeric
            } != {k: v for k, v in b.items() if k not in numeric}:
                result["reason"] = f"ordered row metadata differs: {key}"
                return result
            for field in numeric:
                av, bv = a[field], b[field]
                equal = av == bv and (
                    av != 0 or math.copysign(1, av) == math.copysign(1, bv)
                )
                exact &= equal
                diff = abs(bv - av)
                if not math.isfinite(diff):
                    result["difference_overflow"] = True
                else:
                    result["max_abs_difference"] = max(
                        result["max_abs_difference"], diff
                    )
                threshold = atol + rtol * abs(av)
                if not math.isfinite(diff) or not math.isfinite(threshold):
                    # Overflow must not turn a failed comparison into inf <= inf.
                    from decimal import Decimal, localcontext

                    with localcontext() as ctx:
                        ctx.prec = 1100
                        da, db = Decimal(av), Decimal(bv)
                        accepted = abs(db - da) <= Decimal(atol) + Decimal(rtol) * abs(
                            da
                        )
                else:
                    accepted = diff <= threshold
                within &= av == bv or accepted
    result.update(exact=exact, within_tolerance=within, metadata_equal=True)
    if result["difference_overflow"]:
        result["max_abs_difference"] = None
    if not within:
        result["reason"] = "numerical difference exceeds the declared tolerance"
    return result


def _validate_tolerances(rtol, atol):
    for key, value in (("rtol", rtol), ("atol", atol)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{key} must be finite and nonnegative")


def _run_child(request, timeout, source_root):
    with tempfile.TemporaryDirectory(prefix="sft-wick-resource-worker-") as tmp:
        request_path, result_path = (
            Path(tmp) / "request.json",
            Path(tmp) / "result.json",
        )
        request_path.write_text(json.dumps(request, allow_nan=False))
        env = dict(os.environ)
        env["PYTHONPATH"] = str(source_root) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONTRACEMALLOC"] = "0"
        env.setdefault("PYTHONHASHSEED", "0")
        env["PYTHONPYCACHEPREFIX"] = str(Path(tmp) / "empty-bytecode-cache")
        command = [
            sys.executable,
            "-B",
            "-m",
            "sft_wick.workflow._resource_worker",
            str(request_path),
            str(result_path),
        ]
        try:
            completed = subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return dict(
                ok=False,
                failure="timeout",
                timeout_seconds=timeout,
                stderr=str(exc.stderr or "")[-4000:],
            )
        if not result_path.exists():
            return dict(
                ok=False,
                failure="worker did not return a result",
                returncode=completed.returncode,
                stderr=completed.stderr[-4000:],
            )
        try:
            result = json.loads(result_path.read_text())
        except (ValueError, OSError) as exc:
            return dict(
                ok=False,
                failure=f"invalid worker response: {exc}",
                returncode=completed.returncode,
            )
        result["returncode"] = completed.returncode
        if completed.returncode != 0:
            result["ok"] = False
        result["stderr"] = completed.stderr[-4000:]
        return result


def _directory_identity(path):
    info = path.stat()
    return info.st_dev, info.st_ino


def _write_report(report, result, *, force, parent_identity):
    """Replace a directory entry atomically without following late file aliases.

    The directory descriptor pins the already validated parent, including if
    callbacks retarget a directory symlink. Existing ordinary report permission
    bits are retained; new reports use private 0600 permissions.
    """
    directory = os.open(report.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    temporary = f".{report.name}.{uuid.uuid4().hex}.tmp"
    created = False
    try:
        info = os.fstat(directory)
        if (info.st_dev, info.st_ino) != parent_identity:
            raise ValueError("report directory changed during the experiment")
        mode = 0o600
        if force:
            try:
                existing = os.stat(report.name, dir_fd=directory, follow_symlinks=False)
                if stat.S_ISREG(existing.st_mode):
                    mode = stat.S_IMODE(existing.st_mode) & 0o777
            except FileNotFoundError:
                pass
        fd = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory
        )
        created = True
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fchmod(stream.fileno(), mode)
        if force:
            os.replace(
                temporary, report.name, src_dir_fd=directory, dst_dir_fd=directory
            )
            created = False
        else:
            os.link(
                temporary,
                report.name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
    finally:
        if created:
            os.unlink(temporary, dir_fd=directory)
        os.close(directory)


def compare_resources(
    config: str | Path,
    variants: str | Path,
    report: str | Path,
    *,
    repeats: int = 2,
    timeout: float = 300.0,
    rtol: float = 0.0,
    atol: float = 0.0,
    force: bool = False,
) -> dict:
    """Run explicit resource variants and write a complete JSON experiment report.

    ``variants`` is a YAML mapping of names to the six permitted sweep
    controls. Baseline retains the original controls. All three parallelism
    layers must be serial, QMC must have a fixed seed, and output/disk caches
    are disabled consistently. Every timing/memory pass uses a fresh process.

    User callbacks run repeatedly; callers are responsible for determinism,
    chunk compatibility and external effects. The report hashes known local
    inputs, not every unknown dependency. Returns the report mapping; failed
    execution/equivalence/input gates set ``ok=False`` and suppress ratios.
    Existing reports are preserved unless ``force=True``.
    """
    _validate_tolerances(rtol, atol)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
        raise ValueError("repeats must be an integer >= 2 for repeat stability")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be finite and positive")
    config, variants = (Path(p).absolute() for p in (config, variants))
    report = Path(report).resolve()
    config_text, variants_text = config.read_bytes().decode(
        "utf-8"
    ), variants.read_bytes().decode("utf-8")
    raw, choices = _safe_mapping(config_text, "config"), _safe_mapping(
        variants_text, "variants"
    )
    _validate_raw(raw)
    _validate_variants(choices)
    # The fixed-model promise is held constant, never introduced by a variant.
    # Reject invalid effective budgets before importing any configured module.
    for choice in choices.values():
        _validate_lookup_contract({**raw["sweep"], **choice})
    source_root = Path(__file__).resolve().parents[2]
    known = _known_files(raw, config.parent)
    before = input_fingerprints(config, variants, known, source_root)
    links_before = _input_links(config, variants, known, source_root)
    source_manifest = {
        str(p.absolute()): before[str(p.absolute())] for p in _source_files(source_root)
    }
    protected = {Path(p) for p in before}
    collision = (
        report in protected or report == source_root or source_root in report.parents
    )
    if report.exists():
        collision |= any(p.exists() and report.samefile(p) for p in protected)
    if collision:
        raise ValueError(
            "report must not replace a config, variant, declared input or source file"
        )
    if (
        before[str(config)] != hashlib.sha256(config_text.encode("utf-8")).hexdigest()
        or before[str(variants)]
        != hashlib.sha256(variants_text.encode("utf-8")).hexdigest()
    ):
        raise ValueError("config or variants changed while freezing inputs")
    if report.exists() and not force:
        raise FileExistsError(f"report exists: {report}; use --force to replace it")
    if any(value is None for value in before.values()):
        raise ValueError("a declared local input file is missing")
    if os.open not in os.supports_dir_fd or os.link not in os.supports_dir_fd:
        raise ValueError("this platform lacks directory-relative atomic report writes")
    report.parent.mkdir(parents=True, exist_ok=True)
    parent_identity = _directory_identity(report.parent)
    cases = {"baseline": {}, **choices}
    result = dict(
        schema_version=1,
        ok=False,
        config=str(config),
        variants_file=str(variants),
        config_text=config_text,
        variants_text=variants_text,
        tolerances=dict(rtol=rtol, atol=atol),
        repeats=repeats,
        measurement_scope="timing: config loading separately from complete build/expand/propagators/sweep/totals; traced peak: config loading plus workflow; no configured outputs or disk caches",
        limits="known local inputs hashed; unknown dependencies and external callback side effects are not frozen; not a sandbox or total-memory budget",
        input_fingerprints_before=before,
        input_links_before=links_before,
        execution_order=[],
        runs=[],
        cases={},
        environment=dict(
            python=sys.version,
            executable=sys.executable,
            platform=platform.platform(),
            source_root=str(source_root),
            thread_environment={k: os.environ.get(k) for k in _THREAD_ENV},
            parent_hash_seed=os.environ.get("PYTHONHASHSEED"),
            worker_hash_seed=os.environ.get("PYTHONHASHSEED", "0"),
            parent_tracemalloc=os.environ.get("PYTHONTRACEMALLOC"),
            worker_startup_tracemalloc="0",
            worker_bytecode_disabled=True,
            worker_bytecode_cache="fresh empty per-worker prefix; existing bytecode is not read",
        ),
        failures=[],
    )
    for repeat in range(repeats):
        names = list(cases) if repeat % 2 == 0 else list(reversed(cases))
        for mode in ("timing", "memory"):
            for name in names:
                current = input_fingerprints(config, variants, known, source_root)
                if (
                    current != before
                    or _input_links(config, variants, known, source_root)
                    != links_before
                ):
                    result["failures"].append("known inputs changed before a run")
                    break
                request = dict(
                    config_text=config_text,
                    base_dir=str(config.parent),
                    resource_overrides=copy.deepcopy(cases[name]),
                    mode=mode,
                    input_fingerprints=before,
                    source_root=str(source_root),
                    source_manifest=source_manifest,
                    input_links=links_before,
                )
                result["execution_order"].append(
                    dict(case=name, repeat=repeat, mode=mode)
                )
                run = _run_child(request, timeout, source_root)
                run.update(case=name, repeat=repeat, mode=mode)
                result["runs"].append(run)
                if not run.get("ok"):
                    result["failures"].append(
                        f"{name} {mode} repeat {repeat}: {run.get('failure', 'worker failed')}"
                    )
                if (
                    input_fingerprints(config, variants, known, source_root) != before
                    or _input_links(config, variants, known, source_root)
                    != links_before
                ):
                    result["failures"].append("known inputs changed during a run")
                    break
            if result["failures"]:
                break
        if result["failures"]:
            break
    after = input_fingerprints(config, variants, known, source_root)
    result["input_fingerprints_after"] = after
    links_after = _input_links(config, variants, known, source_root)
    result["input_links_after"] = links_after
    result["known_inputs_unchanged"] = after == before and links_after == links_before
    result["execution_count"] = len(result["runs"])
    _summarize(result, cases, rtol, atol)
    _write_report(report, result, force=force, parent_identity=parent_identity)
    return result


def _summarize(result, cases, rtol, atol):
    baseline = next(
        (
            r.get("result")
            for r in result["runs"]
            if r["case"] == "baseline" and r.get("ok")
        ),
        None,
    )
    all_good = not result["failures"] and result["known_inputs_unchanged"]
    for name, controls in cases.items():
        runs = [r for r in result["runs"] if r["case"] == name and r.get("ok")]
        payload = runs[0].get("result") if runs else None
        stable = bool(payload) and len(runs) == 2 * result["repeats"]
        checks = []
        if payload is not None:
            checks = [compare_results(payload, r.get("result")) for r in runs]
            stable = stable and all(c["exact"] for c in checks)
        equivalent = compare_results(baseline, payload, rtol=rtol, atol=atol)
        valid = stable and (
            equivalent["exact"] if rtol == atol == 0 else equivalent["within_tolerance"]
        )
        all_good &= valid
        timing = [r["workflow_seconds"] for r in runs if r["mode"] == "timing"]
        memory = [r["traced_peak_bytes"] for r in runs if r["mode"] == "memory"]
        rss = [r.get("rss_lifetime_peak_bytes") for r in runs if r["mode"] == "memory"]
        loads = [r["config_load_seconds"] for r in runs if r["mode"] == "timing"]
        combined = [
            r["config_load_seconds"] + r["workflow_seconds"]
            for r in runs
            if r["mode"] == "timing"
        ]
        result["cases"][name] = dict(
            resource_overrides=controls,
            stable=stable,
            equivalence_gate_passed=valid,
            stability_checks=checks,
            equivalence=equivalent,
            timing_seconds=timing,
            median_workflow_seconds=statistics.median(timing) if timing else None,
            median_config_load_seconds=statistics.median(loads) if loads else None,
            median_load_plus_workflow_seconds=(
                statistics.median(combined) if combined else None
            ),
            traced_peak_bytes=memory,
            median_traced_peak_bytes=statistics.median(memory) if memory else None,
            rss_lifetime_peak_bytes=rss,
        )
    result["ok"] = bool(all_good)
    if not result["ok"] and not result["failures"]:
        result["failures"].append(
            "complete result stability or equivalence gate failed"
        )
    # Do not publish attractive ratios for an experiment with any failed gate.
    if result["ok"]:
        base = result["cases"]["baseline"]
        for case in result["cases"].values():
            case["workflow_time_ratio"] = (
                case["median_workflow_seconds"] / base["median_workflow_seconds"]
            )
            peak = base["median_traced_peak_bytes"]
            case["traced_peak_ratio"] = (
                case["median_traced_peak_bytes"] / peak if peak else None
            )
