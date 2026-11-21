"""Support release reviews by comparing the physical results of two versions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from ._version_worker import _coordinate, _key, _normalize_rows, _row_key
from .resource_compare import (
    _THREAD_ENV,
    _directory_identity,
    _hash,
    _known_files,
    _refuse_deep_config,
    _safe_mapping,
    _source_files,
    _write_report,
)

_ROUND_OFF = 4.5e-16
_WORKER = Path(__file__).with_name("_version_worker.py")
_MOVEMENTS = {"moved", "within uncertainty"}


def _file(parent, value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty file path")
    path = (parent / value).resolve()
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def _read_mapping(path, label, parsed_hashes):
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    if parsed_hashes.setdefault(str(path), digest) != digest:
        raise ValueError(f"input changed while loading corpus: {path}")
    return _safe_mapping(contents.decode("utf-8"), label)


def _entries(path, label, key, parsed_hashes):
    raw = _read_mapping(path, label, parsed_hashes)
    if set(raw) != {key} or not isinstance(raw[key], list):
        raise ValueError(f"{label} requires a {key!r} list")
    if not all(isinstance(entry, dict) for entry in raw[key]):
        raise ValueError(f"{label} entries must be mappings")
    return raw[key]


def _load_cases(path, parsed_hashes):
    cases, inputs = {}, {path}
    entries = _entries(path, "corpus", "cases", parsed_hashes)
    if not entries:
        raise ValueError("corpus must contain at least one case")
    for entry in entries:
        name = entry.get("id")
        if not isinstance(name, str) or not name.strip() or name in cases:
            raise ValueError("each case needs a unique, nonempty string id")
        declared_inputs = entry.get("inputs", [])
        if not isinstance(declared_inputs, list):
            raise ValueError(f"{name}: inputs must be a list")
        inputs.update(_file(path.parent, item, name) for item in declared_inputs)
        keys = set(entry) - {"id", "inputs"}
        if keys == {"config"}:
            config = _file(path.parent, entry["config"], f"{name} config")
            raw = _read_mapping(config, f"{name} config", parsed_hashes)
            _refuse_deep_config(raw)
            inputs.add(config)
            for file in _known_files(raw, config.parent):
                inputs.update({file, file.resolve()})
            raw["output"] = []
            for section in ("expand", "propagators", "sweep"):
                mapping = raw.get(section)
                if not isinstance(mapping, dict):
                    raise ValueError(f"{name}: config requires a {section!r} mapping")
                mapping["n_jobs"] = 1
                if section != "sweep":
                    mapping["cache_path"] = None
            cases[name] = {"config": raw, "base_dir": str(config.parent)}
        elif keys == {"driver", "method"}:
            driver = _file(path.parent, entry["driver"], f"{name} driver")
            method = entry["method"]
            if not isinstance(method, str) or not method.strip():
                raise ValueError(f"{name}: driver method must be a nonempty string")
            cases[name] = {"driver": str(driver), "method": method}
            inputs.add(driver)
        else:
            raise ValueError(f"{name}: specify config, or driver and method")
    return cases, inputs


def _load_declarations(path, cases, parsed_hashes):
    if path is None:
        return []
    declarations = _entries(path, "declarations", "changes", parsed_hashes)
    for change in declarations:
        if set(change) != {"case", "where", "kind", "reason"}:
            raise ValueError("a change requires case, where, kind and reason")
        if not isinstance(change["case"], str) or change["case"] not in cases:
            raise ValueError(f"declaration names an unknown case: {change['case']!r}")
        if change["kind"] not in ("moved", "redrawn"):
            raise ValueError("declared change kind must be moved or redrawn")
        if not isinstance(change["reason"], str) or not change["reason"].strip():
            raise ValueError("declared change needs a nonempty reason")
        where = change["where"]
        if not isinstance(where, dict) or not all(isinstance(k, str) and k for k in where):
            raise ValueError("declared where must be a mapping with string coordinate names")
        change["where"] = {name: _coordinate(value) for name, value in where.items()}
    return declarations


def _project_root(path):
    options = [root for root in (path, path / "src") if (root / "sft_wick/__init__.py").is_file()]
    if len(options) != 1:
        raise ValueError(f"expected one sft_wick package at {path} or its src directory")
    return options[0]


def _extract_wheel(path, destination):
    try:
        with ZipFile(path) as archive:
            seen = set()
            for member in archive.infolist():
                name = PurePosixPath(member.filename)
                if not name.parts or name.is_absolute() or ".." in name.parts or "\\" in member.filename:
                    raise ValueError(f"invalid wheel entry: {member.filename!r}")
                if member.is_dir():
                    continue
                if name in seen:
                    raise ValueError(f"duplicate wheel entry: {member.filename!r}")
                seen.add(name)
                if name.parts[0] != "sft_wick" and not (
                    name.parts[0].startswith("sft_wick-")
                    and name.parts[0].endswith(".dist-info")
                    and name.name == "METADATA"
                ):
                    continue
                target = destination.joinpath(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as copy:
                    shutil.copyfileobj(source, copy)
    except BadZipFile as exc:
        raise ValueError(f"invalid wheel {path}: {exc}") from exc


def _tree_fingerprint(root):
    files = _source_files(root / "sft_wick")
    metadata = (root.parent if root.name == "src" else root) / "pyproject.toml"
    if metadata.is_file():
        files.add(metadata)
    files.update(root.glob("sft_wick-*.dist-info/METADATA"))
    manifest = []
    for file in sorted(files):
        relative = os.path.relpath(file, root)
        manifest.append((relative, _hash(file)))
    return hashlib.sha256(json.dumps(manifest).encode()).hexdigest()


def _install_files(path):
    if not path.is_dir():
        return {path}
    root = _project_root(path)
    project = root.parent if root.name == "src" else root
    return {path, root / "sft_wick", project / "pyproject.toml",
            *root.glob("sft_wick-*.dist-info/METADATA")}


def _prepare_install(path, destination):
    wheel = path.is_file() and path.suffix == ".whl"
    if not wheel and not path.is_dir():
        raise ValueError(f"expected a local wheel, installed package or source tree: {path}")
    if wheel:
        digest = _hash(path)
        _extract_wheel(path, destination)
        root = _project_root(destination)
        kind = "wheel"
    else:
        root = _project_root(path)
        digest = _tree_fingerprint(root)
        kind = "source" if root.name == "src" else "installed"
    return root, dict(artifact=str(path), kind=kind, sha256=digest, version=None)


def _run_case(root, case, timeout, directory):
    request = directory / "request.json"
    response = directory / "response.json"
    response.unlink(missing_ok=True)
    try:
        payload = json.dumps({"root": str(root), **case}, allow_nan=False)
    except (TypeError, ValueError) as exc:
        return dict(ok=False, failure=f"could not encode worker request: {exc}")
    request.write_text(payload, encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    env.update(dict.fromkeys(_THREAD_ENV, "1"), SFT_WICK_PROGRESS="0", PYTHONHASHSEED="0")
    command = [
        sys.executable, "-s", "-B", "-X", f"pycache_prefix={directory / 'empty-cache'}",
        str(_WORKER), str(request), str(response),
    ]
    try:
        proc = subprocess.run(command, cwd=directory, env=env, capture_output=True,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return dict(ok=False, failure=f"worker timed out after {timeout:g} s")
    if not response.is_file():
        return dict(ok=False, failure=f"worker produced no result (exit {proc.returncode})",
                    stderr=proc.stderr[-4000:].decode(errors="replace"))
    try:
        returned = json.loads(response.read_text(encoding="utf-8"))
        if proc.returncode != 0 or returned.get("ok") is not True:
            return dict(ok=False, failure=returned["failure"])
        returned["rows"] = _normalize_rows(returned["rows"])
    except (AttributeError, KeyError, TypeError, ValueError, OSError) as exc:
        return dict(ok=False, failure=f"invalid worker result: {exc}")
    return returned


def _classify(baseline, candidate, qmc):
    """Compare floating-point values before considering a movement declaration."""
    low, high = baseline["value"], candidate["value"]
    if struct.pack(">d", low) == struct.pack(">d", high):
        return "bitwise"
    distance = abs(high - low)
    scale = max(abs(low), abs(high))
    if distance == 0 or distance / scale <= _ROUND_OFF:
        return "round-off"
    old_error, new_error = baseline["error"], candidate["error"]
    if qmc and old_error is not None and new_error is not None:
        if math.isfinite(distance):
            inside = distance <= old_error + new_error
        else:
            inside = abs(high / scale - low / scale) <= old_error / scale + new_error / scale
        if inside:
            return "within uncertainty"
    return "moved"


def _matching_declarations(declarations, name, coordinates):
    return [
        index for index, item in enumerate(declarations)
        if item["case"] == name and all(
            key in coordinates and _key(coordinates[key]) == _key(value)
            for key, value in item["where"].items()
        )
    ]


def _compare_case(name, old, new, declarations, used):
    result = {"id": name, "baseline": {k: v for k, v in old.items() if k != "rows"},
              "candidate": {k: v for k, v in new.items() if k != "rows"}}
    if not old["ok"] or not new["ok"]:
        label = "refused on both sides" if not old["ok"] and not new["ok"] else "refused on one side"
        return dict(result, baseline=old, candidate=new, classification=label, rows=[], ok=False)

    left = {_row_key(row["coordinates"]): row for row in old["rows"]}
    right = {_row_key(row["coordinates"]): row for row in new["rows"]}
    qmc = all(side["method"].startswith("qmc") for side in (old, new))
    rows = []
    for key in dict.fromkeys((*left, *right)):
        baseline, candidate = left.get(key), right.get(key)
        coordinates = (baseline or candidate)["coordinates"]
        row = dict(coordinates=coordinates, baseline=baseline, candidate=candidate, declared=False)
        if baseline is None or candidate is None:
            row.update(classification="row only on one side", ok=False)
        else:
            classification = _classify(baseline, candidate, qmc)
            row.update(classification=classification, ok=classification not in _MOVEMENTS)
            if classification in _MOVEMENTS:
                matches = _matching_declarations(declarations, name, coordinates)
                if len(matches) == 1:
                    index = matches[0]
                    change = declarations[index]
                    if change["kind"] == "redrawn" and not qmc:
                        row["failure"] = "redrawn requires QMC on both sides"
                    else:
                        used.add(index)
                        row.update(ok=True, declared=True, reason=change["reason"],
                                   declaration=index)
                        if change["kind"] == "redrawn":
                            row["classification"] = "redrawn"
                elif matches:
                    row["failure"] = "several declarations match this row"
        rows.append(row)
    return dict(result, classification="compared", rows=rows,
                ok=all(row["ok"] for row in rows))


def _input_hashes(paths):
    return {str(path): (_hash(path) if path.is_file() else None) for path in sorted(paths)}


def compare_versions(
    corpus: str | Path,
    baseline: str | Path,
    candidate: str | Path,
    report: str | Path,
    *,
    declared: str | Path | None = None,
    timeout: float = 300.0,
    force: bool = False,
) -> dict:
    """Compare two local package versions on a corpus in fresh processes.

    Write the per-row values, decisions and input identities to a JSON
    report. An undeclared change beyond round-off, a missing physical row,
    or a failed worker leaves the returned report's ``ok`` false.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    corpus, baseline, candidate, report = (Path(p).resolve() for p in (corpus, baseline, candidate, report))
    declared = None if declared is None else Path(declared).resolve()
    parsed_hashes = {}
    cases, inputs = _load_cases(corpus, parsed_hashes)
    declarations = _load_declarations(declared, cases, parsed_hashes)
    if declared is not None:
        inputs.add(declared)
    protected = inputs | _install_files(baseline) | _install_files(candidate)
    package_directories = {
        entry.resolve() for entry in protected if entry.name == "sft_wick" and entry.is_dir()
    }
    if report in {entry.resolve() for entry in protected} or package_directories.intersection(report.parents):
        raise ValueError("report must not replace an input or package file")
    if report.exists() and not force:
        raise FileExistsError(f"report exists: {report}; use --force to replace it")
    report.parent.mkdir(parents=True, exist_ok=True)
    directory_identity = _directory_identity(report.parent)
    hashes = _input_hashes(inputs)
    hashes.update(parsed_hashes)

    used, comparisons, failures = set(), [], []
    with tempfile.TemporaryDirectory(prefix="sft-wick-version-") as temp:
        directory = Path(temp)
        artifacts = [
            _prepare_install(path, directory / side)
            for path, side in ((baseline, "baseline"), (candidate, "candidate"))
        ]
        for name, case in cases.items():
            old, new = (_run_case(root, case, timeout, directory) for root, _ in artifacts)
            for response, (_, identity) in zip((old, new), artifacts):
                if response["ok"] and identity["version"] is None:
                    identity["version"] = response["source"]["version"]
            comparisons.append(_compare_case(name, old, new, declarations, used))
        for (root, identity), install in zip(artifacts, (baseline, candidate)):
            current = _hash(install) if identity["kind"] == "wheel" else _tree_fingerprint(root)
            if current != identity["sha256"]:
                failures.append(f"installation changed during comparison: {install}")
    after = _input_hashes(inputs)
    if hashes != after:
        failures.append("corpus, declarations or a known input changed during comparison")
    for result in comparisons:
        if not result["ok"]:
            failures.append(f"{result['id']}: {result['classification']} has unresolved differences")
    counts = Counter(row["classification"] for result in comparisons for row in result["rows"])
    data = dict(schema=1, ok=not failures, baseline=artifacts[0][1], candidate=artifacts[1][1],
                cases=comparisons, counts=dict(counts), failures=failures,
                declarations=declarations,
                unused_declarations=[i for i in range(len(declarations)) if i not in used],
                input_sha256=hashes, input_sha256_after=after,
                python=sys.version, roundoff_rtol=_ROUND_OFF)
    _write_report(report, data, force=force, parent_identity=directory_identity)
    return data
