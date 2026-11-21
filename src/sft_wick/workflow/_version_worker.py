"""Standalone worker shared by the baseline and candidate installations.

Run this file directly. The modules that the workflow imports must come
from the requested installation, including releases predating this file.
"""

from __future__ import annotations

import json
import math
import re
import runpy
import sys
from numbers import Integral, Real
from pathlib import Path


def _coordinate(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and math.isfinite(value):
        return float(value)
    if isinstance(value, (tuple, list)):
        return [_coordinate(part) for part in value]
    raise ValueError(f"invalid result coordinate {value!r}")


def _key(value):
    """Hashable coordinates with numeric equality across releases."""
    if isinstance(value, list):
        return tuple(_key(part) for part in value)
    if isinstance(value, bool):
        return (bool, value)
    return value


def _row_key(coordinates):
    return tuple(sorted((name, _key(value)) for name, value in coordinates.items()))


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite real number")
    return float(value)


def _normalize_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("workflow returned no comparable rows")
    result = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"coordinates", "value", "error"}:
            raise ValueError("result rows require coordinates, value and error")
        coords = row["coordinates"]
        if not isinstance(coords, dict) or not coords or not all(
            isinstance(name, str) and name for name in coords
        ):
            raise ValueError("coordinates must be a nonempty mapping with string keys")
        coords = {name: _coordinate(value) for name, value in coords.items()}
        key = _row_key(coords)
        if key in seen:
            raise ValueError(f"duplicate physical row {coords!r}")
        seen.add(key)
        error = None if row["error"] is None else _number(row["error"], "error")
        if error is not None and error < 0:
            raise ValueError("error must be nonnegative")
        result.append(dict(coordinates=coords, value=_number(row["value"], "value"), error=error))
    return result


def _workflow_rows(sweep, totals):
    """Use the workflow's physical totals, independent of diagram indices.

    Sum the per-diagram reported errors for each total, without assuming
    that the integration errors in different diagrams are independent.
    """
    fractions = totals.to_dict(orient="records")
    if not fractions:
        return []
    names = [name for name in fractions[0] if name != "value"]
    errors = {}
    for row in sweep.rows:
        key = _row_key({name: _coordinate(row[name]) for name in names})
        raw_error = row.get("error")
        error = None if raw_error is None else _number(raw_error, "diagram error")
        if error is not None and error < 0:
            raise ValueError("diagram error must be nonnegative")
        errors.setdefault(key, []).append(error)
    result = []
    for row in fractions:
        coordinates = {name: _coordinate(row[name]) for name in names}
        parts = errors.pop(_row_key(coordinates))
        error = None if None in parts else math.fsum(parts)
        result.append(dict(coordinates=coordinates, value=row["value"], error=error))
    if errors:
        raise ValueError("the workflow totals omit sweep rows")
    return result


def _version(package, root):
    declared = getattr(package, "__version__", None)
    if declared is not None:
        return str(declared)
    metadata = sorted(root.glob("sft_wick-*.dist-info/METADATA"))
    if len(metadata) == 1:
        found = re.search(r"^Version:\s*(\S+)", metadata[0].read_text(), re.M)
        return found[1] if found else None
    pyproject = (root.parent if root.name == "src" else root) / "pyproject.toml"
    if not pyproject.is_file():
        return None
    project = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", pyproject.read_text(), re.M | re.S)
    if project and re.search(r'^name\s*=\s*["\']sft[-_]wick["\']', project[1], re.M):
        version = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', project[1], re.M)
        if version:
            return version[1]
    return None


def _run(request):
    root = Path(request["root"])
    sys.path.insert(0, str(root))
    import sft_wick

    expected = (root / "sft_wick" / "__init__.py").resolve()
    if Path(sft_wick.__file__).resolve() != expected:
        raise ValueError("worker imported sft-wick from a different installation")
    source = {"version": _version(sft_wick, root)}
    if "driver" in request:
        driver = Path(request["driver"])
        sys.path.insert(1, str(driver.parent))
        returned = runpy.run_path(str(driver))["run"]()
        method = request["method"]
    else:
        from sft_wick.workflow.config import _parse_workflow, run_workflow

        raw = request["config"]
        cfg = _parse_workflow(raw, base_dir=Path(request["base_dir"]))
        settings = getattr(cfg, "sweep", None)
        method = getattr(settings, "method", raw["sweep"].get("method"))
        seed = getattr(settings, "seed", raw["sweep"].get("seed", 42))
        if isinstance(method, str) and method.startswith("qmc") and seed is None:
            raise ValueError("QMC comparisons require a fixed sweep.seed")
        sweep, totals = run_workflow(cfg, progress=False)
        returned = _workflow_rows(sweep, totals)
    if not isinstance(method, str) or not method:
        raise ValueError("workflow did not identify its integration method")
    return dict(ok=True, source=source, method=method, rows=_normalize_rows(returned))


def main():
    destination = Path(sys.argv[2])
    try:
        result = _run(json.loads(Path(sys.argv[1]).read_text()))
    except Exception as exc:
        result = dict(ok=False, failure=f"{type(exc).__name__}: {exc}")
    destination.write_text(json.dumps(result, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
