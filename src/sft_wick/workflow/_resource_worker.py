"""Private fresh-process worker for explicit resource comparisons."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import platform
import sys
import time
import traceback
from pathlib import Path


def _rss():
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (
            int(value)
            if sys.platform == "darwin"
            else int(value * 1024) if sys.platform.startswith("linux") else None
        )
    except (ImportError, AttributeError):
        return None


def run(request):
    import sft_wick
    import yaml

    from .config import _parse_workflow, run_workflow
    from .resource_compare import encode_results
    from .resource_compare import _hash, _source_files
    from .resources import resource_controls

    source_root = Path(__file__).resolve().parents[2]
    if str(source_root) != request["source_root"]:
        raise ValueError("worker loaded a different source tree")
    expected = request["input_fingerprints"]

    def check_inputs():
        for path, target in request["input_links"].items():
            if str(Path(path).resolve()) != target:
                raise ValueError(f"known worker input link changed: {path}")
        for path, fingerprint in expected.items():
            if not Path(path).is_file() or _hash(path) != fingerprint:
                raise ValueError(f"known worker input changed: {path}")
        sources = {str(p.absolute()) for p in _source_files(source_root)}
        if sources != set(request["source_manifest"]):
            raise ValueError("worker source inventory changed")

    check_inputs()
    tracing = request["mode"] == "memory"
    if tracing:
        import tracemalloc

        tracemalloc.start()
    start = time.perf_counter()
    raw = copy.deepcopy(yaml.safe_load(request["config_text"]))
    raw["output"] = []
    raw["expand"]["cache_path"] = None
    raw["propagators"]["cache_path"] = None
    raw["sweep"].update(request["resource_overrides"])
    cfg = _parse_workflow(raw, base_dir=Path(request["base_dir"]))
    config_seconds = time.perf_counter() - start
    start = time.perf_counter()
    sweep, totals = run_workflow(cfg, progress=False)
    workflow_seconds = time.perf_counter() - start
    peak = None
    if tracing:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    rss = _rss()
    check_inputs()
    # Metadata and complete-result serialization occur after timing, tracing
    # and RSS sampling, so they cannot inflate the reported workflow cost.
    versions = {}
    for package in ("sft-wick", "numpy", "scipy", "pandas", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(
        ok=True,
        config_load_seconds=config_seconds,
        workflow_seconds=workflow_seconds,
        traced_peak_bytes=peak,
        rss_lifetime_peak_bytes=rss,
        result=encode_results(sweep, totals),
        resource_controls=resource_controls(cfg),
        versions=versions,
        package_path=str(Path(sft_wick.__file__).resolve()),
        source_fingerprints=request["source_manifest"],
        python=sys.version,
        platform=platform.platform(),
    )


def main():
    destination = Path(sys.argv[2])
    try:
        result = run(json.loads(Path(sys.argv[1]).read_text()))
        code = 0
    except Exception as exc:
        result = dict(
            ok=False,
            failure=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        code = 1
    destination.write_text(json.dumps(result, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
