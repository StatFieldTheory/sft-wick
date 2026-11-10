r"""The package side of a comparison, in its own process.

``python -m validation.worker [--lowering validation.lower] [--cache-size
4] < jobs.jsonl > results.jsonl`` reads one JSON job per line and writes
one JSON result per line, so the process that compares never loads
sft_wick.  ``--lowering`` names the module that lowers a ModelSpec onto the
release under test (``validation.lower``, or ``validation.lower_v042`` for
0.4.2).

A job holds ``model`` and ``observable`` (the JSON of
:mod:`validation.modelspec`), ``max_weight`` and ``method``, and optionally
``settings``, ``orders``, ``c_source``, ``flags``, ``n_grid_t``, ``id``,
``timeout`` (a positive number of seconds, default :data:`DEFAULT_TIMEOUT`,
or ``null`` for none) and ``expect_package`` (the directory that must hold
the loaded ``sft_wick`` package, a ``src`` or ``site-packages`` directory;
the environment variable ``VALIDATION_EXPECT_PACKAGE`` sets a default).

A result holds ``id``, the package's location, version and the source of
that version, and ``status``: ``ok``, ``error`` or ``timeout``.  A result
with status ``ok`` holds ``values`` as ``[tag, value, error]`` rows, the C
route the package reports (``c_route``), the method, settings and
expansion flags it ran, the time to
build the expansion and propagators (``build_seconds``, absent when
``cached``) and the time to evaluate (``seconds``).  The quadrature tables
of ``c_source='table'`` are built lazily by the first evaluation of a key,
so their time and their failures fall under ``evaluate``.  Any other
result carries the ``stage`` that failed: ``parse`` (the line is not JSON),
``job`` (the job is malformed, or the package is from the wrong tree),
``lower``, ``expand``, ``propagators`` or ``evaluate``.  An error carries
its ``origin`` and the traceback: ``validation`` for the lowering,
``package`` for sft_wick, ``callback`` for an oracle callable (the closed
form of C, a Campbell cumulant) that the package called and that raised,
``other`` otherwise.  A timeout and a parse failure carry neither.  Every result counts the warnings raised
(``warnings_total``) and lists the most common kinds.

While the worker runs, file descriptor 1 (which ``sys.stdout`` writes to
and child processes inherit) points at stderr, so stdout carries only
results, whatever the package, a callable or a child process prints.  At most ``--cache-size`` lowered expansions are
kept, the least recently used dropped first.
"""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
import re
import sys
import time
import traceback
import warnings
from pathlib import Path
from types import ModuleType

from .stages import STAGE_ATTR, JobTimeout, deadline

__all__ = ["DEFAULT_TIMEOUT", "DEFAULT_CACHE_SIZE", "JobTimeout", "deadline",
           "package_identity", "run_job", "main"]

DEFAULT_TIMEOUT = 600.0
DEFAULT_CACHE_SIZE = 4
MAX_WARNING_KINDS = 20
_WARNING_KINDS_KEPT = 1000
VALIDATION_DIR = Path(__file__).resolve().parent


def _project_version(pyproject: Path, name: str) -> str | None:
    """``version`` of the ``[project]`` table, if it is the project
    ``name``."""
    text = pyproject.read_text()
    table = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    if table is None:
        return None
    body = table.group(1)
    found_name = re.search(r'^name\s*=\s*"([^"]+)"', body, re.M)
    version = re.search(r'^version\s*=\s*"([^"]+)"', body, re.M)
    if found_name is None or version is None:
        return None
    if found_name.group(1).replace("_", "-") != name.replace("_", "-"):
        return None
    return version.group(1)


def package_identity(module: ModuleType) -> dict:
    """Where ``module`` was loaded from and its version, read from the
    package's own source: ``__version__``, else the ``pyproject.toml`` of a
    source tree (``src/<name>``), else the one ``.dist-info`` beside an
    unpacked wheel (several leave the version unknown), else the installed
    metadata (stale for an editable install)."""
    path = Path(module.__file__).resolve().parent
    name = module.__name__.split(".")[0]
    version = getattr(module, "__version__", None)
    source = "__version__"
    if version is None:
        pyproject = path.parent.parent / "pyproject.toml"
        dist = sorted(path.parent.glob(f"{name}-*.dist-info/METADATA"))
        if pyproject.is_file() and path.parent.name == "src":
            version, source = _project_version(pyproject, name), str(pyproject)
        elif len(dist) == 1:
            found = re.search(r"^Version:\s*(\S+)", dist[0].read_text(), re.M)
            version, source = (found.group(1) if found else None), str(dist[0])
        elif dist:
            source = "several dist-info: " + ", ".join(
                d.parent.name for d in dist)
        else:
            from importlib import metadata
            try:
                version = metadata.version(name.replace("_", "-"))
            except metadata.PackageNotFoundError:
                version = None
            source = "importlib.metadata"
    return {"package": str(path), "version": version,
            "version_source": source}


def _classify(filename: str, package_dir: Path) -> str | None:
    path = Path(filename).resolve()
    if VALIDATION_DIR in path.parents:
        return None if path.name in ("worker.py", "stages.py") else "validation"
    if package_dir == path.parent or package_dir in path.parents:
        return "package"
    return None


def _innermost(kinds: list) -> str | None:
    """The origin of a list of frame kinds, outermost first: the innermost
    kind, ``callback`` when a validation frame was called from a package
    frame (the package called an oracle callable)."""
    origin = None
    for kind in kinds:
        if kind == "validation" and origin in ("package", "callback"):
            origin = "callback"
        elif kind is not None:
            origin = kind
    return origin


def _origin(exc: BaseException, package_dir: Path) -> str:
    """``validation``, ``package`` or ``callback``: the innermost frame in
    either tree of the deepest exception in the chain that has one (a frame
    in numpy or scipy belongs to its caller), with ``callback`` for a
    validation frame below a package frame.  A joblib worker's traceback,
    carried as text in the cause, is read from its file names."""
    found = "other"
    seen: set = set()
    chain = []
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        chain.append(exc)
        exc = exc.__cause__ or exc.__context__
    for link in chain:
        kinds = []
        tb = link.__traceback__
        while tb is not None:
            kinds.append(_classify(tb.tb_frame.f_code.co_filename,
                                   package_dir))
            tb = tb.tb_next
        remote = getattr(link, "tb", None)
        if isinstance(remote, str):
            kinds += [_classify(f, package_dir)
                      for f in re.findall(r'File "([^"]+)"', remote)]
        found = _innermost(kinds) or found
    return found


class _WarningCounter:
    """Counts warnings by kind, keeping at most ``_WARNING_KINDS_KEPT``."""

    def __init__(self) -> None:
        self.total = 0
        self.kinds: collections.Counter = collections.Counter()

    def __call__(self, message, category, *args, **kwargs) -> None:
        self.total += 1
        kind = f"{category.__name__}: {str(message)[:200]}"
        if kind in self.kinds or len(self.kinds) < _WARNING_KINDS_KEPT:
            self.kinds[kind] += 1
        else:
            self.kinds["(other kinds)"] += 1

    def report(self) -> dict:
        return {"warnings_total": self.total, "warnings": [
            [text, n] for text, n in self.kinds.most_common(MAX_WARNING_KINDS)]}


def _check_identity(identity: dict, expected: str | None) -> None:
    if not expected:
        return
    folder = Path(expected).resolve()
    package = Path(identity["package"])
    if package.parent != folder:
        raise RuntimeError(f"sft_wick was loaded from {package}, not from "
                           f"{folder}")


def _cached(cache: dict, key: str):
    low = cache.pop(key, None)
    if low is not None:
        cache[key] = low
    return low


def run_job(job: dict, cache: dict, lowering: ModuleType,
            cache_size: int = DEFAULT_CACHE_SIZE) -> dict:
    """One job, as the module docstring describes.  ``cache`` keeps the
    lowered expansion of each (model, observable, weight, C source, flags,
    grid), at most ``cache_size`` of them."""
    from .modelspec import model_from_dict, observable_from_dict

    identity = package_identity(lowering.sw)
    out = {"id": job.get("id") if isinstance(job, dict) else None,
           **identity}
    key = None
    t_start = time.perf_counter()
    counter = _WarningCounter()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = counter
        try:
            if not isinstance(job, dict):
                raise TypeError("a job is a JSON object")
            _check_identity(identity, job.get(
                "expect_package", os.environ.get("VALIDATION_EXPECT_PACKAGE")))
            with deadline(job.get("timeout", DEFAULT_TIMEOUT)) as enforced:
                out["timeout_enforced"] = enforced
                key = json.dumps([job["model"], job["observable"],
                                  job["max_weight"],
                                  job.get("c_source", "oracle"),
                                  job.get("flags"), job.get("n_grid_t", 41)],
                                 sort_keys=True)
                low = _cached(cache, key)
                out["cached"] = low is not None
                if low is None:
                    t0 = time.perf_counter()
                    low = lowering.Lowered(
                        model_from_dict(job["model"]),
                        observable_from_dict(job["observable"]),
                        job["max_weight"], job.get("flags"),
                        job.get("c_source", "oracle"),
                        job.get("n_grid_t", 41))
                    out["build_seconds"] = time.perf_counter() - t0
                    cache[key] = low
                    while len(cache) > max(1, cache_size):
                        cache.pop(next(iter(cache)))
                t0 = time.perf_counter()
                values = low.evaluate(job["method"], orders=job.get("orders"),
                                      **job.get("settings", {}))
                out["seconds"] = time.perf_counter() - t0
            out.update(ok=True, status="ok", method=job["method"],
                       settings=job.get("settings", {}),
                       c_route=str(low.props.c_source),
                       flags={k: bool(v) for k, v in low.flags.items()},
                       values=[[list(t), v, e] for t, (v, e) in values.items()])
        except JobTimeout as exc:
            cache.pop(key, None)
            out.update(ok=False, status="timeout", error=str(exc),
                       stage=getattr(exc, STAGE_ATTR, "job"))
        except Exception as exc:  # noqa: BLE001  (a refusal is a result)
            out.update(ok=False, status="error",
                       error=f"{type(exc).__name__}: {exc}",
                       stage=getattr(exc, STAGE_ATTR, "job"),
                       origin=_origin(exc, Path(identity["package"])),
                       traceback=traceback.format_exc(limit=-12))
    out.update(counter.report())
    out["wall_seconds"] = time.perf_counter() - t_start
    return out


def _parse(raw: bytes):
    """A job from one line of stdin, or the reason it is not one."""
    try:
        return json.loads(raw.decode("utf-8")), None
    except (ValueError, RecursionError) as exc:
        return None, f"bad job line: {type(exc).__name__}: {str(exc)[:200]}"


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m validation.worker")
    parser.add_argument("--lowering", default="validation.lower")
    parser.add_argument("--cache-size", type=int, default=DEFAULT_CACHE_SIZE)
    args = parser.parse_args(argv)
    os.environ["SFT_WICK_QUIET_CACHE"] = "1"
    sys.stdout.flush()
    results = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    cache: dict = {}
    lowering = importlib.import_module(args.lowering)
    for raw in sys.stdin.buffer:
        if not raw.strip():
            continue
        job, problem = _parse(raw)
        if problem is None:
            result = run_job(job, cache, lowering, args.cache_size)
        else:
            result = {"id": None, **package_identity(lowering.sw),
                      "ok": False, "status": "error", "stage": "parse",
                      "error": problem}
        results.write(json.dumps(result) + "\n")
        results.flush()
    results.close()


if __name__ == "__main__":
    main()
