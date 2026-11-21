r"""The oracle rows of ``tools/mutant_fuzz.py`` and their cache.

The oracle row of a drawn case (:func:`validation.fuzz.oracle_of`) is
computed once and cached in a directory outside the project, keyed by the
case's model, observable and weight and by the digest of the oracle's code
(:func:`oracle_key`, :func:`oracle_sha`), so that the unmutated tree and
every mutant are judged against one evaluation.  :func:`oracle_rows`
computes the missing rows in at most ``jobs`` pool processes of
:mod:`validation.fuzz`, each limited to ``max_rss`` bytes.  A row lost to
the deadline, the memory limit or an error in the oracle is cached with
those limits and computed again under larger ones (:func:`reusable`); a row
lost with its pool process is not cached.  This module imports no sft_wick
code.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import inspect
import json
import pathlib
import sys
import time
from collections.abc import Iterable, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation import fuzz  # noqa: E402
from validation.draw import Case  # noqa: E402
from validation.modelspec import model_to_dict, observable_to_dict  # noqa: E402
from validation.stages import JobMemory, JobTimeout  # noqa: E402

#: Files whose bytes decide the oracle's value, and the function of
#: validation/fuzz.py that picks its tags and scale.
ORACLE_FILES = ("validation/hierarchy.py", "validation/modelspec.py")
ORACLE_DIRS = ("examples/reference",)
ORACLE_FUNCTIONS = (fuzz.oracle_of,)


def _digest_files(paths: Iterable[pathlib.Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(path.relative_to(ROOT).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def oracle_sha() -> str:
    """sha256 of :data:`ORACLE_FILES`, the Python files of
    :data:`ORACLE_DIRS` and the source of :data:`ORACLE_FUNCTIONS`."""
    files = [ROOT / f for f in ORACLE_FILES]
    for d in ORACLE_DIRS:
        files += sorted((ROOT / d).glob("*.py"))
    h = hashlib.sha256(_digest_files(files).encode())
    for function in ORACLE_FUNCTIONS:
        h.update(inspect.getsource(function).encode())
    return h.hexdigest()


ORACLE_SHA = oracle_sha()


def oracle_key(case: Case) -> str:
    """The cache key of the oracle row of ``case``: its model, observable
    and weight, and the oracle's code; not its flags or spelling."""
    text = json.dumps([model_to_dict(case.model),
                       observable_to_dict(case.observable), case.max_weight,
                       ORACLE_SHA], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def check_cache_dir(path: pathlib.Path) -> pathlib.Path:
    """``path``, which must lie outside the project."""
    resolved = pathlib.Path(path).resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise SystemExit(f"--oracle-cache {path} lies inside {ROOT}; put "
                         "the cache outside the project")
    return pathlib.Path(path)


def _oracle_task(case: Case, timeout: float, max_rss: float) -> dict:
    """The oracle of one case, in a pool process."""
    t0 = time.perf_counter()
    memory: dict = {}
    out: dict = {}
    try:
        ref, scale = fuzz.oracle_of(case, timeout, max_rss, memory)
        out.update(rows=[[list(t), float(v)] for t, v in ref.items()],
                   scale=float(scale))
    except JobTimeout as exc:
        out.update(lost="oracle-timeout", error=str(exc))
    except JobMemory as exc:
        out.update(lost="oracle-memory", error=str(exc))
    except Exception as exc:  # noqa: BLE001  (recorded, not compared)
        out.update(lost="oracle-error", error=f"{type(exc).__name__}: {exc}")
    out["seconds"] = time.perf_counter() - t0
    if memory.get("peak"):
        out["rss_mb"] = memory["peak"] / 2 ** 20
    return out


def reusable(entry: dict, timeout: float, max_rss: float) -> bool:
    """Whether a cached entry answers for a run with these limits: a
    computed row always; a lost one only when it was lost under limits at
    least as large."""
    if "lost" not in entry:
        return True
    limits = entry.get("limits") or {}
    return (limits.get("timeout", 0.0) >= timeout
            and limits.get("max_rss", 0.0) >= max_rss)


def oracle_rows(cases: Sequence[Case], cache: pathlib.Path, jobs: int,
                timeout: float, max_rss: float, log=print) -> dict:
    """``{oracle_key: entry}`` for ``cases``, from the cache where it holds
    the key (:func:`reusable`), else computed in at most ``jobs`` processes
    and written."""
    cache.mkdir(parents=True, exist_ok=True)
    out, todo = {}, {}
    for case in cases:
        key = oracle_key(case)
        path = cache / f"{key}.json"
        if key in out or key in todo:
            continue
        entry = json.loads(path.read_text()) if path.exists() else None
        if entry is not None and reusable(entry, timeout, max_rss):
            out[key] = entry
        else:
            todo[key] = case
    if not todo:
        return out
    log(f"oracle: {len(todo)} to compute, {len(out)} cached")
    if str(TOOLS) not in sys.path:
        sys.path.append(str(TOOLS))  # a pool process imports this module
    pool = fuzz._new_pool(jobs)
    try:
        futures = {pool.submit(_oracle_task, case, timeout, max_rss): key
                   for key, case in todo.items()}
        for future in cf.as_completed(futures):
            key = futures[future]
            keep = True
            try:
                entry = future.result()
            except Exception as exc:  # noqa: BLE001  (a lost process)
                entry = dict(lost="oracle-error",
                             error=f"{type(exc).__name__}: {exc}",
                             seconds=0.0)
                keep = False
            entry.update(oracle_sha=ORACLE_SHA, seed=todo[key].seed)
            if "lost" in entry:
                entry["limits"] = dict(timeout=timeout, max_rss=max_rss)
            if keep:
                (cache / f"{key}.json").write_text(json.dumps(entry))
            out[key] = entry
            log(f"oracle: seed {todo[key].seed} "
                f"{entry.get('lost', 'ok')} {entry['seconds']:.1f} s")
    except BaseException:
        # As validation/fuzz.py's driver: kill, then forget the pool.
        # Killing the processes and then waiting for the pool can hang on a
        # process the pool starts after the kill (one task per process), so
        # the normal path below shuts the pool down only.
        fuzz._stop_pool(pool)
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    return out
