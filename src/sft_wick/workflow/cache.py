"""Opt-in disk caching for expensive wrapper outputs.

Uses ``joblib.dump/load`` — already a transitive dependency of the
package (via :func:`integrate_diagrams`'s parallel worker) and the
standard choice for caching computed Python objects in the
scientific-Python ecosystem (same backend as ``sklearn``'s
``Memory``).  Cache files are always produced and consumed by the same
trusted process; the key check below also rejects files whose content
hash does not match the declared specification hash.

Design: caching is **always explicit**.  The user supplies
``cache_path`` to an operation that can be cached; the cache key is a
short content hash of the relevant spec objects.  If no
``cache_path`` is supplied, a one-line reminder is printed
(suppressible with the ``SFT_WICK_QUIET_CACHE`` environment variable).

Two reasons for opt-in:

1. **Cache invalidation is subtle for physics workflows**: a user may
   change a field value mid-session and expect results to update.  A
   silent cache would mask such edits.
2. **Cache size can be large**: propagator splines on fine grids can
   reach tens of MB, which would silently bloat a repository or a
   cluster job's output directory.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable

_REMINDER_SEEN: set[str] = set()


def _reminder_quiet() -> bool:
    return os.environ.get("SFT_WICK_QUIET_CACHE", "0") not in ("0", "")


def hash_spec(obj: Any, length: int = 12) -> str:
    """Return a short hex hash of ``obj``'s canonical serialisation.

    Falls back on ``repr(obj)`` when the object isn't serialisable by
    joblib — callers should keep spec objects pickle-friendly.
    """
    try:
        from joblib import hash as _joblib_hash

        return _joblib_hash(obj)[:length]
    except Exception:
        return hashlib.sha256(repr(obj).encode("utf-8")).hexdigest()[:length]


def load_or_compute(
    cache_path: str | Path | None,
    spec_obj: Any,
    compute_fn: Callable[[], Any],
    *,
    operation_name: str = "operation",
) -> Any:
    """Load a cached result matching ``spec_obj``'s hash from
    ``cache_path``; otherwise call ``compute_fn``, save the result to
    ``cache_path``, and return it.

    When ``cache_path`` is None, ``compute_fn`` is always invoked and
    a one-shot reminder is printed pointing to the kwarg.

    Args:
        cache_path: directory or file path.  If it names a directory
            (or a non-existing file with no ``.joblib`` suffix), the
            file is ``<dir>/<operation>_<key>.joblib`` inside that
            directory.  ``None`` disables caching.
        spec_obj: picklable object used to derive the cache key.
        compute_fn: zero-argument callable producing the result.
        operation_name: label used in the reminder and file name.

    Returns:
        The (possibly cached) result.
    """
    if cache_path is None:
        _maybe_remind(operation_name)
        return compute_fn()

    from joblib import dump as _jdump, load as _jload

    path = Path(cache_path)
    key = hash_spec(spec_obj)

    if path.is_dir() or (not path.exists() and path.suffix == ""):
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"{operation_name}_{key}.joblib"
    else:
        file_path = path

    if file_path.exists():
        try:
            payload = _jload(file_path)
            if isinstance(payload, dict) and payload.get("key") == key:
                return payload["value"]
        except Exception:
            pass  # corrupt or stale cache — recompute

    value = compute_fn()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    _jdump({"key": key, "value": value}, file_path)
    return value


def _maybe_remind(operation_name: str) -> None:
    """Print a one-shot reminder that caching is available."""
    if operation_name in _REMINDER_SEEN or _reminder_quiet():
        return
    _REMINDER_SEEN.add(operation_name)
    print(
        f"[sft-wick] Tip: {operation_name} is cacheable.  Pass "
        f"`cache_path='path/to/cache/dir'` to save results for re-use.  "
        f"Silence with env SFT_WICK_QUIET_CACHE=1.",
        flush=True,
    )
