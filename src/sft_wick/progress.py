"""Progress reporting shared by the table builders, the integrators and
the workflow layer.

Reporting never changes a number: every helper here wraps an iteration
that already existed and only *observes* it.  Three switches decide
whether anything is shown, in this precedence:

1. an explicit :func:`progress` context (what the L1 ``progress=`` kwarg
   and the CLI ``--quiet`` flag set): ``True`` / ``False`` / a callable;
2. the ``SFT_WICK_PROGRESS`` environment variable -- ``0``, ``false`` or
   ``off`` silences everything, ``1`` forces the text fallback on even
   when stderr is not a terminal;
3. otherwise bars are shown only when stderr is an interactive terminal,
   so library use inside pytest or a batch job stays quiet by default.

``tqdm`` is used when it is importable (``pip install sft-wick[progress]``);
without it, or when stderr is not a TTY, a plain one-line-per-step text
fallback is printed to stderr at most every few seconds.  A callable
setting receives ``(description, done, total)`` events instead and
nothing is printed.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from typing import Any, Callable, Iterable, Sequence

_FALSY = {"0", "false", "off", "no"}
_TRUTHY = {"1", "true", "on", "yes"}

#: Stack of explicit settings pushed by :func:`progress`.
_SETTINGS: list[Any] = []


def _env_setting() -> bool | None:
    raw = os.environ.get("SFT_WICK_PROGRESS")
    if raw is None:
        return None
    val = raw.strip().lower()
    if val in _FALSY:
        return False
    if val in _TRUTHY:
        return True
    return None


def current_setting() -> Any:
    """Resolve the active setting: ``True``, ``False`` or a callable."""
    if _SETTINGS:
        setting = _SETTINGS[-1]
        if setting is not None:
            return setting
    env = _env_setting()
    if env is not None:
        return env
    try:
        return bool(sys.stderr.isatty())
    except Exception:  # pragma: no cover - exotic stderr replacements
        return False


def is_enabled() -> bool:
    """Whether anything (bars, callbacks or stage banners) is reported."""
    return current_setting() is not False


@contextlib.contextmanager
def progress(setting: Any):
    """Scope a progress setting: ``True`` / ``False`` / callable / ``None``.

    ``None`` defers to the enclosing scope (then the environment, then the
    TTY check); this is what makes ``progress=None`` the transparent
    default of every L1 entry point.
    """
    _SETTINGS.append(setting)
    try:
        yield
    finally:
        _SETTINGS.pop()


def _tqdm_class():
    if _env_setting() is None and not sys.stderr.isatty():
        return None
    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        return None
    return tqdm


#: Bars stay silent for loops that finish within this many seconds, so a
#: closed-form table build (milliseconds) does not print a dozen lines.
_DELAY = 1.0


class _TextBar:
    """Plain-stderr fallback: one line at most every ``interval`` seconds,
    and nothing at all for loops shorter than :data:`_DELAY`."""

    def __init__(self, total: int, desc: str, unit: str, interval: float = 2.0):
        self.total = int(total)
        self.desc = desc
        self.unit = unit
        self.interval = interval
        self.done = 0
        self.t0 = time.perf_counter()
        self._last = self.t0
        self._last_frac = 0.0
        self._printed = False

    def _emit(self, force: bool = False) -> None:
        now = time.perf_counter()
        frac = self.done / self.total if self.total else 1.0
        if (now - self.t0) < _DELAY and not (force and self._printed):
            return
        if not force and (now - self._last) < self.interval and (frac - self._last_frac) < 0.1:
            return
        self._printed = True
        elapsed = now - self.t0
        eta = (elapsed / frac - elapsed) if frac > 0 else float("nan")
        eta_s = f"{eta:.0f}s" if eta == eta else "?"
        print(
            f"[sft-wick] {self.desc}: {self.done}/{self.total} {self.unit} "
            f"({100.0 * frac:.0f}%) elapsed {elapsed:.1f}s eta {eta_s}",
            file=sys.stderr, flush=True,
        )
        self._last = now
        self._last_frac = frac

    def update(self, n: int = 1) -> None:
        self.done += int(n)
        self._emit(force=(self.done >= self.total))

    def close(self) -> None:
        if self.done < self.total:
            self._emit(force=True)


class _CallbackBar:
    def __init__(self, cb: Callable, total: int, desc: str):
        self.cb, self.total, self.desc, self.done = cb, int(total), desc, 0
        self.cb(desc, 0, self.total)

    def update(self, n: int = 1) -> None:
        self.done += int(n)
        self.cb(self.desc, self.done, self.total)

    def close(self) -> None:
        pass


@contextlib.contextmanager
def progress_bar(total: int, desc: str, unit: str = "it"):
    """Yield an ``update(n=1)`` callable for a loop of ``total`` steps.

    A no-op when reporting is disabled; a ``tqdm`` bar, the text fallback,
    or a user callback otherwise.
    """
    setting = current_setting()
    if setting is False or total <= 0:
        yield lambda n=1: None
        return
    if callable(setting):
        bar: Any = _CallbackBar(setting, total, desc)
    else:
        tqdm = _tqdm_class()
        if tqdm is not None:
            bar = tqdm(
                total=total, desc=f"[sft-wick] {desc}", unit=unit,
                file=sys.stderr, dynamic_ncols=True, leave=True,
                mininterval=0.5, delay=_DELAY,
            )
        else:
            bar = _TextBar(total, desc, unit)
    try:
        yield bar.update
    finally:
        bar.close()


def progress_map(
    fn: Callable,
    tasks: Sequence,
    desc: str,
    *,
    n_jobs: int = 1,
    unit: str = "it",
    serial_below: int = 5,
) -> list:
    """``[fn(t) for t in tasks]`` -- serially or through joblib -- with a bar.

    The parallel branch streams results back in submission order
    (``return_as='generator'``), so the bar advances as workers finish
    and the returned list is identical to the serial one.  Small batches
    (``len(tasks) < serial_below``) run serially to avoid loky's start-up
    cost, matching the thresholds the builders used before this helper.
    """
    tasks = list(tasks)
    out: list = []
    with progress_bar(len(tasks), desc, unit=unit) as tick:
        if n_jobs == 1 or len(tasks) < serial_below:
            for t in tasks:
                out.append(fn(t))
                tick()
        else:
            from joblib import Parallel, delayed
            gen = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
                delayed(fn)(t) for t in tasks
            )
            for r in gen:
                out.append(r)
                tick()
    return out


@contextlib.contextmanager
def stage(name: str, detail: str = ""):
    """Print ``[sft-wick] <name> ...`` and the elapsed time on exit.

    Used by the CLI for its stage banners; silent when reporting is off
    or routed to a callback.
    """
    setting = current_setting()
    show = setting is True
    t0 = time.perf_counter()
    if show:
        suffix = f" ({detail})" if detail else ""
        print(f"[sft-wick] {name}{suffix} ...", file=sys.stderr, flush=True)
    try:
        yield
    finally:
        if show:
            print(f"[sft-wick] {name} done in {time.perf_counter() - t0:.1f}s",
                  file=sys.stderr, flush=True)
