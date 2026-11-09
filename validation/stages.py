"""Where a lowering failed, and how long a step may take.

:func:`stage` marks an exception with the step that raised it (``lower``,
``expand``, ``propagators``, ``evaluate``), so that ``validation/worker.py``
can report it.  :func:`deadline` raises :class:`JobTimeout` when a step runs
past its time; the worker puts each job under it and the fuzzer each oracle
evaluation.  Imports no sft-wick code.
"""
from __future__ import annotations

import contextlib
import math
import signal
import threading
from typing import Iterator

__all__ = ["STAGE_ATTR", "stage", "JobTimeout", "deadline", "check_timeout",
           "ALARM_REPEAT"]

#: The attribute that carries the stage on an exception.
STAGE_ATTR = "validation_stage"
#: The alarm repeats at this interval until a :class:`JobTimeout` escapes:
#: one raised inside a weakref callback or ``__del__`` is only printed.
ALARM_REPEAT = 1.0


@contextlib.contextmanager
def stage(name: str) -> Iterator[None]:
    """Mark an exception raised inside with ``name``, unless an inner stage
    already marked it."""
    try:
        yield
    except BaseException as exc:
        if not hasattr(exc, STAGE_ATTR):
            try:
                setattr(exc, STAGE_ATTR, name)
            except AttributeError:
                pass
        raise


class JobTimeout(BaseException):
    """A step ran past its deadline.  It derives from ``BaseException`` so that
    no ``except Exception`` in the package swallows it."""


def check_timeout(value) -> float | None:
    """``None``, or a finite positive number of seconds."""
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"timeout must be a positive number of seconds or "
                         f"null, got {value!r}")
    return float(value)


@contextlib.contextmanager
def deadline(seconds: float | None) -> Iterator[bool]:
    """Raise :class:`JobTimeout` after ``seconds`` of wall time, and again
    every :data:`ALARM_REPEAT` seconds until it escapes.  Yields whether the
    limit is enforced: it needs ``SIGALRM`` and the main thread."""
    seconds = check_timeout(seconds)
    if (seconds is None or not hasattr(signal, "setitimer")
            or threading.current_thread() is not threading.main_thread()):
        yield False
        return
    message = f"no result within {seconds:g} s"

    def on_alarm(signum, frame):
        raise JobTimeout(message)

    previous = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, on_alarm)
        signal.setitimer(signal.ITIMER_REAL, seconds, ALARM_REPEAT)
        yield True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)
