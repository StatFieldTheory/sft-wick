"""Where a lowering failed, and how long a step may take.

:func:`stage` marks an exception with the step that raised it (``lower``,
``expand``, ``propagators``, ``evaluate``), so that ``validation/worker.py``
can report it.  :func:`deadline` raises :class:`JobTimeout` when a step runs
past its time; the worker puts each job under it and the fuzzer each oracle
evaluation.  :func:`memory_limit` raises :class:`JobMemory` when this
process's resident memory passes a limit; the fuzzer puts each oracle
evaluation under it, and :func:`resident_bytes` measures a worker from its
parent.  :func:`exit_with_parent` ends a process whose parent has gone.
Imports no sft-wick code.
"""
from __future__ import annotations

import contextlib
import math
import os
import signal
import threading
import time
from typing import Iterator

__all__ = ["STAGE_ATTR", "stage", "JobTimeout", "deadline", "check_timeout",
           "ALARM_REPEAT", "JobMemory", "memory_limit", "resident_bytes",
           "MEMORY_POLL", "exit_with_parent", "PARENT_POLL"]

#: The attribute that carries the stage on an exception.
STAGE_ATTR = "validation_stage"
#: The alarm repeats at this interval until a :class:`JobTimeout` escapes:
#: one raised inside a weakref callback or ``__del__`` is only printed.
ALARM_REPEAT = 1.0
#: Seconds between two samples of resident memory.
MEMORY_POLL = 0.25
#: Seconds between two checks that a process's parent is still there.
PARENT_POLL = 1.0


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


class JobMemory(BaseException):
    """A step went past its resident-memory limit.  A ``BaseException``, as
    :class:`JobTimeout` is."""


def resident_bytes(pid: int | None = None, children: bool = True) -> int:
    """Resident memory of process ``pid`` (this process by default), with
    that of every process it started when ``children``, in bytes.  A process
    that has exited, or exits while it is read, counts 0."""
    import psutil

    try:
        proc = psutil.Process(pid)
        total = proc.memory_info().rss
        kids = proc.children(recursive=True) if children else []
    except psutil.Error:
        return 0
    for child in kids:
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total


@contextlib.contextmanager
def memory_limit(limit: float | None,
                 poll: float | None = None) -> Iterator[dict]:
    """Sample this process's resident memory every ``poll`` seconds (default
    :data:`MEMORY_POLL`) in a thread, and raise :class:`JobMemory` in the
    main thread once a sample is above ``limit`` bytes, again every
    :data:`ALARM_REPEAT` seconds until it escapes.  Yields a dict whose
    ``peak`` is the largest sample so far and whose ``enforced`` says
    whether the limit is enforced: it needs ``SIGUSR1`` and the main thread.
    A step inside one long call of compiled code is reached only when the
    call returns.  With ``limit`` ``None`` nothing is sampled and ``peak``
    is ``None``."""
    if limit is None:
        yield dict(peak=None, enforced=False)
        return
    poll = MEMORY_POLL if poll is None else poll
    info = dict(peak=resident_bytes(children=False), enforced=False)
    enforce = (hasattr(signal, "pthread_kill")
               and hasattr(signal, "SIGUSR1")
               and threading.current_thread() is threading.main_thread())
    armed = threading.Event()
    stop = threading.Event()
    main = threading.main_thread().ident

    def on_signal(signum, frame):
        if armed.is_set():
            raise JobMemory(f"resident memory above {limit / 2**30:.3g} GB")

    def watch():
        sent = None
        while not stop.wait(poll):
            info["peak"] = max(info["peak"], resident_bytes(children=False))
            if not enforce or info["peak"] <= limit:
                continue
            now = time.monotonic()
            if sent is None or now - sent >= ALARM_REPEAT:
                signal.pthread_kill(main, signal.SIGUSR1)
                sent = now

    previous = signal.getsignal(signal.SIGUSR1) if enforce else None
    if enforce:
        signal.signal(signal.SIGUSR1, on_signal)
        armed.set()
        info["enforced"] = True
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        yield info
    finally:
        armed.clear()
        stop.set()
        watcher.join()
        if enforce:
            signal.signal(signal.SIGUSR1, previous)


def exit_with_parent(parent: int | None = None,
                     poll: float | None = None) -> threading.Thread:
    """End this process, with no cleanup, once its parent has gone: a daemon
    thread compares the parent's pid with ``parent`` (by default the parent
    at the call) every ``poll`` seconds (default :data:`PARENT_POLL`).  When
    ``parent`` is given and is already not the parent, the process ends at
    once, so a parent that died while this process started is caught too.
    A worker or a pool process whose run was killed then stops instead of
    finishing its case unwatched."""
    poll = PARENT_POLL if poll is None else poll
    parent = os.getppid() if parent is None else int(parent)

    def leave():
        try:
            os.write(2, f"pid {os.getpid()}: parent {parent} has gone, "
                        f"exiting\n".encode())
        finally:
            os._exit(1)

    if os.getppid() != parent:
        leave()

    def watch():
        while True:
            time.sleep(poll)
            if os.getppid() != parent:
                leave()

    thread = threading.Thread(target=watch, daemon=True,
                              name="exit-with-parent")
    thread.start()
    return thread
