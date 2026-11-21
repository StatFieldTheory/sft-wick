"""One pytest run of the parent-commit gate under its time and memory limits.

Loaded by ``tools/parent_gate.py``, which re-exports :func:`run_limited`.
A run is started in a session of its own, so its first process (the root)
has a pid equal to its process group and session numbers.  The root is not
reaped until its group has been killed: the gate learns that it exited with
``waitid(..., WNOWAIT)``, which leaves it a zombie, and a zombie's pid, and
so its group number, is not given out again.  Every run therefore ends in
the same order, whether the root exited, a limit was passed, the gate was
stopped or an exception was raised: SIGKILL to the root's group and to the
root, then to the run's processes outside that group that a new process
table confirms, then to the root's group again, and only then the reap.
When ``ps`` cannot be started the sends to the root and its group still
happen, and the run raises the ``OSError``.

A process is known by its pid and its start time, and forgotten once it is
no longer in the process table.  The run's processes are the root, every
process seen before with the same start time, every process in the process
group of one of those, and every descendant by parent pid of any of them.
A test's process that starts a new session is counted and killed once a
poll has seen it, and a process that reuses the pid of one that has exited
is neither.  A process that leaves both the run's groups and its parent
chain between two polls, such as a daemon made by a double fork, is not
counted or killed.  Their memory is summed with :data:`MEMORY_MEASURE`: the physical
footprint on macOS (compressed pages included), anonymous and shared
resident memory plus swap on Linux, the resident set elsewhere.

A signal of :data:`HANDLED_SIGNALS` sets the module-level flag
:data:`_STOPPED` and raises ``SystemExit``; the handler calls nothing.  In
the main thread a whole :func:`run_limited` is one deferred block: the
handler only sets the flag, the run's waits read it every
:data:`STOP_CHECK_SECONDS`, the run is ended in the order above, and the
``SystemExit`` is raised when the block ends.  A stopped gate (a signal or
:func:`stop_all`) starts no new run: the run raises :class:`Stopped`.
Standard library only.
"""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import dataclasses
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

POLL_SECONDS = 0.2
#: The longest a wait of a run sleeps before it reads the stop flag again.
STOP_CHECK_SECONDS = 0.05
#: Signals that stop a run from outside the gate; a serial pytest process that
#: ends by one of them, when the gate did not kill it, gives no verdict.  A
#: run whose pytest-xdist worker crashed, or that pytest reports as
#: interrupted (exit 2), gives none either (``tools/parent_gate.py``).
EXTERNAL_SIGNALS = (signal.SIGKILL, signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
#: Signals on which the gate kills its runs and exits ``128 + signal``.
HANDLED_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


class Stopped(Exception):
    """The gate was stopped: by a signal, by :func:`stop_all` or by an exception of the replay."""


@dataclasses.dataclass(frozen=True)
class Outcome:
    """What :func:`run_limited` saw of one run.

    ``peak_mb`` is the largest polled sum of :data:`MEMORY_MEASURE` over the
    run's processes; ``maxrss_mb`` is the largest resident set of the child
    process alone (``ru_maxrss``), which also covers the time between polls.
    """

    exit: int | None
    output: str
    seconds: float
    peak_mb: float
    maxrss_mb: float
    timed_out: bool
    memory_killed: bool


# ---- Memory of one process ----

def _footprint_reader():
    """``pid -> bytes`` of the physical footprint (macOS), or None."""
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    except OSError:
        return None
    flavor, offset = 2, 16 + 7 * 8  # RUSAGE_INFO_V2: uuid[16], 7 uint64, ri_phys_footprint

    def read(pid: int) -> int | None:
        buf = ctypes.create_string_buffer(512)
        if libproc.proc_pid_rusage(pid, flavor, buf) != 0:
            return None
        return int.from_bytes(buf.raw[offset:offset + 8], "little")
    return read


def _linux_reader(pid: int) -> int | None:
    """RssAnon + RssShmem + VmSwap of ``/proc/<pid>/status`` in bytes, or None."""
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    total, found = 0, False
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key in ("RssAnon", "RssShmem", "VmSwap") and value.split():
            total += int(value.split()[0]) * 1024
            found = True
    return total if found else None


def _no_reader(pid: int) -> int | None:
    """No reader: :func:`memory_bytes` takes the resident set of the process table."""
    return None


def _select_reader() -> tuple[str, Callable[[int], int | None]]:
    """The name of the memory measure of this platform and its ``pid -> bytes`` reader."""
    if sys.platform == "darwin":
        footprint = _footprint_reader()
        if footprint is not None:
            return "physical footprint", footprint
    if sys.platform.startswith("linux"):
        return "anonymous and shared resident memory plus swap", _linux_reader
    return "resident set size", _no_reader


MEMORY_MEASURE, _memory_of = _select_reader()


# ---- The processes of a run ----

class Process(NamedTuple):
    """One row of the process table."""

    ppid: int
    pgid: int
    rss: int
    start: str


def _process_table() -> dict[int, Process]:
    """pid -> :class:`Process`, from one ``ps``; ``start`` is the start time as text."""
    proc = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,pgid=,rss=,lstart="],
                          capture_output=True, text=True,
                          env={**os.environ, "LC_ALL": "C", "TZ": "UTC"})
    table = {}
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) > 4 and all(f.isdigit() for f in fields[:4]):
            pid, ppid, pgid, rss = (int(f) for f in fields[:4])
            table[pid] = Process(ppid, pgid, rss * 1024, " ".join(fields[4:]))
    return table


class Tree:
    """The processes of one run, each known by its pid and start time.

    A process is a member when it is the root and the root is not yet
    reaped, when it was a member at the last call with the same start time,
    when it is in the process group of such a process, or when it descends
    by parent pid from any member.  A group holds only processes of the run
    while one of its members is a confirmed process of the run, because a
    pid is not given out again while a process group of that number exists.
    The lock orders :meth:`kill` and :meth:`reap` between threads; the
    signal handler never takes it.
    """

    def __init__(self, root: int) -> None:
        self.root = root
        self.root_alive = True
        self.seen: dict[int, str] = {}
        self.lock = threading.Lock()

    def root_reaped(self) -> None:
        """The root was reaped: its pid may now be another process's."""
        self.root_alive = False

    def members(self, table: Mapping[int, Process]) -> list[int]:
        """The run's live processes in ``table``; :attr:`seen` becomes exactly them."""
        seen = self.seen
        confirmed = {pid for pid, start in seen.items()
                     if pid in table and table[pid].start == start}
        if self.root_alive and self.root in table:
            confirmed.add(self.root)
        groups = {table[pid].pgid for pid in confirmed}
        found = confirmed | {pid for pid, row in table.items() if row.pgid in groups}
        children: dict[int, list[int]] = {}
        for pid, row in table.items():
            children.setdefault(row.ppid, []).append(pid)
        stack = list(found)
        while stack:
            for child in children.get(stack.pop(), []):
                if child not in found:
                    found.add(child)
                    stack.append(child)
        self.seen = {pid: table[pid].start for pid in found}
        return sorted(found)

    def root_exited(self) -> bool:
        """Whether the root has exited, which leaves it unreaped (a zombie)."""
        return os.waitid(os.P_PID, self.root, os.WEXITED | os.WNOHANG | os.WNOWAIT) \
            is not None

    def _kill_root_group(self) -> None:
        """SIGKILL the root's group and the root; both numbers are the run's until the reap."""
        _send_kills([(os.killpg, self.root), (os.kill, self.root)])

    def kill(self) -> OSError | None:
        """SIGKILL the run's live processes and leave the root unreaped.

        The root's group and the root first, then the members outside that
        group that a new process table confirms, with their groups, then
        the root's group again.  Nothing is sent once the root is reaped.
        Returns the ``OSError`` of a process table that could not be read.
        """
        with self.lock:
            if not self.root_alive:
                return None
            self._kill_root_group()
            try:
                table = _process_table()
            except OSError as exc:
                error = exc
            else:
                error = None
                pids = [pid for pid in self.members(table) if table[pid].pgid != self.root]
                groups = sorted({table[pid].pgid for pid in pids})
                _send_kills([(os.killpg, group) for group in groups]
                            + [(os.kill, pid) for pid in pids])
            self._kill_root_group()
        return error

    def reap(self) -> tuple[int, object]:
        """Wait for the root and mark it reaped, in one step; its wait status and usage."""
        with self.lock:
            _, status, usage = os.wait4(self.root, 0)
            self.root_reaped()
        return status, usage


def _send_kills(sends: Sequence[tuple[Callable[[int, int], None], int]]) -> None:
    """Send SIGKILL by each ``(send, number)``; a number with no process is passed over.

    macOS answers ``killpg`` of a group that holds only a zombie with EPERM.
    """
    for send, number in sends:
        try:
            send(number, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            continue


def memory_bytes(table: Mapping[int, Process], pids: Sequence[int]) -> int:
    """The summed :data:`MEMORY_MEASURE` of ``pids``; the resident set where it is unreadable."""
    total = 0
    for pid in pids:
        value = _memory_of(pid)
        row = table.get(pid)
        total += value if value is not None else (row.rss if row else 0)
    return total


# ---- Stopping ----

_LIVE: dict[int, Tree] = {}
_LIVE_LOCK = threading.Lock()
#: Set by a handled signal or :func:`stop_all`: no further run starts and live runs end.
_STOPPED = False


def stop_all() -> None:
    """Start no further run and kill every live one; each run's own thread reaps it."""
    global _STOPPED
    _STOPPED = True
    with _LIVE_LOCK:
        trees = list(_LIVE.values())
    for tree in trees:
        tree.kill()


@atexit.register
def _kill_live() -> None:
    with _LIVE_LOCK:
        trees = list(_LIVE.values())
    for tree in trees:
        tree.kill()


def clear_stop() -> None:
    """Let runs start again, once no thread of a stopped replay runs one."""
    global _STOPPED
    _STOPPED = False


class _Deferral:
    """How deep the main thread is in :func:`_signals_deferred`, and a signal held meanwhile."""

    depth = 0
    pending: int | None = None


def _on_signal(signum: int, frame) -> None:
    """Set the stop flag and raise ``SystemExit``, or hold it while deferred; no call at all."""
    global _STOPPED
    _STOPPED = True
    if _Deferral.depth:
        _Deferral.pending = signum
        return
    raise SystemExit(128 + signum)


@contextlib.contextmanager
def _signals_deferred() -> Iterator[None]:
    """In the main thread, hold a handled signal's ``SystemExit`` until the block ends."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    _Deferral.depth += 1
    try:
        yield
    finally:
        _Deferral.depth -= 1
        if not _Deferral.depth and _Deferral.pending is not None:
            pending, _Deferral.pending = _Deferral.pending, None
            raise SystemExit(128 + pending)


def install_signal_handlers() -> dict[int, object]:
    """Handle :data:`HANDLED_SIGNALS` in the main thread; the previous handlers."""
    clear_stop()
    if threading.current_thread() is not threading.main_thread():
        return {}
    return {signum: signal.signal(signum, _on_signal) for signum in HANDLED_SIGNALS}


def restore_signal_handlers(previous: Mapping[int, object]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _check_stop() -> None:
    if _STOPPED:
        raise Stopped("the gate was stopped; no further run is started")


def _sleep(seconds: float) -> None:
    """Sleep ``seconds``, in steps of :data:`STOP_CHECK_SECONDS`, until the gate is stopped."""
    end = time.monotonic() + seconds
    while not _STOPPED:
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(left, STOP_CHECK_SECONDS))


# ---- One run ----

def _maxrss_bytes(value: int) -> int:
    return value if sys.platform == "darwin" else value * 1024


def _poll(tree: Tree, start: float, timeout: float | None, max_bytes: int | None):
    """Poll until the root exits, a limit is passed or the gate is stopped.

    The peak memory and whether the time or the memory limit ended the
    run.  The root is left unreaped.
    """
    peak = 0
    while not tree.root_exited():
        table = _process_table()
        used = memory_bytes(table, tree.members(table))
        peak = max(peak, used)
        memory_killed = bool(max_bytes) and used > max_bytes
        timed_out = timeout is not None and time.monotonic() - start > timeout
        if memory_killed or timed_out or _STOPPED:
            return peak, timed_out, memory_killed and not timed_out
        _sleep(POLL_SECONDS)
    return peak, False, False


def run_limited(command: Sequence[str], *, cwd: Path, env: Mapping[str, str],
                timeout: float | None, max_rss_bytes: int | None, log: Path) -> Outcome:
    """Run ``command`` in a session of its own, with its output in ``log``.

    Every :data:`POLL_SECONDS` the memory of the run's processes is summed;
    they are killed above ``max_rss_bytes`` or after ``timeout`` seconds.
    Raises :class:`Stopped` when the gate is stopped before or during the
    run, and the ``OSError`` when the process table could not be read at
    its end.  In the main thread a handled signal's ``SystemExit`` is raised
    once the run is ended and reaped.
    """
    with _signals_deferred():
        return _run(command, cwd=cwd, env=env, timeout=timeout,
                    max_rss_bytes=max_rss_bytes, log=log)


def _run(command: Sequence[str], *, cwd: Path, env: Mapping[str, str],
         timeout: float | None, max_rss_bytes: int | None, log: Path) -> Outcome:
    _check_stop()
    start = time.monotonic()
    with open(log, "w", encoding="utf-8", errors="replace") as sink:
        proc = subprocess.Popen(list(command), cwd=cwd, env=dict(env), stdout=sink,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
        tree = Tree(proc.pid)
        with _LIVE_LOCK:
            _LIVE[proc.pid] = tree
        try:
            peak, timed_out, memory_killed = _poll(tree, start, timeout, max_rss_bytes)
        finally:
            try:
                error = tree.kill()
            finally:
                status, usage = tree.reap()
                proc.returncode = os.waitstatus_to_exitcode(status)
                with _LIVE_LOCK:
                    _LIVE.pop(proc.pid, None)
    if error is not None:
        raise error
    _check_stop()
    output = log.read_text(encoding="utf-8", errors="replace")
    return Outcome(proc.returncode, output, time.monotonic() - start, peak / 2**20,
                   _maxrss_bytes(usage.ru_maxrss) / 2**20, timed_out, memory_killed)
