r"""Random differential runs of the package against the oracle.

``python -m validation.fuzz run DIR --count 300`` draws cases with
:func:`validation.draw.draw` from consecutive seeds, computes the oracle in
this process and sends every integrator's job to ``python -m
validation.worker`` in a subprocess, so this process never loads sft_wick.
It writes ``DIR/records.jsonl`` (one record per case and integrator),
``DIR/summary.json`` and ``DIR/run.json`` (the arguments of every
invocation).  ``--resume`` runs only the (seed, integrator) pairs that have
no record; a run without it clears ``records.jsonl`` and ``run.json``.
``python -m validation.fuzz summary DIR`` prints the summary again;
``python -m validation.fuzz shrink DIR --seed S --method M`` reduces a
failing case (:mod:`validation.shrink`) under the two memory limits below.

``--package DIR`` names the directory that holds the ``sft_wick`` under
test (default this tree's ``src``), ``--lowering`` the module that lowers
onto it, and ``--extra-path`` directories the worker also searches, for a
lowering outside this tree.

Memory is bounded per process and for the machine.  ``--max-rss-gb``
(default :data:`DEFAULT_MAX_RSS_GB`) limits the resident memory of each
worker, with any process it starts, read by its parent every
:data:`~validation.stages.MEMORY_POLL` seconds, and ``--oracle-max-rss-gb``
(default :data:`DEFAULT_ORACLE_MAX_RSS_GB`) that of each oracle evaluation
(:func:`~validation.stages.memory_limit`).  ``--min-available-gb`` is the
memory the run leaves to the rest of the machine (default
:data:`DEFAULT_MIN_AVAILABLE_GB`, or a quarter of the physical memory when
that is less): a case starts only when the memory available, less both
limits of every running case and of the new one, is still that much
(:func:`admits`), and with no case running once that much is available.  A
limit of 0 is no limit.  ``--jobs`` (default the number of processors less
4) caps the cases that run at once.  Both limits are of a whole process: a
worker's counts what the earlier integrators of its case left, and before
Python 3.13 an oracle's counts what earlier cases left in its pool process;
from 3.13 each case runs in a new process.  Available and physical memory
are the host's (``psutil.virtual_memory``; on macOS available is inactive
plus free memory), resident memory leaves out compressed pages, and cgroup
limits are not read.  Records keep the largest samples, ``peak_rss_mb`` of
the worker and ``oracle_rss_mb`` of the oracle.

A run stopped by an interrupt or ``SIGTERM`` kills its pool and the
workers.  A pool process or a worker whose parent has gone ends itself
within a second (:func:`~validation.stages.exit_with_parent`), so a run
killed outright leaves no case running either.  A pool process that dies
breaks the pool, and every case running in it fails: each of them runs
again, alone, in a new pool, and a case that kills its pool process twice
while it runs alone is recorded as ``harness-error``.  The fuzzer needs
psutil.

A record's ``verdict`` is one of:

* ``pass``: every tag within tolerance; ``fail``: a tag outside it;
* ``refusal``: the package raised;
* ``callback-error``: an oracle callable that the package called (the closed
  form of C, a Campbell cumulant) raised, which a bad call by the package
  or a fault in the callable can cause;
* ``lowering-error``: the lowering raised before reaching the package;
* ``timeout``: past the job's deadline; ``killed``: the worker gave no
  answer within the deadline plus :data:`WATCHDOG_MARGIN`; ``memory``: the
  worker went past ``--max-rss-gb`` and was killed; ``crashed``: the
  worker exited (its return code and the end of its stderr are kept);
* ``oracle-error``, ``oracle-timeout``, ``oracle-memory``: the oracle
  raised, ran past ``--oracle-timeout`` or went past
  ``--oracle-max-rss-gb``, so nothing was compared;
* ``harness-error``: anything else, including a worker that ran another
  method, settings or flags than the job asked for.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import concurrent.futures.process  # noqa: F401  (BrokenProcessPool)
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

from .compare import compare_values, tolerance
from .draw import Case, draw
from .hierarchy import moments_with_scale, tags_up_to
from .modelspec import model_to_dict, observable_to_dict
from .stages import (MEMORY_POLL, JobMemory, JobTimeout, deadline,
                     exit_with_parent, memory_limit, resident_bytes)

__all__ = ["METHODS", "Worker", "run_case", "verdict", "summarise", "main"]

ROOT = Path(__file__).resolve().parents[1]
#: Integrators in the order a case runs them, with their settings.
METHODS = {"gauss_legendre": dict(n_gauss=14),
           "qmc_vectorized": dict(n_samples=2 ** 14, seed=7),
           "nquad": {}}
#: Extra wall time the parent allows past a job's own deadline.
WATCHDOG_MARGIN = 60.0
#: Seconds a worker that closed its output may take to exit on its own, so
#: that its return code is kept.
CRASH_GRACE = 5.0
DEFAULT_ORACLE_TIMEOUT = 1800.0
#: Resident memory a worker may use, in GB.
DEFAULT_MAX_RSS_GB = 2.0
#: Resident memory an oracle evaluation may use, in GB.
DEFAULT_ORACLE_MAX_RSS_GB = 8.0
#: Memory a run leaves to the rest of the machine, in GB, at most a quarter
#: of the physical memory.
DEFAULT_MIN_AVAILABLE_GB = 8.0
#: Seconds between two checks of available memory while a case waits.
AVAILABLE_POLL = 5.0
GB = 2.0 ** 30
_STDERR_TAIL = 2000


# ---------------------------------------------------------------------------
# The worker process
# ---------------------------------------------------------------------------

def _mb(peak: int) -> float | None:
    """Megabytes, or ``None`` for a job that ended before the first
    sample."""
    return peak / 2**20 if peak else None


class Worker:
    """``python -m validation.worker`` with one job in flight at a time.  A
    worker that gives no answer within the job's deadline plus
    :data:`WATCHDOG_MARGIN` is killed, and so is one whose resident memory,
    with that of any process it started, is above ``max_rss`` bytes; one
    that exits is reported with its return code and stderr.  Each way the
    next job starts a new one.  A result carries ``peak_rss_mb``, the
    largest of the samples taken every
    :data:`~validation.stages.MEMORY_POLL` seconds while the job ran
    (``None`` when it ended before the first)."""

    def __init__(self, package: Path, lowering: str = "validation.lower",
                 extra_path: tuple = (), max_rss: float | None = None):
        self.package, self.lowering = Path(package), lowering
        self.extra_path = tuple(map(str, extra_path))
        self.max_rss = max_rss
        self.proc = None
        self.stderr = None

    def _start(self) -> None:
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [str(self.package), str(ROOT), *self.extra_path]),
            OMP_NUM_THREADS="1", VALIDATION_EXPECT_PACKAGE=str(self.package),
            VALIDATION_PARENT_PID=str(os.getpid()))
        self.stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "validation.worker", "--lowering",
             self.lowering], cwd=ROOT, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.stderr)

    def _lost(self, job: dict, status: str, why: str) -> dict:
        code = None
        if self.proc is not None:
            if status != "crashed" and self.proc.poll() is None:
                self.proc.kill()
            elif self.proc.poll() is None:
                try:
                    self.proc.stdin.close()
                except OSError:
                    pass
                try:
                    self.proc.wait(timeout=CRASH_GRACE)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            code = self.proc.wait()
        tail = ""
        if self.stderr is not None:
            self.stderr.seek(0)
            tail = self.stderr.read().decode(errors="replace")[-_STDERR_TAIL:]
        self.close()
        return {"id": job.get("id"), "ok": False, "status": status,
                "error": why, "returncode": code, "stderr_tail": tail}

    def run(self, job: dict) -> dict:
        if self.proc is None or self.proc.poll() is not None:
            self.close()
            self._start()
        try:
            self.proc.stdin.write((json.dumps(job) + "\n").encode())
            self.proc.stdin.flush()
        except OSError as exc:
            return self._lost(job, "crashed", f"writing the job: {exc}")
        wait = (job.get("timeout") or 3600.0) + WATCHDOG_MARGIN
        end = time.monotonic() + wait
        peak = 0
        with selectors.DefaultSelector() as sel:
            sel.register(self.proc.stdout, selectors.EVENT_READ)
            while not sel.select(timeout=min(MEMORY_POLL, max(
                    0.0, end - time.monotonic()))):
                peak = max(peak, resident_bytes(self.proc.pid))
                if self.max_rss is not None and peak > self.max_rss:
                    return dict(self._lost(
                        job, "memory", f"resident memory above "
                        f"{self.max_rss / GB:.3g} GB"), peak_rss_mb=_mb(peak))
                if time.monotonic() >= end:
                    return dict(self._lost(
                        job, "killed", f"no answer within {wait:g} s"),
                        peak_rss_mb=_mb(peak))
        line = self.proc.stdout.readline()
        if not line:
            return dict(self._lost(job, "crashed", "the worker exited"),
                        peak_rss_mb=_mb(peak))
        try:
            return dict(json.loads(line), peak_rss_mb=_mb(peak))
        except ValueError as exc:
            return dict(self._lost(job, "crashed",
                                   f"a torn result line: {exc}"),
                        peak_rss_mb=_mb(peak))

    def close(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()
            for stream in (self.proc.stdin, self.proc.stdout):
                if not stream.closed:
                    stream.close()
            self.proc = None
        if self.stderr is not None:
            self.stderr.close()
            self.stderr = None


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def job_of(case: Case, method: str, timeout: float) -> dict:
    """The worker job of one case and integrator."""
    return dict(id=f"{case.seed}:{method}", model=model_to_dict(case.model),
                observable=observable_to_dict(case.observable),
                max_weight=case.max_weight, method=method,
                settings=METHODS[method], flags=case.flags, timeout=timeout)


def oracle_of(case: Case, timeout: float | None = None,
              max_rss: float | None = None, memory: dict | None = None
              ) -> tuple:
    """The oracle's values at every tag of the case, and their scale, under
    a deadline (:class:`~validation.stages.JobTimeout` past it) and a limit
    of ``max_rss`` bytes of resident memory
    (:class:`~validation.stages.JobMemory` past it).  ``memory``, when
    given, receives ``peak``, the largest sample of resident memory."""
    tags = tags_up_to(case.model, case.max_weight)
    with deadline(timeout), memory_limit(max_rss) as info:
        try:
            return moments_with_scale(case.model, case.observable, tags)
        finally:
            if memory is not None:
                memory["peak"] = info["peak"]


def features(case: Case) -> list:
    """Labels of what the case exercises, for the coverage table."""
    m, obs = case.model, case.observable
    out = [f"kind:{case.kind}", f"N={m.n_components}",
           f"legs={len(obs.legs)}", f"weight={case.max_weight}"]
    out += [name for name, on in (
        ("source", m.source is not None), ("F", m.quadratic is not None),
        ("G", m.cubic is not None), ("white", m.white is not None),
        ("scalar R", len(set(m.gamma)) == 1),
        ("two times", bool(obs.times)), ("integrated", bool(
            obs.integrate_over)), ("connected", obs.target == "connected"))
        if on]
    out += sorted({f"OU:{blk.spatial.kind}" for blk in m.ou})
    if m.poisson is not None:
        out.append(f"Poisson:{m.poisson.pulse}:{m.poisson.orders}")
    if m.multiplicative is not None:
        out.append(f"multiplicative:{m.multiplicative.interpretation}")
    out += [f"cumulant:{c.channel}" for c in m.cumulants]
    out += [f"flag:{k}" for k, v in case.flags.items() if v and k != "diag_R"]
    return out


_ERROR_VERDICTS = {"package": "refusal", "callback": "callback-error",
                   "validation": "lowering-error"}
_STAGE_VERDICTS = {"expand": "refusal", "propagators": "refusal",
                   "evaluate": "refusal", "lower": "lowering-error"}


def verdict(result: dict, ref: dict, scale: float, method: str,
            c_source: str = "oracle", job: dict | None = None) -> dict:
    """The verdict of a worker result against the oracle's values (see the
    module docstring), with the compared rows of a result that has
    values."""
    status = result.get("status")
    if status == "ok":
        if job is not None:
            ran = (result.get("method"), result.get("settings"),
                   result.get("flags"))
            if ran != (job["method"], job["settings"], job["flags"]):
                return dict(verdict="harness-error",
                            error=f"the worker ran {ran}")
        package = {tuple(t): (v, e) for t, v, e in result["values"]}
        rows = compare_values(ref, package, tolerance(method, c_source),
                              scale=scale)
        worst = max(rows, key=lambda r: r.ratio, default=None)
        return dict(verdict="pass" if all(r.ok for r in rows) else "fail",
                    worst_ratio=worst.ratio if worst else 0.0,
                    worst_tag=list(worst.tag) if worst else None,
                    rows=[[list(r.tag), r.oracle, r.package, r.error,
                           r.allowed, r.ratio] for r in rows])
    if status == "error":
        return dict(verdict=_ERROR_VERDICTS.get(
            result.get("origin"), _STAGE_VERDICTS.get(result.get("stage"),
                                                      "harness-error")))
    if status in ("timeout", "killed", "crashed", "memory"):
        return dict(verdict=status)
    return dict(verdict="harness-error")


_KEPT = ("status", "error", "stage", "origin", "traceback", "seconds",
         "build_seconds", "c_route", "version", "package", "warnings_total",
         "returncode", "stderr_tail", "peak_rss_mb")


def run_case(seed: int, package: str, lowering: str, timeout: float,
             methods: tuple = tuple(METHODS), case: Case | None = None,
             extra_path: tuple = (),
             oracle_timeout: float | None = DEFAULT_ORACLE_TIMEOUT,
             max_rss: float | None = None,
             oracle_max_rss: float | None = None) -> list:
    """Every integrator's record of one case; ``extra_path`` is added to the
    worker's module search path (for a lowering outside this tree),
    ``max_rss`` bytes limit each worker and ``oracle_max_rss`` bytes the
    oracle."""
    case = case or draw(seed)
    base = dict(seed=case.seed, kind=case.kind, features=features(case),
                model=model_to_dict(case.model),
                observable=observable_to_dict(case.observable),
                max_weight=case.max_weight, flags=case.flags)
    t0 = time.perf_counter()
    memory: dict = {}
    lost = None
    try:
        ref, scale = oracle_of(case, oracle_timeout, oracle_max_rss, memory)
    except JobTimeout as exc:
        lost = dict(verdict="oracle-timeout", error=str(exc))
    except JobMemory as exc:
        lost = dict(verdict="oracle-memory", error=str(exc))
    except Exception as exc:  # noqa: BLE001  (recorded, not compared)
        lost = dict(verdict="oracle-error",
                    error=f"{type(exc).__name__}: {exc}")
    base["oracle_seconds"] = time.perf_counter() - t0
    if memory.get("peak"):
        base["oracle_rss_mb"] = memory["peak"] / 2**20
    if lost is not None:
        return [dict(base, method=m, **lost) for m in methods]
    out = []
    worker = Worker(Path(package), lowering, extra_path, max_rss)
    try:
        for method in methods:
            record = dict(base, method=method, settings=METHODS[method])
            try:
                job = job_of(case, method, timeout)
                result = worker.run(job)
                record.update({k: result[k] for k in _KEPT if k in result})
                record.update(verdict(result, ref, scale, method, job=job))
            except Exception as exc:  # noqa: BLE001  (one record, not the run)
                worker.close()
                record.update(verdict="harness-error",
                              error=f"{type(exc).__name__}: {exc}",
                              traceback=traceback.format_exc(limit=-8))
            out.append(record)
    finally:
        worker.close()
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def records_of(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def summarise(records: list) -> dict:
    by_method: dict = collections.defaultdict(collections.Counter)
    worst: dict = {}
    coverage: dict = collections.defaultdict(collections.Counter)
    errors: collections.Counter = collections.Counter()
    for r in records:
        method, v = r["method"], r["verdict"]
        by_method[method][v] += 1
        if v == "pass":
            worst[method] = max(worst.get(method, 0.0), r["worst_ratio"])
        for f in r["features"]:
            coverage[f][v] += 1
        if v not in ("pass", "fail"):
            first = (r.get("error") or "").splitlines()[:1]
            errors[f"{v} | {r.get('stage', '')} | "
                   f"{first[0][:160] if first else ''}"] += 1
    fails = sorted(([r["seed"], r["method"], r["worst_tag"], r["worst_ratio"]]
                    for r in records if r["verdict"] == "fail"),
                   key=lambda x: -x[3])
    return dict(cases=len({r["seed"] for r in records}),
                records=len(records),
                by_method={m: dict(c) for m, c in by_method.items()},
                worst_passing_ratio=worst, fails=fails,
                errors=errors.most_common(),
                coverage={f: dict(c) for f, c in sorted(coverage.items())})


def _print_summary(summary: dict) -> None:
    print(f"{summary['cases']} cases, {summary['records']} records")
    for method, counts in summary["by_method"].items():
        worst = summary["worst_passing_ratio"].get(method, 0.0)
        print(f"  {method:16s} {dict(sorted(counts.items()))}  worst passing "
              f"ratio {worst:.3g}")
    for seed, method, tag, ratio in summary["fails"][:40]:
        print(f"  FAIL seed {seed} {method} tag {tag} ratio {ratio:.3g}")
    for text, n in summary["errors"][:40]:
        print(f"  {n:4d} x {text}")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

class _Progress:
    """Writes a case's records to ``sink`` and prints the count every ten
    cases."""

    def __init__(self, total: int, t0: float, sink):
        self.total, self.t0, self.sink, self.n = total, t0, sink, 0

    def write(self, records: list) -> None:
        self.n += 1
        for record in records:
            self.sink.write(json.dumps(record) + "\n")
        self.sink.flush()
        if self.n % 10 == 0 or self.n == self.total:
            print(f"{self.n}/{self.total} cases, "
                  f"{time.perf_counter() - self.t0:.0f} s", flush=True)


def _lost_records(seed: int, methods: tuple, error: str) -> list:
    return [dict(seed=seed, method=m, features=[], verdict="harness-error",
                 error=error) for m in methods]


def _settle(group: list, breaks: collections.Counter, isolate: set,
            pending: collections.deque, todo: dict) -> list:
    """The cases that failed together when a pool broke.  A break counts for
    a case only when it was the one case in the pool.  A case with fewer
    than two goes back to the front of ``pending``, to run alone; the
    others come back as ``(seed, records)``."""
    alone = len(group) == 1
    out = []
    for seed, why in reversed(group):
        if alone:
            breaks[seed] += 1
        if breaks[seed] < 2:
            isolate.add(seed)
            pending.appendleft((seed, todo[seed]))
            continue
        isolate.discard(seed)
        out.append((seed, _lost_records(
            seed, todo[seed], f"the case killed its pool process twice while "
                              f"it ran alone: {why}")))
    return out


def _plan(args) -> tuple:
    """The records file and the ``{seed: methods}`` still to run; appends
    the invocation to ``run.json``."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "records.jsonl"
    done = {(r["seed"], r["method"]) for r in records_of(path)} if (
        path.exists() and args.resume) else set()
    if not args.resume:
        path.write_text("")
    todo = {s: tuple(m for m in args.methods if (s, m) not in done)
            for s in range(args.start, args.start + args.count)}
    todo = {s: m for s, m in todo.items() if m}
    history = out / "run.json"
    runs = (json.loads(history.read_text()) if history.exists()
            and args.resume else [])
    runs.append({k: v for k, v in vars(args).items() if k != "command"}
                | {"settings": {m: METHODS[m] for m in args.methods}})
    history.write_text(json.dumps(runs, indent=1))
    return path, todo


def limits_of(args) -> dict:
    """``--max-rss-gb`` and ``--oracle-max-rss-gb`` in bytes, ``None`` for a
    limit of 0."""
    return dict(max_rss=args.max_rss_gb * GB if args.max_rss_gb else None,
                oracle_max_rss=(args.oracle_max_rss_gb * GB
                                if args.oracle_max_rss_gb else None))


def _run(args) -> None:
    path, todo = _plan(args)
    limits = limits_of(args)
    with path.open("a") as sink:
        _drive(args, todo, limits, _Progress(len(todo), time.perf_counter(),
                                             sink))
    summary = summarise(records_of(path))
    (path.parent / "summary.json").write_text(json.dumps(summary, indent=1))
    _print_summary(summary)


def _collect(done, running: dict, group: list, isolate: set, todo: dict,
             progress) -> bool:
    """Write the records of the finished futures; a future that failed with
    the pool goes to ``group`` instead.  Whether one did."""
    broken = False
    for fut in done:
        seed = running.pop(fut)
        try:
            records = fut.result()
        except cf.process.BrokenProcessPool as exc:
            broken = True
            group.append((seed, str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001  (a lost case)
            records = _lost_records(seed, todo[seed],
                                    f"{type(exc).__name__}: {exc}")
        isolate.discard(seed)
        progress.write(records)
    return broken


def _drive(args, todo: dict, limits: dict, progress) -> None:
    """Run every case of ``todo`` through a pool, under the admission rule,
    replacing the pool when a process of it dies."""
    per_case = (limits["max_rss"] or 0.0) + (limits["oracle_max_rss"] or 0.0)
    reserve = args.min_available_gb * GB
    pool = _new_pool(args.jobs)
    pending = collections.deque(todo.items())
    running: dict = {}
    breaks: collections.Counter = collections.Counter()
    isolate: set = set()
    group: list = []
    broken = waiting = False
    try:
        while pending or running or group:
            broken = _fill(pool, running, pending, args, limits,
                           per_case, reserve, isolate) or broken
            if not running and not broken:
                if not waiting:
                    print(f"waiting for {reserve / GB:g} GB of available "
                          f"memory", flush=True)
                waiting = True
                time.sleep(AVAILABLE_POLL)
                continue
            waiting = False
            done, _ = cf.wait(running, timeout=None if broken else
                              AVAILABLE_POLL,
                              return_when=cf.ALL_COMPLETED if broken
                              else cf.FIRST_COMPLETED)
            broken = _collect(done, running, group, isolate, todo,
                              progress) or broken
            if broken and not running:
                for seed, records in _settle(group, breaks, isolate,
                                             pending, todo):
                    progress.write(records)
                group, broken = [], False
                _stop_pool(pool)
                pool.shutdown(wait=False, cancel_futures=True)
                if pending:
                    print("a pool process died; starting a new pool",
                          flush=True)
                    pool = _new_pool(args.jobs)
    except BaseException:
        _stop_pool(pool)
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()


def _stop_pool(pool) -> None:
    """Kill the pool's processes and every process they started, so that an
    interrupted run leaves no case running.  Every process is listed before
    the first is killed, so that none is lost to a reparenting, and a pool
    process that has exited is skipped, since its pid may name another
    process by now.  Call it before ``shutdown``, which forgets the
    processes."""
    import psutil

    procs = []
    for each in list((getattr(pool, "_processes", None) or {}).values()):
        if each.exitcode is not None:
            continue
        try:
            proc = psutil.Process(each.pid)
            procs += [proc, *proc.children(recursive=True)]
        except psutil.Error:
            continue
    for proc in procs:
        try:
            proc.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=5)


def _pool_process(parent: int) -> None:
    """Start a pool process: the default ``SIGTERM`` disposition (a forked
    process inherits the run's handler, under which it would outlive the
    pool's ``terminate``), and an end once ``parent``, the run, has gone.
    :func:`_pool_context` forks or spawns, so the run is the parent."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    exit_with_parent(parent)


def _pool_context():
    """Spawn from Python 3.13 (a new process per case needs it), else the
    default start method when it forks or spawns, else spawn: a pool
    process must be the run's child for :func:`_pool_process`."""
    import multiprocessing

    if sys.version_info >= (3, 13):
        return multiprocessing.get_context("spawn")
    ctx = multiprocessing.get_context()
    if ctx.get_start_method() in ("fork", "spawn"):
        return ctx
    return multiprocessing.get_context("spawn")


def _new_pool(jobs: int):
    """The pool of a run (:func:`_pool_process` starts each process)."""
    return cf.ProcessPoolExecutor(max_workers=jobs, mp_context=_pool_context(),
                                  initializer=_pool_process,
                                  initargs=(os.getpid(),),
                                  **_fresh_processes())


def _on_term(signum, frame):
    raise SystemExit(128 + signum)


def available_bytes() -> int:
    """Memory the machine has available for new processes, in bytes."""
    import psutil

    return psutil.virtual_memory().available


def physical_bytes() -> int:
    """The machine's physical memory, in bytes."""
    import psutil

    return psutil.virtual_memory().total


def default_reserve_gb() -> float:
    """:data:`DEFAULT_MIN_AVAILABLE_GB`, or a quarter of the physical memory
    when that is less."""
    return min(DEFAULT_MIN_AVAILABLE_GB, physical_bytes() / 4 / GB)


def admits(available: float, running: int, per_case: float,
           reserve: float) -> bool:
    """Whether a case may start.  With none running, once ``reserve`` bytes
    are available.  Otherwise each running case and the new one count at
    their limits, ``per_case`` bytes, whatever they use now, and what is
    available less that must still be ``reserve``."""
    if running == 0:
        return available >= reserve
    return available - (running + 1) * per_case >= reserve


def _fresh_processes() -> dict:
    """A new process for each case where the pool can do it (Python 3.13
    and later)."""
    return dict(max_tasks_per_child=1) if sys.version_info >= (3, 13) else {}


def _fill(pool, running: dict, pending: collections.deque, args,
          limits: dict, per_case: float, reserve: float,
          isolate: frozenset | set = frozenset()) -> bool:
    """Start cases from ``pending`` until ``args.jobs`` run, while
    :func:`admits` lets each start; a case it does not admit waits for the
    next call.  A case in ``isolate`` starts only in an empty pool, and
    nothing starts beside it.  Whether the pool turned out broken, in which
    case the case goes back to ``pending``."""
    while pending and len(running) < args.jobs:
        if any(seed in isolate for seed in running.values()):
            return False
        seed, methods = pending[0]
        if seed in isolate and running:
            return False
        if (reserve or per_case) and not admits(
                available_bytes(), len(running), per_case, reserve):
            return False
        pending.popleft()
        try:
            fut = pool.submit(run_case, seed, args.package, args.lowering,
                              args.timeout, methods,
                              extra_path=tuple(args.extra_path),
                              oracle_timeout=args.oracle_timeout, **limits)
        except cf.process.BrokenProcessPool:
            pending.appendleft((seed, methods))
            return True
        running[fut] = seed
    return False


def _check_memory_args(parser, args) -> None:
    """psutil is there, the limits are 0 or more, and the reserve (resolved
    to its default when not given) is below the physical memory."""
    try:
        import psutil  # noqa: F401
    except ImportError:
        parser.error("the fuzzer reads memory with psutil: "
                     "pip install -e '.[dev]'")
    for name in ("max_rss_gb", "oracle_max_rss_gb", "min_available_gb"):
        value = getattr(args, name, None)
        if value is not None and not value >= 0:
            parser.error(f"--{name.replace('_', '-')} must be 0 or more, "
                         f"got {value:g}")
    if args.command != "run":
        return
    if args.min_available_gb is None:
        args.min_available_gb = default_reserve_gb()
    if args.min_available_gb * GB >= physical_bytes():
        parser.error(f"--min-available-gb {args.min_available_gb:g} is not "
                     f"below the physical memory, "
                     f"{physical_bytes() / GB:.3g} GB")


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m validation.fuzz")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "summary", "shrink"):
        p = sub.add_parser(name)
        p.add_argument("out")
        p.add_argument("--package", default=str(ROOT / "src"))
        p.add_argument("--lowering", default="validation.lower")
        p.add_argument("--extra-path", nargs="*", default=[])
        p.add_argument("--timeout", type=float, default=240.0)
        p.add_argument("--oracle-timeout", type=float,
                       default=DEFAULT_ORACLE_TIMEOUT)
    for name in ("run", "shrink"):
        sub.choices[name].add_argument("--max-rss-gb", type=float,
                                       default=DEFAULT_MAX_RSS_GB)
        sub.choices[name].add_argument("--oracle-max-rss-gb", type=float,
                                       default=DEFAULT_ORACLE_MAX_RSS_GB)
    run = sub.choices["run"]
    run.add_argument("--start", type=int, default=0)
    run.add_argument("--count", type=int, default=300)
    run.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)
                                                     - 4))
    run.add_argument("--min-available-gb", type=float, default=None)
    run.add_argument("--methods", nargs="+", default=list(METHODS),
                     choices=list(METHODS))
    run.add_argument("--resume", action="store_true")
    small = sub.choices["shrink"]
    small.add_argument("--seed", type=int, required=True)
    small.add_argument("--method", required=True, choices=list(METHODS))
    small.add_argument("--budget", type=int, default=200)
    args = parser.parse_args(argv)
    if args.command in ("run", "shrink"):
        _check_memory_args(parser, args)
    if args.command == "run":
        if threading.current_thread() is threading.main_thread():
            previous = signal.signal(signal.SIGTERM, _on_term)
            try:
                _run(args)
            finally:
                signal.signal(signal.SIGTERM, previous)
        else:
            _run(args)
    elif args.command == "summary":
        _print_summary(summarise(records_of(Path(args.out) / "records.jsonl")))
    else:
        from .shrink import main as shrink_main
        shrink_main(args)


if __name__ == "__main__":
    main()
