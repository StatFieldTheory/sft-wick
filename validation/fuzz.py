r"""Random differential runs of the package against the oracle.

``python -m validation.fuzz run DIR --count 300`` draws cases with
:func:`validation.draw.draw` from consecutive seeds, computes the oracle in
this process and sends every integrator's job to ``python -m
validation.worker`` in a subprocess, so this process never loads sft_wick.
It writes ``DIR/records.jsonl`` (one record per case and integrator),
``DIR/summary.json`` and ``DIR/run.json`` (the arguments of every
invocation).  ``--resume`` runs only the (seed, integrator) pairs that have
no record.  ``python -m validation.fuzz summary DIR`` prints the summary
again; ``python -m validation.fuzz shrink DIR --seed S --method M`` reduces
a failing case (:mod:`validation.shrink`).

``--package DIR`` names the directory that holds the ``sft_wick`` under
test (default this tree's ``src``), ``--lowering`` the module that lowers
onto it, and ``--extra-path`` directories the worker also searches, for a
lowering outside this tree.

A record's ``verdict`` is one of:

* ``pass``: every tag within tolerance; ``fail``: a tag outside it;
* ``refusal``: the package raised;
* ``callback-error``: an oracle callable that the package called (the closed
  form of C, a Campbell cumulant) raised, which a bad call by the package
  or a fault in the callable can cause;
* ``lowering-error``: the lowering raised before reaching the package;
* ``timeout``: past the job's deadline; ``killed``: the worker gave no
  answer within the deadline plus :data:`WATCHDOG_MARGIN`; ``crashed``: the
  worker exited (its return code and the end of its stderr are kept);
* ``oracle-error``, ``oracle-timeout``: the oracle raised or ran past
  ``--oracle-timeout``, so nothing was compared;
* ``harness-error``: anything else, including a worker that ran another
  method, settings or flags than the job asked for.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import selectors
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from .compare import compare_values, tolerance
from .draw import Case, draw
from .hierarchy import moments_with_scale, tags_up_to
from .modelspec import model_to_dict, observable_to_dict
from .stages import JobTimeout, deadline

__all__ = ["METHODS", "Worker", "run_case", "verdict", "summarise", "main"]

ROOT = Path(__file__).resolve().parents[1]
#: Integrators in the order a case runs them, with their settings.
METHODS = {"gauss_legendre": dict(n_gauss=14),
           "qmc_vectorized": dict(n_samples=2 ** 14, seed=7),
           "nquad": {}}
#: Extra wall time the parent allows past a job's own deadline.
WATCHDOG_MARGIN = 60.0
DEFAULT_ORACLE_TIMEOUT = 1800.0
_STDERR_TAIL = 2000


# ---------------------------------------------------------------------------
# The worker process
# ---------------------------------------------------------------------------

class Worker:
    """``python -m validation.worker`` with one job in flight at a time.  A
    worker that gives no answer within the job's deadline plus
    :data:`WATCHDOG_MARGIN` is killed; one that exits is reported with its
    return code and stderr.  Either way the next job starts a new one."""

    def __init__(self, package: Path, lowering: str = "validation.lower",
                 extra_path: tuple = ()):
        self.package, self.lowering = Path(package), lowering
        self.extra_path = tuple(map(str, extra_path))
        self.proc = None
        self.stderr = None

    def _start(self) -> None:
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [str(self.package), str(ROOT), *self.extra_path]),
            OMP_NUM_THREADS="1", VALIDATION_EXPECT_PACKAGE=str(self.package))
        self.stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "validation.worker", "--lowering",
             self.lowering], cwd=ROOT, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.stderr)

    def _lost(self, job: dict, status: str, why: str) -> dict:
        code = None
        if self.proc is not None:
            if status == "killed" and self.proc.poll() is None:
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
        with selectors.DefaultSelector() as sel:
            sel.register(self.proc.stdout, selectors.EVENT_READ)
            ready = sel.select(timeout=wait)
        if not ready:
            return self._lost(job, "killed", f"no answer within {wait:g} s")
        line = self.proc.stdout.readline()
        if not line:
            return self._lost(job, "crashed", "the worker exited")
        try:
            return json.loads(line)
        except ValueError as exc:
            return self._lost(job, "crashed", f"a torn result line: {exc}")

    def close(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()
            for stream in (self.proc.stdin, self.proc.stdout):
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


def oracle_of(case: Case, timeout: float | None = None) -> tuple:
    """The oracle's values at every tag of the case, and their scale, under
    a deadline (:class:`~validation.stages.JobTimeout` past it)."""
    tags = tags_up_to(case.model, case.max_weight)
    with deadline(timeout):
        return moments_with_scale(case.model, case.observable, tags)


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
    if status in ("timeout", "killed", "crashed"):
        return dict(verdict=status)
    return dict(verdict="harness-error")


_KEPT = ("status", "error", "stage", "origin", "traceback", "seconds",
         "build_seconds", "c_route", "version", "package", "warnings_total",
         "returncode", "stderr_tail")


def run_case(seed: int, package: str, lowering: str, timeout: float,
             methods: tuple = tuple(METHODS), case: Case | None = None,
             extra_path: tuple = (),
             oracle_timeout: float | None = DEFAULT_ORACLE_TIMEOUT) -> list:
    """Every integrator's record of one case; ``extra_path`` is added to the
    worker's module search path (for a lowering outside this tree)."""
    case = case or draw(seed)
    base = dict(seed=case.seed, kind=case.kind, features=features(case),
                model=model_to_dict(case.model),
                observable=observable_to_dict(case.observable),
                max_weight=case.max_weight, flags=case.flags)
    t0 = time.perf_counter()
    try:
        ref, scale = oracle_of(case, oracle_timeout)
    except JobTimeout as exc:
        return [dict(base, method=m, verdict="oracle-timeout", error=str(exc),
                     oracle_seconds=time.perf_counter() - t0)
                for m in methods]
    except Exception as exc:  # noqa: BLE001  (recorded, not compared)
        return [dict(base, method=m, verdict="oracle-error",
                     error=f"{type(exc).__name__}: {exc}",
                     oracle_seconds=time.perf_counter() - t0)
                for m in methods]
    base["oracle_seconds"] = time.perf_counter() - t0
    out = []
    worker = Worker(Path(package), lowering, extra_path)
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

def _run(args) -> None:
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
    t0 = time.perf_counter()
    pool = cf.ProcessPoolExecutor(max_workers=args.jobs)
    try:
        with path.open("a") as sink:
            futures = {pool.submit(run_case, s, args.package, args.lowering,
                                   args.timeout, methods,
                                   extra_path=tuple(args.extra_path),
                                   oracle_timeout=args.oracle_timeout): s
                       for s, methods in todo.items()}
            for n, fut in enumerate(cf.as_completed(futures), 1):
                seed = futures[fut]
                try:
                    records = fut.result()
                except Exception as exc:  # noqa: BLE001  (a lost case)
                    records = [dict(seed=seed, method=m, features=[],
                                    verdict="harness-error",
                                    error=f"{type(exc).__name__}: {exc}")
                               for m in todo[seed]]
                for record in records:
                    sink.write(json.dumps(record) + "\n")
                sink.flush()
                if n % 10 == 0 or n == len(futures):
                    print(f"{n}/{len(futures)} cases, "
                          f"{time.perf_counter() - t0:.0f} s", flush=True)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    summary = summarise(records_of(path))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    _print_summary(summary)


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
    run = sub.choices["run"]
    run.add_argument("--start", type=int, default=0)
    run.add_argument("--count", type=int, default=300)
    run.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)
                                                     - 4))
    run.add_argument("--methods", nargs="+", default=list(METHODS),
                     choices=list(METHODS))
    run.add_argument("--resume", action="store_true")
    small = sub.choices["shrink"]
    small.add_argument("--seed", type=int, required=True)
    small.add_argument("--method", required=True, choices=list(METHODS))
    small.add_argument("--budget", type=int, default=200)
    args = parser.parse_args(argv)
    if args.command == "run":
        _run(args)
    elif args.command == "summary":
        _print_summary(summarise(records_of(Path(args.out) / "records.jsonl")))
    else:
        from .shrink import main as shrink_main
        shrink_main(args)


if __name__ == "__main__":
    main()
