"""The random cases of the fuzzer (``validation/draw.py``), the fuzzer
(``validation/fuzz.py``) and the shrinker (``validation/shrink.py``) on a
fixed set of seeds.

The fuzz runs themselves are evidence, not tests.  Here:

* twenty seeds chosen to cover every feature the draw produces, each
  among the fastest on Gauss-Legendre: the case each seed draws is pinned
  by a digest, the oracle's value at its largest non-trivial tag is pinned,
  and the package agrees with the oracle at the declared tolerance;
* the flags a drawn model admits hold in the closed-form C (diagonal when
  ``diag_C`` is on and not otherwise, ``c·I`` under ``iso_C``), and each
  condition of isotropy is needed;
* one case with its flags on runs end to end through the worker subprocess
  on all three integrators;
* a lowering with a planted fault (the cubic drift scaled by 1.001) fails
  exactly the cases with a cubic drift, and the shrinker reduces one of
  them to a case whose channels are the source and the cubic drift;
* the verdict of every worker status, origin and stage, the summary's
  counts, and the watchdog: a worker whose job is stuck in compiled code is
  killed, and one that exits is reported with its return code;
* the memory limits: a worker past its limit is killed, an oracle past its
  limit raises, both reach the records through a run, a case starts only
  when every running case and the new one fit under their limits with the
  reserve left, a run waits for the reserve before its first case and ends
  once every case has run, a run stopped by ``SIGTERM`` or killed outright
  leaves no case running, and a pool process that dies is replaced, the
  case that killed it recorded and the case started beside it run again
  alone; the shrinker runs under the same two limits.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import itertools
import json
import threading
import time

import numpy as np
import pytest

import validation.lower as lw
import validation.lowering_parts as lp
from validation.compare import compare_values, tolerance
from validation.draw import admissible_flags, draw
import validation.fuzz as fz
import validation.stages as stages
from validation.fuzz import METHODS, ROOT, run_case, summarise, verdict
from validation.shrink import shrink
from validation.stages import (JobMemory, JobTimeout, memory_limit,
                               resident_bytes)
from validation.hierarchy import moments_with_scale, tags_up_to
from validation.modelspec import (Kernel, ModelSpec, OUBlock, model_to_dict,
                                  observable_to_dict)

#: seed -> (digest of the drawn case, the oracle's largest non-trivial tag,
#: its value)
PINNED = {
    3: ("f486157610d0e0a4", (0, 0, 1), 0.19751212714378039),
    31: ("290e32240af2453d", (0, 1, 0, 0, 0, 1), -0.02259668903510387),
    33: ("bdfa5dbd6e102db8", (0, 0, 0, 2, 0), 0.07739741254075931),
    36: ("b60699ae6c1f8f99", (0, 0, 1, 0, 0), 0.2361350531869894),
    90: ("e6029195f4146022", (1, 0, 0, 1, 0), 0.3761686541778702),
    111: ("4387278694d644a1", (1,), 0.029027832470631915),
    114: ("6b58e88f620cc72d", (0, 0, 1, 0, 0), -0.2778670955799529),
    115: ("f952b7a9178dc2f2", (0, 0, 1), -0.003975471795966948),
    126: ("306ff7c7f011cfcf", (0, 0, 1, 0, 0), 0.07904396931756681),
    134: ("8786957c56362817", (0, 0, 0, 2), 0.042946599314231594),
    146: ("736b8708e2b53ef2", (0, 0, 0, 1, 0), 0.02767548481651758),
    151: ("c7e54699309b24c4", (1, 1, 0), 0.04824125826703164),
    152: ("f9c85876f948a8d6", (0, 0, 0, 2, 0), 0.039645434810868345),
    172: ("b2c2dfd887fd871e", (0, 1, 0, 0, 1), -0.02450277498453415),
    202: ("309b9dba6e0c453f", (0, 0, 1), 0.03580627821933865),
    207: ("0c20247f8abf090d", (1, 0, 0, 1, 0), -0.06389580956914867),
    214: ("58a1903abb79e40c", (0, 0, 0, 0, 1), -0.07658215593832465),
    233: ("b42fcb98fcefb03a", (0, 0, 1, 0, 0), 0.1434404035667054),
    266: ("d945d1c9c7770989", (1, 0, 0, 1, 0, 0), -0.03194929137434738),
    292: ("b969fec28dd762bf", (2, 0, 0, 0), 0.01713158569089783),
}


def _digest(case) -> str:
    blob = json.dumps([model_to_dict(case.model),
                       observable_to_dict(case.observable), case.max_weight,
                       case.flags], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@pytest.mark.parametrize("seed", sorted(PINNED))
def test_VZ1_pinned_cases_come_back_and_the_package_agrees(seed):
    digest, tag, value = PINNED[seed]
    case = draw(seed)
    assert _digest(case) == digest
    tags = tags_up_to(case.model, case.max_weight)
    ref, scale = moments_with_scale(case.model, case.observable, tags)
    assert ref[tag] == pytest.approx(value, rel=1e-12, abs=1e-300)
    got = lw.Lowered(case.model, case.observable, case.max_weight,
                     case.flags).evaluate("gauss_legendre",
                                          **METHODS["gauss_legendre"])
    rows = compare_values(ref, got, tolerance("gauss_legendre", "oracle"),
                          scale=scale)
    assert all(r.ok for r in rows), [r for r in rows if not r.ok]


def test_VZ2_the_pinned_seeds_cover_every_drawn_feature():
    from validation.fuzz import features

    pinned = set().union(*(features(draw(s)) for s in PINNED))
    drawn = set().union(*(features(draw(s)) for s in range(300)))
    assert drawn - pinned == set()
    assert {"kind:general", "kind:diagonal", "kind:isotropic", "G",
            "flag:iso_C", "connected", "integrated", "two times"} <= pinned


def _c_at_random_points(case, rng, n=6):
    """The closed-form C at ``n`` random point pairs, coincident points and
    coincident times included."""
    model = case.model
    d = len(case.observable.positions[0][1])
    x = rng.uniform(-1, 1, size=(n, d))
    y = rng.uniform(-1, 1, size=(n, d))
    y[0] = x[0]
    t1 = model.t_min + rng.uniform(0.1, 1.5, size=n)
    t2 = model.t_min + rng.uniform(0.1, 1.5, size=n)
    t2[1] = t1[1]
    return lp.closed_form_c(model)(x, t1, y, t2)


def test_VZ3_the_admissible_flags_hold_in_the_closed_form():
    """``diag_C`` on: every off-diagonal entry of C is 0; off: some entry is
    not.  ``iso_C`` on: C is ``c·I``.  Over 300 draws, then on hand-built
    models where one condition of isotropy fails at a time."""
    rng = np.random.default_rng(3)
    counts = {"diag_C": 0, "not diag_C": 0, "iso_C": 0}
    for seed in range(300):
        case = draw(seed)
        C = _c_at_random_points(case, rng)
        off = ~np.eye(case.model.n_components, dtype=bool)
        if case.flags["diag_C"]:
            counts["diag_C"] += 1
            assert np.all(C[:, off] == 0.0), seed
        elif case.model.n_components > 1:
            counts["not diag_C"] += 1
            assert np.any(C[:, off] != 0.0), seed
        if case.flags["iso_C"]:
            counts["iso_C"] += 1
            diag = np.diagonal(C, axis1=1, axis2=2)
            np.testing.assert_allclose(diag, diag[:, :1] * np.ones_like(diag),
                                       rtol=1e-13, atol=0.0)
    assert min(counts.values()) >= 30, counts
    block = OUBlock(0.5, 0.8, Kernel("gaussian", (0.9,)), (1.0, 0.0))
    other = OUBlock(0.5, 0.8, Kernel("gaussian", (0.9,)), (0.0, 1.0))
    base = ModelSpec(gamma=(1.0, 1.0), ou=(block, other),
                     white=((0.1, 0.0), (0.0, 0.1)), t_min=0.2)
    assert admissible_flags(base)["iso_C"]
    for broken in (
            dataclasses.replace(base, white=((0.1, 0.0), (0.0, 0.3))),
            dataclasses.replace(base, gamma=(1.0, 1.2)),
            dataclasses.replace(base, ou=(block, dataclasses.replace(
                other, sigma_t=0.9))),
            dataclasses.replace(base, ou=(block,))):
        flags = admissible_flags(broken)
        assert flags["diag_C"] and not flags["iso_C"], broken


def test_VZ4_one_case_end_to_end_through_the_worker():
    """Seed 72 is isotropic, with every flag on and seven non-zero
    interacting tags; the verdict checks that the worker ran the method,
    settings and flags of each job."""
    case = draw(72)
    assert all(case.flags.values())
    records = run_case(72, str(ROOT / "src"), "validation.lower", 120.0)
    assert [r["method"] for r in records] == list(METHODS)
    for r in records:
        assert r["verdict"] == "pass", r
        assert r["package"] == str(ROOT / "src" / "sft_wick")
        interacting = [row for row in r["rows"] if sum(row[0]) and row[1]]
        assert len(interacting) >= 5 and 0.0 < r["worst_ratio"] < 1.0


PLANTED = '''
"""validation.lower with the cubic drift scaled by 1.001."""
import dataclasses

import numpy as np

from validation import lower

sw = lower.sw


class Lowered(lower.Lowered):
    def __init__(self, model, obs, *args, **kwargs):
        if model.cubic is not None:
            model = dataclasses.replace(
                model, cubic=np.asarray(model.cubic) * 1.001)
        super().__init__(model, obs, *args, **kwargs)
'''


def test_VZ5_a_planted_fault_is_found_and_shrunk_to_its_channels(tmp_path):
    (tmp_path / "planted_lowering.py").write_text(PLANTED)
    package, extra = str(ROOT / "src"), (tmp_path,)
    verdicts = {}
    for seed in range(8):
        (record,) = run_case(seed, package, "planted_lowering", 120.0,
                             ("gauss_legendre",), extra_path=extra)
        verdicts[seed] = (record["verdict"], "G" in record["features"])
    assert {s for s, (v, g) in verdicts.items() if v == "fail"} == {
        s for s, (v, g) in verdicts.items() if g} == {1, 7}
    small, steps, verdict = shrink(draw(7), "gauss_legendre", package,
                                   "planted_lowering", 120.0,
                                   extra_path=extra)
    assert small.model.channels() == ("h", "G")
    assert verdict["verdict"] == "fail" and sum(verdict["worst_tag"]) >= 1
    assert "drop quadratic" in steps and "drop component 0" in steps
    # Seed 7 reads component 1 of three; with component 0 gone the leg reads
    # component 0 of two.
    assert draw(7).model.n_components == 3
    assert [leg.comp for leg in draw(7).observable.legs] == [1]
    assert small.model.n_components == 2
    assert [leg.comp for leg in small.observable.legs] == [0]


def test_VZ6_every_worker_result_gets_its_verdict(monkeypatch):
    ref, scale = {(0,): 0.5, (1,): 0.0}, 0.5
    job = dict(method="gauss_legendre", settings=METHODS["gauss_legendre"],
               flags={"diag_R": True})
    ok = dict(status="ok", values=[[[0], 0.5, 0.0]], method=job["method"],
              settings=job["settings"], flags=job["flags"])
    assert verdict(ok, ref, scale, "gauss_legendre", job=job)["verdict"] == (
        "pass")
    off = dict(ok, values=[[[0], 0.5 + 1e-6, 0.0]])
    assert verdict(off, ref, scale, "gauss_legendre")["verdict"] == "fail"
    for changed in (dict(flags={}), dict(settings={}), dict(method="nquad")):
        assert verdict(dict(ok, **changed), ref, scale, "gauss_legendre",
                       job=job)["verdict"] == "harness-error", changed
    errors = {("package", "evaluate"): "refusal",
              ("callback", "evaluate"): "callback-error",
              ("validation", "lower"): "lowering-error",
              ("other", "expand"): "refusal",
              ("other", "lower"): "lowering-error",
              ("other", "job"): "harness-error",
              (None, "parse"): "harness-error"}
    for (origin, stage), want in errors.items():
        result = dict(status="error", origin=origin, stage=stage)
        assert verdict(result, ref, scale, "nquad")["verdict"] == want
    for status in ("timeout", "killed", "crashed", "memory"):
        assert verdict(dict(status=status), ref, scale, "nquad")[
            "verdict"] == status
    assert verdict(dict(status="new"), ref, scale, "nquad")["verdict"] == (
        "harness-error")
    for raised, want in ((ValueError("no"), "oracle-error"),
                         (JobTimeout("slow"), "oracle-timeout"),
                         (JobMemory("big"), "oracle-memory")):
        def oracle(case, *args, raised=raised):
            raise raised
        monkeypatch.setattr(fz, "oracle_of", oracle)
        records = run_case(0, str(ROOT / "src"), "validation.lower", 60.0)
        assert [r["verdict"] for r in records] == [want] * len(METHODS)


def test_VZ7_the_summary_counts_the_records():
    def record(seed, method, v, ratio=0.0, error=None):
        return dict(seed=seed, method=method, verdict=v, worst_ratio=ratio,
                    worst_tag=[1], features=["F"], error=error)

    records = [record(0, "nquad", "pass", 0.25), record(1, "nquad", "pass", 0.5),
               record(2, "nquad", "fail", 30.0),
               record(0, "gauss_legendre", "timeout", error="no result"),
               record(1, "gauss_legendre", "fail", 3.0)]
    summary = summarise(records)
    assert summary["cases"] == 3 and summary["records"] == 5
    assert summary["by_method"] == {"nquad": {"pass": 2, "fail": 1},
                                    "gauss_legendre": {"timeout": 1,
                                                       "fail": 1}}
    assert summary["worst_passing_ratio"] == {"nquad": 0.5}
    assert summary["fails"] == [[2, "nquad", [1], 30.0],
                                [1, "gauss_legendre", [1], 3.0]]
    assert summary["errors"] == [("timeout |  | no result", 1)]
    assert summary["coverage"]["F"] == {"pass": 2, "fail": 2, "timeout": 1}


STUCK = '''
import os, signal, time
from validation import lower

sw = lower.sw
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})


class Lowered(lower.Lowered):
    def evaluate(self, method, *args, **kwargs):
        if os.environ.get("PLANTED") == "exit":
            os._exit(3)
        if os.environ.get("PLANTED") == "sysexit":
            raise SystemExit(5)
        time.sleep(30.0)
'''


def test_VZ8_a_worker_that_hangs_is_killed_and_one_that_exits_is_named(
        tmp_path, monkeypatch):
    """The worker's alarm handler runs in its main thread between bytecodes,
    so it cannot stop a job stuck in one long call of compiled code, here a
    sleep with SIGALRM blocked in that thread: the parent's watchdog kills
    it.  A worker that exits is reported with its return code, at once or
    through the interpreter's shutdown."""
    (tmp_path / "stuck_lowering.py").write_text(STUCK)
    monkeypatch.setattr(fz, "WATCHDOG_MARGIN", 1.0)
    case = draw(0)
    worker = fz.Worker(ROOT / "src", "stuck_lowering", (tmp_path,))
    try:
        start = time.perf_counter()
        result = worker.run(fz.job_of(case, "gauss_legendre", 0.5))
        assert result["status"] == "killed"
        assert time.perf_counter() - start < 15.0
        # a long deadline: a new worker has to start before it exits
        monkeypatch.setenv("PLANTED", "exit")
        result = worker.run(fz.job_of(case, "gauss_legendre", 60.0))
        assert (result["status"], result["returncode"]) == ("crashed", 3)
        monkeypatch.setenv("PLANTED", "sysexit")
        result = worker.run(fz.job_of(case, "gauss_legendre", 60.0))
        assert (result["status"], result["returncode"]) == ("crashed", 5)
    finally:
        worker.close()


def test_VZ9_resume_runs_only_the_missing_pairs(tmp_path):
    """A resume with a second integrator adds that integrator's records and
    nothing else, and ``run.json`` keeps both invocations."""
    out = tmp_path / "run"
    base = [str(out), "--count", "2", "--jobs", "2", "--timeout", "60",
            "--min-available-gb", "0"]
    fz.main(["run", *base, "--methods", "gauss_legendre"])
    fz.main(["run", *base, "--methods", "gauss_legendre", "qmc_vectorized",
             "--resume"])
    records = fz.records_of(out / "records.jsonl")
    pairs = sorted((r["seed"], r["method"]) for r in records)
    assert pairs == [(0, "gauss_legendre"), (0, "qmc_vectorized"),
                     (1, "gauss_legendre"), (1, "qmc_vectorized")]
    runs = json.loads((out / "run.json").read_text())
    assert [r["methods"] for r in runs] == [
        ["gauss_legendre"], ["gauss_legendre", "qmc_vectorized"]]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["records"] == 4


HUNGRY = '''
import os, time
import numpy as np
from validation import lower

sw = lower.sw


class Lowered(lower.Lowered):
    def evaluate(self, method, *args, **kwargs):
        size = int(open(os.environ["PLANTED_SIZE"]).read())
        if not size:
            return {}
        block = np.ones(size, dtype=np.uint8)
        time.sleep(30.0)
        return {(0,): (float(block[0]), 0.0)}
'''


def test_VZ10_a_worker_past_its_memory_limit_is_killed(tmp_path,
                                                         monkeypatch):
    """The parent reads the worker's resident memory while a job runs and
    kills it past the limit, long before the job's deadline."""
    (tmp_path / "hungry_lowering.py").write_text(HUNGRY)
    size = tmp_path / "size"
    size.write_text("0")
    monkeypatch.setattr(fz, "MEMORY_POLL", 0.05)
    monkeypatch.setenv("PLANTED_SIZE", str(size))
    case = draw(0)
    worker = fz.Worker(ROOT / "src", "hungry_lowering", (tmp_path,))
    try:
        assert worker.run(fz.job_of(case, "gauss_legendre", 60.0))[
            "status"] == "ok"
        limit = resident_bytes(worker.proc.pid) + 200 * 2**20
        worker.max_rss = limit
        size.write_text(str(600 * 2**20))
        start = time.perf_counter()
        result = worker.run(fz.job_of(case, "qmc_vectorized", 60.0))
        assert result["status"] == "memory", result
        assert time.perf_counter() - start < 20.0
        assert result["peak_rss_mb"] > limit / 2**20
        assert worker.proc is None
    finally:
        worker.close()


def _grow(step: int, steps: int) -> list:
    blocks = []
    for _ in range(steps):
        blocks.append(np.ones(step, dtype=np.uint8))
        time.sleep(0.05)
    return blocks


def test_VZ11_an_oracle_past_its_memory_limit_raises(monkeypatch):
    """In the main thread the limit raises :class:`JobMemory` and keeps the
    largest sample; in another thread it only measures; a step that stays
    below the limit returns."""
    base = resident_bytes(children=False)
    limit = base + 150 * 2**20
    with pytest.raises(JobMemory):
        with memory_limit(limit, poll=0.02) as info:
            _grow(20 * 2**20, 40)
    assert info["enforced"] and info["peak"] > limit
    with memory_limit(resident_bytes(children=False) + 400 * 2**20,
                      poll=0.02) as info:
        _grow(10 * 2**20, 5)
    assert info["enforced"]
    # The sample at entry reads 0 and every later one more; freed pages can
    # stay resident and be reused, so the real memory need not grow.
    samples = itertools.count()
    monkeypatch.setattr(stages, "resident_bytes",
                        lambda pid=None, children=True: next(samples))
    seen = {}

    def other():
        with memory_limit(0, poll=0.02) as inner:
            time.sleep(0.2)
        seen.update(inner)

    thread = threading.Thread(target=other)
    thread.start()
    thread.join()
    assert seen["enforced"] is False and seen["peak"] > 0
    with memory_limit(None) as off:
        pass
    assert off == {"peak": None, "enforced": False}


def test_VZ12_a_run_waits_for_the_reserve_and_ends(tmp_path, monkeypatch,
                                                    capsys):
    """With less than ``--min-available-gb`` available the first case waits,
    and starts once the memory is there; a resume with nothing left ends
    even while memory is short.  A reserve at or above the physical memory
    is refused, and so is a negative or undefined limit; the default
    reserve, at most a quarter of the physical memory, reaches the run."""
    readings = iter([0, 0])
    monkeypatch.setattr(fz, "available_bytes",
                        lambda: next(readings, 2 ** 40))
    monkeypatch.setattr(fz, "AVAILABLE_POLL", 0.01)
    out = tmp_path / "run"
    base = ["run", str(out), "--count", "1", "--jobs", "1", "--timeout",
            "60", "--methods", "gauss_legendre"]
    fz.main([*base, "--min-available-gb", "1"])
    assert capsys.readouterr().out.count("waiting for 1 GB") == 1
    records = fz.records_of(out / "records.jsonl")
    assert [(r["seed"], r["verdict"]) for r in records] == [(0, "pass")]
    assert records[0]["oracle_rss_mb"] > 0
    runs = json.loads((out / "run.json").read_text())
    assert (runs[0]["max_rss_gb"], runs[0]["oracle_max_rss_gb"],
            runs[0]["min_available_gb"]) == (
        fz.DEFAULT_MAX_RSS_GB, fz.DEFAULT_ORACLE_MAX_RSS_GB, 1.0)
    monkeypatch.setattr(fz, "available_bytes", lambda: 0)
    fz.main([*base, "--min-available-gb", "1", "--resume"])
    assert "waiting" not in capsys.readouterr().out
    assert json.loads((out / "summary.json").read_text())["records"] == 1
    monkeypatch.setattr(fz, "physical_bytes", lambda: 8 * fz.GB)
    fz.main([*base, "--resume"])
    runs = json.loads((out / "run.json").read_text())
    assert runs[-1]["min_available_gb"] == 2.0
    for bad in (["--min-available-gb", "8"], ["--max-rss-gb", "-1"],
                ["--oracle-max-rss-gb", "nan"]):
        with pytest.raises(SystemExit):
            fz.main([*base, *bad, "--resume"])
    for total, reserve in ((8 * fz.GB, 2.0), (96 * fz.GB, 8.0)):
        monkeypatch.setattr(fz, "physical_bytes", lambda total=total: total)
        assert fz.default_reserve_gb() == reserve


class _Pool:
    """Records what a run would submit, and runs nothing."""

    def __init__(self):
        self.calls = []

    def submit(self, fn, seed, *args, **kwargs):
        self.calls.append((seed, kwargs))
        return object()


def test_VZ13_a_case_starts_only_when_every_limit_fits(monkeypatch):
    """Every running case and the new one count at their limits: with 25 GB
    available, 10 GB per case and a reserve of 4 GB, two cases start and
    the third waits; nothing starts while memory is short, with no reserve
    either, and with memory to spare the cases start up to ``--jobs``."""
    G = fz.GB
    assert fz.admits(1 * G, 0, 10 * G, 1 * G)
    assert not fz.admits(0.5 * G, 0, 10 * G, 1 * G)
    assert fz.admits(38 * G, 2, 10 * G, 8 * G)
    assert not fz.admits(37 * G, 2, 10 * G, 8 * G)
    args = argparse.Namespace(jobs=3, package="p", lowering="l", timeout=1.0,
                              extra_path=[], oracle_timeout=1.0)
    limits = dict(max_rss=2 * G, oracle_max_rss=8 * G)
    pool, running = _Pool(), {}
    pending = collections.deque((s, ("gauss_legendre",)) for s in range(4))
    monkeypatch.setattr(fz, "available_bytes", lambda: 25 * G)
    fz._fill(pool, running, pending, args, limits, 10 * G, 4 * G)
    assert [seed for seed, _ in pool.calls] == [0, 1] and len(pending) == 2
    assert {k: pool.calls[0][1][k] for k in limits} == limits
    monkeypatch.setattr(fz, "available_bytes", lambda: 0)
    for reserve in (4 * G, 0.0):
        assert fz._fill(pool, running, pending, args, limits, 10 * G,
                        reserve) is False
        assert len(pool.calls) == 2
    monkeypatch.setattr(fz, "available_bytes", lambda: 2 ** 50)
    fz._fill(pool, running, pending, args, limits, 10 * G, 0.0)
    assert len(pool.calls) == 3 and len(pending) == 1 and len(running) == 3


def test_VZ14_the_limits_reach_the_records_through_a_run(tmp_path,
                                                          monkeypatch):
    """Through the command line, a worker limit below any worker's size
    kills the worker, whose job never answers, at its first sample; through
    :func:`run_case`, an oracle that grows past its limit gives
    ``oracle-memory`` on every integrator, with the largest sample kept."""
    (tmp_path / "stuck_lowering.py").write_text(STUCK)
    out = tmp_path / "run"
    fz.main(["run", str(out), "--count", "1", "--jobs", "1", "--timeout",
             "60", "--methods", "gauss_legendre", "--max-rss-gb", "0.001",
             "--min-available-gb", "0", "--lowering", "stuck_lowering",
             "--extra-path", str(tmp_path)])
    (record,) = fz.records_of(out / "records.jsonl")
    assert record["verdict"] == "memory" and record["peak_rss_mb"] > 1.0

    import validation.stages as stages

    monkeypatch.setattr(stages, "MEMORY_POLL", 0.02)
    limit = resident_bytes(children=False) + 100 * 2**20

    def hungry(*args):
        _grow(20 * 2**20, 40)
        raise AssertionError("the limit did not stop the oracle")

    monkeypatch.setattr(fz, "moments_with_scale", hungry)
    records = run_case(0, str(ROOT / "src"), "validation.lower", 60.0,
                       oracle_max_rss=limit)
    assert [r["verdict"] for r in records] == ["oracle-memory"] * len(METHODS)
    assert all(r["oracle_rss_mb"] > limit / 2**20 for r in records)


BLOCKED = '''
import os, subprocess, sys, time
from validation import lower

sw = lower.sw


class Lowered(lower.Lowered):
    def evaluate(self, method, *args, **kwargs):
        pidfile = os.environ.get("PLANTED_PIDFILE")
        if pidfile:
            child = subprocess.Popen([sys.executable, "-c",
                                      "import time; time.sleep(600)"])
            with open(pidfile + ".tmp", "w") as fh:
                fh.write(str(child.pid))
            os.rename(pidfile + ".tmp", pidfile)
        time.sleep(600.0)
'''

KILLS_PARENT = '''
import hashlib, json, os, signal, time
from validation import lower
from validation.modelspec import model_to_dict

sw = lower.sw


def _first(marker, me):
    """Whether ``me`` is the first case that reached the marker."""
    tmp = f"{marker}.{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(me)
    try:
        os.link(tmp, marker)
    except FileExistsError:
        pass
    finally:
        os.unlink(tmp)
    with open(marker) as fh:
        return fh.read() == me


def _new(path):
    """Whether this call created ``path``."""
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return False
    return True


def _log(marker, what, me):
    with open(f"{marker}.log", "a") as fh:
        fh.write(f"{what} {me}\\n")


def _started(marker):
    with open(f"{marker}.log") as fh:
        return {line.split()[1] for line in fh if line.startswith("start")}


def _side_by_side(marker, me, first_call):
    """The ``first`` mode: on its first call the killer waits until the other
    case has started, and the other case stays in the pool until it breaks."""
    if _first(marker, me):
        end = time.monotonic() + 20.0
        while (first_call and len(_started(marker)) < 2
               and time.monotonic() < end):
            time.sleep(0.05)
        _log(marker, "kill", me)
        os.kill(os.getppid(), signal.SIGKILL)
    elif first_call:
        time.sleep(30.0)


class Lowered(lower.Lowered):
    def evaluate(self, method, *args, **kwargs):
        mode, marker = os.environ["PLANTED"], os.environ["PLANTED_MARKER"]
        me = hashlib.sha256(json.dumps(model_to_dict(self.model),
                                       sort_keys=True).encode()).hexdigest()
        once = mode == "once" and not os.path.exists(marker)
        if once:
            open(marker, "w").close()
        if once or mode == "always":
            os.kill(os.getppid(), signal.SIGKILL)
        if mode == "first":
            first_call = _new(f"{marker}.{me}")
            _log(marker, "start", me)
            _side_by_side(marker, me, first_call)
        return super().evaluate(method, *args, **kwargs)
'''


def _model_digest(seed: int) -> str:
    """The digest :data:`KILLS_PARENT` gives the case of ``seed``."""
    return hashlib.sha256(json.dumps(model_to_dict(draw(seed).model),
                                     sort_keys=True).encode()).hexdigest()


def _start_run(tmp_path, lowering: str, source: str, *extra, env=None):
    import os
    import subprocess
    import sys

    (tmp_path / f"{lowering}.py").write_text(source)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(ROOT / "src"), str(ROOT)]), **(env or {}))
    return subprocess.Popen(
        [sys.executable, "-m", "validation.fuzz", "run", str(tmp_path / "run"),
         "--timeout", "60", "--methods", "gauss_legendre",
         "--min-available-gb", "0", "--lowering", lowering, "--extra-path",
         str(tmp_path), *extra],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _family_with_worker(run, seconds: float = 60.0) -> list:
    """The run's descendants once its worker has started."""
    import psutil

    end = time.monotonic() + seconds
    while time.monotonic() < end:
        family = psutil.Process(run.pid).children(recursive=True)
        try:
            if any("validation.worker" in " ".join(c.cmdline())
                   for c in family):
                return family
        except psutil.Error:
            pass
        time.sleep(0.2)
    pytest.fail("the worker did not start")


def _kill_all(run, family) -> None:
    import psutil

    for proc in family:
        try:
            proc.kill()
        except psutil.Error:
            pass
    if run.poll() is None:
        run.kill()
    run.wait()


def test_VZ15_a_terminated_run_leaves_no_process_behind(tmp_path):
    """``SIGTERM`` to a run whose worker blocks for ten minutes ends the run
    at once and kills the pool process, the worker and a process the
    worker started, which has no watch of its own and would run on."""
    import psutil

    pidfile = tmp_path / "grandchild"
    run = _start_run(tmp_path, "blocked_lowering", BLOCKED, "--count", "1",
                     "--jobs", "1", env={"PLANTED_PIDFILE": str(pidfile)})
    family = []
    try:
        family = _family_with_worker(run)
        end = time.monotonic() + 30.0
        while not pidfile.exists() and time.monotonic() < end:
            time.sleep(0.1)
        family.append(psutil.Process(int(pidfile.read_text())))
        run.terminate()
        assert run.wait(timeout=15) == 128 + 15
        gone, alive = psutil.wait_procs(family, timeout=10)
        assert alive == []
    finally:
        _kill_all(run, family)


def test_VZ16_a_run_killed_outright_leaves_no_process_behind(tmp_path):
    """``SIGKILL`` leaves the run no chance to clean up: the pool process and
    the worker see their parent gone and end themselves.  A process given a
    parent that is not its own ends at once, before any watch, and says
    why; the pool forks or spawns, so a pool process is the run's child,
    under a default of forkserver too."""
    import subprocess
    import sys

    import psutil

    gone = subprocess.run(
        [sys.executable, "-c", "import time; from validation.stages import "
         "exit_with_parent; exit_with_parent(1, poll=3600); time.sleep(3600)"],
        cwd=ROOT, capture_output=True, timeout=30)
    assert gone.returncode == 1 and b"parent 1 has gone" in gone.stderr
    method = subprocess.run(
        [sys.executable, "-c", "import multiprocessing as mp; "
         "mp.set_start_method('forkserver'); from validation.fuzz import "
         "_pool_context; print(_pool_context().get_start_method())"],
        cwd=ROOT, capture_output=True, timeout=30, text=True)
    assert method.stdout.strip() == "spawn", method.stderr

    run = _start_run(tmp_path, "blocked_lowering", BLOCKED, "--count", "1",
                     "--jobs", "1")
    family = []
    try:
        family = _family_with_worker(run)
        run.kill()
        run.wait(timeout=15)
        gone, alive = psutil.wait_procs(family, timeout=10)
        assert alive == []
    finally:
        _kill_all(run, family)


def _finish(run, timeout: float = 120.0) -> tuple:
    """The run's output once it ends; a run past ``timeout`` is killed with
    its processes, and the test fails."""
    import subprocess

    import psutil

    try:
        out, err = run.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_all(run, psutil.Process(run.pid).children(recursive=True))
        run.communicate()
        raise
    assert run.returncode == 0, err.decode()[-2000:]
    return out.decode(), err.decode()


def _planted_run(tmp_path, mode: str, *extra):
    tmp_path.mkdir(exist_ok=True)
    run = _start_run(tmp_path, "killing_lowering", KILLS_PARENT, *extra,
                     env={"PLANTED": mode,
                          "PLANTED_MARKER": str(tmp_path / "marker")})
    out, err = _finish(run)
    records = fz.records_of(tmp_path / "run" / "records.jsonl")
    assert (tmp_path / "run" / "summary.json").exists()
    return out, sorted((r["seed"], r["verdict"], r.get("error", ""))
                       for r in records)


def test_VZ17_a_pool_process_that_dies_is_replaced(tmp_path):
    """A lowering that kills its pool process: on the first call only, the
    case runs again alone in a new pool and passes, and the next case runs;
    on every call, the case, alone in the pool, is recorded as a
    ``harness-error`` once it has killed it twice.  Either way the run ends
    with a summary."""
    out, records = _planted_run(tmp_path / "once", "once", "--count", "2",
                                "--jobs", "1")
    assert out.count("a pool process died") == 1
    assert [r[:2] for r in records] == [(0, "pass"), (1, "pass")]
    out, records = _planted_run(tmp_path / "always", "always", "--count", "1",
                                "--jobs", "1")
    assert out.count("a pool process died") == 1
    ((seed, verdict, error),) = records
    assert verdict == "harness-error" and "twice while it ran alone" in error


def test_VZ19_shrink_runs_under_the_memory_limits(tmp_path, monkeypatch):
    """``shrink`` gives its worker and every oracle the limits of a run, by
    default and from its command line, where 0 is no limit; an oracle past
    its limit leaves no usable case."""
    import validation.shrink as sh

    seen = []

    class Recorder:
        def __init__(self, package, lowering, extra_path=(), max_rss=None):
            seen.append(("worker", max_rss))

        def close(self):
            pass

    def oracle(case, timeout=None, max_rss=None, memory=None):
        seen.append(("oracle", max_rss))
        raise JobMemory("planted")

    monkeypatch.setattr(sh, "Worker", Recorder)
    monkeypatch.setattr(sh, "oracle_of", oracle)
    case = draw(0)
    out = tmp_path / "run"
    out.mkdir()
    (out / "records.jsonl").write_text(json.dumps(dict(
        seed=0, method="gauss_legendre", kind=case.kind,
        model=model_to_dict(case.model),
        observable=observable_to_dict(case.observable),
        max_weight=case.max_weight, flags=case.flags)) + "\n")
    base = ["shrink", str(out), "--seed", "0", "--method", "gauss_legendre"]
    for extra, worker, oracle_limit in (
            ([], fz.DEFAULT_MAX_RSS_GB * fz.GB,
             fz.DEFAULT_ORACLE_MAX_RSS_GB * fz.GB),
            (["--max-rss-gb", "1", "--oracle-max-rss-gb", "0"], fz.GB, None)):
        seen.clear()
        with pytest.raises(ValueError, match="does not fail"):
            fz.main([*base, *extra])
        assert seen == [("worker", worker), ("oracle", oracle_limit)]
    with pytest.raises(SystemExit):
        fz.main([*base, "--oracle-max-rss-gb", "-1"])


def test_VZ18_a_case_that_kills_its_pool_takes_no_other_case_down(tmp_path):
    """Two cases start side by side (no memory limits, so on any machine);
    the first to reach the lowering waits for the other, then kills its
    pool process on every call.  Both fail together and neither break is
    counted; each runs again alone, the killer twice more, and only the
    killer is recorded as a ``harness-error``."""
    out, records = _planted_run(tmp_path / "first", "first", "--count", "2",
                                "--jobs", "2", "--max-rss-gb", "0",
                                "--oracle-max-rss-gb", "0")
    log = [tuple(line.split()) for line in
           (tmp_path / "first" / "marker.log").read_text().splitlines()]
    kill = next(i for i, (what, _) in enumerate(log) if what == "kill")
    killer = log[kill][1]
    assert {who for what, who in log[:kill] if what == "start"} == {
        _model_digest(0), _model_digest(1)}
    assert [who for what, who in log if what == "kill"] == [killer] * 3
    assert sorted(r[1] for r in records) == ["harness-error", "pass"]
    ((seed, _, _),) = [r for r in records if r[1] == "harness-error"]
    assert _model_digest(seed) == killer
