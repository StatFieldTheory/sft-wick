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
  counts, and the watchdog: a worker whose job blocks the alarm is killed,
  and one that exits is reported with its return code.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time

import numpy as np
import pytest

import validation.lower as lw
import validation.lowering_parts as lp
from validation.compare import compare_values, tolerance
from validation.draw import admissible_flags, draw
import validation.fuzz as fz
from validation.fuzz import METHODS, ROOT, run_case, summarise, verdict
from validation.shrink import shrink
from validation.stages import JobTimeout
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
    for status in ("timeout", "killed", "crashed"):
        assert verdict(dict(status=status), ref, scale, "nquad")[
            "verdict"] == status
    assert verdict(dict(status="new"), ref, scale, "nquad")["verdict"] == (
        "harness-error")
    for raised, want in ((ValueError("no"), "oracle-error"),
                         (JobTimeout("slow"), "oracle-timeout")):
        def oracle(case, timeout=None, raised=raised):
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
        time.sleep(30.0)
'''


def test_VZ8_a_worker_that_hangs_is_killed_and_one_that_exits_is_named(
        tmp_path, monkeypatch):
    """The worker's own alarm cannot reach a job that blocks SIGALRM: the
    parent's watchdog kills it.  A worker that exits is reported with its
    return code."""
    (tmp_path / "stuck_lowering.py").write_text(STUCK)
    monkeypatch.setattr(fz, "WATCHDOG_MARGIN", 1.0)
    case = draw(0)
    worker = fz.Worker(ROOT / "src", "stuck_lowering", (tmp_path,))
    try:
        start = time.perf_counter()
        result = worker.run(fz.job_of(case, "gauss_legendre", 0.5))
        assert result["status"] == "killed"
        assert time.perf_counter() - start < 15.0
        monkeypatch.setenv("PLANTED", "exit")
        result = worker.run(fz.job_of(case, "gauss_legendre", 0.5))
        assert (result["status"], result["returncode"]) == ("crashed", 3)
    finally:
        worker.close()


def test_VZ9_resume_runs_only_the_missing_pairs(tmp_path):
    """A resume with a second integrator adds that integrator's records and
    nothing else, and ``run.json`` keeps both invocations."""
    out = tmp_path / "run"
    base = [str(out), "--count", "2", "--jobs", "2", "--timeout", "60"]
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
