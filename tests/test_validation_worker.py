"""The package side of a comparison in its own process
(``validation/worker.py``).

One stream of jobs runs through ``python -m validation.worker``.  Every
line of stdout must be a result.  Jobs that differ only in C source, flags,
orders or weight must not share a cached expansion, and each must equal the
lowering run in this process.  Lines that are not JSON, not UTF-8 or nested
too deeply, a malformed timeout, refusals of the lowering and of the
package, a package from the wrong directory and a job past its deadline
each come back as a result with its stage (and origin, for an error), and
the stream goes on.  A second stream runs a lowering that prints through
``sys.stdout`` and straight to file descriptor 1, and stdout still carries
only results.  In this process: the deadline is not swallowed by an
``except Exception``, the cache drops the least recently used expansion,
and the version is read from a source tree and from an unpacked wheel.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

import validation.lower as lw
from validation import demo_cases as dc
from validation.modelspec import (Kernel, Leg, ModelSpec, Observable, OUBlock,
                                  model_to_dict, observable_to_dict)
from validation.worker import JobTimeout, deadline, package_identity, run_job

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.search(r'^version\s*=\s*"([^"]+)"',
                    (ROOT / "pyproject.toml").read_text(), re.M).group(1)

DIAGONAL = dc.demo5_model("isotropic")
DIAGONAL_OBS = Observable([(0, "x"), (1, "y")], {"x": 0.0, "y": 0.7}, 2.1)
RNG = np.random.default_rng(42)
SLOW = ModelSpec(gamma=(0.7, 1.3), source=(0.3, -0.5),
                 quadratic=RNG.normal(scale=0.4, size=(2, 2, 2)),
                 cubic=RNG.normal(scale=0.2, size=(2, 2, 2, 2)),
                 ou=(OUBlock(0.45, 0.6, Kernel("exponential", (0.8,)),
                             (1.0, -0.4)),), t_min=0.3)
SLOW_OBS = Observable([(0, "x"), (1, "y")], {"x": 0.0, "y": 0.9}, 1.6)


def _job(id_, model=DIAGONAL, obs=DIAGONAL_OBS, weight=2, **extra):
    job = dict(id=id_, model=model_to_dict(model),
               observable=observable_to_dict(obs), max_weight=weight,
               method="gauss_legendre", settings=dict(n_gauss=12))
    job.update(extra)
    return json.dumps(job)


STREAM = [
    _job("default"),
    _job("again"),
    _job("auto", c_source="auto"),
    _job("no diag_C", flags=dict(diag_C=False)),
    _job("orders", orders=[0, 1]),
    _job("weight 1", weight=1),
    "{not json",
    "[1, 2]",
    _job("no such method", method="no_such_method"),
    _job("response leg", obs=Observable([Leg(0, "x", "psi"), (1, "y")],
                                        DIAGONAL_OBS.positions, 2.1)),
    _job("wrong tree", expect_package="/nonexistent/tree"),
    _job("slow", model=SLOW, obs=SLOW_OBS, method="nquad", settings={},
         timeout=1.0),
    _job("after the timeout"),
    b"\xff\xfe not utf-8".decode("latin-1"),
    "[" * 1_000_000,
    _job("timeout as text", timeout="5"),
    _job("negative timeout", timeout=-1.0),
    _job("last"),
]
#: The lines of STREAM that are not JSON jobs, by position.
BAD_LINES = {6: "parse", 7: "job", 13: "parse", 14: "parse"}


@pytest.fixture(scope="module")
def results():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"),
               SFT_WICK_QUIET_CACHE="0",
               VALIDATION_EXPECT_PACKAGE=str(ROOT / "src"))
    data = "\n".join(STREAM).encode("latin-1") + b"\n"
    out = subprocess.run([sys.executable, "-m", "validation.worker"],
                         input=data, cwd=ROOT, env=env, capture_output=True,
                         check=True, timeout=600)
    lines = out.stdout.decode().splitlines()
    assert len(lines) == len(STREAM), out.stdout[:2000]
    parsed = [json.loads(line) for line in lines]
    return {r["id"] if r["id"] is not None else f"line {k}": r
            for k, r in enumerate(parsed)}


def test_VW1_jobs_that_differ_in_route_or_flags_do_not_share_a_cache(results):
    ok = {k: r for k, r in results.items() if r["ok"]}
    assert set(ok) == {"default", "again", "auto", "no diag_C", "orders",
                       "weight 1", "after the timeout", "last"}
    for r in results.values():
        if "package" in r:
            assert r["package"] == str(ROOT / "src" / "sft_wick")
            assert r["version"] == VERSION
    assert [ok[k]["cached"] for k in ("default", "again", "auto")] == [
        False, True, False]
    assert ok["auto"]["c_route"] == "closed_form:builtin"
    assert ok["default"]["c_route"] == "closed_form:user"
    assert ok["default"]["flags"]["diag_C"]
    assert not ok["no diag_C"]["flags"]["diag_C"]
    assert "build_seconds" in ok["default"] and "build_seconds" not in ok[
        "again"]
    expected = {
        "default": ({}, None), "auto": (dict(c_source="auto"), None),
        "no diag_C": (dict(flags=dict(diag_C=False)), None),
        "orders": ({}, [0, 1]), "weight 1": (dict(max_weight=1), None)}
    for name, (kw, orders) in expected.items():
        low = lw.Lowered(DIAGONAL, DIAGONAL_OBS, kw.get("max_weight", 2),
                         kw.get("flags"), kw.get("c_source", "oracle"))
        want = low.evaluate("gauss_legendre", orders=orders, n_gauss=12)
        got = {tuple(t): v for t, v, _ in ok[name]["values"]}
        assert got.keys() == want.keys(), name
        for t in got:
            assert got[t] == pytest.approx(want[t][0], rel=1e-14, abs=1e-17)


def test_VW2_failures_come_back_as_results_with_stage_and_origin(results):
    for position, stage in BAD_LINES.items():
        bad = results[f"line {position}"]
        assert (bad["status"], bad["stage"]) == ("error", stage), bad
        assert bad["version"] == VERSION
    assert results["line 7"]["error"].startswith("TypeError")
    assert results["line 7"]["origin"] == "other"
    for name in ("timeout as text", "negative timeout"):
        bad = results[name]
        assert (bad["stage"], bad["error"].split(":")[0]) == ("job",
                                                              "ValueError")
    method = results["no such method"]
    assert (method["status"], method["stage"], method["origin"]) == (
        "error", "evaluate", "package")
    leg = results["response leg"]
    assert (leg["stage"], leg["origin"]) == ("lower", "validation")
    assert leg["error"].startswith("NotImplementedError")
    tree = results["wrong tree"]
    assert tree["error"].startswith("RuntimeError") and "not from" in tree[
        "error"]
    assert (tree["stage"], tree["origin"]) == ("job", "other")
    slow = results["slow"]
    assert slow["status"] == "timeout" and slow["timeout_enforced"]
    assert slow["stage"] in ("expand", "propagators", "evaluate")
    assert "origin" not in slow and slow["wall_seconds"] < 30.0
    assert results["after the timeout"]["ok"]


PRINTING = """
import os, sys
from validation import lower
sw = lower.sw


class Lowered(lower.Lowered):
    def __init__(self, *args, **kwargs):
        print("a print from the lowering")
        sys.__stdout__.write("a write to sys.__stdout__\\n")
        sys.__stdout__.flush()
        os.write(1, b"a write to file descriptor 1\\n")
        super().__init__(*args, **kwargs)
"""


def test_VW3_stdout_carries_only_results_whatever_the_lowering_prints(
        tmp_path):
    (tmp_path / "printing_lowering.py").write_text(PRINTING)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(ROOT / "src"), str(ROOT), str(tmp_path)]))
    jobs = "\n".join([_job("one", weight=1), _job("two", weight=0)])
    out = subprocess.run([sys.executable, "-m", "validation.worker",
                          "--lowering", "printing_lowering"], input=jobs,
                         cwd=ROOT, env=env, capture_output=True, text=True,
                         check=True, timeout=300)
    results = [json.loads(line) for line in out.stdout.splitlines()]
    assert [r["id"] for r in results] == ["one", "two"]
    assert all(r["ok"] for r in results)
    for text in ("a print", "sys.__stdout__", "file descriptor 1"):
        assert out.stderr.count(text) == 2


def test_VW4_the_deadline_is_not_swallowed_and_the_cache_is_bounded():
    start = time.perf_counter()
    with pytest.raises(JobTimeout):
        with deadline(0.2):
            try:
                time.sleep(5.0)
            except Exception:  # noqa: BLE001  (what the package may do)
                pass
    assert time.perf_counter() - start < 2.0
    with deadline(None) as enforced:
        assert not enforced
    for bad in (0, -1.0, "5", True, math.nan, math.inf):
        with pytest.raises(ValueError, match="timeout"):
            with deadline(bad):
                pass
    cache: dict = {}
    jobs = [json.loads(_job(name, weight=weight, timeout=None))
            for name, weight in (("a", 0), ("b", 1), ("a again", 0))]
    cached = [run_job(job, cache, lw, cache_size=1)["cached"] for job in jobs]
    assert cached == [False, False, False] and len(cache) == 1
    jobs.append(json.loads(_job("a once more", weight=0, timeout=None)))
    assert run_job(jobs[-1], cache, lw, cache_size=1)["cached"]


def _fake_package(path: Path, version_attr: str | None = None):
    path.mkdir(parents=True)
    (path / "__init__.py").write_text("")
    module = types.ModuleType("sft_wick")
    module.__file__ = str(path / "__init__.py")
    if version_attr:
        module.__version__ = version_attr
    return module


def test_VW5_the_version_comes_from_the_package_s_own_source(tmp_path):
    tree = _fake_package(tmp_path / "repo" / "src" / "sft_wick")
    (tmp_path / "repo" / "pyproject.toml").write_text(
        '[build-system]\nrequires = []\n\n[project]\nname = "sft-wick"\n'
        'version = "9.8.7"\n\n[tool.other]\nversion = "0"\n')
    assert package_identity(tree)["version"] == "9.8.7"
    wheel = _fake_package(tmp_path / "wheel" / "sft_wick")
    meta = tmp_path / "wheel" / "sft_wick-0.4.2.dist-info"
    meta.mkdir()
    (meta / "METADATA").write_text("Metadata-Version: 2.1\nName: sft-wick\n"
                                   "Version: 0.4.2\n")
    identity = package_identity(wheel)
    assert identity["version"] == "0.4.2"
    assert identity["version_source"].endswith("METADATA")
    stale = tmp_path / "wheel" / "sft_wick-0.10.0.dist-info"
    stale.mkdir()
    (stale / "METADATA").write_text("Version: 0.10.0\n")
    identity = package_identity(wheel)
    assert identity["version"] is None
    assert identity["version_source"].startswith("several dist-info")
    own = _fake_package(tmp_path / "own" / "sft_wick", version_attr="1.2.3")
    assert package_identity(own)["version"] == "1.2.3"
