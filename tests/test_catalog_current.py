"""The validation catalogue (docs/verification/catalog.rst) must match the
test suite as it is: every test file has a FILE_META entry and the
collected counts are the committed ones."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "gen_test_catalog", ROOT / "tools" / "gen_test_catalog.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gen_test_catalog"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_catalog_is_current():
    tool = _load_tool()
    expected = tool.render(tool.collect_counts())
    committed = tool.OUT.read_text()
    assert committed == expected, (
        "docs/verification/catalog.rst is stale -- run "
        "`python tools/gen_test_catalog.py`"
    )


# --------------------------------------------------------------------------
# The suite must not acquire new assertions weakened by approx's hidden floor.
# --------------------------------------------------------------------------

def _bare_rel_sites():
    """Every ``approx(..., rel=...)`` call that does not also pass ``abs=``."""
    import ast

    out = []
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name != "approx":
                continue
            kwargs = {kw.arg for kw in node.keywords if kw.arg}
            if "rel" in kwargs and "abs" not in kwargs:
                out.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return out


def test_every_approx_rel_site_states_its_abs():
    """``pytest.approx(x, rel=r)`` compares against ``max(r * expected, abs)``
    with ``abs`` defaulting to **1e-12**, so a site written with a tight
    ``rel`` on a small quantity silently enforces the floor instead.

    Whether that has happened depends on the RUNTIME magnitude of the
    compared value, so it cannot be read off the source -- which is exactly
    why it went unnoticed until v0.4.1, when two such assertions turned out
    to be unable to fail at all.  ``tools/approx_audit.py`` measures the
    damage after the fact; this test prevents it, by requiring every site to
    state its absolute tolerance rather than inherit one invisibly.

    ``abs=0.0`` is the right answer wherever a pure relative check is meant,
    and it is a NO-OP at any site whose ``rel`` is already in force:
    ``max(rel * expected, 0.0) == rel * expected`` there.  Pass a deliberate
    non-zero ``abs`` only where the compared quantity can legitimately reach
    zero -- a structural cancellation, a symmetry-enforced null -- and say
    which in a comment.

    See issue #5.
    """
    sites = _bare_rel_sites()
    assert not sites, (
        f"{len(sites)} approx(rel=...) site(s) do not state abs=; each inherits "
        f"the hidden 1e-12 floor. Add abs=0.0 for a pure relative check, or a "
        f"justified non-zero floor where the quantity can reach zero:\n  "
        + "\n  ".join(sites)
    )


# --------------------------------------------------------------------------
# A published timing must be able to notice that its subject changed.
# --------------------------------------------------------------------------

#: The settings each README timing row was measured under.  A timing is only
#: meaningful for the configuration it measured, and these are the fields that
#: move it: the integrator, its resolution, and the worker count.
#:
#: This is deliberately NOT a file hash.  A hash fails on a comment edit, which
#: trains the reader to re-bless it without looking -- converting a real signal
#: into a ritual.  These four fields fail only when something that can actually
#: change the runtime changes.
TIMED_CONFIGS = {
    "examples/quickstart.yaml":            {"method": None, "n_gauss": None, "n_samples": 4096, "n_jobs": None},
    "examples/demo1_config.yaml":          {"method": None, "n_gauss": None, "n_samples": 8192, "n_jobs": None},
    "examples/demo1/L2/config.yaml":       {"method": "gauss_legendre", "n_gauss": 24, "n_samples": None, "n_jobs": -1},
    "examples/demo2/L2/config_FF.yaml":    {"method": "qmc_vectorized", "n_gauss": None, "n_samples": 32768, "n_jobs": -1},
    "examples/demo2/L2/config_FK.yaml":    {"method": "gauss_legendre", "n_gauss": 32, "n_samples": None, "n_jobs": -1},
    "examples/demo3/config_FK.yaml":       {"method": "gauss_legendre", "n_gauss": 32, "n_samples": None, "n_jobs": 1},
    "examples/demo3/config_F3K.yaml":      {"method": "gauss_legendre", "n_gauss": 12, "n_samples": None, "n_jobs": -1},
}


def test_readme_timings_still_describe_these_configs():
    """The README's wall-clock table must not outlive the configs it timed.

    The row for ``examples/demo1/L2/config.yaml`` once read "4.6 min (28
    workers), 32768 samples".  That number was never *wrong*: nobody's machine
    got slower.  Its SUBJECT was replaced -- the config became
    ``method: gauss_legendre`` with ``n_gauss: 24``, whose own comment says
    ~15 min on 28 cores, and under a deterministic rule a sample count is not
    even a meaningful quantity.  A caveat about hardware would not have caught
    that, because the hardware was not the variable.

    The measurements themselves cannot be automated: the table spans three
    machines and needs an idle one, so generating it at build time would
    replace three careful numbers with one machine-dependent number.  What CAN
    be automated is the INVALIDATION.  This test pins what each row measured;
    when a config changes, the row stops being a silently-wrong number and
    becomes a failure that names itself.

    If this fails: re-measure that row on an idle machine and update both the
    README and the entry here.  Do not simply re-bless the entry -- that
    restores the exact defect it exists to prevent.
    """
    import yaml

    drifted = []
    for rel_path, pinned in sorted(TIMED_CONFIGS.items()):
        path = ROOT / rel_path
        assert path.exists(), f"{rel_path} is in the README timing table but does not exist"
        sweep = (yaml.safe_load(path.read_text()) or {}).get("sweep", {}) or {}
        actual = {k: sweep.get(k) for k in pinned}
        if actual != pinned:
            changed = {k: (pinned[k], actual[k]) for k in pinned if pinned[k] != actual[k]}
            drifted.append(f"{rel_path}: " + ", ".join(
                f"{k} was {was!r} now {now!r}" for k, (was, now) in changed.items()))
    assert not drifted, (
        "the README's wall-clock table measured a different configuration than "
        "the one committed here, so those rows now describe something that no "
        "longer exists. Re-measure on an idle machine, update the README row "
        "AND the pin in TIMED_CONFIGS:\n  " + "\n  ".join(drifted)
    )
