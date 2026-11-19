"""Settings for parallel runs of the test suite (``pytest -n``, pytest-xdist),
and the fixtures of the tests that run with and without pynauty."""
import os

import pytest

#: With this variable set to ``"1"`` (as in CI), a test that needs pynauty
#: fails when pynauty cannot be imported, instead of being skipped.
REQUIRE_PYNAUTY = "SFT_WICK_REQUIRE_PYNAUTY"


def pytest_configure(config):
    """Cap the process pools that ``n_jobs=-1`` starts inside an xdist worker.

    joblib's loky backend sizes a pool by the CPU count, which it reads from
    ``LOKY_MAX_CPU_COUNT`` when that is set.  Without the cap every worker of
    ``pytest -n 24`` on a 28-core machine could start 28 processes of its own.
    Two processes still take the parallel path that those tests compare with
    the serial one.  A run without ``-n`` is left unchanged.
    """
    if os.environ.get("PYTEST_XDIST_WORKER"):
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")


def _pynauty_or_skip():
    """The pynauty module.  Without it the calling test is skipped, or fails
    when ``SFT_WICK_REQUIRE_PYNAUTY=1``."""
    try:
        import pynauty
    except ImportError as exc:
        if os.environ.get(REQUIRE_PYNAUTY) == "1":
            pytest.fail(
                f"{REQUIRE_PYNAUTY}=1, but pynauty cannot be imported: {exc!r}",
                pytrace=False,
            )
        pytest.skip(f"pynauty is not installed ({exc})")
    return pynauty


@pytest.fixture
def pynauty_module():
    """The pynauty module, for a test that cannot run without it."""
    return _pynauty_or_skip()


@pytest.fixture(params=["builtin", "pynauty"])
def canonical_method_name(request):
    """Each value of ``canonical=``: ``"builtin"`` always runs, and
    ``"pynauty"`` needs pynauty as :func:`pynauty_module` does."""
    if request.param == "pynauty":
        _pynauty_or_skip()
    return request.param
