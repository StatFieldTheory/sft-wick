"""Settings for parallel runs of the test suite (``pytest -n``, pytest-xdist)."""
import os


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
