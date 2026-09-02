"""Compatibility shim: the R-contracted kappa^(3) lives with the demo it
belongs to, ``examples/demo2/k3_R_coupling.py``."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "demo2"))
from k3_R_coupling import *  # noqa: F401,F403,E402
from k3_R_coupling import k3_R, kappa3_raw, coupling_fn, coupling_fn_vectorized, GAMMA  # noqa: E402
