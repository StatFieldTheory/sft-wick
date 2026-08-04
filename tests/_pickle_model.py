"""Module-level (therefore picklable) model functions for the pickle tests.

Lambdas and closures are not picklable, so a cache built from them could never
round-trip regardless of this branch.  These make the *cache* the thing under
test.
"""
import numpy as np

MU, D, SX = 1.0, 0.5, 1.0


def R_time(t, tp):
    return np.exp(-MU * (t - tp))


def kappa2(n1, t1, n2, t2):
    a = np.atleast_1d(np.asarray(n1, dtype=float)).ravel()[0]
    b = np.atleast_1d(np.asarray(n2, dtype=float)).ravel()[0]
    return np.eye(1) * np.exp(-abs(a - b) / SX)


def sigma2(n1, t, n2):
    return np.array([[2.0 * D]])


def c_value_fn(n1, t1, n2, t2):
    return np.array([[(D / MU) * (np.exp(-MU * abs(t1 - t2))
                                  - np.exp(-MU * (t1 + t2)))]])
