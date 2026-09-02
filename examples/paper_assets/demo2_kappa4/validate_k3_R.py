"""Validate k3_R against 3-D adaptive quadrature of the raw leg integral."""
import sys, time
sys.path.insert(0, ".")
import numpy as np
from scipy.integrate import nquad
from k3_R_contracted import k3_R, kappa3_raw, GAMMA

def brute(t1, t2, t3, x=(0.0, 0.0, 0.0)):
    def f(u3, u2, u1):
        R = np.exp(-GAMMA * ((t1 - u1) + (t2 - u2) + (t3 - u3)))
        return R * kappa3_raw(u1, u2, u3, *x)
    # legs: u_i in [0, t_i']; split each at the others' partner times to help the cusps
    opts = dict(epsabs=1e-13, epsrel=1e-9, limit=200)
    val, err = nquad(f, [[0, t3], [0, t2], [0, t1]], opts=[opts, opts, opts])
    return val, err

for (t1, t2, t3) in [(1.0, 1.0, 1.0), (3.0, 1.5, 1.5), (1.5, 3.0, 1.5), (10.0, 4.0, 4.0), (0.3, 0.2, 0.25), (15.0, 15.0, 15.0)]:
    t0 = time.perf_counter(); ref, err = brute(t1, t2, t3); tb = time.perf_counter() - t0
    got = k3_R([t1], [t2], [t3], [1.0], [1.0], [1.0])[0]
    print(f"t'=({t1},{t2},{t3}): brute {ref:.10e} (±{err:.1e}, {tb:.1f}s)  k3_R {got:.10e}  rel {abs(got-ref)/abs(ref):.2e}")
# with spatial factors
ref, err = brute(3.0, 1.5, 1.5, x=(0.0, 0.5, 0.5))
got = k3_R([3.0], [1.5], [1.5], [np.exp(-0.5)], [np.exp(-0.5)], [1.0])[0]
print(f"spatial x=(0,.5,.5): brute {ref:.10e} k3_R {got:.10e} rel {abs(got-ref)/abs(ref):.2e}")
