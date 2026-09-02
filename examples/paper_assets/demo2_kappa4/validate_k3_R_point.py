import sys, time
sys.path.insert(0, ".")
import numpy as np
from scipy.integrate import nquad
from k3_R_contracted import k3_R, kappa3_raw, GAMMA
t1, t2, t3 = map(float, sys.argv[1:4])
xs = tuple(map(float, sys.argv[4:7])) if len(sys.argv) > 4 else (0.0, 0.0, 0.0)
def f(u3, u2, u1):
    return np.exp(-GAMMA * ((t1 - u1) + (t2 - u2) + (t3 - u3))) * kappa3_raw(u1, u2, u3, *xs)
opts = dict(epsabs=1e-14, epsrel=1e-7, limit=100)
t0 = time.perf_counter()
ref, err = nquad(f, [[0, t3], [0, t2], [0, t1]], opts=[opts, opts, opts])
s = lambda a, b: np.exp(-abs(a - b))
got = k3_R([t1], [t2], [t3], [s(xs[0], xs[1])], [s(xs[0], xs[2])], [s(xs[1], xs[2])])[0]
print(f"t'=({t1},{t2},{t3}) x={xs}: nquad {ref:.10e} (±{err:.1e}, {time.perf_counter()-t0:.0f}s)  k3_R {got:.10e}  rel {abs(got-ref)/abs(ref):.2e}", flush=True)
