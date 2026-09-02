"""Validate k4_R against QMC of the raw 4-D leg integral."""
import sys, time
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import qmc
from k4_R_contracted import k4_R, GAMMA
from k4_coupling import kappa4_amplitude
for tp in [(1.0, 1.0, 1.0, 1.0), (3.0, 1.5, 1.5, 1.5), (2.0, 2.0, 0.5, 0.5), (10.0, 4.0, 4.0, 4.0), (15.0, 15.0, 15.0, 15.0)]:
    tp = np.array(tp)
    vals = []
    for seed in range(4):
        u = qmc.Sobol(4, scramble=True, seed=seed).random(2**21) * tp
        f = np.exp(-GAMMA * np.sum(tp[None, :] - u, axis=1)) * kappa4_amplitude(np.zeros((4, u.shape[0])), u.T)
        vals.append(f.mean() * np.prod(tp))
    t0 = time.perf_counter()
    s = {(i, j): np.ones(1) for i in range(4) for j in range(4) if i != j}
    got = k4_R(tp[:, None], s)[0]
    print(f"t'={tuple(tp)}: QMC {np.mean(vals):.6e} ± {np.std(vals)/2:.1e}   k4_R {got:.6e} ({time.perf_counter()-t0:.2f}s)   rel {abs(got-np.mean(vals))/abs(np.mean(vals)):.1e}", flush=True)
