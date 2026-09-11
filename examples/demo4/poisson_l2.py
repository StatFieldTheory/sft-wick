"""Demo 4's hooks for the YAML configs (``config_level_a*.yaml``).

The YAML loader reads each callable as an attribute of a module named by
path, so this module binds the frozen-dataclass callables of
:mod:`poisson_system` to the two parameter sets.  Instances rather than
functions keep their ``repr`` stable, which the expansion and propagator
caches key on.
"""
from __future__ import annotations

import poisson_noise as nz
import poisson_system as dsys

# Exponential pulses: a general κ² (the pulses mix the components), the
# exact C, and the R-contracted cumulants.
KAPPA2_EXP = dsys.Kappa2Colored(nz.PARAMS_EXP)
C_EXP = dsys.ClosedFormC(nz.PARAMS_EXP)
K3_R_EXP = dsys.RContractedKappa(m=3, p=nz.PARAMS_EXP)
K4_R_EXP = dsys.RContractedKappa(m=4, p=nz.PARAMS_EXP)

# White pulses: the white-noise amplitude, the exact C, the R-contracted
# cumulants, and the raw κ³ (an ``equal_time`` vertex for white pulses).
SIGMA2_WHITE = dsys.Sigma2White(nz.PARAMS_WHITE)
C_WHITE = dsys.ClosedFormC(nz.PARAMS_WHITE)
K3_R_WHITE = dsys.RContractedKappa(m=3, p=nz.PARAMS_WHITE)
K4_R_WHITE = dsys.RContractedKappa(m=4, p=nz.PARAMS_WHITE)
K3_RAW_WHITE = dsys.RawKappa(m=3, p=nz.PARAMS_WHITE)
