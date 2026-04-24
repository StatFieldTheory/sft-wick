"""High-level user-facing workflow API.

Example::

    import sft_wick as sw
    import numpy as np

    F = np.zeros((2, 2, 2))
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = F[1, 1, 0] = 0.5

    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        # Pass the BARE F tensor — the wrapper applies the MSR
        # factor (F_MSR = −i · F) automatically.
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
            spatial=sw.ExponentialSpatial(sigma_x=1.0),
        )),
    )

    expansion = system.expand(("phi_a(x)", "phi_b(y)"),
                              orders=[0, 2, 4])
    props     = system.propagators(t_max=15.0, n_grid_t=60)

    sweep = expansion.sweep(
        props,
        positions_grid={"x": [0.0], "y": [0.0, 0.5, 1.0, 2.5]},
        t_final_grid=[1.0, 15.0],
        component_pairs=[(0, 0), (1, 1)],
    )

    print(sweep.totals())
"""

from .expansion import Expansion
from .propagators import Propagators
from .result import Result, SweepResult
from .specs import (
    ConstantImpulse,
    CustomImpulse,
    CustomKernel,
    DiagonalA,
    ExplicitR,
    ExponentialSpatial,
    ExponentialTemporal,
    FieldSpec,
    GaussianNoise,
    GaussianSpatial,
    GaussianTemporal,
    GeneralKappa2,
    Kappa2,
    LegendreAngular,
    LinearOp,
    LocalVertex,
    NonLocalVertex,
    SeparableRotation,
    SeparableTranslation,
    Sigma2,
)
from .system import System

__all__ = [
    "ConstantImpulse",
    "CustomImpulse",
    "CustomKernel",
    "DiagonalA",
    "Expansion",
    "ExplicitR",
    "ExponentialSpatial",
    "ExponentialTemporal",
    "FieldSpec",
    "GaussianNoise",
    "GaussianSpatial",
    "GaussianTemporal",
    "GeneralKappa2",
    "Kappa2",
    "LegendreAngular",
    "LinearOp",
    "LocalVertex",
    "NonLocalVertex",
    "Propagators",
    "Result",
    "SeparableRotation",
    "SeparableTranslation",
    "Sigma2",
    "SweepResult",
    "System",
]
