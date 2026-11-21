"""Compare one exponential-pulse demo 4 FF configuration through L1."""

from math import fsum

from poisson_noise import PARAMS_EXP
from poisson_system import make_system, propagators_for


def run_for(method):
    positions = {"x": 0.0, "y": 0.8}
    t_final = 1.7
    system = make_system(PARAMS_EXP, f_amplitude=1.0, cumulants=())
    props = propagators_for(system, PARAMS_EXP, t_max=2.2)
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2], diag_C=False)
    options = {"n_gauss": 32} if method == "gauss_legendre" else {"n_samples": 32768, "seed": 42}
    result = expansion.evaluate(
        props, positions=positions, t_final=t_final, component_pair=(0, 0),
        orders=[2], method=method, **options,
    )
    return [dict(
        coordinates=dict(positions, t_final=t_final, a=0, b=0, order=2),
        value=float(result.total),
        error=fsum(float(row["error"]) for row in result.per_diagram) if result.per_diagram else None,
    )]
