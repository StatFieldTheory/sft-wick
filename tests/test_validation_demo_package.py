"""The lowering reproduces the package values committed for demos 4, 5 and
7 from a ModelSpec.

``validation/demo_package_cases.py`` lists the 1,497 committed package
values whose route the lowering takes too; the evidence run reproduces all
of them (the largest relative difference 3.8e-16, QMC included).  Here up
to five fast deterministic evaluations of each demo part and integrator
(:data:`CHOSEN`) run through ``validation/lower.py``: each value must come back to
1e-12 relative (1e-15 absolute for a stored zero) and lie within the
declared tolerance of the oracle.  QMC rows are left to the evidence run,
since scipy's Sobol scrambling can change between releases.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from validation.compare import compare_values, tolerance
from validation.demo_package_cases import package_cases
from validation.hierarchy import moments
from validation.lower import Lowered

#: Committed package values per demo part and integrator.
COUNTS = {
    ("demo4", "gauss_legendre"): 18, ("demo4 order 4", "gauss_legendre"): 4,
    ("demo5 part A", "gauss_legendre"): 57, ("demo5 part A", "nquad"): 24,
    ("demo5 part A", "qmc"): 9, ("demo5 part A", "qmc_scalar"): 9,
    ("demo5 part A", "qmc_vectorized"): 18,
    ("demo5 part C", "gauss_legendre"): 260, ("demo5 part C", "nquad"): 100,
    ("demo5 part C", "qmc_scalar"): 88, ("demo5 part C", "qmc_vectorized"): 112,
    ("demo7 item a", "gauss_legendre"): 108, ("demo7 item a", "nquad"): 108,
    ("demo7 item a", "qmc_scalar"): 108, ("demo7 item a", "qmc_vectorized"): 108,
    ("demo7 item d", "gauss_legendre"): 30, ("demo7 item d", "qmc_scalar"): 30,
    ("demo7 item d", "qmc_vectorized"): 30,
    ("demo7 item e", "gauss_legendre"): 102, ("demo7 item e", "qmc_scalar"): 87,
    ("demo7 item e", "qmc_vectorized"): 87,
}
#: Per demo part and deterministic integrator, the evaluations checked here,
#: named by the first committed value each carries: up to five spread over
#: those that took under a second in the evidence run (demo 4 at order 4
#: has none, so its one evaluation runs as a heavy test).  They cover every
#: tag the case tables map (demo 4: order 0, FK3 and FF, FFK4, the
#: tadpole) and a two-time case with distinct components, where swapping
#: the two times changes the value.
CHOSEN = {
    ('demo4', 'gauss_legendre'): (
        'demo4 white <phi_0(x) phi_0(y)> FFK4 #3',
        'demo4 white <phi_0(x) phi_1(y)> FFK4 #7',
        'demo4 white <phi_0(x)> F tadpole #16',
        'demo4 white <phi_0(x) phi_1(y)> order 0 #4',
        'demo4 white <phi_1(x) phi_0(y)> FK3 #9',
    ),
    ('demo4 order 4', 'gauss_legendre'): (
        'demo4 white <phi_1(x) phi_1(y)> F3K3 #3',
    ),
    ('demo5 part A', 'gauss_legendre'): (
        'demo5 diagonal <phi_0(x) phi_0(y)> order 2 gauss_legendre #15',
        'demo5 diagonal <phi_1(x) phi_1(y)> order 2 gauss_legendre #32',
        'demo5 isotropic <phi_1(x) phi_0(y) phi_0(z)> order 1 gauss_legendre #37',
        'demo5 mixing <phi_0(x) phi_1(y)> order 2 gauss_legendre #20',
        'demo5 mixing <phi_1(x)> order 3 gauss_legendre #12',
    ),
    ('demo5 part A', 'nquad'): (
        'demo5 diagonal <phi_0(x) phi_0(y)> order 2 nquad #17',
        'demo5 diagonal <phi_1(x) phi_1(y)> order 2 nquad #34',
        'demo5 isotropic <phi_0(x)> order 1 nquad #4',
        'demo5 mixing <phi_0(x) phi_1(y)> order 2 nquad #22',
        'demo5 mixing <phi_1(x)> order 1 nquad #11',
    ),
    ('demo5 part C', 'gauss_legendre'): (
        'demo5c ito, scalar R, gauss_legendre phi_a(x) [0] [0, 0] #0',
        'demo5c ito, scalar R, gauss_legendre phi_a(x) phi_b(y) [1, 0] [0, 0] #0',
        'demo5c stratonovich, scalar R, gauss_legendre phi_a(x) [0] [0, 0] #0',
        'demo5c stratonovich, scalar R, gauss_legendre phi_a(x) phi_b(y) [0, 0] [0, 0] #0',
        'demo5c stratonovich, scalar R, gauss_legendre, two times phi_a(x, t1) phi_b(y, t2) [0, 1] [0, 0] #0',
    ),
    ('demo5 part C', 'nquad'): (
        'demo5c ito, matrix R, nquad phi_a(x) [0] [0, 0] #0',
        'demo5c ito, matrix R, nquad phi_a(x) [1] [0, 0] #0',
        'demo5c stratonovich, matrix R, nquad phi_a(x) [0] [0, 0] #0',
        'demo5c stratonovich, matrix R, nquad phi_a(x) [1] [0, 0] #0',
    ),
    ('demo7 item a', 'gauss_legendre'): (
        'demo7 translation-exp phi_a(x) phi_b(y) [0, 1] gauss_legendre (0, 1) #4',
        'demo7 translation-exp phi_a(x) phi_b(y) [1, 1] gauss_legendre (0, 1) #64',
        'demo7 translation-exp-matrixR phi_a(x) phi_b(y) [1, 0] gauss_legendre (0, 2) #128',
        'demo7 translation-exp-white-matrixR phi_a(x) phi_b(y) [0, 1] gauss_legendre (0, 2) #152',
        'demo7 translation-exp-white-matrixR phi_a(x) phi_b(y) [1, 1] gauss_legendre (0, 2) #212',
    ),
    ('demo7 item a', 'nquad'): (
        'demo7 translation-exp phi_a(x) phi_b(y) [0, 1] nquad (0, 1) #43',
        'demo7 translation-exp phi_a(x) phi_b(y) [1, 0] nquad (0, 2) #23',
        'demo7 translation-exp-matrixR phi_a(x) phi_b(y) [0, 1] nquad (0, 1) #115',
        'demo7 translation-exp-matrixR phi_a(x) phi_b(y) [1, 1] nquad (0, 1) #139',
        'demo7 translation-exp-white-matrixR phi_a(x) phi_b(y) [1, 1] nquad (0, 1) #211',
    ),
    ('demo7 item d', 'gauss_legendre'): (
        'demo7 shot3d white phi_a(x) phi_b(y) [0, 1] gauss_legendre (1, 1) #3',
        'demo7 shot3d white phi_a(x) phi_b(y) phi_c(z) [0, 1, 1] gauss_legendre (0, 1) #18',
        'demo7 shot3d white-matrixR phi_a(x) phi_b(y) [0, 1] gauss_legendre (1, 1) #58',
        'demo7 shot3d white-matrixR phi_a(x) phi_b(y) [1, 1] gauss_legendre (1, 1) #70',
        'demo7 shot3d white-matrixR phi_a(x) phi_b(y) phi_c(z) [1, 1, 0] gauss_legendre (0, 1) #79',
    ),
    ('demo7 item e', 'gauss_legendre'): (
        'demo7 translation-exp phi_a(x) phi_b(y) [0, 1] gauss_legendre (0, 1) #313',
        'demo7 translation-exp phi_a(x) phi_b(y) [1, 1] gauss_legendre (0, 1) #358',
        'demo7 translation-exp phi_a(x) phi_b(y) phi_c(z) [1, 1, 0] gauss_legendre (0, 2) #428',
        'demo7 translation-exp-matrixR phi_a(x) phi_b(y) [1, 1] gauss_legendre (0, 1) #413',
        'demo7 translation-exp-matrixR phi_a(x) phi_b(y) phi_c(z) [1, 1, 0] gauss_legendre (0, 2) #441',
    ),
}


def _evaluations(group: str, method: str) -> list:
    """The :data:`CHOSEN` evaluations of ``(group, method)``, each with the
    cases it carries."""
    evals = defaultdict(list)
    for c in package_cases():
        if (c.group, c.method) == (group, method):
            evals[(c.model, c.observable, c.max_weight, c.orders, c.settings,
                   c.c_source, c.flags)].append(c)
    by_first = {cases[0].source: (key, cases) for key, cases in evals.items()}
    return [by_first[source] for source in CHOSEN[(group, method)]]


def test_VP1_the_cases_cover_every_route_the_lowering_shares():
    counts = Counter((c.group, c.method) for c in package_cases())
    assert dict(counts) == COUNTS
    assert sum(COUNTS.values()) == 1497


@pytest.mark.parametrize("group,method", [
    pytest.param(*key, marks=pytest.mark.heavy) if key[0] == "demo4 order 4"
    else key for key in CHOSEN])
def test_VP2_committed_package_values_come_back(group, method):
    checked = 0
    for (model, obs, weight, orders, settings, c_source, flags), cases in (
            _evaluations(group, method)):
        low = Lowered(model, obs, weight, dict(flags), c_source)
        got = low.evaluate(method, orders=orders, **dict(settings))
        for case in cases:
            value = sum(got.get(t, (0.0, 0.0))[0] for t in case.tags)
            tol = 1e-12 * abs(case.expected) if case.expected else 1e-15
            assert abs(value - case.expected) <= tol, case.source
            checked += 1
        tags = sorted(set(got) | {t for c in cases for t in c.tags})
        rows = compare_values(moments(model, obs, tags), got,
                              tolerance(method, c_source))
        assert all(r.ok for r in rows), [r for r in rows if not r.ok]
    assert checked >= len(CHOSEN[(group, method)])
