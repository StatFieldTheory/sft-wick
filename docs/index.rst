sft-wick: Wick Contractions for Statistical Field Theory
=========================================================

**sft-wick** automates the computation of perturbative expansions in the
path-integral formalism for stochastic differential equations.  Given an
observable and an interaction action, it applies Wick's theorem to express
arbitrary field moments in terms of two-point propagators --- the correlation
function :math:`C` and the response function :math:`R`.

.. math::

   \langle \mathcal{O} \rangle_S
   = \sum_{n=0}^{N} \frac{(-1)^n}{n!}\,
     \langle \mathcal{O}\, S_{\mathrm{int}}^{\,n} \rangle_{S_0}

Key Features
------------

- **Three-layer API** --- pick your abstraction: L0 symbolic / L1
  workflow (``System``, ``Expansion``, ``Propagators``,
  ``SweepResult``) / L2 YAML + CLI (``sft-wick run config.yaml``).
- **No SymPy dependency** --- custom lightweight expression tree with exact
  rational arithmetic via ``fractions.Fraction``.
- **MSR-optimised contraction** --- exploits the constraint
  :math:`\langle\psi\,\psi\rangle = 0` to skip vanishing pairings entirely.
- **Feynman diagram generation** --- ``networkx``-based graph representation
  with ``matplotlib`` rendering (correlation :math:`C` as solid blue lines,
  response :math:`R` as dashed red arrows).
- **LaTeX output** --- every expression renders to publication-ready LaTeX,
  with configurable propagator names.
- **Numerical evaluation** --- QMC with causal simplex mapping and
  O(1/N) convergence; separable / rotational / general spatial
  homogeneity modes; optional closed-form C bypass.
- **Time-dependent linear operator** and **spacetime-dependent
  non-local couplings** both supported end-to-end through the L1
  API.

Quick Start (L2 — config file)
------------------------------

**The recommended entry point for any new analysis.**  Write a YAML
config once, run it from the shell, iterate with ``--override`` or
edits.  The CLI registers automatically on install.

.. code-block:: yaml

   # demo1_config.yaml
   system:
     field: {name: phi, n_components: 2}
     linear: {type: diagonal, gamma: [1.0, 1.0]}
     vertices:
       - name: F
         coupling:                              # bare F; MSR factor
           - [[0.0, 0.0], [0.0, 1.0]]           # applied automatically
           - [[0.0, 0.5], [0.5, 0.0]]
     noise:
       kappa2:
         type: separable_translation
         temporal: {type: exponential, lam: 0.05, sigma_t: 0.3}
         spatial:  {type: exponential, sigma_x: 1.0}

   expand:
     observable: ["phi_a(x)", "phi_b(y)"]
     orders: [0, 2, 4]

   propagators: {t_max: 15.0, n_grid_t: 60}

   sweep:
     positions_grid: {x: [0.0], y: [0.0, 0.5, 1.0, 2.5]}
     t_final_grid: [1.0, 15.0]
     component_pairs: [[0, 0], [1, 1]]
     n_samples: 8192
     seed: 42

.. code-block:: bash

   sft-wick run demo1_config.yaml
   sft-wick run demo1_config.yaml --override sweep.seed=7
   sft-wick run demo1_config.yaml --dry-run            # validate only

See :doc:`user_guide/workflow` for the complete YAML schema and
``examples/demo1_config.yaml`` / ``examples/demo2_config.yaml``
for non-local vertices, closed-form :math:`C` hooks, and dynamic
couplings.

Quick Start (L1 — Python, for programmatic use)
-----------------------------------------------

The same workflow expressed directly in Python — use this when
embedding the pipeline inside a larger script:

.. code-block:: python

   import numpy as np
   import sft_wick as sw

   F = np.zeros((2, 2, 2))
   F[0, 1, 1] = 1.0
   F[1, 0, 1] = F[1, 1, 0] = 0.5

   system = sw.System(
       field=sw.FieldSpec("phi", n_components=2),
       linear=sw.DiagonalA(gamma=[1.0, 1.0]),
       vertices=[sw.LocalVertex("F", coupling=F)],   # bare F
       noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
           temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
           spatial=sw.ExponentialSpatial(sigma_x=1.0),
       )),
   )

   expansion = system.expand(("phi_a(x)", "phi_b(y)"),
                             orders=[0, 2, 4])
   props = system.propagators(t_max=15.0, n_grid_t=60)

   sweep = expansion.sweep(
       props,
       positions_grid={"x": [0.0], "y": [0.0, 0.5, 1.0, 2.5]},
       t_final_grid=[1.0, 15.0],
       component_pairs=[(0, 0), (1, 1)],
   )

   print(sweep.totals())

For the L0 symbolic API (fine-grained control over individual
pairings, canonical forms, custom simplification), see
:doc:`user_guide/index`.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   theory/index
   getting_started
   user_guide/index
   examples/index
   verification/index
   api/index
   contributing
   changelog

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
