Changelog
=========

Version 0.2.0
-------------

*First release on PyPI —* ``pip install sft-wick``.

Builds two higher-level API layers and a numerical evaluation pipeline
on top of the L0 symbolic core.

Added
~~~~~

- **L1 workflow API** (``System``, ``Expansion``, ``Propagators``,
  ``Result``, ``SweepResult``) — immutable, high-level interface for
  specifying a physics system and expanding / evaluating field moments,
  with single-point ``evaluate`` and grid ``sweep``.
- **L2 YAML + CLI** (``sft-wick run config.yaml``) — fully declarative,
  reproducible workflows; ``--override key=value`` for parameter scans;
  every L1 constructor argument maps 1:1 onto a YAML field.
- **Numerical evaluation pipeline** (``evaluate.py``) — propagator
  caches, three spatial-homogeneity modes (translation / rotation /
  general), and three integrators: Sobol QMC (default), tensor-product
  Gauss–Legendre, and adaptive ``nquad``.
- **Dynamic (spacetime-dependent) couplings** for non-local vertices,
  with per-sample and vectorised contracts.
- **R-contracted non-local vertices** (``already_R_contracted=True``)
  that cut integration dimensionality for narrow-kernel ``κ^(m)``.
- Closed-form ``C`` hook (``c_closed_form``) that bypasses ``dblquad``
  for separable kernels, and a Gauss–Legendre ``C`` builder
  (``c_method="gauss_legendre"``).
- ``integrate_over`` observable convention (fixed-time 2-point
  correlator vs time-integrated moment).
- Time-dependent linear drift via ``γ(t)`` callables and a fully
  explicit closed-form ``R(t₁, t₂)`` escape hatch.
- Parallel evaluation via joblib across propagator builds, expansion,
  and sweeps.
- ``compute_moment_numerical`` using nauty graph isomorphism for
  canonical labelling.

Packaging
~~~~~~~~~

- Published to PyPI; installable via ``pip install sft-wick``.
- Continuous-integration workflow (tests on Python 3.10–3.12 plus a
  packaging check) and a Trusted-Publishing release workflow.


Version 0.1.0
-------------

*Initial release.*

- Core Wick contraction engine with MSR-optimised pairing enumeration
- Custom symbolic expression tree (no SymPy dependency)
- Exact rational arithmetic via ``fractions.Fraction``
- Local and non-local interaction vertices
- Perturbative expansion via ``compute_moment()`` to arbitrary order
- Multi-pass expression simplification pipeline
- Feynman diagram generation (``networkx.MultiGraph``)
- Matplotlib-based diagram rendering
- Configurable LaTeX output with ``LaTeXFormatter``
- Support for scalar and multi-component fields
- 46 tests covering all modules
