Changelog
=========

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
