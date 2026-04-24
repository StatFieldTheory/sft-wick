Contributing
============

Contributions are welcome!  This page describes how to set up the
development environment and the key design invariants to respect.


Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/sft-wick/sft-wick.git
   cd sft-wick
   pip install -e ".[dev,docs]"
   pytest tests/ -v


Running Tests
-------------

.. code-block:: bash

   # All tests
   pytest tests/ -v

   # Single file
   pytest tests/test_wick.py -v

   # Single test by name
   pytest tests/test_wick.py::test_generate_valid_pairings -v

The test suite has **two tiers**:

- *Unit tests* (``test_wick.py``, ``test_simplify.py``, ``test_propagators.py``,
  etc.) covering expressions, fields, propagators, Wick contractions,
  perturbative expansion, and simplification — the per-module tests that
  guard individual functions.
- *Deductive tests* (``test_deductive_expansion.py``,
  ``test_deductive_numerics.py``) that cross-check every transformation
  in the package against an independent reference — see
  :doc:`/verification/index` for the full overview and rationale.

110+ deductive tests run in ~78 s and prove that each elementary
transformation does what it should, independently of simulation.


Code Style
----------

- **Frozen dataclasses** for all expression types (immutable, hashable).
- **Type hints** on all public function signatures.
- **Google-style docstrings** (compatible with ``sphinx.ext.napoleon``).
- Standard Python formatting conventions.


Architecture Overview
---------------------

The core data flow is a pipeline:

1. :mod:`~sft_wick.fields` --- ``Field`` declares a field type;
   ``FieldOperator`` is a concrete instance with a unique UID.

2. :mod:`~sft_wick.vertices` + :mod:`~sft_wick.action` --- ``Vertex``
   defines an interaction term; ``VertexInstance`` is a fresh copy with
   non-overlapping indices.  ``Action`` holds a list of vertices.

3. :mod:`~sft_wick.wick` --- ``generate_valid_pairings()`` enumerates
   complete pairings skipping :math:`\psi`--:math:`\psi` pairs.
   ``wick_contract()`` sums over all valid pairings.

4. :mod:`~sft_wick.propagators` --- ``contract_pair()`` maps
   :math:`(\phi,\phi) \to C`, :math:`(\phi,\psi) \to R`,
   :math:`(\psi,\psi) \to 0`.

5. :mod:`~sft_wick.perturbation` --- ``compute_moment()`` is the main
   entry point: expands vertex combinations, instantiates vertices,
   calls ``wick_contract()``, and returns a ``PerturbativeResult``.

6. :mod:`~sft_wick.simplify` --- Multi-pass simplification pipeline.

7. :mod:`~sft_wick.expressions` --- Custom symbolic expression tree.

8. :mod:`~sft_wick.diagrams` + :mod:`~sft_wick.drawing` --- Feynman
   diagram graph representation and matplotlib rendering.


Key Design Invariants
---------------------

When contributing, please respect these invariants:

**Expression immutability.**
  All expression types must be frozen dataclasses.  Never mutate an
  expression after creation.

**UID uniqueness.**
  Every ``FieldOperator`` must receive a unique UID from the global
  counter.  Call ``reset_uid_counter()`` at the start of each test
  for reproducibility.

**IndexContext isolation.**
  Each call to ``compute_moment()`` creates a fresh ``IndexContext``
  per vertex combination to avoid index collisions.

**Canonical R ordering.**
  The response propagator ``R`` always puts the physical field on
  the left: :math:`R_{ij}(x,x') = \langle\phi_i(x)\,\psi_j(x')\rangle`.

**No SymPy dependency.**
  The expression tree is deliberately self-contained.  Do not introduce
  SymPy as a dependency.


Adding a New Expression Type
-----------------------------

1. Create a frozen dataclass inheriting from
   :class:`~sft_wick.expressions.Expr`.
2. Implement ``to_latex()``, ``__eq__()``, and ``__hash__()``.
3. Update ``_flatten``, ``_absorb_rationals``, and
   ``_eliminate_zeros`` in :mod:`~sft_wick.simplify` if the new type
   can nest other expressions.
4. Add the type to ``__all__`` in ``__init__.py``.
5. Write tests.


Building Documentation
----------------------

.. code-block:: bash

   cd docs
   make html

The output is in ``docs/_build/html/``.  Open ``index.html`` to review.
