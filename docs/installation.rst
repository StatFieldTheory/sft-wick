Installation
============

Requirements
------------

- Python >= 3.10
- `numpy <https://numpy.org/>`_ >= 1.24
- `scipy <https://scipy.org/>`_ >= 1.10
- `networkx <https://networkx.org/>`_ >= 3.0
- `matplotlib <https://matplotlib.org/>`_ >= 3.7
- `pandas <https://pandas.pydata.org/>`_ >= 2.0
- `PyYAML <https://pyyaml.org/>`_ >= 6.0
- `tabulate <https://pypi.org/project/tabulate/>`_ >= 0.9
- `joblib <https://joblib.readthedocs.io/>`_ >= 1.3


Install from PyPI
-----------------

The recommended way to install ``sft-wick`` is from PyPI:

.. code-block:: bash

   pip install sft-wick

This pulls in all runtime dependencies and registers the ``sft-wick``
command-line entry point.


Install from Source
-------------------

Clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/StatFieldTheory/sft-wick.git
   cd sft-wick
   pip install -e .

The ``parallel`` extra is retained for compatibility with older install
commands; ``joblib`` is now part of the default runtime install.

Progress bars use ``tqdm`` when it is installed; without it the same
information is printed as plain lines on stderr::

   pip install "sft-wick[progress]"

.. code-block:: bash

   pip install -e ".[parallel]"

To include development dependencies (``pytest``, ``pytest-cov``, ``pytest-xdist``):

.. code-block:: bash

   pip install -e ".[dev]"

To include documentation-building dependencies (Sphinx, Furo, etc.):

.. code-block:: bash

   pip install -e ".[docs]"


Optional: pynauty
-----------------

:func:`~sft_wick.perturbation.compute_moment_numerical` groups topologies
by a canonical key.  It computes the key with pynauty when pynauty is
installed and with a built-in canonicaliser (standard library only)
otherwise.  Both give the same classes, so the same terms in the same
order.  On the sets timed in the README the built-in canonicaliser takes
1.6 to 2.7 times as long per key as pynauty (about 17 µs against 9 µs
per topology for the cubic vertex at order 6, on one shared machine with
CPython 3.14).  The ``nauty`` extra
installs it:

.. code-block:: bash

   pip install "sft-wick[nauty]"

``canonical="auto"`` (the default) takes pynauty when it imports.  An
installed pynauty that fails to import, or one older than 2.8.8, gives a
``RuntimeWarning`` and the built-in canonicaliser.  ``canonical="builtin"``
never imports pynauty, and ``canonical="pynauty"`` raises ``ImportError``
when pynauty is missing, fails to import or is older than 2.8.8.

**Licence.**  sft-wick is BSD-3-Clause.  pynauty is licensed under the GNU
GPL, version 3 or later.  It is not a required dependency.  The package
imports it in one module only, ``sft_wick._canonical_nauty``, and never
under ``canonical="builtin"``.  Installing the extra adds a package under
the GPL to the environment.

**Platforms.**  pynauty 2.8.8.1 on PyPI has one wheel, for CPython 3.12 on
Linux x86_64 (glibc 2.39 or later), and a source distribution.  On every
other platform and Python version pip builds pynauty from source, which
needs a C compiler.  pynauty declares Linux and Unix as its operating
systems; on Windows, use the built-in canonicaliser.

The tests that need pynauty are skipped when it does not import.  With
``SFT_WICK_REQUIRE_PYNAUTY=1``, as in the main CI test job, they fail
instead.


Verify Installation
-------------------

.. code-block:: bash

   python -c "from sft_wick import compute_moment; print('OK')"

Run the test suite:

.. code-block:: bash

   pytest tests/ -v


Development Setup
-----------------

1. Clone and install with dev + docs extras:

   .. code-block:: bash

      pip install -e ".[dev,docs]"

2. Run tests:

   .. code-block:: bash

      pytest tests/ -v

3. Build documentation locally:

   .. code-block:: bash

      cd docs
      make html
      open _build/html/index.html
