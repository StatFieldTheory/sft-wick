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

To include development dependencies (``pytest``, ``pytest-cov``):

.. code-block:: bash

   pip install -e ".[dev]"

To include documentation-building dependencies (Sphinx, Furo, etc.):

.. code-block:: bash

   pip install -e ".[docs]"


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
