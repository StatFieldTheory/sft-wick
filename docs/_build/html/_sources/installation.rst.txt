Installation
============

Requirements
------------

- Python >= 3.10
- `networkx <https://networkx.org/>`_ >= 3.0
- `matplotlib <https://matplotlib.org/>`_ >= 3.7


Install from Source
-------------------

Clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/sft-wick/sft-wick.git
   cd sft-wick
   pip install -e .

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
