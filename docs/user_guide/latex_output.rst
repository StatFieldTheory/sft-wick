LaTeX Output
============

Every expression in **sft-wick** can be rendered as publication-ready
LaTeX.


The ``to_latex()`` Method
-------------------------

All expression types implement ``to_latex()``:

.. code-block:: python

   result = compute_moment(obs, action, order=1)

   # Single expression
   print(result.order(0).to_latex())
   # e.g. "C_{ab}(x, y) C_{cd}(z, w) + ..."

   # Full result (order by order)
   print(result.to_latex())
   # O(0): ...
   # O(1): ...

Examples of LaTeX output for each expression type:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Expression
     - LaTeX
   * - ``Rational(3, 4)``
     - ``\frac{3}{4}``
   * - ``Symbol('F', ('i','j','k'))``
     - ``F_{ijk}``
   * - ``Propagator('C', 'a', 'b', 'x', 'y')``
     - ``C_{ab}(x, y)``
   * - ``SumOverIndex('i', 3, body)``
     - ``\sum_{i=1}^{3} ...``
   * - ``IntegralOver('y', body)``
     - ``\int \mathrm{d}y\, ...``
   * - ``KroneckerDelta('i', 'j')``
     - ``\delta_{ij}``
   * - ``DiracDelta('x', 'y')``
     - ``\delta(x - y)``


Custom Propagator Names
-----------------------

Use :class:`~sft_wick.latex.LaTeXFormatter` to replace the default
propagator names ``C`` and ``R``:

.. code-block:: python

   from sft_wick import LaTeXFormatter

   fmt = LaTeXFormatter(propagator_names={
       'C': 'G',
       'R': r'R^{\mathrm{ret}}'
   })
   print(fmt.format(result.order(0)))
   # G_{ab}(x, y) instead of C_{ab}(x, y)


Equation Formatting
-------------------

``format_equation()`` wraps an expression in an equation:

.. code-block:: python

   fmt.format_equation(r'\langle O \rangle', result.total)
   # \langle O \rangle = <formatted expression>


Aligned Environment
-------------------

``format_aligned()`` generates a LaTeX ``align`` environment showing
each perturbative order on its own line:

.. code-block:: python

   print(fmt.format_aligned(result.order_terms))

Output:

.. code-block:: latex

   \begin{align}
     O(0) &= \ldots \\
     O(1) &= \ldots \\
   \end{align}


Integrating into Papers
-----------------------

The LaTeX output is designed to paste directly into ``.tex`` documents.
Typical workflow:

1. Run the calculation in a Python script or Jupyter notebook.
2. Call ``to_latex()`` or ``format_aligned()`` to get the LaTeX string.
3. Paste into your paper inside a ``\begin{equation}`` or
   ``\begin{align}`` environment.

.. tip::

   In Jupyter notebooks, use ``IPython.display.Math`` to render
   expressions inline:

   .. code-block:: python

      from IPython.display import Math
      Math(result.order(0).to_latex())
