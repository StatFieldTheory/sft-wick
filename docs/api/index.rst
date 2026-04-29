API Reference
=============

Complete reference for every public class and function in **sft-wick**.

The top-level package re-exports the entire public API, so all objects
listed below are accessible directly from ``sft_wick``:

.. code-block:: python

   from sft_wick import Field, Vertex, Action, compute_moment

.. toctree::
   :maxdepth: 2
   :caption: L1 + L2 — User-facing workflow API

   workflow

.. toctree::
   :maxdepth: 2
   :caption: L0 — Raw API

   fields
   expressions
   vertices
   action
   wick
   propagators
   perturbation
   simplify
   evaluate
   diagrams
   drawing
   drawing_tikz
   render_style
   indices
   latex
   util
