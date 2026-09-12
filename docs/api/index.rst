API Reference
=============

Complete reference for every public class and function in **sft-wick**.

The top-level package re-exports the user-facing API — every name in
``sft_wick.workflow.__all__`` and the L0 public types — so most objects
listed below are accessible directly from ``sft_wick``:

.. code-block:: python

   from sft_wick import Field, Vertex, Action, compute_moment

The remaining helpers are imported from their own module:
``sft_wick.workflow.config.run_workflow``, and ``hash_spec`` /
``load_or_compute`` from ``sft_wick.workflow.cache``.

.. toctree::
   :maxdepth: 2
   :caption: L1 + L2 — User-facing workflow API

   workflow
   closed_forms
   estimate

.. toctree::
   :maxdepth: 2
   :caption: L0 — Raw API

   fields
   expressions
   vertices
   action
   wick
   propagators
   spectral
   selfconsistency
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
   progress
