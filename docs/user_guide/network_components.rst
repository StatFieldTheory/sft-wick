Network sites as field components
=================================

``NetworkSystem`` represents a real field at each site of a finite graph.
The column vector of site values evolves according to

.. math::

   dx_i = \left[-\gamma x_i-D\sum_j L_{ij}x_j+
                 \sum_p g_{p,i}x_i^p\right]dt+dW_i,\qquad
   \mathbb{E}[dW_i\,dW_j]=Q_{ij}\,dt.

Rows of the graph Laplacian sum to zero. An edge from source :math:`j` to
target :math:`i` has weight :math:`-L_{ij}`. Supply the white covariance
rate :math:`Q` as a common variance, a vector of independent site variances,
or a symmetric positive semidefinite matrix. Each entry in ``local_drift``
gives the signed coefficient in the equation above. Each insertion counts
as one perturbative order. Polynomial terms of either sign can be used for
finite-order coefficients.

This example computes just the second-order coefficient of an unequal-time
two-point function between sites on opposite sides of an eight-site ring.

.. code-block:: python

   import numpy as np
   from sft_wick.network import NetworkSystem

   n = 8
   eye = np.eye(n)
   laplacian = 2 * eye - np.roll(eye, 1, axis=1) - np.roll(eye, -1, axis=1)
   sites = np.arange(n)
   model = NetworkSystem(
       laplacian, gamma=0.68, diffusion=0.34, noise=0.9 + 0.3*np.cos(0.7*sites),
       local_drift={3: 0.31*np.cos(0.9*sites) - 0.05}, t_min=0.3,
   )
   expansion = model.expand(["phi_a(x)", "phi_b(y)"], orders=[2])
   result = expansion.evaluate(
       model.propagators(0.92), positions={"x": 0.0, "y": 0.0},
       external_times={"x": 0.92, "y": 0.77}, component_pair=(0, 4),
       t_final=0.92, method="gauss_legendre", n_gauss=8,
       contraction_backend="einsum",
   )
   coefficient = result.total

The component tuple selects the sites. The spatial position labels distinguish
the two insertions and carry their separate external times.
``start="zero"`` gives an initially empty Gaussian field at ``t_min``.
With ``start="stationary"`` and positive :math:`\gamma`, the initial
covariance solves the linear stationary Lyapunov equation. The nonlinear
perturbation starts at ``t_min`` in both cases.

The free propagators use the matrix exponential of the drift. Symmetric
graphs use orthonormal Laplacian modes when the eigensolver resolves the
slowest positive decay rate. A weak bridge can bring this rate close to the
conserved mode. Such graphs and directed graphs use the matrix
exponential directly and integrate the white covariance by a block
exponential with time doubling. When a long evolution could lose a
conserved mode, the solver forms the generator from the edge rates in
physical coordinates and doubles the response and driven covariance in
extended precision. The same physical coordinates preserve weakly forced
sites. For signed noise the unequal-time covariance can select a decaying
mode even when the response is nearly uniform. For :math:`t>s` the response multiplies
the equal-time covariance on the left. The Wick expansion keeps both
component indices on each response and covariance line. At each local
vertex all legs have the same node, so only one node index is summed.
``model.system`` also exposes the ordinary Wick system with its full
coupling tensors. The memory needed to store a bare degree-:math:`p`
tensor grows as :math:`N^{p+1}`.
