Static Gaussian disorder
========================

A quenched coupling is drawn once for the whole trajectory. Consider
independently driven sites with empty initial data,

.. math::

   dx_i = \left[-\gamma x_i + g\sum_j J_{ij} x_j - u x_i^3\right]dt
          + \sqrt{q}\,dW_i .

For independent Gaussian entries the covariance of the matrix is
:math:`\langle J_{ij}J_{kl}\rangle=(v/N)\delta_{ik}\delta_{jl}`.
The real symmetric Gaussian ensemble (GOE) adds the exchange term
:math:`(v/N)\delta_{il}\delta_{jk}`. The diagonal entries then have
variance :math:`2v/N`.

.. code-block:: python

   from sft_wick.disorder import RandomCoupling

   law = RandomCoupling(n_sites=6, variance=1.0, ensemble="iid")
   obs = [(0, 2, 0.7), (0, 2, 1.2)]   # (copy, site, time)
   second = law.coefficient(obs, pairs=1, t_min=0.2)  # 0.04157221368
   fourth = law.coefficient(obs, pairs=2, t_min=0.2)  # 0.00453000852

These are coefficients of :math:`g^2` and :math:`g^4` in the same-site
two-time covariance. The call with ``pairs=p, cubic=k`` returns the
coefficient of :math:`g^{2p}u^k`. The parameter ``noise`` is the
white-noise covariance rate :math:`q`. It defaults to one, as does
``decay``. Perturbative coefficients refer to the specified finite time.

Two thermal histories can share the same matrix. Conditional on the
matrix their noises are independent. The connected four-point function
below measures the variance across matrices of a thermal two-point
function, since both conditional one-point means vanish in this model.

.. code-block:: python

   twins = [(c, 2, t) for c in (0, 1) for t in (0.7, 1.2)]
   variance = law.coefficient(twins, pairs=1, copies=2, connected=True,
                              t_min=0.2)  # 0.005161803986

The group vertex supplied by ``law.vertex(copies=2)`` can also be used
in ``System(grouped_vertices=...)`` with a general free response and
noise. Give the field :math:`2N` components ordered by copy, then by site.
To model independent thermal histories, use a block-diagonal response and
independent noise blocks. Inside a copy the noise may be correlated. The
two factors of the random coupling have independent time arguments. The
grouped-vertex action accounts for the Gaussian factor of one half.
Every vertex insertion supplies two powers of the random drift. Multiply
``variance`` by :math:`g^2` to include a specified amplitude in the vertex.

For isotropic free sites, ``coefficient`` counts the independent site and
copy labels in each Wick contraction before integrating its time dependence.
This keeps the two times of the disorder vertex and the labels common to
all lines touching it. It avoids constructing a dense covariance of all
tensor entries. The general ``System`` path materializes that tensor when
the couplings are built and checks its workspace first.

For an ordered iid tensor of rank :math:`r\ge2`, pass ``rank=r``.
The first index is the output site and the other :math:`r-1` indices
multiply physical fields at the same time. Each entry has variance
:math:`v/N^{r-1}`. Distinct permutations of the inputs are independent
entries. The GOE exchange pairing applies to matrices.

Averaging each propagator over a relaxation spectrum and then using those
averages in an interacting diagram can change a coefficient. For one site
with unit damping and unit noise at time one, the coefficient of
:math:`u g^2` is :math:`-0.4887923943` with shared disorder. Replacing
the three lines of the cubic diagram by their individual spectral
averages gives :math:`-0.2370198305`. Those lines depend on the same
matrix in the quenched model.
