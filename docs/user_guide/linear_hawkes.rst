Finite-window Hawkes counts
===========================

A linear Hawkes network has an event rate at each node. An event increases
subsequent rates, and that increase decays exponentially. The following
example calculates a joint cumulant of counts in three windows. Node numbers
start at zero. The model starts with empty history at time zero.

.. code-block:: python

   from sft_wick.hawkes import LinearHawkes

   model = LinearHawkes(
       baseline=[0.49, 0.73],
       excitation=[[0.15, 0.38], [0.26, 0.14]],
       decay=[1.31, 0.82],
       mark_moments=[1.0, 2.0, 6.0],
   )
   windows = [(0, 0.20, 0.83), (1, 0.48, 1.30), (0, 1.01, 1.64)]
   kappa = model.count_cumulant(windows)  # 0.0393049833110508

An event at source node :math:`j` increments excitation :math:`i` by
:math:`A_{ij}M`. The same independent nonnegative mark :math:`M` multiplies
all target increments from that event. Each event increases its own counter
by one. The example supplies the first three raw moments of a unit-mean
exponential mark. The default moments describe a unit mark with certainty.
Excitation at node :math:`i` decays at its own rate :math:`\beta_i`, and
its event rate is baseline plus excitation. A scalar baseline, excitation
and decay specify a one-node model. Supply moments of an actual common
mark distribution. Validation checks nonnegativity, log-convexity and
the deterministic case.

Use one window for a mean, two for a covariance or three for a connected
third cumulant. Repeating a window gives the corresponding marginal
cumulant. Windows start on or after the absolute initial time ``t_min``.
With empty history, only clusters from immigrant sources that can reach
every counted node contribute to their joint cumulant. The calculation keeps
the excitation paths from these sources to the counted nodes. The cumulant
is zero when there is no such source.
Mean counts and the mean event rates used by the jump vertices follow a
closed set of first-moment equations. Solving them separately keeps the
faster growth of the covariance in a supercritical network from degrading
these means. The solver uses a wider arithmetic range when time scales or
event rates require it. This calculation also observes ``workspace_max_bytes``.

The count calculation forms quadrature nodes in time elapsed since ``t_min``.
The default third-order calculation refines the quadrature until all endpoint
contributions converge. Pass ``n_gauss`` to select a fixed Gauss–Legendre
rule. Long prehistories or fast decay rates may require more nodes.
For independent Poisson counters the joint cumulant is the rate times
the length of the common time interval on one node. For a coupled process,
a window so short and late that endpoint rounding swamps the difference
raises an arithmetic error.

Wick system and jump vertices
-----------------------------

For custom observables use ``model.system.expand(...)`` and
``model.propagators(t_max)`` with ``diag_R=False`` and ``diag_C=False`` in
the expansion. The first half of the field components are excitations, the
second half counts. Both are centred on their evolving means. The retarded
response and two-time covariance come from matrix exponentials for the
finite-history mean and covariance. The covariance is zero when either time
is at or before ``t_min``. The white covariance rate also enters the system
as a time-dependent impulse.

The supplied mark moments set the maximum derivative order :math:`m` in
the Kramers–Moyal generator. For each order the jump tensor contains the
joint raw moments of excitation and count increments. Multiplying that
tensor by the mean event rate gives the ``K<m>mean`` vertex for
:math:`m\geq3`. The quadratic mean-rate term supplies the Gaussian
propagator. Multiplying by the excitation fluctuation gives the
``K<m>rate`` vertex for :math:`m\geq2`. The Wick vertex spec supplies the
factorial and the Martin–Siggia–Rose phase.

For this linear rate law the order-zero Wick diagrams give the exact
second cumulant and the one-vertex diagrams give the exact third cumulant
before numerical quadrature. The finite-window readout takes signed
differences across all window endpoints and builds only the required jump
orders. Accessing ``model.system`` builds all supplied orders. The dense
joint jump tensors and matrix-exponential workspace grow with node count.
``workspace_max_bytes`` sets the preflight budget for each construction,
with a default of 64 MiB.
