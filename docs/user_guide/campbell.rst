Marked Poisson sources
======================

A point source can affect several measurements through the same random
event. Give a ``CampbellDriver`` one row per event quadrature point. Its
weight is the expected number of events represented by that row, including
the rate and integration measure. Its entries are the responses of the
measurements to one event with unit mark. One independent draw of the mark
multiplies the entire row.

.. code-block:: python

   from sft_wick.campbell import CampbellDriver, ExponentialMark

   law = CampbellDriver(
       weights=[0.4, 0.8],
       responses=[[1.0, 0.2], [0.5, 1.3]],
       mark=ExponentialMark(),
   )
   phi = law.characteristic([1.0, -0.5])
   covariance = law.mixed_cumulant([0, 1])
   free = law.saturating_mean(output=0)

For event vectors :math:`h_e`, expected counts :math:`w_e`, and mark
:math:`M`, the continuous branch of the log characteristic function is

.. math::

   \log\phi(k)=\sum_e w_e\big(\mathbb E_M e^{iM k\cdot h_e}-1\big).

The default mark is one. For a discrete joint mark law, include one row per
mark outcome and multiply the event weight by its probability. Continuous
positions, times, and marks can likewise be integrated by quadrature. Refine
those nodes to check the continuum limit. The no-event probability
:math:`\exp(-\sum_e w_e)` contributes to every characteristic function.
With ``compensated=True``, the mean is subtracted from the output and the
no-event outcome is shifted by the same amount.

For joint affine or quadratic functions of Gaussian variables, the existing
:doc:`gaussian_quadratic` law provides the corresponding exact
characteristic function and mixed cumulants.

The positive observable
-----------------------

For an uncompensated nonnegative measurement :math:`X`,
``saturating_mean`` averages :math:`X/(1+X)` using the positive Laplace
transform of the Poisson law. It retains all orders in the event rate and
all powers of the jump. For a smooth observable supplied as a Fourier sum,
``fourier_expectation(frequencies, coefficients)`` averages that sum
directly. The caller controls the Fourier truncation and weights, including
any transform normalization.

A quadratic drift changes the final observation by
:math:`\epsilon \int R(t,s)Y_s Z_s\,ds` to first order. Build the joint
responses for :math:`(X_t,Y_s,Z_s)` at each insertion time. Then
``saturating_quadratic_response(1, 2, output=0)`` gives
:math:`\mathbb E[Y_sZ_s/(1+X_t)^2]`. Multiply this value by the retarded
kernel, the quadratic coupling and the time quadrature weight, and sum over
insertion times. Split the source integration at :math:`s` when the response
starts at the event time. For example, a single quadrature insertion with
weight 0.03 and response 0.9 contributes the following coefficient.

.. code-block:: python

   coefficient = 0.03 * 0.9 * law.saturating_quadratic_response(1, 1, output=0)

For Fourier observables, ``quadratic_characteristic(k, left, right)`` returns
:math:`\mathbb E[Y Z e^{i k\cdot X}]`. Multiplication by the corresponding
:math:`i k_j` gives the derivative of the Fourier mode with respect to the
perturbed output. These insertions include two responses from a single event
as well as contributions from different events. The nonlinear calculation
retains only terms through first order in the drift coefficient.

Custom marks
------------

A mark object supplies a vectorized ``increment(z, *, compensated)``
returning :math:`\mathbb E[e^{zM}]-1` or, for compensation,
:math:`\mathbb E[e^{zM}]-1-z\mathbb E[M]`. This can implement a known
characteristic function even when higher moments diverge. Compensation
requires a finite mean. Cumulants and quadratic insertions also use
``tilted_moment(order, z)``, which returns
:math:`\mathbb E[M^{\mathrm{order}} e^{zM}]`. A compensated mark can supply
``tilted_first_increment(z)`` for a more accurate evaluation of
:math:`\mathbb E[M(e^{zM}-1)]` near zero. The positive readouts require
``nonnegative = True`` and a real Laplace transform for nonpositive real
arguments. A custom law should report an error when a requested transform or
moment does not exist.

Numerical range
---------------

For very large or small rates or response amplitudes, the positive readouts
integrate in the logarithm of the Laplace variable. For the quadratic
response, the two powers of that variable are combined with the two
response legs before the result is rounded. This retains a finite answer
when the unweighted squared amplitude exceeds the floating-point range.

For the built-in marks, ``characteristic`` evaluates a large deterministic
centering phase as a separate rotation. ``quadratic_characteristic`` retains
the decay of the Fourier mode until it is combined with the signed
insertion. Use these functions directly when the unwrapped phase is large.
The imaginary part returned by ``log_characteristic`` has binary64
precision, which can lose whole radians and overflows if the unwrapped
phase is outside that range. The built-in
unit and exponential marks also support a projected event angle outside
binary64 when the requested Fourier readout fits. Unit marks use a reduced
phase, and exponential marks retain the logarithm of the projected angle
when taking inverse powers. Custom marks must accept representable Fourier
arguments and supply representable transforms and tilted moments.
