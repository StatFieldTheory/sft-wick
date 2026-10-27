Analytic OU white cumulants
===========================

``OUWhiteCumulant`` evaluates the response contraction of a declared static
equal-time noise cumulant. Its amplitude is a finite real or complex tensor
``A[a1,...,am]`` of shape ``(N,)*m``, with ``m>=2``. It assumes a constant
diagonal Ornstein-Uhlenbeck response, nonnegative decay rates, a common finite
lower time bound and spatially constant amplitude. It does not recognize these
assumptions from an arbitrary callback.

The response product shares one noise time. With ``T=min(t_i)``,
``Delta=T-t_min`` and ``G=sum(gamma[a_i])``, the result is

.. code-block:: text

   A * exp(-sum(gamma[a_i]*(t_i-T))) * Delta * psi(G*Delta)
   psi(x) = -expm1(-x)/x,  psi(0) = 1

It is zero for ``T<=t_min``. Each absorbed white vertex removes one common time
integration dimension. It does not remove one independent time per leg.
Noncommuting matrix responses, time-dependent rates, spatial kernels and
automatic rewriting of a System are outside this adapter.

Explicit workflow construction
------------------------------

Use the System-aware factory to match the response actually used by the
workflow:

.. code-block:: python

   from dataclasses import replace
   import numpy as np
   from sft_wick import (
       DiagonalA, FieldSpec, GaussianNoise, OUWhiteCumulant,
       SeparableTranslation, ExponentialTemporal, ExponentialSpatial, System,
   )

   base = System(
       field=FieldSpec("phi", 2),
       linear=DiagonalA([0.7, 1.3]),
       noise=GaussianNoise(kappa2=SeparableTranslation(
           temporal=ExponentialTemporal(lam=0.2, sigma_t=0.3),
           spatial=ExponentialSpatial(sigma_x=1.0),
       )),
       t_min=0.2,
   )
   amplitude = np.arange(1, 9, dtype=float).reshape(2, 2, 2) / 7
   leaf = OUWhiteCumulant.from_system(amplitude, base)
   vertex = leaf.as_vertex("K", system=base)
   system = replace(base, nonlocal_vertices=(vertex,))

The factory accepts the native constant ``DiagonalA`` specification and checks
the component count, actual lowered rates and lower time bound. In particular,
``DiagonalA`` collapses rates considered equal by its existing ``allclose``
rule to the first component's scalar response; ``from_system`` snapshots that
actual response. Explicit responses, callable rates and overridden System or
DiagonalA subclasses are refused. ``as_vertex`` checks the supplied System
again. Revalidate if its response or time bound changes.

The returned vertex uses the existing ``already_R_contracted`` mechanism. The
leaf returns the **bare** cumulant; ``NonLocalVertex`` applies its MSR phase once.
Do not put that phase into the amplitude. Complex bare amplitudes are supported
by the primitive. In the existing workflow, each evaluated diagram or batch
must satisfy the real-projection check after observable rotation. Arbitrary
complex amplitudes may therefore be rejected; imaginary parts cannot simply
cancel between different diagrams at the end.

Standalone evaluation and costs
-------------------------------

.. code-block:: python

   direct = OUWhiteCumulant(amplitude, [0.7, 1.3], t_min=0.2)
   tensor = direct([0.0]*3, [0.8, 1.1, 0.9])  # shape (2,2,2)
   times = np.array([[0.8, 1.0], [1.1, 1.2], [0.9, 1.1]])
   batch = direct(np.zeros_like(times), times)  # shape (2,2,2,2)

Times have shape ``(m,)`` or ``(m,S)``. Spatial input must contain ``m`` legs;
its values do not affect this spatially constant tensor. The minimum time
creates coincident-time kinks, which the adapter declares to the integrator.
Scalar and batched inputs obey the same finite-time and support checks.

Inputs and prepared component sums use immutable snapshots; copy, deepcopy and
pickle retain this property. Default ``chunk_size=128`` bounds samples in a work
block. ``max_output_bytes=64*1024**2`` rejects an oversized amplitude or returned
batch. Work arrays scale as ``chunk_size*N**m``; this cap does not bound total
process memory, and input validation copies the ``m*S`` times. Large component
count or cumulant order still makes the amplitude itself expensive.
The ordinary finite-scale branch reuses its private work arrays while retaining
the same ordered operations, admission checks and scaled fallback. It leaves
the caller's arrays and immutable snapshots intact. When all component rates
have identical float64 bit patterns and ``N>1``, it computes the common time
factor over singleton component axes, then multiplies the complete amplitude.
This avoids repeated exponentials and factor scratch without assuming tensor
symmetry, separability or sparse support. The stored ordered rate sum and
leg-by-leg decay additions are retained. Single-component and heterogeneous
rates, including mixed signed zeros, keep the original factor dimensions;
the scaled fallback and full-component overflow checks remain unchanged.
The returned tensor and its output budget are still full-sized. Setup adds a
small rate-comparison cost, and tiny scalar calls need not become faster;
``tools/benchmark_ou_white_cumulant.py`` measures positive and negative
controls. Exact-byte probes describe the tested
environment, rather than guaranteeing every NumPy/platform combination.
A larger explicit
``chunk_size`` can reduce repeated small-block overhead, but increases work
memory; measure the full workload before selecting it. Block-wide admission
can choose different floating-point paths for mixed extreme samples, so
different chunk sizes do not carry a general bitwise-equivalence promise.
The returned full tensor batch and downstream dynamic contraction remain
allocated for all samples; this local work-array reuse does not stream the
QMC pipeline or bound its total memory.
The tensor order must leave room for one additional sample axis within the
installed NumPy version's dimension limit; the constructor rejects an order
that cannot support its documented batched output.

Stable removable limits and logarithmic scaling cover tiny rates, overflowing
intermediate rate/time sums and temporal underflow compensated by a large
amplitude. Real and imaginary parts are scaled independently, since a complex
magnitude can overflow while both components remain finite. A detected
unrepresentable final component raises ``OverflowError``; genuine final
float64 underflow is allowed. These evaluations are not interval certificates
of rounding errors near a range boundary.

``tests/test_ou_white_cumulant.py`` compares independent common-time quadrature,
high-precision extreme cases and actual nonzero raw/absorbed workflow channels.
``tools/benchmark_ou_white_cumulant.py`` compares raw equal-time and analytic
routes with the same bare amplitudes, components, external times and integrator.
It records refined numerical controls, full setup plus evaluation, repeated
evaluation and traced allocations. Run formal timings serially; ``--smoke``
does not establish performance acceptance. Traced peaks omit untracked native
allocations, and maximum RSS covers the whole process.

A lower integration dimension does not guarantee a shorter runtime. Small
Gauss-Legendre grids and mixed drift/cumulant channels can spend more time in
tensor preparation and adapter checks than they save in quadrature. Compare
setup plus evaluation, warm evaluations, numerical error and memory for the
same workload before selecting the analytic route. A QMC sample count in the
reduced dimension is a different approximation and must be checked against a
refined numerical reference, rather than assumed equivalent by sample count.

See :doc:`../api/white_cumulant` for the API reference.
