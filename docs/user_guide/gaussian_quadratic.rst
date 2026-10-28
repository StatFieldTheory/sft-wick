Finite Gaussian quadratic observables
=====================================

``GaussianQuadratic`` prepares the probability law of an explicitly declared
real observable of a finite real Gaussian input:

.. math::

   z\sim N(\mu,C),\qquad Y=b+\ell^{\mathsf T}z+z^{\mathsf T}Az.

The symmetric part of :math:`A` defines the same observable. Preparation uses an
exactly symmetric :math:`A` unchanged, subnormal entries included, and replaces
any other :math:`A` by its symmetric part with each entry rounded once to
float64. :math:`A` is converted to float64 first, so an integer entry beyond
``2**53`` or a wider float is rounded at that conversion too, and the
statements below are about the converted matrix. Each rounded entry is then within ``2**-53`` times the largest
magnitude in :math:`A` of the exact one. The bound is relative to that largest
entry: a mixed cumulant that another observable reaches only through a
subnormal pair of :math:`A`, beside entries of normal size, can lose all its
digits. That bound cannot hold when every
entry of :math:`A` is below ``2**-1022`` in magnitude, and preparation then
raises ``ValueError`` if an entry of the symmetric part is not a float64
value: rounding it can change it by up to its whole value, to zero included.
Multiplying :math:`A` by a power of two is exact and clears the refusal once
its largest entry reaches ``2**-1022``. Negative quadratic
eigenvalues are allowed; the observable need not be positive. The covariance
must be positive semidefinite. Physical Gaussian input is required: a formal
joint MSR physical/response propagator block is not an ordinary sampling
covariance.

This adapter sums exactly the rings and chains of the declared quadratic
observable, within floating-point preprocessing and evaluation error. It does
not extract an observable from ``Expansion``, replace a full nonlinear dynamics
calculation or include missing higher-response terms such as :math:`X_1^3X_3`.
It does not calculate a density or certify a PDF inversion.

Preparation and source reuse
----------------------------

.. code-block:: python

   import numpy as np
   from sft_wick import GaussianQuadratic

   covariance = np.array([[1.2, 0.25], [0.25, 0.7]])
   quadratic = np.array([[0.15, 0.08], [0.08, -0.12]])
   law = GaussianQuadratic(covariance, quadratic, [0.4, -0.3],
                           constant=0.2, mean=[0.12, -0.17])
   sources = np.linspace(-0.3, 0.3, 1000)
   cgf = law.cgf(sources, chunk_size=64)
   centered_cgf = law.cgf(sources, centered=True, chunk_size=64)
   log_phi = law.log_characteristic(np.linspace(-10, 10, 1000))
   coefficients = law.taylor_coefficients(4)  # entries for powers t**0 .. t**4
   one_coefficient = law.taylor_coefficient(4)

Inputs are snapshots; later edits to input arrays cannot change a prepared law.
Shallow/deep copies share the immutable law, and pickle reconstruction restores
immutable array backing.
The prepared numeric state consists of eigenvalues and rotated linear weights,
plus scalar mean metadata, requiring :math:`O(r)` storage. Source evaluation costs :math:`O(r)` per source.
``chunk_size`` bounds broadcast rows, so temporary source memory is
:math:`O(\text{chunk_size}\,r)`. Preparation still requires dense matrices and
cubic work; a one-shot small problem may gain nothing from preparation.

Covariance validation is deliberately strict: the supplied matrix must be
exactly symmetric and its computed eigenvalues must be nonnegative. No negative
or small positive eigenvalue is clipped. An exactly singular covariance can
acquire a negative eigensolver result from roundoff and be rejected. If a
covariance factor is known, prefer the explicit factor route:

.. code-block:: python

   factor = np.array([[1.0], [2.0]])
   singular_law = GaussianQuadratic.from_factor(
       factor, np.eye(2), [1.0, -1.0], mean=[0.2, 0.3])
   # This declares C = factor @ factor.T; no covariance eigensolver is needed.

The factor may have zero or dependent columns. ``latent_dimension`` counts its
columns and is not an assertion of matrix rank. Factor construction also copies
the inputs. Rank-zero input gives a deterministic observable.

Rotated linear endpoints are stored as squared float64 weights. A nonzero square
must be a normal float64 value: preparation rejects zero/subnormal underflow and
square overflow. This avoids large relative errors from rounded subnormal
weights. Rescale the observable and inversely rescale its sources when needed;
the adapter does not silently replace such a linear law by a constant.

Formula and domain
------------------

Set :math:`C=LL^{\mathsf T}`,
:math:`b_0=b+\ell^{\mathsf T}\mu+\mu^{\mathsf T}A\mu`,
:math:`B=L^{\mathsf T}AL`, and
:math:`h=L^{\mathsf T}(\ell+2A\mu)`. With
:math:`B=U\Lambda U^{\mathsf T}` and :math:`v=U^{\mathsf T}h`,

.. math::

   K(t)=b_0t-\frac12\sum_j\log(1-2t\lambda_j)
        +\frac{t^2}{2}\sum_j\frac{v_j^2}{1-2t\lambda_j}.

``real_domain`` is an open interval; every :math:`1-2t\lambda_j` must be
positive. Endpoints and outside sources raise ``ValueError``. A complex source
must have its real part in the same interval. Characteristic evaluation uses
:math:`t=i\omega` and individual spectral logarithms anchored at zero; taking
the principal logarithm of a determinant product can wrap the phase incorrectly.
``log_characteristic`` preserves this continuous branch. ``characteristic`` can
underflow to zero for very small magnitudes.

Centering subtracts the law's first moment and preserves higher cumulants. Its
ring term uses a small-argument series to avoid cancellation near zero. For
small-argument rows, uncentered evaluation combines the same stable ring with
the accurately summed, source-scaled first moment. Larger rows use the raw
spectral logarithms to avoid artificial cancellation from a large centering
term. Cumulants obey

.. math::

   \kappa_1=b_0+\operatorname{tr}B,\qquad
   \kappa_n=2^{n-1}(n-1)!\operatorname{tr}(B^n)
           +n!2^{n-3}h^{\mathsf T}B^{n-2}h\quad(n\ge2).

The Taylor coefficient for :math:`t^n` is :math:`\kappa_n/n!`. Cumulants and
Taylor coefficients first combine exactly equal-absolute-eigenvalue groups, then
combine signed scaled spectral powers in log space before
deciding whether the final result fits float64. They do not separately convert
an overflowing factorial or form an underflowing unscaled power. A true final
overflow raises, and a true final underflow can return zero. Severe signed
cancellation and near-pole sources can still require higher precision. Positive
covariance eigenvalues are all retained by default.

First moments use exact binary-float summation before final conversion, so a
finite signed trace is not lost to an overflowing intermediate sum. A true
unrepresentable first moment raises explicitly from its float64 accessor.
A finite small-source product can still be evaluated when that accessor would
overflow; the zero-source anchor remains valid.

Severe signed cancellation between distinct spectral groups triggers a targeted
exact binary-rational fallback for the prepared spectral data. A work bound
prevents unbounded integer-power allocation; an excessively large cancellation
problem raises ``ArithmeticError`` asking for higher precision. This does not
certify the preceding floating-point eigensystem or represent an arbitrary
precision implementation of the whole adapter.

Whitening, nonzero-mean shifts, and projected matrices/endpoints are prepared
with float64 matrix operations. Ill-conditioned input cancellations can lose
small contributions during that preparation. The exact fallbacks protect
selected contractions of prepared data; they do not certify those matrix
operations or guarantee accurate results for arbitrarily conditioned inputs.

``taylor_coefficient(n)`` calculates one coefficient without evaluating any
lower-order terms. It can return a finite odd coefficient even when lower even
coefficients overflow. ``taylor_coefficients(max_order)`` requires every entry
in the returned array to be representable.

Quadrature conventions
----------------------

``from_weighted_grid(C, A, weights, ell, ...)`` accepts sampled *colored nodal
covariance*, kernel coefficients and strictly positive quadrature weights. Its
finite observable is :math:`b+\ell^{\mathsf T}Wz+z^{\mathsf T}WAWz`, with
:math:`W=\operatorname{diag}(w_i)` and covariance :math:`C`. The product
:math:`WAW` is formed in float64 and prepared as the quadratic matrix. Its two
triangles round in a different order and can differ in the last bit even for a
symmetric :math:`A`; it is then symmetrized like any other quadratic matrix
that is not exactly symmetric.

``from_white_noise(intensity, A, weights, ell, ...)`` instead accepts nonnegative
white-noise intensity :math:`c_i`. Its nodal covariance is
:math:`\operatorname{diag}(c_i/w_i)`, yielding
:math:`B_{ij}=\sqrt{w_ic_i}A_{ij}\sqrt{w_jc_j}` and
:math:`h_i=\sqrt{w_ic_i}\ell_i` for zero mean. The two conventions are distinct.
For component-valued grids, repeat weights consistently with the flattened
component ordering. Means in either constructor are raw nodal means.
White-noise preparation uses the equivalent variable :math:`x=Wz`, with factor
:math:`\operatorname{diag}(\sqrt{w_i}\sqrt{c_i})` and mean :math:`W\mu`.
It avoids forming a potentially underflowing :math:`c_i/w_i` ratio. An
unrepresentable weighted factor or mean is rejected; a positive weighted factor
must be a normal float64 value.

These constructors establish a finite law, not a continuum error certificate.
The symmetric Nyström weighting and regularity-dependent determinant convergence
are discussed in `Bornemann, Sections 2–6 <https://arxiv.org/pdf/0804.2543>`_.
Analytic-kernel convergence rates do not automatically apply to kinked kernels.

Explicit spectral approximation
-------------------------------

.. code-block:: python

   approximate = law.compress(retain_modes=1)
   approximation = approximate.cgf(0.2)
   conditional_bound = approximate.cgf_error_bound(0.2)

Compression is opt-in and keeps the largest absolute quadratic eigenvalues. It
replaces omitted modes by an independent Gaussian with their exact spectral
mean and computed spectral variance. Both linear tail variance and zero-eigenvalue linear modes
are included. Compression preserves total mean and variance, remains a valid
probability law, and retains no parent or full-spectrum arrays. It saves repeated
source work, while initial preparation still computes the full dense spectrum.
Compression is refused when a required tail summary overflows or underflows its
float64 calculation, rather than returning a law with invalid tail metadata.
The tail's exposed ``mean`` is a rounded diagnostic. Separate exact binary-sum
scalar metadata preserves the original total first moment even when rounded
tail and retained means would otherwise cancel away a small remainder.
Variance preservation and the following error bound are subject to the stated
floating-point spectral-summary error.

For omitted modes let :math:`\delta=\max|\lambda_j|`,
:math:`T_2=\sum\lambda_j^2`, :math:`H_0=\sum v_j^2`. If
:math:`q=2|t|\delta<1`,

.. math::

   |K(t)-K_{\rm approx}(t)|\le
   \frac{|t|^3\delta}{1-q}\left(\frac43T_2+H_0\right).

The bound is conditional on spectral summaries computed in float64; it is not an
interval-certified bound on eigensolver, matrix preprocessing, continuous
quadrature, nonlinear closure or density errors. Invalid bound sources raise.
``real_domain`` describes the approximate law and can expand after compression;
``reference_real_domain`` retains the original prepared law's interval. Further
compression of an already approximate law is refused. Higher polynomial Taylor
corrections alone need not define a valid probability law.

Correlated observables
----------------------

``GaussianQuadraticFamily`` accepts quadratic shape ``(m,N,N)``, optional linear
shape ``(m,N)``, optional constant shape ``(m,)`` and one shared input mean and
covariance. Its factor constructor handles the same singular-input contract.

.. code-block:: python

   from sft_wick import GaussianQuadraticFamily

   family = GaussianQuadraticFamily(covariance, [quadratic, np.eye(2)])
   direction_law = family.project([1.0, -0.1])
   sweep = direction_law.cgf(sources)
   covariance_01 = family.mixed_cumulant([0, 1])
   coefficient_22 = family.joint_coefficient([2, 2])

Projection prepares a fixed real source direction once. ``family.cgf(J)`` accepts
one real joint source vector and ``family.log_characteristic(omega)`` accepts
one real frequency vector; each forms and diagonalizes its projected matrix.
Different matrices need not commute, so one common eigensystem is not assumed.
General complex joint source vectors are outside this family interface.

Mixed cumulants retain the full slot-permutation sum and repeated-index
multiplicities. Mixed cumulants and monomial coefficients are supported through
total order four; higher orders raise. A monomial coefficient divides the mixed
cumulant by the product of the multi-index factorials.

Every returned mixed cumulant and joint coefficient is one of two things, with
respect to the prepared matrices and endpoints: the float64 contraction, kept
only when a rigorous bound on its rounding error is at most ``1e-12`` of its
magnitude, or the correctly rounded exact value. ``ArithmeticError`` is raised
when the bound does not admit the float64 value and the exact evaluation
exceeds its work or memory limit (given below). The evaluation sets its own
NumPy error state, so a call inside ``np.errstate(all="raise")`` returns the
same value.

Unless the exact evaluation is refused (see below), a float64 value is also
kept only where version 0.7.0 kept it: the contraction raised no
floating-point exception, and its magnitude is above ``1e-10`` times the
Frobenius-norm bound that 0.7.0 used to detect cancellation. Every value
that 0.7.0 returned is therefore returned with the same bits or as the
correctly rounded exact value, except where the bound does not admit it and
the exact evaluation exceeds its limit; that query raises ``ArithmeticError``.
When the exact evaluation is refused, a float64 value that the bound admits is
returned even if it raised an underflow or the 0.7.0 test flags it. Version
0.7.0 raised ``ArithmeticError`` on all of these queries, because its own
exact evaluation stopped at ``1e6`` operations.

The float64 contraction sums its signed traces and endpoint terms exactly; its
matrix products are ordinary float64 products. The bound follows the
componentwise error analysis of matrix products (Higham, *Accuracy and
Stability of Numerical Algorithms*, chapter 3) and is formed from the
elementwise absolute values of the matrices and endpoints, with an allowance
for underflow and flush-to-zero. It does not rely
on floating-point exception flags, which BLAS matrix products do not reliably
report. The model assumes matrix products that form each entry as a sum of its
products in some order, with or without fused multiply-add, as OpenBLAS, MKL,
Accelerate and NumPy's own loops do. The bound costs one absolute contraction
per class of words with equal absolute value, and the admission decision is
made in exact rational arithmetic. On one core the bound took 25 to 120
microseconds per query at ``N = 4`` and ``N = 32``. A query at ``N = 4`` takes
0.06 to 0.5 milliseconds, so admitted queries there were about 20 percent
(order four) to 70 percent (order two) slower than in version 0.7.0. They were
4 to 25 percent slower at ``N = 32`` and 3 to 14 percent slower at ``N = 128``
and ``N = 256``.

The bound is about ``(n - 1) N u`` times the same statistic built from the
absolute matrices and endpoints, for order ``n``, latent dimension ``N`` and
``u = 2**-53``. A float64 value is therefore kept only when its magnitude is
at least about ``1.1e-4 (n - 1) N`` times that absolute statistic. Dense
indefinite matrices, and weakly correlated observables, fall below this from
``N`` of a few tens at orders three and four and from ``N`` of a few hundred at
order two; in a test family of dense positive semidefinite matrices,
fourth-order queries fell below it from ``N = 64`` to ``128`` and third-order
queries from ``N = 500``.

A mixed cumulant whose absolute contractions all vanish, for example between
observables built on disjoint blocks of the input, is a proved zero and returns
``0.0`` without a signed contraction. A vanishing absolute contraction is
accepted only when every term is shown to be at least ``2**-1000``, or when the
zero patterns of the matrices show that no term exists; a term that underflowed
to zero is not taken for a zero. The pattern test does not depend on the
magnitudes, so it also proves zeros whose entries are too large or too small
for the error bound.

Uncertified values are evaluated exactly. These include zeros forced by a
symmetry, small overlaps between blocks and strongly cancelling contractions.
Each prepared array is split into integer digit slices of ``w`` bits on a
common power-of-two grid, with ``w = (53 - ceil(log2 N)) // 2``. Every partial
sum of a product of two slices is then an integer below ``2**53``, so a
float64 matrix product (BLAS) forms it exactly in any summation order, with or
without fused multiply-add. Each such product is compared with a checksum
modulo ``2**64`` computed in NumPy's integer loops; a mismatch, which only a
product outside the classical model would produce, issues a
``RuntimeWarning`` and the product is recomputed in NumPy's int64 loops.
Traces and matrix-vector products of slices use NumPy's own loops.

The exact evaluation first uses only the leading bits of each array. Every
matrix and endpoint keeps the bits within ``k`` binades of its largest entry,
with ``k`` about 64 plus the binary logarithm of ``N`` times a product of
absolute row and column sums over the magnitude of the float64 value. The
dropped bits of an entry are smaller than one power of two, so the dropped
part of the statistic is bounded by that power times ``N`` and the same sums.
The statistic of the kept bits is evaluated exactly and returned when both
ends of this bound round to the same float64; that float64 is then the
correctly rounded exact value. Otherwise a second attempt keeps 64 more
binades, and a third evaluates the full arrays. For dense matrices whose
entries lie within a few binades of each other nothing is dropped and the
first attempt is the full evaluation. An underflow inside the float64
contraction, or the 0.7.0 cancellation test, also leads to the exact
evaluation. Times on one core of a full query (bound, float64 contraction and
exact evaluation) on dense indefinite matrices, with a distinct observable in
every slot:

.. list-table::
   :header-rows: 1

   * - ``N``
     - order 2
     - order 3
     - order 4
   * - 256
     - 0.005 s
     - 0.014 s
     - 0.05 s
   * - 500
     - 0.024 s
     - 0.07 s
     - 0.29 s
   * - 1000
     - 0.09 s
     - 0.41 s
     - 1.6 s
   * - 2000
     - 0.42 s
     - 2.3 s
     - 9.6 s

The float64 contraction alone takes about a sixth (order two) to two fifths
(order four) of these times. The bound holds the absolute values of the
distinct matrices and, at order four, up to two absolute pair products. The
exact evaluation holds the int32 slices of the distinct matrices, up to two
exact pair products and seven dimension-square scratch arrays; a fourth-order
query at ``N = 1000`` peaked at about 165 MiB.

Matrices with entries over many binades, such as couplings that decay with
distance, need far fewer slices for their leading bits than for all of them.
Times on one core of full queries on four random-sign symmetric matrices
multiplied by ``exp(-|i - j| / l)``, none of them certified:

.. list-table::
   :header-rows: 1

   * - ``N``
     - ``l``
     - smallest entry
     - ``(0, 1, 2, 3)``
     - ``(0, 0, 1, 1)``
     - ``(0, 1, 2)``
   * - 1200
     - 3.07
     - 1.5e-172
     - 3.2 s
     - 1.9 s
     - 0.75 s
   * - 1500
     - 3.84
     - 3.7e-172
     - 5.5 s
     - 3.4 s
     - 1.3 s
   * - 2000
     - 20
     - 1.1e-48
     - 12 s
     - 5.6 s
     - 2.7 s

The leading bits took five or six slices per matrix where the full arrays
take 30 (``N = 1500``) and 10 or 11 (``N = 2000``). The float64 contraction
alone takes 0.14 to 3.4 s for these queries.

Before forming any product the exact evaluation estimates its work and its
peak memory from the slice counts, following its evaluation order. It raises
``ArithmeticError`` above ``1e13`` weighted multiply-adds (BLAS multiply-adds
plus 64 for each multiply-add in NumPy's own loops) or 4 GiB at the peak.
Dense matrices whose entries carry full 53-bit mantissas need three or four
slices; for them the memory limit is reached at about ``N = 4500`` for
fourth-order queries with four distinct observables, ``N = 5350`` at order
three and ``N = 9450`` at order two. At those sizes the float64 contraction
alone would still return a value, but not a certified one. Each attempt
checks its own work and memory before forming a product, and the refusal
applies to the first attempt that exceeds them. The slice count of a full
evaluation grows with the range of binary exponents among the entries of an
array, so arrays whose entries span hundreds of binades are refused at much
smaller ``N`` when the full evaluation is needed. It is needed when the leading
bits cannot decide the rounding: for an exact zero that the zero patterns do
not prove, such as a zero forced by a symmetry, for a value within the bound
of a point halfway between two float64 values, when the float64 contraction
overflows and gives no magnitude, and when an absolute row or column sum of an
array overflows. It can also be needed for a strongly cancelling value whose
float64 contraction is more than about ``2**64`` times larger, because the
first attempt then keeps too few bits. Joint coefficients divide before
deciding final float64 range, so a coefficient can be finite even when its
unscaled cumulant overflows.

The guarantee concerns the prepared matrices and endpoints. Whitening,
projection, the covariance eigendecomposition and the symmetrization of a
quadratic matrix that is not exactly symmetric are float64 operations whose
errors it does not cover. That symmetrization rounds each entry of
:math:`(A+A^{\mathsf T})/2` once. At or above ``2**-1022`` in magnitude the
error is at most ``2**-53`` of the entry, and below it at most ``2**-1075``,
to zero included. A declared entry below ``2**-1022`` that float64 cannot hold
is refused only when no entry of its matrix reaches ``2**-1022``; an entry of
the whitened matrix is never refused. An exactly symmetric matrix is not
rounded.

Reuse of selected queries
~~~~~~~~~~~~~~~~~~~~~~~~~

Repeated indices can produce identical ordered matrix words. The native mixed
calculation computes each such word once and appends its scalar terms for every
original permutation, in the same order. It does not identify cyclic or reversed
words, reorder noncommuting matrices, or multiply a rounded sum by a repetition
count. The rounding bound and the exact evaluation do group cyclic shifts, and
reversals of symmetric matrices, because absolute traces and exact traces are
unchanged by them; each group is weighted by its number of slot permutations.

For native float64 fourth-order queries with at least three distinct observable indices,
latent dimension at least 256 and one prefix matrix no larger than 8 MiB, the
calculation can also retain the last identical ordered two-matrix product.
The latent dimension counts factor columns, not algebraic rank; the byte cap
permits dimensions through 1024.
Later factors retain their original left association, and scalar terms retain
their original order. At most one prefix matrix is
retained within that query; it is released on return. Smaller queries,
two-observable patterns, oversized prefixes and custom array/family protocols
use the original route. This trades up to one additional dimension-square work
matrix for fewer dense products; it does not promise lower peak memory or
uniformly faster queries. Independently of the prefix, the rounding bound of
every native query holds the absolute values of its distinct matrices and, at
order four, up to two absolute pair products.

For a known set of queries used repeatedly, preparation is explicit:

.. code-block:: python

   selected = family.prepare_statistics(
       indices=[(0, 0, 1, 1), (1, 0)],
       coefficients=[(2, 2)],
   )
   covariance_10 = selected.mixed_cumulant((1, 0))
   coefficient_22 = selected.joint_coefficient((2, 2))

``GaussianQuadraticStatistics`` snapshots each requested public result as a
float64 scalar. It retains neither the parent family nor dense contractions.
Ordered cumulant keys and full Taylor-power keys are separate; preparation
computes coefficients independently, so a finite coefficient remains available
when its corresponding cumulant overflows. Builtin numeric ``OverflowError``
and ``ArithmeticError`` outcomes preserve their type and message on lookup.
Invalid keys fail during preparation; a valid unselected key raises ``KeyError``.
Copy, deepcopy and pickle preserve the immutable snapshots.

The native family is required: subclass query overrides need not be pure.
Preparation caps distinct entries at 1,024 (configurable downward with
``max_entries``), with a separate 65,536-integer key budget and a 65,536-query
submission limit before deduplication. Supplied sequences must have actual
lengths matching their declarations; consumption is bounded. Its time and
memory cost must be included for one-shot use. A warm lookup does not execute a
matrix contraction.

This interface supplies selected individual statistics, not a directional
polynomial. For example, separately rounded means for
``A0=diag(1e16,1)`` and ``A1=diag(-1e16,0)`` can sum to zero even though the
direct projected mean for direction ``[1,1]`` is one. A fixed direction should
still use ``family.project(direction)`` and retain that returned law.

Validation and performance evidence
-----------------------------------

``tests/test_gaussian_quadratic.py`` checks scalar connected public Wick pairings,
matrix covariance routing, independent Gauss-Hermite integrals, noncommuting
joint cumulants and numerical/input boundaries. The central quadratic trace
identity is independently documented by
`Lancaster (1954) <https://academic.oup.com/jrsssb/article/16/2/247/7026669>`_.
The linear term and tail inequality follow by Gaussian square completion and
power-series remainders.

``tools/benchmark_gaussian_quadratic.py`` records source/input hashes and compares
the prepared spectrum against the same finite law evaluated using one Cholesky
factorization and triangular solve per real source. It separates preparation,
first calls, repeated sweeps, retained numeric storage and traced allocations.
Run formal timings serially with thread counts pinned. ``--smoke`` is a tiny
correctness control and does not establish performance acceptance. Traced peaks
exclude untracked native BLAS allocations; process maximum RSS is global.

``tools/benchmark_joint_statistic_reuse.py`` compares native ordered-word reuse
against the earlier implementation archived in ``tests/fixtures`` (unpacked to
a temporary directory unless ``--reference-src`` names another copy) and
compares selected-statistic lookup against direct queries. The shared parent family is prepared outside query
timings. Bank preparation, warm lookup and an actually timed combined
preparation-plus-query interval are reported separately. It includes repeated
and all-distinct fourth-order keys, low orders and one-shot use; low-order or
one-shot queries can lose to their preparation overhead. Admission compares
float64 result bits and builtin numeric error outcomes. Formal measurements
use six balanced paired repetitions per fresh process; retain three processes
and all raw repetitions when judging performance.

See :doc:`../api/quadratic` for the API reference.
