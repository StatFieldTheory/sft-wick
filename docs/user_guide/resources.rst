Choosing time and memory tradeoffs
==================================

Resource controls keep the requested model, method and integration resolution
fixed. Smaller blocks reduce some temporary arrays while adding dispatch
overhead; retaining static preparation can save repeated work while using more
memory. There is no universally optimal block size.

Existing controls
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Sweep option
     - Scope
     - Tradeoff
   * - ``gl_chunk_size``
     - GL coordinates, mapped times and evaluated factors
     - Default 4096 nodes per block; a grid no larger than one block is
       the eager grid.  Smaller blocks reduce workspace; all nodes are still
       evaluated.  Each block has a fixed Python cost: on an order-4 sweep,
       blocks of 512 nodes took twice as long as eager GL and blocks of 64
       ten times.  See :ref:`gl-batch-memory` for choosing the value.
   * - ``qmc_coupling_chunk_size``
     - Dynamic coupling construction in vectorized QMC only
     - Smaller raw tensor batches; full Sobol coordinates, R/C arrays,
       contracted outputs and global statistics remain allocated.
   * - ``reuse_integrands``
     - Prepared static integrands per component tuple within one sweep
     - Retains static tensors to amortize preparation over grid points;
       dynamic diagrams retain ordinary preparation.

QMC chunking requires sample-separable callbacks with stable dtype and trailing
shape. Results must not depend on batch size, invocation count or mutable state;
``coupling_vectorized: true`` alone does not establish this contract. Chunking
changes callback grouping and can change native output-budget admission or
exception order. None of these options bounds process RSS, model-owned caches,
callback-retained arrays or worker memory.

``run`` and ``run --dry-run`` print each requested control's scope and
configuration-known status. ``conditional`` means diagrams, cache dispatch or
integration dimensions still determine use; it is not an execution measurement.
The summary invokes no additional callbacks. ``expand.n_jobs`` parallelizes
diagram integration inside each grid point; ``sweep.n_jobs`` parallelizes grid
points, with serial fallback for at most two points. ``propagators.n_jobs``
applies to eager C-table construction; lazy tables are built serially. Workers
can retain their own model, caches and workspaces. Normal ``run`` still supports
these parallel modes.

Explicit comparison
-------------------

Create a variants YAML beside an existing workflow. For the packaged quickstart
configuration, a static preparation comparison is::

    reuse:
      reuse_integrands: true

Run the explicit experiment::

    sft-wick compare-resources quickstart.yaml \
      --variants variants.yaml --json resource-report.json

``baseline`` is automatic and preserves the original resource settings. A
variant may change only ``reuse_integrands``, ``gl_chunk_size``,
``qmc_coupling_chunk_size``, ``propagator_cache_bytes``,
``contraction_backend`` and ``contraction_max_bytes``. It cannot change the model, seed, sample count,
nodes, method, axes or diagram filters. Names must be unique and cannot be
``baseline``. Null chunk caps and boolean reuse flags are accepted; chunk caps
must otherwise be positive integers.

For a workflow already using ``method: gauss_legendre``, compare an explicitly
chosen block size::

    smaller_blocks:
      gl_chunk_size: 128

GL block reductions can change floating-point rounding. Declare a numerical
budget before running, for example::

    sft-wick compare-resources gl-config.yaml \
      --variants gl-variants.yaml --json gl-resource-report.json \
      --rtol 1e-12 --atol 1e-14

These tolerances illustrate the syntax; they are not a certified budget for
every observable. Cancellation can require a scientifically justified absolute
tolerance. QMC requires a fixed seed. Initially the comparison tool requires
``expand.n_jobs = propagators.n_jobs = sweep.n_jobs = 1`` and does not compare
multiprocessing. It never chooses a winner or modifies the original config.
Existing reports require ``--force`` to replace. Declared inputs and source files
are protected even with force.
Reports are installed atomically without following file aliases created during
the experiment. A changed report directory is rejected. Existing ordinary report
permissions are retained; new reports use private permissions. This initial
comparison tool requires directory-relative atomic filesystem operations and
rejects unsupported platforms. Standard ``run`` has no such requirement.

Measurement and acceptance
--------------------------

The default two rounds use forward then reverse case order. Every timing pass
and separate memory pass runs in a new process, serially. The execution count
is ``2 * repeats * (1 + number_of_variants)``; ``--repeats`` must be at least two.
User modules and callbacks are loaded and executed repeatedly. They must be
deterministic and suitable for repeated execution. This tool is not a sandbox
and cannot prevent arbitrary module or callback side effects.

Each run disables configured outputs and expansion/propagator disk caches.
It measures a workflow from scratch, including system building, expansion,
propagators, sweep and totals, without output serialization. Configuration
loading is recorded separately; the report also gives loading-plus-workflow
time, excluding process startup. Timing passes disable tracing; memory passes
trace configuration loading plus workflow. Traced peaks are not total memory.
Fresh-process RSS is the lifetime high-water mark, including imports and config
loading; it is not an incremental query peak, and baseline RSS is not subtracted.
Unsupported platforms report null RSS. Encoding occurs after all measurements.

The report records full ordered sweep rows, three axis metadata sets, ordered
total columns/records, execution order, versions, platform, active package/source
information and relevant environment values. Workers prepend the active source
tree to ``PYTHONPATH``, disable bytecode writes and startup tracing, redirect
bytecode reads to a fresh empty cache prefix, and supply
``PYTHONHASHSEED=0`` only when unset. Thread settings are preserved. An explicit
``PYTHONHASHSEED=random`` remains random and may fail the strict stability gate.

All results within each case, including memory passes, must be exactly stable.
Against baseline, row metadata and ordering must match exactly; each diagram's
value and error and each total value are compared separately. Default tolerances
are zero and require exact equality, including the sign of floating zero.
Reports distinguish exact equality from equivalence within an explicit finite
nonnegative tolerance. Empty rows, nonfinite results, timeouts, worker failures,
known input changes or failed result gates produce a failure report and nonzero
CLI status. Ratios are withheld if any gate fails; matching totals alone is
insufficient.

The original YAML and variants text, declared local module files, coupling array
files and all Python files in the active source tree are hashed before and after
runs. Workers independently verify these known inputs. Unknown imports,
dependency data and external state are not fully frozen. A finite experiment
establishes this configuration's repeatability and measured tradeoff, not a
general speed guarantee or a scientific convergence certificate.

Explicit propagator batch cache
-------------------------------

For a fixed model, repeated propagator batch lookups may be cached with an
explicit byte budget. Add the following fields to the existing ``sweep`` block::

    sweep:
      # Existing positions, times, components, method and resolution go here.
      propagator_cache_contract: fixed_model
      propagator_cache_bytes: 16777216

``propagator_cache_bytes`` is null by default; null and zero disable this cache.
Otherwise it must be a nonnegative integer. A positive budget requires
``propagator_cache_contract: fixed_model``. The declaration is a promise from
the caller: model parameters, response/correlation data and all callbacks read
by the propagator must remain fixed while the cache is active. Equal arguments
must produce equal native numerical outputs. Callbacks must not rely on call
counts, random draws, mutable state or side effects. A dataclass, vectorized
callback flag or read-only array alone does not establish this promise.

The cache uses exact argument representations, including array dtype, shape and
contents; it does not round coordinates or use nearby-time matches. Stored
arrays are privately copied, and cache hits return separate copies preserving writeability.
Changing a returned array cannot change a later hit. Unsupported input/output
protocols follow the ordinary lookup path. Entries use a least-recently-used
policy; an entry larger than the cap is evaluated without being retained.

The budget covers retained native input-key and output-array bytes. It does
not bound Python metadata, temporary key/output copies, prepared tensors,
propagator tables, model data or process RSS. Process workers have independent
caches and budgets. In serial execution the lifetime is one sweep or one
``evaluate`` call. Serialization clears active entries, so process workers
start with empty task/batch-local caches. Contents are not persisted to disk
or reused between models or separate invocations.
This cache can save repeated lookups but exact key construction and copying
also add work. A tiny or frequently evicted cache can be slower.

Keep the fixed-model declaration in the original configuration when comparing
budgets; variants cannot add or change that scientific promise. For example::

    cache_16_mib:
      propagator_cache_bytes: 16777216
    cache_disabled:
      propagator_cache_bytes: 0

Then run ``compare-resources`` with the same zero default tolerances. Cache-only
comparisons are expected to retain exact complete results for the declared
fixed model. Independent process RSS and tracing measurements remain necessary
to assess the actual memory cost.

Opt-in contraction backends
---------------------------

For a static coupling, the integrand of one diagram at one sample is a sum
over component assignments: the coupling coefficient of the assignment times
its C and R propagator values, the scalar R product and the Jacobian. The
default ``contraction_backend: ordered`` adds the assignments one at a time.
Two opt-in backends compute the same sum in a different order.

``einsum``
    Contracts the prepared coefficient tensor with the propagator batches,
    two operands at a time, along a path chosen to keep the operation count
    small. A diagonal C table (``diag_C``), an R whose nonzero entries lie on
    the diagonal and a fixed component index are first merged into the index
    structure, so entries that are zero by construction are not multiplied.

``factorized``
    Keeps the original coupling tensors of each monomial of the coupling sum
    (the ``F`` and ``κ`` factors) instead of the coefficient tensor over all
    propagator indices, and contracts every monomial the same way. This
    avoids building the coefficient tensor.

Either backend is selected in the ``sweep`` block::

    sweep:
      contraction_backend: einsum
      contraction_max_bytes: 67108864

or with ``contraction_backend=`` on :meth:`~sft_wick.workflow.Expansion.evaluate`,
:meth:`~sft_wick.workflow.Expansion.sweep`, ``integrate_moment`` and
``integrate_diagrams``. The choice changes only how each diagram's sum is
evaluated; samples, nodes, propagator tables and diagrams are the same.

Accuracy
~~~~~~~~

A different summation order rounds differently, so a backend result differs
from the ordered one in its last bits. Both satisfy the same bound. At each
sample let

* ``S`` be the sum over assignments of the absolute value of the
  coefficient times its propagator factors, where the coefficient is the one
  the ordered fold multiplies (for ``factorized``, the sum of the monomials
  at that assignment),
* ``A`` be the number of component assignments, ``T`` the number of monomials
  (``T = 1`` for ``einsum``) and ``m`` the number of factors in one product,
  and ``u = 2**-53``.

Then

.. math::

   |\mathrm{result} - \mathrm{exact}| \le 1.01\,(m + A T)\,u\,S ,

where "exact" is the sum computed in exact arithmetic from the same float64
inputs. The bound has no underflow term: a backend runs only when no product
of nonzero inputs can fall below ``2**-1021``, so an underflowing operation
loses less than ``u/2`` of the magnitudes it combines, and the ordered fold
does not underflow on these inputs either. The bound is relative to ``S``,
not to the result: when the products cancel, every summation order, the
ordered one included, can leave a large relative error in the result.
``tests/test_contraction_error_bound.py`` checks the bound against an exact
rational reference on generic signed diagrams.

``factorized`` contracts each monomial separately and adds the results. Its
error would grow with the sum of the monomial magnitudes, which exceeds
``S`` when monomials of opposite sign cancel at an assignment, while the
ordered fold adds the monomials into one coefficient first. A sum of
monomials is therefore prepared for ``factorized`` only when no two of its
monomials have opposite signs at any assignment; then the two sums are
equal and the bound above holds. The check runs once per diagram, on the
signs of the coupling tensors, assignment by assignment. When that would
take more than ``2**27`` element operations (the number of assignments
times the number of tensors plus one, summed over the monomials; for
example two monomials of three tensors over six indices of 17 components),
the sum is prepared only if the signs of its tensors alone settle the
question: every tensor holds entries of one sign apart from zeros, and
every monomial has the same resulting sign.

The largest errors observed on 330 diagrams captured from
:meth:`~sft_wick.workflow.Expansion.evaluate` (``N`` = 2, 3 and 5, signed
``F`` and static ``κ³``, distinct rates, orders 2 to 4, ``diag_C`` on and
off), with every diagram forced past the cost rule and compared with an
exact rational sum at three samples, are

============  ===================================  ==========
Backend       Largest error in units of ``u S``    Diagrams
============  ===================================  ==========
ordered       2.13                                 330
einsum        1.80                                 330
factorized    0.56                                 8
============  ===================================  ==========

``factorized`` prepared 9 of the 330 diagrams; the other 321 are sums whose
monomials have opposite signs at some assignment. The factorized reference
multiplies out every monomial and was computed for the 8 of them with
``A T`` at most 20000.

When a backend runs
~~~~~~~~~~~~~~~~~~~

A backend returns a value for a diagram call only when

* every input array it reads is finite, and so is every entry the ordered
  fold multiplies that the backend leaves out: entries at components a
  diagonal C table (``diag_C``) rules out, which the ordered fold multiplies
  by that table's zero, and R entries the ordered fold multiplies before it
  reaches a later R factor whose support excludes the assignment,
* no product or partial sum of those entries can overflow, in any
  evaluation order,
* no product of nonzero inputs can fall below ``2**-1021``, in any
  evaluation order (summed over the factors, the binary exponent of the
  smallest nonzero magnitude, counted as 0 for a factor whose magnitudes
  are at least 1, is at least -1021); for ``factorized`` the rational
  prefactor and every rational of a monomial count as factors of their own
  in this rule and the previous one, because the ordered preparation
  multiplies them one at a time, so rationals whose product is moderate do
  not hide a partial product that overflows or underflows,
* for ``factorized``, the monomials have no opposite signs at any
  assignment (checked once, when the integrand is prepared),
* every coefficient passes the phase check of the ordered fold,
* NumPy's underflow policy is ``"ignore"`` (the default; see
  ``numpy.seterr``),
* the planned workspace fits ``contraction_max_bytes``, and
* its estimated cost is below the ordered fold's.

Otherwise that call runs on the ordered fold. A backend therefore never
returns inf or NaN where the ordered fold returns a finite value, and every
error the ordered fold raises (an imaginary diagram value from a mis-specified
action, an index-free C that is not a multiple of the identity) is raised with
the same type and message. An inf or NaN among the entries the ordered fold
multiplies by a diagonal C table's zero, or a product there that overflows,
gives NaN, and an overflow or invalid operation among the R entries it
multiplies before an excluding support is a floating-point event (an error
under ``numpy.seterr(over="raise")``); in both cases the backend declines
and the call returns that NaN or raises that error. A backend never returns
``-0.0``. A non-finite value that the ordered fold never reads, for example
one multiplied only by zero coefficients, makes the backend decline.

Dynamic (callable) couplings, propagator batches that are not float64,
component labels that are numeric literals or unresolved, and subclasses of
the package classes always use the ordered fold. ``factorized`` also requires
a coupling sum of products of static tensors, rationals and ``i`` whose
summation indices all appear in propagators, with every monomial real after
the observable's phase. A ``factorized`` call declines when a summation
index is carried by none of the arrays it contracts for some monomial: no
coupling tensor of that monomial carries it, and the only propagator that
does is an R without a partner index (read as its trace) or an R folded
into the scalar R product. The ordered fold adds that monomial's product
once for every value of the index. Replacing package methods at run time is
unsupported and is not detected.

Cost rule
~~~~~~~~~

The cost estimates are computed from array shapes, R supports and the zero
pattern of the coefficient. ``einsum`` reads the pattern from the coefficient
array; ``factorized`` counts the nonzero coefficients once, when it prepares
the integrand, and keeps the pattern itself for coefficients of at most 65536
entries. Where the per-assignment sign check is skipped, the count is the
largest count of a single monomial, which can only underestimate the
ordered fold's work. The estimates involve no timing, so the choice of
backend, and hence the result, does not depend on machine load. The ordered
estimate counts the Python work per assignment and one vector operation per
factor of each contributing assignment. The backend estimate counts fixed costs, passes
over the propagator batches and the operations of the planned pairwise
contraction. The constants were fitted to timings of 275 diagram calls on
Python 3.10 with NumPy 2.2 and Python 3.14 with NumPy 2.4, and shifted so
that the ordered fold is underestimated and the backend overestimated.
Small diagrams, for which the ordered fold takes tens of microseconds, keep
the ordered fold; most of them are declined from their array counts and
sample count alone, before any array is inspected. So does a ``factorized``
sum with many monomials over a small coefficient, where contracting every
monomial costs more than the ordered fold.

The contraction path is chosen greedily: at each step two operands that
share an index are contracted, the pair with the fewest loop iterations
first. Two coupling tensors without a common index are never multiplied
into an outer product while another pair is left, which keeps the
intermediates of a ``factorized`` monomial at the size of one tensor times
its propagator batches.

Measured speed
~~~~~~~~~~~~~~

Timings below are from one Apple M-series machine with one BLAS thread,
Python 3.14 and NumPy 2.4. For one diagram call (the static contraction
alone) on the 330 captured diagrams, with the cost rule in force and the
default 64 MiB budget:

=========  ==========================  =============================
Samples    einsum: admitted, speed     factorized: admitted, speed
=========  ==========================  =============================
1024       204 of 330, 7.8x to 108x    2 of 330, 11x to 79x
4096       300 of 330, 2.0x to 117x    2 of 330, 11x to 63x
65536      308 of 330, 2.5x to 107x    4 of 330, 2.1x to 98x
=========  ==========================  =============================

The speed is the ordered time divided by the backend time, over the admitted
diagrams; no admitted call was slower than the ordered fold. ``factorized``
runs on few of these diagrams because most of their coupling sums have
monomials of opposite sign.

End to end, :meth:`~sft_wick.workflow.Expansion.evaluate` with
``qmc_vectorized`` spends most of its time outside the static contraction,
in coefficient preparation, propagator lookups and sampling, so the gain is
smaller than per call. The columns give the ordered time divided by the
backend time:

===  ======  ======  =======  ==========  =======  ==========
N    Orders  diag_C  Samples  ordered s   einsum   factorized
===  ======  ======  =======  ==========  =======  ==========
2    2, 3    off     4096     0.09        1.01     0.98
2    2, 3    off     65536    1.27        1.01     1.00
2    2, 3    on      4096     0.07        1.02     0.99
2    2, 3    on      65536    1.01        1.00     1.01
3    2, 3    off     4096     0.21        1.12     1.02
3    2, 3    off     65536    2.14        1.12     1.02
3    2, 3    on      4096     0.08        1.01     1.01
3    2, 3    on      65536    1.17        1.01     1.01
5    2, 3    off     4096     2.02        1.17     1.06
5    2, 3    off     65536    9.95        1.56     1.10
5    2, 3    on      4096     0.10        0.98     1.01
5    2, 3    on      65536    1.50        1.10     1.04
2    4       on      4096     1.37        1.02     0.95
3    4       on      4096     2.58        1.31     0.98
===  ======  ======  =======  ==========  =======  ==========

Times are the best of two runs; differences of 2 percent are within the
run-to-run spread. When a call of either backend declines, the evaluation
does what the ordered fold does after the declined attempt. Most small
calls decline from their array counts alone. For calls of 30 to 340
microseconds a declined ``einsum`` call still took 7 to 14 percent longer
(median over the diagrams of a run) than the ordered fold, and
``gauss_legendre`` runs made of such calls took 5 to 9 percent longer.
``factorized`` attempts to prepare every diagram; a sum it declines at
preparation costs the attempt, 2 to 25 percent of the ordered coefficient
preparation, which accounts for the ratios below 1 in the order-4 rows.

Workspace budget
~~~~~~~~~~~~~~~~

``contraction_max_bytes`` must be a positive integer and defaults to 64 MiB.
It bounds the arrays a backend allocates for one diagram call: copies of the
propagator batches with the sample axis last, every intermediate of the
planned contraction, the rotated coefficient and its zero mask for
``einsum``, and the output. A call whose plan exceeds it runs on the ordered
fold. It is not a limit on total memory or process RSS.

Comparing complete results
~~~~~~~~~~~~~~~~~~~~~~~~~~

A backend can be compared with the ordered fold on a complete workflow::

    dense_backend:
      contraction_backend: einsum
    dense_backend_with_cache:
      contraction_backend: einsum
      propagator_cache_bytes: 16777216

The second variant also requires ``fixed_model`` in the original workflow.
Declare a tolerance before running, for example::

    sft-wick compare-resources dense-config.yaml \
      --variants dense-variants.yaml --json dense-resource-report.json \
      --rtol 1e-12 --atol 1e-14

The bound above holds per diagram and per sample; a tolerance for an
observable whose diagrams cancel has to account for ``S``. The full resource
comparison checks each diagram's value and error as well as the totals and
axis metadata.

The lazy coefficient of ``factorized``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preparation snapshots the native float64 or complex128 coupling arrays of
the monomials and keeps the symbolic product and term order. A sum whose
monomials have opposite signs at some assignment is not prepared this way;
the ordered preparation builds its coefficient. The returned
object preserves the public ``DiagramIntegrand`` interface. Reading
``ig.coupling_array`` materializes the genuine coefficient ndarray, with the
original dtype and the values the ordered preparation computes, then
disables fusion so later array mutations are observed. Factor snapshots own
their data; later mutations of the input tensors do not alter the prepared
integrand. Pickling a still-lazy integrand preserves its snapshots without
materializing the coefficient. Scalar QMC, zero-dimensional evaluation and
``nquad`` materialize the coefficient instead of using fusion, and so does
the first call that declines.

For this option, ``contraction_max_bytes`` also checks the conservative
ndarray payload needed to materialize a known native coefficient. Public
array access or a dense fallback **raises before allocation** if that payload
exceeds the budget. Increase the budget or rebuild the integrand with
``ordered`` to permit the materialization. Switching an already prepared lazy
object to ordered dispatch does not remove its materialization policy.

The low-level builder supports the same choice::

    ig = diagram.build_integrand(
        coupling_values, fixed_indices,
        contraction_backend="factorized", contraction_max_bytes=67108864,
    )

When using the low-level wrapper, also pass
``integrate_moment(ig, ..., contraction_backend="factorized")``. Its default
is ordered dispatch even for a prepared factorized object; that dispatch
materializes the coefficient. ``Expansion.evaluate`` and ``Expansion.sweep``
forward the selected backend through both preparation and integration.

Shared propagator table identities
----------------------------------

``propagators.cache_path`` can reuse tables across models with an identical
supported native linear/noise specification. Native keys include complete
array content, dtype and shape; interactions that do not affect R or C do not
invalidate these tables. The key never uses a truncated display representation
as numerical identity.

Explicit response callbacks, user closed-form callbacks and other unknown or
modified protocols compute normally but bypass this shared disk cache. Their
external state has no automatic version identity. This differs from the
query-local lookup cache: its explicit ``fixed_model`` promise is sufficient
only for that query's bounded lifetime. No automatic global numerical bank is
created, and model mutations during evaluation remain outside both contracts.
