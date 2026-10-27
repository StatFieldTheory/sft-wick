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

Opt-in dense tensor contraction
-------------------------------

The default ``contraction_backend: ordered`` retains the existing component
assignment fold. ``contraction_backend: einsum`` requests a dense contraction
for admitted static integrands::

    sweep:
      contraction_backend: einsum
      contraction_max_bytes: 67108864

``contraction_max_bytes`` must be a positive integer and defaults to 64 MiB.
It controls the admitted contraction workspace; it is not a total memory or
process RSS limit. The ``einsum`` backend consumes the existing prepared
coefficient tensor, so it retains that tensor's preparation cost.

Admission is deliberately narrow: native float64 real factors, a validated
static coefficient phase and fully resolved ordinary component indices with
full-matrix propagator batches. Dynamic couplings, sparse or nonfinite inputs,
diagonal-C routing, unresolved/literal indices, custom numerical protocols,
extreme factor scales, unsupported dtypes, strict floating-error settings and
over-budget paths retain the ordered implementation. A requested backend can
therefore legitimately fall back for some or all diagrams.

Changing contraction order changes floating-point rounding, even when every
input is finite. The backend does not certify an error bound for arbitrary
models or cancellations. Each user must choose and validate a numerical budget
for the observable; a small total difference alone is insufficient because
different diagram errors can cancel. The full resource comparison checks each
ordered diagram value and error as well as the totals and axis metadata.

For a dense workflow, the variants file can contain::

    dense_backend:
      contraction_backend: einsum
    dense_backend_with_cache:
      contraction_backend: einsum
      propagator_cache_bytes: 16777216

The second variant also requires ``fixed_model`` in the original workflow.
Declare a budget before running, for example::

    sft-wick compare-resources dense-config.yaml \
      --variants dense-variants.yaml --json dense-resource-report.json \
      --rtol 1e-12 --atol 1e-14

These example tolerances are syntax, not a universal numerical guarantee.
The standard workflow has no tolerance acceptance flag: the explicit
comparison is the validation tool, while backend selection is the execution
request. Source settings, diagrams and integration resolution remain fixed.

Original coupling-factor fusion
-------------------------------

``contraction_backend: factorized`` additionally requests preparation from the
original static coupling factors::

    sweep:
      contraction_backend: factorized
      contraction_max_bytes: 67108864

For admitted native monomials or bounded native sums of monomials, preparation
snapshots its native float64 or complex128 arrays and retains the original
symbolic product and term order. Terms reuse the private tensor snapshots.
Vectorized QMC and GL can contract each term's factors directly with full-matrix
propagator batches. This avoids preparing the dense coefficient over all
propagator component indices. It uses the same integration nodes and propagator
routing; changed accumulation order still requires complete-result tolerance
validation. It does not establish a numerical error bound for an arbitrary
model.

This initial implementation admits at most 32 monomial terms in a top-level
native sum. Admission requires ordinary resolved indices, no coupling-only
index sums, a finite pure real or imaginary phase for each nonzero coupling
factor, bounded scales and unchanged native numerical protocols. Every term
must independently project to a finite real contribution under the diagram's
phase. Imaginary contributions cannot be admitted by cancellation between
terms. Propagator batches must also pass the native dtype, finite-value and
scale checks; finite inputs alone do not establish a rounding bound.

For a multi-term sum, every monomial's selected real or imaginary array
components must additionally have uniform nonzero signs. After including each
term's scalar phase, the monomial contributions must share the same sign.
Mixed-sign slices or opposing term signs retain the original ordered path;
fusion does not admit cross-monomial real cancellation. A single monomial may
still use mixed-sign factors under the existing phase and range checks. These
sign checks constrain the coefficient sum, not all later kernel contractions
or numerical integration, and do not certify a complete-result error budget.

Unsupported or larger sums, deltas, dynamic couplings, custom accessors,
unsupported dtypes and kernel routing keep the original coefficient path when
it fits the explicit budget. Runtime refusal of fusion also restores the
original coefficient path. Sparse factors and per-component phase errors are
not replaced by cancellation across components or terms.

An existing compiled coefficient plan also retains its original path;
requesting ``factorized`` on previously evaluated diagram terms may fall back.
Fresh workflow preparation is the supported cost comparison.

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

The returned object preserves the public ``DiagramIntegrand`` interface.
Reading ``ig.coupling_array`` materializes the genuine original ndarray with
its original dtype and ordered component values, then disables fusion so
subsequent array mutations are observed. Dense restoration follows the original
symbolic products and the existing builtin sum evaluation of coefficient terms;
it does not substitute the fused batch accumulation. Factor snapshots own their data;
later mutations of the input tensors do not alter the prepared integrand.
Pickling a still-lazy integrand preserves its snapshots without materializing
the dense coefficient. Scalar QMC, zero-dimensional evaluation and ``nquad``
materialize the coefficient instead of using batch fusion.

For this option, ``contraction_max_bytes`` also checks the conservative ndarray
payload needed for known native coefficient materialization. Public array
access or a dense fallback **raises before allocation** if that payload exceeds
the budget. Increase the budget or rebuild the integrand with ``ordered`` to permit
the original materialization. Switching an already prepared lazy object to
ordered dispatch does not remove its materialization policy. Dynamic and custom
protocols with unknown shapes are not bounded by this admission check. Python
metadata, other integrands, model data, process RSS and private BLAS workspaces
remain outside this cap.

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
