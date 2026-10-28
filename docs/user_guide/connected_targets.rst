Connected source targets
========================

``System.expand(..., target="connected")`` keeps the incidence-connected
contributions before component routing and diagram canonicalization when
incidence ownership permits; otherwise it filters each exact routing. The
default ``target="moment"`` retains the existing moment expansion::

    expansion = system.expand(
        ("phi_a(x)", "phi_b(y)", "phi_c(z)", "phi_d(w)"),
        orders=[0, 1, 2], target="connected",
    )

The same keyword is available on ``compute_moment`` and
``compute_moment_numerical``. The selected target is recorded on
``PerturbativeResult.target`` and ``Expansion.target``. Connected and moment
expansions use distinct cache keys; existing moment cache keys are unchanged.
In YAML use ``expand.target: connected``; the CLI summary displays the target.
Standalone numerical result tables and exports retain their existing schema
and do not carry this target metadata. Preserve the YAML/configuration or the
``Expansion`` alongside saved totals and label connected results explicitly.

What is connected
-----------------

Every occurrence in the observable list is a separate source insertion. Each
interaction vertex is one incidence node, including all legs of a nonlocal
input-cumulant vertex. Propagator pairs join these nodes. Retained graphs have
one connected component containing every insertion and every vertex; vacuum
components are excluded.

Neither shared component indices nor coincident coordinates merge source
insertions. Use distinct external labels at interacting orders and assign them
the same physical position through ``positions`` if needed. For example, four
physical fields at one point still request a fourth joint cumulant; they do
not request the variance of a composite ``phi**2`` observable. Composite
observable grouping is not part of this API. Empty connected observables are
refused.

The implementation first checks an exact necessary edge-count condition. It
then filters spatial representatives when each coordinate label has one
incidence owner. A coordinate label shared by an external occurrence and an
internal leg uses exact per-routing occurrence filtering. Equal-time
and absorbed-response aliases retain the original vertex incidence; filtering
takes place before those lowering operations. Signed coefficients,
multiplicities, ordered tensor slots and MSR phases are preserved.

:func:`~sft_wick.perturbation.compute_moment` and
:func:`~sft_wick.perturbation.compute_moment_numerical` name the vertex points
and indices around the observable's labels (see
:class:`~sft_wick.indices.IndexContext`), so an external label such as ``y_0``
is never also an internal one and the shared-label filtering is reached only
by callers that build their own vertex instances.

Normalization and interpretation
--------------------------------

Under a centered Gaussian free functional and a factorizing action/integration
domain, the linked-cluster identity identifies the connected sum with source
derivatives of ``log Z(J) - log Z(0)``, coefficient by coefficient in the formal
perturbation series. The default causal Itô MSR ``System`` has ``Z(0)=1``:
vacuum contractions require a forbidden response cycle. With physical field
insertions, its connected target therefore supplies formal joint cumulants.
External response insertions supply formal MSR source correlators.

The raw ``Action`` API permits broader vertices. Its legacy moment target does
not divide by a nontrivial vacuum partition function. To compare a connected
target with independent moment-partition cumulants in such a theory, first
normalize the moments by ``Z(0)`` and perform coefficient convolutions at the
same total vertex order. ``ito=False`` is a symbolic switch with the existing
numerical restrictions, rather than a normalized Stratonovich prescription.
Finite input cumulants and a formal expansion establish neither a realizable
probability law nor convergence of the series. An absorbed-response callable
must implement its declared R-contracted kernel.

Structural zeros in scans
-------------------------

A free Gaussian fourth connected source correlator is exactly zero. Connected
``evaluate`` includes zero selected orders in ``Result.by_order`` while
``Result.per_diagram`` stays empty for those orders. Connected scans preserve
each grid/order zero as a row with ``value=error=0`` and null ``diagram_idx``,
``vertex_type`` and ``n_cross_C``. These rows are coordinate/order placeholders,
not diagrams. ``totals`` and CSV exports retain the zeros;
``by_vertex_type_totals`` does not assign them a vertex family. Plot with
``hue="order"`` to show zero order coefficients. Moment scans retain their
existing row behavior.

Scalar numerical coupling repair
--------------------------------

This implementation also repairs a preexisting scalar numerical-generator
failure: scalar vertex symbols used ``(None, ...)`` as component indices,
although scalar ``VertexInstance`` symbols have no component indices. Numeric
generation now omits those ``None`` entries. This changes a failing scalar
evaluation into the supported scalar value for both targets; it is separate
from connected filtering. Native two- and three-component ordered terms were
checked against the frozen preimplementation source on bounded probes.
