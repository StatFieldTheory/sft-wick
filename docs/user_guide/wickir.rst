WickIR
======

WickIR is a JSON representation of a single
:class:`~sft_wick.perturbation.DiagramTerm`. It preserves the exact rational
coefficient, the coupling expression tree, the propagators and their ordered
endpoints, component sums, integration variables, equal-time aliases, and
absorbed response edges.

.. code-block:: python

   import json
   from sft_wick.ir import from_ir, ir_digest, to_ir

   expansion = system.expand(["phi_a(x)", "phi_b(y)"], orders=2, iso_C=False)
   diagram = expansion.diagrams(2)[0]
   record = to_ir(
       diagram, target=expansion.target, t_min=system.t_min,
       iso_R=system.iso_R, iso_C=False,
   )
   payload = json.dumps(record)
   restored = from_ir(json.loads(payload))
   assert restored == diagram
   digest = ir_digest(record)

``from_ir`` restores the diagram while the record retains the surrounding
conventions and optional call data. For an isolated term, the default time
origin is zero, the default target is ``moment``, and null isotropy flags
denote unspecified provenance. Pass the flags used at expansion time to
document an override. Numerical R and C kernels and coupling values are
bound by the caller.

Format
------

* ``schema`` is ``"wickir/v1"`` for ordinary terms and ``"wickir/v1.1"`` for
  terms with grouped vertices, fixed delays or explicit time deltas.
* ``diagram`` holds the inputs needed to reconstruct the term. The
  ``propagators`` list includes absorbed R edges and their component
  indices. Component labels are strings, with ``"1"`` denoting the first
  literal component and null denoting an index-free propagator leg.
  Each ``r_absorbed_pairs`` item has the same physical-then-response
  endpoint order. Each ``equal_time_aliases`` item maps an internal
  label to its time representative. The ``vertex_types`` array contains
  source templates for multiplicity terms. Each entry is an ordered
  point list paired with
  ``[name, local, equal_time, already_R_contracted, fields, leg_symmetry]``.
  Each ordered field is ``[name, role, n_components]``.
* ``conventions`` describes expansion, numerical, and time conventions.
  ``response_phase`` controls display. The raw numerical diagram includes
  :math:`(-i)^{n_\mathrm{response}}`, with an additional
  :math:`i^{n_\mathrm{external\_response}}` rotation when projecting the
  observable to real values. R is retarded with :math:`\Theta(0)=0`.
  Absorbed R edges join spatial groups and identify times while their
  numerical R factor is one. With ``iso_C=True``, the scalar value is the
  common diagonal entry. Explicit component deltas remain in the coupling
  expression.
* ``spatial_structure`` records direction groups and causal orderings from
  the production analyzer. A separate evaluator should derive these from
  the diagram before checking them against this field.
* ``evaluation`` holds the supplied positions, fixed external times and
  ``t_final``, with null for omitted inputs. The optional refusal record
  contains a stable ``code``. For example, two external points of the
  same R group at different coordinates produce
  ``r_group_position_conflict``. Pass the cache's ``n_components`` to
  check absorbed R indices. With two or more components, an absorbed R
  edge with two different explicit component indices produces
  ``invalid_absorbed_r_indices``. The production evaluator skips this
  check with one component. If the dimension is omitted, the same
  mismatch appears as ``conditional_rejection`` with
  ``min_n_components: 2``. The checks also cover swept and pinned time
  point names and conflicts. Model and backend checks require their
  respective numerical inputs.

Bind MSR-adjusted coupling tensors when evaluating a record. ``bare_to_msr`` records
the convention used to obtain them from a System's bare inputs. For
example, a local drift receives :math:`-i` and a nonlocal cumulant of rank
:math:`m` receives :math:`-i^m/m!`. Callers using
:meth:`~sft_wick.workflow.system.System.build_coupling_values` already have
the adjusted values.

External times default to ``t_final`` and allow per-point overrides.
``integrate_over`` lists swept external labels, with ``"all"`` for every
external point and null for none. The integration interval starts at
``t_min``. ``swept_external_ceiling`` sets the upper bound for swept
external times to ``t_final``. ``integration_ceiling`` gives the global
upper bound for internal times as the latest of ``t_final`` and the pinned
times. Causal R ordering can further restrict either range. Earlier v1
records leave the external ceiling implicit and remain readable.

An expression node's ``type`` key identifies its arguments.

.. list-table::
   :header-rows: 1

   * - Type
     - Arguments
   * - ``rational``
     - ``numerator``, ``denominator``
   * - ``symbol``
     - ``name``, ``indices``, ``spatial_args``, ``local``
   * - ``propagator``
     - ``kind``, ``index_left``, ``index_right``, ``spatial_left``, ``spatial_right``
   * - ``sum``, ``product``
     - ``terms`` or ``factors``
   * - ``sum_over_index``
     - ``index_name``, ``dimension``, ``body``
   * - ``integral_over``
     - ``variable``, ``body``
   * - ``kronecker_delta``, ``dirac_delta``
     - ``index1`` and ``index2``, or ``arg1`` and ``arg2``
   * - ``time_delta`` (v1.1)
     - ``point``, ``anchor``, ``lag``
   * - ``imaginary_unit``
     - None

The digest covers the complete JSON record, including the numerical
context when present. The workflow's on-disk expansion cache has a
separately versioned key based on the action, observable, and expansion
settings. A change to the coupling values or kernel parameters can reuse
the same expansion while numerical evaluation uses the current system.

Typed diagram identities
------------------------

``diagram_ids(diagram)`` gives stable IDs for the typed spatial graphs that
a term represents. Each identity keeps external labels, directed R edges,
unordered C edges, coupling names and ordered leg incidence, equal-time
groups and R absorption. For the same source template information,
renaming internal labels or changing component routing and numerical
coefficients preserves the graph IDs. Use ``ir_digest`` to identify the
complete serialized diagram.

A default symbolic term can collect several typed graphs into its coupling
sum. Use ``routing="multiplicity"`` in both expansion engines when comparing
one row per typed orbit and preserve the original field order and
dimensions. Older or collected terms whose source templates are unavailable
use the typed spatial information in their coupling expression. That
fallback and the full-template route use different identity inputs. The
built-in canonical search gives the same encoding across topology
grouping backends.

SymPy
-----

With the ``sft-wick[cas]`` extra installed, ``to_sympy(diagram)`` emits
the raw component-summed integrand. It includes the rational coefficient
and response phase. Grouped clocks and fixed delays appear as Dirac deltas
of the form :math:`\delta(t_\mathrm{point} - t_\mathrm{anchor} + \mathrm{lag})`,
where the lag is zero for a time-only group. The ``time:`` symbols name
the times of the corresponding ``point:`` labels. Absorbed R factors and
their time aliases are omitted. The caller applies those aliases together
with the causal time integration and the observable's final phase rotation.

``from_sympy`` translates the supported SymPy nodes back into a Wick
expression. Both functions also work on individual expression trees.
This bridge preserves the ordered argument lists of tensor and propagator
calls and uses native SymPy rationals, sums, products and deltas. Component
labels remain symbols with their original spelling, including numeric
labels such as ``'1'`` and ``'+1'``. For numerical binding, pass
``fixed_indices`` to ``to_sympy``. Values in that mapping are zero-based
and take precedence over the one-based literal interpretation of free
numeric labels. An empty mapping applies just the literal defaults.
Component sums keep their own bound indices. The JSON format carries the
full diagram and its integration context.

Grouped vertices and fixed delays
---------------------------------

A term with grouped field slots or a fixed delay uses ``wickir/v1.1``.
The diagram includes ``group_templates`` in the form
``[coupling_name, coordinate_labels, ordered_field_roles, group_ids, mode]``.
For ``spacetime`` each group has one coordinate. For ``time`` the slots keep
separate spatial coordinates and ``equal_time_aliases`` records the common
clocks. Each ``time_lags`` entry ``[delayed, anchor, lag]`` states that the
delayed point is evaluated at the anchor time minus the positive lag.
Typed graph identities include this direction, the lag value and the ordered
group incidence. Ordinary terms retain their v1 wire representation and IDs.
