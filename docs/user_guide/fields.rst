Fields and Field Operators
==========================

Every calculation in **sft-wick** begins with **field declarations** and
the **field operators** derived from them.


Field Types
-----------

The :class:`~sft_wick.fields.FieldType` enum defines the two MSR field
species:

- ``FieldType.PHYSICAL``: the physical field :math:`\phi`
- ``FieldType.RESPONSE``: the response (auxiliary) field :math:`\psi`

You can pass either the enum value or a string:

.. code-block:: python

   from sft_wick import Field, FieldType

   phi = Field('phi', FieldType.PHYSICAL)
   psi = Field('psi', 'response')          # string shorthand


Creating Fields
---------------

A :class:`~sft_wick.fields.Field` is a frozen dataclass that declares a
field species:

.. code-block:: python

   # Scalar field (1 component)
   phi = Field('phi', 'physical')

   # Multi-component field (N components)
   phi = Field('phi', 'physical', n_components=3)

Properties:

- :attr:`~sft_wick.fields.Field.is_scalar`: ``True`` when
  ``n_components == 1``
- :attr:`~sft_wick.fields.Field.is_physical` /
  :attr:`~sft_wick.fields.Field.is_response`: type checks


Creating Field Operators
------------------------

A :class:`~sft_wick.fields.FieldOperator` is a concrete instance of a
field with a bound component index and spatial argument.  Create one by
*calling* a ``Field``:

.. code-block:: python

   # Scalar: one argument (spatial)
   op = phi('x')             # phi(x)

   # Multi-component: two arguments (component, spatial)
   op = phi('a', 'x')        # phi_a(x)

Each call produces an operator with a **globally unique integer UID**.
The contraction engine uses it to distinguish two copies of the same
field in the same product (e.g. :math:`\phi(x)\,\phi(x)`).

A letter component label such as ``'a'`` is a free label: its component
is given at evaluation time through ``fixed_indices`` (values count from
0).  A component label that parses as an integer is a literal component,
and literals count from 1: ``phi('1', 'x')`` is component 0 and
``phi('2', 'x')`` component 1.  Coupling tensors, propagators and the
Kronecker deltas inserted by ``diag_C``, ``diag_R``, ``iso_C`` and
``iso_R`` all read a literal this way.
:func:`~sft_wick.perturbation.compute_moment` and
:func:`~sft_wick.perturbation.compute_moment_numerical` raise
``ValueError`` for a literal outside ``1..n_components``, ``'0'``
included.  An L1 observable string takes a component label that starts
with a letter (``'phi_1(x)'`` does not parse); there the component tuple
of ``Expansion.evaluate`` chooses the components.


UID System and Test Reproducibility
------------------------------------

UIDs are assigned from a global counter.  For reproducible tests (so
that operator UIDs start from zero each time), call
:func:`~sft_wick.fields.reset_uid_counter` before the test:

.. code-block:: python

   from sft_wick import reset_uid_counter

   reset_uid_counter()
   op1 = phi('x')  # uid == 0
   op2 = phi('y')  # uid == 1


Field Operator Properties
-------------------------

A ``FieldOperator`` exposes the following attributes:

- ``field``: the parent :class:`~sft_wick.fields.Field` declaration
- ``component_index``: component label (``None`` for scalars)
- ``spatial_arg``: spatial argument string
- ``uid``: unique integer ID
- ``field_type`` / ``is_physical`` / ``is_response``: delegated from
  the parent field
