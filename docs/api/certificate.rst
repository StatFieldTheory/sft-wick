``sft_wick._certificate``: Count Certificates
==============================================

The records and the error of ``certify=True`` on
:func:`~sft_wick.perturbation.compute_moment`,
:func:`~sft_wick.perturbation.compute_moment_numerical` and
:meth:`~sft_wick.workflow.system.System.expand`.  ``CountCertificate`` and
``CertificateError`` are exported by ``sft_wick``; the other records are
reached through the attributes of a certificate.

Value checks
------------

For each sequence of at most 14 operator occurrences, the certified
expansions compare the emitted diagram terms with a separate enumeration of
labelled pairings. The largest layout in this budget has 135,135 candidate
pairings before the causality and connectivity filters. Both sums use the same
deterministic synthetic couplings and propagator matrices. The matrices respect
the requested diagonal or isotropic constraints. For dense matrices only C
reverses by transposition. The numerical engine also checks a second fixture
whose matrices distinguish different points of the same vertex type. The
comparison tolerance is 1e-12 relative to the larger magnitude of the two sums.
The synthetic component basis keeps the first three components of each field
and any literal external component labels. A tensor enlarged past the numeric
budget by those extra components gives a count-only row.

Read ``value_checks`` on each sequence row for the two complex values
(as real/imaginary pairs) and their relative discrepancy. ``value_skip``
records the reason when a certified expansion only checks counts, for example
above the operator budget or above order zero with ``collect_topology=False``.
An index reduction joining unequal component dimensions or a propagator
joining scalar and vector fields also gives a count-only row.
These probes check the contraction and its weights at synthetic inputs.
The caller's coupling functions, propagator tables, time integration and
quadrature require separate validation. Certified ``System.expand`` recomputes the result,
including these checks, when an expansion cache exists.

.. currentmodule:: sft_wick

.. autoclass:: CountCertificate
   :members:
   :show-inheritance:

.. autoexception:: CertificateError
   :show-inheritance:

.. automodule:: sft_wick._certificate
   :members: CERTIFICATE_SCHEMA, OrderCount, SequenceCount, CertifiedTerms,
             counter_inputs
   :show-inheritance:
   :member-order: bysource

.. autoclass:: sft_wick._value_certificate.ValueComparison
   :members:

``sft_wick._certify``: Pairing Counter
--------------------------------------

.. automodule:: sft_wick._certify
   :members: count_valid_pairings, UnsupportedLayout
   :show-inheritance:
   :member-order: bysource
