``sft_wick._certificate``: Count Certificates
==============================================

The records and the error of ``certify=True`` on
:func:`~sft_wick.perturbation.compute_moment`,
:func:`~sft_wick.perturbation.compute_moment_numerical` and
:meth:`~sft_wick.workflow.system.System.expand`.  ``CountCertificate`` and
``CertificateError`` are exported by ``sft_wick``; the other records are
reached through the attributes of a certificate.

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

``sft_wick._certify``: Pairing Counter
--------------------------------------

.. automodule:: sft_wick._certify
   :members: count_valid_pairings, UnsupportedLayout
   :show-inheritance:
   :member-order: bysource
