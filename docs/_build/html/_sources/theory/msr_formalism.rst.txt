The MSR Formalism
=================

The Martin--Siggia--Rose (MSR) formalism (also known as the
Janssen--de Dominicis formalism) provides a field-theoretic framework
for studying stochastic differential equations (SDEs) using
path-integral methods.

From Langevin Equations to Path Integrals
-----------------------------------------

Consider a stochastic process described by a Langevin equation:

.. math::

   \frac{\partial \phi_i(x,t)}{\partial t}
   = -\frac{\delta \mathcal{H}[\phi]}{\delta \phi_i(x,t)}
   + \eta_i(x,t)

where :math:`\phi_i` is the physical field of interest,
:math:`\mathcal{H}[\phi]` is an energy functional, and :math:`\eta_i`
is Gaussian white noise with correlator

.. math::

   \langle \eta_i(x,t)\,\eta_j(x',t') \rangle
   = 2 D_{ij}\,\delta(x-x')\,\delta(t-t').

The MSR procedure recasts averages over noise realisations as a
field-theoretic path integral by introducing an auxiliary **response
field** :math:`\psi_i` (sometimes written :math:`\tilde\phi_i` or
:math:`\hat\phi_i`):

.. math::

   \langle \mathcal{O}[\phi] \rangle
   = \int \mathcal{D}\phi\,\mathcal{D}\psi\;
     \mathcal{O}[\phi]\;
     e^{-S[\phi,\psi]}

The two field types in **sft-wick** correspond to:

- :math:`\phi` --- the **physical field** (``FieldType.PHYSICAL``)
- :math:`\psi` --- the **response field** (``FieldType.RESPONSE``)


Free and Interaction Actions
----------------------------

The MSR action splits into a free (Gaussian) part and an interaction part:

.. math::

   S[\phi,\psi] = S_0[\phi,\psi] + S_{\mathrm{int}}[\phi,\psi]

The **free action** :math:`S_0` is quadratic in the fields and determines
the two-point functions (propagators).  The **interaction action**
:math:`S_{\mathrm{int}}` contains all nonlinear terms --- these are the
terms represented by :class:`~sft_wick.vertices.Vertex` objects in
**sft-wick**.


Propagators
-----------

The free two-point functions derived from :math:`S_0` are:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Contraction
     - Propagator
     - Physical meaning
   * - :math:`\langle \phi_i(x)\,\phi_j(x') \rangle_{S_0}`
     - :math:`C_{ij}(x,x')`
     - Correlation (equal-time or two-time)
   * - :math:`\langle \phi_i(x)\,\psi_j(x') \rangle_{S_0}`
     - :math:`R_{ij}(x,x')`
     - Response (retarded Green's function)
   * - :math:`\langle \psi_i(x)\,\psi_j(x') \rangle_{S_0}`
     - :math:`0`
     - Vanishes by construction

The vanishing of the :math:`\psi`--:math:`\psi` contraction is a
fundamental consequence of the MSR construction (the noise is integrated
out exactly) and is the key constraint exploited by **sft-wick** for
efficient enumeration of Wick contractions.

.. note::

   **Convention in sft-wick:** for the response propagator *R*, the
   physical field is always placed on the **left**:

   .. math::

      R_{ij}(x,x') = \langle \phi_i(x)\,\psi_j(x') \rangle_{S_0}


Itô Convention and Causality
----------------------------

The response propagator is retarded:
:math:`R(t,t') \propto \Theta(t - t')`.  The **Itô discretisation**
convention sets :math:`\Theta(0) = 0`, which implies:

.. math::

   R(x,x) = 0

This eliminates equal-point response contractions (self-response
tadpoles).  More generally, **any closed loop of response propagators
vanishes** by causality:

.. math::

   R(a_1,a_2)\,R(a_2,a_3)\,\cdots\,R(a_n,a_1) = 0

because the retarded nature of :math:`R` would require
:math:`t_1 > t_2 > \cdots > t_n > t_1`, which is impossible.

In **sft-wick**, both rules are applied by default when ``ito=True``.


Response Phase Convention
-------------------------

In some formulations of the MSR action, the contraction of a physical
field with a response field picks up a phase:

.. math::

   \langle \phi_i(a)\,\psi_j(b) \rangle_{S_0}
   = -\mathrm{i}\,R_{ij}(a,b)

When ``response_phase=True`` (the default in
:func:`~sft_wick.perturbation.compute_moment`), each term in the final
result is multiplied by :math:`(-\mathrm{i})^n`, where *n* is the
number of response propagators in that term.


The Partition Function :math:`Z=1`
-----------------------------------

A remarkable property of the MSR formalism is that the partition function
equals unity:

.. math::

   Z = \int \mathcal{D}\phi\,\mathcal{D}\psi\;
       e^{-S[\phi,\psi]} = 1.

This means there is no denominator in the perturbative expansion of
moments, greatly simplifying the diagrammatic analysis (no vacuum
diagram subtraction is needed).


Perturbative Expansion
----------------------

Since :math:`Z=1`, the expectation value of an observable
:math:`\mathcal{O}[\phi]` can be expanded directly:

.. math::

   \langle \mathcal{O} \rangle_S
   = \sum_{n=0}^{N} \frac{(-1)^n}{n!}\,
     \langle \mathcal{O}\, S_{\mathrm{int}}^{\,n} \rangle_{S_0}

Each term in the sum involves a Gaussian expectation
:math:`\langle\cdots\rangle_{S_0}`, which is evaluated via **Wick's
theorem** (see :doc:`wick_theorem`).  The function
:func:`~sft_wick.perturbation.compute_moment` implements this expansion
up to any finite order *N*.
