Changing the integration variable
=================================

Every integrator places its nodes linearly between causal bounds in the
time ``t`` of the system, and the C and R tables are built on uniform grids
in ``t``.  When the kernels are smooth in another variable ``u`` but steep in
``t``, pose the system in ``u``.  No option of the package is needed: the
fields are rescaled and every kernel picks up a fixed power of the Jacobian.
This page gives the factors and an example.

When it matters
---------------

A weak-lensing calculation along the line of sight is the case that
motivated this page.  The time of the Sachs equations is the affine
parameter ``λ``, with ``dλ = a² dχ`` (``a`` the scale factor, ``χ`` the
comoving distance).  Towards high redshift ``λ`` saturates:
``dχ/dλ = (1 + z)²`` is 4 at ``z = 1`` and ``1.2·10⁶`` at ``z = 1100``.  A
kernel that is smooth in ``χ`` is then steep in ``λ`` near the source, and
Gauss-Legendre nodes uniform in ``λ`` converge slowly.

The Einstein-de Sitter toy of ``tests/test_time_variable_recipe.py``
(TV4: one component, the Sachs response, a smooth C, a cubic vertex,
``<Φ(x) Φ(y)>`` with ``Φ = ∫ φ dλ``, source at ``z = 1100``), relative to
the ``χ`` formulation at 32 nodes per time variable:

.. list-table::
   :header-rows: 1

   * - nodes
     - posed in ``λ``
     - posed in ``χ``
   * - 8
     - -2.5e-1
     - -2.4e-5
   * - 12
     - -2.0e-1
     - +4.5e-10
   * - 16
     - -1.7e-1
     - +3.6e-15
   * - 32
     - -1.2e-1
     - 0

The cost per evaluation is the same in both variables at equal node count;
the order-2 channel has four time variables, so its cost grows as ``n⁴``.

The rescaling
-------------

Let ``t = f(u)`` be increasing, with ``J(u) = dt/du > 0``.  Rescale the
physical and response fields as

.. math::

   \tilde\varphi(u) = J(u)\,\varphi(f(u)), \qquad
   \tilde\psi(u) = \psi(f(u)) / J(u).

The MSR action keeps its form in ``u`` with the ingredients below.  With
this split a cubic local vertex with one ψ leg stays the same constant
tensor, so its diagrams stay on the static coupling path.

.. list-table::
   :header-rows: 1

   * - ingredient
     - in ``t``
     - in ``u``
   * - response
     - ``R(t, t')``
     - ``J(u) R(f(u), f(u')) / J(u')``
   * - ``DiagonalA`` rate
     - ``γ(t)``
     - ``J(u) γ(f(u)) − d ln J / du``
   * - C (closed form)
     - ``C(t₁, t₂)``
     - ``J(u₁) J(u₂) C``
   * - coloured noise ``κ²``
     - ``κ²(t₁, t₂)``
     - ``J(u₁)² J(u₂)² κ²``
   * - white noise ``σ²``
     - ``σ²(t)``
     - ``J(u)³ σ²``
   * - local vertex, one ψ leg and ``p`` φ legs
     - ``F``
     - ``J(u)^(2 − p) F`` (cubic: unchanged; quartic: ``F / J``)
   * - raw non-local ``κ^(m)`` at leg times ``u_i``
     - ``κ(t₁, …, t_m)``
     - ``Π_i J(u_i)² κ``
   * - ``equal_time`` ``κ^(m)``
     - ``κ(t)``
     - ``J(u)^(m+1) κ``
   * - ``already_R_contracted`` ``K_R`` at partner times ``u'_i``
     - ``K_R(t'₁, …, t'_m)``
     - ``Π_i J(u'_i) K_R``
   * - integrated external ``∫ φ dt``
     - ``φ``
     - ``φ~`` with unit weight in ``u``
   * - external ``φ`` at a fixed time
     - ``φ(t_x)``
     - ``φ~(u_x) / J(u_x)``
   * - ``t_min``, ``t_final``, ``external_times``, ``t_max``
     - ``t``
     - ``f⁻¹(t)``

The ``DiagonalA`` row follows from the response row: ``J(u)/J(u')`` is
``exp(∫ d ln J)``, which shifts the rate.  A matrix drift becomes
``J A + (d ln J / du) I``.  Rows that change a static tensor into a function
of time (the quartic vertex, a raw or ``equal_time`` ``κ^(m)``) make it a
callable coupling; pass it with ``coupling_vectorized=True``.

Every row is checked by ``tests/test_time_variable_recipe.py``: one system
in ``t`` and the same system in ``u`` with the curved map
``t = u + 0.4 u²``, two components, couplings and cumulants with no index
symmetry, ``t_min = 0.3``.  The package's own quadrature of C from ``κ²``
and ``σ²`` agrees to 5e-14 (TV1).  Diagrams with every vertex type above,
fixed and integrated externals, orders 0 to 2, agree to 4.6e-12 (TV2).
Each rule with one factor of ``J`` missing or wrong misses by 16 per cent to
a factor 4 (TV3).

What changes for the kernels
----------------------------

* A separable kernel such as ``ExponentialTemporal`` is a function of
  ``t₁ − t₂``.  After the rescaling it is not a function of ``u₁ − u₂``, so
  the built-in closed-form C (``c_closed_form='auto'``) and the separable
  fast path do not apply.  Supply a closed-form C in ``u`` or a
  ``GeneralKappa2``.
* A kink of C on its time diagonal stays on the diagonal, because ``f`` is
  increasing.  A kernel declared with ``has_diagonal_kink`` or
  ``has_coincident_time_kinks`` keeps its declaration.
* A missing factor changes values without an error.  Compare the system in
  ``u`` with the system in ``t`` at a node count where the ``t`` version has
  converged, as the test above does.

Example
-------

The ``χ`` system of TV4, written from its ``λ`` system.  With
``J = dλ/dχ = a² = (1 − χ)⁴`` and the Sachs response
``R(λ, λ') = (D(λ')/D(λ))²``, ``D = a χ``:

.. code-block:: python

   import numpy as np
   import sft_wick as sw

   def jac(chi):                 # J = d lambda / d chi
       return (1.0 - chi) ** 4

   def dbar(chi):
       return (1.0 - chi) ** 2 * chi

   def R_chi(c, c_prime):        # J(c) R / J(c')
       if c < c_prime:
           return 0.0
       return jac(c) * (dbar(c_prime) / dbar(c)) ** 2 / jac(c_prime)

   def C_chi(n1, c1, n2, c2):    # J(c1) J(c2) C
       return np.array([[jac(c1) * jac(c2) * C_of_chi(n1, c1, n2, c2)]])

   system = sw.System(
       field=sw.FieldSpec("phi", n_components=1),
       linear=sw.ExplicitR(R_time=R_chi),
       vertices=[sw.LocalVertex("F", coupling=np.full((1, 1, 1), -1.0))],
       noise=sw.GaussianNoise(kappa2=sw.GeneralKappa2(
           fn=lambda n1, t1, n2, t2: np.zeros((1, 1)))))
   props = system.propagators(t_max=chi_s, c_closed_form=C_chi,
                              c_closed_form_only=True, diag_C=False)

``C_of_chi`` is C as a function of ``χ`` and ``chi_s`` the comoving distance
of the source.  The cubic vertex keeps ``F = −1``.  The observable
``Φ = ∫ φ dλ`` is ``∫ φ~ dχ``, so ``integrate_over="all"`` with
``t_final=chi_s`` evaluates it unchanged.
