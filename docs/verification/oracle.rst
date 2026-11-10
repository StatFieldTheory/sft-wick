The oracle and the fuzzer
=========================

``validation/`` compares the package with an exact reference on models drawn
at random.  It ships in the sdist and not in the wheel.  The reference, the
oracle, is the moment hierarchy of a finite Markov embedding of the model;
it shares no code with the package.  The tests of ``validation/`` are listed
in :doc:`catalog`; the fuzz runs below are records of runs, not tests.

The oracle
----------

``validation/modelspec.py`` describes a model and an observable as plain
data.  A model has N components, a constant rate per component, a source,
quadratic and cubic drift, Ornstein-Uhlenbeck blocks loaded onto the
components with a spatial kernel each (constant, exponential, Gaussian,
damped cosine, Legendre, or one that is not homogeneous), white noise,
affine multiplicative noise read as Itô or as Stratonovich, compound-Poisson
noise with white or exponential pulses, static cumulants of order 1 and
higher, and white cumulants of order 2 and higher.  An observable is a
product of legs at distinct labels, each at its own external time or at the
final one, some integrated over the window, with the moment or the joint
cumulant as its target.

``validation/hierarchy.py`` compiles the description into the moment
hierarchy of its embedding, on ``examples/reference/ito_moments.py`` and
``hormander_moments.py``.  R is local in space and every vertex acts at one
point, so the fields at the observation points and the states of the noise
form a finite-dimensional Markov process.  Each vertex type is a channel,
and every term of the generator carries a tag that counts its channels, so
the hierarchy returns one value per channel content: the sum of the
package's diagrams with those vertices.  The oracle takes at most two
distinct external times and no response legs.  From model descriptions it
reproduces the 1,772 hierarchy values committed for demos 4, 5 and 7 to
within 4.2e-15 relative.

``tests/test_validation_imports.py`` walks the imports of every oracle
module and of the two engines.  Only the lowerings,
``validation/lower.py`` and ``validation/lower_v042.py``, may import
sft_wick, and ``validation/worker.py`` runs them in a process of their own.
``validation/lowering_parts.py`` holds the closed form of C that the
lowerings give the package; it imports no sft_wick, and no oracle module may
import it.  A subprocess then runs the oracle on a case of every demo and on
a model with every noise block, and checks that no sft_wick module was
loaded.

The lowering and the comparison
-------------------------------

``validation/lower.py`` lowers a description onto the L1 API (``System``,
``Expansion``, ``Propagators``) and sums the package's diagrams by the
oracle's channels.  It does not lower response legs, or white Poisson pulses
beside multiplicative noise.  ``validation/lower_v042.py`` lowers onto the
0.4.2 release and refuses what 0.4.2 lacks, multiplicative noise and the
joint-cumulant target.  ``python -m validation.worker`` runs a lowering on
JSON lines in its own process, with a time limit per job; its result names
the package it loaded, the route of C, and the stage and origin of a
failure.

``validation/compare.py`` compares tag by tag.  A value passes when

.. math::

   |v_\mathrm{package} - v_\mathrm{oracle}| \le
   \mathrm{rel}\, |v_\mathrm{oracle}| + k_\mathrm{err}\, e + \mathrm{floor}\, s,

where :math:`e` is the package's own error estimate and :math:`s` a scale
from ``hierarchy.moments_with_scale``: for a moment the largest absolute
value the oracle returns, and for a joint cumulant the largest term its
subtraction cancels, a moment of a subset of the legs or a product of a
cumulant and a moment, at any tag at or below a compared one.  An exact zero
is compared against that scale.  A row that is not finite fails.

======================================  ========  ==============  ========
Integrator, with C in closed form       rel       k_err           floor
======================================  ========  ==============  ========
``gauss_legendre`` (14 nodes)           1e-10     0               1e-12
``nquad``                               1e-7      10              1e-12
``qmc_vectorized`` (2^14 samples)       1e-9      10              1e-8
quadrature tables of C, any method      1e-4      10              1e-6
======================================  ========  ==============  ========

Drawn cases
-----------

``validation/draw.py`` turns a seed into a model, an observable, the largest
tag weight compared and the expansion flags the model admits.  Six in ten
models are general: widths and loadings differ between components, and so
do the rates in nine of ten, amplitudes take both signs, legs come in a
random order and ``t_min`` is not 0.  Points are distinct, except under
multiplicative noise, where every leg sits at one point because the package
refuses multiplicative noise at two positions.  Two in ten models are
drawn with a diagonal C, and two in ten are isotropic, with equal rates and
``C = c I``, so ``iso_R`` and ``iso_C`` apply.  ``diag_C`` is switched on
wherever C comes out diagonal, which holds for about one general model in
ten as well: 45 % of the cases in all.  Only
numpy's ``default_rng`` draws, in a fixed order, so a seed gives the same
case on every platform.  ``tests/test_validation_draw.py`` checks properties
of the draw on seeds that a derandomized hypothesis chooses: a seed gives
one case, the flags agree with the model, and the observable is one the
oracle and the package both take.

Running the fuzzer
------------------

.. code-block:: bash

   # draw cases from seeds 0 to 299 and compare on two integrators
   python -m validation.fuzz run runs/main --count 300 \
       --methods gauss_legendre qmc_vectorized

   # add an integrator to the same seeds, then add seeds 300 to 599
   python -m validation.fuzz run runs/main --count 300 --methods nquad --resume
   python -m validation.fuzz run runs/main --count 600 --resume

   # print the summary again; reduce a failing case to its channels
   python -m validation.fuzz summary runs/main
   python -m validation.fuzz shrink runs/main --seed 5 --method gauss_legendre

   # replay on another release: an unpacked wheel and its lowering
   python -m validation.fuzz run runs/v042 --package wheel/unpacked \
       --lowering validation.lower_v042 --count 1000

The oracle runs in the fuzzer's process and each integrator's job in the
worker, so the oracle's process never imports the package.  A run writes
``records.jsonl``, one record per case and integrator, ``summary.json`` and
``run.json``, the arguments of every invocation; a run without
``--resume`` clears ``records.jsonl`` and ``run.json``.  A record's verdict is one of:

* ``pass`` or ``fail``, the comparison;
* ``refusal``: the package raised; ``lowering-error``: the lowering raised
  before reaching the package; ``callback-error``: an oracle callable that
  the package called raised;
* ``timeout``: the job ran past its deadline in the worker; ``killed``: the
  worker gave no answer within the deadline and a margin; ``memory``: the
  worker went past its memory limit; ``crashed``: the worker exited;
* ``oracle-error``, ``oracle-timeout``, ``oracle-memory``: the oracle raised,
  ran past its time limit or went past its memory limit, and nothing was
  compared;
* ``harness-error``: anything else, such as a worker that ran other
  settings than the job asked for.

``validation/shrink.py`` removes one feature of a
failing case at a time, a drift term, a noise block, a leg, a component and
so on, and keeps the removal while the comparison still fails on the same
integrator.

Memory is bounded per process and for the machine.  ``--max-rss-gb``
(default 2) limits the resident memory of each worker, read by the fuzzer
every quarter second, and ``--oracle-max-rss-gb`` (default 8) that of each
oracle evaluation; a job past its limit is stopped and recorded.  ``shrink``
takes the same two limits.  Both limits are of a whole process: a worker's
counts what the earlier integrators of its case left, and before Python
3.13, where a pool process runs many cases, an oracle's counts what earlier
cases left in it; from 3.13 each case runs in a new process.
``--min-available-gb`` is the memory a run leaves to the rest of the machine
(default 8, or a quarter of the physical memory when that is less).  A case
starts only when the memory available, less both limits of every running
case and of the new one, is still that much; with no case running, a case
starts once that much is available.  A limit of 0 is no limit.  ``--jobs``
(default the number of processors less 4) caps the cases that run at once.
Available and physical memory are the host's, as psutil reports them (on
macOS, inactive plus free memory); cgroup limits are not read.  A run
stopped by an interrupt or ``SIGTERM`` kills its pool and its workers, and a
pool process or a worker whose parent has gone ends itself within a second,
so a run killed outright leaves no case running either.  When a pool process
dies, every case running in the pool runs again, alone, in a new pool, and a
case that kills its pool process twice while it runs alone is recorded as
``harness-error``.  On 200 cases the workers stayed below 0.9 GB,
Gauss-Legendre at 14 nodes and QMC at 2^14 samples alike, and 5 of the 200
oracles went past 4 GB.

Results
-------

On this release, 300 cases (seeds 0 to 299): every comparison that finished
agreed, 299 on Gauss-Legendre, 300 on QMC and 254 on ``nquad``; the others
ran past 240 s.  A fault of 1e-3 planted in any one of nine quantities (F,
h, S2, W3, the Poisson rate, g1, an OU amplitude, ``t_min``, a rate) failed
exactly the cases where it reaches a compared value.

10,000 cases (seeds 0 to 9999) on Gauss-Legendre at 14 nodes and QMC at
2^14 samples, 240 s per job, a worker limit of 2 GB and an oracle limit of
8 GB, six cases at a time, on Python 3.14 (a new process for each case):

==================  =====  ====  =======  =============
Integrator          pass   fail  timeout  oracle-memory
==================  =====  ====  =======  =============
``gauss_legendre``  9,916  0     13       71
``qmc_vectorized``  9,926  3     0        71
==================  =====  ====  =======  =============

The largest Gauss-Legendre difference that passed is 0.002 of its tolerance.
The three QMC failures:

=====  ==================  ==========================  ===============
Seed   Ratio to tolerance  QMC minus oracle (error)    GL minus oracle
=====  ==================  ==========================  ===============
3843   1.48                -1.5e-8 (1.0e-9)            1.0e-17
5872   1.10                5.0e-11 (9.3e-13)           -4.3e-19
8646   1.30                7.0e-9 (4.1e-10)            -3.0e-18
=====  ==================  ==========================  ===============

Each QMC value lies 15 to 53 times its own error estimate from the oracle.
With Sobol seeds 8 and 9 at 2^14 samples each difference is below its error
estimate, and at 2^18 samples it is 1.1e-11 or less on all three seeds.  The
error estimate of the single run was too small; the tolerance was not
changed.

The 13 Gauss-Legendre timeouts all have exponential Poisson pulses with a
third cumulant, and an integrated leg or a largest weight of 3; QMC passed
all 13.  The 71 oracles past 8 GB all belong to models with three
components, three legs, a cubic drift and a largest weight of 2.  Both sets
were run again, the timeouts with 1200 s per job and the oracles one case at
a time with a limit of 16 GB:

=========================================  =====  ======  ===========================
Run again                                  Cases  Passed  Not compared
=========================================  =====  ======  ===========================
Gauss-Legendre timeouts, 1200 s per job    13     11      2 past 1200 s
oracles past 8 GB, 16 GB: Gauss-Legendre   71     36      32 past 16 GB, 3 past 240 s
oracles past 8 GB, 16 GB: QMC              71     39      32 past 16 GB
=========================================  =====  ======  ===========================

No comparison failed.  The Gauss-Legendre jobs that passed took 227 to
1,185 s; the oracles that finished took 76 to 162 s and 8,102 to 15,742 MB.
Two of them stayed just below 8 GB this time, so a peak moves by about
100 MB from one run to the next.

Resident memory, sampled every quarter second, in MB of 2^20 bytes (the
limits are in GB of 2^30 bytes).  The worker row is over the 19,858 jobs
that ran, less 267 that answered before the first sample; the oracle row
is over the 9,929 oracles that finished.  The other 71 were stopped at
8 GB.

=======  ======  ======  ======  ========
Process  Median  90 %    99 %    Largest
=======  ======  ======  ======  ========
worker   109     258     463     1,008
oracle   63      272     3,091   8,017
=======  ======  ======  ======  ========

``nquad`` was not run on these cases; its comparisons are the 300 above.

On the 0.4.2 wheel, 1,000 cases gave 749 failing comparisons of the 893 that
finished.  Each failure was compared again with every known cause removed,
then with each put back in turn; the failures this left over were run on the
later releases and on commits between 0.5.0 and 0.6.0.  Every failure is
attributed except 19 whose variant ran past 240 s.  A failure needs every
error whose removal it takes to pass, so the last column sums to more than
749.

=================================================================  ===================  ==========  ========
Error in 0.4.2                                                     Fixed in             First seed  Failures
=================================================================  ===================  ==========  ========
``System.t_min`` does not reach the integrators                    0.5.0                1           668
``iso_C=True`` multiplies C by N                                   0.5.0                5           300
the diagonal-C fast path ignores pinned labels (``nquad``)         0.5.0                5           80
Gauss-Legendre and ``nquad`` do not split at kinks                 0.5.0, 0.6.0, 0.7.0  5           95
a callable cumulant is evaluated at one leg order                  0.5.0                235         13
an ``equal_time`` vertex takes the leg structure of one routing    0.6.0                140         11
=================================================================  ===================  ==========  ========

The CHANGELOG entry gives the smallest case found for each error and the
counts that rest on a step past its time limit.

Not covered
-----------

The draw does not produce one component or more than three, tags of weight
above 3 (diagrams of order 4 and higher), more than one integrated leg,
temporal kernels other than the exponential (``GaussianTemporal``, an
``ExplicitR`` spline, a general κ²), four or more legs, or a dense drift.
``qmc_scalar``, ``qmc`` and the package's quadrature tables of C are not
drawn either: every comparison takes C in closed form.  The oracle does not
compute three or more distinct external times or response legs.
