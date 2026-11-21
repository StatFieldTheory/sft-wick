Comparing numerical results across versions
===========================================

``sft-wick compare-versions`` runs the same cases against two local sft-wick
versions and writes a JSON report. The inputs may be local wheels, directories
containing an installed ``sft_wick`` package, or source trees. The command
starts a fresh process for each case and version, with the same interpreter
and dependencies. Configure any dependencies needed by older releases in
that environment.

Create a corpus manifest containing one or more workflow configurations.
Resolve paths relative to the manifest. Each workflow still resolves its own
input paths relative to its YAML file. Use ``inputs`` to list additional local
files read by the case, such as files imported by a callable module.

.. code-block:: yaml

   cases:
     - id: brownian-pinned
       config: pinned.yaml
     - id: shot-noise
       config: shot_noise.yaml
       inputs: [shot_noise_covariance.py]

.. code-block:: console

   sft-wick compare-versions corpus.yaml \
     --baseline sft_wick-0.6.1-py3-none-any.whl --candidate source-tree \
     --declared changes.yaml --json drift.json

The comparison disables workflow outputs and expansion and propagator caches.
The workers run the workflow stages serially and fix the Python hash seed.
QMC workflows require a fixed ``sweep.seed``.
The CLI limits the time for each case and version to ``--timeout`` seconds
(300 by default). Use ``--force`` to replace an existing report.

Each workflow row in the report is the sum across diagrams at one perturbative
order and one set of sweep coordinates. Rows match by coordinates, including
external times and components when present. This makes the comparison
independent of diagram indices. One reported error for each side is the sum
of the per-diagram errors at that row, with no assumption of independent
errors.

The numerical comparison uses the following classifications.

* ``bitwise`` means identical binary64 values.
* ``round-off`` means an absolute difference no larger than
  ``4.5e-16 * max(abs(baseline), abs(candidate))``.
* ``within uncertainty`` means a larger difference within the sum of the
  two reported error envelopes on QMC runs. This still needs a declaration.
* ``moved`` means a difference beyond round-off without the QMC condition.
* ``redrawn`` marks a QMC movement explicitly declared as a sampling change.

A declaration names a case and a subset of its row coordinates, specifies
the version change, and gives a reason. A row can match one declaration.
An empty ``where`` selects all moved rows of the named case. A declaration may
classify a QMC movement as ``moved`` or ``redrawn``.

.. code-block:: yaml

   changes:
     - case: brownian-pinned
       where: {order: 0, t_y: 0.65}
       kind: moved
       reason: Split quadrature at the pinned external time.

The report records the baseline and candidate values, error estimates,
declarations, and input fingerprints. A wheel fingerprint covers the
archive bytes. A source or installed-tree fingerprint covers the Python
files under ``sft_wick`` and its version metadata. A refusal, missing row,
or undeclared movement fails the gate and the CLI exits with status 1.

Cases using the Python API
--------------------------

For a case spanning releases that use different YAML interfaces, supply a
local Python driver and the method it runs. The driver imports from the
version selected by the worker. Its zero-argument ``run()`` returns a list of
dicts with unique ``coordinates``, a finite numerical ``value``, and an
``error`` (a nonnegative finite number or ``None``). Use paths relative to
the driver file for adjacent input files. Set a fixed seed in QMC drivers
so both versions receive the same sampling input.

.. code-block:: yaml

   cases:
     - id: shot-noise-python
       driver: shot_noise_comparison.py
       method: gauss_legendre
