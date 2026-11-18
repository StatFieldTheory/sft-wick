The mutation harness
====================

``tools/mutation_check.py`` measures whether the test suite notices defects.
It applies one textual edit, a mutant, to a file of the package under test,
by default ``src/sft_wick``, runs the pytest selector named for that edit,
restores the file and reports whether a selected test failed.
``tools/mutation_tree.py`` holds the code that starts processes on a tree,
copies it and keeps the journal described below.  Both use the standard
library only and are not part of the wheel, which packages ``src/sft_wick``
alone.  Their tests are listed in :doc:`catalog`.

The corpus
----------

The mutants are kept in ``tools/mutations/`` unless ``--corpus`` names
another directory, one module per source area, each with a list ``ENTRIES``
of ``(label, path, old, new, tests, why)``.  ``path`` is relative to the
package directory, ``src/sft_wick`` by default, ``old`` is replaced by ``new``,
``tests`` is a pytest selector and ``why`` names the defect the mutant
stands for.  The harness reads every ``*.py`` module of the directory, in
the order of the file names, when it starts.  It stops with exit status 3,
naming the module and the position, if a module cannot be run, has no
``ENTRIES`` list, or has an entry that is not six strings, whose ``path`` is
absolute or contains ``..``, whose ``old`` is empty or equal to ``new``,
whose label or selector is empty, or whose label another entry already
has.  The corpus is empty for now: with no entry the harness prints that
the corpus has no entry and exits with status 2.

Verdicts
--------

Two checks run on the unmutated source before a mutant is written:

* ``old`` occurs once in the target file, overlapping occurrences counted.
  No occurrence is ``STALE``, two or more are ``AMBIGUOUS``.  A target that
  is not a file, is not UTF-8, or resolves outside the package directory
  through a symbolic link is ``STALE``.
* the selector exits 0 with at least one passing test, counted from the
  JUnit report.  Otherwise the entry is ``SELECTOR BROKEN``.  Each distinct
  selector runs once.

This repository's ``addopts`` deselects the tests marked ``slow`` and
``heavy``.  A selector whose only matches carry one of those markers runs no
test, and the verdict is ``SELECTOR BROKEN`` with "no passing test" and the
two markers in the reason.  With ``--project-root``, the reason asks to check
the selector and the markers of the project's ``addopts`` instead.  A
``-m`` in the selector replaces the one in ``addopts``, so ``-m slow`` or
``-m 'slow or heavy'`` selects such a test.

With the mutant applied, pytest's exit code decides:

=====  ================  ==================================================
code   verdict           meaning
=====  ================  ==================================================
1      ``KILLED``        tests ran and at least one failed
0      ``SURVIVED``      the selected tests pass on the mutated source
other  ``ERROR rc=N``    no test verdict: 2 interrupted, for example by a
                         collection error; 3 pytest internal error; 4
                         usage error; 5 no tests collected; negative if
                         killed by a signal
none   ``TIMEOUT``       the run exceeded its time limit and was killed
=====  ================  ==================================================

A collection error is not counted as a kill, because a mutant that breaks
an import says nothing about the assertions of the selected tests.  A
``TIMEOUT`` is not counted as a kill either: the suite has no per-test time
limit, so a mutant that makes a test loop would stop the test run rather
than fail it.

The exit status is 0 when every selected mutant is ``KILLED``
(``BASELINE OK`` under ``--baseline-only``) and every target file has the
bytes it had at the start, and also when the requested shard is empty; 1
otherwise; 2 when no label matches ``-k``, which includes an empty corpus;
and 3 when the layout options, the corpus, the ``--json`` directory or the
tree cannot be used.

Running it
----------

.. code-block:: bash

   python tools/mutation_check.py                    # every mutant, in place
   python tools/mutation_check.py --baseline-only    # the two checks only
   python tools/mutation_check.py -k iso_C --shard 1/4
   python tools/mutation_check.py --matrix --json kills.json
   python tools/mutation_check.py --root ../mutant-tree
   python tools/mutation_check.py --timeout 600

``-k`` keeps the mutants whose label contains a substring and ``--shard I/N``
keeps every N-th of those.  ``--tests SELECTOR`` replaces the selector of
every selected mutant.  ``--json FILE`` writes one record per mutant with
its verdict, the pytest exit code and the failing test files; the directory
of ``FILE`` is checked before the sweep starts.

Every pytest run starts in a session of its own.  At its time limit, or
when the harness is interrupted, the whole process group is killed, so
pytest-xdist workers and processes the tests started end with it.
``--timeout SECONDS`` sets the limit of every run.  Without it a baseline
run has no limit and a mutated run may take 10 times the duration of its
selector's baseline run plus 30 seconds.

``--matrix`` runs pytest without ``-x``, so every selected test runs, and
lists under each mutant the test files with a failing or erroring test,
read from the ``file`` attribute of the JUnit report.  The verdict rule is
the same as without it.

``--root DIR`` copies the tree to ``DIR``, which must be absent or an empty
directory outside the tree, and runs the sweep in the copy.  ``.git``,
``.claude``, ``.agent-state``, the caches, ``_build`` directories and the
journal are left out of the copy, and symbolic links are copied as links.
A copy that fails is removed and the harness exits with status 3.  The
project is not written, and the copy is left in place.

Another project's layout
------------------------

Four options describe the project under test.  Without them the harness
tests this repository.

==================  ======================================================
option              meaning, and the default
==================  ======================================================
``--project-root``  the project to test, swept in place unless ``--root``
                    is given; the repository that holds the tool
``--package``       the import name of the package, dotted for a
                    subpackage; ``sft_wick``
``--source-root``   the directory, relative to the project root, that is
                    put on the import path, with the package under it;
                    ``src``
``--corpus``        the directory of the corpus modules, inside or outside
                    the project; ``tools/mutations`` beside the tool
``--extra-path``    a further directory, relative to the project root, put
                    on the import path after the source root; repeatable;
                    none
==================  ======================================================

``--root`` keeps its meaning: it names the destination of a copy, and
``--project-root`` names the tree that is copied.  ``--project-root``,
``--corpus``, ``--root`` and ``--json`` are read against the working
directory; ``--source-root`` and ``--extra-path`` against the project root.
``--corpus`` is required whenever ``--project-root``, ``--package`` or
``--source-root`` is given, so that the corpus of this repository is never
applied to another project.  For a package ``pkg`` under ``lib/``, run
from the calling project's root with its own corpus:

.. code-block:: bash

   python path/to/mutation_check.py --project-root . --package pkg \
       --source-root lib --corpus ci/mutations

The ``path`` of each corpus entry is then relative to ``lib/pkg``, and
every process the harness starts gets ``lib``, as an absolute path, first
in ``PYTHONPATH`` and in pytest's ``pythonpath``, followed by the extra
paths.  The ``-o pythonpath`` option replaces the ``pythonpath`` list of
the project's pytest configuration, so a project whose tests import from a
second directory names it with ``--extra-path``.  The package may be a
namespace package.  The harness stops with exit status 3, naming the
option, before any test runs when ``--corpus`` is missing as described
above, when the package name is not a dotted import name, when the project
root, the source root, an extra path or the package directory is not a
directory, when the source root or an extra path is absolute or contains
``..``, when the package directory resolves outside the project root
through a symbolic link, or when the corpus is not a directory.

In Python, the same description is a ``Layout``, a frozen dataclass with
the fields ``root``, ``package``, ``source_root``, ``corpus``,
``extra_paths`` and ``deselected_markers``, the markers named in that
reason.
``DEFAULT_LAYOUT`` holds the values of this repository.  The functions that
need the layout, such as ``load_corpus``, ``run_mutation``, ``sweep``,
``mutated_copy``, ``run_in_tree`` and ``origin_problem``, take it as the
keyword ``layout`` and default to ``DEFAULT_LAYOUT``.  The two tool files
name this repository's package and ``src`` only in ``DEFAULT_LAYOUT``.
``tests/test_mutation_harness_layout.py`` runs the script on a toy package
``toypkg`` under ``lib/`` with its corpus outside the toy tree, and checks
that ``sft_wick`` occurs in none of the commands, environments, journals or
output of those runs.

``mutated_copy``, ``run_in_tree`` and ``run_mutant_in_copy`` give other
tools a copy of the tree with one mutant applied, and run a command on it
with the environment described below.  ``mutated_copy`` checks the target
in the copy as well, so that a link copied from the tree cannot lead the
write back into it.

Which tree is imported
----------------------

An installed ``sft_wick`` can shadow the tree under test.  An editable
install of another checkout puts that checkout's ``src`` on ``sys.path``
through a ``.pth`` file, and a sweep that imported it would report every
mutant as surviving.  The harness guards against this in three ways:

* every process it starts runs with the tree as working directory and with
  the tree's source root, as an absolute path, first in ``PYTHONPATH``.
  ``PYTHONPATH`` entries come before the entries of ``.pth`` files, and the
  child processes of the tests inherit the variable.
* pytest gets ``-o pythonpath=<tree>/<source root>``, followed by the extra
  paths, each quoted because pytest splits the value with ``shlex``.  The
  ``pythonpath`` of ``pyproject.toml`` takes precedence over ``PYTHONPATH``
  in the pytest process, and is read relative to the directory pytest
  starts in.
* before a sweep, one Python process with that environment imports the
  package and the harness stops with exit status 3 unless the package
  comes from the tree's source root.  The probe also catches a second copy
  of the package in the project root itself, which ``python -c`` and
  ``python -m pytest`` put ahead of ``PYTHONPATH``.

``tests/test_mutation_harness_trees.py`` puts a second toy package first on
``PYTHONPATH`` and checks, from a log written by the toy tests, that both
the pytest process and a child process it starts import the mutated tree,
in place and under ``--root``.  A test that starts a child process with an
environment of its own, or with ``python -I`` or ``-E``, is outside this
guarantee.  The tests of this repository that set ``PYTHONPATH`` for a
child put their own tree's ``src`` there.

Restoring the source
--------------------

A sweep without ``--root`` rewrites the source in place.  The original bytes
are restored in a ``finally``.  SIGINT, SIGTERM and SIGHUP are turned into
an exit that runs that ``finally``; the signal handler and an ``atexit``
hook restore the file as well.

Before a mutant is written, the harness writes ``.mutation-inflight.json``
at the root of the tree, holding the target's path, its original bytes and
a digest of the mutant, and removes it after the restore.  SIGKILL cannot
be caught, so a killed run leaves the journal behind.  The next run reads
it before anything else: if the file still holds the recorded mutant, or
already holds its original bytes, the harness puts the original back and
removes the journal; if the file has changed since, it stops with exit
status 3 and leaves both alone.  No other file is touched.

``PYTHONDONTWRITEBYTECODE`` is set, so no run leaves a ``.pyc`` that a later
run of a same-length mutant could load.  Run one in-place sweep per tree at
a time.  A sweep that has to run beside other work on the same tree uses
``--root``.
