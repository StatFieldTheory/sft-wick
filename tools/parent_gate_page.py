"""The page of the parent-commit gate's replay, rendered from its record alone.

Loaded by ``tools/parent_gate_record.py``, which re-exports
:func:`render_page`.  The page is ``docs/verification/parent_gate.rst``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

#: The verdicts in the order the gate tries them, with their rules.
VERDICT_RULES = (
    ("no_code_change", "the change makes no difference under ``src/``"),
    ("no_touched_tests", "the change adds, modifies or renames no ``tests/**/test_*.py``"),
    ("environment", "side head: the change's own run of its touched files has a failing "
                    "or erroring test, a collection error, a time or memory kill, an exit "
                    "code outside 0, 1 and 5, or no report"),
    ("no_evidence", "no test of the touched files passes on the head"),
    ("environment", "side parent: the parent's own run of the touched files that exist "
                    "there has one of the faults of the head side"),
    ("overlay_not_run", "the run on the parent's source wrote no report, or exited "
                        "outside 0 and 1, without being killed at a limit"),
    ("accepted", "a test that passes on the head fails on the parent's source with an "
                 "assertion"),
    ("crash_only", "tests that pass on the head fail on the parent's source, at least "
                   "one by raising and none with an assertion"),
    ("resource_only", "they fail there only at a time or memory limit, possibly beside "
                      "names the change adds"),
    ("new_api_only", "they fail there only through names the change adds"),
    ("passes_on_parent", "no test that passes on the head fails on the parent's source"),
)

INTRODUCTION = """\
``tools/parent_gate.py`` runs the tests a change adds or modifies on the
source of the change's parent.  A test written for a fix should fail there:
if it passes on the parent's source, it does not tell the fixed code from
the code before the fix.  For a change from a parent P to a head C the gate
makes up to three pytest runs of C's added, modified and renamed
``tests/**/test_*.py``, {workers} in the replay below, with ``-m "not slow"``,
``--continue-on-collection-errors``, ``PYTHONHASHSEED=0`` and
``MUTATION_HARNESS=1``:

- **head**: on C's own tree;
- **parent's own**: P's versions of the touched files that exist at P, on
  P's tree;
- **overlay**: C's touched files on a tree that is C everywhere except
  ``src/``, which is P's.  Fixtures, ``examples/``, ``validation/`` and
  ``conftest.py`` come from C.

Each tree is a ``git archive`` of its commit, and each run checks first that
a process started there imports the package from that tree's ``src``.  In a
pull request P is the merge base of the base branch and the head; in the
replay below it is the first parent.

Kinds of failure
----------------

A test that passes on the head and fails on the overlay is classed by the
exception the JUnit report gives and the frame that raised it.  In a chained
traceback that frame is the one of the last exception, the one that failed
the test:

- **missing name**: an ``ImportError``, ``ModuleNotFoundError``,
  ``AttributeError`` or ``NameError`` raised outside ``src/`` whose missing
  name is defined in C's source and not in P's (an import, a
  ``monkeypatch.setattr``, an attribute), or a ``TypeError`` for a keyword
  that C's function or class takes and P's does not;
- **assertion**: an ``AssertionError`` or pytest's ``Failed``
  (``pytest.fail``, "DID NOT RAISE", a strict XPASS) raised in the call or
  in a fixture, from a frame outside ``src/``;
- **crash**: any other exception, an assertion or a missing name inside
  P's ``src/`` included;
- **time or memory**: a run killed at its time limit or above its memory
  limit.

A file that does not import on the parent's source counts each of its tests
with the kind of that import error.
"""

CANNOT_JUDGE = """\
What the gate cannot judge
--------------------------

- A change with no difference under ``src/`` passes: there is no source to
  tell apart.  A vacuous test added with no source change passes too.
- A test that fails on the parent's source only by raising cannot be told
  from a test of an option value the parent refuses with an exception:
  both are crashes.  Such a change is rejected unless one of its tests also
  asserts.
- A dependency version the run does not have hides a defect that needs it;
  the replay classes such a commit as environment only through a reviewed
  entry of ``DEPENDENCY_TRIGGERS`` in ``tools/parent_gate_record.py``.

Writing a fix's tests so the gate can see them
----------------------------------------------

- Import a helper the fix adds inside the test that uses it, not at module
  level, so the file still imports on the parent's source and its other
  tests run there.
- For a fix of a crash, compare the returned value in a test that also
  asserts it, as the refusal tests do with ``pytest.raises``.
- For a fix of a run time or memory that grows without bound, give the test
  an input small enough to finish on the parent's source and assert the
  count or the refusal the fix adds, so the parent fails an assertion
  rather than a limit.

The workflow
------------

``.github/workflows/parent-gate.yml`` runs the gate on every pull request,
with the copy of ``tools/`` from the base branch when it has one, and with
``--pytest-workers 2``.  The job is advisory: its verdict is in the job
summary and the uploaded ``parent-gate.json``, and a failing job does not
fail the workflow.  An undetermined verdict (exit 3) does not fail the job;
it means that the change's own tests, or the parent's, fail under the gate.
The job stays advisory while the replay below accepts less than 90 % of
its denominator; ``tests/test_parent_gate.py`` fails if the workflow
blocks before then.
"""


def _escape(text: str) -> str:
    return re.sub(r"([\\`*_|])", r"\\\1", text)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f} %"


def _table(widths: str, header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [".. list-table::", "   :header-rows: 1", f"   :widths: {widths}", ""]
    for row in [header, *rows]:
        lines += [f"   * - {row[0]}", *(f"     - {cell}" for cell in row[1:])]
    return lines + [""]


def _figure_line(s: Mapping) -> str:
    env, by = s["environment"], s["by_verdict"]
    return (f"{s['accepted']} of {s['denominator']} commits pass the gate, "
            f"{_percent(s['rate'])}: {by.get('accepted', 0)} with a witness, "
            f"{by.get('no_code_change', 0)} with no source change; {env['total']} classed "
            f"environment and not counted: {env['head']} head, {env['parent']} parent, "
            f"{env['dependency']} dependency, of them {env['timeout']} timeouts and "
            f"{env['memory']} memory kills.")


def _commit_cell(entry: Mapping) -> str:
    if entry["witnesses"]:
        witness = entry["witnesses"][0]
        return f"``{witness['id']}`` ({witness['exception']})"
    return _escape(entry["reason"])


def _section_lines(title: str, s: Mapping, record: Mapping) -> list[str]:
    lines = [title, "-" * len(title), ""]
    if s["counted"]:
        lines += [_figure_line(s), ""]
        lines += _table("70 30", ["Figure", "Value"],
                        [[_escape(label), _percent(value)]
                         for label, value in s["figures"].items()])
    else:
        lines += [f"{s['population']} commits; {s['accepted']} of {s['denominator']} "
                  f"outside environment pass the gate ({_percent(s['rate'])}).", ""]
    lines += _table("60 40", ["Verdict", "Commits"],
                    [[f"``{v}``", str(n)] for v, n in s["by_verdict"].items()])
    triggered = [e for e in s["commits"] if e.get("side") == "dependency"]
    if triggered:
        lines += ["Commits classed environment by a dependency trigger:", ""]
        lines += _table("15 55 30", ["Commit", "Subject", "Reason"],
                        [[f"``{e['sha'][:7]}``", _escape(e["subject"]), _escape(e["reason"])]
                         for e in triggered])
    lines += _table("10 40 15 35", ["Commit", "Subject", "Verdict", "First witness or reason"],
                    [[f"``{e['sha'][:7]}``", _escape(e["subject"]), f"``{e['verdict']}``",
                      _commit_cell(e)] for e in s["commits"]])
    return lines


def _introduction(record: Mapping) -> str:
    workers = record["settings"].get("pytest_workers", 0)
    how = f"each with {workers} pytest-xdist workers" if workers >= 2 else "each serial"
    return INTRODUCTION.format(workers=how)


def _environment_sentence(env: Mapping, digest: str) -> str:
    """Where the replay ran, its distributions and digest, then its memory measure."""
    versions = [f"{k} {v}" for k, v in env.items() if k not in ("python", "platform", "memory")]
    listed = " and ".join([", ".join(versions[:-1]), versions[-1]] if versions[:-1] else versions)
    memory = (f" The memory limit of a run applies to the summed {env['memory']} of its "
              "processes." if env.get("memory") else "")
    return (f"The replay below ran under Python {env.get('python')} on {env.get('platform')}"
            + (f", with {listed}" if listed else "") + f"; rules digest ``{digest[:16]}``."
            + memory)


def render_page(record: Mapping) -> str:
    """The page of a replay record."""
    decisions = record["decisions"]
    env = record["environment"]
    counted, earlier = record["counted"], record["earlier"]
    lines = [".. This file is GENERATED by tools/parent_gate.py render; do not edit.", "",
             "Tests that fail before their fix", "================================", "",
             *_introduction(record).splitlines(), "", "Verdicts", "--------", "",
             "The first rule that applies gives the verdict.", ""]
    lines += _table("20 60 20", ["Verdict", "Rule", "Decision"],
                    [[f"``{v}``", rule, decisions.get(v, "")] for v, rule in VERDICT_RULES])
    lines += [*CANNOT_JUDGE.splitlines(), "", _environment_sentence(env, record["rules_digest"]), ""]
    lines += _section_lines(f"Replay over the fix commits of {counted['spec']}", counted, record)
    if earlier:
        lines += _section_lines("Earlier history (not counted)", earlier, record)
    return "\n".join(lines).rstrip("\n") + "\n"
