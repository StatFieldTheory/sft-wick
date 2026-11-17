"""Mutation entries for ``workflow/``, the high-level layer.

Each entry is ``(label, path, old, new, tests, why)``: the fields of
``Mutation`` in ``tools/mutation_check.py``, which reads ``ENTRIES`` from
every module of this directory.  ``path`` is relative to ``src/sft_wick``.
``old`` must occur once in that file.  ``tests`` is a pytest selector that
passes on unmutated source; a test marked ``slow`` or ``heavy`` needs a
``-m`` in the selector.
"""

ENTRIES: list[tuple[str, str, str, str, str, str]] = []
