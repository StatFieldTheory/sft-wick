"""Mutation entries for ``indices.py`` and ``propagators.py``, the names and
contractions the expansion is built from.

Each entry is ``(label, path, old, new, tests, why, kind)``: the fields of
``Mutation`` in ``tools/mutation_check.py``, which reads ``ENTRIES`` from
every module of this directory.  ``path`` is relative to ``src/sft_wick``.
``old`` must occur once in that file.  ``tests`` is a pytest selector that
passes on unmutated source; a test marked ``slow`` or ``heavy`` needs a
``-m`` in the selector.  ``kind`` is ``historical`` for a defect a commit
fixed, whose ``why`` names the commit and the CHANGELOG heading, and
``operator`` for a small edit at a site the tests should guard.
"""

ENTRIES: list[tuple[str, str, str, str, str, str, str]] = [
    (
        "historical-generated-names-take-observable-labels",
        "indices.py",
        "            labels.add(op.component_index)\n"
        "    return frozenset(labels)\n",
        "            labels.add(op.component_index)\n"
        "    return frozenset()\n",
        "tests/test_observable_label_collision.py",
        'No observable label is reserved, so a vertex point or index can be '
        'generated with the name of an external one (y_0, i_1) and merged with '
        'it.  Fixed in 0ab76ab; CHANGELOG 0.7.0, "Fixed: observable labels '
        'spelled like the generated names".',
        "historical",
    ),
    (
        "operator-equal-point-r-kept",
        "propagators.py",
        "    if r_vanishes([(phi_op.spatial_arg, psi_op.spatial_arg)]):\n"
        "        return None\n",
        "    if False:\n"
        "        return None\n",
        "tests/test_propagators.py",
        "contract_pair keeps the equal-point R(x, x), which Theta(0) = 0 sets "
        "to zero.",
        "operator",
    ),
]
