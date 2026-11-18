"""Mutation entries for ``perturbation.py``, the expansion.

Each entry is ``(label, path, old, new, tests, why, kind)``: the fields of
``Mutation`` in ``tools/mutation_check.py``, which reads ``ENTRIES`` from
every module of this directory.  ``path`` is relative to ``src/sft_wick``.
``old`` must occur once in that file.  ``tests`` is a pytest selector that
passes on unmutated source; a test marked ``slow`` or ``heavy`` needs a
``-m`` in the selector.  ``kind`` is ``historical`` for a defect a commit
fixed, whose ``why`` names the commit and the CHANGELOG heading, and
``operator`` for a small edit at a site the tests should guard.

No entry edits the canonical forms or the graph enumeration of the
expansion.
"""

ENTRIES: list[tuple[str, str, str, str, str, str, str]] = [
    # ---- historical ----
    (
        "historical-shared-external-label-accepted",
        "perturbation.py",
        "    if not repeated:\n"
        "        return\n"
        "    lab = repeated[0]\n",
        "    if True:\n"
        "        return\n"
        "    lab = repeated[0]\n",
        "tests/test_coincident_external_labels.py",
        'Two external operators sharing a spatial label are expanded at '
        'interacting orders, which collapses their component routings.  Fixed '
        'in 8771c52 and, for compute_moment_numerical, 10ddce1; CHANGELOG '
        '0.4.0, "Breaking: external operators may no longer share a spatial '
        'label", and 0.4.2, "Fixed: `compute_moment_numerical` bypassed the '
        'coincident-label guard".',
        "historical",
    ),
    (
        "historical-callable-coupling-at-one-leg-order",
        "perturbation.py",
        "            key = key_for(e.name, tuple(e.spatial_args))\n",
        "            key = key_for(e.name, next((legs for n, legs in "
        "legs_by_key.values() if n == e.name), tuple(e.spatial_args)))\n",
        "tests/test_nonlocal_leg_order.py",
        'A callable non-local coupling is evaluated once, at the legs of the '
        'first term, and every term reads that tensor with its own indices.  '
        'Fixed in c9daf92; CHANGELOG 0.5.0, "Fixed: a callable non-local '
        'coupling was evaluated at one leg order".',
        "historical",
    ),
    (
        "historical-callable-copies-share-one-leg-structure",
        "perturbation.py",
        "        classes = frozenset(\n"
        "            frozenset(s.spatial_args)\n"
        "            for s, v in zip(syms, vertices)\n"
        "            if v.equal_time\n"
        "        )\n"
        "        absorbed = frozenset(\n"
        "            leg\n"
        "            for s, v in zip(syms, vertices)\n"
        "            if v.already_R_contracted\n"
        "            for leg in s.spatial_args\n"
        "        )\n",
        "        classes = frozenset(\n"
        "            frozenset(vi.spatial_variables)\n"
        "            for vi, v in zip(vertex_instances, vertices)\n"
        "            if v.equal_time\n"
        "        )\n"
        "        absorbed = frozenset(\n"
        "            leg\n"
        "            for vi, v in zip(vertex_instances, vertices)\n"
        "            if v.already_R_contracted\n"
        "            for leg in vi.spatial_variables\n"
        "        )\n",
        "tests/test_callable_vertex_copies.py",
        'The leg structure of the copies of an already_R_contracted vertex '
        'is read off the vertex instances, so every term of a coupling sum '
        'that routes the legs between the copies is integrated with the '
        'measure of one routing.  Fixed in b35e247; CHANGELOG 0.6.0, '
        '"Fixed: the copies of one vertex shared a leg structure".  The '
        'equal_time half of that fix does not come back: since 266b217 the '
        'canonical form keeps the legs of each equal_time copy apart, so '
        'every equal_time term already has the structure of the instances.',
        "historical",
    ),
    (
        "historical-delta-reads-a-literal-from-0",
        "perturbation.py",
        "    component = pinned_component(i, index_map)\n"
        "    if component is None:\n"
        "        raise KeyError(\n"
        "            f\"KroneckerDelta index",
        "    component = index_map[i] if i in index_map else (\n"
        "        int(i) if str(i).isdigit() else None)\n"
        "    if component is None:\n"
        "        raise KeyError(\n"
        "            f\"KroneckerDelta index",
        "tests/test_literal_component_labels.py",
        'A Kronecker delta of the scalar and batched coupling interpreters '
        'reads a literal component label from 0 while symbols and legs read it '
        'from 1.  The compiled coupling plans keep the fix.  Fixed in 3ef85c3; '
        'CHANGELOG 0.7.0, "Fixed: literal component labels in Kronecker deltas '
        '(L0)".',
        "historical",
    ),
    # ---- operator ----
    (
        "operator-split-keeps-the-first-term-of-a-sum",
        "perturbation.py",
        "            return Sum(tuple(rewrite(t) for t in e.terms))\n",
        "            return Sum(tuple(rewrite(t) for t in e.terms[:1]))\n",
        "tests/test_nonlocal_leg_order.py",
        "Splitting a callable coupling by leg order keeps only the first term "
        "of each sum.",
        "operator",
    ),
    (
        "operator-split-leaves-products-unrenamed",
        "perturbation.py",
        "            return Product(tuple(rewrite(f) for f in e.factors))\n",
        "            return e\n",
        "tests/test_nonlocal_leg_order.py",
        "Symbols inside a product keep the name of the callable, so no leg "
        "order is evaluated for them.",
        "operator",
    ),
    (
        "operator-split-records-reversed-legs",
        "perturbation.py",
        "            legs_by_key[key] = (name, legs)\n",
        "            legs_by_key[key] = (name, legs[::-1])\n",
        "tests/test_nonlocal_leg_order.py",
        "Each leg order of a callable coupling is evaluated at its legs in "
        "reverse order.",
        "operator",
    ),
    (
        "operator-apply-diagonal-needs-both-flags",
        "perturbation.py",
        "        diag_C = diag_C or iso_C\n"
        "        if not diag_R and not diag_C:\n"
        "            return self\n",
        "        diag_C = diag_C or iso_C\n"
        "        if not diag_R or not diag_C:\n"
        "            return self\n",
        "tests/test_perturbation.py tests/test_simplify.py",
        "apply_diagonal leaves a term unchanged unless both diag_R and diag_C "
        "are on.",
        "operator",
    ),
    (
        "operator-apply-diagonal-coupling-keeps-old-indices",
        "perturbation.py",
        "        substituted_coupling = simplify(_apply_index_sub(self.coupling_sum, sub))\n",
        "        substituted_coupling = simplify(self.coupling_sum)\n",
        "tests/test_perturbation.py tests/test_simplify.py",
        "apply_diagonal renames the propagators but not the coupling, which "
        "keeps indices that are no longer summed.",
        "operator",
    ),
    (
        "operator-apply-diagonal-leaves-propagator-indices",
        "perturbation.py",
        "        new_props = [_apply_index_sub(p, sub) for p in self.propagators]\n",
        "        new_props = list(self.propagators)\n",
        "tests/test_perturbation.py tests/test_simplify.py",
        "apply_diagonal renames the coupling but not the propagators, which "
        "keep indices that are no longer summed.",
        "operator",
    ),
]
