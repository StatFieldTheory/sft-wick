"""Mutation entries for ``evaluate.py``, the numerical pipeline.

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
    # ---- historical ----
    (
        "historical-iso-c-trace",
        "evaluate.py",
        "    return c if batched else c[0]\n",
        "    return diag.sum(axis=1) if batched else diag.sum(axis=1)[0]\n",
        "tests/test_iso_c_value.py",
        'An index-free C under iso_C is the trace N c instead of c.  Fixed in '
        'f6c4e5d; CHANGELOG 0.5.0, "Fixed: `iso_C=True` multiplied every C '
        'propagator by N".',
        "historical",
    ),
    (
        "historical-t-min-not-reaching-integrators",
        "evaluate.py",
        "            f\"t_min={model_t_min}, from which C was built.\"\n"
        "        )\n"
        "    return t_min\n",
        "            f\"t_min={model_t_min}, from which C was built.\"\n"
        "        )\n"
        "    return 0.0\n",
        "tests/test_system_t_min.py",
        'The time integrals start at 0 when the system\'s t_min is passed, '
        'while the propagators are built from t_min.  Fixed in f6c4e5d; '
        'CHANGELOG 0.5.0, "Fixed: `System.t_min` did not reach the '
        'integrators".',
        "historical",
    ),
    (
        "historical-r-group-takes-any-position",
        "evaluate.py",
        "    pair = _conflicting_group_positions(spatial, positions)\n"
        "    if pair is None:\n"
        "        return\n",
        "    return\n",
        "tests/test_group_position_conflict.py",
        'Two points of one R-connected direction group at different positions '
        'are not refused at any of the evaluate.py entry points that call '
        '_require_one_position_per_group, and the group takes whichever point '
        'the frozenset yields first.  Fixed in ae0dbf9; CHANGELOG 0.6.0, "Breaking: two '
        'points of one R-connected group may not be at different positions".',
        "historical",
    ),
    (
        "historical-r-cache-mismatch-accepted",
        "evaluate.py",
        "        if kind:\n"
        "            return _R_MISMATCH[kind].format(\n",
        "        if False:\n"
        "            return _R_MISMATCH[kind].format(\n",
        "tests/test_r_cache_mismatch.py",
        'R propagators whose component indices do not match the kind of R in '
        'the cache are evaluated: the leg components are summed instead of '
        'pinned, or an index-free R is read as its trace.  Fixed in 043720d; '
        'CHANGELOG 0.5.0, "Fixed: L0 terms evaluated against a cache with the '
        'other kind of R".',
        "historical",
    ),
    (
        "historical-no-cut-at-a-pinned-external-time",
        "evaluate.py",
        "            cut_times.append((var, float(fixed_times[other])))\n",
        "            continue\n",
        "tests/test_kink_split_external_time.py",
        'A kink between an integration time and an external pinned at its own '
        'time is not cut, so Gauss-Legendre is algebraic there.  Fixed in '
        '46134e3; CHANGELOG 0.6.0, "Fixed: the domain is cut at a kink against '
        'a fixed external time".',
        "historical",
    ),
    (
        "historical-matrix-r-reads-another-propagators-indices",
        "evaluate.py",
        "        absorbed = set(self.spatial.r_absorbed_pairs)\n"
        "        return tuple(\n"
        "            p\n"
        "            for p in self.diagram_term.propagators\n",
        "        absorbed = set(self.spatial.r_absorbed_pairs)\n"
        "        props = self.diagram_term.propagators\n"
        "        return tuple(\n"
        "            next(q for q in props if q.kind == \"R\" and "
        "(q.spatial_left, q.spatial_right) == (p.spatial_left, p.spatial_right))\n"
        "            for p in self.diagram_term.propagators\n",
        "tests/test_matrix_r_index_and_zero_dim.py",
        'Under a matrix R each R propagator is resolved by its endpoints, so '
        'two R propagators between the same two points both read the indices '
        'of the first.  Fixed in ecb19fb; CHANGELOG 0.5.0, "Fixed: two R '
        'propagators between the same two points shared one set of component '
        'indices".',
        "historical",
    ),
    (
        "historical-diag-c-fast-path-ignores-pinned-labels",
        "evaluate.py",
        "        return None, DiagramIntegrand._resolve_component("
        "idx_name, self.fixed_indices)\n",
        "        return None, DiagramIntegrand._resolve_component(idx_name, {})\n",
        "tests/test_diag_fast_component_labels.py",
        'The diag_C scalar fast path resolves a C leg against an empty map, so '
        'a pinned observable label falls back to the trace.  Fixed in '
        '774ff9d; CHANGELOG 0.5.0, "Fixed: a pinned observable component label '
        'was ignored in the diag_C fast path".',
        "historical",
    ),
    (
        "historical-no-split-at-a-declared-coupling-kink",
        "evaluate.py",
        "        if not _declares(fn, \"has_coincident_time_kinks\"):\n"
        "            continue\n",
        "        if True:\n"
        "            continue\n",
        "tests/test_kink_split_nquad_couplings.py",
        'A coupling callable that declares has_coincident_time_kinks gives no '
        'kink pair, so Gauss-Legendre and nquad integrate across the '
        'coincident leg or partner times without a split.  Fixed in 964c2c9; '
        'CHANGELOG 0.6.0, "Improved: `nquad` splits at kinks, and coupling '
        'callables can declare kinks".',
        "historical",
    ),
    (
        "historical-gauss-legendre-not-split-at-c-kinks",
        "evaluate.py",
        "        c_kink = _shifted_panel is None and self._c_kink_at(\n"
        "            cache, t_min, t_ceiling, positions, direction\n"
        "        )\n"
        "        split = None if _shifted_panel is not None else self._sum_kink_pieces(\n"
        "            lambda orderings, cuts, **panel: self.integrate_moment_gauss_legendre(\n",
        "        c_kink = False\n"
        "        split = None if _shifted_panel is not None else self._sum_kink_pieces(\n"
        "            lambda orderings, cuts, **panel: self.integrate_moment_gauss_legendre(\n",
        "tests/test_gl_white_noise_kinks.py",
        'Gauss-Legendre takes no C propagator between two unordered times as '
        'a kink pair, so a white-noise C is integrated across its time '
        'diagonal and converges as n^-2.  Fixed in f62230c, which split the '
        'domain at those pairs, and 6b391f1, which repeats the split in each '
        'piece; CHANGELOG 0.5.0, "Improved: Gauss-Legendre splits the time '
        'domain where the integrand is kinked".  The edit also removes the C '
        'kinks found later, among them a kernel with a cusp (e77edef; '
        'CHANGELOG 0.6.0, "Fixed: a κ² with a `|Δt|` cusp is declared as '
        'kinking C") and an undeclared closed-form C (53300b6; CHANGELOG '
        'Unreleased, "Fixed: an undeclared kink of a closed-form C was '
        'integrated without a split").',
        "historical",
    ),
    # ---- operator ----
    (
        "operator-iso-c-accepts-an-uneven-diagonal",
        "evaluate.py",
        "    if np.any(_departs_relative(diag, c[:, None])) or (\n",
        "    if False or (\n",
        "tests/test_iso_c_value.py",
        "iso_C accepts a C whose diagonal entries differ and contributes their "
        "mean.",
        "operator",
    ),
    (
        "operator-iso-c-accepts-off-diagonal-entries",
        "evaluate.py",
        "        off is not None and np.any(off > _ISO_C_RTOL * np.abs(c)[:, None, None])\n",
        "        False and np.any(off > _ISO_C_RTOL * np.abs(c)[:, None, None])\n",
        "tests/test_iso_c_value.py",
        "iso_C accepts a C with off-diagonal entries and contributes its "
        "diagonal only.",
        "operator",
    ),
    (
        "operator-absorbed-r-factors-kept",
        "evaluate.py",
        "    if not spatial.r_absorbed_pairs:\n"
        "        return spatial.r_propagators\n",
        "    if True:\n"
        "        return spatial.r_propagators\n",
        "tests/test_R_contracted_vertex.py",
        "The R factors absorbed into an already_R_contracted vertex are "
        "multiplied in again, giving R(t, t) = 0 after the leg aliases onto "
        "its partner.",
        "operator",
    ),
    (
        "operator-first-kept-r-factor-dropped",
        "evaluate.py",
        "    return tuple(p for p in spatial.r_propagators if p not in absorbed)\n",
        "    return tuple(p for p in spatial.r_propagators if p not in absorbed)"
        "[1:]\n",
        "tests/test_R_contracted_vertex.py",
        "Beside an absorbed R, the first kept R factor is left out of the "
        "product.",
        "operator",
    ),
    (
        "operator-retarded-at-equal-times",
        "evaluate.py",
        "    return t_left > t_right\n",
        "    return t_left >= t_right\n",
        "tests/test_msr_numerics_regressions.py",
        "Theta(0) = 1 in the numerical R: an R propagator at equal times is "
        "R(t, t) instead of 0.",
        "operator",
    ),
]
