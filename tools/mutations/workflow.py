"""Mutation entries for ``workflow/``, the high-level layer.

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
        "historical-diagonal-flags-on-a-non-diagonal-system",
        "workflow/system.py",
        "            if _offdiagonal(fn(*args)):\n"
        "                return True\n",
        "            if False:\n"
        "                return True\n",
        "tests/test_l1_structure_guards.py",
        'The probe of a callable R, kappa2 or sigma2 never reports an '
        'off-diagonal entry, so System.expand and System.propagators accept '
        'diag_R=True and diag_C=True on a dense R or a component-mixing noise '
        'and keep only its diagonal.  The later per-position check of '
        'Expansion.evaluate (e93b720) is left in place.  Fixed in f6c4e5d; CHANGELOG 0.5.0, "Fixed: L1 defaults that '
        'dropped off-diagonal R and C entries".',
        "historical",
    ),
    # ---- operator ----
    (
        "operator-msr-factor-over-m-not-m-factorial",
        "workflow/specs.py",
        "    return -i_power(m) / math.factorial(m)\n",
        "    return -i_power(m) / m\n",
        "tests/test_higher_cumulants.py",
        "The MSR factor of a non-local vertex is -(i^m)/m instead of "
        "-(i^m)/m!, which differs from order 3.",
        "operator",
    ),
    (
        "operator-msr-factor-sign",
        "workflow/specs.py",
        "    return -i_power(m) / math.factorial(m)\n",
        "    return i_power(m) / math.factorial(m)\n",
        "tests/test_higher_cumulants.py",
        "The MSR factor of a non-local vertex is +(i^m)/m! instead of "
        "-(i^m)/m!.",
        "operator",
    ),
    (
        "operator-stratonovich-drift-doubled",
        "workflow/specs.py",
        '        return alpha * np.einsum("ikj,jk->i", g1, g0), alpha * linear\n',
        '        return 2.0 * alpha * np.einsum("ikj,jk->i", g1, g0), alpha * linear\n',
        "tests/test_multiplicative_noise_l1.py",
        "The constant part of the noise-induced Stratonovich drift is "
        "g1 g0 instead of g1 g0 / 2.",
        "operator",
    ),
]
