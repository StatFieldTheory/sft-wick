# Test fixtures

`quadratic_white_reference_source.tar.gz` holds an earlier implementation of
`src/sft_wick/quadratic.py` and `src/sft_wick/white_cumulant.py`, written
before the reuse optimisations of 0.7.0. The file bytes are unchanged from
that implementation; the archive holds only these two files.

SHA-256 of the two members:

- `quadratic.py`: `eee500133e818e60cdda1c0ea268a8cf6f61a69d2ea4b0b6e7423187eaf7ed89`
- `white_cumulant.py`: `83849f2081fad5b3e316699d8d48afd69f0f528176a7aae2e430703608088a66`

`tests/test_quadratic_white_reference_parity.py` loads both modules from the
archive, checks each against its SHA-256 pin before executing it, and
compares the current implementation with it for byte-level numerical parity.
`tests/test_quadratic_symmetrization.py` loads `quadratic.py` the same way and
compares the prepared matrices, endpoints and constants, and the single-law
spectra and cumulants, on inputs whose halves are exact in float64, given in C
order, in Fortran order and as transposed views.
The archive sits with the tests so that a clean checkout or a source
distribution can run them. Do not regenerate it when the implementation
changes: it is the fixed reference.
