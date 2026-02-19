# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_wick.py -v

# Run a single test by name
pytest tests/test_wick.py::test_generate_valid_pairings -v
```

## Architecture

`sft-wick` computes perturbative expansions in the MSR (Martin-Siggia-Rose) formalism using Wick's theorem. The key formula is:

⟨O⟩_S = Σ_{n=0}^{N} (-1)^n / n! ⟨O S_int^n⟩_{S₀}

**Field types**: φ (physical) and ψ (response). The critical MSR constraint is that ψ-ψ contractions vanish, so only C = ⟨φφ⟩ and R = ⟨φψ⟩ propagators appear.

### Core data flow

1. `fields.py`: `Field` declares a field type; `FieldOperator` is a concrete instance with a unique integer UID, component index, and spatial argument.
2. `vertices.py` + `action.py`: `Vertex` defines an interaction term template; `VertexInstance` is a fresh copy with non-overlapping indices. `Action` holds a list of vertices.
3. `wick.py`: `generate_valid_pairings()` enumerates complete pairings of field operators, skipping ψ-ψ pairs early. `wick_contract()` sums over all valid pairings.
4. `propagators.py`: `contract_pair()` maps (φ,φ)→C, (φ,ψ)→R, (ψ,ψ)→0.
5. `perturbation.py`: `compute_moment()` is the main entry point — expands vertex combinations, instantiates vertices, calls `wick_contract()`, builds diagrams, and returns a `PerturbativeResult`.
6. `simplify.py`: Multi-pass simplification pipeline (flatten, absorb rationals, eliminate zeros, canonical ordering, collect like terms).
7. `expressions.py`: Custom symbolic expression tree using frozen dataclasses and `fractions.Fraction`. All expression types are immutable and hashable. No SymPy dependency.
8. `diagrams.py` + `drawing.py`: networkx `MultiGraph` representation of Feynman diagrams; matplotlib-based rendering.

### Key design choices

- All expression types (`Rational`, `Propagator`, `Symbol`, `Sum`, `Product`, etc.) are frozen dataclasses — immutable and hashable, safe for use as dict keys or in sets.
- `FieldOperator` UIDs are globally unique integers that distinguish copies of the same field (e.g., two φ(x) operators in a vertex expansion).
- `IndexContext` generates fresh index names to prevent collisions when instantiating multiple copies of the same vertex.
- `reset_uid_counter()` is exposed for test reproducibility.
