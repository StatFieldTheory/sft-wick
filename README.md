# sft-wick

[![PyPI](https://img.shields.io/pypi/v/sft-wick)](https://pypi.org/project/sft-wick/)
[![Python versions](https://img.shields.io/pypi/pyversions/sft-wick)](https://pypi.org/project/sft-wick/)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue)](https://sft-wick.readthedocs.io)
[![arXiv](https://img.shields.io/badge/arXiv-2606.19480-b31b1b)](https://arxiv.org/abs/2606.19480)
[![DOI](https://zenodo.org/badge/1162069108.svg)](https://doi.org/10.5281/zenodo.20776358)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)

**Feynman-diagram expansion and evaluation for stochastic field theories.**

> 📖 **Documentation (API reference · user guide · theory background): <https://sft-wick.readthedocs.io>**

`sft-wick` automates perturbative calculations for **stochastic (partial) differential equations** in the Martin–Siggia–Rose (MSR) response-field formalism. Starting from a Langevin-type field equation — a deterministic drift plus noise that may be non-Gaussian and spatially correlated — it builds the interaction action, applies **Wick's theorem** to expand arbitrary field moments order by order, and writes every term using just two two-point propagators: the correlation function *C* = ⟨φφ⟩ and the response (Green's) function *R* = ⟨φψ⟩. Diagrams are enumerated symbolically, rendered as Feynman graphs, and evaluated numerically — end to end, from a YAML config to theory-vs-simulation curves. For **self-consistent** (DMFT-style) problems, where the propagators define a self-energy that in turn defines the propagators, `solve_self_consistency` supplies the fixed-point iteration and its diagnostics.

## Installation

```bash
pip install sft-wick
```

For development (editable install with the test suite):

```bash
git clone https://github.com/StatFieldTheory/sft-wick.git
cd sft-wick
pip install -e ".[dev]"
```

Dependencies: `numpy`, `scipy`, `networkx`, `matplotlib`, `pandas`, `pyyaml`, `tabulate`, `joblib`. The `parallel` extra is kept for compatibility with older install commands. For development: `pytest`, `pytest-cov`.

The install also registers a CLI entry point:

```bash
sft-wick run config.yaml         # execute a full YAML-configured workflow
sft-wick run config.yaml --override sweep.seed=7 --dry-run
```

## Three-layer API — start with L2

`sft-wick` exposes three progressively higher-level entry points.
**The recommended entry point for any new analysis is L2: write a
YAML config, run it with the CLI, iterate.**  Drop down only when
the physics genuinely requires Python-level control.

| Layer | Entry point | Use when |
|---|---|---|
| **L2 — YAML + CLI** ✨ | `sft-wick run config.yaml` | **Default choice.** Reproducible, shareable, diff-able; no Python code at the call site; runs identically on a laptop or a cluster; `--override` lets you scan parameters from the shell. |
| **L1 — Python workflow** | `System`, `Expansion`, `Propagators`, `SweepResult` | You need to script custom pre/post-processing around the sweep, or compose multiple systems programmatically. |
| **L0 — raw symbolic** | `compute_moment`, `Field`, `Vertex`, `Action`, `PropagatorCache`, `DiagramIntegrand` | You need fine-grained control over pairings, Itô flags, canonical forms, or custom simplifications. Research into the symbolic machinery. |

The Sphinx docs' "Workflow API" chapter (`docs/user_guide/workflow.rst`) covers L1/L2 end-to-end; the L0 reference below is the complete specification of the underlying symbolic machinery.

## Quick Start — L2 (config file)

Write `demo1_config.yaml`:

```yaml
system:
  field: {name: phi, n_components: 2}
  linear: {type: diagonal, gamma: [1.0, 1.0]}
  vertices:
    - name: F
      coupling:                                    # bare F; MSR factor
        - [[0.0, 0.0], [0.0, 1.0]]                 # applied automatically
        - [[0.0, 0.5], [0.5, 0.0]]
  noise:
    kappa2:
      type: separable_translation
      temporal: {type: exponential, lam: 0.05, sigma_t: 0.3}
      spatial:  {type: exponential, sigma_x: 1.0}

expand:
  observable: ["phi_a(x)", "phi_b(y)"]
  orders: [0, 2, 4]

propagators: {t_max: 15.0, n_grid_t: 60}

sweep:
  positions_grid: {x: [0.0], y: [0.0, 0.5, 1.0, 2.5]}
  t_final_grid: [1.0, 15.0]
  component_pairs: [[0, 0], [1, 1]]
  n_samples: 8192
  seed: 42

output:
  - {type: table, format: markdown, path: results.md}
  - {type: npz, path: results.npz}
```

Run from the shell:

```bash
sft-wick run demo1_config.yaml                # full pipeline
sft-wick run demo1_config.yaml --override sweep.seed=7
sft-wick run demo1_config.yaml --dry-run      # validate + summarize
```

See `examples/demo1_config.yaml` and `examples/demo2_config.yaml` for
richer examples (non-local vertex, closed-form C hook, dynamic
couplings).

## Quick Start — L1 (Python, for programmatic use)

The exact same workflow above, written directly in Python when you
need to integrate it into a larger script:

```python
import numpy as np
import sft_wick as sw

F = np.zeros((2, 2, 2))
F[0, 1, 1] = 1.0
F[1, 0, 1] = F[1, 1, 0] = 0.5

system = sw.System(
    field=sw.FieldSpec("phi", n_components=2),
    linear=sw.DiagonalA(gamma=[1.0, 1.0]),
    vertices=[sw.LocalVertex("F", coupling=F)],   # bare F
    noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
        temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
        spatial=sw.ExponentialSpatial(sigma_x=1.0),
    )),
)

expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2, 4])
props = system.propagators(t_max=15.0, n_grid_t=60)

sweep = expansion.sweep(
    props,
    positions_grid={"x": [0.0], "y": [0.0, 0.5, 1.0, 2.5]},
    t_final_grid=[1.0, 15.0],
    component_pairs=[(0, 0), (1, 1)],
)

print(sweep.totals())    # long-format pandas DataFrame
```

## Raw API (L0) Quick Start

```python
from sft_wick import Field, Vertex, Action, compute_moment

# Define scalar fields
phi = Field('phi', 'physical')
psi = Field('psi', 'response')

# Compute <psi(x) phi(x) phi(x) phi(x)>_{S_0}
obs = [psi('x'), phi('x'), phi('x'), phi('x')]
result = compute_moment(obs, Action(vertices=[]), order=0)
print(result.order(0).to_latex())
# Output: 3 R(x, x) C(x, x)
```

## Self-consistent solutions

Some problems are a **fixed point**: propagators define a self-energy, the
self-energy defines new propagators, repeat. `solve_self_consistency` runs
that loop, mixes, and reports what actually happened.

```python
from sft_wick import solve_self_consistency

def step(state):
    sigma = self_energy_from_diagrams(state)   # sft-wick computes this
    return dyson_solve(sigma)                  # you supply this

result = solve_self_consistency(initial, step, tol=1e-8, damping=0.3)
if not result:                       # bool(result) is result.converged
    raise RuntimeError(result.summary())
R, C = result.state
```

The **Dyson solve is deliberately not provided**: it is model-specific and is
genuinely an integral-equation solve rather than a diagram evaluation, so a
wrong general one would be worse than none.

The result is never a bare state, because a non-converged iteration looks
exactly like a converged one if you only print the last state. It carries
`converged`, the full residual history, and a `reason` — `converged`,
`diverged`, `oscillating` (use damping) or `max_iter`. See
[the API page](https://sft-wick.readthedocs.io/en/latest/api/selfconsistency.html)
for the four specific ways a fixed-point loop can report a solution it never
found, and what this one does about each.

## Background

In the MSR formalism for SDEs, path-integral averages involve two types of fields:

- **Physical field** φ (phi): the field of interest
- **Response field** ψ (psi): the auxiliary conjugate field

The free two-point functions (propagators) are:

| Contraction | Propagator | Meaning |
|---|---|---|
| ⟨φ\_i(x) φ\_j(x')⟩\_{S₀} | C\_{ij}(x, x') | Correlation function |
| ⟨φ\_i(x) ψ\_j(x')⟩\_{S₀} | R\_{ij}(x, x') | Response (Green's) function |
| ⟨ψ\_i(x) ψ\_j(x')⟩\_{S₀} | 0 | Vanishes by construction |

Since the MSR partition function Z = 1, the perturbative expansion is simply:

```
⟨O⟩_S = Σ_{n=0}^{N} (-1)^n / n! ⟨O S_int^n⟩_{S_0}
```

Each term is evaluated via **Wick's theorem**: the expectation of a product of fields equals the sum over all complete pairings of the product of two-point functions.

## Usage Guide

### 1. Defining Fields

```python
from sft_wick import Field

# Scalar fields (single component)
phi = Field('phi', 'physical')
psi = Field('psi', 'response')

# Multi-component fields
phi = Field('phi', 'physical', n_components=3)
psi = Field('psi', 'response', n_components=3)
```

### 2. Creating Field Operators

Field operators are concrete instances with bound component indices and spatial arguments.

```python
# Scalar: phi(spatial_arg)
op = phi('x')          # φ(x)

# Multi-component: phi(component_index, spatial_arg)
op = phi('a', 'x')     # φ_a(x)
op = psi('b', 'y')     # ψ_b(y)
```

### 3. Defining Interaction Vertices

Vertices represent terms in the interaction action S\_int.

```python
from sft_wick import Vertex

# Local vertex: ∫ F_{ijk} φ_i(x) φ_j(x) ψ_k(x) dx
# All fields share the same spatial argument.
v1 = Vertex(fields=[phi, phi, psi], coupling='F')

# Non-local vertex: ∬ K_{ij}(x, x') ψ_i(x) ψ_j(x') dx dx'
# Each field gets its own spatial argument.
v2 = Vertex(fields=[psi, psi], coupling='K', local=False)
```

### 4. Computing Perturbative Expansions

```python
from sft_wick import Action, compute_moment

action = Action(vertices=[v1])
obs = [psi('a', 'x'), phi('b', 'x'), phi('c', 'x'), phi('d', 'x')]

result = compute_moment(obs, action, order=1)

# Access individual orders
print(result.order(0).to_latex())
print(result.order(1).to_latex())

# Full result
print(result.to_latex())
```

### 5. Feynman Diagrams

Each non-vanishing Wick contraction corresponds to a Feynman diagram:

- **Vertices** (■): interaction points from S\_int
- **External points** (●): observable field operators
- **C propagator** (blue solid line): correlation φ-φ
- **R propagator** (red dashed arrow): response φ-ψ

```python
# Draw all diagrams
result.draw_diagrams()

# Draw only diagrams at a specific order
result.draw_diagrams(order=1)

# Access diagram topology
for d_info in result.diagrams_by_order[1]:
    fd = d_info.to_feynman_diagram()
    print(fd.summary())
    print(f"  Loops: {fd.n_loops}, Connected: {fd.is_connected}")
```

### 6. LaTeX Formatting

```python
from sft_wick import LaTeXFormatter

# Default names
print(result.order(0).to_latex())
# C_{ab}(x, y)

# Custom propagator names
fmt = LaTeXFormatter(propagator_names={
    'C': 'G',
    'R': r'R^{\mathrm{ret}}'
})
print(fmt.format(result.order(0)))
# G_{ab}(x, y)

# LaTeX align environment for order-by-order display
print(fmt.format_aligned(result.order_terms))
```

### 7. Direct Wick Contraction

For low-level access without the perturbative machinery:

```python
from sft_wick import wick_contract, contract_pair

# Contract a product of fields
ops = [phi('a', 'x'), phi('b', 'y'), phi('c', 'z'), phi('d', 'w')]
expr, pairings = wick_contract(ops)
print(expr.to_latex())
# C_{ab}(x, y) C_{cd}(z, w) + C_{ac}(x, z) C_{bd}(y, w) + C_{ad}(x, w) C_{bc}(y, z)

# Contract a single pair
prop = contract_pair(phi('a', 'x'), psi('b', 'y'))
print(prop.to_latex())
# R_{ab}(x, y)
```

## Examples

### Zeroth-Order Moment

```python
phi = Field('phi', 'physical')
psi = Field('psi', 'response')

obs = [psi('x'), phi('x'), phi('x'), phi('x')]
result = compute_moment(obs, Action(vertices=[]), order=0)
print(result.order(0).to_latex())
# 3 R(x, x) C(x, x)
```

The three terms arise because ψ can pair with any of the three φ's (producing R), and the remaining two φ's pair together (producing C).

### First-Order Perturbation

```python
phi = Field('phi', 'physical')
psi = Field('psi', 'response')

v = Vertex(fields=[phi, psi], coupling='g')
action = Action(vertices=[v])

obs = [phi('x'), phi('y')]
result = compute_moment(obs, action, order=1)
print(result.order(1).to_latex())
# ∫ dy₀ (-g) [R(x, y₀) C(y, y₀) + R(y, y₀) C(x, y₀) + R(y₀, y₀) C(x, y)]
```

### Multi-Component Four-Point Function

```python
phi = Field('phi', 'physical', n_components=3)

obs = [phi('a', 'x'), phi('b', 'y'), phi('c', 'z'), phi('d', 'w')]
result = compute_moment(obs, Action(vertices=[]), order=0)
print(result.order(0).to_latex())
# C_{ab}(x, y) C_{cd}(z, w) + C_{ac}(x, z) C_{bd}(y, w) + C_{ad}(x, w) C_{bc}(y, z)
```

### Non-Local Interaction

```python
phi = Field('phi', 'physical', n_components=2)
psi = Field('psi', 'response', n_components=2)

v_nonlocal = Vertex(fields=[psi, psi], coupling='K', local=False)
action = Action(vertices=[v_nonlocal])

obs = [phi('a', 'x'), phi('b', 'y')]
result = compute_moment(obs, action, order=1)
```

## API Reference

### Core Functions

| Function | Description |
|---|---|
| `compute_moment(observable, action, order, ito=True, response_phase=True, collect_topology=True)` | Perturbative expansion of ⟨O⟩\_S up to given order |
| `compute_moment_numerical(observable, action, order, coupling_values, fixed_indices, ..., n_jobs=1)` | Fast numerical path using nauty canonical labeling for diagram grouping. Enables order-6 calculations. Requires `pynauty`. Parallelization uses `joblib` (`n_jobs=-1`). |
| `wick_contract(operators, ito=True)` | Apply Wick's theorem to a product of field operators |
| `contract_pair(op1, op2, ito=True)` | Contract two field operators into a propagator |
| `apply_response_phase(expr)` | Multiply each term by (−i)^n for n response propagators |
| `collect_by_diagram(expr)` | Group terms by Feynman diagram isomorphism, factor out propagators |
| `collect_by_topology(expr)` | Alias for `collect_by_diagram` (backward compat) |
| `integrate_moment(integrand, lambda_f, cache, ...)` | Integrate a single diagram's contribution (QMC or nquad) |
| `integrate_diagrams(diagram_terms, coupling_values, lambda_f, cache, ..., n_jobs=1)` | Batch-integrate a list of diagram terms, optionally in parallel (`n_jobs=-1`) |
| `simplify(expr)` | Simplify an expression (flatten, collect terms, eliminate zeros) |
| `reset_uid_counter()` | Reset field operator UID counter (for reproducible tests) |

### Classes

| Class | Description |
|---|---|
| `Field` | Field declaration (name, type, component count) |
| `FieldOperator` | Concrete field instance with bound index and position |
| `Vertex` | Interaction vertex template (local or non-local) |
| `VertexInstance` | Instantiated vertex with fresh internal indices |
| `Action` | Collection of vertices defining S\_int |
| `PerturbativeResult` | Result container with order-by-order expressions and diagrams |
| `FeynmanDiagram` | Graph representation of a diagram (networkx MultiGraph) |
| `DiagramRenderer` | Matplotlib-based diagram visualizer |
| `PropagatorModel` | User-provided R\_time and κ² callables for numerical evaluation |
| `PropagatorCache` | Caches C propagators (spline-interpolated or dblquad) |
| `DiagramIntegrand` | Ready-to-integrate object combining coupling coefficients and spatial structure |
| `LaTeXFormatter` | Configurable LaTeX output |
| `ImaginaryUnit` | The imaginary unit i, used in phase factors |

### Expression Types

| Type | Description | Example LaTeX |
|---|---|---|
| `Rational(num, den)` | Exact rational number | `\frac{1}{2}` |
| `Symbol(name, indices, spatial_args)` | Named tensor/coupling | `F_{ijk}` |
| `Propagator(kind, il, ir, sl, sr)` | Two-point function | `C_{ab}(x, y)` |
| `ImaginaryUnit()` | Imaginary unit | `\mathrm{i}` |
| `Sum(terms)` | Sum of expressions | `a + b + c` |
| `Product(factors)` | Product of expressions | `a b c` |
| `SumOverIndex(index, dim, body)` | Index summation | `\sum_{i=1}^{N} ...` |
| `IntegralOver(var, body)` | Spatial integration | `\int dx ...` |
| `KroneckerDelta(i, j)` | Component delta | `δ_{ij}` |
| `DiracDelta(x, y)` | Spatial delta | `δ(x - y)` |

## Conventions and Options

### Itô prescription (`ito=True`, default)

By default, the Itô discretisation convention Θ(0)=0 is applied:

- **Equal-point R vanishes**: R(x,x) = 0 — eliminates self-response contractions and intra-vertex tadpoles in local vertices.
- **Causal R-loops vanish**: Any closed loop of response propagators R(a,b)R(b,c)...R(z,a) = 0, since this would require a cyclic time ordering t\_a > t\_b > ... > t\_a, which is impossible for the retarded propagator.

Pass `ito=False` to keep these terms symbolic.

### Response phase convention (`response_phase=True`, default)

The MSR convention ⟨φ(a) ψ(b)⟩ = −i R(a,b) is implemented by multiplying each term by (−i)^n, where n is the number of response propagators R in that term. This is applied after simplification so that like-term collection is unaffected.

Pass `response_phase=False` to get raw R propagators without the phase factor.

### Diagram-based term collection (`collect_topology=True`, default)

Terms whose Feynman diagrams are isomorphic — under relabeling of dummy integration variables and accounting for C propagator symmetry C(x,y) = C(y,x) — are grouped together. The algorithm computes a canonical graph form for each term by trying all permutations of internal spatial variables. Propagators are factored out with canonical component indices, and coupling coefficients are summed with appropriately permuted indices to produce expressions like (F\_{ijk} + F\_{ikj}) R C.

At second order and above, spatial-variable relabeling (e.g. y\_0 ↔ y\_1 for two copies of the same vertex) merges additional equivalent pairings.

Pass `collect_topology=False` to keep all pairings expanded individually.

## Design Notes

- **No SymPy dependency**: Uses a custom lightweight expression tree with `fractions.Fraction` for exact rational arithmetic.
- **Frozen dataclasses**: All expression types are immutable and hashable, safe for use in sets and dicts.
- **Unique operator IDs**: Each `FieldOperator` carries a unique integer ID, so that two copies of φ\_a(x) in the same product are properly distinguished during contraction.
- **Optimized contraction**: Two engines are available. The operator-level engine (`generate_valid_pairings`) skips ψ-ψ pairings at construction time. The spatial-level engine (`wick_contract_spatial`, used by default when `collect_topology=True`) enumerates spatial topologies instead of operator-level pairings, computing a multiplicity for each — this avoids the combinatorial explosion from component-index routing and provides orders-of-magnitude speedup at high perturbative orders.
- **Feynman diagrams**: Built on `networkx.MultiGraph` (supporting multiple edges between the same pair of nodes) with `matplotlib` rendering.

## Performance

### `compute_moment` (symbolic path)

The default `compute_moment` builds full symbolic expressions and groups diagrams via brute-force canonical form search (trying all k! permutations of internal spatial variables). Performance at each perturbative order:

| Order | Operators | Topologies | Diagrams | Time  |
|-------|-----------|------------|----------|-------|
| 2     | 8         | 12         | 6        | <0.01s |
| 4     | 14        | 1,416      | 64       | ~1.5s |
| 6     | 20        | 738,900    | 1,088    | infeasible (hours/OOM) |

### `compute_moment_numerical` (nauty path)

Replaces the O(k!) canonical form search with the nauty graph isomorphism algorithm (via `pynauty`), reducing diagram grouping from hours to seconds at order 6:

| Order | Topologies | Nauty grouping | Component routing | Total  |
|-------|------------|----------------|-------------------|--------|
| 4     | 1,416      | 0.02s          | 0.25s             | ~0.3s  |
| 6     | 738,900    | ~12s           | ~8 min            | ~10 min |

Pass `n_jobs=-1` to parallelize across CPU cores (`joblib` is installed by default).

### Known bottlenecks and future directions

At order 6, the dominant cost is **component routing** (`_enumerate_component_routings`), called once per spatial topology (738K calls). Potential improvements:

- **Cache routing for isomorphic topologies**: topologies in the same nauty canonical group are graph-isomorphic. If the nauty permutation can be applied at the operator level, routing need only run once per canonical group (~1K calls instead of ~738K). This requires mapping operator UIDs across isomorphic graphs.
- **Einsum-based coupling evaluation**: for constant coupling tensors, the coupling sum can be computed via `np.einsum` tensor contraction rather than symbolic expression evaluation, eliminating the combinatorial component-index enumeration entirely.
- **Vectorized QMC integration** ✅: `DiagramIntegrand.integrate_moment_qmc_vectorized()` replaces the Python for-loop over Sobol samples with batch propagator evaluation via `PropagatorCache.C_diagonal_batch()` and `R_time_batch()`. Achieves 18–22× speedup over the scalar `integrate_moment_qmc()` with identical results.
- **GPU acceleration**: the simulation (Euler–Maruyama) and QMC integration are embarrassingly parallel across realizations/samples and would benefit from JAX `vmap` or similar frameworks.

## Testing

```bash
pytest tests/ -v
```

**460 tests** (a few minutes on a laptop).  The suite is organised into
eight deductive phases:

1. Phase 1 — Symbolic expansion (`test_deductive_expansion.py`)
2. Phase 2 — Propagator numerics (`test_deductive_numerics.py::TestClosedFormC` etc.)
3. Phase 3 — Full diagram evaluation
4. Phase 4 — Alternative-path consistency (vectorised, parallel, nauty)
5. Phase 5 — Spatial homogeneity modes (translation / rotation / general)
6. Phase 6 — White-noise component
7. Phase 7 — Dynamic non-local coupling + L1/L2 workflow round-trip
   (`test_workflow.py`, `test_workflow_config.py`)
8. Phase 8 — Time-dependent linear operator (`test_diagonal_A_time_dependent.py`)

See `docs/verification/index.rst` for the per-phase test matrix, tolerances, and design rationale.

## Repository layout

| Path | Contents |
|------|----------|
| `src/sft_wick/` | Package source: diagram enumeration, propagators, numerical evaluation, drawing, and the `workflow/` high-level API + CLI |
| `examples/` | Worked examples — `demo1/` (Gaussian noise), `demo2/` (non-Gaussian, non-zero κ³), and tutorial notebooks |
| `tests/` | pytest suite (eight deductive phases) |
| `docs/` | Sphinx documentation (ReadTheDocs source) |

## Worked examples (reproducible test runs)

Two end-to-end examples ship with committed inputs **and** outputs, each covering
symbolic diagram expansion *and* numerical evaluation against a direct Langevin
simulation:

```bash
# demo1 — Gaussian driving noise
python examples/demo1/run_simulation.py            # writes sim_cache.npz (~50k realisations)
# then run examples/demo1/analysis.ipynb           # diagrams + theory-vs-simulation figures

# demo2 — non-Gaussian noise (non-zero third cumulant kappa^3)
python examples/demo2/run_simulation.py --alpha 0.6   # non-Gaussian
python examples/demo2/run_simulation.py --alpha 0.0   # Gaussian control
# then run examples/demo2/analysis.ipynb           # kappa^3 cross-check + FK channel
```

The `sim_cache.npz` files and reference figures are committed, so a reviewer can
re-run the scripts and diff against the shipped outputs. See also
`examples/nonlocal_vertex_2pt.ipynb` for a non-local-vertex tutorial.

## Documentation

Full documentation (API reference, user guide, theory background) is hosted at
<https://sft-wick.readthedocs.io>.

## Citation

If you use `sft-wick`, please cite the paper:

```bibtex
@misc{zhang2026sftwickformalismpackagefeynmandiagram,
      title={sft-wick: A formalism and package for Feynman-diagram expansion and evaluation in stochastic field theories},
      author={Zheng Zhang},
      year={2026},
      eprint={2606.19480},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2606.19480},
}
```

A specific software version can additionally be referenced via its Zenodo archive ([DOI:10.5281/zenodo.20776358](https://doi.org/10.5281/zenodo.20776358)).

## License

`sft-wick` is released under the BSD 3-Clause License — see [LICENSE](LICENSE).
