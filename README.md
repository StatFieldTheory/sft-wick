# sft-wick

Wick's theorem contractions for statistical field theory perturbative calculations.

`sft-wick` automates the computation of perturbative expansions in the path integral formalism for stochastic differential equations. Given an observable and an interaction action, it applies Wick's theorem to express arbitrary field moments in terms of two-point propagators — the correlation function C and the response function R.

## Installation

```bash
pip install -e ".[dev]"
```

Dependencies: `networkx`, `matplotlib`. For parallel diagram evaluation: `pip install -e ".[parallel]"` (adds `joblib`). For development: `pytest`, `pytest-cov`.

## Quick Start

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

## Testing

```bash
pytest tests/ -v
```

114 tests covering expressions, fields, propagators, Wick contractions (operator-level and spatial-level), perturbative expansion, simplification, diagram-based collection, Itô prescription, causal R-loop elimination, and response phase convention.
