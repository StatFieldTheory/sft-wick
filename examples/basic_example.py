"""Basic examples of using sft-wick for Wick contraction calculations.

Demonstrates:
1. Scalar field Wick contraction (zeroth order)
2. Multi-component field Wick contraction
3. Perturbative expansion with interaction vertices
4. Feynman diagram generation
"""

from sft_wick import (
    Field,
    Vertex,
    Action,
    compute_moment,
    reset_uid_counter,
    LaTeXFormatter,
)

# ============================================================
# Example 1: Scalar fields, zeroth order
# <psi(x) phi(x) phi(x) phi(x)>_{S_0} = 3 R(x,x) C(x,x)
# ============================================================
print("=" * 60)
print("Example 1: <psi(x) phi(x) phi(x) phi(x)>_{S_0}")
print("=" * 60)

reset_uid_counter()

phi = Field("phi", "physical")
psi = Field("psi", "response")

# Define a dummy action (not used at order 0)
v = Vertex(fields=[phi, phi, psi], coupling="F")
action = Action(vertices=[v])

obs = [psi("x"), phi("x"), phi("x"), phi("x")]
result = compute_moment(obs, action, order=0)

print(f"Result: {result.order(0).to_latex()}")
print()

# ============================================================
# Example 2: Multi-component fields, zeroth order
# <phi_a(x) phi_b(y) phi_c(z) phi_d(w)>_{S_0}
# = C_{ab}(x,y) C_{cd}(z,w) + C_{ac}(x,z) C_{bd}(y,w) + C_{ad}(x,w) C_{bc}(y,z)
# ============================================================
print("=" * 60)
print("Example 2: <phi_a(x) phi_b(y) phi_c(z) phi_d(w)>_{S_0}")
print("=" * 60)

reset_uid_counter()

phi = Field("phi", "physical", n_components=3)
psi = Field("psi", "response", n_components=3)

obs = [phi("a", "x"), phi("b", "y"), phi("c", "z"), phi("d", "w")]
result = compute_moment(obs, Action(vertices=[]), order=0)

print(f"Result: {result.order(0).to_latex()}")
print()

# ============================================================
# Example 3: First-order perturbation with phi-psi vertex
# S_int = int g phi(z) psi(z) dz
# <phi(x) phi(y)>_S up to order 1
# ============================================================
print("=" * 60)
print("Example 3: First-order perturbation")
print("S_int = int g phi(z) psi(z) dz")
print("<phi(x) phi(y)>_S up to order 1")
print("=" * 60)

reset_uid_counter()

phi = Field("phi", "physical")
psi = Field("psi", "response")

v = Vertex(fields=[phi, psi], coupling="g")
action = Action(vertices=[v])

obs = [phi("x"), phi("y")]
result = compute_moment(obs, action, order=1)

print(f"Order 0: {result.order(0).to_latex()}")
print(f"Order 1: {result.order(1).to_latex()}")
print()

# Diagram info
for order_n, diagrams in result.diagrams_by_order.items():
    for d in diagrams:
        fd = d.to_feynman_diagram()
        print(f"  Order {order_n}: {fd.summary()}")
print()

# ============================================================
# Example 4: Multi-component with cubic vertex
# S_int = int F_{ijk} phi_i(z) phi_j(z) psi_k(z) dz
# <psi_a(x) phi_b(x) phi_c(x) phi_d(x)>_S at order 0
# ============================================================
print("=" * 60)
print("Example 4: Multi-component cubic vertex")
print("S_int = int F_{ijk} phi_i phi_j psi_k dz")
print("<psi_a(x) phi_b(x) phi_c(x) phi_d(x)>_S at order 0")
print("=" * 60)

reset_uid_counter()

phi = Field("phi", "physical", n_components=3)
psi = Field("psi", "response", n_components=3)

v = Vertex(fields=[phi, phi, psi], coupling="F")
action = Action(vertices=[v])

obs = [psi("a", "x"), phi("b", "x"), phi("c", "x"), phi("d", "x")]
result = compute_moment(obs, action, order=0)

print(f"Order 0: {result.order(0).to_latex()}")
print()

# ============================================================
# Example 5: Non-local vertex
# S_int = iint K_{ij}(z, z') psi_i(z) psi_j(z') dz dz'
# ============================================================
print("=" * 60)
print("Example 5: Non-local vertex")
print("S_int = iint K_{ij}(z,z') psi_i(z) psi_j(z') dz dz'")
print("=" * 60)

reset_uid_counter()

phi = Field("phi", "physical", n_components=2)
psi = Field("psi", "response", n_components=2)

v_local = Vertex(fields=[phi, psi], coupling="g")
v_nonlocal = Vertex(fields=[psi, psi], coupling="K", local=False)
action = Action(vertices=[v_local, v_nonlocal])

obs = [phi("a", "x"), phi("b", "y")]
result = compute_moment(obs, action, order=1)

print(f"Order 0: {result.order(0).to_latex()}")
print(f"Order 1: {result.order(1).to_latex()}")
print()

# ============================================================
# Example 6: LaTeX formatter with custom propagator names
# ============================================================
print("=" * 60)
print("Example 6: Custom LaTeX formatting")
print("=" * 60)

reset_uid_counter()

phi = Field("phi", "physical")
psi = Field("psi", "response")

obs = [phi("x"), phi("y")]
result = compute_moment(obs, Action(vertices=[]), order=0)

formatter = LaTeXFormatter(propagator_names={"C": "G", "R": r"R^{\mathrm{ret}}"})
print(f"Default: {result.order(0).to_latex()}")
print(f"Custom:  {formatter.format(result.order(0))}")
