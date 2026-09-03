# Order-1 diagrams of $\langle\varphi_a(x)\varphi_b(y)\varphi_c(z)\rangle$ (cubic vertex, N = 3)

`compute_moment(obs, action, order=1)` returns 6 `DiagramTerm`s. Wick's theorem pairs the 6 operators in 15 ways, of which 6 pair the vertex's psi with one of its own phi's and vanish under the Ito prescription (R(x, x) = 0), leaving 9; the topology engine groups those into the 6 distinct topologies here, with pairing multiplicities [2, 1, 2, 1, 1, 2] summing to 9, already folded into each term's rational prefactor.  (<psi psi> = 0 removes nothing at order 1 -- there is only one psi in the expression; it starts pruning at order 2.)  Each line is the term's full LaTeX: coefficient (rational prefactor times the MSR phase), coupling sum, propagators, integrals and index sums.

1. multiplicity 2, propagators `R_{ai_{2}}(x, y_{0}) C_{bi_{0}}(y, y_{0}) C_{i_{1}c}(y_{0}, z)`  
   $$ \sum_{i_{0}=1}^{3} \sum_{i_{1}=1}^{3} \sum_{i_{2}=1}^{3} \int \mathrm{d}y_{0}\, \mathrm{i} (F_{i_{0}i_{1}i_{2}} + F_{i_{1}i_{0}i_{2}}) R_{ai_{2}}(x, y_{0}) C_{bi_{0}}(y, y_{0}) C_{i_{1}c}(y_{0}, z) $$

2. multiplicity 1, propagators `R_{ai_{2}}(x, y_{0}) C_{bc}(y, z) C_{i_{0}i_{1}}(y_{0}, y_{0})`  
   $$ \sum_{i_{0}=1}^{3} \sum_{i_{1}=1}^{3} \sum_{i_{2}=1}^{3} \int \mathrm{d}y_{0}\, \mathrm{i} (F_{i_{0}i_{1}i_{2}}) R_{ai_{2}}(x, y_{0}) C_{bc}(y, z) C_{i_{0}i_{1}}(y_{0}, y_{0}) $$

3. multiplicity 2, propagators `R_{bi_{2}}(y, y_{0}) C_{ai_{0}}(x, y_{0}) C_{i_{1}c}(y_{0}, z)`  
   $$ \sum_{i_{0}=1}^{3} \sum_{i_{1}=1}^{3} \sum_{i_{2}=1}^{3} \int \mathrm{d}y_{0}\, \mathrm{i} (F_{i_{0}i_{1}i_{2}} + F_{i_{1}i_{0}i_{2}}) R_{bi_{2}}(y, y_{0}) C_{ai_{0}}(x, y_{0}) C_{i_{1}c}(y_{0}, z) $$

4. multiplicity 1, propagators `R_{bi_{2}}(y, y_{0}) C_{ac}(x, z) C_{i_{0}i_{1}}(y_{0}, y_{0})`  
   $$ \sum_{i_{0}=1}^{3} \sum_{i_{1}=1}^{3} \sum_{i_{2}=1}^{3} \int \mathrm{d}y_{0}\, \mathrm{i} (F_{i_{0}i_{1}i_{2}}) R_{bi_{2}}(y, y_{0}) C_{ac}(x, z) C_{i_{0}i_{1}}(y_{0}, y_{0}) $$

5. multiplicity 1, propagators `R_{ci_{2}}(z, y_{0}) C_{ab}(x, y) C_{i_{0}i_{1}}(y_{0}, y_{0})`  
   $$ \sum_{i_{0}=1}^{3} \sum_{i_{1}=1}^{3} \sum_{i_{2}=1}^{3} \int \mathrm{d}y_{0}\, \mathrm{i} (F_{i_{0}i_{1}i_{2}}) R_{ci_{2}}(z, y_{0}) C_{ab}(x, y) C_{i_{0}i_{1}}(y_{0}, y_{0}) $$

6. multiplicity 2, propagators `R_{ci_{2}}(z, y_{0}) C_{ai_{0}}(x, y_{0}) C_{bi_{1}}(y, y_{0})`  
   $$ \sum_{i_{0}=1}^{3} \sum_{i_{1}=1}^{3} \sum_{i_{2}=1}^{3} \int \mathrm{d}y_{0}\, \mathrm{i} (F_{i_{0}i_{1}i_{2}} + F_{i_{1}i_{0}i_{2}}) R_{ci_{2}}(z, y_{0}) C_{ai_{0}}(x, y_{0}) C_{bi_{1}}(y, y_{0}) $$

