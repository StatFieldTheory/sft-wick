# Order-1 diagrams of $\langle\varphi_a(x)\varphi_b(x)\varphi_c(y)\rangle$ (cubic vertex, N = 3)

`compute_moment(obs, action, order=1)` returns exactly four `DiagramTerm`s. Wick's theorem produces 6 pairings; the four distinct topologies carry pairing multiplicities [1, 2, 1, 2], already folded into each term's rational prefactor.  Each line is the term's full LaTeX: coefficient (rational prefactor times the MSR phase), coupling sum, propagators, integrals and index sums.

1. multiplicity 1, propagators `R_{ai_{2}}(x, y_{0}) C_{bc}(x, y) C_{i_{0}i_{1}}(y_{0}, y_{0})`  
   $$ \sum_{i_{0}=1}^{3} \sum_{i_{1}=1}^{3} \sum_{i_{2}=1}^{3} \int \mathrm{d}y_{0}\, \mathrm{i} (F_{i_{0}i_{1}i_{2}}) R_{ai_{2}}(x, y_{0}) C_{bc}(x, y) C_{i_{0}i_{1}}(y_{0}, y_{0}) $$

2. multiplicity 2, propagators `R_{ai_{2}}(x, y_{0}) C_{bi_{0}}(x, y_{0}) C_{ci_{1}}(y, y_{0})`  
   $$ \sum_{i_{0}=1}^{3} \sum_{i_{1}=1}^{3} \sum_{i_{2}=1}^{3} \int \mathrm{d}y_{0}\, \mathrm{i} (F_{i_{0}i_{1}i_{2}} + F_{i_{1}i_{0}i_{2}}) R_{ai_{2}}(x, y_{0}) C_{bi_{0}}(x, y_{0}) C_{ci_{1}}(y, y_{0}) $$

3. multiplicity 1, propagators `R_{ci_{2}}(y, y_{0}) C_{ab}(x, x) C_{i_{0}i_{1}}(y_{0}, y_{0})`  
   $$ \sum_{i_{0}=1}^{3} \sum_{i_{1}=1}^{3} \sum_{i_{2}=1}^{3} \int \mathrm{d}y_{0}\, \mathrm{i} (F_{i_{0}i_{1}i_{2}}) R_{ci_{2}}(y, y_{0}) C_{ab}(x, x) C_{i_{0}i_{1}}(y_{0}, y_{0}) $$

4. multiplicity 2, propagators `R_{ci_{2}}(y, y_{0}) C_{ai_{0}}(x, y_{0}) C_{bi_{1}}(x, y_{0})`  
   $$ \sum_{i_{0}=1}^{3} \sum_{i_{1}=1}^{3} \sum_{i_{2}=1}^{3} \int \mathrm{d}y_{0}\, \mathrm{i} (F_{i_{0}i_{1}i_{2}} + F_{i_{1}i_{0}i_{2}}) R_{ci_{2}}(y, y_{0}) C_{ai_{0}}(x, y_{0}) C_{bi_{1}}(x, y_{0}) $$

