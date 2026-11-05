# Diagram sources

* `level_b_FK3_*.tex`: the 2 order-2 F.kappa^(3) diagrams that
  give the leading non-Gaussian signal in xi_01.
* `level_a_kappa3_*.tex`: the level-A diagram (1 rendered).
  The level-A numbers agree with the closed form to 1e-16.

Each has a `_standalone` twin that compiles on its own.
This file and the sources are written by `make_figures.py`.

* `level_a_kappa3_1.tex  [K3]  \int \mathrm{d}y_{0}\, \int \mathrm{d}y_{1}\, \int \mathrm{d}y_{2}\, -\mathrm{i} (K3_{abc}(y_{0}, y_{1}, y_{2}) + K3_{acb}(y_{0}, y_{2}, y_{1}) + K3_{bac}(y_{1}, y_{0}, y_{2}) + K3_{bca}(y_{1}, y_{2}, y_{0}) + K3_{cab}(y_{2}, y_{0}, y_{1}) + K3_{cba}(y_{2}, y_{1}, y_{0})) R(x, y_{0}) R(y, y_{1}) R(z, y_{2})`
* `level_b_FK3_1.tex  [FK3]  \sum_{i_{0}=1}^{2} \sum_{i_{1}=1}^{2} \int \mathrm{d}y_{0}\, \int \mathrm{d}y_{1}\, \int \mathrm{d}y_{2}\, \int \mathrm{d}y_{3}\, (F_{ai_{0}i_{1}} K3_{bi_{0}i_{1}}(y_{1}, y_{2}, y_{3}) + F_{ai_{1}i_{0}} K3_{bi_{0}i_{1}}(y_{1}, y_{2}, y_{3}) + F_{ai_{0}i_{1}} K3_{i_{0}bi_{1}}(y_{2}, y_{1}, y_{3}) + F_{ai_{1}i_{0}} K3_{i_{0}bi_{1}}(y_{2}, y_{1}, y_{3}) + F_{ai_{0}i_{1}} K3_{i_{0}i_{1}b}(y_{2}, y_{3}, y_{1}) + F_{ai_{1}i_{0}} K3_{i_{0}i_{1}b}(y_{2}, y_{3}, y_{1})) R(x, y_{0}) R(y, y_{1}) R(y_{0}, y_{2}) R(y_{0}, y_{3})`
* `level_b_FK3_2.tex  [FK3]  \sum_{i_{0}=1}^{2} \sum_{i_{1}=1}^{2} \int \mathrm{d}y_{0}\, \int \mathrm{d}y_{1}\, \int \mathrm{d}y_{2}\, \int \mathrm{d}y_{3}\, (F_{bi_{0}i_{1}} K3_{ai_{0}i_{1}}(y_{1}, y_{2}, y_{3}) + F_{bi_{1}i_{0}} K3_{ai_{0}i_{1}}(y_{1}, y_{2}, y_{3}) + F_{bi_{0}i_{1}} K3_{i_{0}ai_{1}}(y_{2}, y_{1}, y_{3}) + F_{bi_{1}i_{0}} K3_{i_{0}ai_{1}}(y_{2}, y_{1}, y_{3}) + F_{bi_{0}i_{1}} K3_{i_{0}i_{1}a}(y_{2}, y_{3}, y_{1}) + F_{bi_{1}i_{0}} K3_{i_{0}i_{1}a}(y_{2}, y_{3}, y_{1})) R(y, y_{0}) R(x, y_{1}) R(y_{0}, y_{2}) R(y_{0}, y_{3})`
