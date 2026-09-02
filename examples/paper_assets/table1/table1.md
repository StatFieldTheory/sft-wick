| Order n | Observable | Operators 3n+m | Raw pairings (3n+m−1)!! | Distinct diagrams | Wall-clock (s) | Reduction |
|---|---|---|---|---|---|---|
| 1 | $\langle\varphi_a(x)\varphi_b(y)\varphi_c(z)\rangle$ | 6 | 15 | 6 | 3.12e-04 | 2 |
| 2 | $\langle\varphi_a(x)\varphi_b(y)\rangle$ | 8 | 105 | 6 | 0.00143 | 18 |
| 3 | $\langle\varphi_a(x)\varphi_b(y)\varphi_c(z)\rangle$ | 12 | 10395 | 80 | 0.118 | 130 |

Machine: arm64 Darwin, Python 3.14.3; best of three runs, single core.

The diagram counts and reduction factors are exact and machine independent.  The wall-clock column is not: it is sensitive to load on the measuring machine (a factor ~2.7 has been observed between an idle and a busy run of this same script), so treat it as an order of magnitude unless it was measured on an idle machine.
