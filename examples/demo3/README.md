# Demo 3 — filtered Poisson (shot) noise

A non-Gaussian validation example built so that **every** claim has an
independent reference, and so that the pieces that are usually estimated
are computed instead.

```
dφ_a/dt = −γ φ_a + s F_abc φ_b φ_c + η_a
η_a(x,t) = Σ_k h w(x−x_k) g(t−s_k) − ⟨·⟩,  events ~ Poisson(ν)
g(τ) = Θ(τ) e^{−τ/σ_t},   w(x) = e^{−|x|/σ_x}
```

Campbell's theorem gives every cumulant in closed form.  Three structural
advantages over a deformed-Gaussian field:

| | demo 3 |
|---|---|
| non-Gaussianity knob | one number, `n = ν σ_t σ_x`: skewness `0.6285/√n`, excess kurtosis `0.5/n`, with `κ₂` held fixed by compensating `h` |
| R-contracted vertex | **exact** — all cumulants factor through one source time, so the `m` leg integrals collapse to a 1-D integral with a `2^m` closed form (any `m`, no new ideas) |
| reference simulation | **event-exact** — no time stepping and no spatial discretisation at `F = 0`; the only error is Monte Carlo |

## Files

| file | what it is |
|---|---|
| `shot_noise.py` | cumulants `X_m`, `T_m`, `κ_m`; the R-contracted kernel `K_R = ν h^m X_m T̃_m`; the `NonLocalVertex` callables |
| `simulate.py` | event-exact sampling of `η` and of the free field `φ` |
| `simulate_b.py` | ETD integrator for the interacting case: linear part and noise exact, only the `F` term discretised |
| `system.py` | the `System` builders for both levels |
| `level_a.py` | level A (`F = 0`) — the exact test, plus the `1/√n` law |
| `level_b.py` | level B — `ξ₀₁ = Fκ³ + F³κ³ + F³κ⁵`, all three computed |
| `make_figures.py` | paper figures + TikZ diagram sources |
| `config_FK.yaml`, `config_F3K.yaml` | the same physics through the L2 CLI |
| `L2/` | closed-form `C` and `κ²` modules the YAML layer needs |
| `INTERPRETATION.md` | **read this** — validation ledger, error budget, what is *not* established |

## Reproducing

```bash
conda activate sft-wick
cd examples/demo3
python level_a.py          # ~3 min
python level_b.py          # ~25 min (first run also enumerates order 4, ~2.5 min)
python make_figures.py
sft-wick run config_FK.yaml
```

Add `--quick` to `level_a.py` / `level_b.py` for a fast pass.  Tests:

```bash
pytest tests/test_demo3_shot_noise.py tests/test_demo3_levels.py -q
```
