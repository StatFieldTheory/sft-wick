# Historical release drift corpora

These cases compare local release artifacts with the same input and runtime.
Run the commands from the repository root using an environment with the
current `sft-wick compare-versions` command.

## Pinned external time, v0.6.1 to v0.7.0

Use the released v0.6.1 wheel and a source archive of v0.7.0. The expected
change is the finite-node Gauss–Legendre result for the interior pinned time
`t_y = 0.65`. The analytic result is `0.43875`. The Gauss–Legendre endpoint,
QMC, and adaptive-quadrature cases provide controls.

```sh
drift_run=$(mktemp -d)
mkdir "$drift_run/v070"
git archive v0.7.0 pyproject.toml src | tar -x -C "$drift_run/v070"
sft-wick compare-versions tools/drift_corpus/v061-v070.yaml \
  --baseline /path/to/sft_wick-0.6.1-py3-none-any.whl \
  --candidate "$drift_run/v070" \
  --declared tools/drift_corpus/v061-v070-declared.yaml \
  --json "$drift_run/v061-v070.json"
```

Compare this corpus against a source archive of `main` without the historical
declaration for a separate current-release check.

## Selected demo channels, v0.5.0 to v0.6.0

The generator retrieves the original YAML and auxiliary modules from the
v0.5.0 tag. It selects a small set of demo 1–3 cells, an L1 demo 4
exponential-pulse FF cell, and a two-component demo 5 white-plus-OU mixing
cell. The declarations cover the expected Gauss–Legendre kink-split changes
and the QMC redraw following the Sobol precision change. Compare these
selected cells individually. The analytic or zero-integration cells and
channels without an active kink can remain unchanged.

```sh
python tools/drift_corpus/prepare_history.py "$drift_run/demo-corpus"
mkdir "$drift_run/v050" "$drift_run/v060"
git archive v0.5.0 pyproject.toml src | tar -x -C "$drift_run/v050"
git archive v0.6.0 pyproject.toml src | tar -x -C "$drift_run/v060"
sft-wick compare-versions "$drift_run/demo-corpus/v050-v060.yaml" \
  --baseline "$drift_run/v050" --candidate "$drift_run/v060" \
  --declared "$drift_run/demo-corpus/v050-v060-declared.yaml" \
  --json "$drift_run/v050-v060.json" --timeout 450
```
