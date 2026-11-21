"""Build the v0.5.0 to v0.6.0 comparison inputs from the v0.5.0 tag."""

from __future__ import annotations

import argparse
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
METHODS = ("gauss_legendre", "qmc_vectorized")


def archived(path):
    return subprocess.check_output(["git", "show", f"v0.5.0:{path}"], cwd=REPO)


def write_yaml(output, name, data):
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def export_examples(output, demo, files):
    inputs = []
    for original, replacement in files.items():
        relative = Path(demo) / replacement
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archived(f"examples/{demo}/{original}"))
        inputs.append(relative.as_posix())
    return inputs


def sweep(config, *, times, n_gauss, components, orders, positions, vertex_types=None):
    config["expand"]["orders"] = orders
    config["expand"].pop("cache_path", None)
    config["propagators"].pop("cache_path", None)
    config["sweep"].update(
        t_final_grid=times, n_gauss=n_gauss, n_samples=32768, seed=42, n_jobs=1,
        component_pairs=components, orders=orders, positions_grid=positions,
    )
    if vertex_types is not None:
        config["sweep"]["vertex_types"] = vertex_types
    config["output"] = []
    return config


def yaml_cases(output, demo, config, *, inputs=(), channel=""):
    key = f"{demo}-{channel}" if channel else demo
    cases = []
    for method, tag in zip(METHODS, ("gl", "qmc")):
        data = deepcopy(config)
        data["sweep"]["method"] = method
        name = f"{demo}/{tag}{'-' + channel if channel else ''}.yaml"
        write_yaml(output, name, data)
        cases.append(dict(id=f"{key}-{tag}", config=name, inputs=list(inputs)))
    return cases


def demo4_cases(output):
    inputs = export_examples(
        output, "demo4", {"poisson_noise.py": "poisson_noise.py", "poisson_system.py": "poisson_system.py"}
    )
    (output / "demo4/run_ff.py").write_bytes((HERE / "demo4_ff.py").read_bytes())
    cases = []
    for method, tag in zip(METHODS, ("gl", "qmc")):
        name = f"demo4/{tag}.py"
        (output / name).write_text(
            f"from run_ff import run_for\n\n\ndef run():\n    return run_for({method!r})\n",
            encoding="utf-8",
        )
        cases.append(dict(id=f"demo4-{tag}", driver=name, method=method,
                          inputs=[*inputs, "demo4/run_ff.py"]))
    return cases


def demo5_config():
    # Use a discrete cross-component mixing cell with white and OU noise.
    model = yaml.safe_load((HERE / "brownian/gl.yaml").read_text())
    model["system"].update(
        t_min=0.5, field=dict(name="phi", n_components=2),
        linear=dict(type="diagonal", gamma=[1.0, 1.0]),
        vertices=[dict(name="F", coupling=[
            [[0.30, -0.25], [0.10, 0.20]], [[-0.15, 0.35], [0.40, -0.20]],
        ])],
        noise=dict(
            kappa2=dict(
                type="separable_translation",
                temporal=dict(type="exponential", lam=0.4, sigma_t=0.6),
                spatial=dict(type="gaussian", sigma_x=0.9),
            ),
            sigma2=dict(type="constant", amplitude=[[0.5, 0.2], [0.2, 0.35]]),
        ),
    )
    model["expand"].update(observable=["phi_a(x)", "phi_b(y)"], diag_C=False)
    model["propagators"].update(t_max=2.6, diag_C=False)
    model["sweep"].pop("external_times_grid")
    model["sweep"]["integrate_over"] = None
    return sweep(model, times=[2.1], n_gauss=12, components=[[0, 1]],
                 orders=[0, 2], positions={"x": [0.0], "y": [0.7]})


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    positions = {"x": [0.0], "y": [0.5]}
    demo1 = sweep(yaml.safe_load(archived("examples/demo1/L2/config.yaml")),
                  times=[1.0, 15.0], n_gauss=24, components=[[0, 0]],
                  orders=[0, 2, 4], positions=positions)
    demo2 = sweep(yaml.safe_load(archived("examples/demo2/L2/config_FF.yaml")),
                  times=[1.0, 15.0], n_gauss=24, components=[[0, 0]],
                  orders=[0, 2], positions=positions)
    demo2_inputs = export_examples(output, "demo2", {"L2/c_closed_form.py": "c_closed_form.py"})
    helpers = ("shot_noise.py", "L2/c_closed_form.py", "L2/kappa2_general.py")
    demo3_inputs = export_examples(output, "demo3", {path: path for path in helpers})
    demo3 = sweep(yaml.safe_load(archived("examples/demo3/config_FK.yaml")),
                  times=[1.0, 3.0], n_gauss=12, components=[[0, 1]],
                  orders=[2], positions=positions)
    demo3_ff = deepcopy(demo3)
    demo3_ff["system"]["nonlocal_vertices"] = []
    demo3_ff["sweep"].update(component_pairs=[[0, 0]], vertex_types=["F"])

    cases = yaml_cases(output, "demo1", demo1)
    cases += yaml_cases(output, "demo2", demo2, inputs=demo2_inputs)
    cases += yaml_cases(output, "demo3", demo3, inputs=demo3_inputs)
    cases += yaml_cases(output, "demo3", demo3_ff, inputs=demo3_inputs, channel="ff")
    cases += demo4_cases(output)
    cases += yaml_cases(output, "demo5", demo5_config())
    write_yaml(output, "v050-v060.yaml", dict(cases=cases))

    gl_reason = "The finite-node rule splits integration cells at a colored covariance kink."
    qmc_reason = "The release increases the Sobol sequence precision from 30 to 64 bits."
    changes = [
        dict(case=f"{name}-{method}", where={"order": order}, kind=kind, reason=reason)
        for method, names, kind, reason in (
            ("gl", ("demo1", "demo2", "demo3-ff"), "moved", gl_reason),
            ("qmc", ("demo1", "demo2", "demo3", "demo3-ff", "demo4", "demo5"), "redrawn", qmc_reason),
        )
        for name in names
        for order in ((2, 4) if name == "demo1" else (2,))
    ]
    write_yaml(output, "v050-v060-declared.yaml", dict(changes=changes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for the corpus")
    prepare(parser.parse_args().output)
