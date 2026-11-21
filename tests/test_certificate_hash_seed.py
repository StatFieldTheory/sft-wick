"""The entire certificate, including value diagnostics, is seed-stable."""

import json
import os
from pathlib import Path
import subprocess
import sys


_PROBE = """
from dataclasses import asdict
import json

from sft_wick import Action, Field, Vertex, compute_moment, compute_moment_numerical

phi = Field('phi', 'physical', 2)
psi = Field('psi', 'response', 2)
action = Action([Vertex([psi, phi, phi], name, local=True) for name in ('F', 'G')])
result = {}
for target in ('moment', 'connected'):
    obs = [phi('a', 'x'), phi('b', 'y')]
    result[target] = [
        asdict(compute_moment(obs, action, 2, target=target, certify=True).certificate),
        asdict(compute_moment_numerical(obs, action, 2, target=target,
                                      canonical='builtin', certify=True).certificate),
    ]
print(json.dumps(result, sort_keys=True, allow_nan=False))
"""


def test_certificates_are_identical_under_five_python_hash_seeds():
    root = Path(__file__).resolve().parents[1]
    results = []
    for seed in ("0", "1", "7", "42", "113"):
        env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONHASHSEED=seed)
        result = subprocess.run([sys.executable, "-c", _PROBE], cwd=root, env=env,
                                check=True, capture_output=True, timeout=30)
        results.append(result.stdout)
    assert results == [results[0]] * len(results)

    for certificates in json.loads(results[0]).values():
        for cert in certificates:
            assert any(row["value_checks"] for order in cert["orders"]
                       if order["order"] == 2 for row in order["sequences"])
