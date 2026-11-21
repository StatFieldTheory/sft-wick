"""Exact unfiltered counts precede the dry run's ordinary cost estimate."""

from pathlib import Path

import pytest
import yaml

from sft_wick import Action, Field, Vertex, perturbation
from sft_wick._routing_count import _preflight_counts
from sft_wick.workflow import System
from sft_wick.workflow.cli import main


QUICKSTART = Path(__file__).resolve().parents[1] / "examples" / "quickstart.yaml"


@pytest.mark.parametrize("target, pairings, classes", [
    ("moment", 15_120, 64),
    ("connected", 14_400, 56),
])
def test_dry_run_prints_cubic_counts_before_its_only_expansion(
        tmp_path, capsys, monkeypatch, target, pairings, classes):
    doc = yaml.safe_load(QUICKSTART.read_text())
    doc["expand"].update(orders=[0, 2, 4], target=target)
    doc["sweep"].update(orders=[0, 2], n_samples=128, t_final_grid=[1.0])
    config = tmp_path / "cubic.yaml"
    config.write_text(yaml.safe_dump(doc))

    expand = System.expand
    route = perturbation._enumerate_component_routings
    expansion_started = False

    def checked_expand(system, **kwargs):
        nonlocal expansion_started
        assert not expansion_started
        assert not kwargs.get("certify", False)
        before = capsys.readouterr().out
        assert f"expansion preflight ({target}, before sweep filters)" in before
        assert "order 0: pairings=1, weighted pairings=1, typed classes=n/a" in before
        assert (f"order 4: pairings={pairings}, weighted pairings={pairings}, "
                f"typed classes={classes}") in before
        expansion_started = True
        return expand(system, **kwargs)

    def checked_route(*args, **kwargs):
        assert expansion_started, "preflight enumerated component routings"
        return route(*args, **kwargs)

    monkeypatch.setattr(System, "expand", checked_expand)
    monkeypatch.setattr(perturbation, "_enumerate_component_routings", checked_route)
    assert main(["run", str(config), "--dry-run", "-q"]) == 0
    assert expansion_started
    after = capsys.readouterr().out
    diagrams = next(line for line in after.splitlines() if "   diagrams:" in line)
    assert "order 0: 1" in diagrams and "order 2:" in diagrams
    assert "order 4" not in diagrams
    assert "ROUGH TOTAL:" in after


def test_preflight_distinguishes_pairing_totals_and_sequence_multiplicity():
    phi, psi = Field("phi", "physical"), Field("psi", "response")
    action = Action([Vertex([psi, phi, phi], name, local=True) for name in ("F", "G")])
    odd, mixed = _preflight_counts([phi("x"), phi("y")], action, [1, 2], "moment")

    assert odd.pairings == odd.weighted_pairings == 0
    assert [row.classes for row in odd.sequences] == [0, 0]
    assert [(row.vertices, row.counter, row.multinomial, row.classes)
            for row in mixed.sequences] == [
        (("F", "F"), 30, 1, 6),
        (("F", "G"), 30, 2, 12),
        (("G", "G"), 30, 1, 6),
    ]
    assert mixed.pairings == 90
    assert mixed.weighted_pairings == 120
