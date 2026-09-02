"""The validation catalogue (docs/verification/catalog.rst) must match the
test suite as it is: every test file has a FILE_META entry and the
collected counts are the committed ones."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "gen_test_catalog", ROOT / "tools" / "gen_test_catalog.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gen_test_catalog"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_catalog_is_current():
    tool = _load_tool()
    expected = tool.render(tool.collect_counts())
    committed = tool.OUT.read_text()
    assert committed == expected, (
        "docs/verification/catalog.rst is stale -- run "
        "`python tools/gen_test_catalog.py`"
    )
