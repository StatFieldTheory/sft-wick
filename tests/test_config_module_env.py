"""A module a config names changes nothing a later process reads.

Loading one used to put its directory at the front of ``PYTHONPATH``, so
every process started afterwards read that directory before the standard
library: a ``random.py`` beside the config replaced the real one there.
Workers never needed it.  joblib's loky backend hands each worker the
``sys.path`` of the parent.

ME1 -- the environment is the same after a load, one that succeeds or one
       that is refused, with ``PYTHONPATH`` set or unset.  Importing joblib
       comes first: its first import sets ``KMP_INIT_AT_FORK``.
ME2 -- a process started after the load imports the standard library's
       module, not the file beside the config.
ME3 -- a loky worker imports what the module imports from its directory, and
       reads the standard library first: ``random`` while it starts, which
       only a ``PYTHONPATH`` entry reaches, and ``colorsys`` later, which an
       entry at the front of ``sys.path`` reaches.  In a new interpreter,
       since a pool kept from an earlier test holds the path it started with.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import joblib  # noqa: F401  (its first import sets KMP_INIT_AT_FORK)
import pytest

from sft_wick.workflow.config import ConfigError, _load_callable_from_module

ROOT = Path(__file__).resolve().parents[1]
MODULE = "def fn(x):\n    return x\n"
STDLIB_NAME = "random.py"
SHADOW = "raise RuntimeError('the file beside the config')\n"


@pytest.fixture(autouse=True)
def path_as_it_was(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))


@pytest.mark.parametrize(
    "text, refused",
    [(MODULE, False), ("raise RuntimeError('at import')\n", True),
     ("other = 1\n", True)],
    ids=["loaded", "raises", "no_attribute"])
@pytest.mark.parametrize("before", [None, "", "/somewhere/else"],
                         ids=["unset", "empty", "set"])
def test_ME1_the_environment_is_the_same_after_a_load(tmp_path, monkeypatch, text,
                                                      refused, before):
    if before is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", before)
    environment = dict(os.environ)
    module = tmp_path / "env_probe_module.py"
    module.write_text(text)
    if refused:
        with pytest.raises(ConfigError):
            _load_callable_from_module(module, "fn")
    else:
        assert _load_callable_from_module(module, "fn")(3) == 3
    sys.modules.pop("env_probe_module", None)
    assert dict(os.environ) == environment


def test_ME1_the_module_still_imports_what_is_beside_it(tmp_path):
    (tmp_path / "env_probe_helper.py").write_text("VALUE = 7\n")
    module = tmp_path / "env_probe_user.py"
    module.write_text("import env_probe_helper\n"
                      "def fn():\n    return env_probe_helper.VALUE\n")
    assert _load_callable_from_module(module, "fn")() == 7
    sys.modules.pop("env_probe_user", None)
    sys.modules.pop("env_probe_helper", None)


def test_ME2_a_later_process_reads_the_standard_library(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    (tmp_path / STDLIB_NAME).write_text(SHADOW)
    module = tmp_path / "env_probe_later.py"
    module.write_text(MODULE)
    _load_callable_from_module(module, "fn")
    sys.modules.pop("env_probe_later", None)
    done = subprocess.run(
        [sys.executable, "-c", "import random; print(random.__file__)"],
        capture_output=True, text=True, cwd=tmp_path.parent)
    assert done.returncode == 0, done.stderr
    assert str(tmp_path) not in done.stdout


SCRIPT = textwrap.dedent("""
    import sys
    from pathlib import Path

    from joblib import Parallel, delayed

    from sft_wick.workflow.config import _load_callable_from_module


    def stdlib_module():
        import colorsys
        return colorsys.__file__


    folder = Path(sys.argv[1])
    fn = _load_callable_from_module(folder / "uses_helper.py", "fn")
    assert Parallel(n_jobs=2, backend="loky")(
        delayed(fn)(i) for i in range(4)) == [0, 7, 14, 21]
    found = Parallel(n_jobs=2, backend="loky")(
        delayed(stdlib_module)() for _ in range(2))
    assert not any(str(folder) in path for path in found), found
    print("workers imported the helper")
""")


def test_ME3_a_worker_imports_what_the_module_imports(tmp_path):
    folder = tmp_path / "config"
    folder.mkdir()
    (folder / "helper_beside_it.py").write_text(
        "def times_seven(x):\n    return 7 * x\n")
    (folder / "uses_helper.py").write_text(
        "import helper_beside_it\n"
        "def fn(x):\n    return helper_beside_it.times_seven(x)\n")
    (folder / STDLIB_NAME).write_text(SHADOW)
    (folder / "colorsys.py").write_text(SHADOW)
    # Python puts the folder of the script first, so it is another folder.
    script = tmp_path / "drive.py"
    script.write_text(SCRIPT)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    done = subprocess.run([sys.executable, str(script), str(folder)],
                          capture_output=True, text=True, env=env,
                          cwd=tmp_path, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "workers imported the helper" in done.stdout
