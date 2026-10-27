"""Fresh-process contracts for the deferred public Matplotlib exports."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

PREFIX = """
import importlib, inspect, pickle, sys
import sft_wick as sw
assert 'sft_wick.drawing' not in sys.modules
assert not any(n == 'matplotlib' or n.startswith('matplotlib.') for n in sys.modules)
"""


CASES = {
    "discovery": """
assert {'DiagramRenderer', 'drawing'} <= set(dir(sw))
assert sw.TikzRenderer.__module__ == 'sft_wick.drawing_tikz'
assert 'DiagramRenderer' not in vars(sw)
try:
    sw.nonexistent_public_export
except AttributeError as e:
    assert str(e) == "module 'sft_wick' has no attribute 'nonexistent_public_export'"
else:
    raise AssertionError('missing AttributeError')
assert 'matplotlib' not in sys.modules
""",
    "symbolic": """
phi = sw.Field('phi', 'physical', n_components=2)
result = sw.compute_moment([phi('a', 'x'), phi('b', 'y')], sw.Action([]), order=0)
assert result.to_latex()
assert 'matplotlib' not in sys.modules
""",
    "numeric": """
import numpy as np
kernel = sw.OUWhiteCumulant(np.ones((2, 2)), gamma=np.array([0.75, 1.25]))
assert kernel is not None
assert 'matplotlib' not in sys.modules
""",
    "class_first": """
cls = sw.DiagramRenderer
from sft_wick import DiagramRenderer
from sft_wick.drawing import DiagramRenderer as direct
assert cls is DiagramRenderer is direct is sw.drawing.DiagramRenderer
assert vars(sw)['DiagramRenderer'] is cls
""",
    "module_first": """
module = sw.drawing
assert module is importlib.import_module('sft_wick.drawing')
assert sw.DiagramRenderer is module.DiagramRenderer
""",
    "direct_first": """
from sft_wick.drawing import DiagramRenderer
assert sw.DiagramRenderer is DiagramRenderer
assert sw.drawing.DiagramRenderer is DiagramRenderer
""",
    "from_module": """
from sft_wick import drawing
assert drawing is sw.drawing
assert sw.DiagramRenderer is drawing.DiagramRenderer
""",
    "star": """
scope = {}
exec('from sft_wick import *', scope)
assert set(scope) - {'__builtins__'} == set(sw.__all__)
assert scope['DiagramRenderer'] is sw.drawing.DiagramRenderer
""",
    "members": """
members = dict(inspect.getmembers(sw))
assert members['DiagramRenderer'] is sw.drawing.DiagramRenderer
""",
    "pickle": """
cls = sw.DiagramRenderer
assert cls.__module__ == 'sft_wick.drawing'
assert pickle.loads(pickle.dumps(cls)) is cls
obj = cls(figsize=(4, 3))
# The pre-change renderer's mappingproxy style is not stdlib-pickleable.
try:
    pickle.dumps(obj)
except TypeError as e:
    assert str(e) == "cannot pickle 'mappingproxy' object"
else:
    raise AssertionError('renderer instance pickle behavior changed')
try:
    from joblib.externals import cloudpickle
except ImportError:
    import cloudpickle
assert cloudpickle.loads(cloudpickle.dumps(cls)) is cls
copy = cloudpickle.loads(cloudpickle.dumps(obj))
assert type(copy) is cls and copy.figsize == (4, 3)
""",
    "reload_unloaded": """
importlib.reload(sw)
assert 'sft_wick.drawing' not in sys.modules
assert 'matplotlib' not in sys.modules
assert 'DiagramRenderer' in dir(sw)
""",
    "reload_loaded": """
cls = sw.DiagramRenderer
sw.DiagramRenderer = object()
importlib.reload(sw)
assert sw.DiagramRenderer is cls is sw.drawing.DiagramRenderer
""",
    "reload_submodule": """
old = sw.DiagramRenderer
importlib.reload(sw.drawing)
current = sw.drawing.DiagramRenderer
assert current is not old
importlib.reload(sw)
assert sw.DiagramRenderer is current
""",
    "concurrent": """
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
gate = Barrier(8)
def request(i):
    gate.wait(timeout=30)
    return sw.drawing.DiagramRenderer if i % 2 else sw.DiagramRenderer
with ThreadPoolExecutor(max_workers=8) as pool:
    classes = list(pool.map(request, range(8)))
assert all(cls is sw.DiagramRenderer for cls in classes)
""",
    "blocked_retry": """
from importlib.abc import MetaPathFinder
class BlockPlotting(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'matplotlib' or fullname.startswith('matplotlib.'):
            raise ModuleNotFoundError('plotting sentinel', name=fullname)
blocker = BlockPlotting()
sys.meta_path.insert(0, blocker)
for name in ('DiagramRenderer', 'drawing', 'DiagramRenderer'):
    try:
        getattr(sw, name)
    except ModuleNotFoundError as e:
        assert str(e) == 'plotting sentinel' and e.name == 'matplotlib'
    else:
        raise AssertionError('missing plotting error')
    assert name not in vars(sw)
    assert 'sft_wick.drawing' not in sys.modules
sys.meta_path.remove(blocker)
assert sw.DiagramRenderer is sw.drawing.DiagramRenderer
""",
    "backend_render": """
import matplotlib
matplotlib.use('Agg')
diagram = sw.FeynmanDiagram()
a = diagram.add_external_point(label='a', field_type='physical', component='a', spatial='x')
b = diagram.add_external_point(label='b', field_type='physical', component='b', spatial='y')
diagram.add_propagator(a, b, kind='C')
figure = sw.DiagramRenderer().draw(diagram).figure
figure.canvas.draw()
assert matplotlib.get_backend().lower() == 'agg'
assert figure.axes and figure.canvas.buffer_rgba().nbytes > 0
import matplotlib.pyplot as plt
plt.close(figure)
""",
}


@pytest.fixture(scope="module")
def plotting_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("lazy-drawing") / "mpl"


@pytest.mark.parametrize("case", CASES)
def test_fresh_process_export_contract(case, plotting_cache):
    environment = os.environ.copy()
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONOPTIMIZE="0",
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(plotting_cache),
    )
    # Pin the same selected source tree even if the editable install points elsewhere.
    source = str(Path(__file__).resolve().parents[1] / "src")
    script = f"import sys; sys.path.insert(0, {source!r})\n" + PREFIX + CASES[case]
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        env=environment,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
