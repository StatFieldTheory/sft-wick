"""The parent-commit gate, ``tools/parent_gate.py``, and its replay record.

A toy git repository in ``tmp_path`` holds a package ``toypkg`` and one
branch per kind of change.  Each verdict follows from how its branch is
built: a test that passes on the parent's source, a test that fails there
with an assertion, a name the change adds, a crash, a time or memory limit.
The tests at the end read the committed replay record,
``docs/verification/parent_gate_replay.json``, and its page,
``docs/verification/parent_gate.rst``; they fail while either is missing.

No test here imports ``sft_wick``, so none can kill a mutant.  Under the
mutation harness the file is skipped.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import errno
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "verification" / "parent_gate_replay.json"
PAGE = ROOT / "docs" / "verification" / "parent_gate.rst"
CHANGELOG = ROOT / "CHANGELOG.md"

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("MUTATION_HARNESS")),
    reason="run by the mutation harness")


def _load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load_tool("parent_gate")


@pytest.fixture(scope="module")
def record_tools(gate):
    return gate.record_tools()


# ---- The toy repository ----

def _isolated_git_env() -> dict[str, str]:
    """No inherited ``GIT_*`` variables and no user or system configuration
    (the pattern of ``tests/test_repo_hygiene.py``)."""
    env = {key: val for key, val in os.environ.items() if not key.startswith("GIT_")}
    ident = {"NAME": "test", "EMAIL": "test@example.invalid"}
    for role in ("AUTHOR", "COMMITTER"):
        env.update({f"GIT_{role}_{key}": value for key, value in ident.items()})
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return env


PYPROJECT = """\
[project]
name = "toypkg"
version = "0"

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = ["slow: slow toy tests"]
"""

#: The parts of the toy package's ``__init__.py``, in order.  ``area`` holds
#: the defect: it returns 3 r^2.
SOURCE_PARTS = {
    "header": "import math\nimport time\n\nSCALE = 2.0\n",
    "area": "def area(r):\n    return 3.0 * r * r\n",
    "double": "def double(x):\n    return SCALE * x\n",
    "ratio": "def ratio(a, b):\n    return a / b\n",
    "root": "def root(x):\n    assert x >= 0\n    return math.sqrt(x)\n",
    "integrate": ("def integrate(method=\"simpson\"):\n"
                  "    if method != \"simpson\":\n"
                  "        raise ValueError(f\"unknown method {method!r}\")\n"
                  "    return 0.5\n"),
    "call": "def call(fn):\n    return fn()\n",
    "extra": "",
}
AREA_PI = "def area(r):\n    return math.pi * r * r\n"
TEST_HEADER = "import math\n\nimport pytest\n\nimport toypkg\n\n\n"
PACKAGE = "src/toypkg/__init__.py"


def _source(**parts: str) -> dict[str, str]:
    merged = {**SOURCE_PARTS, **parts}
    text = "\n\n".join(merged[key] for key in SOURCE_PARTS if merged[key])
    return {PACKAGE: text}


def _test(body: str) -> str:
    return TEST_HEADER + body


WITNESS = _test(
    "def test_area_is_pi():\n"
    "    assert toypkg.area(1.0) == pytest.approx(math.pi, rel=1e-12, abs=0.0)\n")

PERIMETER = ("def perimeter(r):\n"
             "    return 2.0 * math.pi * r\n")
PERIMETER_TEST = (
    "import math\n\nimport pytest\n\nfrom toypkg import perimeter\n\n\n"
    "def test_perimeter():\n"
    "    assert perimeter(1.0) == pytest.approx(2.0 * math.pi, rel=1e-12, abs=0.0)\n")
RATIO_INF = ("def ratio(a, b):\n    if b == 0:\n        return math.inf\n"
             "    return a / b\n")
RATIO_TEST = _test("def test_ratio_at_zero():\n    assert toypkg.ratio(1.0, 0.0) == math.inf\n")
RATIO_NAMED = ("def ratio(a, b):\n    if b == 0:\n"
               "        raise ZeroDivisionError(\"the divisor b is zero\")\n"
               "    return a / b\n")
#: Box gains a dataclass field and Disc a keyword; Ring takes it on both sides.
SHAPES = ("import dataclasses\n\n\n"
          "@dataclasses.dataclass(frozen=True)\nclass Box:\n    side: float\n{box}\n\n"
          "class Disc:\n    def __init__(self, r{disc}):\n        self.r = r\n{disc_body}\n\n"
          "class Ring:\n    def __init__(self, r, scale=1.0):\n"
          "        self.r, self.scale = r, scale\n")
SHAPES_PARENT = SHAPES.format(box="", disc="", disc_body="")
SHAPES_HEAD = SHAPES.format(box="    scale: float = 1.0\n", disc=", scale=1.0",
                            disc_body="        self.scale = scale\n")
PARSE = ("def parse(text):\n    try:\n        return float(text)\n"
         "    except ValueError:\n{handler}")
SLOW_TEST = _test("import time\n\n\ndef test_area_slowly():\n    time.sleep(20)\n"
                  "    assert toypkg.area(1.0) == pytest.approx(math.pi, rel=1e-12, abs=0.0)\n")
#: A name the child process of the escape test holds, so a test can look for it.
CHILD_MARKER = "toy_escaping_child"


@dataclasses.dataclass(frozen=True)
class Case:
    head: str
    parent: str
    settings: tuple[tuple[str, float], ...] = ()


class Toy:
    """A git repository built commit by commit."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.env = _isolated_git_env()
        self.cases: dict[str, Case] = {}
        self.refs: dict[str, str] = {}
        root.mkdir(parents=True)
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.root, env=self.env, check=True,
            capture_output=True, text=True).stdout.strip()

    def commit(self, message: str, files: dict[str, str | None]) -> str:
        for name, text in files.items():
            if text is None:
                self.git("rm", "-q", name)
                continue
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            self.git("add", name)
        self.git("commit", "-q", "--allow-empty", "-m", message)
        return self.git("rev-parse", "HEAD")

    def branch(self, name: str, start: str) -> None:
        self.git("checkout", "-q", "-b", name, start)

    def case(self, name: str, files: dict[str, str | None], *,
             start: str | None = None, settings: dict | None = None) -> None:
        start = start or self.refs["B0"]
        self.branch(name, start)
        head = self.commit(f"fix: {name.replace('_', ' ')}", files)
        self.cases[name] = Case(head, start, tuple(sorted((settings or {}).items())))


def _build_toy(root: Path) -> Toy:
    toy = Toy(root)
    toy.refs["B0"] = toy.commit("feat: a toy package", {
        "pyproject.toml": PYPROJECT,
        **_source(),
        "examples/expected.json": json.dumps({"unit_area": 3.0}) + "\n",
        "tests/test_double.py": _test(
            "def test_double():\n    assert toypkg.double(3.0) == 6.0\n"),
        "tests/test_area_loose.py": _test(
            "def test_area_loose():\n"
            "    assert toypkg.area(1.0) == pytest.approx(3.0, rel=0.1, abs=0.0)\n"),
    })
    b0 = toy.refs["B0"]
    toy.branch("moved", b0)
    toy.refs["moved"] = toy.commit("docs: a readme", {"README.md": "toy\n"})

    fixed = _source(area=AREA_PI)
    # Five changes whose new tests also pass on the parent.
    toy.case("constant", {**fixed, "tests/test_constant.py": _test(
        "def test_scale():\n    assert toypkg.SCALE == 2.0\n")})
    toy.case("wide_approx", {**fixed, "tests/test_wide.py": _test(
        "def test_area_wide():\n"
        "    assert toypkg.area(1.0) == pytest.approx(math.pi, abs=4.0)\n")})
    toy.case("false_branch", {**fixed, "tests/test_branch.py": _test(
        "def test_area_large():\n"
        "    value = toypkg.area(1.0)\n"
        "    if value > 100.0:\n"
        "        assert value == pytest.approx(math.pi, rel=1e-12, abs=0.0)\n")})
    toy.case("unchanged_function", {**fixed, "tests/test_double_again.py": _test(
        "def test_double_again():\n    assert toypkg.double(3.0) == 6.0\n")})
    toy.case("skipped", {**fixed, "tests/test_skipped.py": _test(
        "@pytest.mark.skip(reason=\"later\")\n"
        "def test_area_is_pi():\n"
        "    assert toypkg.area(1.0) == pytest.approx(math.pi, rel=1e-12, abs=0.0)\n")})

    # Tests that fail on the parent with an assertion.
    toy.case("witness", {**fixed, "tests/test_area.py": WITNESS})
    toy.case("refusal", {
        **_source(area="def area(r):\n    if r < 0:\n"
                       "        raise ValueError(\"negative radius\")\n"
                       "    return 3.0 * r * r\n"),
        "tests/test_refusal.py": _test(
            "def test_negative_radius_is_refused():\n"
            "    with pytest.raises(ValueError):\n"
            "        toypkg.area(-1.0)\n")})
    toy.case("existing_test_tightened", {**fixed, "tests/test_area_loose.py": _test(
        "def test_area_loose():\n"
        "    assert toypkg.area(1.0) == pytest.approx(math.pi, rel=1e-12, abs=0.0)\n")})
    toy.case("fixture_assertion", {**fixed, "tests/test_fixture.py": _test(
        "@pytest.fixture(scope=\"module\")\n"
        "def unit_area():\n"
        "    value = toypkg.area(1.0)\n"
        "    assert value == pytest.approx(math.pi, rel=1e-12, abs=0.0)\n"
        "    return value\n\n\n"
        "def test_unit_area_is_positive(unit_area):\n"
        "    assert unit_area > 0\n")})

    # Own runs that fail in this environment.
    toy.branch("parent_env_base", b0)
    p_env = toy.commit("feat: area reads a dependency", {
        **_source(area="def area(r):\n    import toypkg_removed_dependency\n"
                       "    return 3.0 * r * r\n"),
        "tests/test_area_env.py": _test(
            "def test_area_value():\n"
            "    assert toypkg.area(2.0) == pytest.approx(12.0, rel=1e-12, abs=0.0)\n")})
    toy.case("parent_env", {**fixed, "tests/test_area_env.py": _test(
        "def test_area_value():\n"
        "    assert toypkg.area(2.0) == pytest.approx(4.0 * math.pi, rel=1e-12, abs=0.0)\n\n\n"
        "def test_area_unit():\n"
        "    assert toypkg.area(1.0) == pytest.approx(math.pi, rel=1e-12, abs=0.0)\n")},
        start=p_env)
    toy.case("head_env", {**fixed, "tests/test_head_env.py": _test(
        "def test_area_is_three():\n    assert toypkg.area(1.0) == 3.0\n")})

    # Names the change adds.
    toy.case("new_api", {**_source(extra=PERIMETER),
                         "tests/test_perimeter.py": PERIMETER_TEST})
    toy.case("new_api_beside_witness", {
        **_source(area=AREA_PI, extra=PERIMETER),
        "tests/test_perimeter.py": PERIMETER_TEST, "tests/test_area.py": WITNESS})
    toy.case("monkeypatched_new_helper", {
        **_source(area="def _area_factor():\n    return math.pi\n\n\n"
                       "def area(r):\n    return _area_factor() * r * r\n"),
        "tests/test_helper.py": _test(
            "def test_area_uses_its_factor(monkeypatch):\n"
            "    monkeypatch.setattr(toypkg, \"_area_factor\", lambda: 2.0)\n"
            "    assert toypkg.area(1.0) == 2.0\n")})
    toy.case("new_keyword", {
        **_source(area="def area(r, *, scale=1.0):\n"
                       "    return scale * math.pi * r * r\n"),
        "tests/test_keyword.py": _test(
            "def test_area_with_a_scale():\n"
            "    value = toypkg.area(1.0, scale=1.0)\n"
            "    assert value == pytest.approx(math.pi, rel=1e-12, abs=0.0)\n")})

    # Crashes of the parent's source.
    toy.case("crash", {**_source(ratio=RATIO_INF), "tests/test_ratio.py": RATIO_TEST})
    toy.case("assert_in_source", {
        **_source(root="def root(x):\n    if x < 0:\n        return math.nan\n"
                       "    return math.sqrt(x)\n"),
        "tests/test_root.py": _test(
            "def test_root_of_a_negative_is_nan():\n"
            "    assert math.isnan(toypkg.root(-1.0))\n")})
    toy.case("api_value_refused", {
        **_source(integrate="def integrate(method=\"simpson\"):\n"
                            "    if method == \"exact\":\n        return 1.0\n"
                            "    if method != \"simpson\":\n"
                            "        raise ValueError(f\"unknown method {method!r}\")\n"
                            "    return 0.5\n"),
        "tests/test_integrate.py": _test(
            "def test_exact_method():\n"
            "    assert toypkg.integrate(method=\"exact\") == 1.0\n")})

    # Limits on the parent.
    area_test = {"tests/test_area_value.py": WITNESS}
    toy.branch("slow_base", b0)
    p_slow = toy.commit("feat: a slow area", _source(
        area="def area(r):\n    time.sleep(60)\n    return 3.0 * r * r\n"))
    toy.case("timeout_on_parent", {**fixed, **area_test}, start=p_slow,
             settings={"overlay_floor": 2.0, "time_limit": 60.0})
    toy.branch("big_base", b0)
    p_big = toy.commit("feat: a large area", _source(
        area="def area(r):\n    blob = b\"\\x01\" * (600 * 2**20)\n"
             "    time.sleep(3)\n    return 3.0 * r * r + 0.0 * len(blob)\n"))
    toy.case("memory_on_parent", {**fixed, **area_test}, start=p_big,
             settings={"max_rss_gb": 0.25})

    # An overlay that writes no report; files outside src/.
    toy.case("conftest_new_name", {
        **_source(extra="FIXTURE_SCALE = 1.0\n"),
        "tests/conftest.py": (
            "import pytest\n\nfrom toypkg import FIXTURE_SCALE\n\n\n"
            "@pytest.fixture\ndef fixture_scale():\n    return FIXTURE_SCALE\n"),
        "tests/test_fixture_scale.py": (
            "def test_fixture_scale(fixture_scale):\n    assert fixture_scale == 1.0\n")})
    toy.case("example_data", {
        **fixed,
        "examples/expected.json": json.dumps({"unit_area": 3.141592653589793}) + "\n",
        "tests/test_expected.py": _test(
            "from pathlib import Path\nimport json\n\n\n"
            "def test_area_matches_the_example():\n"
            "    path = Path(__file__).resolve().parents[1] / \"examples\" / \"expected.json\"\n"
            "    expected = json.loads(path.read_text())[\"unit_area\"]\n"
            "    assert toypkg.area(1.0) == pytest.approx(expected, rel=1e-12, abs=0.0)\n")})

    _add_chained_and_mixed_cases(toy, b0)
    _add_name_cases(toy, b0)
    _add_process_cases(toy, b0)

    # No source difference.
    toy.case("tests_only", {"tests/test_double.py": _test(
        "def test_double():\n    assert toypkg.double(3.0) == 6.0\n\n\n"
        "def test_double_zero():\n    assert toypkg.double(0.0) == 0.0\n")})
    toy.case("test_only_vacuous", {"tests/test_vacuous.py": _test(
        "def test_scale():\n    assert toypkg.SCALE == 2.0\n")})

    # A pull request whose base has moved on: M1 on main fixes area too.
    toy.branch("main_ahead", b0)
    toy.refs["M1"] = toy.commit("fix: area is pi r squared", fixed)
    toy.branch("pr", b0)
    toy.refs["PR1"] = toy.commit("fix: area, with its test",
                                 {**fixed, "tests/test_area.py": WITNESS})
    toy.refs["PR2"] = toy.commit("docs: the readme", {"README.md": "toy\n"})

    # A linear history for the population and the replay.
    toy.branch("history", b0)
    toy.refs["in_1"] = toy.commit("fix: area is pi r squared",
                                  {**fixed, "tests/test_area.py": WITNESS})
    toy.commit("feat: a perimeter", {**_source(area=AREA_PI, extra=PERIMETER),
                                     "tests/test_perimeter.py": PERIMETER_TEST})
    toy.commit("fix: the readme", {"README.md": "toy\n"})
    toy.refs["in_2"] = toy.commit("fix(scale): name the unit", {
        **_source(area=AREA_PI, extra=PERIMETER + "\n\nUNIT = 1.0\n"),
        "tests/test_constant.py": _test(
            "def test_scale():\n    assert toypkg.SCALE == 2.0\n")})
    toy.commit("fix: a nested test", {"tests/sub/test_x.py": _test(
        "def test_x():\n    assert toypkg.SCALE == 2.0\n")})
    tip = toy.git("rev-parse", "HEAD")
    toy.branch("side", tip)
    toy.commit("docs: a side note", {"NOTES.md": "side\n"})
    toy.git("checkout", "-q", "history")
    toy.git("merge", "-q", "--no-ff", "--no-edit", "-m", "fix: merge the side note", "side")
    toy.refs["history"] = toy.git("rev-parse", "HEAD")
    toy.git("checkout", "-q", "main")
    return toy


def _add_chained_and_mixed_cases(toy: Toy, b0: str) -> None:
    """Failures chained to an exception of the parent's source, and mixed kinds."""
    toy.case("chained_assertion", {
        **_source(ratio=RATIO_NAMED),
        "tests/test_ratio_named.py": _test(
            "def test_ratio_names_the_divisor():\n"
            "    with pytest.raises(ZeroDivisionError, match=\"divisor b\"):\n"
            "        toypkg.ratio(1.0, 0.0)\n\n\n"
            "def test_ratio_message_in_a_handler():\n"
            "    try:\n        toypkg.ratio(1.0, 0.0)\n"
            "    except ZeroDivisionError as exc:\n"
            "        assert \"divisor\" in str(exc)\n")})
    toy.branch("parse_base", b0)
    p_parse = toy.commit("feat: parse", _source(
        extra=PARSE.format(handler="        assert False, \"not a number\"\n")))
    toy.case("assert_in_source_chained", {
        **_source(extra=PARSE.format(handler="        return math.nan\n")),
        "tests/test_parse.py": _test(
            "def test_parse_of_a_word_is_nan():\n"
            "    assert math.isnan(toypkg.parse(\"x\"))\n")}, start=p_parse)
    toy.case("witness_beside_crash", {
        **_source(area=AREA_PI, ratio=RATIO_INF),
        "tests/test_area.py": WITNESS, "tests/test_ratio.py": RATIO_TEST})
    toy.branch("factor_base", b0)
    p_factor = toy.commit("feat: area calls a factor", _source(
        area="def area(r):\n    return _factor() * r * r\n"))
    toy.case("nameerror_in_parent_source", {
        **_source(area="def _factor():\n    return math.pi\n\n\n"
                       "def area(r):\n    return _factor() * r * r\n"),
        "tests/test_area.py": WITNESS}, start=p_factor)


def _add_name_cases(toy: Toy, b0: str) -> None:
    """Names the change adds, read from messages the other cases do not give."""
    toy.branch("shapes_base", b0)
    p_shapes = toy.commit("feat: shapes", _source(extra=SHAPES_PARENT))
    toy.case("new_init_keyword", {
        **_source(extra=SHAPES_HEAD),
        "tests/test_box.py": _test(
            "def test_box_takes_a_scale():\n"
            "    assert toypkg.Box(2.0, scale=0.5).scale == 0.5\n\n\n"
            "def test_disc_takes_a_scale():\n"
            "    assert toypkg.Disc(1.0, scale=0.5).scale == 0.5\n")}, start=p_shapes)
    toy.case("string_monkeypatch", {
        **_source(area="def _area_factor():\n    return math.pi\n\n\n"
                       "def area(r):\n    return _area_factor() * r * r\n"),
        "tests/test_helper_string.py": _test(
            "def test_area_uses_its_factor(monkeypatch):\n"
            "    monkeypatch.setattr(\"toypkg._area_factor\", lambda: 2.0)\n"
            "    assert toypkg.area(1.0) == 2.0\n")})
    toy.case("importorskip_new_module", {
        "src/toypkg/geometry.py": "import math\n\n\ndef circle(r):\n    return math.pi * r * r\n",
        "tests/test_geometry.py": (
            "import math\n\nimport pytest\n\n"
            "geometry = pytest.importorskip(\"toypkg.geometry\")\n\n\n"
            "def test_circle():\n"
            "    assert geometry.circle(1.0) == pytest.approx(math.pi, rel=1e-12, abs=0.0)\n")})
    toy.case("source_only", _source(area=AREA_PI))


def _add_process_cases(toy: Toy, b0: str) -> None:
    """Runs killed from outside, a symbolic link for src/, slow and escaping runs."""
    fixed = _source(area=AREA_PI)
    for name, number, workers in (("killed_by_sigkill", "SIGKILL", 0),
                                  ("killed_by_sigusr1", "SIGUSR1", 0),
                                  ("killed_by_sigint", "SIGINT", 0),
                                  ("worker_killed_by_sigkill", "SIGKILL", 2)):
        toy.case(name, {**fixed, "tests/test_killed.py": _test(
            "import os\nimport signal\n\n\n"
            "def test_the_run_is_killed():\n"
            f"    os.kill(os.getpid(), signal.{number})\n")},
            settings={"pytest_workers": workers} if workers else None)
    toy.case("escaping_child", {**fixed, "tests/test_escape.py": _test(
        "import subprocess\nimport sys\n\n\n"
        "def test_area_after_a_child():\n"
        "    code = \"import time; " + CHILD_MARKER + " = b'x' * (400 * 2**20); "
        "time.sleep(4)\"\n"
        "    subprocess.run([sys.executable, \"-c\", code], start_new_session=True, check=True)\n"
        "    assert toypkg.area(1.0) == pytest.approx(math.pi, rel=1e-12, abs=0.0)\n")},
        settings={"max_rss_gb": 0.25})
    toy.case("slow_head", {**fixed, "tests/test_slow.py": SLOW_TEST})
    toy.branch("symlinked_src", b0)
    toy.git("mv", "src", "lib")
    (toy.root / "src").symlink_to("lib")
    toy.git("add", "src")
    head = toy.commit("fix: area, with src as a link", {
        "lib/toypkg/__init__.py": fixed[PACKAGE], "tests/test_area.py": WITNESS})
    toy.cases["symlinked_src"] = Case(head, b0)
    toy.branch("slow_history", b0)
    toy.refs["slow_1"] = toy.commit("fix: area, slowly", {
        **fixed, "tests/test_slow_a.py": SLOW_TEST})
    toy.refs["slow_2"] = toy.commit("fix: a unit, slowly", {
        **_source(area=AREA_PI, extra="UNIT = 1.0\n"), "tests/test_slow_b.py": SLOW_TEST})


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    return _build_toy(tmp_path_factory.mktemp("toy") / "repo")


@pytest.fixture(scope="module")
def judged(gate, toy, tmp_path_factory):
    """A judgement of a toy case, made once per case and base."""
    cache = {}

    def run(case: str, *, base: str | None = None):
        key = (case, base)
        if key not in cache:
            spec = toy.cases[case]
            cache[key] = gate.judge(
                toy.root, spec.head, base=base,
                settings=gate.Settings(**dict(spec.settings)),
                work_dir=tmp_path_factory.mktemp("judge"))
        return cache[key]

    return run


def _overlay_failures(judgement):
    return [r for r in judgement.runs["overlay"].results
            if r.outcome in ("failed", "error", "not_run")]


# ---- Verdicts on the toy repository ----

@pytest.mark.parametrize("mode", ["commit", "pull_request"])
@pytest.mark.parametrize("case", ["constant", "wide_approx", "false_branch",
                                  "unchanged_function", "skipped"])
def test_a_test_that_passes_on_the_parent_is_rejected(judged, toy, case, mode):
    base = toy.refs["moved"] if mode == "pull_request" else None
    judgement = judged(case, base=base)
    assert judgement.decision == "reject"
    assert judgement.verdict == ("no_evidence" if case == "skipped"
                                 else "passes_on_parent")
    assert judgement.parent == toy.refs["B0"]
    assert judgement.parent_rule == ("merge_base" if base else "first_parent")
    if case != "skipped":
        assert any(r.outcome == "passed" for r in judgement.runs["head"].results)


@pytest.mark.parametrize("case, exception, phase, existing", [
    ("witness", "AssertionError", "call", False),
    ("refusal", "Failed", "call", False),
    ("existing_test_tightened", "AssertionError", "call", True),
    ("fixture_assertion", "AssertionError", "setup", False),
    ("chained_assertion", "AssertionError", "call", False),
])
def test_a_test_that_fails_on_the_parent_with_an_assertion_is_accepted(
        judged, case, exception, phase, existing):
    judgement = judged(case)
    assert (judgement.verdict, judgement.decision) == ("accepted", "pass")
    assert judgement.witnesses
    for witness in judgement.witnesses:
        assert witness.kind == "assertion"
        assert witness.exception == exception
        assert witness.phase == phase
        assert witness.existing is existing


@pytest.mark.parametrize("side", ["parent", "head"])
def test_a_failing_own_run_is_environment(judged, side):
    judgement = judged(f"{side}_env")
    assert judgement.verdict == "environment"
    assert (judgement.side, judgement.cause) == (side, "failure")
    assert judgement.decision == "undetermined"
    assert judgement.runs["overlay"] is None
    if side == "parent":
        failing = [r.id for r in judgement.runs["parent_own"].results
                   if r.outcome in ("failed", "error")]
        assert failing == ["tests/test_area_env.py::test_area_value"]
        assert [w.id for w in judgement.witnesses] == failing
    else:
        assert judgement.runs["parent_own"] is None


@pytest.mark.parametrize("case, verdict", [
    ("new_api", "new_api_only"),
    ("new_api_beside_witness", "accepted"),
    ("monkeypatched_new_helper", "new_api_only"),
    ("new_keyword", "new_api_only"),
    ("new_init_keyword", "new_api_only"),
    ("string_monkeypatch", "new_api_only"),
], ids=["alone", "beside_a_witness", "monkeypatched_helper", "new_keyword",
        "init_keyword", "monkeypatched_by_name"])
def test_a_name_the_change_adds_is_not_a_witness(judged, case, verdict):
    judgement = judged(case)
    assert judgement.verdict == verdict
    failures = _overlay_failures(judgement)
    missing = [r for r in failures if r.kind == "missing_api"]
    assert missing, failures
    assert all(r.id.startswith(("tests/test_perimeter.py", "tests/test_helper.py",
                                "tests/test_keyword.py", "tests/test_box.py",
                                "tests/test_helper_string.py")) for r in missing)
    if verdict == "accepted":
        assert [w.id for w in judgement.witnesses] == ["tests/test_area.py::test_area_is_pi"]
    else:
        assert judgement.decision == "reject"
        assert all(r.kind == "missing_api" for r in failures)


@pytest.mark.parametrize("case, exception", [
    ("crash", "ZeroDivisionError"),
    ("assert_in_source", "AssertionError"),
    ("api_value_refused", "ValueError"),
    ("assert_in_source_chained", "AssertionError"),
    ("nameerror_in_parent_source", "NameError"),
], ids=["exception", "assert_in_source", "refused_option", "assert_in_source_chained",
        "name_error_in_source"])
def test_a_crash_of_the_parent_is_its_own_verdict(judged, case, exception):
    judgement = judged(case)
    assert judgement.verdict == "crash_only"
    assert judgement.decision == "reject"
    failures = _overlay_failures(judgement)
    assert failures
    assert [(r.kind, r.exception) for r in failures] == [("crash", exception)]
    assert [w.id for w in judgement.witnesses] == [r.id for r in failures]


@pytest.mark.parametrize("case, kind, flag", [
    ("timeout_on_parent", "timeout", "timed_out"),
    ("memory_on_parent", "memory", "memory_killed"),
], ids=["timeout", "memory"])
def test_a_resource_limit_on_the_parent_is_not_a_witness(gate, toy, tmp_path, case, kind, flag):
    spec = toy.cases[case]
    settings = gate.Settings(**dict(spec.settings))
    judgement = gate.judge(toy.root, spec.head, settings=settings, work_dir=tmp_path)
    assert (judgement.verdict, judgement.decision) == ("resource_only", "reject")
    overlay = judgement.runs["overlay"]
    assert getattr(overlay, flag)
    assert [(r.outcome, r.kind) for r in _overlay_failures(judgement)] == [("not_run", kind)]
    if kind == "timeout":
        limit = gate._later_limit(settings, judgement.runs["head"])
        assert limit < 60.0
        assert overlay.seconds < limit + 5.0


def test_an_overlay_that_writes_no_report_is_not_a_pass(judged):
    judgement = judged("conftest_new_name")
    assert (judgement.verdict, judgement.decision) == ("overlay_not_run", "reject")
    overlay = judgement.runs["overlay"]
    assert overlay.exit == 4
    assert overlay.has_report is False
    assert "FIXTURE_SCALE" in judgement.reason


def test_files_outside_src_come_from_the_change(judged):
    judgement = judged("example_data")
    assert (judgement.verdict, judgement.decision) == ("accepted", "pass")
    assert [w.id for w in judgement.witnesses] == [
        "tests/test_expected.py::test_area_matches_the_example"]


@pytest.mark.parametrize("case", ["tests_only", "test_only_vacuous"], ids=["tests_only", "vacuous"])
def test_a_change_without_a_source_difference_passes_without_a_run(judged, case):
    judgement = judged(case)
    assert judgement.verdict == "no_code_change"
    assert judgement.decision == "pass"
    assert judgement.src_changed is False
    assert all(run is None for run in judgement.runs.values())


def test_a_pull_request_is_judged_against_the_merge_base(gate, toy, tmp_path):
    alone = gate.judge(toy.root, toy.refs["PR2"], work_dir=tmp_path / "alone")
    assert alone.verdict == "no_code_change"
    assert alone.parent == toy.refs["PR1"]
    judgement = gate.judge(toy.root, toy.refs["PR2"], base=toy.refs["M1"],
                           work_dir=tmp_path / "pr")
    assert judgement.parent == toy.refs["B0"]
    assert judgement.parent_rule == "merge_base"
    assert judgement.base == toy.refs["M1"]
    assert (judgement.verdict, judgement.decision) == ("accepted", "pass")


@pytest.mark.parametrize("case, code", [("witness", 0), ("constant", 1), ("parent_env", 3),
                                        ("source_only", 1)])
def test_the_command_line_exits_by_decision(gate, toy, tmp_path, case, code):
    decoy = tmp_path / "decoy.git"
    env = {**os.environ, "GIT_DIR": str(decoy), "GIT_WORK_TREE": str(tmp_path)}
    out, summary = tmp_path / "verdict.json", tmp_path / "summary.md"
    summary.write_text("before\n")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "parent_gate.py"), "judge",
         "--repo", str(toy.root), "--head", toy.cases[case].head,
         "--json", str(out), "--summary", str(summary),
         "--work-dir", str(tmp_path / "work")],
        env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == code, proc.stdout + proc.stderr
    verdict = json.loads(out.read_text())
    expected = {0: "pass", 1: "reject", 3: "undetermined"}[code]
    assert verdict["decision"] == expected
    assert verdict["head"] == toy.cases[case].head
    assert verdict["rules_digest"] == gate.rules_digest()
    text = summary.read_text()
    assert text.startswith("before\n")
    assert f"### Parent-commit gate: {verdict['verdict']} ({expected})" in text
    assert not decoy.exists()


def test_an_assertion_beside_a_crash_is_accepted(judged):
    judgement = judged("witness_beside_crash")
    assert (judgement.verdict, judgement.decision) == ("accepted", "pass")
    assert [w.id for w in judgement.witnesses] == ["tests/test_area.py::test_area_is_pi"]
    assert {r.kind for r in _overlay_failures(judgement)} == {"assertion", "crash"}


def test_a_change_to_the_source_without_a_test_is_rejected(judged):
    judgement = judged("source_only")
    assert (judgement.verdict, judgement.decision) == ("no_touched_tests", "reject")
    assert judgement.src_changed is True
    assert all(run is None for run in judgement.runs.values())


@pytest.mark.parametrize("exception, message", [
    ("AttributeError", "'Holder' object has no attribute 'bar'"),
    ("ImportError", "cannot import name 'bar' from 'toypkg' (<tree>/src/toypkg/__init__.py)"),
    ("NameError", "name 'bar' is not defined"),
])
def test_a_name_the_parent_also_defines_is_not_missing(gate, exception, message):
    defines = gate.Names(frozenset({"toypkg"}), {"toypkg": frozenset({"bar"})},
                         frozenset({"bar"}), {})
    lacks = gate.Names(frozenset({"toypkg"}), {"toypkg": frozenset()}, frozenset(), {})
    assert gate.missing_name(exception, message, defines, lacks)
    assert not gate.missing_name(exception, message, defines, defines)


def test_an_overlay_that_skips_every_test_names_its_exit(judged):
    judgement = judged("importorskip_new_module")
    assert (judgement.verdict, judgement.decision) == ("overlay_not_run", "reject")
    overlay = judgement.runs["overlay"]
    assert (overlay.exit, overlay.has_report) == (5, True)
    assert "exited 5" in judgement.reason
    assert "no report" not in judgement.reason


@pytest.mark.parametrize("case, killed", [("killed_by_sigkill", "SIGKILL"),
                                          ("killed_by_sigint", "exited 2"),
                                          ("worker_killed_by_sigkill", "xdist worker"),
                                          ("killed_by_sigusr1", None)],
                         ids=["SIGKILL", "SIGINT", "SIGKILL_of_a_worker", "SIGUSR1"])
def test_a_run_killed_from_outside_gives_no_verdict(gate, judged, case, killed):
    if killed:
        with pytest.raises(gate.GateError, match=killed):
            judged(case)
        return
    judgement = judged(case)
    assert (judgement.verdict, judgement.side, judgement.cause) == (
        "environment", "head", "failure")


def test_a_source_root_that_is_a_symbolic_link_is_refused(gate, judged):
    with pytest.raises(gate.GateError, match="symbolic link"):
        judged("symlinked_src")


def _processes_mentioning(text: str) -> list[int]:
    out = subprocess.run(["ps", "-A", "-o", "pid=,command="], capture_output=True,
                         text=True).stdout
    return [int(line.split(None, 1)[0]) for line in out.splitlines() if text in line]


def _none_left(text: str, seconds: float = 3.0) -> list[int]:
    deadline = time.monotonic() + seconds
    while (left := _processes_mentioning(text)) and time.monotonic() < deadline:
        time.sleep(0.1)
    return left


def _kill_leftovers(proc: subprocess.Popen, text: str) -> None:
    proc.kill()
    proc.wait()
    for pid in _processes_mentioning(text):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


def _wait_for(base: Path, pattern: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while not list(base.glob(pattern)):
        assert time.monotonic() < deadline, f"no {base / pattern} after {seconds} s"
        time.sleep(0.1)


def test_a_process_in_a_new_session_counts_for_the_memory_limit(judged):
    judgement = judged("escaping_child")
    assert (judgement.verdict, judgement.side, judgement.cause) == (
        "environment", "head", "memory")
    assert judgement.runs["head"].memory_killed
    assert _none_left(CHILD_MARKER) == []


def test_a_terminated_judge_kills_its_run_and_removes_its_trees(toy, tmp_path):
    work = tmp_path / "work"
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "tools" / "parent_gate.py"), "judge",
         "--repo", str(toy.root), "--head", toy.cases["slow_head"].head,
         "--work-dir", str(work)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        _wait_for(work, "head.log", 60.0)
        time.sleep(1.0)
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=15)
        assert proc.returncode == 128 + signal.SIGTERM
        assert _none_left(str(work)) == []
        assert not (work / "C").exists()
    finally:
        _kill_leftovers(proc, str(work))


def test_an_interrupted_replay_stops_its_runs_and_starts_no_more(toy, tmp_path):
    out = tmp_path / "out"
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "tools" / "parent_gate.py"), "replay",
         "--repo", str(toy.root), "--range", f"{toy.refs['B0']}..{toy.refs['slow_2']}",
         "--out", str(out), "--jobs", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        _wait_for(out, "trees/*/head.log", 60.0)
        time.sleep(1.0)
        proc.send_signal(signal.SIGINT)
        proc.communicate(timeout=15)
        assert proc.returncode == 128 + signal.SIGINT
        assert _none_left(str(out)) == []
        assert list((out / "commits").glob("*.json")) == []
        assert len(list((out / "logs").glob("*-head.txt"))) == 1
        assert [row[0] for row in _log_rows(out / "replay.log")][-1] == "interrupted"
    finally:
        _kill_leftovers(proc, str(out))


def _run_module_script(body: str) -> subprocess.CompletedProcess:
    """``body`` run in a new interpreter with ``run`` the loaded tools/parent_gate_run.py."""
    head = ("import importlib.util, os, signal, sys, threading, time\n"
            f"path = {str(ROOT / 'tools' / 'parent_gate_run.py')!r}\n"
            "spec = importlib.util.spec_from_file_location('run_under_test', path)\n"
            "run = importlib.util.module_from_spec(spec)\n"
            "sys.modules[spec.name] = run\n"
            "spec.loader.exec_module(run)\n"
            "run.install_signal_handlers()\n")
    return subprocess.run([sys.executable, "-c", head + body], capture_output=True,
                          text=True, timeout=30)


def test_a_signal_handler_takes_no_lock_and_waits_for_a_run_to_be_registered():
    held = _run_module_script(
        "with run._LIVE_LOCK:\n"
        "    os.kill(os.getpid(), signal.SIGTERM)\n"
        "    time.sleep(10)\n")
    assert held.returncode == 128 + signal.SIGTERM, held.stderr
    deferred = _run_module_script(
        "try:\n"
        "    with run._signals_deferred():\n"
        "        os.kill(os.getpid(), signal.SIGTERM)\n"
        "        time.sleep(0.2)\n"
        "        print('registered', run._STOPPED, flush=True)\n"
        "except SystemExit as exc:\n"
        "    print('raised', exc.code, flush=True)\n")
    assert deferred.stdout.split() == ["registered", "True", "raised", "143"], deferred.stderr


def test_a_replay_that_raises_leaves_no_stop_behind(gate, record_tools, toy, tmp_path,
                                                    monkeypatch):
    def broken_judge(*args, **kwargs):
        raise ValueError("a defect in one judgement")

    commits = record_tools.population(toy.root, f"{toy.refs['B0']}..{toy.refs['history']}")
    monkeypatch.setattr(gate, "judge", broken_judge)
    with pytest.raises(ValueError, match="a defect"):
        record_tools.replay(toy.root, commits[:1], gate=gate, out_dir=tmp_path / "out",
                            jobs=1)
    assert not gate.run_tools._STOPPED
    outcome = gate.run_limited([sys.executable, "-c", "pass"], cwd=tmp_path,
                               env=dict(os.environ), timeout=60, max_rss_bytes=2**30,
                               log=tmp_path / "later.log")
    assert outcome.exit == 0


def test_a_process_is_known_by_its_pid_and_start_time_and_forgotten_once_gone(gate):
    run = gate.run_tools
    row = run.Process
    tree = run.Tree(100)
    first = {100: row(1, 100, 0, "s100"), 200: row(100, 100, 0, "s200"),
             300: row(200, 300, 0, "s300")}
    assert tree.members(first) == [100, 200, 300]
    # 200 exits: 300 is no longer a descendant, 301 joined its group after the last poll.
    escaped = {100: row(1, 100, 0, "s100"), 300: row(1, 300, 0, "s300"),
               301: row(1, 300, 0, "s301")}
    assert tree.members(escaped) == [100, 300, 301]
    gone = {100: row(1, 100, 0, "s100")}
    assert tree.members(gone) == [100]
    assert set(tree.seen) == {100}
    # Pid 300 now leads an unrelated group, after one poll or with no poll between.
    reused = {**gone, 300: row(1, 300, 0, "other"), 302: row(300, 300, 0, "s302")}
    assert tree.members(reused) == [100]
    direct = run.Tree(100)
    assert direct.members(first) == [100, 200, 300]
    assert direct.members(reused) == [100]
    tree.root_reaped()
    assert tree.members({100: row(1, 100, 0, "other"), 101: row(100, 100, 0, "s101")}) == []


def test_a_run_never_kills_a_process_that_reused_a_pid_it_saw(gate):
    run = gate.run_tools
    leader = subprocess.Popen(["sleep", "30"], start_new_session=True)
    other = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        time.sleep(0.3)
        tree = run.Tree(leader.pid)
        assert tree.members(run._process_table()) == [leader.pid]
        tree.seen = {**tree.seen, other.pid: "a start time other does not have"}
        assert other.pid not in tree.members(run._process_table())
        tree.kill()
        assert leader.wait(timeout=5) == -signal.SIGKILL
        time.sleep(0.3)
        assert other.poll() is None
    finally:
        for proc in (leader, other):
            proc.kill()
            proc.wait()


def _process_row(pid: int) -> str:
    """``pid ppid pgid command`` of ``pid`` from ``ps``, empty when there is no such process."""
    return subprocess.run(["ps", "-o", "pid=,ppid=,pgid=,command=", "-p", str(pid)],
                          capture_output=True, text=True).stdout.strip()


def test_a_process_left_in_the_run_s_group_after_the_last_poll_is_killed(gate, tmp_path,
                                                                        monkeypatch):
    run = gate.run_tools
    monkeypatch.setattr(run, "POLL_SECONDS", 2.0)
    root_file, orphan_file = tmp_path / "root.pid", tmp_path / "orphan.pid"
    # The subshell exits at once: the sleep's parent becomes pid 1, its group stays the run's.
    script = (f"echo $$ > {shlex.quote(str(root_file))}; sleep 0.5; "
              f"(sleep 47.25 & echo $! > {shlex.quote(str(orphan_file))}); exit 0")
    try:
        outcome = run.run_limited(["sh", "-c", script], cwd=tmp_path, env=dict(os.environ),
                                  timeout=30, max_rss_bytes=2**30, log=tmp_path / "run.log")
        assert outcome.exit == 0
        time.sleep(0.3)
        assert "sleep 47.25" not in _process_row(int(orphan_file.read_text()))
    finally:
        if orphan_file.is_file() and root_file.is_file():
            orphan, root = int(orphan_file.read_text()), int(root_file.read_text())
            row = _process_row(orphan).split()
            if "47.25" in row and row[2] == str(root):
                os.kill(orphan, signal.SIGKILL)


def test_a_run_whose_process_table_cannot_be_read_is_still_killed(gate, tmp_path,
                                                                  monkeypatch):
    run = gate.run_tools
    roots: list[int] = []

    def no_ps():
        roots.extend(run._LIVE)
        raise BlockingIOError(errno.EAGAIN, "fork: resource temporarily unavailable")

    monkeypatch.setattr(run, "_process_table", no_ps)
    try:
        with pytest.raises(BlockingIOError):
            run.run_limited(["sleep", "60.75"], cwd=tmp_path, env=dict(os.environ),
                            timeout=30, max_rss_bytes=2**30, log=tmp_path / "run.log")
        assert roots
        assert not set(roots) & set(run._LIVE)
        for pid in set(roots):
            with pytest.raises(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)
    finally:
        for pid in set(roots):
            with run._LIVE_LOCK:
                run._LIVE.pop(pid, None)
            try:
                if os.waitpid(pid, os.WNOHANG)[0]:
                    continue
            except ChildProcessError:
                continue
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)


def test_the_signal_handler_calls_nothing_that_can_block(gate):
    run = gate.run_tools
    handler = run._on_signal.__code__
    calls: list[str] = []

    def inside(frame) -> bool:
        while frame is not None:
            if frame.f_code is handler:
                return True
            frame = frame.f_back
        return False

    def profile(frame, event, arg) -> None:
        if event == "call" and frame.f_code is not handler and inside(frame.f_back):
            calls.append(frame.f_code.co_name)
        elif event == "c_call" and inside(frame):
            calls.append(getattr(arg, "__qualname__", repr(arg)))

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(SystemExit) as raised:
            run._on_signal(signal.SIGTERM, None)
    finally:
        sys.setprofile(previous)
        run.clear_stop()
    assert raised.value.code == 128 + signal.SIGTERM
    assert calls == []


def test_a_signal_just_after_the_reap_sends_nothing_to_the_reaped_pid(tmp_path):
    log = tmp_path / "run.log"
    done = _run_module_script(
        "reaped, after = [], []\n"
        "real = {name: getattr(os, name) for name in ('wait4', 'kill', 'killpg', 'waitpid')}\n"
        "def wait4(pid, options):\n"
        "    result = real['wait4'](pid, options)\n"
        "    if result[0]:\n"
        "        reaped.append(result[0])\n"
        "        real['kill'](os.getpid(), signal.SIGTERM)\n"
        "    return result\n"
        "def recorded(name):\n"
        "    def call(pid, *args):\n"
        "        if pid in reaped:\n"
        "            after.append(name)\n"
        "        return real[name](pid, *args)\n"
        "    return call\n"
        "os.wait4 = wait4\n"
        "for name in ('kill', 'killpg', 'waitpid'):\n"
        "    setattr(os, name, recorded(name))\n"
        "from pathlib import Path\n"
        "try:\n"
        "    run.run_limited(['sh', '-c', 'exit 3'], cwd=Path('.'), env=dict(os.environ),\n"
        f"                    timeout=30, max_rss_bytes=2**30, log=Path({str(log)!r}))\n"
        "    print('returned', flush=True)\n"
        "except SystemExit as exc:\n"
        "    print('exit', exc.code, 'reaped', len(reaped), 'after', after, flush=True)\n")
    assert done.stdout.split() == ["exit", "143", "reaped", "1", "after", "[]"], (
        done.stdout + done.stderr)


@pytest.mark.parametrize("where", ["default_behind_a_link", "relative"])
def test_the_work_directory_may_be_relative_or_behind_a_link(toy, tmp_path, where):
    real, link = tmp_path / "real", tmp_path / "link"
    real.mkdir()
    link.symlink_to(real)
    command = [sys.executable, str(ROOT / "tools" / "parent_gate.py"), "judge",
               "--repo", str(toy.root), "--head", toy.cases["witness"].head]
    if where == "relative":
        command += ["--work-dir", "rel"]
    proc = subprocess.run(command, cwd=link, env={**os.environ, "TMPDIR": str(link)},
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert " accepted (pass)" in proc.stdout


def test_the_runs_inherit_no_pytest_python_or_git_setting(gate, tmp_path, monkeypatch):
    outer = tmp_path / "outer"
    tree_dir = outer / "trees" / "C"
    (tree_dir / "src" / "toypkg").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(outer)], env=_isolated_git_env(), check=True)
    inherited = {"PYTEST_ADDOPTS": "-x", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                 "PYTEST_DEBUG": "1", "PYTHONWARNINGS": "error", "PYTHONOPTIMIZE": "2",
                 "PYTHONDEVMODE": "1"}
    for key, value in {**inherited, "PYTHONPATH": str(tmp_path / "elsewhere")}.items():
        monkeypatch.setenv(key, value)
    layout = gate.tree.Layout(root=tree_dir, package="toypkg", source_root=Path("src"),
                              corpus=tree_dir)
    env = gate.run_env(tree_dir, layout, gate.Settings())
    assert not set(inherited) & set(env)
    assert env["PYTHONPATH"] == os.pathsep.join(
        str(entry) for entry in layout.path_entries(tree_dir))
    probe = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=tree_dir, env=env,
                           capture_output=True, text=True)
    assert probe.returncode != 0, probe.stdout


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
@pytest.mark.parametrize("flag", ["--time-limit", "--max-rss-gb"])
def test_a_limit_must_be_finite_and_positive(gate, tmp_path, capsys, flag, value):
    with pytest.raises(SystemExit) as info:
        gate.main(["judge", "--repo", str(tmp_path), "--head", "HEAD", flag, value])
    assert info.value.code == 2
    assert flag in capsys.readouterr().err


@pytest.mark.parametrize("text", ["{", "{}"], ids=["not_json", "no_fields"])
def test_a_malformed_record_is_a_tool_error(gate, tmp_path, capsys, text):
    record = tmp_path / "record.json"
    record.write_text(text)
    page = tmp_path / "page.rst"
    page.write_text("page\n")
    assert gate.main(["render", "--record", str(record), "--check", "--out", str(page)]) == 2
    assert "parent_gate: error:" in capsys.readouterr().err


# ---- Population, replay and record ----

def test_population_selects_fix_commits_that_touch_a_test_file(record_tools, toy):
    commits = record_tools.population(toy.root, f"{toy.refs['B0']}..{toy.refs['history']}")
    assert [c.sha for c in commits] == [toy.refs["in_1"], toy.refs["in_2"]]
    assert [c.touched for c in commits] == [("tests/test_area.py",), ("tests/test_constant.py",)]


def _log_rows(path: Path) -> list[list[str]]:
    return [line.split("\t") for line in path.read_text().splitlines() if line]


def test_the_replay_runs_two_at_a_time_resumes_and_writes_no_local_path(gate, toy, tmp_path):
    out, record_path = tmp_path / "out", tmp_path / "record.json"
    args = ["replay", "--repo", str(toy.root),
            "--range", f"{toy.refs['B0']}..{toy.refs['history']}",
            "--out", str(out), "--jobs", "2", "--record", str(record_path)]
    assert gate.main(args) == 0
    record = json.loads(record_path.read_text())
    counted = record["counted"]
    assert [c["sha"] for c in counted["commits"]] == [toy.refs["in_1"], toy.refs["in_2"]]
    assert [c["verdict"] for c in counted["commits"]] == ["accepted", "passes_on_parent"]
    assert counted["population"] == 2
    assert counted["denominator"] == 2
    assert counted["rate"] == 0.5
    assert record["earlier"] is None
    text = record_path.read_text()
    assert str(tmp_path) not in text and str(tmp_path.resolve()) not in text
    assert str(Path.home()) not in text

    rows = _log_rows(out / "replay.log")
    runs = [row for row in rows if row[0] == "run"]
    assert len(runs) == 2
    (start_a, end_a), (start_b, end_b) = [(row[1], row[2]) for row in runs]
    assert start_a < end_b and start_b < end_a

    files = sorted((out / "commits").glob("*.json"))
    before = {f: (f.read_bytes(), f.stat().st_mtime_ns) for f in files}
    assert gate.main(args) == 0
    assert {f: (f.read_bytes(), f.stat().st_mtime_ns) for f in files} == before
    later = _log_rows(out / "replay.log")[len(rows):]
    assert sorted(row[0] for row in later) == ["reused", "reused"]


def _full_commit(sha: str, subject: str, verdict: str, witness: str) -> dict:
    """A per-commit judgement as the replay writes it, made in memory."""
    counts = {"passed": 1, "failed": 0, "error": 0, "skipped": 0, "not_run": 0, "absent": 0}
    failed = {**counts, "passed": 0, "failed": 1}
    run = {"seconds": 1.0, "peak_rss_mb": 50.0, "per_file": {"tests/test_x.py": counts}}
    return {
        "format": 1, "head": sha, "subject": subject, "verdict": verdict,
        "side": None, "cause": None, "reason": "made in memory",
        "touched": [{"path": "tests/test_x.py", "status": "A", "old_path": None}],
        "runs": {"head": run, "parent_own": None,
                 "overlay": {**run, "per_file": {"tests/test_x.py": failed}}},
        "tests": [{"id": witness, "head": "passed", "parent_own": None, "overlay": "failed",
                   "phase": "call", "kind": "assertion", "exception": "AssertionError",
                   "message": "", "existing": False}],
        "witnesses": [witness],
    }


def _record_of(gate, record_tools, entries, settings=None) -> dict:
    counted = record_tools.section("a" * 40 + ".." + "b" * 40, entries, counted=True)
    settings = settings if settings is not None else gate.Settings()
    return record_tools.build_record(gate, counted=counted, earlier=None,
                                     settings=dataclasses.asdict(settings),
                                     environment={"python": "3"})


def test_the_record_writer_refuses_a_development_name(gate, record_tools, tmp_path):
    task_like = "T" + "-" + "1" + "23"
    witness = f"tests/test_x.py::test_y[{task_like}]"
    entry = record_tools.commit_entry(
        _full_commit("c" * 40, "fix: x", "accepted", witness), gate.DECISION)
    path = tmp_path / "record.json"
    assert record_tools.write_record(_record_of(gate, record_tools, [entry]), path) == 2
    assert not path.exists()

    entry = record_tools.commit_entry(
        _full_commit("c" * 40, "fix: x", "accepted", "tests/test_x.py::test_y"), gate.DECISION)
    entry = {**entry, "reason": "made \u2014 in memory"}
    assert record_tools.write_record(_record_of(gate, record_tools, [entry]), path) == 2
    assert not path.exists()

    subject = "fix: one \u2014 two" + " " + "-" * 2 + " three"
    entry = record_tools.commit_entry(
        _full_commit("d" * 40, subject, "accepted", "tests/test_x.py::test_y"), gate.DECISION)
    assert record_tools.write_record(_record_of(gate, record_tools, [entry]), path) == 0
    written = json.loads(path.read_text())
    assert written["counted"]["commits"][0]["subject"] == "fix: one, two, three"
    assert "\u2014" not in path.read_text()


@pytest.mark.parametrize("workers, words", [
    (0, "each serial in the replay below,"),
    (2, "each with 2 pytest-xdist workers in the replay below,")])
def test_the_page_states_how_the_replay_ran(gate, record_tools, workers, words):
    entry = record_tools.commit_entry(
        _full_commit("a" * 40, "fix: x", "accepted", "tests/test_x.py::test_y"), gate.DECISION)
    record = _record_of(gate, record_tools, [entry],
                        settings=gate.Settings(pytest_workers=workers))
    page = " ".join(record_tools.render_page(record).split())
    assert words in page
    assert "``--pytest-workers 2``" in page


def test_a_dependency_trigger_applies_only_to_passes_on_parent(gate, record_tools):
    trigger = record_tools.DependencyTrigger(
        dependency="toylib", minimum="2.0", below=None,
        changelog_heading="### Fixed: toylib 2.0")
    passing = record_tools.commit_entry(
        _full_commit("e" * 40, "fix: x", "passes_on_parent", "tests/test_x.py::t"), gate.DECISION)
    classed = record_tools.apply_trigger(passing, trigger)
    assert (classed["verdict"], classed["side"], classed["cause"]) == (
        "environment", "dependency", "failure")
    assert classed["gate_verdict"] == "passes_on_parent"
    assert classed["decision"] == "undetermined"
    assert passing["verdict"] == "passes_on_parent"
    accepted = record_tools.commit_entry(
        _full_commit("f" * 40, "fix: x", "accepted", "tests/test_x.py::t"), gate.DECISION)
    kept = record_tools.apply_trigger(accepted, trigger)
    assert kept["verdict"] == "accepted"
    assert kept["trigger_unused"] is True


def _function_docstring_line(source: str, name: str) -> int:
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node.body[0].lineno
    raise AssertionError(name)


#: Functions that decide a verdict, one per stage, and their files.
RULE_FUNCTIONS = (("parent_gate.py", "decide"), ("parent_gate.py", "_overlay_verdict"),
                  ("parent_gate.py", "run_fault"), ("parent_gate_report.py", "failure_kind"),
                  ("parent_gate_report.py", "missing_name"),
                  ("parent_gate_report.py", "read_report"),
                  ("parent_gate_run.py", "run_limited"),
                  ("parent_gate_run.py", "_select_reader"))


def _add_a_statement(source: str, name: str) -> str:
    """``source`` with a ``pass`` statement added at the end of function ``name``."""
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines(keepends=True)
    last = lines[node.body[-1].lineno - 1]
    indent = last[: len(last) - len(last.lstrip())]
    return "".join(lines[: node.end_lineno] + [f"{indent}pass\n"] + lines[node.end_lineno:])


def test_the_rules_digest_follows_the_rules_not_their_docstrings(gate, tmp_path):
    def copy_tools(name: str) -> Path:
        dest = tmp_path / name
        dest.mkdir()
        for tool in ("parent_gate.py", "parent_gate_report.py", "parent_gate_record.py",
                     "parent_gate_page.py", "parent_gate_run.py", "mutation_tree.py"):
            shutil.copy(ROOT / "tools" / tool, dest / tool)
        return dest

    def edit(tools: Path, tool: str, change) -> None:
        path = tools / tool
        path.write_text(change(path.read_text()))

    original = gate.rules_digest()
    assert gate.rules_digest(copy_tools("same")) == original

    docs = copy_tools("docs")

    def docstring_and_comment(text: str) -> str:
        lines = text.splitlines(keepends=True)
        at = _function_docstring_line(text, "decide") - 1
        indent = lines[at][: len(lines[at]) - len(lines[at].lstrip())]
        lines[at] = lines[at].replace('"""', '"""Edited.  ', 1)
        lines.insert(at, f"{indent}# a comment that decides nothing\n")
        return "".join(lines)

    edit(docs, "parent_gate.py", docstring_and_comment)
    assert gate.rules_digest(docs) == original

    changes = {
        "assertion_classes": ("parent_gate_report.py",
                              'ASSERTION_CLASSES = ("AssertionError", "Failed")',
                              'ASSERTION_CLASSES = ("AssertionError", "Failed", "ValueError")'),
        "run_flag": ("parent_gate.py", '"--tb=short"', '"--tb=long"'),
        "pytest_flags": ("mutation_tree.py", '"-o", "junit_family=xunit1")',
                         '"-o", "junit_family=xunit2")'),
    }
    for name, (tool, old, new) in changes.items():
        tools = copy_tools(name)
        assert old in (tools / tool).read_text(), name
        edit(tools, tool, lambda text, old=old, new=new: text.replace(old, new, 1))
        assert gate.rules_digest(tools) != original, name

    for tool, name in RULE_FUNCTIONS:
        tools = copy_tools(f"code_{name}")
        edit(tools, tool, lambda text, name=name: _add_a_statement(text, name))
        assert gate.rules_digest(tools) != original, name

    missing = copy_tools("missing")

    def drop_tree_env(text: str) -> str:
        node = next(n for n in ast.parse(text).body
                    if isinstance(n, ast.FunctionDef) and n.name == "tree_env")
        lines = text.splitlines(keepends=True)
        return "".join(lines[: node.lineno - 1] + lines[node.end_lineno:])

    edit(missing, "mutation_tree.py", drop_tree_env)
    with pytest.raises(gate.GateError, match="tree_env"):
        gate.rules_digest(missing)


def test_the_digest_keeps_code_beside_a_docstring_and_every_line_of_a_string(gate, tmp_path):
    def text_of(source: str) -> str:
        path = tmp_path / "rules.py"
        path.write_text(source, encoding="utf-8")
        return gate.normalised_source(path, ["f", "X"])

    base = text_of('def f():\n    """Doc μ."""; return 1\n\nX = """a\n"""\n')
    # Docstring text, a comment-only line and a blank line outside strings: same.
    assert text_of('def f():\n    """Other νν."""; return 1\n\n'
                   'X = """a\n"""\n') == base
    assert text_of('def f():\n    # a comment\n\n    """Doc μ."""; return 1\n\n'
                   'X = """a\n"""\n') == base
    # Code on the docstring's line, and blank or '#' lines inside a string: different.
    assert text_of('def f():\n    """Doc μ."""; return 2\n\nX = """a\n"""\n') != base
    assert text_of('def f():\n    """Doc μ."""; return 1\n\nX = """a\n\n"""\n') != base
    assert text_of('def f():\n    """Doc μ."""; return 1\n\nX = """a\n# b\n"""\n') != base


#: The module-level statement kinds a rule file may hold: imports, definitions,
#: assignments to names and the ``__main__`` guard.  The digest reads each.
DIGESTED_KINDS = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign,
                  ast.AnnAssign, ast.If)


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
        and isinstance(node.value.value, str)


def _binds_out_of_sight(node: ast.stmt) -> bool:
    """Whether ``node`` binds or changes something the digest does not key it by.

    An assignment to an attribute, a subscript or a starred name, a ``*``
    import, or a walrus evaluated at module level.
    """
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        targets = []
    for target in targets:
        if any(not isinstance(part, (ast.Name, ast.Tuple, ast.List, ast.Load, ast.Store))
               for part in ast.walk(target)):
            return True
    if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
        return True
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        evaluated = [*node.decorator_list, *node.args.defaults,
                     *(d for d in node.args.kw_defaults if d is not None)]
    elif isinstance(node, ast.ClassDef):
        evaluated = [*node.decorator_list, *node.bases, *node.keywords]
    else:
        evaluated = [node]
    return any(isinstance(inner, ast.NamedExpr) for part in evaluated
               for inner in ast.walk(part))


def _statements_the_digest_does_not_read(gate, source: str) -> list[int]:
    """Lines of the module-level statements of ``source`` that ``_definitions`` keys by
    no name, or that bind out of its sight; the module docstring is exempt."""
    module = ast.parse(source)
    keyed = {id(node) for node in gate._definitions(module).values()}
    unread = []
    for index, node in enumerate(module.body):
        if index == 0 and _is_docstring(node):
            continue
        if not isinstance(node, DIGESTED_KINDS) or id(node) not in keyed \
                or _binds_out_of_sight(node):
            unread.append(node.lineno)
    return unread


def test_every_module_level_statement_of_a_rule_file_is_one_the_digest_reads(gate):
    unread = [f"{name}:{line}" for name in gate.RULE_FILES
              for line in _statements_the_digest_does_not_read(
                  gate, (ROOT / "tools" / name).read_text(encoding="utf-8"))]
    assert unread == []
    module = "\n".join(['"""Doc."""', "import os", "X = 1", "def f():", "    pass",
                        "if __name__ == '__main__':", "    f()"])
    assert _statements_the_digest_does_not_read(gate, module) == []
    hidden = ["os.sep = '/'", "X[0] = 1", "a, *X = 1, 2", "*X, = (1,)", "X += 1",
              "X.y: int = 1", "setattr(os, 'sep', '/')", "globals()['X'] = 2",
              "from os import *", "Y = (Z := 1)", "def g(x=(W := 1)):\n    pass",
              "del X", "if X:\n    Y = 2", "import os"]
    for statement in hidden:
        assert _statements_the_digest_does_not_read(gate, f"{module}\n{statement}\n"), \
            statement


def _changed_statement(node: ast.stmt) -> ast.stmt:
    """A copy of ``node``, of the same kind, whose text and effect differ."""
    node = copy.deepcopy(node)
    if isinstance(node, ast.Import):
        node.names.append(ast.alias("json"))
    elif isinstance(node, ast.ImportFrom):
        node.names.append(ast.alias(node.names[0].name, "_changed"))
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        node.value = ast.Subscript(ast.Tuple([node.value], ast.Load()), ast.Constant(1),
                                   ast.Load())
    else:
        node.body.append(ast.Pass())
    return ast.fix_missing_locations(node)


@pytest.mark.parametrize("kind", DIGESTED_KINDS, ids=lambda kind: kind.__name__)
def test_a_change_to_a_statement_of_each_accepted_kind_changes_the_digest(gate, tmp_path,
                                                                         kind):
    tools = tmp_path / "tools"
    tools.mkdir()
    for tool in (*gate.RULE_FILES, "parent_gate_record.py", "parent_gate_page.py",
                 "mutation_tree.py"):
        shutil.copy(ROOT / "tools" / tool, tools / tool)
    for name in gate.RULE_FILES:
        source = (tools / name).read_text(encoding="utf-8")
        module = ast.parse(source)
        digested = {id(node) for key, node in gate._definitions(module).items()
                    if key not in gate.NOT_RULES}
        chosen = [node for node in module.body
                  if isinstance(node, kind) and id(node) in digested
                  and getattr(node, "module", None) != "__future__"]
        if chosen:
            break
    else:
        pytest.fail(f"no {kind.__name__} statement of a rule file is digested")
    node = chosen[0]
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    lines = source.splitlines(keepends=True)
    changed = ast.unparse(_changed_statement(node)) + "\n"
    (tools / name).write_text("".join(lines[: start - 1] + [changed] + lines[node.end_lineno:]),
                              encoding="utf-8")
    assert gate.rules_digest(tools) != gate.rules_digest(), f"{name}:{node.lineno}"


def test_the_page_names_the_memory_measure_in_a_sentence_of_its_own(gate, record_tools):
    entry = record_tools.commit_entry(
        _full_commit("a" * 40, "fix: x", "accepted", "tests/test_x.py::test_y"), gate.DECISION)
    record = {**_record_of(gate, record_tools, [entry]),
              "environment": gate.environment_versions()}
    page = " ".join(record_tools.render_page(record).split())
    assert "with memory" not in page
    assert (f"The memory limit of a run applies to the summed {gate.MEMORY_MEASURE} of "
            "its processes.") in page
    assert f"numpy {record['environment']['numpy']}" in page


def test_an_edit_that_keeps_the_base_neither_starts_nor_cancels_a_gate_run():
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "parent-gate.yml").read_text(encoding="utf-8"))
    assert "concurrency" not in workflow
    job = workflow["jobs"]["parent-gate"]
    condition = "github.event.action != 'edited' || github.event.changes.base"
    assert job["if"] == condition
    group = job["concurrency"]["group"]
    assert f"(({condition}) && 'judge' || github.run_id)" in group
    assert job["concurrency"]["cancel-in-progress"] is True


# ---- The committed record ----

#: Non-merge commits of v0.4.2..<recorded end> whose subject starts with
#: fix and that add or modify a tests/test_*.py, counted with git log by a
#: script separate from the tool.
COUNTED_POPULATION = 86
EARLIER_POPULATION = 27
#: The rate the replay must reach before the workflow may block a pull request.
BLOCKING_RATE = 0.90
WORKFLOW = ROOT / ".github" / "workflows" / "parent-gate.yml"

#: The decision of each verdict: the tool's table, restated, with the
#: owner's rulings on crash_only (rejected) and no_code_change (passes).
DECISIONS = {
    "no_code_change": "pass", "no_touched_tests": "reject",
    "environment": "undetermined", "no_evidence": "reject",
    "overlay_not_run": "reject", "accepted": "pass", "crash_only": "reject",
    "resource_only": "reject", "new_api_only": "reject",
    "passes_on_parent": "reject",
}


def test_the_decisions_are_the_owner_rulings(gate):
    assert gate.DECISION == DECISIONS


def _committed(path: Path) -> Path:
    if not path.is_file():
        pytest.fail(
            f"{path.relative_to(ROOT)} is missing: write it with "
            "`python tools/parent_gate.py replay --range v0.4.2..<sha> "
            "--earlier-range v0.4.2 --out <dir> --record "
            "docs/verification/parent_gate_replay.json`, then "
            "`python tools/parent_gate.py render --record "
            "docs/verification/parent_gate_replay.json --out "
            "docs/verification/parent_gate.rst`", pytrace=False)
    return path


def _record() -> dict:
    return json.loads(_committed(RECORD).read_text())


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _figures(commits: list[dict]) -> dict:
    by = {}
    for commit in commits:
        by[commit["verdict"]] = by.get(commit["verdict"], 0) + 1
    e = by.get("environment", 0)
    d = len(commits) - e
    a, k, c = by.get("accepted", 0), by.get("no_code_change", 0), by.get("crash_only", 0)
    o = sum(1 for commit in commits if commit.get("side") == "dependency")
    passed = sum(1 for commit in commits
                 if commit["verdict"] != "environment" and DECISIONS[commit["verdict"]] == "pass")
    return {
        "by_verdict": by, "environment": e, "denominator": d, "accepted": passed,
        "rate": _ratio(passed, d),
        "figures": {
            "crash_only rejected, no_code_change passes": _ratio(a + k, d),
            "crash_only passes, no_code_change passes": _ratio(a + k + c, d),
            "crash_only rejected, no_code_change left out": _ratio(a, d - k),
            "crash_only passes, no_code_change left out": _ratio(a + c, d - k),
            "dependency triggers counted as rejections": _ratio(a + k, d + o),
        },
    }


def test_the_replay_figures_follow_from_its_commits_and_bound_the_workflow(gate):
    record = _record()
    counted = record["counted"]
    commits = counted["commits"]
    assert len(commits) == counted["population"] == COUNTED_POPULATION
    start, end = counted["range"].split("..")
    assert re.fullmatch("[0-9a-f]{40}", start) and re.fullmatch("[0-9a-f]{40}", end)
    for commit in commits:
        assert commit["verdict"] in DECISIONS, commit["sha"]
        assert commit["decision"] == gate.DECISION[commit["verdict"]], commit["sha"]
        assert commit["decision"] == DECISIONS[commit["verdict"]], commit["sha"]
        if commit["verdict"] == "accepted":
            assert any(w["kind"] == "assertion" for w in commit["witnesses"]), commit["sha"]
        if commit["verdict"] == "environment":
            assert commit["side"] in ("head", "parent", "dependency"), commit["sha"]
            assert commit["cause"] in ("failure", "timeout", "memory"), commit["sha"]
    mine = _figures(commits)
    assert counted["environment"]["total"] == mine["environment"]
    assert counted["denominator"] == mine["denominator"]
    assert counted["by_verdict"] == mine["by_verdict"]
    assert counted["accepted"] == mine["accepted"]
    assert counted["rate"] == pytest.approx(mine["rate"], rel=1e-12, abs=0.0)
    for label, value in mine["figures"].items():
        assert counted["figures"][label] == (
            None if value is None else pytest.approx(value, rel=1e-12, abs=0.0)), label
    print(json.dumps({"rate": mine["rate"], **mine["figures"]}, indent=1))
    advisory = "continue-on-error: true" in WORKFLOW.read_text()
    assert advisory or mine["rate"] >= BLOCKING_RATE


def _entry(section: dict, short: str) -> dict:
    matches = [c for c in section["commits"] if c["sha"].startswith(short)]
    assert len(matches) == 1, short
    return matches[0]


def test_each_dependency_trigger_names_a_version_the_replay_lacks(record_tools):
    record = _record()
    changelog = CHANGELOG.read_text()
    sections = [s for s in (record["counted"], record["earlier"]) if s]
    for commit in (c for s in sections for c in s["commits"]):
        assert not commit.get("trigger_unused"), commit["sha"]
        if commit.get("side") != "dependency":
            continue
        trigger = record_tools.DEPENDENCY_TRIGGERS[commit["sha"]]
        assert commit["gate_verdict"] == "passes_on_parent", commit["sha"]
        version = record["environment"][trigger.dependency]
        assert not record_tools.version_in_range(version, trigger.minimum, trigger.below), (
            commit["sha"], version)
        assert trigger.changelog_heading in changelog, commit["sha"]


def _not_passing_on_parent(commit: dict, path: str) -> tuple[int, int]:
    counts = commit["per_file"][path]
    head = counts["head"]["passed"]
    return head, head - counts["overlay"]["passed"]


def test_the_replay_matches_the_failure_counts_the_changelog_states():
    record = _record()
    group = _entry(record["counted"], "ae0dbf9")
    assert _not_passing_on_parent(group, "tests/test_group_position_conflict.py") == (33, 20)
    rate = _entry(record["counted"], "e0014d6")
    names = {w["id"].split("::")[-1] for w in rate["witnesses"]}
    assert {"test_T5_default_cache_grid_gives_R_to_1e_3",
            "test_T5b_gamma_spline_error_falls_faster_than_h_squared"} <= names
    regressions = _entry(record["earlier"], "2fc0865")
    assert _not_passing_on_parent(regressions, "tests/test_msr_numerics_regressions.py") == (17, 15)


def _fix_commits(rev_range: str) -> list[str]:
    """Shas of the population of ``rev_range``, read from git log here."""
    out = subprocess.run(
        ["git", "log", "--no-merges", "--reverse", "--format=%x00%H%x1f%s",
         "--name-status", "-M", rev_range],
        cwd=ROOT, env=_isolated_git_env(), check=True, capture_output=True,
        text=True).stdout
    shas = []
    for block in out.split("\x00")[1:]:
        head, _, rest = block.partition("\n")
        sha, subject = head.split("\x1f", 1)
        paths = [(line.split("\t")[0], line.split("\t")[-1])
                 for line in rest.splitlines() if "\t" in line]
        if subject.startswith("fix") and any(
                status[0] in "AMR" and re.match(r"^tests/test_[^/]*\.py$", path)
                for status, path in paths):
            shas.append(sha)
    return shas


def test_the_replay_names_the_fix_commits_of_its_range():
    record = _record()
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    counted, earlier = record["counted"], record["earlier"]
    start, end = counted["range"].split("..")
    for rev in (start, end, earlier["range"]):
        probe = subprocess.run(["git", "cat-file", "-e", f"{rev}^{{commit}}"], cwd=ROOT,
                               env=_isolated_git_env(), capture_output=True)
        if probe.returncode != 0:
            pytest.skip(f"{rev} is not in this checkout (a shallow clone)")
    assert [c["sha"] for c in counted["commits"]] == _fix_commits(counted["range"])
    assert [c["sha"] for c in earlier["commits"]] == _fix_commits(earlier["range"])
    assert len(earlier["commits"]) == EARLIER_POPULATION


def test_the_replay_was_made_by_the_current_rules(gate):
    record = _record()
    assert record["rules_digest"] == gate.rules_digest()
    assert record["settings"]["markers"] == "not slow"
    assert record["decisions"] == gate.DECISION


def test_the_replay_page_is_current(record_tools):
    record = _record()
    page = _committed(PAGE).read_text()
    assert page == record_tools.render_page(record), (
        "docs/verification/parent_gate.rst is stale: run `python tools/parent_gate.py "
        "render --record docs/verification/parent_gate_replay.json --out "
        "docs/verification/parent_gate.rst`")
