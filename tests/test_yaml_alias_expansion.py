"""A file whose aliases expand past the limit is refused where it is read.

A list that holds the one before it twice doubles at each level.  YAML keeps
one object, so a key nothing reads passes ``validate``, and the first walk
that copies or prints the value does not end: ``_plain`` for
``--print-resolved``, ``repr`` for the cache key of ``run``.

AE1 -- each place and shape is refused at once, with a dotted path, in one
       line, at any depth.
AE2 -- both sides of the limit, and what stays legal: a value written out,
       merged keys, aliases that repeat little.
AE3 -- the count is the size of the tree, compared with a recursive walk on
       random files.
       A file nested past the limit is refused too, written out or by alias,
       by the config loader; ``compare-resources`` only reads such a file.
AE4 -- the commands: ``validate``, ``--print-resolved``, ``run`` and
       ``compare-resources``.
"""

from __future__ import annotations

import random
import signal
import time
from pathlib import Path

import pytest
import yaml

from sft_wick.workflow import config as cf
from sft_wick.workflow.cli import main
from sft_wick.workflow.config import ConfigError, _refuse_expansion, _refuse_nesting
from sft_wick.workflow.resource_compare import _safe_mapping

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = (ROOT / "examples" / "quickstart.yaml").read_text()


@pytest.fixture
def hang_fails():
    """A walk of the expanded value would not end in the life of the test.
    The failure is a ``BaseException``, which no handler in the parser
    catches."""

    def expired(signum, frame):
        pytest.fail("still expanding")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(10)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


def _diamond(depth: int) -> str:
    """A flow list of ``depth`` levels, each holding the last one twice."""
    text = "&l0 [1.0]"
    for i in range(1, depth + 1):
        text = f"&l{i} [{text}, *l{i - 1}]"
    return text


def _mapping_diamond(depth: int) -> str:
    text = "&l0 {a: 1}"
    for i in range(1, depth + 1):
        text = f"&l{i} {{a: {text}, b: *l{i - 1}}}"
    return text


def _in_quickstart(key: str, value: str) -> str:
    """Quick start with ``key: value`` added at the end of a block."""
    block, name = key.rsplit(".", 1)
    data = yaml.safe_load(QUICKSTART)
    node = data
    for part in block.split("."):
        node = node[int(part)] if part.isdigit() else node[part]
    node[name] = "@@"
    text = yaml.safe_dump(data, sort_keys=False)
    return text.replace("'@@'", value)


def _write(tmp_path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def _refused(path: Path) -> ConfigError:
    with pytest.raises(ConfigError) as info:
        cf.load_workflow_config(path)
    return info.value


PLACES = {
    "unread_list_at_output": ("output.0.x", _diamond(40), "output[0].x"),
    "unread_mapping_at_output": ("output.0.hue", _mapping_diamond(40),
                                 "output[0].hue"),
    "facet_col": ("output.0.facet_col", _diamond(40), "output[0].facet_col"),
    "linear_cache_key": ("system.linear.t_max_cache", _diamond(40),
                         "system.linear.t_max_cache"),
    "depth_of_the_limit": ("output.0.x", _diamond(20), "output[0].x"),
    "very_deep": ("output.0.x", _diamond(300), "output[0].x"),
}


@pytest.mark.parametrize("key, value, path", PLACES.values(), ids=PLACES)
def test_AE1_a_diamond_is_refused_at_once(tmp_path, hang_fails, key, value, path):
    started = time.perf_counter()
    error = _refused(_write(tmp_path, _in_quickstart(key, value)))
    assert time.perf_counter() - started < 2.0
    assert (error.code, error.path) == ("invalid_value", path)
    assert "\n" not in str(error)
    assert len(str(error)) < 160


def test_AE1_no_one_key_holds_it(tmp_path, hang_fails):
    """Many keys share one value: the place is the block that holds them."""
    data = yaml.safe_load(QUICKSTART)
    data["output"][0]["@"] = "@@"
    text = yaml.safe_dump(data, sort_keys=False)
    many = ", ".join(["*big"] * 200)
    big = ", ".join(["1.0"] * 10_000)
    value = f"[[&big [{big}]], {many}]"
    error = _refused(_write(tmp_path, text.replace("'@': '@@'", f"x: {value}")))
    assert error.code == "invalid_value"
    assert error.path == "output[0].x"


def test_AE1_keys_that_share_alike_name_the_top(hang_fails):
    with pytest.raises(ConfigError) as info:
        _refuse_expansion(_flat(2000, 2000))
    assert info.value.path == "config"


def test_AE1_a_vertex_list_sharing_one_tensor(tmp_path, hang_fails):
    """Each tensor is under the limit of one tensor; the file is not."""
    row = ", ".join(["1.0"] * 100)
    table = ", ".join([f"[{row}]"] * 100)
    cube = f"&cube [[{table}]]"
    vertices = [f"{{name: F{i}, coupling: {'*cube' if i else cube}}}"
                for i in range(200)]
    data = yaml.safe_load(QUICKSTART)
    data["system"]["vertices"] = "@@"
    text = yaml.safe_dump(data, sort_keys=False)
    text = text.replace("'@@'", "[" + ", ".join(vertices) + "]")
    error = _refused(_write(tmp_path, text))
    assert error.path == "system.vertices"


def _flat(items: int, uses: int) -> dict:
    """``uses`` keys share one list of ``items`` numbers."""
    shared = [1.0] * items
    return {f"k{i}": shared for i in range(uses)}


@pytest.mark.parametrize("items, uses", [(100, 11), (1000, 2), (10, 101)],
                         ids=["rows", "pair", "many_keys"])
def test_AE2_both_sides_of_a_limit(monkeypatch, items, uses):
    monkeypatch.setattr(cf, "_MAX_ALIAS_VALUES", (uses - 2) * items)
    _refuse_expansion(_flat(items, uses - 1))
    with pytest.raises(ConfigError):
        _refuse_expansion(_flat(items, uses))


def test_AE2_the_limit_itself():
    assert cf._MAX_ALIAS_VALUES == 10**6
    _refuse_expansion(_flat(1000, 1001))
    with pytest.raises(ConfigError):
        _refuse_expansion(_flat(1001, 1001))


def test_AE2_a_value_written_out_is_not_counted():
    """The file holds it, so reading it cost what any walk costs."""
    _refuse_expansion({"a": [[1.0] * 1000 for _ in range(5000)]})


def test_AE2_a_merge_of_numbers_is_not_counted():
    base = {f"k{i}": i for i in range(1000)}
    _refuse_expansion({"uses": [dict(base) for _ in range(2000)]})


def test_AE2_a_repeated_number_is_not_counted():
    one = 1.0
    _refuse_expansion({"a": [one] * 5_000_000})


@pytest.mark.parametrize("path", sorted(ROOT.glob("examples/**/*.yaml")),
                         ids=lambda path: path.stem)
def test_AE2_no_shipped_config_comes_near_it(path):
    assert cf._values_added_by_aliases(cf._load_yaml(path.read_text())) < 10**4


def _nested(depth: int) -> str:
    return "[" * depth + "1.0" + "]" * depth


def _alias_chain(depth: int) -> str:
    """Each list holds the one before it once, so little is repeated."""
    links = ["&l0 [1.0]"] + [f"&l{i} [*l{i - 1}]" for i in range(1, depth + 1)]
    return "[" + ", ".join(links) + "]"


@pytest.mark.parametrize("text", [_nested(400), _alias_chain(500), _alias_chain(1400)],
                         ids=["written_out", "chain", "long_chain"])
@pytest.mark.parametrize("extra", [[], ["--print-resolved"]],
                         ids=["validate", "print_resolved"])
def test_AE2_nesting_is_refused_in_one_line(tmp_path, capsys, text, extra):
    path = _write(tmp_path, _in_quickstart("output.0.x", text))
    assert main(["validate", str(path), *extra]) == 2
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1
    assert "output[0].x" in lines[0]
    assert "nested more than 100 levels" in lines[0]


def test_AE2_both_sides_of_the_nesting_limit():
    assert cf._MAX_NESTING == 100
    value: list = [1.0]
    for _ in range(cf._MAX_NESTING - 1):
        value = [value]
    _refuse_nesting(value)
    with pytest.raises(ConfigError):
        _refuse_nesting([value])


def test_AE2_a_tensor_of_the_most_axes_is_not_too_deep():
    tensor: list = [1.0]
    for _ in range(cf._MAX_TENSOR_AXES - 1):
        tensor = [tensor]
    _refuse_nesting({"system": {"vertices": [{"coupling": tensor}]}})


def test_AE2_a_nesting_of_mappings_names_its_end():
    value: dict = {"a": 1}
    for _ in range(cf._MAX_NESTING):
        value = {"a": value}
    with pytest.raises(ConfigError) as info:
        _refuse_nesting(value)
    assert info.value.path == ".".join(["a"] * cf._MAX_NESTING)


def _walked(value) -> int:
    if isinstance(value, dict):
        return 1 + sum(_walked(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return 1 + sum(_walked(v) for v in value)
    return 1


def _written(value) -> int:
    seen, pending, total = set(), [value], 1
    while pending:
        item = pending.pop()
        if isinstance(item, (dict, list, tuple)) and id(item) not in seen:
            seen.add(id(item))
            items = item.values() if isinstance(item, dict) else item
            total += len(items)
            pending.extend(items)
    return total


def _random_value(rng: random.Random) -> dict:
    """Containers holding containers built before them, shared at random."""
    made: list = [[1.0], {"a": 2}]
    for _ in range(rng.randrange(1, 14)):
        parts = [rng.choice(made) if rng.random() < 0.7 else 3.0
                 for _ in range(rng.randrange(0, 4))]
        kind = rng.random()
        if kind < 0.5:
            made.append(parts)
        elif kind < 0.7:
            made.append(tuple(parts))
        else:
            made.append({f"k{i}": p for i, p in enumerate(parts)})
    return {"top": rng.choice(made), "other": rng.choice(made)}


def test_AE3_the_count_is_the_tree_less_what_is_written():
    rng = random.Random(20261031)
    repeated = 0
    for _ in range(3000):
        value = _random_value(rng)
        added = cf._values_added_by_aliases(value)
        assert added == _walked(value) - _written(value)
        repeated += added > 0
    assert repeated > 1000


def test_AE3_the_count_does_not_recurse():
    value: list = [1.0]
    for _ in range(100_000):
        value = [value]
    assert cf._values_added_by_aliases({"a": value}) == 0
    with pytest.raises(ConfigError):
        _refuse_nesting({"a": value})


@pytest.mark.parametrize(
    "extra", [[], ["--print-resolved"]], ids=["validate", "print_resolved"])
def test_AE4_validate_prints_one_line(tmp_path, capsys, hang_fails, extra):
    path = _write(tmp_path, _in_quickstart("output.0.x", _diamond(40)))
    assert main(["validate", str(path), *extra]) == 2
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("sft-wick: config error [invalid_value] output[0].x:")


@pytest.mark.parametrize("extra", [["--dry-run"], []], ids=["dry_run", "run"])
def test_AE4_run_prints_one_line(tmp_path, capsys, hang_fails, extra):
    path = _write(tmp_path, _in_quickstart("output.0.x", _diamond(40)))
    assert main(["run", str(path), *extra]) == 2
    assert len(capsys.readouterr().err.strip().splitlines()) == 1


def test_AE4_compare_resources_refuses_it(hang_fails):
    text = _in_quickstart("output.0.x", _diamond(40))
    with pytest.raises(ValueError, match=r"config: invalid YAML: output\[0\]\.x: "):
        _safe_mapping(text, "config")


def test_AE4_compare_resources_reads_a_long_chain():
    text = "s:\n- &a0 {n: 1}\n" + "".join(
        f"- &a{i} {{n: *a{i - 1}}}\n" for i in range(1, 700))
    assert _safe_mapping(text, "config")["s"][1]["n"] == {"n": 1}


def test_AE4_an_override_is_counted_with_the_file(tmp_path, hang_fails):
    path = _write(tmp_path, QUICKSTART)
    diamond: list = [1.0]
    for _ in range(40):
        diamond = [diamond, diamond]
    with pytest.raises(ConfigError) as info:
        cf.load_workflow_config(path, {"output[0].x": diamond})
    assert info.value.path == "output[0].x"
