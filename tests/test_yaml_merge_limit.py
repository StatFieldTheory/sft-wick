"""A file whose merges would stall PyYAML is refused before PyYAML lists them.

PyYAML copies a merged mapping's pairs into each mapping it lands in, so a
chain that merges the mapping before it twice doubles at each link, and 600
bytes take seconds inside ``yaml.safe_load``.

ML1 -- the shapes of growth below are refused at once, with the dotted path
       of the merge key, in one line, and naming that path allocates no more
       than the file holds.
ML2 -- both sides of the limit: a file at it reads as ``yaml.safe_load``
       reads it, one pair more is refused.
ML3 -- the count is the walk of ``flatten_mapping`` in numbers: it matches
       the lists PyYAML builds on every mapping, on hand-written files and on
       random ones, including a merge that reaches a mapping still being
       merged.  It stops where PyYAML stops, and costs no frame PyYAML does
       not.
ML4 -- the commands: ``validate`` and ``run`` print one line and exit with
       status 2, and ``compare-resources`` does not call it a repeated key.
"""

from __future__ import annotations

import random
import signal

import pytest
import yaml

from sft_wick.workflow import config as cf
from sft_wick.workflow.cli import main
from sft_wick.workflow.config import ConfigError, _load_yaml
from sft_wick.workflow.resource_compare import _safe_mapping


@pytest.fixture
def hang_fails():
    """Reading a growing merge would not end in the life of the test.  The
    failure is a ``BaseException``, which no handler in the parser catches.
    The alarm counts wall-clock time, and these tests take under a second
    alone but several times that on a CI runner shared by four workers, so
    it allows a minute."""

    def expired(signum, frame):
        pytest.fail("still merging")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(60)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


class _Merges:
    """The pairs PyYAML's merges add, watched at ``flatten_mapping``.

    Each mapping it flattens is kept with the pairs written in it, before
    the first flatten, and ``added`` is what the mappings hold now beyond
    those.  A mapping flattened again, or merged into itself, counts once.
    """

    def __init__(self, monkeypatch) -> None:
        written: dict = {}
        flatten = yaml.constructor.SafeConstructor.flatten_mapping

        def watched(constructor, node):
            if node not in written:
                written[node] = sum(
                    key.tag != cf._MERGE_TAG for key, _ in node.value)
            flatten(constructor, node)

        monkeypatch.setattr(
            yaml.constructor.SafeConstructor, "flatten_mapping", watched)
        self._written = written

    def added(self) -> int:
        return sum(len(node.value) - kept
                   for node, kept in self._written.items())


def _doubling(depth: int) -> str:
    """Each mapping merges the one before it twice."""
    lines = ["m0: &m0 {a: 1}"]
    for i in range(1, depth + 1):
        lines.append(f"m{i}: &m{i} {{<<: [*m{i - 1}, *m{i - 1}]}}")
    return "\n".join(lines) + "\n"


def _four_sources(depth: int) -> str:
    lines = ["m0: &m0 {a: 1}"]
    for i in range(1, depth + 1):
        lines.append(f"m{i}: &m{i} {{<<: [{', '.join([f'*m{i - 1}'] * 4)}]}}")
    return "\n".join(lines) + "\n"


def _two_keys(depth: int) -> str:
    """The doubling with two ``<<`` keys, a repeat refused only after the
    file is read."""
    lines = ["m0: &m0 {a: 1}"]
    for i in range(1, depth + 1):
        lines.append(f"m{i}: &m{i} {{<<: *m{i - 1}, <<: *m{i - 1}}}")
    return "\n".join(lines) + "\n"


def _back_references(keys: int) -> str:
    """Each merge key reaches the mapping it is in.  The merge resumes the
    keys still waiting, so the mapping doubles at each key, and a count that
    takes a mapping being merged at its own keys sees none of it."""
    merges = ", ".join("<<: {<<: *a}" for _ in range(keys))
    return f"top: &a {{k: 1, {merges}}}\n"


def _one_more_key_each(links: int) -> str:
    """No doubling: the pairs grow with the square of the links."""
    lines = ["m0: &m0 {a: 1}"]
    for i in range(1, links + 1):
        lines.append(f"m{i}: &m{i} {{<<: *m{i - 1}, k{i}: 1}}")
    return "\n".join(lines) + "\n"


SHAPES = {
    "two_sources": (_doubling(40), "m19.<<"),
    "four_sources": (_four_sources(40), "m10.<<"),
    "two_keys": (_two_keys(40), "m19.<<"),
    "back_references": (_back_references(40), "top.<<"),
    "merged_at_the_top": (_doubling(40) + "<<: *m40\n", "<<"),
    "in_a_list": (_doubling(40).replace("m40:", "l: [{<<: *m39}]\nm40:"),
                  "m19.<<"),
    "a_few_pairs_before_it": (
        "b: &b {a: 1, c: 2}\nu0: {<<: *b}\nu1: {<<: *b}\n" + _doubling(40),
        "m19.<<"),
    "square_of_the_links": (_one_more_key_each(3000), "m1414.<<"),
}


@pytest.mark.parametrize("text, path", SHAPES.values(), ids=SHAPES)
def test_ML1_growth_is_refused_at_once(hang_fails, monkeypatch, text, path):
    """PyYAML has added no more than the limit when the refusal comes.  A
    refusal after the file is read adds 4.5e6 pairs on the square shape, and
    one after the list that passes the limit adds that list too."""
    merges = _Merges(monkeypatch)
    with pytest.raises(ConfigError) as info:
        _load_yaml(text)
    assert merges.added() <= cf._MAX_MERGED_PAIRS
    assert (info.value.code, info.value.path) == ("invalid_value", path)
    assert "\n" not in str(info.value)
    assert len(str(info.value)) < 120


def test_ML1_a_chain_of_any_depth_costs_no_more(hang_fails, monkeypatch):
    """The file stops at the same link however far it goes on: PyYAML adds
    the same pairs before the refusal at every depth."""
    merges = _Merges(monkeypatch)
    seen = []
    for depth in (24, 400, 4000):
        before = merges.added()
        with pytest.raises(ConfigError) as info:
            _load_yaml(_doubling(depth))
        seen.append((merges.added() - before, info.value.path))
    assert seen == [seen[0]] * 3
    assert 0 < seen[0][0] <= cf._MAX_MERGED_PAIRS


def test_ML1_a_mapping_as_a_key_is_refused_by_yaml(hang_fails):
    """PyYAML hashes the key before it merges anything in it."""
    with pytest.raises(yaml.YAMLError, match="unhashable"):
        _load_yaml(_doubling(40) + "? {<<: *m40}\n: 1\n")


def test_ML1_a_refused_chain_names_the_link_that_passes_the_limit():
    """``m18`` brings the file to 2**19 - 2 pairs, ``m19`` to 2**20 - 2."""
    assert 2**19 - 2 <= cf._MAX_MERGED_PAIRS < 2**20 - 2
    assert _load_yaml(_doubling(18))["m18"] == {"a": 1}


def test_ML1_the_path_of_a_refusal_lists_each_source_once():
    """A mapping of many keys over one list named each source per key: 98 kB
    asked for 8 GB."""
    node = yaml.compose("x: &x {}\ns: &s [*x, *x]\nm: {<<: *s, <<: *s, k: 1}\n")
    mapping = node.value[2][1]
    children = cf._children(mapping, None, {mapping: list(mapping.value)})
    assert len(children) == 2


def test_ML1_a_refusal_after_a_repeated_list_names_the_right_place():
    text = ("e: &e {}\ns: &s [*e, *e, *e]\n"
            "m: {<<: *s, <<: *s, d: {x: 1}}\n" + _doubling(40))
    with pytest.raises(ConfigError) as info:
        _load_yaml(text)
    assert info.value.path == "m19.<<"


def _uses(mappings: int, keys: int) -> str:
    """``mappings`` mappings merge one of ``keys`` keys."""
    body = ", ".join(f"k{i}: {i}" for i in range(keys))
    uses = "\n".join("- {<<: *base}" for _ in range(mappings))
    return f"base: &base {{{body}}}\nuses:\n{uses}\n".replace("- ", "  - ")


@pytest.mark.parametrize("mappings, keys", [(1, 64), (2, 32), (4, 16), (64, 1)])
def test_ML2_both_sides_of_a_limit(monkeypatch, mappings, keys):
    monkeypatch.setattr(cf, "_MAX_MERGED_PAIRS", 64)
    text = _uses(mappings, keys)
    assert _load_yaml(text) == yaml.safe_load(text)
    with pytest.raises(ConfigError) as info:
        _load_yaml(_uses(mappings, keys + 1))
    assert info.value.code == "invalid_value"
    with pytest.raises(ConfigError):
        _load_yaml(_uses(mappings + 1, keys))


def test_ML2_both_sides_of_the_limit_itself():
    assert cf._MAX_MERGED_PAIRS == 10**6
    text = _uses(1000, 1000)
    assert len(_load_yaml(text)["uses"]) == 1000
    with pytest.raises(ConfigError):
        _load_yaml(_uses(1000, 1001))


def test_ML2_a_merge_of_no_mapping_is_not_counted():
    """PyYAML refuses it, and says so before the limit could."""
    text = "a: {<<: 1}\n"
    with pytest.raises(yaml.YAMLError) as info:
        _load_yaml(text)
    with pytest.raises(yaml.YAMLError) as plain:
        yaml.safe_load(text)
    assert str(info.value).endswith(str(plain.value).rsplit("\n", 1)[-1])


@pytest.mark.parametrize(
    "text",
    [
        "b: &b {x: 1, y: 2}\nm: {<<: [*b, *b, *b, 1]}\n",
        "b: &b {x: 1, y: 2}\nm: {<<: [*b, *b, *b, [*b]]}\n",
        "b: &b {x: 1, y: 2}\nm: {<<: *b, <<: *b, <<: 1}\n",
        "b: &b {x: 1, y: 2}\nm: {<<: [{<<: 1}, *b, *b, *b]}\n",
    ],
    ids=["scalar_last", "list_last", "key_last", "inside_a_source"])
def test_ML2_a_file_pyyaml_refuses_is_refused_by_pyyaml(monkeypatch, text):
    """The merges past the mistake were never made, so the limit stays out
    of it and the error keeps its line."""
    monkeypatch.setattr(cf, "_MAX_MERGED_PAIRS", 3)
    with pytest.raises(yaml.constructor.ConstructorError) as plain:
        yaml.safe_load(text)
    with pytest.raises(yaml.YAMLError) as info:
        _load_yaml(text)
    assert not isinstance(info.value, ConfigError)
    assert str(info.value) == str(plain.value)


def test_ML2_a_count_does_not_go_down_a_chain_pyyaml_never_reaches():
    links = "\n".join(f"m{i}: &m{i} {{<<: *m{i - 1}}}" for i in range(1, 2001))
    text = f"m0: &m0 {{a: 1}}\n{links}\n<<: [1, *m2000]\n"
    with pytest.raises(yaml.YAMLError, match="expected a mapping for merging"):
        _load_yaml(text)


def _counted(text: str):
    """Read ``text`` with PyYAML, counting before each merge as the parser
    does.  Returns the mappings read and the count."""
    written: dict = {}
    count = cf._MergeCount(written)

    class Loader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            cf._keep_as_written(node, written)
            if node in written:
                count.flatten(node)
            return super().construct_mapping(node, deep=deep)

    loader = Loader(text)
    try:
        root = loader.get_single_node()
        if root is not None:
            loader.construct_document(root)
    finally:
        loader.dispose()
    return written, count


def _assert_exact(text: str) -> None:
    written, count = _counted(text)
    for mapping in written:
        assert count.sizes[mapping] == len(mapping.value)


HAND_WRITTEN = {
    "empty": "{}\n",
    "no_merge": "a: {b: 1}\n",
    "one": "a: &a {x: 1}\nb: {<<: *a, y: 2}\n",
    "two_sources": "a: &a {x: 1}\nb: &b {y: 2}\nc: {<<: [*a, *b]}\n",
    "same_source_twice": "a: &a {x: 1}\nb: {<<: [*a, *a]}\n",
    "two_keys": "a: &a {x: 1}\nb: {<<: *a, <<: *a}\n",
    "in_place": "a: {<<: {x: 1}, y: 2}\n",
    "in_place_in_a_list": "a: {<<: [{x: 1}, {<<: {y: 2}}]}\n",
    "self": "a: &a {x: 1, <<: *a}\n",
    "two_step_cycle": "a: &a {x: 1, <<: &b {y: 2, <<: *a}}\n",
    "resumed": "a: &a {x: 1, <<: {y: 2, <<: *a}, <<: {z: 3}}\n",
    "resumed_twice": "a: &a {x: 1, <<: {<<: *a}, <<: {<<: *a}, <<: {<<: *a}}\n",
    "back_references": _back_references(6),
    "chain": _doubling(6),
}


@pytest.mark.parametrize("text", HAND_WRITTEN.values(), ids=HAND_WRITTEN)
def test_ML3_the_count_is_the_list_pyyaml_builds(text):
    _assert_exact(text)


def _random_file(rng: random.Random) -> str:
    """A flow mapping whose merges reach any anchor already opened, also one
    that is still open."""
    anchors: list[str] = []

    def mapping(depth: int) -> str:
        name = f"n{len(anchors)}"
        anchors.append(name)
        entries = []
        for _ in range(rng.randrange(1, 5)):
            kind = rng.random()
            if kind < 0.4:
                entries.append(f"k{rng.randrange(5)}: {rng.randrange(3)}")
            elif kind < 0.6 and depth < 4:
                entries.append(f"k{rng.randrange(5)}: {mapping(depth + 1)}")
            elif kind < 0.85:
                entries.append(f"<<: {source(depth)}")
            else:
                items = ", ".join(source(depth) for _ in range(rng.randrange(1, 4)))
                entries.append(f"<<: [{items}]")
        return f"&{name} {{{', '.join(entries)}}}"

    def source(depth: int) -> str:
        if anchors and rng.random() < 0.7:
            return f"*{rng.choice(anchors)}"
        return mapping(min(depth + 1, 4))

    return mapping(0) + "\n"


def test_ML3_the_count_is_exact_on_random_files():
    rng = random.Random(20261030)
    checked = merged = 0
    for _ in range(1500):
        text = _random_file(rng)
        try:
            written, count = _counted(text)
        except (yaml.YAMLError, RecursionError, TypeError):
            continue
        for mapping in written:
            assert count.sizes[mapping] == len(mapping.value), text
        checked += 1
        merged += count.added > 0
    assert checked > 1000
    assert merged > 300


def test_ML3_the_count_adds_what_the_merges_add():
    written, count = _counted(HAND_WRITTEN["two_sources"])
    plain = sum(k.tag != cf._MERGE_TAG for e in written.values() for k, _ in e)
    assert count.added == sum(len(m.value) for m in written) - plain == 2


def _deepest_chain(tail: str) -> int:
    """The most links ``_load_yaml`` reads, ending in the mapping ``tail``."""

    def reads(links: int) -> bool:
        lines = [f"m0: &m0 {tail}"]
        lines += [f"m{i}: &m{i} {{<<: *m{i - 1}}}" for i in range(1, links + 1)]
        lines.append(f"<<: *m{links}")
        try:
            _load_yaml("\n".join(lines) + "\n")
        except yaml.YAMLError:
            return False
        return True

    low, high = 1, 2000
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (middle, high) if reads(middle) else (low, middle)
    return low


def test_ML3_an_empty_merge_list_costs_no_frame():
    assert _deepest_chain("{<<: []}") == _deepest_chain("{a: 1}")


@pytest.mark.parametrize("command", [["validate"], ["run", "--dry-run"], ["run"]],
                         ids=["validate", "dry_run", "run"])
def test_ML4_a_command_prints_one_line(tmp_path, capsys, hang_fails, command):
    path = tmp_path / "chain.yaml"
    path.write_text(_doubling(40))
    assert main([command[0], str(path), *command[1:]]) == 2
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("sft-wick: config error [invalid_value] m19.<<: ")


def test_ML4_compare_resources_does_not_call_it_a_repeated_key(hang_fails):
    with pytest.raises(ValueError) as info:
        _safe_mapping(_doubling(40), "config")
    assert str(info.value).startswith("config: invalid YAML: m19.<<: ")
    assert "duplicate" not in str(info.value)


def test_ML4_a_repeated_key_is_still_called_one():
    with pytest.raises(ValueError, match="duplicate mapping key"):
        _safe_mapping("a: 1\na: 2\n", "config")
