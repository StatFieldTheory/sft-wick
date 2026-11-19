"""Method selection, mixing guards, refusals, hash seeds and import rules
of the typed canonical keys of :mod:`sft_wick._canonical`.

``canonical="auto"`` takes pynauty when it imports, the built-in method
with no warning when pynauty is not installed, and the built-in method with
one RuntimeWarning when an installed pynauty is broken or too old.  Keys
carry their method tag and layout digest, and keys of two methods or two
layouts are refused where keys are grouped.  The classes and ``|Aut_V|``
are tested in ``tests/test_typed_canonical_key.py``.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import os
import subprocess
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import pytest

import sft_wick
from sft_wick import Action, Vertex, perturbation
from sft_wick._canonical import (
    ENCODING_VERSION,
    BuiltinKeyer,
    CanonicalMethod,
    ClassGroups,
    SequenceLayout,
    make_keyer,
    require_same_method,
    resolve_canonical_method,
)
from sft_wick.expressions import Propagator
from sft_wick.perturbation import _vertex_sequences, compute_moment_numerical

from . import _canonical_models as models

SRC = Path(sft_wick.__file__).resolve().parent
TESTS = Path(__file__).resolve().parent

BUILTIN = CanonicalMethod("builtin")
#: A pynauty method as a tag only: keys with this tag are built by hand, so
#: the tests that use it run without pynauty.
NAUTY_TAG_ONLY = CanonicalMethod("pynauty", "2.8.8.1", "word=8;byteorder=little")


def _layout(case: models.Case) -> SequenceLayout:
    if case.seq is not None:
        return SequenceLayout.from_sequence(case.seq, case.observable)
    return SequenceLayout.from_plain(**case.plain.kwargs())


# ----------------------------------------------------------------------
# method selection and refusals
# ----------------------------------------------------------------------

BROKEN_PYNAUTY = """
raise ModuleNotFoundError(
    "No module named 'pynauty.nautywrap'", name="pynauty.nautywrap"
)
"""
FAKE_PYNAUTY = """
__version__ = {version!r}


class Graph:
    def __init__(self, number_of_vertices, **kwargs):
        self.number_of_vertices = number_of_vertices


def certificate(graph):
    return bytes(8)
"""


def _hide_pynauty(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pynauty", None)
    for name in [n for n in sys.modules if n.startswith("pynauty.")]:
        monkeypatch.delitem(sys.modules, name)


def _install_fake_pynauty(monkeypatch, tmp_path: Path, source: str) -> None:
    """Put a package ``pynauty`` with ``source`` first on the path; the real
    one, if loaded, comes back after the test."""
    package = tmp_path / "fake" / "pynauty"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(source)
    _hide_pynauty(monkeypatch)
    monkeypatch.delitem(sys.modules, "pynauty")
    monkeypatch.syspath_prepend(str(tmp_path / "fake"))


def _resolve_recording(requested: str):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        method = resolve_canonical_method(requested)
    return method, [w for w in caught if issubclass(w.category, RuntimeWarning)], caught


def _assert_install_hint(error: ImportError) -> None:
    assert 'pip install "sft-wick[nauty]"' in str(error)
    assert 'canonical="builtin"' in str(error)


def _small_case() -> models.Case:
    return models.engine_case("cubic", 2, "F,F")


def _auto_picks_pynauty(monkeypatch, tmp_path, request):
    pynauty = request.getfixturevalue("pynauty_module")
    word = len(pynauty.certificate(pynauty.Graph(1)))
    detail = f"word={word};byteorder={sys.byteorder}"
    want = CanonicalMethod("pynauty", pynauty.__version__, detail)
    method, _runtime, caught = _resolve_recording("auto")
    assert method == want and caught == []
    assert resolve_canonical_method() == want == resolve_canonical_method("pynauty")


def _auto_without_pynauty_is_silent(monkeypatch, tmp_path, request):
    _hide_pynauty(monkeypatch)
    for requested in ("auto", "builtin"):
        method, _runtime, caught = _resolve_recording(requested)
        assert method == BUILTIN and caught == []


def _explicit_pynauty_without_it_raises(monkeypatch, tmp_path, request):
    _hide_pynauty(monkeypatch)
    with pytest.raises(ImportError) as raised:
        resolve_canonical_method("pynauty")
    _assert_install_hint(raised.value)
    assert isinstance(raised.value.__cause__, ModuleNotFoundError)
    assert raised.value.__cause__.name == "pynauty"
    with pytest.raises(ImportError):
        compute_moment_numerical(
            models.observable(models.XY), Action([models.F]), 1, canonical="pynauty"
        )


def _unknown_value_raises_before_enumeration(monkeypatch, tmp_path, request):
    with pytest.raises(ValueError) as raised:
        resolve_canonical_method("fast")
    assert all(word in str(raised.value) for word in ("auto", "builtin", "pynauty"))

    def no_enumeration(*args, **kwargs):
        raise AssertionError("enumeration started before the method was resolved")

    monkeypatch.setattr(perturbation, "_vertex_sequences", no_enumeration)
    monkeypatch.setattr(perturbation, "_order_zero", no_enumeration)
    with pytest.raises(ValueError):
        compute_moment_numerical(
            models.observable(models.XY), Action([models.F]), 2, canonical="fast"
        )


def _broken_install_warns_once(monkeypatch, tmp_path, request):
    _install_fake_pynauty(monkeypatch, tmp_path, BROKEN_PYNAUTY)
    method, runtime, caught = _resolve_recording("auto")
    assert method == BUILTIN
    assert len(runtime) == len(caught) == 1
    assert "pynauty.nautywrap" in str(runtime[0].message)
    with pytest.raises(ImportError) as raised:
        resolve_canonical_method("pynauty")
    _assert_install_hint(raised.value)
    assert getattr(raised.value.__cause__, "name", None) == "pynauty.nautywrap"


def _engine_warning_names_its_caller(monkeypatch, tmp_path, request):
    """The warning of ``compute_moment_numerical`` names the caller's line,
    so the default filter shows it once per calling line."""
    _install_fake_pynauty(monkeypatch, tmp_path, BROKEN_PYNAUTY)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_moment_numerical(models.observable(models.XY), Action([models.F]), 1)
    (runtime,) = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert Path(runtime.filename) == Path(__file__)


def _old_version_warns_once(monkeypatch, tmp_path, request):
    _install_fake_pynauty(monkeypatch, tmp_path, FAKE_PYNAUTY.format(version="2.8.7"))
    method, runtime, caught = _resolve_recording("auto")
    assert method == BUILTIN
    assert len(runtime) == len(caught) == 1
    assert "2.8.7" in str(runtime[0].message)
    with pytest.raises(ImportError) as raised:
        resolve_canonical_method("pynauty")
    _assert_install_hint(raised.value)


def _versions_compare_as_numbers(monkeypatch, tmp_path, request):
    _install_fake_pynauty(monkeypatch, tmp_path, FAKE_PYNAUTY.format(version="2.10.0"))
    method, _runtime, caught = _resolve_recording("auto")
    assert method == CanonicalMethod("pynauty", "2.10.0", f"word=8;byteorder={sys.byteorder}")
    assert caught == []


def _tags_name_method_and_version(monkeypatch, tmp_path, request):
    builtin_text = f"sft-wick-canon/{ENCODING_VERSION};method=builtin"
    nauty_text = (
        f"sft-wick-canon/{ENCODING_VERSION};method=pynauty;pynauty=2.8.8.1;word=8;byteorder=little"
    )
    assert ENCODING_VERSION == 1
    assert BUILTIN.tag == len(builtin_text).to_bytes(2, "big") + builtin_text.encode("ascii")
    assert NAUTY_TAG_ONLY.tag == len(nauty_text).to_bytes(2, "big") + nauty_text.encode("ascii")
    assert BUILTIN.label == "builtin/1"
    assert NAUTY_TAG_ONLY.label == "pynauty/1 (pynauty 2.8.8.1, word=8, little)"
    case = _small_case()
    layout = _layout(case)
    builtin_key = make_keyer(layout, BUILTIN).key(case.topologies[0])
    nauty_key = NAUTY_TAG_ONLY.tag + layout.digest + b"certificate"
    newer = CanonicalMethod("pynauty", "2.9.0", "word=8;byteorder=little")
    assert require_same_method(builtin_key, builtin_key) == builtin_text
    assert require_same_method(nauty_key, nauty_key) == nauty_text
    with pytest.raises(ValueError) as raised:
        require_same_method(builtin_key, nauty_key)
    assert builtin_text in str(raised.value) and nauty_text in str(raised.value)
    with pytest.raises(ValueError):
        require_same_method(nauty_key, newer.tag + layout.digest + b"certificate")


def _class_groups_check_the_prefix(monkeypatch, tmp_path, request):
    """Point labels restart in every sequence, so a key of (F, F, F, G) is
    refused by the groups of (F, F, G, G) through the layout digest."""
    ffgg = models.engine_case("F_G", 4, "F,F,G,G")
    fffg = models.engine_case("F_G", 4, "F,F,F,G")
    keyer = make_keyer(_layout(ffgg), BUILTIN)
    other = make_keyer(_layout(fffg), BUILTIN)
    first, *rest = ffgg.topologies
    second = next(p for p in rest if keyer.key(p) != keyer.key(first))
    groups = ClassGroups(keyer.prefix)
    groups.add(keyer.key(first), ["a"])
    groups.add(keyer.key(second), ["b"])
    groups.add(keyer.key(first), ["c", "d"])
    assert groups.groups() == [["a", "c", "d"], ["b"]]
    assert other.prefix != keyer.prefix
    with pytest.raises(ValueError):
        groups.add(other.key(first), ["e"])
    with pytest.raises(ValueError):
        groups.add(NAUTY_TAG_ONLY.tag + _layout(ffgg).digest + b"certificate", ["f"])


def _engine_refuses_mixed_keys(monkeypatch, tmp_path, request):
    own = BUILTIN.tag
    original = BuiltinKeyer.key
    calls = itertools.count()

    def every_second_foreign(self, props):
        key = original(self, props)
        return key if next(calls) % 2 == 0 else NAUTY_TAG_ONLY.tag + key[len(own):]

    monkeypatch.setattr(BuiltinKeyer, "key", every_second_foreign)
    with pytest.raises(ValueError):
        compute_moment_numerical(
            models.observable(models.XY), Action([models.F]), 2, canonical="builtin"
        )
    assert next(calls) > 1


def _invalid_input_refused(method: CanonicalMethod) -> None:
    case = _small_case()
    layout = _layout(case)
    keyer = make_keyer(layout, method)
    props = list(case.topologies[0])
    unknown_label = models.relabel(props, {props[0].spatial_left: "nowhere"})
    first = props[0]
    unknown_kind = [Propagator("Q", None, None, first.spatial_left, first.spatial_right)]
    unknown_kind += props[1:]
    for bad in (unknown_label, unknown_kind, props[:-1]):
        with pytest.raises(ValueError):
            keyer.key(bad)
    for bad in (unknown_label, unknown_kind):
        with pytest.raises(ValueError):
            layout.point_automorphism_order(bad)
    # Two externals and one cubic vertex leave three phi for one psi: no
    # complete pairing, so every topology is refused.
    obs = models.observable(models.XY)
    (odd,) = list(_vertex_sequences(obs, Action([models.F]), 1, "moment"))
    unpaired = make_keyer(SequenceLayout.from_sequence(odd, obs), method)
    with pytest.raises(ValueError, match="admits no complete pairing"):
        unpaired.key([Propagator("R", None, None, "x", min(odd.points))])
    named_by_tuple = Vertex([models.PSI, models.PHI, models.PHI], ("F",))
    with pytest.raises(TypeError):
        SequenceLayout.from_plain(
            externals=("x",),
            local_points=(("z", named_by_tuple),),
            nonlocal_instances=(),
            n_r=1,
            n_c=1,
        )


def _invalid_input_refused_by_builtin(monkeypatch, tmp_path, request):
    _invalid_input_refused(BUILTIN)


def _invalid_input_refused_by_pynauty(monkeypatch, tmp_path, request):
    request.getfixturevalue("pynauty_module")
    _invalid_input_refused(resolve_canonical_method("pynauty"))


def _keyer_refuses_another_pynauty_build(monkeypatch, tmp_path, request):
    """A key tagged with a pynauty build is made by that build only."""
    request.getfixturevalue("pynauty_module")
    other = CanonicalMethod("pynauty", "0.0.1", f"word=8;byteorder={sys.byteorder}")
    with pytest.raises(ValueError, match="cannot be made with the installed"):
        make_keyer(_layout(_small_case()), other)


GUARD_SCENARIOS = {
    "auto picks pynauty": _auto_picks_pynauty,
    "auto without pynauty is silent": _auto_without_pynauty_is_silent,
    "explicit pynauty without it raises": _explicit_pynauty_without_it_raises,
    "unknown value raises before enumeration": _unknown_value_raises_before_enumeration,
    "broken install warns once": _broken_install_warns_once,
    "engine warning names its caller": _engine_warning_names_its_caller,
    "old version warns once": _old_version_warns_once,
    "versions compare as numbers": _versions_compare_as_numbers,
    "tags name method and version": _tags_name_method_and_version,
    "class groups check the prefix": _class_groups_check_the_prefix,
    "engine refuses mixed keys": _engine_refuses_mixed_keys,
    "invalid input refused by builtin": _invalid_input_refused_by_builtin,
    "invalid input refused by pynauty": _invalid_input_refused_by_pynauty,
    "keyer refuses another pynauty build": _keyer_refuses_another_pynauty_build,
}


@pytest.mark.parametrize("scenario", list(GUARD_SCENARIOS))
def test_method_selection_and_mixing_guards(scenario, monkeypatch, tmp_path, request):
    """``auto`` takes pynauty when it imports, the built-in with no warning
    when pynauty is not installed, and the built-in with one RuntimeWarning
    when an installed pynauty is broken or older than 2.8.8.  Keys carry
    their method tag and layout digest, and a key of another method or
    another layout is refused where keys are grouped."""
    GUARD_SCENARIOS[scenario](monkeypatch, tmp_path, request)


# ----------------------------------------------------------------------
# hash seeds and imports
# ----------------------------------------------------------------------

HASH_SEEDS = ("0", "1", "2")

#: sha256 of the sorted built-in keys of every topology of {F, G} at order
#: 4 and {F, K} at order 3, as the hash-seed script prints it under hash
#: seeds 0, 1 and 2.  A later change of the key bytes needs a new
#: ``ENCODING_VERSION`` and a new value here.
PINNED_BUILTIN_KEY_DIGEST = "68a228d5a570297b862c36ab660d37963c3227b8445b8e51f8d7893ba592a291"

SEED_SCRIPT = """
import hashlib
import json
import sys

from sft_wick import Action, Field, Vertex
from sft_wick._canonical import SequenceLayout, make_keyer, resolve_canonical_method
from sft_wick.perturbation import _vertex_sequences, compute_moment_numerical
from sft_wick.wick import wick_contract_spatial

phi = Field("phi", "physical")
psi = Field("psi", "response")
F = Vertex([psi, phi, phi], "F")
G = Vertex([psi, phi, phi], "G")
K = Vertex([psi, psi, psi], "K", local=False)
method = resolve_canonical_method(sys.argv[1])
counts, keys, terms = {}, [], hashlib.sha256()
for vertices, labels, order in (([F, G], "xy", 4), ([F, K], "xyz", 3)):
    action = Action(vertices)
    obs = [phi(label) for label in labels]
    for seq in _vertex_sequences(obs, action, order, "moment"):
        keyer = make_keyer(SequenceLayout.from_sequence(seq, obs), method)
        found = [keyer.key(list(p)) for p, _ in wick_contract_spatial(seq.operators).values()]
        name = ",".join(vi.vertex.coupling for vi in seq.instances)
        counts[f"{order}:{name}"] = len(set(found))
        keys.extend(found)
    out = compute_moment_numerical(
        [phi(label) for label in labels], action, order, canonical=sys.argv[1])
    for n in sorted(out):
        for term in out[n]:
            terms.update(repr((n, term.propagators, term.coupling_sum)).encode())
digest = hashlib.sha256()
for key in sorted(keys):
    digest.update(len(key).to_bytes(4, "big") + key)
print(json.dumps({"label": method.label, "counts": counts,
                  "keys": digest.hexdigest(), "terms": terms.hexdigest()}))
"""


@lru_cache(maxsize=None)
def _seed_run(method_name: str, seed: str) -> dict:
    """What the script prints in a process of its own with this hash seed.
    The interpreter runs without ``-I`` and ``-E``, which ignore
    ``PYTHONHASHSEED``."""
    done = subprocess.run(
        [sys.executable, "-c", SEED_SCRIPT, method_name],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(SRC.parent)},
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_keys_do_not_depend_on_the_hash_seed(canonical_method_name):
    """loky workers start with hash seeds of their own, so a key that
    followed the iteration order of a set would split classes between
    workers.  The class counts, the keys and the term order agree under three
    seeds and between the methods, and the built-in keys equal a pinned
    digest, so a change of their bytes is seen."""
    runs = [_seed_run(canonical_method_name, seed) for seed in HASH_SEEDS]
    assert runs == [runs[0]] * len(runs)
    assert runs[0]["counts"]["4:F,F,G,G"] == 364
    if canonical_method_name == "builtin":
        assert runs[0]["keys"] == PINNED_BUILTIN_KEY_DIGEST
    else:
        builtin = _seed_run("builtin", HASH_SEEDS[0])
        assert runs[0]["counts"] == builtin["counts"]
        assert runs[0]["terms"] == builtin["terms"]


#: sha256 of the built-in keys of the hand-built topologies, in order.  The
#: search on the strongly regular join meets inequivalent leaves (19 of
#: them); no other graph here and no engine sequence pinned above does.  So
#: a change of which leaf the search keeps shows here through that one
#: graph.  A change of these bytes needs a new ``ENCODING_VERSION`` and a
#: new value.
PINNED_HAND_BUILT_KEY_DIGEST = "6d089c3686fb6a28f5935a1784d284a01078cfb59ecee09409c29f13bec99c9e"


def test_builtin_keys_of_hand_built_graphs_are_pinned():
    """The built-in keys of the hand-built graphs equal a pinned digest."""
    cases = models.hand_built_cases() + (
        models.strongly_regular_join(),
        models.loops_and_two_cycles(8),
    )
    digest = hashlib.sha256()
    for case in cases:
        keyer = make_keyer(_layout(case), BUILTIN)
        for props in case.topologies:
            key = keyer.key(props)
            digest.update(len(key).to_bytes(4, "big") + key)
    assert digest.hexdigest() == PINNED_HAND_BUILT_KEY_DIGEST


def _imports(path: Path) -> list[tuple[str, int, bool]]:
    """``(module, relative level, inside a function)`` of every import in
    ``path``; ``from . import x`` gives ``("x", 1, ...)``."""
    found = []

    def visit(node: ast.AST, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import):
                found.extend((alias.name, 0, in_function) for alias in child.names)
            elif isinstance(child, ast.ImportFrom):
                if child.module:
                    found.append((child.module, child.level, in_function))
                else:
                    found.extend((alias.name, child.level, in_function) for alias in child.names)
            inner = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            visit(child, in_function or inner)

    visit(ast.parse(path.read_text(encoding="utf-8")), False)
    return found


def test_reference_and_package_import_rules():
    """The reference shares no code with the package; pynauty, which is
    GPL, is imported by one module of the package, lazily; ``_canonical``
    needs only the standard library and its search module, and the search
    module only the standard library."""
    assert {name for name, _level, _in in _imports(TESTS / "brute_relabel.py")} == {"itertools"}
    importers: dict[str, list[bool]] = {}
    for path in sorted(SRC.rglob("*.py")):
        for name, _level, inside in _imports(path):
            if name.split(".")[0] == "pynauty":
                importers.setdefault(path.relative_to(SRC).as_posix(), []).append(inside)
    assert list(importers) == ["_canonical_nauty.py"]
    assert all(importers["_canonical_nauty.py"])
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    allowed = {
        "_canonical.py": {("_canonical_search", 1, False), ("_canonical_nauty", 1, True)},
        "_canonical_search.py": set(),
    }
    for module, relative in allowed.items():
        for name, level, inside in _imports(SRC / module):
            if level == 0:
                assert name.split(".")[0] in stdlib, (module, name)
            else:
                assert (name, level, inside) in relative, (module, name)
