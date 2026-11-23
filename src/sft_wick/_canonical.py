"""Typed canonical keys of labelled topologies, and their point automorphism
order.

:func:`sft_wick.perturbation.compute_moment_numerical` emits its terms class
by class.  A class is an orbit of the relabelling group ``G`` of one vertex
sequence acting on its labelled topologies (propagator multisets, R directed
from its phi side, C unordered):

* the points of one local vertex value permute among themselves;
* the instances of one non-local vertex value permute as whole leg tuples,
  leg ``l`` onto leg ``l``, because a non-local coupling is evaluated in
  leg order and need not be symmetric;
* the external points stay fixed.

``|G|`` is the product of ``n_t!`` over the vertex values ``t`` of the
sequence.  The point automorphism order ``|Aut_V|`` of a topology is the
number of elements of ``G`` that fix it.  It counts no swap of parallel
propagators, so ``|G| / |Aut_V|`` is the size of its class.

A vertex value is the whole :class:`~sft_wick.vertices.Vertex`: two
templates with one coupling name and different fields or flags are two
types.  The graph has one node per external point, per integrated point and
per non-local instance, with the initial colour keys ``(0, i, 0)`` for the
``i``-th external label in sorted order, ``(1, t, 0)`` for the point of a
local instance of type ``t``, ``(2, t, l)`` for leg ``l`` of a non-local
instance of type ``t`` and ``(3, t, 0)`` for the instance itself, joined to
its legs; ``t`` is the rank of the vertex value in the sequence.  Every set
is sorted before it is iterated, so a key does not depend on the hash seed.

Two methods compute the key.  ``"builtin"`` runs colour refinement and an
individualisation-refinement search with automorphism pruning on a point
multigraph, where parallel propagators are one arc label
(:mod:`sft_wick._canonical_search`); it needs only the standard library.
``"pynauty"`` asks pynauty for the certificate of an undirected graph with
dummy nodes for the propagators (:mod:`sft_wick._canonical_nauty`).  Both
give the same classes.  A key is ``u16 tag length | tag | 16-byte layout
digest | certificate``, so keys of two methods, of two pynauty builds or of
two layouts with different digests never compare equal.  The digest holds
the vertex values, the cells and the counts, not the order of the sequence
or the external label names, so the layouts of two sequences with the same
vertex values, as an action that lists one vertex twice gives, share one
digest, and their keys can compare equal.  ``|Aut_V|`` always comes from the
built-in search, as an exact int.
"""

from __future__ import annotations

import hashlib
import operator
import re
import struct
import warnings
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import factorial, prod
from typing import Any, Protocol

from ._canonical_search import PointGraph, PropagatorLike

#: Version of the key bytes.  A change of the layout digest or of the
#: certificate format of either method needs a new value.
ENCODING_VERSION = 1

#: Oldest pynauty release whose certificates the keys accept.
MIN_PYNAUTY_VERSION = (2, 8, 8)

_ACCEPTED = ("auto", "builtin", "pynauty")
_TAG_ROOT = f"sft-wick-canon/{ENCODING_VERSION}"
_INSTALL_HINT = (
    'Install it with pip install "sft-wick[nauty]" (pynauty is GPLv3+), '
    'or pass canonical="builtin" for the built-in method, which gives the '
    "same classes."
)

# Colour key categories of the graph nodes.
_EXTERNAL, _LOCAL, _LEG, _INSTANCE = 0, 1, 2, 3


# --- methods ---------------------------------------------------------


@dataclass(frozen=True)
class CanonicalMethod:
    """The method that computes canonical keys, and the build it uses.

    ``name`` is ``"builtin"`` or ``"pynauty"``.  For pynauty, ``backend`` is
    ``pynauty.__version__`` and ``detail`` is
    ``"word=<bytes per setword>;byteorder=<sys.byteorder>"``, because a
    certificate is a list of native setwords.  The built-in method has
    neither.
    """

    name: str
    backend: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if self.name not in ("builtin", "pynauty"):
            raise ValueError(
                f"unknown canonical method {self.name!r}; the methods are "
                "'builtin' and 'pynauty'"
            )
        for part in (self.backend, self.detail):
            if not isinstance(part, str) or not part.isascii():
                raise ValueError(f"backend and detail must be ASCII str, got {part!r}")
        if self.name == "builtin" and (self.backend or self.detail):
            raise ValueError("the built-in method takes no backend and no detail")
        if self.name == "pynauty" and not self.backend:
            raise ValueError("a pynauty method needs the pynauty version")
        if len(self.tag_text) > 0xFFFF:
            raise ValueError("the tag of a canonical method must fit in 65535 bytes")

    @property
    def tag_text(self) -> str:
        """The tag of every key of this method, as text."""
        text = f"{_TAG_ROOT};method={self.name}"
        if self.name == "pynauty":
            text += f";pynauty={self.backend}"
            if self.detail:
                text += f";{self.detail}"
        return text

    @property
    def tag(self) -> bytes:
        """The tag as a key starts with it: u16 big-endian length, then the
        ASCII text."""
        text = self.tag_text.encode("ascii")
        return struct.pack(">H", len(text)) + text

    @property
    def label(self) -> str:
        """A short description, e.g. ``"builtin/1"`` or
        ``"pynauty/1 (pynauty 2.8.8.1, word=8, little)"``."""
        head = f"{self.name}/{ENCODING_VERSION}"
        if self.name == "builtin":
            return head
        parts = [f"pynauty {self.backend}"]
        for item in filter(None, self.detail.split(";")):
            name, _, value = item.partition("=")
            parts.append(value if name == "byteorder" else item)
        return f"{head} ({', '.join(parts)})"


BUILTIN_METHOD = CanonicalMethod("builtin")


def resolve_canonical_method(requested: str = "auto", *, stacklevel: int = 2) -> CanonicalMethod:
    """The method that ``canonical=requested`` selects.

    * ``"builtin"`` never imports pynauty.
    * ``"pynauty"`` raises :class:`ImportError`, chained to the cause, when
      pynauty cannot be imported or is older than 2.8.8.
    * ``"auto"`` takes pynauty when it can be used.  Without pynauty
      installed it takes the built-in method with no message.  When an
      installed pynauty fails to import or is older than 2.8.8 it emits one
      :class:`RuntimeWarning` quoting the cause and takes the built-in
      method.  ``stacklevel`` goes to :func:`warnings.warn`: the default
      names the caller of this function, and a function that calls it for
      its own caller passes 3.  Python's default filter shows the warning
      once per line it names.

    Both methods give the same classes, so the choice changes no term and
    no term order.

    Raises:
        ValueError: If ``requested`` is not one of the three values.
        ImportError: For ``"pynauty"`` when pynauty cannot be used.
    """
    if not isinstance(requested, str) or requested not in _ACCEPTED:
        raise ValueError(
            f"canonical must be 'auto', 'builtin' or 'pynauty', got {requested!r}"
        )
    if requested == "builtin":
        return BUILTIN_METHOD
    try:
        return _usable_pynauty()
    except ImportError as exc:
        if requested == "pynauty":
            raise ImportError(
                f'canonical="pynauty" needs pynauty 2.8.8 or newer, which '
                f"cannot be used here: {exc}. {_INSTALL_HINT}"
            ) from exc
        if isinstance(exc, ModuleNotFoundError) and exc.name == "pynauty":
            return BUILTIN_METHOD
        warnings.warn(
            f"pynauty is installed but cannot be used ({exc}); the canonical "
            'keys use the built-in method.  Pass canonical="pynauty" to raise '
            'instead, or canonical="builtin" to skip pynauty.',
            RuntimeWarning,
            stacklevel=stacklevel,
        )
        return BUILTIN_METHOD


def _usable_pynauty() -> CanonicalMethod:
    """The pynauty method of the installed build; ImportError when pynauty
    cannot be imported or is too old."""
    from ._canonical_nauty import probe

    method = probe()
    if _version_tuple(method.backend) < MIN_PYNAUTY_VERSION:
        raise ImportError(
            f"pynauty {method.backend} is older than 2.8.8, or its version "
            "cannot be read"
        )
    return method


def _version_tuple(text: str) -> tuple[int, ...]:
    """The leading dotted numbers of a version string, as ints."""
    found = re.match(r"\d+(?:\.\d+)*", text, re.ASCII)
    return tuple(int(part) for part in found.group().split(".")) if found else ()


# --- layout of a vertex sequence -------------------------------------


def _type_identity(vertex: Any) -> tuple:
    """The value of a vertex template as plain sortable data: two vertices
    have one identity exactly when they compare equal."""
    coupling = vertex.coupling
    if not isinstance(coupling, str):
        raise TypeError(
            f"vertex {vertex!r}: the coupling name must be a str, got "
            f"{type(coupling).__name__}"
        )
    fields = []
    for field in vertex.fields:
        kind = getattr(field.field_type, "value", field.field_type)
        if not isinstance(field.name, str) or not isinstance(kind, str):
            raise TypeError(
                f"vertex {vertex!r}: field names and field types must be str, "
                f"got {field.name!r} of type {kind!r}"
            )
        where = f"vertex {vertex!r}: n_components of field {field.name!r}"
        fields.append((field.name, kind, _count(field.n_components, where)))
    identity = (
        coupling,
        bool(vertex.local),
        bool(vertex.equal_time),
        bool(vertex.already_R_contracted),
        tuple(fields),
    )
    symmetry = getattr(vertex, "leg_symmetry", "none")
    if getattr(vertex, "leg_groups", ()):
        return (*identity, symmetry, vertex.leg_groups, vertex.group_mode, vertex.group_lags)
    return identity if symmetry == "none" else (*identity, symmetry)


def _count(value: object, name: str) -> int:
    """``value`` as a count: an int, or a float with an integral value,
    which compares equal to that int as a ``Field`` does."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    try:
        count = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an int, got {value!r}") from None
    if count < 0:
        raise ValueError(f"{name} must be at least 0, got {count}")
    return count


@dataclass(frozen=True)
class SequenceLayout:
    """The graph structure that every labelled topology of one vertex
    sequence shares, as plain data that does not depend on the method.

    Nodes are numbered externals first, then the integrated points, then
    one node per non-local instance.  ``labels`` holds the label of each
    point node, ``colours`` the colour key of every node, and
    ``instance_legs`` the leg nodes of each instance node in leg order.
    ``n_r`` and ``n_c`` are the numbers of R and C propagators of every
    topology; ``n_c`` is ``-1`` for a sequence whose operators admit no
    complete pairing, so that every topology is refused.  ``group_order`` is
    ``|G|``, and ``digest`` is a 16-byte blake2b digest of the encoding
    version, the vertex values, the cells and the counts.

    Build one with :meth:`from_sequence` or :meth:`from_plain`.
    """

    labels: tuple
    colours: tuple[tuple[int, int, int], ...]
    instance_legs: tuple[tuple[int, ...], ...]
    type_identities: tuple[tuple, ...]
    n_r: int
    n_c: int
    group_order: int
    digest: bytes

    @classmethod
    def from_sequence(cls, seq: Any, observable: Iterable[Any]) -> SequenceLayout:
        """The layout of a vertex sequence of
        :func:`sft_wick.perturbation._vertex_sequences` and its observable.

        Raises:
            ValueError: If the instances do not cover ``seq.points`` exactly
                once, or an external label is an integrated point.
            TypeError: If a vertex has a coupling or field name that is not
                a str.
        """
        local_points = []
        instances = []
        for vi in seq.instances:
            if vi.vertex.local:
                if len(vi.spatial_variables) != 1:
                    raise ValueError(f"local instance with {vi.spatial_variables}")
                local_points.append((vi.spatial_variables[0], vi.vertex))
            else:
                instances.append((vi.vertex, tuple(vi.spatial_variables)))
        covered = {label for label, _vertex in local_points}
        covered.update(label for _vertex, legs in instances for label in legs)
        if covered != set(seq.points):
            raise ValueError(
                "the vertex instances do not cover the integrated points of "
                "the sequence"
            )
        n_r = sum(1 for op in seq.operators if op.field.is_response)
        twice_c = sum(1 for op in seq.operators if op.field.is_physical) - n_r
        n_c = twice_c // 2 if twice_c >= 0 and twice_c % 2 == 0 else -1
        externals = sorted({op.spatial_arg for op in observable})
        return cls._build(externals, local_points, instances, n_r, n_c)

    @classmethod
    def from_plain(
        cls,
        externals: Iterable[Any],
        local_points: Iterable[tuple[Any, Any]],
        nonlocal_instances: Iterable[tuple[Any, Sequence[Any]]],
        n_r: int,
        n_c: int,
    ) -> SequenceLayout:
        """A layout from plain data.

        Args:
            externals: The external labels.
            local_points: ``(label, Vertex)`` of each local instance.
            nonlocal_instances: ``(Vertex, leg labels)`` of each non-local
                instance, in instance order, the legs in field order.
            n_r: Number of R propagators of every topology.
            n_c: Number of C propagators of every topology.

        Raises:
            ValueError: If a label is used twice, a local point has a
                non-local vertex or the reverse, a leg tuple does not match
                its vertex, or a count is negative.
            TypeError: If a vertex has a coupling or field name that is not
                a str, or a count is not an int.
        """
        externals = list(externals)
        if len(set(externals)) != len(externals):
            raise ValueError(f"an external label is repeated in {externals!r}")
        return cls._build(
            sorted(externals),
            list(local_points),
            [(vertex, tuple(legs)) for vertex, legs in nonlocal_instances],
            _count(n_r, "n_r"),
            _count(n_c, "n_c"),
        )

    @classmethod
    def _build(
        cls,
        externals: list[Any],
        local_points: list[tuple[Any, Any]],
        instances: list[tuple[Any, tuple[Any, ...]]],
        n_r: int,
        n_c: int,
    ) -> SequenceLayout:
        _check_instances(local_points, instances)
        types = [_type_identity(v) for _label, v in local_points]
        types += [_type_identity(v) for v, _legs in instances]
        ranked = tuple(sorted(set(types)))
        rank = {value: t for t, value in enumerate(ranked)}
        local_ranks = [rank[value] for value in types[: len(local_points)]]
        instance_ranks = [rank[value] for value in types[len(local_points) :]]

        points = [
            ((_LOCAL, t, 0), p)
            for (p, _v), t in zip(local_points, local_ranks, strict=True)
        ]
        for (_vertex, legs), t in zip(instances, instance_ranks, strict=True):
            points.extend(((_LEG, t, pos), label) for pos, label in enumerate(legs))
        points.sort(key=lambda item: item[0])
        labels = tuple(externals) + tuple(label for _colour, label in points)
        node = {label: i for i, label in enumerate(labels)}
        if len(node) != len(labels):
            repeated = next(x for x in labels if labels.count(x) > 1)
            raise ValueError(f"label {repeated!r} is used twice in the layout")

        nodes_of = sorted(range(len(instances)), key=lambda i: instance_ranks[i])
        legs_of = tuple(tuple(node[x] for x in instances[i][1]) for i in nodes_of)
        colours = [(_EXTERNAL, i, 0) for i in range(len(externals))]
        colours += [colour for colour, _label in points]
        colours += [(_INSTANCE, instance_ranks[i], 0) for i in nodes_of]
        sizes = Counter(colours)
        cells = tuple(sorted(sizes))
        counts = tuple(sizes[c] for c in cells)
        summary = (ENCODING_VERSION, ranked, cells, counts, n_r, n_c)
        return cls(
            labels=labels,
            colours=tuple(colours),
            instance_legs=legs_of,
            type_identities=ranked,
            n_r=n_r,
            n_c=n_c,
            group_order=prod(
                factorial(m) for c, m in sizes.items() if c[0] in (_LOCAL, _INSTANCE)
            ),
            digest=hashlib.blake2b(repr(summary).encode(), digest_size=16).digest(),
        )

    def point_automorphism_order(self, props: Iterable[PropagatorLike]) -> int:
        """``|Aut_V|`` of the topology ``props``: the number of relabellings
        in ``G`` that fix it (:func:`_point_automorphism_order`)."""
        return _point_automorphism_order(props, self)

    def point_graph(self) -> PointGraph:
        """The point multigraph of the built-in method."""
        return PointGraph(self.labels, self.colours, self.instance_legs, self.n_r, self.n_c)


def _check_instances(
    local_points: list[tuple[Any, Any]], instances: list[tuple[Any, tuple[Any, ...]]]
) -> None:
    """Refuse a local point with a non-local vertex, an instance with a
    local vertex, and a leg tuple that does not match its vertex.  A
    non-local vertex with no fields has an instance with no legs."""
    for label, vertex in local_points:
        if not vertex.local:
            raise ValueError(f"local point {label!r} has a non-local vertex")
    for vertex, legs in instances:
        if vertex.local:
            raise ValueError(f"instance {legs!r} has a local vertex")
        if getattr(vertex, "leg_groups", ()):
            if len(legs) != len(set(vertex.slot_points)):
                raise ValueError(f"instance {legs!r} needs one label per grouped coordinate")
        elif len(legs) != len(vertex.fields):
            raise ValueError(f"instance {legs!r} needs one leg per field of its vertex")


def _point_automorphism_order(props: Iterable[PropagatorLike], layout: SequenceLayout) -> int:
    """``|Aut_V|`` of the topology ``props`` on ``layout``, as an exact int,
    from the built-in search whatever method computes the keys.

    Raises:
        ValueError: If the sequence admits no complete pairing, a
            propagator has an unknown endpoint label or a kind other than
            ``"R"`` and ``"C"``, or the counts differ from the layout's.
        RuntimeError: If the order does not divide ``layout.group_order``.
    """
    return _checked_order(layout.point_graph().search(props).group_order(), layout)


def _checked_order(order: int, layout: SequenceLayout) -> int:
    if layout.group_order % order:
        raise RuntimeError(
            f"the point automorphism order {order} does not divide the order "
            f"{layout.group_order} of the relabelling group"
        )
    return order


# --- keyers ---------------------------------------------------------


class BuiltinKeyer:
    """Keys of the topologies of one layout, by the built-in search.

    Holds plain data only, so it pickles into worker processes.
    """

    __slots__ = ("method", "layout", "prefix", "_graph")

    def __init__(self, layout: SequenceLayout, method: CanonicalMethod) -> None:
        if method.name != "builtin":
            raise ValueError(f"BuiltinKeyer computes builtin keys, not {method.label}")
        self.method = method
        self.layout = layout
        #: Every key of this keyer starts with it.
        self.prefix = method.tag + layout.digest
        self._graph = layout.point_graph()

    def key(self, props: Iterable[PropagatorLike]) -> bytes:
        """The canonical key of the topology ``props``; ValueError as for
        :func:`_point_automorphism_order`."""
        return self._graph.key(self._graph.search(props), self.prefix)

    def key_and_point_automorphism_order(
        self, props: Iterable[PropagatorLike]
    ) -> tuple[bytes, int]:
        """The key and ``|Aut_V|`` of ``props`` from one search."""
        search = self._graph.search(props)
        order = _checked_order(search.group_order(), self.layout)
        return self._graph.key(search, self.prefix), order


class Keyer(Protocol):
    """What :func:`make_keyer` returns, for either method."""

    method: CanonicalMethod
    layout: SequenceLayout
    prefix: bytes

    def key(self, props: Iterable[PropagatorLike]) -> bytes: ...


def make_keyer(layout: SequenceLayout, method: CanonicalMethod) -> Keyer:
    """The keyer of ``method`` for ``layout``: a :class:`BuiltinKeyer` or a
    :class:`sft_wick._canonical_nauty.NautyKeyer`."""
    if method.name == "builtin":
        return BuiltinKeyer(layout, method)
    from ._canonical_nauty import NautyKeyer

    return NautyKeyer(layout, method)


# --- grouping keys ---------------------------------------------------


def _tag_text(key: bytes) -> str:
    """The tag a key starts with, as text."""
    if len(key) < 2:
        raise ValueError(f"{key!r} is too short to be a canonical key")
    (length,) = struct.unpack(">H", key[:2])
    raw = key[2 : 2 + length]
    if len(raw) != length or not raw.isascii():
        raise ValueError(f"{key[:40]!r}... does not start with a canonical-key tag")
    return raw.decode("ascii")


def _safe_tag(key: bytes) -> str:
    try:
        return repr(_tag_text(key))
    except ValueError:
        return "(none readable)"


def require_same_method(*keys: bytes) -> str:
    """The tag text that every key carries.

    Raises:
        ValueError: If two keys carry different tags (two methods, or two
            pynauty builds), naming both tags, or if no key is given.
    """
    if not keys:
        raise ValueError("require_same_method needs at least one key")
    first = _tag_text(keys[0])
    for key in keys[1:]:
        other = _tag_text(key)
        if other != first:
            raise ValueError(
                f"canonical keys of two methods cannot be compared: {first!r} "
                f"and {other!r}"
            )
    return first


class ClassGroups:
    """Items grouped by canonical key, in the order each key is first seen.

    Every key must start with ``prefix``, the method tag and the layout
    digest of one vertex sequence.  Point labels restart in every sequence,
    so a check of the tag alone would accept the key of a sequence with
    other vertex values or counts.  Two sequences of one observable with
    the same vertex values (an action that lists one vertex twice) have one
    digest and identical layouts, so their keys pass the check and mean the
    same classes.
    """

    __slots__ = ("prefix", "_groups")

    def __init__(self, prefix: bytes) -> None:
        self.prefix = bytes(prefix)
        self._groups: dict[bytes, list] = {}

    def add(self, key: bytes, items: list) -> None:
        """Append ``items`` to the group of ``key``.

        Raises:
            ValueError: If ``key`` does not start with ``prefix``.
        """
        if not key.startswith(self.prefix):
            raise ValueError(
                f"a canonical key with tag {_safe_tag(key)} does not start with "
                f"the prefix of these groups (tag {_safe_tag(self.prefix)} and "
                "the layout digest of the sequence): keys of two methods or of "
                "two layouts were mixed"
            )
        group = self._groups.get(key)
        if group is None:
            self._groups[key] = list(items)
        else:
            group.extend(items)

    def groups(self) -> list[list]:
        """The item lists, in first-seen key order."""
        return [list(items) for items in self._groups.values()]
