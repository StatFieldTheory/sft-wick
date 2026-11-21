"""Canonical keys by pynauty, the ``"pynauty"`` method of
:mod:`sft_wick._canonical`.

pynauty is GPLv3+ and optional (``pip install "sft-wick[nauty]"``).  This is
the only module of the package that imports it, and it does so inside the
functions that need it, so importing this module never imports pynauty.

The graph given to pynauty is undirected.  Its nodes are those of the point
multigraph of the built-in method (externals, integrated points, one node
per non-local instance), then one tail and one head dummy per R and one
dummy per C:

* ``R(a, b)`` is the path ``a - tail - head - b``, with ``a`` on the phi
  side;
* ``C(a, b)`` is the path ``a - c - b``, and ``C(a, a)`` the edge ``a - c``;
* each instance node is joined to each of its legs.

The ordered partition is the layout's cells in colour-key order, then the
tail, head and C dummies, one cell each.  A colour-preserving automorphism
of this graph maps dummies to dummies of the same kind and keeps incidence,
so it induces a point relabelling that keeps the propagator multiset with
the R directions; every such relabelling lifts back.  The two methods
therefore give the same classes.  The certificate does not encode the
partition, so the layout digest in the key prefix carries it.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

from ._canonical import CanonicalMethod, SequenceLayout
from ._canonical_search import PropagatorLike


def probe() -> CanonicalMethod:
    """The pynauty method of the installed pynauty: its version, the bytes
    per setword of its certificates and the byte order of this machine.

    Raises:
        ImportError: If pynauty cannot be imported; the error is passed on
            unchanged.
    """
    import pynauty

    version = getattr(pynauty, "__version__", None)
    version = version if isinstance(version, str) and version else "unknown"
    word = len(pynauty.certificate(pynauty.Graph(1)))
    return CanonicalMethod("pynauty", version, f"word={word};byteorder={sys.byteorder}")


class NautyKeyer:
    """Keys of the topologies of one layout, by pynauty.

    The colouring, the instance edges and the dummy offsets are computed
    once per layout.  The keyer holds plain data only and no pynauty
    object, so it pickles into worker processes, which import pynauty
    themselves.
    """

    __slots__ = (
        "method",
        "layout",
        "prefix",
        "_node",
        "_n_r",
        "_n_c",
        "_tail0",
        "_c0",
        "_total",
        "_colouring",
        "_base",
    )

    def __init__(self, layout: SequenceLayout, method: CanonicalMethod) -> None:
        if method.name != "pynauty":
            raise ValueError(f"NautyKeyer computes pynauty keys, not {method.label}")
        installed = probe()
        if installed != method:
            raise ValueError(
                f"keys tagged {method.label} cannot be made with the installed "
                f"{installed.label}"
            )
        self.method = method
        self.layout = layout
        #: Every key of this keyer starts with it.
        self.prefix = method.tag + layout.digest
        self._node = {label: i for i, label in enumerate(layout.labels)}
        self._n_r = layout.n_r
        self._n_c = layout.n_c
        n0 = len(layout.colours)
        n_r, n_c = layout.n_r, max(layout.n_c, 0)
        self._tail0 = n0
        self._c0 = n0 + 2 * n_r
        self._total = self._c0 + n_c
        cells: dict[tuple[int, int, int], list[int]] = {}
        for v, colour in enumerate(layout.colours):
            cells.setdefault(colour, []).append(v)
        colouring = [frozenset(cells[colour]) for colour in sorted(cells)]
        tails, heads = range(n0, n0 + n_r), range(n0 + n_r, self._c0)
        dummies = (tails, heads, range(self._c0, self._total))
        colouring += [frozenset(cell) for cell in dummies if cell]
        self._colouring = tuple(colouring)
        first = len(layout.labels)
        legs_of = enumerate(layout.instance_legs)
        self._base = {first + i: list(legs) for i, legs in legs_of}

    def key(self, props: Iterable[PropagatorLike]) -> bytes:
        """The canonical key of the topology ``props``.

        Raises:
            ValueError: If the sequence admits no complete pairing, a
                propagator has an unknown endpoint label or a kind other
                than ``"R"`` and ``"C"``, or the counts differ from the
                layout's.
        """
        if self._n_c < 0:
            raise ValueError(
                "this vertex sequence admits no complete pairing, so it has no topologies"
            )
        import pynauty

        node = self._node
        n_r = self._n_r
        adj = dict(self._base)
        tail = self._tail0
        c = self._c0
        for p in props:
            try:
                a = node[p.spatial_left]
                b = node[p.spatial_right]
            except KeyError:
                raise ValueError(
                    f"propagator {p!r} has an endpoint that is neither an "
                    "external nor a point of this vertex sequence"
                ) from None
            kind = p.kind
            if kind == "R":
                adj[tail] = [a, tail + n_r]
                adj[tail + n_r] = [b]
                tail += 1
            elif kind == "C":
                adj[c] = [a, b] if a != b else [a]
                c += 1
            else:
                raise ValueError(f"propagator {p!r} has kind {kind!r}, not 'R' or 'C'")
        found_r, found_c = tail - self._tail0, c - self._c0
        if found_r != n_r or found_c != self._n_c:
            raise ValueError(
                f"the topology has {found_r} R and {found_c} C propagators; every "
                f"topology of this sequence has {n_r} and {self._n_c}"
            )
        graph = pynauty.Graph(
            self._total,
            directed=False,
            adjacency_dict=adj,
            vertex_coloring=self._colouring,
        )
        return self.prefix + pynauty.certificate(graph)
