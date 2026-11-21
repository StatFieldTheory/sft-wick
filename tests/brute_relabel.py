"""Classes and stabilisers of labelled topologies, by trying relabellings.

This is the reference for ``tests/test_typed_canonical_key.py``.  It imports
nothing from ``sft_wick`` and reads plain data only:

* a topology is an iterable of propagators ``(kind, a, b)``, where ``kind``
  is ``"R"`` (``a`` on the phi side, ``b`` on the psi side) or ``"C"``
  (unordered);
* ``points`` are the labels of the integrated points.  Every other label is
  external and is never moved;
* ``local_vertex`` maps the point of each local vertex instance to a vertex
  token;
* ``instances`` lists the non-local vertex instances as ``(token, legs)``,
  with ``legs`` the tuple of leg labels in field order.

Tokens are hashable values, and equal tokens stand for equal vertices.

A relabelling sigma, a permutation of ``points``, acts on the triple made of
the sorted propagators, the map from local point to token, and the set of
``(token, legs)``.  The topologies of one vertex sequence share the last two
parts, so sigma maps one triple of the sequence onto another only if it maps
those two parts onto themselves.  :func:`relabelling_group` lists these
permutations.  It tries every permutation of the points that keeps the token
of each local point, since any other permutation changes the map, and keeps
those that also map the instance set onto itself.  The class of a topology
is its smallest image under these permutations, and its stabiliser is the
number of them that leave it unchanged.
"""
from itertools import permutations, product


def relabelling_group(points, local_vertex, instances):
    """Every permutation of ``points`` that maps ``local_vertex`` and the set
    of ``instances`` onto themselves, as dicts from a label to its image."""
    points = sorted(points)
    instances = tuple((token, tuple(legs)) for token, legs in instances)
    legs = sorted(label for _token, leg_tuple in instances for label in leg_tuple)
    if sorted(list(local_vertex) + legs) != points:
        raise ValueError("every point must be one local point or one leg")
    fibres = _fibres(points, local_vertex)
    own = set(instances)
    group = []
    for images in product(*(permutations(fibre) for fibre in fibres)):
        sigma = {}
        for fibre, image in zip(fibres, images):
            sigma.update(zip(fibre, image))
        if {(token, _moved(leg_tuple, sigma)) for token, leg_tuple in instances} == own:
            group.append(sigma)
    return group


def image(topology, sigma):
    """The sorted propagators of ``topology`` after relabelling by ``sigma``,
    with the two ends of every C in sorted order."""
    out = []
    for kind, a, b in topology:
        if kind not in ("R", "C"):
            raise ValueError(f"unknown propagator kind {kind!r}")
        a, b = sigma.get(a, a), sigma.get(b, b)
        if kind == "C" and b < a:
            a, b = b, a
        out.append((kind, a, b))
    return tuple(sorted(out))


def orbit_data(topology, group):
    """``(class, stabiliser)`` of ``topology`` under ``group``: its smallest
    image, and the number of permutations that leave it unchanged."""
    own = image(topology, {})
    images = [image(topology, sigma) for sigma in group]
    return min(images), sum(1 for other in images if other == own)


def _fibres(points, local_vertex):
    """The points grouped by local token, then the non-local legs."""
    fibres = []
    tokens = []
    for label in points:
        if label not in local_vertex:
            continue
        token = local_vertex[label]
        if token in tokens:
            fibres[tokens.index(token)].append(label)
        else:
            tokens.append(token)
            fibres.append([label])
    fibres.append([label for label in points if label not in local_vertex])
    return [tuple(fibre) for fibre in fibres]


def _moved(labels, sigma):
    return tuple(sigma.get(label, label) for label in labels)
