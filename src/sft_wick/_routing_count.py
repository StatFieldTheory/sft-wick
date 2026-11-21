"""The count certificate from the routing enumeration.

:func:`routing_count_certificate` runs the steps of
:func:`~sft_wick.perturbation.compute_moment_numerical` up to the routings
(spatial topologies, the connected pruning, the component routings and
their connected filter) and counts them, with the built-in canonical key of
each topology for the class fields.  At positive orders it builds no
DiagramTerm and holds the topologies of one vertex sequence and the routings
of one topology at a time.  At order zero it uses the engine's ordinary
contraction, which builds its pairing list and terms.  This runs serially.
"""

from __future__ import annotations

from . import perturbation as _perturbation
from . import wick as _wick


def routing_count_certificate(observable, action, order: int, *, target: str = "moment"):
    """The :class:`~sft_wick.CountCertificate` of the routings of
    ``observable`` under ``action`` at orders 0 to ``order``.

    The checks of the inputs are those of
    :func:`~sft_wick.perturbation.compute_moment_numerical`, and an empty
    observable is refused as under ``certify=True``.  The engine functions
    are looked up on their modules at each call.

    Raises:
        CertificateError: When a check of the certificate fails.
        ValueError: As :func:`~sft_wick.perturbation.compute_moment_numerical`
            does, and for an empty observable.
        TypeError: When a coupling or field name is not a str.
    """
    p = _perturbation
    p.validate_target(target, observable)
    p.check_distinct_external_labels(observable, order)
    p.check_literal_component_labels(observable)
    p._refuse_empty_observable(observable)
    from ._canonical import BUILTIN_METHOD, BuiltinKeyer, SequenceLayout
    from ._certificate import CertificateRun

    run = CertificateRun("routings", observable, action, target,
                         canonical=BUILTIN_METHOD.label)
    for n in range(order + 1):
        tally = run.begin_order(n)
        if n == 0 and p._order_zero_contracts(observable, target):
            tally.add_r_classes(0, *p._r_class_census(observable))
            tally.add_routings(0, len(p._order_zero(observable, target, 0)[1]))
        for seq in p._vertex_sequences(observable, action, n, target,
                                       combinations=tally.combinations):
            tally.add_r_classes(seq.index, *p._r_class_census(seq.operators))
            _count_sequence(tally, seq, observable, BuiltinKeyer, SequenceLayout,
                            BUILTIN_METHOD)
        run.end_order(tally)
    return run.finish()


def _count_sequence(tally, seq, observable, keyer_class, layout_class, method) -> None:
    """Add the routings and topologies of one vertex sequence."""
    p = _perturbation
    topologies = _wick.wick_contract_spatial(seq.operators)
    if seq.incidence is not None:
        topologies = seq.incidence.prune_topologies(topologies)
    if not topologies:
        return
    keyer = keyer_class(layout_class.from_sequence(seq, observable), method)
    for ref_props, rep_pairing in topologies.values():
        routings = p._enumerate_component_routings(
            ref_props, rep_pairing, seq.operators, seq.points)
        if seq.incidence is not None:
            routings = seq.incidence.select_routings(routings)
        if routings:
            tally.add_routings(seq.index, len(routings))
            tally.add_topology(seq.index, keyer.layout, keyer.key(ref_props), ref_props)
