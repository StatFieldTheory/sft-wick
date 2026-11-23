"""Certificates for the engines' pairing counts and small numeric sums.

With ``certify=True``, :func:`~sft_wick.perturbation.compute_moment`,
:func:`~sft_wick.perturbation.compute_moment_numerical`,
:meth:`~sft_wick.workflow.system.System.expand` and
:func:`sft_wick._routing_count.routing_count_certificate` keep one row per
vertex sequence of every order (:class:`SequenceCount`) and check it:

* the number of operator pairings the output stands for (the summed weight
  of the terms, the ``DiagramInfo`` records of ``compute_moment``, or the
  routings) equals :func:`sft_wick._certify.count_valid_pairings` of the
  sequence, which imports nothing from the package;
* the summed term weight is a non-negative integer and every coupling node
  is one the weight reader counts;
* :func:`sft_wick.wick.r_classes` yields each R class once;
* each class of labelled topologies has ``|G| / |Aut_V|`` members, with
  ``|G|`` from the vertex templates and ``|Aut_V|`` from
  :meth:`sft_wick._canonical.SequenceLayout.point_automorphism_order`;
* the multinomial coefficients of the sequences of order ``n`` add up to
  ``v**n`` for ``v`` vertex templates.

When the engine emits numeric terms, the certificate also compares their
values on synthetic arrays with an independently enumerated pairing sum.
The numeric records belong to the same sequence rows as the counts.

Failures are collected over all orders and raised once, as
:class:`CertificateError`, after the last order.  The counter's inputs come
from the vertex templates and the observable (:func:`counter_inputs`), not
from the instantiated sequence.  This module imports no engine module;
:mod:`sft_wick._canonical` is imported inside the class tally and by
``CertificateRun.outcome`` for the encoding version.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from math import factorial, prod
from typing import TYPE_CHECKING, Any

from ._certify import UnsupportedLayout, count_valid_pairings
from .expressions import KroneckerDelta, Product, Rational, Sum, Symbol

if TYPE_CHECKING:
    from ._value_certificate import ValueComparison

#: Version of the record layout of :class:`CountCertificate`.
CERTIFICATE_SCHEMA = 2

#: Failures printed in the message of a :class:`CertificateError`.
_SHOWN_FAILURES = 10

#: The paths, with the compared fields they fill at order 0 and above.
_COMPARED = {
    "compute_moment": (("term_weight", "diagram_infos"),
                       ("term_weight", "diagram_infos")),
    "compute_moment/operator": (("term_weight", "diagram_infos"),
                                ("diagram_infos",)),
    "compute_moment_numerical": (("term_weight",), ("term_weight",)),
    "multiplicity": (("term_weight",), ("term_weight",)),
    "routings": (("routings",), ("routings",)),
}
#: The paths that fill the class fields above order 0.
_CLASS_PATHS = ("compute_moment", "compute_moment_numerical", "multiplicity", "routings")

_INDEPENDENCE = (
    "The counter (sft_wick._certify) shares no code with the contraction "
    "engine."
)


# --- records ----------------------------------------------------------


@dataclass(frozen=True)
class SequenceCount:
    """The counts of one vertex sequence.

    ``counter`` is None only when the counter refused the layout, which is
    a failure.  ``term_weight`` is None when the path does not sum term
    weights, or when the sum is not a non-negative integer or a coupling
    node is unreadable (both failures).  Every other None means the path
    does not compute that quantity.  ``classes`` are the typed classes of
    the labelled topologies; ``group_order`` is ``|G|``, the product of
    ``n_t!`` over the groups of equal vertex templates, and ``orbit_sum``
    the sum over classes of ``|G| / |Aut_V|``.

    ``value_checks`` contains the numeric comparisons that ran. ``value_skip``
    explains a skipped comparison in a certified expansion. These diagnostics
    do not participate in count-certificate equality.
    """

    index: int
    vertices: tuple[str, ...]
    multinomial: int
    counter: int | None
    term_weight: int | None = None
    diagram_infos: int | None = None
    routings: int | None = None
    r_classes: int | None = None
    topologies: int | None = None
    classes: int | None = None
    group_order: int | None = None
    orbit_sum: int | None = None
    note: str = ""
    # Record the numeric probes without changing count-certificate equality.
    value_checks: tuple[ValueComparison, ...] = field(default=(), compare=False)
    value_skip: str = field(default="", compare=False)


@dataclass(frozen=True)
class OrderCount:
    """The rows of one order, one per term of
    ``action.all_vertex_combinations(order)``, in its order."""

    order: int
    n_vertices: int
    sequences: tuple[SequenceCount, ...]

    @property
    def pairings(self) -> int:
        """The counter's pairings summed over the sequences."""
        return sum(row.counter or 0 for row in self.sequences)

    @property
    def weighted_pairings(self) -> int:
        """The counter's pairings times the multinomial coefficients."""
        return sum(row.multinomial * (row.counter or 0) for row in self.sequences)


@dataclass(frozen=True)
class CountCertificate:
    """The count certificate of one call: one
    :class:`~sft_wick._certificate.OrderCount` per order.

    It holds plain data, so
    :func:`dataclasses.asdict` gives plain data.  ``encoding`` is the
    version of the canonical-key colouring when the path computes classes.
    ``canonical`` names the key method; it is not compared, so the
    certificates of the built-in and the pynauty keys compare equal.
    """

    schema: int
    engine: str
    target: str
    observable: tuple[str, ...]
    orders: tuple[OrderCount, ...]
    encoding: int | None
    canonical: str = field(default="", compare=False)

    def order(self, n: int) -> OrderCount:
        """The record of order ``n``; KeyError when it was not computed."""
        for record in self.orders:
            if record.order == n:
                return record
        raise KeyError(n)


class CertificateError(RuntimeError):
    """A count or value check found a difference.

    ``certificate`` holds the rows of every order, ``failures`` one message
    per failed check.
    """

    def __init__(
        self,
        message: str,
        certificate: CountCertificate | None = None,
        failures: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.certificate = certificate
        self.failures = tuple(failures)


class CertifiedTerms(dict):
    """The ``{order: [DiagramTerm, ...]}`` dict of
    ``compute_moment_numerical(certify=True)``, with its
    :class:`CountCertificate` as the attribute ``certificate``.

    ``dict(terms)`` gives a plain dict without the attribute.
    """

    def __init__(self, terms: Any = (), certificate: CountCertificate | None = None):
        super().__init__(terms)
        self.certificate = certificate

    def __reduce__(self):
        return (type(self), (dict(self), self.certificate))


# --- counter inputs ---------------------------------------------------


@dataclass(frozen=True)
class _VertexPoint:
    """The label of a vertex point for the counter; equal to no user
    label."""

    vertex: int
    leg: int


def counter_inputs(observable: Sequence[Any], vertices: Sequence[Any]) -> dict:
    """The keyword arguments of :func:`~sft_wick._certify.count_valid_pairings`
    for ``observable`` and the vertex templates ``vertices``.

    Each external operator is a group of its own with its label and its own
    owner.  A local or field-less vertex is one group on its own point; a
    non-local vertex is one group per leg, all owned by its node.  Reads
    only the observable's fields and labels and the templates' fields and
    ``local`` flags.
    """
    points: list = []
    phi_counts: list[int] = []
    psi_owner: list[int] = []
    owners: list[int] = []

    def add(label, fields, owner) -> None:
        points.append(label)
        phi_counts.append(sum(1 for f in fields if f.is_physical))
        owners.append(owner)
        psi_owner.extend([len(points) - 1] * sum(1 for f in fields if f.is_response))

    for k, op in enumerate(observable):
        add(op.spatial_arg, (op.field,), k)
    base = len(observable)
    for i, vertex in enumerate(vertices):
        fields = tuple(vertex.fields)
        if vertex.leg_groups:
            # Count using the clocks shared by the field slots. Spatial
            # positions and component labels play no role in the Itô rule.
            roots = {g: (root, d) for g, root, d in vertex.group_lags}
            clocks = {}
            for group, f in zip(vertex.leg_groups, fields, strict=True):
                key = roots.get(group, (group, 0.0))
                index = clocks.setdefault(key, len(clocks))
                add(_VertexPoint(i, index), (f,), base + i)
        elif vertex.local or not fields:
            add(_VertexPoint(i, 0), fields, base + i)
        else:
            for leg, f in enumerate(fields):
                add(_VertexPoint(i, leg), (f,), base + i)
    return {
        "points": points,
        "phi_counts": phi_counts,
        "psi_owner": psi_owner,
        "owner_of_group": owners,
    }


# --- term weights -----------------------------------------------------


class _UnreadableNode(Exception):
    """A coupling node the weight reader does not count."""


def _multiplicity(expr: Any) -> Fraction:
    if isinstance(expr, Sum):
        return sum((_multiplicity(term) for term in expr.terms), Fraction(0))
    if isinstance(expr, Product):
        return prod((_multiplicity(f) for f in expr.factors), start=Fraction(1))
    if isinstance(expr, Rational):
        return Fraction(expr.numerator, expr.denominator)
    if isinstance(expr, (Symbol, KroneckerDelta)):
        return Fraction(1)
    raise _UnreadableNode(type(expr).__name__)


def _plain_coupling(expr: Any) -> bool:
    if isinstance(expr, Symbol):
        return True
    return isinstance(expr, Product) and all(
        isinstance(f, (Symbol, KroneckerDelta)) for f in expr.factors
    )


def term_weight(term: Any, prefactor: Rational) -> Fraction:
    """The number of pairings a DiagramTerm stands for:
    ``rational_prefactor / prefactor`` times the multiplicity of its
    coupling sum.

    The multiplicity is the sum over a Sum, the product over a Product, the
    value of a Rational, and 1 for a Symbol and a KroneckerDelta.

    Raises:
        _UnreadableNode: For any other node, with its type name.
    """
    if term.rational_prefactor == prefactor and _plain_coupling(term.coupling_sum):
        return Fraction(1)
    ratio = Fraction(term.rational_prefactor.numerator,
                     term.rational_prefactor.denominator)
    ratio /= Fraction(prefactor.numerator, prefactor.denominator)
    return ratio * _multiplicity(term.coupling_sum)


# --- the run ----------------------------------------------------------


def _group_order(vertex_seq: Iterable[Any]) -> int:
    """``|G|``: the product of ``n_t!`` over groups of equal templates,
    compared by ``==``."""
    groups: list[list] = []
    for vertex in vertex_seq:
        for group in groups:
            if group[0] == vertex:
                group[1] += 1
                break
        else:
            groups.append([vertex, 1])
    return prod(factorial(count) for _vertex, count in groups)


def _row_label(order: int, index: int, names: tuple[str, ...]) -> str:
    shown = ", ".join(names) + ("," if len(names) == 1 else "")
    return f"order {order}, sequence {index} ({shown})"


_COUNT_WORDS = {
    "term_weight": "the emitted terms stand for {}",
    "diagram_infos": "compute_moment built {} DiagramInfo records",
    "routings": "the routing count is {}",
}


class OrderTally:
    """The counts of one order while the engine runs.

    ``combinations`` is the ``(vertex_seq, multinomial)`` list of the order,
    which the engine iterates; ``counters`` the counter's count of each
    (None where the counter refused the layout).
    """

    def __init__(
        self,
        order: int,
        combinations: tuple,
        counters: tuple,
        refusals: tuple[str, ...],
    ) -> None:
        self.order = order
        self.combinations = combinations
        self.counters = counters
        self.refusals = refusals
        n = len(combinations)
        self._weights = [Fraction(0)] * n
        self._unreadable: list[str] = [""] * n
        self._infos = [0] * n
        self._routings = [0] * n
        self._r_classes: list[tuple[int, int] | None] = [None] * n
        self._topologies = [0] * n
        self._classes: list[dict] = [{} for _ in range(n)]
        self._notes = [""] * n

    def add_terms(self, index: int, terms: Iterable[Any], prefactor: Rational) -> None:
        """Add the weights of the terms of sequence ``index``."""
        for term in terms:
            try:
                self._weights[index] += term_weight(term, prefactor)
            except _UnreadableNode as exc:
                self._unreadable[index] = self._unreadable[index] or str(exc)

    def add_diagram_infos(self, index: int, count: int) -> None:
        self._infos[index] += count

    def add_routings(self, index: int, count: int) -> None:
        self._routings[index] += count

    def add_r_classes(self, index: int, yielded: int, distinct: int) -> None:
        """The R classes yielded for the operators of ``index``, and the
        number of distinct ones."""
        self._r_classes[index] = (yielded, distinct)

    def add_topology(
        self, index: int, layout: Any, key: Any, props: Any, *, count: int = 1,
    ) -> None:
        """Record a labelled topology or its weighted orbit, and its
        canonical key. The first member of a class is its representative:
        its point automorphism order (or the message of the RuntimeError
        that refused it) is kept, and the layout and propagators are not."""
        self._topologies[index] += count
        members = self._classes[index].get(key)
        if members is not None:
            members[0] += count
            return
        try:
            aut: int | str = layout.point_automorphism_order(props)
        except RuntimeError as exc:
            aut = str(exc)
        self._classes[index][key] = [count, aut]

    def skip_classes(self, index: int, reason: str) -> None:
        """Leave the class fields of ``index`` empty, with the reason."""
        self._notes[index] = f"classes not computed: {reason}"


class SequenceClassTap:
    """Built-in canonical keys of the topologies of one sequence, for the
    class tally of ``compute_moment``, which groups its terms by another
    form.  The layout is built on the first kept topology; a TypeError of
    the layout (a coupling or field name that is not a str) empties the
    class fields of the row instead of stopping the engine."""

    def __init__(self, tally: OrderTally, seq: Any, observable: Sequence[Any]) -> None:
        self._tally = tally
        self._seq = seq
        self._observable = observable
        self._keyer: Any = None
        self._skipped = False

    def add(self, props: Any) -> None:
        if self._skipped:
            return
        if self._keyer is None:
            from ._canonical import BUILTIN_METHOD, BuiltinKeyer, SequenceLayout

            try:
                layout = SequenceLayout.from_sequence(self._seq, self._observable)
            except TypeError as exc:
                self._skipped = True
                self._tally.skip_classes(self._seq.index, str(exc))
                return
            self._keyer = BuiltinKeyer(layout, BUILTIN_METHOD)
        self._tally.add_topology(
            self._seq.index, self._keyer.layout, self._keyer.key(props), props)


class CertificateRun:
    """The certificate of one call, built order by order."""

    def __init__(
        self,
        engine: str,
        observable: Sequence[Any],
        action: Any,
        target: str,
        *,
        canonical: str = "",
        value_flags: tuple[bool, bool, bool, bool] | None = None,
    ) -> None:
        if engine not in _COMPARED:
            raise ValueError(f"unknown certificate path {engine!r}")
        self.engine = engine
        self.observable = tuple(observable)
        self.action = action
        self.target = target
        self.canonical = canonical
        self._value_flags = value_flags
        self._orders: list[OrderCount] = []
        self._failures: list[str] = []

    def begin_order(self, n: int) -> OrderTally:
        """Enumerate the sequences of order ``n`` once and count each."""
        combinations = tuple(self.action.all_vertex_combinations(n))
        counters: list[int | None] = []
        refusals: list[str] = []
        for vertex_seq, _multinomial in combinations:
            try:
                counters.append(count_valid_pairings(
                    **counter_inputs(self.observable, vertex_seq),
                    target=self.target,
                ))
                refusals.append("")
            except UnsupportedLayout as exc:
                counters.append(None)
                refusals.append(str(exc))
        return OrderTally(n, combinations, tuple(counters), tuple(refusals))

    def end_order(self, tally: OrderTally, terms=(), value_spans=()) -> None:
        """Check the rows of ``tally`` and keep its record."""
        n = tally.order
        n_failures = len(self._failures)
        total = sum(multinomial for _seq, multinomial in tally.combinations)
        n_vertices = len(self.action.vertices)
        if total != n_vertices ** n:
            self._failures.append(
                f"order {n}: the multinomial coefficients of the vertex "
                f"sequences add up to {total}, not {n_vertices}**{n} = "
                f"{n_vertices ** n}"
            )
        rows = tuple(self._row(tally, index) for index in range(len(tally.combinations)))
        if self._value_flags is not None:
            if len(self._failures) != n_failures:
                rows = tuple(replace(row, value_skip="count check failed") for row in rows)
            else:
                rows = self._value_rows(tally, rows, terms, value_spans)
        self._orders.append(OrderCount(n, n_vertices, rows))

    def _value_rows(self, tally, rows, terms, spans):
        from ._value_certificate import (
            MAX_OPERATORS,
            ValueLayoutError,
            compare_values,
            sequence_prefactors,
        )

        spans = {i: (a, b, instances) for i, a, b, instances in spans}
        records = []
        prefactors = sequence_prefactors(len(self.action.vertices), tally.order)
        for row, (vertices, _), prefactor in zip(rows, tally.combinations, prefactors, strict=True):
            if self.engine == "compute_moment/operator" and tally.order > 0:
                records.append(replace(row, value_skip="the operator path emits no numeric terms"))
                continue
            if len(self.observable) + sum(len(vertex.fields) for vertex in vertices) > MAX_OPERATORS:
                records.append(replace(row, value_skip=f"more than {MAX_OPERATORS} operators"))
                continue
            a, b, instances = spans.get(row.index, (0, 0, ()))
            where = _row_label(tally.order, row.index, row.vertices)
            try:
                checks = tuple(compare_values(self.observable, vertices, self.target,
                                              prefactor, terms[a:b], instances,
                                              self._value_flags, self.engine))
            except ValueLayoutError as exc:
                records.append(replace(row, value_skip=str(exc)))
                continue
            except (TypeError, ValueError, KeyError, OverflowError) as exc:
                self._failures.append(f"{where}: value certificate could not evaluate: {exc}")
                records.append(replace(row, value_skip="numeric comparison failed"))
                continue
            for check in checks:
                if not check.passed:
                    self._failures.append(
                        f"{where}: value certificate ({check.probe}) relative error "
                        f"{check.relative_error:.3g}; reference {check.reference}, "
                        f"emitted {check.emitted}"
                    )
            records.append(replace(row, value_checks=checks))
        return tuple(records)

    def finish(self) -> CountCertificate:
        """The certificate; raises :class:`CertificateError` after a failure."""
        certificate, error = self.outcome()
        if error is not None:
            raise error
        return certificate

    def outcome(self) -> tuple[CountCertificate, CertificateError | None]:
        """The certificate and, after a failure, the :class:`CertificateError`
        to raise.  An engine raises it itself after dropping its terms, so
        that the traceback does not keep them."""
        has_classes = self.engine in _CLASS_PATHS and any(
            record.order > 0 for record in self._orders)
        encoding = None
        if has_classes:
            from ._canonical import ENCODING_VERSION

            encoding = ENCODING_VERSION
        certificate = CountCertificate(
            schema=CERTIFICATE_SCHEMA,
            engine=self.engine,
            target=self.target,
            observable=tuple(repr(op) for op in self.observable),
            orders=tuple(self._orders),
            encoding=encoding,
            canonical=self.canonical,
        )
        if self._failures:
            return certificate, CertificateError(
                self._message(), certificate, tuple(self._failures))
        return certificate, None

    # --- checks -------------------------------------------------------

    def _message(self) -> str:
        lines = [f"certificate of {self.engine} (target {self.target!r}) failed:"]
        lines += [f"  {failure}" for failure in self._failures[:_SHOWN_FAILURES]]
        if len(self._failures) > _SHOWN_FAILURES:
            lines.append(f"  and {len(self._failures) - _SHOWN_FAILURES} more")
        lines.append(_INDEPENDENCE)
        return "\n".join(lines)

    def _row(self, tally: OrderTally, index: int) -> SequenceCount:
        n = tally.order
        vertex_seq, multinomial = tally.combinations[index]
        names = tuple(str(v.coupling) for v in vertex_seq)
        where = _row_label(n, index, names)
        counter = tally.counters[index]
        if counter is None:
            self._failures.append(
                f"{where}: the counter cannot take this layout "
                f"({tally.refusals[index]}); no comparison was made"
            )
        values = self._compared_values(tally, index, where)
        for name, value in values.items():
            if value is not None and counter is not None and value != counter:
                self._failures.append(
                    f"{where}: the counter gives {counter} valid pairings; "
                    + _COUNT_WORDS[name].format(value)
                )
        r_classes = tally._r_classes[index]
        if r_classes is not None and r_classes[0] != r_classes[1]:
            self._failures.append(
                f"{where}: r_classes yields {r_classes[0]} R classes, "
                f"{r_classes[1]} of them distinct"
            )
        class_fields = self._class_fields(tally, index, vertex_seq, where)
        return SequenceCount(
            index=index,
            vertices=names,
            multinomial=multinomial,
            counter=counter,
            r_classes=None if r_classes is None else r_classes[0],
            note=tally._notes[index],
            **values,
            **class_fields,
        )

    def _compared_values(self, tally: OrderTally, index: int, where: str) -> dict:
        compared = _COMPARED[self.engine][tally.order > 0]
        values: dict[str, int | None] = {}
        if "term_weight" in compared:
            values["term_weight"] = self._weight(tally, index, where)
        if "diagram_infos" in compared:
            values["diagram_infos"] = tally._infos[index]
        if "routings" in compared:
            values["routings"] = tally._routings[index]
        return values

    def _weight(self, tally: OrderTally, index: int, where: str) -> int | None:
        if tally._unreadable[index]:
            self._failures.append(
                f"{where}: a coupling holds a {tally._unreadable[index]}, which "
                "the weight reader cannot count"
            )
            return None
        weight = tally._weights[index]
        if weight.denominator != 1 or weight < 0:
            self._failures.append(
                f"{where}: the emitted terms stand for {weight} pairings, "
                "which is not a non-negative integer"
            )
            return None
        return int(weight)

    def _class_fields(
        self, tally: OrderTally, index: int, vertex_seq: Any, where: str
    ) -> dict:
        if self.engine not in _CLASS_PATHS or tally.order == 0 or tally._notes[index]:
            return {}
        group_order = _group_order(vertex_seq)
        passed = True
        orbit_sum = 0
        for members, aut in tally._classes[index].values():
            if isinstance(aut, str):
                self._failures.append(
                    f"{where}: the point automorphism order of a class of "
                    f"{members} labelled topologies cannot be used: {aut}"
                )
                passed = False
                continue
            if group_order % aut or members * aut != group_order:
                self._failures.append(
                    f"{where}: a class of {members} labelled topologies has "
                    f"|Aut_V| = {aut} and |G| = {group_order}; its size must be "
                    f"|G| / |Aut_V| = {Fraction(group_order, aut)}"
                )
                passed = False
                continue
            orbit_sum += group_order // aut
        return {
            "topologies": tally._topologies[index],
            "classes": len(tally._classes[index]),
            "group_order": group_order,
            "orbit_sum": orbit_sum if passed else None,
        }
