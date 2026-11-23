"""Versioned, JSON-compatible transport for Wick diagram terms.

See the WickIR guide for the wire format and numerical conventions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from math import isfinite
from numbers import Integral
from operator import index
from typing import Any

from ._ir_identity import diagram_ids
from ._ir_sympy import from_sympy, to_sympy
from .evaluate import (
    _conflicting_group_positions,
    _external_positions,
    _r_index_mismatch,
    _resolve_external_times,
    _resolve_integrate_over,
    analyze_spatial,
)
from .expressions import (
    DiracDelta,
    Expr,
    ImaginaryUnit,
    IntegralOver,
    KroneckerDelta,
    Product,
    Propagator,
    Rational,
    Sum,
    SumOverIndex,
    Symbol,
)
from .perturbation import DiagramTerm

__all__ = ["diagram_ids", "from_ir", "from_sympy", "ir_digest", "to_ir", "to_sympy"]

SCHEMA = "wickir/v1"

# Field names, wire tags, and types are explicit so a dataclass refactor
# cannot silently change the wire format. A list of fields in a descriptor
# denotes a fixed-length array; a one-element tuple denotes a repeated type.
_NODES = {
    Rational: ("rational", {"numerator": int, "denominator": int}),
    Symbol: ("symbol", {
        "name": str, "indices": (str,), "spatial_args": (str,), "local": bool,
    }),
    Propagator: ("propagator", {
        "kind": str, "index_left": str | None, "index_right": str | None,
        "spatial_left": str, "spatial_right": str,
    }),
    Sum: ("sum", {"terms": (Expr,)}),
    Product: ("product", {"factors": (Expr,)}),
    SumOverIndex: ("sum_over_index", {"index_name": str, "dimension": int, "body": Expr}),
    IntegralOver: ("integral_over", {"variable": str, "body": Expr}),
    KroneckerDelta: ("kronecker_delta", {"index1": str, "index2": str}),
    DiracDelta: ("dirac_delta", {"arg1": str, "arg2": str}),
    ImaginaryUnit: ("imaginary_unit", {}),
}
_TAGS = {tag: (cls, fields) for cls, (tag, fields) in _NODES.items()}
_DIAGRAM = {
    "propagators": (Propagator,),
    "coupling_sum": Expr,
    "rational_prefactor": Rational,
    "integration_vars": (str,),
    "summation_indices": ([str, int],),
    "n_response": int,
    "equal_time_aliases": ([str, str],),
    "r_absorbed_pairs": ([str, str],),
    "n_external_response": int,
    # Each entry is [ordered point labels, template]. The template carries
    # name, local/equal-time/absorbed flags, ordered fields and leg symmetry.
    "vertex_types": ([(str,), [str, bool, bool, bool, ([str, str, int],), str]],),
}


def _encode(value, kind):
    if kind is Expr or kind in (Propagator, Rational):
        if kind is not Expr and type(value) is not kind:
            raise TypeError(f"WickIR expects {kind.__name__}, got {type(value).__name__}")
        if type(value) is Propagator and value.kind not in ("R", "C"):
            raise ValueError("WickIR propagator kind must be R or C")
        try:
            tag, fields = _NODES[type(value)]
        except KeyError:
            raise TypeError(f"WickIR cannot encode {type(value).__name__}") from None
        return {"type": tag, **_encode_fields(value, fields)}
    if isinstance(kind, tuple):
        return [_encode(item, kind[0]) for item in value]
    if isinstance(kind, list):
        return [_encode(item, sub) for item, sub in zip(value, kind, strict=True)]
    if kind in (str, str | None) and isinstance(value, str):
        return str(value)
    if kind is int and type(value) is not bool:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        try:
            return index(value)
        except TypeError:
            pass
    return _decode(value, kind)


def _encode_fields(value, fields):
    return {name: _encode(getattr(value, name), kind) for name, kind in fields.items()}


def _require_keys(value, names, context):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"invalid WickIR {context} fields")


def _decode(value, kind):
    if kind is Expr or kind in (Propagator, Rational):
        tag = value.get("type") if isinstance(value, dict) else None
        if not isinstance(tag, str) or tag not in _TAGS:
            raise ValueError(f"unknown WickIR expression type {tag!r}")
        cls, fields = _TAGS[tag]
        if kind is not Expr and cls is not kind:
            raise ValueError(f"expected WickIR {kind.__name__}, got {tag}")
        _require_keys(value, ("type", *fields), tag)
        args = _decode_fields(value, fields)
        if cls is Propagator and args["kind"] not in ("R", "C"):
            raise ValueError("WickIR propagator kind must be R or C")
        if cls is Rational and not args["denominator"]:
            raise ValueError("WickIR rational denominator must be nonzero")
        return cls(**args)
    if isinstance(kind, (tuple, list)):
        if not isinstance(value, list):
            raise ValueError("WickIR sequence must be a JSON array")
        if isinstance(kind, tuple):
            return tuple(_decode(item, kind[0]) for item in value)
        if len(value) != len(kind):
            raise ValueError(f"WickIR array must contain {len(kind)} entries")
        return tuple(_decode(item, sub) for item, sub in zip(value, kind, strict=True))
    if kind == (str | None):
        if value is None or isinstance(value, str):
            return value
    elif type(value) is kind:
        return value
    raise ValueError(f"invalid WickIR value {value!r} for {kind}")


def _decode_fields(value, fields):
    return {name: _decode(value[name], kind) for name, kind in fields.items()}


def _conventions(term, *, target, iso_C, iso_R, t_min, integrate_over):
    if target not in ("moment", "connected"):
        raise ValueError("WickIR target must be moment or connected")
    for name, value in (("iso_C", iso_C), ("iso_R", iso_R)):
        if value is not None and type(value) is not bool:
            raise ValueError(f"WickIR {name} must be a boolean or null")
    return {
        "target": target,
        "ito": True,
        "theta_at_zero": 0,
        "response_phase": term.response_phase,
        "numeric_response_phase": "(-i)^n_response",
        "observable_phase": "i^n_external_response",
        "R_endpoint_order": ["physical", "response"],
        "iso_R": iso_R,
        "iso_C": iso_C,
        "iso_C_scalar": "common_diagonal",
        "coupling_input": "msr_adjusted",
        "bare_to_msr": {
            "local_drift": "-i",
            "nonlocal_cumulant": "-(i^m)/m! (m=rank)",
            "multiplicative_covariance": "1/2",
            "stratonovich_drift": "-i",
            "kramers_moyal": "-(i^m)/m! (m=n_psi)",
            "initial_mean_observable": "+i",
            "initial_mean_translated": "inherit",
        },
        "observable_time": {
            "t_min": t_min,
            "integrate_over": integrate_over,
            "default_pinned_time": "t_final",
            "integration_ceiling": "max(t_final, *external_times.values())",
            "swept_external_ceiling": "t_final",
        },
    }


def _finite_float(value):
    value = float(value)
    if not isfinite(value):
        raise ValueError("WickIR times and positions must be finite")
    return value


def _coordinate(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        return [_coordinate(item) for item in value]
    if type(value) is int:
        return value
    return _finite_float(value)


def _named_values(values, encode, context):
    if values is None:
        return None
    if any(not isinstance(key, str) for key in values):
        raise ValueError(f"WickIR {context} needs external point names as strings")
    return {key: encode(value) for key, value in values.items()}


def _spatial_check(term, positions, integrate_over, external_times, t_final):
    """Use the production guards for the exported structural cross-check."""
    try:
        spatial = analyze_spatial(term)
    except ValueError as error:
        return None, {"code": "invalid_time_ordering", "message": str(error)}

    check = asdict(spatial)
    check["direction_groups"] = [sorted(group) for group in spatial.direction_groups]
    conflict = _conflicting_group_positions(spatial, _external_positions(spatial, positions))
    if conflict:
        return check, {"code": "r_group_position_conflict", "points": list(conflict)}
    try:
        swept = _resolve_integrate_over(integrate_over, list(spatial.external_points))
    except ValueError as error:
        return check, {"code": "invalid_integrated_points", "message": str(error)}
    try:
        _resolve_external_times(
            spatial, set(spatial.external_points) - swept,
            t_final if t_final is not None else 0.0, external_times, swept,
        )
    except ValueError as error:
        return check, {"code": "invalid_external_times", "message": str(error)}
    return check, None


def to_ir(
    term: DiagramTerm,
    *,
    target: str = "moment",
    iso_R: bool | None = None,
    iso_C: bool | None = None,
    t_min: float = 0.0,
    integrate_over=None,
    positions=None,
    external_times=None,
    t_final: float | None = None,
    n_components: int | None = None,
) -> dict[str, Any]:
    """Encode a diagram and optional evaluation context in ``wickir/v1``.

    Supply the originating expansion's target and isotropy flags when known,
    and the system's lower time bound. A raw term does not retain them.
    ``None`` for an isotropy flag means its provenance is unspecified.
    Pass the cache's ``n_components`` to check absorbed R indices as the
    production evaluator does. With unknown dimension, an absorbed R index
    mismatch is recorded as ``evaluation.conditional_rejection`` and applies
    only to multi-component caches. Definite evaluation rejections are
    recorded under ``evaluation.rejection``.
    """
    if type(term) is not DiagramTerm:
        raise TypeError("to_ir expects a DiagramTerm")
    if n_components is not None:
        if not isinstance(n_components, Integral) or isinstance(n_components, bool) or n_components < 1:
            raise ValueError("WickIR n_components must be a positive integer")
        n_components = int(n_components)
    if integrate_over is not None and not isinstance(integrate_over, str):
        points = tuple(integrate_over)
        if any(not isinstance(point, str) for point in points):
            raise ValueError("WickIR integrate_over needs external point names as strings")
        integrate_over = sorted(set(points))
    positions = _named_values(positions, _coordinate, "positions")
    external_times = _named_values(external_times, _finite_float, "external_times")
    if t_final is not None:
        t_final = _finite_float(t_final)
    spatial, rejection = _spatial_check(term, positions, integrate_over, external_times, t_final)
    evaluation = {
        "positions": positions,
        "external_times": external_times,
        "t_final": t_final,
        "rejection": rejection,
    }
    if n_components is not None:
        evaluation["n_components"] = n_components
    if rejection is None and n_components != 1 and term.r_absorbed_pairs:
        absorbed = set(term.r_absorbed_pairs)
        edges = tuple(
            prop for prop in term.propagators
            if (prop.spatial_left, prop.spatial_right) in absorbed
        )
        # For these edges the guard gives the same answer for either R type.
        mismatch = _r_index_mismatch(replace(term, propagators=edges), iso_R=True)
        if mismatch:
            diagnostic = {"code": "invalid_absorbed_r_indices", "message": mismatch}
            if n_components is None:
                evaluation["conditional_rejection"] = {**diagnostic, "min_n_components": 2}
            else:
                evaluation["rejection"] = diagnostic
    record = {
        "schema": SCHEMA,
        "conventions": _conventions(
            term, target=target, iso_R=iso_R, iso_C=iso_C,
            t_min=_finite_float(t_min), integrate_over=integrate_over,
        ),
        "diagram": _encode_fields(term, _DIAGRAM),
        "spatial_structure": spatial,
        "evaluation": evaluation,
    }
    # asdict preserves tuples; ensure callers receive only JSON types.
    return json.loads(_json_bytes(record))


def from_ir(record: dict[str, Any]) -> DiagramTerm:
    """Recreate the term of a parsed WickIR record using known node types.

    Context and cross-checks remain in the record. They are never used to
    reconstruct edges, coupling expressions, phases or integration labels.
    """
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError(f"expected WickIR schema {SCHEMA!r}")
    _require_keys(
        record, ("schema", "conventions", "diagram", "spatial_structure", "evaluation"),
        "record",
    )
    diagram = record["diagram"]
    _require_keys(diagram, _DIAGRAM, "diagram")
    conventions = record["conventions"]
    if not isinstance(conventions, dict) or type(conventions.get("response_phase")) is not bool:
        raise ValueError("WickIR conventions must specify a boolean response_phase")
    if conventions.get("ito") is not True or type(conventions.get("theta_at_zero")) is not int:
        raise ValueError("WickIR v1 requires Itô with integer theta_at_zero=0")
    if conventions["theta_at_zero"] != 0:
        raise ValueError("WickIR v1 requires theta_at_zero=0")
    args = _decode_fields(diagram, _DIAGRAM)
    term = DiagramTerm(**args, response_phase=conventions["response_phase"])
    time = conventions.get("observable_time")
    if not isinstance(time, dict) or type(time.get("t_min")) not in (int, float):
        raise ValueError("WickIR conventions must specify a finite t_min")
    swept = time.get("integrate_over")
    if swept is not None and not isinstance(swept, str):
        _decode(swept, (str,))
    expected = _conventions(
        term, target=conventions.get("target"), iso_C=conventions.get("iso_C"),
        iso_R=conventions.get("iso_R"), t_min=_finite_float(time["t_min"]),
        integrate_over=swept,
    )
    if "swept_external_ceiling" not in time:
        # Early v1 records left the swept external ceiling implicit.
        del expected["observable_time"]["swept_external_ceiling"]
    if conventions != expected:
        raise ValueError("unsupported WickIR v1 conventions")
    return term


def _json_bytes(record):
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("ascii")


def ir_digest(record: dict[str, Any]) -> str:
    """SHA-256 of the full IR record in canonical JSON encoding."""
    return hashlib.sha256(_json_bytes(record)).hexdigest()
