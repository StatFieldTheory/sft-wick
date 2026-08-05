"""Self-consistency driver.

What this is, and what it is not
--------------------------------
A DMFT solution is a *fixed point*: propagators define a self-energy, the
self-energy defines new propagators, repeat.  Three pieces are needed:

  (a) self-energy from the current propagators,
  (b) new propagators from that self-energy (a Dyson / integral-equation
      solve),
  (c) the iteration itself, with the judgement about when it has converged.

sft-wick computes (a) -- that is what the whole package is for.  (b) is
model-specific and is genuinely an integral-equation solve, not a diagram
evaluation; this module does **not** attempt it, because a wrong general Dyson
solver would be worse than none.  What was missing entirely is (c): nothing in
the package iterated, so every DMFT use was one pass of a loop the user had to
write, usually without convergence diagnostics.

So the contract is deliberately thin.  You supply ``step``, a callable that
takes a state and returns the next one -- typically "build the diagrams with
these propagators, get Sigma, solve Dyson, return the new propagators".  This
module runs it, mixes, measures, and is honest about what happened.

Being honest about what happened is the point
----------------------------------------------
A fixed-point iteration that has not converged looks exactly like one that
has, if you only print the last state.  :func:`solve_self_consistency` never
returns a bare state: it returns a
:class:`SelfConsistencyResult` carrying ``converged``, the full residual
history, and a ``reason`` distinguishing the ways it can fail --

* ``"converged"``   -- the residual ``||step(x) - x||`` fell below ``tol``;
* ``"diverged"``    -- the residual is growing, so more iterations will not
  help and the intermediate states may overflow;
* ``"oscillating"`` -- the state has entered a cycle: it is closer to where it
  was TWO steps ago than to where it was one step ago.  Damping is the usual
  cure, and the reason string is what tells you to reach for it;
* ``"max_iter"``    -- none of the above; it simply ran out.

Four categories, each with a sharp test, rather than more with fuzzy ones.

Three things this module is deliberately careful about, because each is a way
to report a solution that was never found:

**Damping must not shrink the residual.**  The state moves by only
``(1 - damping)`` of the step, so a residual read off the state's *movement*
falls as damping rises -- turn damping up and any iteration "converges"
sooner, at a point that is not a fixed point.  The residual here always
measures the step, ``||step(x) - x||``.

**A cycle is a statement about the state, not the residual.**  A two-cycle has
a *constant* residual, indistinguishable from a stall by residual shape alone;
a *growing* residual is divergence, not oscillation.  So cycles are found by
asking whether the state has come back somewhere it has already been.  The
converse trap is just as real: an alternating but genuinely CONVERGENT map
(``x -> a x + b`` with ``a`` just above ``-1``) also returns nearly to where
it was every step, so the cycle test additionally requires that the residual
has stopped improving.

**Growth must be measured against a recent baseline.**  Comparing against the
best residual ever seen condemns any run that starts near a repelling point,
drifts away, and then converges -- "start from the non-interacting solution
and find the interacting one" is exactly that shape.

The caller decides what to do about each; the library's job is to tell them
apart rather than return a plausible-looking number.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

__all__ = [
    "SelfConsistencyResult",
    "max_abs_distance",
    "solve_self_consistency",
]

#: How many past residuals the growth test compares against.  Long enough to
#: let a transient rise and fall, short enough to still catch slow divergence.
_DIVERGENCE_WINDOW = 8

#: How many past states the cycle test keeps.  A window of ``k`` detects
#: cycles of period 2 through ``k + 1``.
_CYCLE_HISTORY = 4


@dataclass(frozen=True)
class SelfConsistencyResult:
    """Outcome of a fixed-point iteration.

    Attributes:
        state: the final state -- **not** necessarily a converged one; check
            :attr:`converged` first.
        converged: whether the residual fell below ``tol``.
        reason: ``"converged"``, ``"diverged"``, ``"oscillating"`` or
            ``"max_iter"``.
        n_iter: iterations actually performed.
        residuals: the residual after each iteration, in order.
    """

    state: Any
    converged: bool
    reason: str
    n_iter: int
    residuals: tuple = field(default=())

    @property
    def residual(self) -> float:
        """The final residual, or ``inf`` if no iteration ran."""
        return float(self.residuals[-1]) if self.residuals else float("inf")

    def __bool__(self) -> bool:
        """``bool(result)`` is :attr:`converged`.

        So ``if not result: ...`` is the natural way to handle a failure, and
        a caller who ignores the outcome entirely and uses ``result.state``
        has at least had to name it.
        """
        return bool(self.converged)

    def summary(self) -> str:
        return (f"{self.reason} after {self.n_iter} iteration(s), "
                f"residual {self.residual:.3e}")


def _as_numeric(x, where: str) -> np.ndarray:
    """``np.asarray`` without forcing a dtype, then check it is numeric.

    Deliberately **not** ``dtype=float``: numpy silently discards the
    imaginary part of a complex array (it warns once per source line, after
    which the warning registry mutes it), and DMFT propagators are routinely
    complex -- sft-wick's own diagram values carry ``i^(-E_psi)`` phases.  A
    distance taken on real parts only reports convergence while the imaginary
    parts are still moving.
    """
    arr = np.asarray(x)
    if arr.dtype.kind not in "biufc":
        raise TypeError(
            f"{where}: states must be numeric (or nested containers of "
            f"numeric arrays); got an array of dtype {arr.dtype!r}."
        )
    return arr


def _rebuild(template: Any, values: list) -> Any:
    """Rebuild a container of ``template``'s type from ``values``.

    Type preservation is a contract, not a nicety: ``step`` must receive the
    same type on iteration 2 that it received on iteration 1, and
    ``result.state``'s type must not depend on whether ``damping`` is zero.
    Namedtuples take their fields positionally, and dict/list SUBCLASSES have
    to be rebuilt as themselves or the caller's methods vanish after one pass.
    """
    if isinstance(template, tuple):
        if hasattr(template, "_fields"):              # namedtuple
            return type(template)(*values)
        return type(template)(values)
    if isinstance(template, list):
        return values if type(template) is list else type(template)(values)
    return values


def _rebuild_mapping(template: Any, pairs: dict) -> Any:
    """Same, for mappings."""
    if type(template) is dict:
        return pairs
    try:
        return type(template)(pairs)
    except Exception:
        return pairs


def _copy_state(x: Any) -> Any:
    """A copy deep enough that ``step`` cannot corrupt the iteration.

    ``step`` is handed a copy, so the numpy idiom ``x += dx; return x`` --
    which returns the object it was given -- cannot make the residual
    identically zero and fake convergence.

    The fallback for an unrecognised type is ``copy.deepcopy``, NOT the
    object itself.  Returning it unchanged would silently withdraw the
    guarantee for exactly the states this module does not enumerate (a pandas
    Series, a dataclass, a tensor), which is the worst place to withdraw it:
    the caller has no signal that the protection lapsed.
    """
    if isinstance(x, dict):
        return _rebuild_mapping(x, {k: _copy_state(v) for k, v in x.items()})
    if isinstance(x, (list, tuple)):
        return _rebuild(x, [_copy_state(v) for v in x])
    if isinstance(x, np.ndarray):
        return copy.deepcopy(x) if x.dtype.kind == "O" else x.copy()
    if x is None or isinstance(x, (bool, int, float, complex, np.number)):
        return x                                       # immutable
    try:
        return copy.deepcopy(x)
    except Exception as exc:                           # pragma: no cover
        raise TypeError(
            f"cannot copy a state of type {type(x).__name__}, so `step` "
            f"cannot be protected from mutating it -- which would make the "
            f"residual identically zero and report convergence at the "
            f"starting point.  Supply a state built from arrays, dicts, "
            f"lists or tuples, or make it deep-copyable."
        ) from exc


def _state_size(x) -> int:
    """Total number of scalar elements in a state."""
    if isinstance(x, dict):
        return sum(_state_size(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(_state_size(v) for v in x)
    return int(np.asarray(x).size)


def max_abs_distance(a: Any, b: Any) -> float:
    """Largest absolute difference between two states.

    Handles a scalar, an array, or any (possibly nested) sequence or mapping
    of those -- which covers the ``(R, C)`` pairs and ``{name: array}`` dicts
    a DMFT state usually is.  Complex arrays are compared by MODULUS, not by
    real part.  Raises on structures that do not match, rather than comparing
    whatever happens to line up.
    """
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            raise TypeError(f"cannot compare {type(a).__name__} with "
                            f"{type(b).__name__}")
        if set(a) != set(b):
            raise ValueError(
                f"states have different keys: {sorted(a)} vs {sorted(b)}")
        if not a:
            return 0.0
        return max(max_abs_distance(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            raise TypeError(f"cannot compare {type(a).__name__} with "
                            f"{type(b).__name__}")
        if len(a) != len(b):
            raise ValueError(
                f"states have different lengths: {len(a)} vs {len(b)}")
        if not a:
            return 0.0
        return max(max_abs_distance(x, y) for x, y in zip(a, b))
    aa = _as_numeric(a, "max_abs_distance")
    bb = _as_numeric(b, "max_abs_distance")
    if aa.shape != bb.shape:
        raise ValueError(f"states have different shapes: {aa.shape} vs "
                         f"{bb.shape}")
    if aa.size == 0:
        return 0.0
    # Subtract in a PROMOTED dtype.  In the input's own integer dtype the
    # difference wraps modulo 2**nbits, so a state far from the fixed point
    # can report a distance small enough to satisfy `tol` -- convergence
    # declared by overflow.  np.abs is the modulus for complex input and
    # propagates NaN, so a NaN anywhere surfaces as a non-finite residual.
    diff = np.subtract(aa, bb,
                       dtype=np.result_type(aa, bb, np.float64))
    return float(np.max(np.abs(diff)))


def _mix(new: Any, old: Any, damping: float) -> Any:
    """``(1 - damping) * new + damping * old``, structure-preserving."""
    if damping == 0.0:
        return new
    if isinstance(new, dict):
        return _rebuild_mapping(
            new, {k: _mix(new[k], old[k], damping) for k in new})
    if isinstance(new, (list, tuple)):
        return _rebuild(new, [_mix(x, y, damping) for x, y in zip(new, old)])
    # Native arithmetic FIRST, so the leaf keeps its own type.  Coercing
    # through np.asarray would hand `step` a plain ndarray from iteration 2
    # onward -- the same "step never receives its own type" defect the
    # container branches above avoid, and for a device array (JAX, torch) it
    # also silently moves the state back to the host after one iteration.
    # Falls back to numeric coercion for a type that cannot do the arithmetic.
    try:
        mixed = (1.0 - damping) * new + damping * old
    except TypeError:
        mixed = None
    if mixed is not None:
        return mixed
    # no dtype coercion here either: a complex state must stay complex
    return (1.0 - damping) * _as_numeric(new, "_mix") + \
        damping * _as_numeric(old, "_mix")


def solve_self_consistency(
    initial: Any,
    step: Callable[[Any], Any],
    *,
    distance: Callable[[Any, Any], float] | None = None,
    tol: float = 1e-8,
    max_iter: int = 100,
    damping: float = 0.0,
    divergence_factor: float = 1e3,
    cycle_tol: float = 0.1,
    callback: Callable[[int, Any, float], None] | None = None,
) -> SelfConsistencyResult:
    """Iterate ``step`` from ``initial`` until the state stops moving.

    Args:
        initial: the starting state.  Anything ``step`` accepts and
            ``distance`` can compare.
        step: ``state -> next_state``.  For DMFT this is "build the diagrams
            with these propagators, extract the self-energy, solve Dyson".
            It is handed a **copy** of the state, so it may mutate its
            argument in place -- the numpy idiom ``x += dx; return x`` returns
            the object it was given, which would otherwise make the residual
            identically zero and fake convergence on iteration 1.
        distance: how far apart two states are; defaults to
            :func:`max_abs_distance`.
        tol: converged when the residual ``||step(x) - x||`` drops below
            this.  Measured on the step, not on the damped movement, so the
            criterion does not loosen as ``damping`` rises.  It is an
            **absolute** tolerance: choose it against the scale of your state,
            since a state whose entries all sit far below ``tol`` satisfies it
            whatever ``step`` does.
        max_iter: give up after this many iterations.
        damping: linear mixing in ``[0, 1)``.  ``0`` is plain iteration;
            larger values trade speed for stability and are the usual cure for
            an ``"oscillating"`` result.  Mixing happens **after** ``step``,
            so ``step`` always sees a genuine state.
        divergence_factor: stop early once the residual exceeds this multiple
            of the smallest residual in the recent window (the last
            ``8`` iterations).  Deliberately *not* the best residual ever
            seen: that would condemn a run which drifts away from a repelling
            starting point before converging.
        cycle_tol: call it a cycle when the state comes back within this
            fraction of one step's movement of a state it visited two to five
            steps ago -- and the residual has stopped improving, meaning it
            fell by less than 1% over two iterations.  For a linear map
            ``x -> a x + b`` that cut lands at ``|a| >= sqrt(0.99) =
            0.99499``: a contraction slower than that is reported
            ``"oscillating"`` rather than run to convergence, because plain
            iteration would need thousands of steps there and damping is the
            real answer.  Cycles of period 6 or more are not detected.
        callback: called as ``callback(iteration, state, residual)`` after
            each iteration, with the POST-mixing state -- for progress output
            or for recording the trajectory.

    Returns:
        A :class:`SelfConsistencyResult`.  **Check ``converged`` before using
        ``state``**: a non-converged final state is returned too, because
        inspecting it is usually how you work out why.
    """
    if not (0.0 <= damping < 1.0):
        raise ValueError(f"damping must be in [0, 1); got {damping!r}.")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1; got {max_iter!r}.")
    if tol < 0:
        raise ValueError(f"tol must be >= 0; got {tol!r}.")
    if _state_size(initial) == 0:
        raise ValueError(
            "initial state contains no elements, so every distance is 0 and "
            "the iteration would report convergence immediately.  This is "
            "almost always an upstream bug (an empty time grid, a "
            "filtered-away component) rather than an intended input."
        )

    dist = distance if distance is not None else max_abs_distance
    state = initial
    history: deque = deque(maxlen=_CYCLE_HISTORY)   # states >= 2 steps back
    residuals: list[float] = []
    cycle_streak = 0

    for it in range(1, int(max_iter) + 1):
        # `step` gets a COPY: an in-place update that returns its own argument
        # would otherwise make `proposed is state`, hence residual 0.
        proposed = step(_copy_state(state))
        # The residual measures the STEP, ||F(x) - x||, not the state's
        # movement.  Those differ by a factor (1 - damping): measuring the
        # movement would make the convergence test depend on a numerical
        # stabilisation knob, so turning damping up would "converge" a
        # calculation that had not.
        res = float(dist(proposed, state))
        residuals.append(res)
        mixed = _mix(proposed, state, damping)
        if mixed is proposed:
            # `_mix` short-circuits to `proposed` at damping == 0 -- the
            # DEFAULT.  If `step` returns a buffer it owns and reuses (the
            # standard preallocated-output idiom), `state` would then alias
            # that buffer, and the next call would overwrite `state` in place
            # BEFORE the residual is taken: dist(proposed, state) compares the
            # buffer with itself, giving exactly 0.0 and "converged" at
            # whatever the second iterate happened to be.  Copying the input
            # to `step` does not help; the aliasing is on the output side.
            mixed = _copy_state(mixed)
        prev, state = state, mixed
        if callback is not None:
            callback(it, state, res)

        if not np.isfinite(res):
            return SelfConsistencyResult(state, False, "diverged", it,
                                         tuple(residuals))
        if res <= tol:
            return SelfConsistencyResult(state, True, "converged", it,
                                         tuple(residuals))

        # --- Divergence first: a growing residual is divergence, whatever
        # else it also resembles.  Measured against the recent window, not
        # the all-time best, so a run that drifts off a repelling start and
        # then converges is not condemned on the way up. ---
        if res > divergence_factor * min(residuals[-_DIVERGENCE_WINDOW:]):
            return SelfConsistencyResult(state, False, "diverged", it,
                                         tuple(residuals))
        # Fast path for an obvious blow-up: three consecutive increases AND an
        # order of magnitude gained across them (~2.2x per step sustained).
        # The magnitude gate is not decoration -- monotonicity alone would
        # condemn a physical transient that rises a few percent then turns.
        if (len(residuals) >= 4
                and all(b > a for a, b in zip(residuals[-4:], residuals[-3:]))
                and residuals[-1] > 10.0 * residuals[-4]):
            return SelfConsistencyResult(state, False, "diverged", it,
                                         tuple(residuals))

        # --- Cycles, measured on the STATE. ---
        # `move` is the actual movement, not the residual: damping shrinks one
        # and not the other, so a ratio against the residual would call every
        # heavily-damped run a cycle.  Comparing against several past states
        # catches period 3, 4, 5 as well as 2 -- reporting those as "max_iter"
        # would tell the caller to raise max_iter, which never terminates.
        move = float(dist(state, prev))
        returned = move > 0 and any(
            float(dist(state, old)) <= cycle_tol * move for old in history
        )
        # An alternating CONTRACTION (x -> a x + b, a just above -1) also
        # returns nearly to where it was each step -- the ratio is |1 + a| --
        # but its residual keeps falling.  A real cycle's does not.
        improving = (len(residuals) >= 3
                     and residuals[-1] < 0.99 * residuals[-3])
        if returned and not improving:
            cycle_streak += 1
            if cycle_streak >= 2:        # must persist, not be a single fluke
                return SelfConsistencyResult(state, False, "oscillating", it,
                                             tuple(residuals))
        else:
            cycle_streak = 0
        history.append(prev)

    return SelfConsistencyResult(state, False, "max_iter", int(max_iter),
                                 tuple(residuals))
