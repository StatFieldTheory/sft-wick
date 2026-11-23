"""Integration domains for fixed delays and zero prehistory.

After substituting each delayed time, a causal edge, a history bound or
a covariance kink has the form x_j - x_i <= c, including x_0 = 0 for
constant limits. Shortest-path closure projects these inequalities exactly
onto any prefix of integration coordinates. The same projected domain
serves adaptive quadrature, tensor quadrature and Sobol sampling.
"""

from fractions import Fraction
from itertools import combinations
from math import inf, isfinite, nextafter, ulp
from numbers import Integral, Real
from operator import itemgetter

import numpy as np


def _time_fraction(value):
    """Keep the exact input value, including NumPy's scalar float types."""
    if isinstance(value, Integral):
        return Fraction(int(value))
    # Older Python versions do not accept a NumPy float32 directly.
    if isinstance(value, np.floating):
        return Fraction(*value.as_integer_ratio())
    return Fraction(value)


def normalize_lags(lags, aliases, internal):
    """Resolve chains once, retaining the spatial labels of all points."""
    from .vertices import flatten_group_lags

    links = {}
    for delayed, anchor, lag in lags:
        delayed = aliases.get(delayed, delayed)
        anchor = aliases.get(anchor, anchor)
        if delayed not in internal or anchor not in internal:
            raise ValueError("fixed time lags must link internal vertex groups")
        if delayed in links or delayed == anchor:
            raise ValueError("fixed time lags need one distinct parent per group")
        if (
            not isinstance(lag, Real)
            or isinstance(lag, bool)
            or not isfinite(lag)
            or lag <= 0
        ):
            raise ValueError("fixed time lags must be finite positive real numbers")
        links[delayed] = (anchor, _time_fraction(lag))
    return flatten_group_lags(links)


def _clock_precision_error(delayed, root):
    return ValueError(
        f"Fixed delay at {delayed!r} relative to {root!r} cannot be resolved "
        "at floating-point clock precision. Shift the time origin or express "
        "the model in elapsed times that retain the delay."
    )


def _clock_spacing(low, high):
    # Step inward at a power-of-two endpoint; the grid on the other
    # side can have twice the spacing inside this interval.
    return max(nextafter(low, inf) - low, high - nextafter(high, -inf))


def shifted_values(time_of, lags):
    """Read actual leg times, preserving the strict order of distinct offsets.

    For an arbitrary user kernel the absolute coordinates themselves must
    distinguish the points. Moving a rounded time to its next float would
    change the requested lag and the value of such a kernel.
    """
    values = {}
    last_root = None
    for label, root, lag in sorted(lags, key=itemgetter(1, 2)):
        if root != last_root:
            raw = time_of(root)
            anchor = (
                np.asarray(raw, dtype=float)
                if isinstance(raw, np.ndarray)
                else float(raw)
            )
            previous = anchor
            previous_lag = 0.0
            last_root = root
        if lag != previous_lag:
            value = anchor - lag
            earlier = value < previous
            if isinstance(earlier, np.ndarray):
                # Batch samplers have already admitted their full root range.
                resolved = earlier.all()
            else:
                resolved = earlier and lag - previous_lag >= _clock_spacing(
                    value, previous
                )
            if not resolved:
                raise _clock_precision_error(label, root)
            previous, previous_lag = value, lag
        values[label] = previous
    return values


def delayed_times(times, lags):
    """Return scalar times including each delayed representative."""
    return {**times, **shifted_values(times.__getitem__, lags)}


def _check_clock_intervals(lags, coordinates, matrix):
    """Require distinguishable absolute times over each feasible anchor range."""
    previous_lags = {}
    for label, root, lag in sorted(lags, key=itemgetter(1, 2)):
        previous = previous_lags.get(root, 0.0)
        previous_lags[root] = lag
        if lag == previous:
            continue
        i, _ = coordinates[root]
        low = float(-matrix[i][0] - Fraction(lag))
        high = float(matrix[0][i] - Fraction(previous))
        if Fraction(lag) - Fraction(previous) < _clock_spacing(low, high):
            raise _clock_precision_error(label, root)


def _domain_precision_error(projection):
    return ValueError(
        f"Fixed-delay integration cannot resolve an interior sample of {projection} "
        "at floating-point clock precision. Shift the time origin or express "
        "the model in elapsed times."
    )


def _check_clock_projections(names, matrix):
    """Refuse domains with fewer than two interior ticks in a projection."""
    steps = []
    for i, name in enumerate(names, 1):
        low, high = -matrix[i][0], matrix[0][i]
        nearest = float(low)
        first = nearest if nearest > low else nextafter(nearest, inf)
        if nextafter(first, inf) >= high:
            raise _domain_precision_error(f"clock {name!r}")
        nearest_zero = max(low, min(0, high))
        steps.append(Fraction(ulp(float(nearest_zero))))

    # A time difference lies on the finer of its clocks' two lattices.
    # This catches a thin strip while allowing ordinary triangles which
    # are narrow only near a tip.
    for i, j in combinations(range(1, len(matrix)), 2):
        step = min(steps[i - 1], steps[j - 1])
        low, high = -matrix[j][i], matrix[i][j]
        if high - low > 2 * step:
            continue
        if (low // step + 2) * step >= high:
            raise _domain_precision_error(
                f"time difference {names[j - 1]!r} - {names[i - 1]!r}"
            )


def _require_interior_sample(name, value, low, high, active=True):
    if np.any(active & ((value <= low) | (value >= high))):
        raise _domain_precision_error(f"clock {name!r}")


def _closure(matrix):
    """Tightest upper bound on x_j - x_i, for every (i,j)."""
    bounds = [row.copy() for row in matrix]
    for k, row in enumerate(bounds):
        for i, outer in enumerate(bounds):
            through = outer[k]
            if through != inf:
                for j, limit in enumerate(row):
                    if limit != inf:
                        outer[j] = min(outer[j], through + limit)
    return bounds


def _positive_volume(bounds):
    # Opposite inequalities summing to zero fix the difference exactly
    # between two otherwise free coordinates (one can be the origin).
    return all(bounds[i][i] >= 0 for i in range(len(bounds))) and all(
        bounds[i][j] + bounds[j][i] > 0 for i, j in combinations(range(len(bounds)), 2)
    )


def _active_faces(matrix, index, upper):
    """Discard bounds that another face always tightens on the projection."""
    offsets = [matrix[j][index] if upper else -matrix[index][j] for j in range(index)]
    active = []
    for j, shift in enumerate(offsets):
        dominated = any(
            k != j
            and (
                matrix[j][k] <= shift - other
                if upper
                else matrix[k][j] <= other - shift
            )
            for k, other in enumerate(offsets)
        )
        if not dominated:
            active.append((j, shift))
    return active


class ShiftedBounds:
    """Projected lower and upper limits, in outermost-first draw order."""

    def __init__(self, names, matrix, clock_checks=()):
        self.names = names
        self.clock_checks = clock_checks
        self.empty = not _positive_volume(matrix)
        self.faces = (
            [
                (
                    [(j, float(x)) for j, x in _active_faces(matrix, index, False)],
                    [(j, float(x)) for j, x in _active_faces(matrix, index, True)],
                )
                for index in range(1, len(names) + 1)
            ]
            if not self.empty
            else []
        )

    @staticmethod
    def limits(faces, previous):
        lower, upper = faces
        lo = previous[lower[0][0]] + lower[0][1]
        hi = previous[upper[0][0]] + upper[0][1]
        for j, offset in lower[1:]:
            lo = np.maximum(lo, previous[j] + offset)
        for j, offset in upper[1:]:
            hi = np.minimum(hi, previous[j] + offset)
        return lo, hi

    def _check_reconstructed_clocks(self, roots, active=True):
        # Both causal edges and the history floor are strict inside the
        # exact domain. Its roots can remain interior even when subtracting
        # their lags rounds one of these constraints to equality.
        for description, (i, left), (j, right) in self.clock_checks:
            if np.any(active & (roots[i] + left <= roots[j] + right)):
                raise _domain_precision_error(description)

    def sample(self, nodes, columns):
        times = np.zeros_like(nodes)
        jacobians = np.ones(len(nodes))
        previous = [0.0]
        if self.empty:
            return times, np.zeros(len(nodes))
        interior = np.ones(len(nodes), dtype=bool)
        for name, faces in zip(self.names, self.faces):
            lo, hi = self.limits(faces, previous)
            width = np.maximum(0.0, hi - lo)
            col = columns[name]
            unit = nodes[:, col]
            times[:, col] = lo + unit * width
            value = times[:, col]
            # A unit-face sample may legitimately lie at an endpoint or
            # leave a subsequent slice empty. An interior prefix of a
            # positive-volume panel must admit an interior next sample.
            interior &= (unit > 0.0) & (unit < 1.0)
            _require_interior_sample(name, value, lo, hi, interior)
            previous.append(value)
            jacobians *= width
        self._check_reconstructed_clocks(previous, interior & (jacobians > 0.0))
        return times, jacobians

    def check_scipy_sample(self, args):
        """Check QUADPACK's interior nodes in innermost-first order."""
        if self.empty:
            return
        previous = [0.0]
        for name, faces, value in zip(self.names, self.faces, reversed(args)):
            lo, hi = self.limits(faces, previous)
            _require_interior_sample(name, value, lo, hi)
            previous.append(value)
        self._check_reconstructed_clocks(previous)

    def scipy_ranges(self, quad_vars):
        if tuple(reversed(quad_vars)) != self.names:
            raise ValueError(
                "shifted integration ranges require innermost-first coordinates"
            )
        if self.empty:
            return [(0.0, 0.0) for _ in quad_vars]

        def make_range(faces):
            def limits(*outer):
                lo, hi = self.limits(faces, [0.0, *reversed(outer)])
                return float(lo), max(float(lo), float(hi))

            return limits

        return [make_range(faces) for faces in reversed(self.faces)]


class ShiftedDomain:
    """An affine causal polytope and its smooth quadrature panels."""

    def __init__(self, spatial, internal, swept, fixed, final, ceiling, lower):
        self.names = (*swept, *internal)
        if not all(isfinite(v) for v in (final, ceiling, lower, *fixed.values())):
            raise ValueError("fixed delay integration requires finite time bounds")
        coordinates = {name: (i, 0) for i, name in enumerate(self.names, 1)}
        coordinates.update({name: (0, _time_fraction(t)) for name, t in fixed.items()})
        for delayed, root, lag in spatial.time_lags:
            index, offset = coordinates[root]
            coordinates[delayed] = (index, offset - Fraction(lag))
        for alias, root in spatial.equal_time_aliases:
            coordinates[alias] = coordinates[root]
        self.coordinates = coordinates

        size = len(self.names) + 1
        matrix = [[0 if i == j else inf for j in range(size)] for i in range(size)]
        final, ceiling, lower = map(_time_fraction, (final, ceiling, lower))
        for i, name in enumerate(self.names, 1):
            matrix[0][i] = final if name in swept else ceiling
            matrix[i][0] = -lower
        # A vertex with a delayed group has no support until *every*
        # participating history is inside the finite model interval.
        clock_checks = []
        for delayed, _, _ in spatial.time_lags:
            i, offset = coordinates[delayed]
            matrix[i][0] = min(matrix[i][0], offset - lower)
            clock_checks.append(
                (f"delayed clock {delayed!r} above the history floor",
                 (i, float(offset)), (0, float(lower)))
            )

        absorbed = set(spatial.r_absorbed_pairs)
        for late, early in spatial.r_propagators:
            if (late, early) in absorbed:
                continue
            i, late_offset = coordinates[late]
            j, early_offset = coordinates[early]
            c = late_offset - early_offset
            if (i and late_offset) or (j and early_offset):
                clock_checks.append(
                    (f"shifted response {late!r} > {early!r}",
                     (i, float(late_offset)), (j, float(early_offset)))
                )
            if i == j and c <= 0:  # the Itô response vanishes at equality
                matrix[0][0] = -1
            matrix[i][j] = min(matrix[i][j], c)
        self.clock_checks = tuple(clock_checks)
        self.matrix = _closure(matrix)
        if _positive_volume(self.matrix):
            _check_clock_intervals(spatial.time_lags, coordinates, self.matrix)
            _check_clock_projections(self.names, self.matrix)

    def crossing(self, early, late):
        """x_j - x_i <= c means time(early) <= time(late)."""
        j, earlier_offset = self.coordinates[early]
        i, later_offset = self.coordinates[late]
        return i, j, later_offset - earlier_offset

    def bounds(self):
        return ShiftedBounds(self.names, self.matrix, self.clock_checks)

    def panels(self, kink_pairs):
        """Split each interior kink and each switch of an active face.

        Fraction arithmetic is confined to the geometry of the panels. It
        prevents a rounded-off equality from reintroducing the same split.
        Evaluations and nodes use ordinary floating point.
        """
        planes = [self.crossing(a, b) for a, b in kink_pairs]
        stack = [self.matrix]
        while stack:
            matrix = stack.pop()
            if not _positive_volume(matrix):
                continue
            # A split can isolate a sliver whose whole projection rounds
            # to one floating-point time. It has zero numerical volume
            # and did not contribute in the original partition.
            if matrix is not self.matrix and any(
                float(-matrix[i][0]) == float(matrix[0][i])
                for i in range(1, len(matrix))
            ):
                continue
            split = next(
                (
                    (i, j, c)
                    for i, j, c in planes
                    if matrix[i][j] > c and matrix[j][i] > -c
                ),
                None,
            )
            if split is None:
                for k in range(1, len(matrix)):
                    for upper in (False, True):
                        faces = _active_faces(matrix, k, upper)
                        if len(faces) > 1:
                            (i, a), (j, b) = faces[:2]
                            split = i, j, a - b
                            break
                    if split is not None:
                        break
            if split is None:
                yield ShiftedBounds(self.names, matrix, self.clock_checks)
                continue
            i, j, c = split
            for source, target, bound in ((i, j, c), (j, i, -c)):
                child = [row.copy() for row in matrix]
                child[source][target] = bound
                stack.append(_closure(child))
