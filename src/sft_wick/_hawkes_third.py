"""Range-safe contractions of the two third-order marked Hawkes diagrams."""

from __future__ import annotations

from decimal import Decimal, localcontext
from itertools import pairwise, product
from math import ceil, log, log2, sqrt

import numpy as np


def needs_wide_third(model, bounds):
    """Check the range of factors before diagrams multiply them by a large rate."""
    times = sorted({0.0}.union(*bounds))
    horizon = times[-1]
    depth = len(model.baseline) + 2
    mean = log2(model.mark_moments[0])
    clock = log2(horizon)
    edges = [log2(x) for row in model.excitation for x in row if x]
    source = [log2(x) for x in model.baseline if x]
    marks = [log2(x) for x in model.mark_moments[:3]]
    # A path from a source to a counter can include every node. A third
    # moment multiplies three such transfers. Use logarithms so the guard
    # itself does not lose weak edges or depend on an intermediate product.
    shortest = min(log2(b - a) for a, b in pairwise(times))
    event_factors = [mean + edge + t for edge in edges for t in (shortest, clock)]
    jumps = max(
        (abs(m + k * edge) for k, m in enumerate(marks, start=1) for edge in edges),
        default=0.0,
    )
    path = 3 * depth * max(map(abs, event_factors), default=0.0)
    # The ordinary evaluator also builds physical, rate-weighted tensors
    # before multiplying by the response legs and integrating over time.
    immigration = max(
        (max(abs(rate), abs(clock + rate)) for rate in source), default=0.0
    )
    raw_clock = max(abs(clock), abs(shortest))
    budget = jumps + path + immigration + 4 * raw_clock + 3 * log2(depth)
    if budget > 900:
        return True
    fastest = max(
        clock + log2(max(model.decay)),
        clock + mean + max(edges, default=0.0) + log2(len(model.baseline)),
    )
    if fastest < 8 and budget + 3 * 2.0**fastest / log(2.0) <= 900:
        return False

    # For rows with sum(E[M]*A_ij/beta_i) <= rho < 1 the excitation flow
    # is a positive contraction in the maximum norm. The constant vector
    # max(mu)/(1-rho) bounds every mean rate. Long common history therefore
    # cannot exponentially amplify an omitted transient. Count responses
    # and covariances of cumulative counts grow at most polynomially in T.
    # Keep the exponential cost for any unavoidable gap between the three
    # observation windows, where decay can be the entire physical signal.
    rho = 0.0
    costs = np.full((len(model.baseline), len(model.baseline)), np.inf)
    for i, (row, beta) in enumerate(zip(model.excitation, model.decay)):
        sources = np.flatnonzero(row)
        row_logs = [mean + log2(row[j]) - log2(beta) for j in sources]
        if any(value >= 0 for value in row_logs):
            return True
        rho = max(rho, sum(2.0**value for value in row_logs))
        # After transients decay a transfer can be as small as E[M]*A/beta.
        # Keep weak multi-generation responses until immigration is applied.
        costs[i, sources] = [-value for value in row_logs]

    if rho >= 1 - 1e-12:
        return True
    # Positive transfers add. For each reachable pair the strongest
    # route gives a lower bound on its integrated response. Negative
    # logarithms turn products of per-generation masses into path lengths.
    # A dense network can reach every node in one generation.
    largest_edge_cost = np.max(costs, where=np.isfinite(costs), initial=0.0)
    np.fill_diagonal(costs, 0.0)
    for j in range(len(costs)):
        costs = np.minimum(costs, costs[:, j, None] + costs[None, j, :])
    route_cost = np.max(costs, where=np.isfinite(costs), initial=0.0)
    # Include the excitation and count readout legs of each response.
    transfer = 3 * (route_cost + 2 * largest_edge_cost)
    budget += max(0.0, transfer - path) - log2(1 - rho)
    gap = max(lo for lo, _ in bounds) - min(hi for _, hi in bounds)
    if gap > 0:
        log_decay = log2(gap) + log2(max(model.decay))
        if log_decay >= 8:
            return True
        budget += 3 * 2.0**log_decay / log(2.0)
    return budget > 900


class _Flow:
    """Exponential action on a positive state, retaining every source separately."""

    def __init__(self, generator, admit):
        self.gen = generator
        self.admit = admit
        self.shift = max(0, -min(generator.diagonal()))
        kernel = generator.copy()
        kernel.flat[:: len(kernel) + 1] += self.shift
        self.rows, self.cols = np.nonzero(kernel)
        self.weights = kernel[self.rows, self.cols]

    def __call__(self, lag, state):
        if not lag or not np.any(state):
            return state
        # A feed-forward jump may be enormous and still lead to a short
        # exponential series. Judge convergence on the physical state.
        if self.shift * lag <= 16:
            factor = self.weights * lag
            term = total = state
            if state.ndim == 2:
                factor = factor[:, None]
            for k in range(1, max(96, len(state) + 40)):
                next_term = np.zeros_like(term)
                np.add.at(next_term, self.rows, factor * term[self.cols])
                term = next_term / Decimal(k)
                updated = total + term
                if not np.any(term) or (
                    k >= len(self.gen) and np.array_equal(updated, total)
                ):
                    return (-self.shift * lag).exp() * updated
                total = updated
        from .hawkes import _wide_action

        return _wide_action(
            lambda: self.gen * lag,
            state,
            1.0,
            lambda: max(abs(self.gen).flat) * lag,
            self.admit,
            "third-cumulant",
        )


def _at_node(rate, covariance, future, readout, amplitude, m2, m3):
    n = len(rate)
    # h_i = W_i + M * V_i is the compensated event's contribution to window
    # i, with W its immediate count and V the downstream response. M has
    # unit mean in the chosen mark units.
    v = future @ amplitude
    w = readout
    triple = w[0] * w[1] * w[2] + m3 * v[0] * v[1] * v[2]
    past_and_future = covariance[:n, n:] + covariance[:n, :n] @ future.T
    result = Decimal(0)
    for k, i, j in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        triple += w[i] * w[j] * v[k] + m2 * v[i] * v[j] * w[k]
        pair = w[i] * w[j] + w[i] * v[j] + w[j] * v[i] + m2 * v[i] * v[j]
        # Pair of simultaneous jumps times the covariance of their
        # predictable source rate with the remaining window.
        result += pair @ past_and_future[:, k]
    return result + rate @ triple


def _gated_third(model, nodes, bounds, n_gauss):
    """Integrate single-jump and pair/rate terms between changes of readout."""
    from .hawkes import _as_decimal, _covariance_generator, _drift

    n = len(model.baseline)
    horizon = Decimal.from_float(max(b for _, b in bounds))
    times = sorted({0.0}.union(*bounds))
    # Scale to a unit horizon and unit mean mark before forming any
    # propagator. The variance and all three jump factors are multiplied
    # before returning to the user's binary64 output type.
    mean = Decimal.from_float(model.mark_moments[0])
    a = _as_decimal(np.asarray(model.excitation)) * (mean * horizon)
    beta = _as_decimal(np.asarray(model.decay)) * horizon
    mu = _as_decimal(np.asarray(model.baseline)) * horizon
    m2 = Decimal.from_float(model.mark_moments[1]) / mean**2
    m3 = Decimal.from_float(model.mark_moments[2]) / mean**3
    if n_gauss is None:
        # Fast feed-forward edges affect amplitudes, while time scales
        # depend on eigenvalues of the drift blocks in directed cycles.
        from scipy.sparse.csgraph import connected_components

        _, labels = connected_components(
            np.asarray(model.excitation) > 0, directed=True, connection="strong"
        )
        bound = max(
            abs(a[i, i] - beta[i])
            + sum(a[i, j] for j in range(n) if j != i and labels[j] == labels[i])
            for i in range(n)
        )
        if bound > (384 // 2) ** 2:
            raise ValueError(
                "the time range needs a finer count quadrature; set n_gauss explicitly"
            )
        count = max(12, ceil(sqrt(float(bound))))
    dimension = n + 3
    size = 2 * n + dimension**2
    storage = 8192 + 160 * (4 + len(times)) * size**2

    def admit(extra, stage):
        model._admit(storage + extra, stage)

    def admit_rule(order):
        # NumPy's Legendre rule uses the dense order-by-order companion matrix.
        admit(64 * order**2, "third-cumulant matrix-exponential state and quadrature")

    admit_rule(count if n_gauss is None else int(n_gauss))
    panels = []
    state = np.r_[mu, mu, np.zeros(dimension**2, dtype=int)].astype(object)
    for left, right in pairwise(times):
        width = (Decimal.from_float(right) - Decimal.from_float(left)) / horizon
        readout = np.zeros((3, n), dtype=int)
        for i, (node, (lo, hi)) in enumerate(zip(nodes, bounds)):
            readout[i, node] = int(lo <= left < hi)
        drift = _drift(a, beta, Decimal(1), readout)
        jump = np.concatenate((a.T, readout.T), axis=1)
        jump2 = jump[:, :, None] * jump[:, None, :]
        jump2[:, :n, :n] *= m2
        forward = _Flow(_covariance_generator(drift, beta, jump2), admit)
        backward = _Flow(drift.T, admit)
        panels.append((width, readout, state, forward, backward))
        if right < times[-1]:
            state = forward(width, state)
    end = np.vstack((np.zeros((n, 3), dtype=int), np.eye(3, dtype=int))).astype(object)
    prepared = []
    for width, readout, start, forward, backward in reversed(panels):
        prepared.append((width, readout, start, forward, backward, end))
        end = backward(width, end)

    def integrate(count):
        admit_rule(count)
        rule, weights = np.polynomial.legendre.leggauss(count)
        total = Decimal(0)
        for width, readout, start, forward, backward, end in prepared:
            subtotal = Decimal(0)
            for x, weight in zip(rule, weights):
                lag = width * ((Decimal.from_float(x) + 1) / 2)
                at = forward(lag, start)
                covariance = at[2 * n :].reshape(dimension, dimension)
                future = backward(width - lag, end)[:n].T
                value = _at_node(at[n : 2 * n], covariance, future, readout, a, m2, m3)
                subtotal += Decimal.from_float(weight) * value
            total += (width / 2) * subtotal
        return total

    if n_gauss is not None:
        value = integrate(int(n_gauss))
    else:
        previous = integrate(count)
        while count < 384:
            count = min(384, 2 * count)
            value = integrate(count)
            if abs(value - previous) <= Decimal("5e-11") * abs(value):
                break
            previous = value
        else:
            raise ArithmeticError(
                "count quadrature did not converge; set n_gauss explicitly"
            )
    return value


def window_third(model, nodes, bounds, n_gauss):
    """Evaluate the third count cumulant without rounding the diagram factors."""
    from .hawkes import _MEAN_CONTEXT

    with localcontext(_MEAN_CONTEXT) as ctx:
        # Two binary64 factors can cancel against a third to about 32 decimal
        # places in E[M]*A - beta. Keep the residual after time normalization.
        ctx.prec = 80
        if n_gauss is not None and any(lo for lo, _ in bounds):
            # Keep the exact fixed-rule convention of the Wick evaluator:
            # each cumulative endpoint combination supplies its own GL panels.
            terms = []
            for bits in product((0, 1), repeat=3):
                stops = [window[bit] for window, bit in zip(bounds, bits)]
                if all(stops):
                    gates = [(0.0, stop) for stop in stops]
                    term = _gated_third(model, nodes, gates, n_gauss)
                    terms.append(term if sum(bits) % 2 else -term)
            value = sum(terms, Decimal(0))
            # The scaled exponential action can retain at least 40 decimal
            # digits. Use the direct gated readout when endpoint subtraction
            # would exhaust that precision.
            if sum(map(abs, terms)) * Decimal("1e-28") > abs(value):
                value = _gated_third(model, nodes, bounds, n_gauss)
        else:
            value = _gated_third(model, nodes, bounds, n_gauss)
        result = float(value)
        if not np.isfinite(result):
            raise ArithmeticError("Hawkes count cumulant exceeds floating-point range")
        return result
