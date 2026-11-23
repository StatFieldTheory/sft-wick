"""Analytic integration of diagrams with static OU/white propagators."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, localcontext
from functools import lru_cache
from itertools import product
from math import comb, exp, factorial, fsum, inf, isfinite, prod

import numpy as np

from ._exp_sum import poset_integral, shifted_power


@lru_cache(maxsize=1024)
def _integrated_exp(rate, degree, precision):
    """Basis of the integral from 0 to t of s**degree * exp(-rate*s)."""
    if rate == 0:
        return (((rate + 1) / (degree + 1), rate, degree + 1),)
    scale = factorial(degree) / rate ** (degree + 1)
    return ((scale, rate * 0, 0),) + tuple(
        (-scale * rate**k / factorial(k), rate, k) for k in range(degree + 1)
    )


def _decay_difference(first, second):
    """Basis of the convolution of two decays, including the repeated rate."""
    mid = (first + second) / 2
    delta = abs(first - second) / 2
    if delta == 0:
        return ((mid * 0 + 1, mid, 1),)
    return (
        (1 / (2 * delta), min(first, second), 0),
        (-1 / (2 * delta), max(first, second), 0),
    )


@lru_cache(maxsize=1024)
def _covariance_basis(
    early_rate, late_rate, noise_rate, colored, white, span, precision
):
    """Terms c*s**p*d**q*exp(-a*s-b*d), where s is age and d is lag."""
    terms = defaultdict(list)
    if span == inf:
        if colored:
            terms[0, 0, early_rate, 0].append(
                colored / (early_rate * (early_rate + noise_rate))
            )
            for coeff, rate, degree in _decay_difference(early_rate, noise_rate):
                terms[0, 0, rate, degree].append(
                    colored * coeff / (early_rate + noise_rate)
                )
        if white:
            terms[0, 0, late_rate, 0].append(white / (early_rate + late_rate))
    elif colored:
        gamma = early_rate
        total_rate = gamma + noise_rate
        for coeff, rate, degree in _decay_difference(total_rate, 2 * gamma):
            for weight, decay, power in _integrated_exp(rate, degree, precision):
                terms[decay, power, gamma, 0].append(2 * colored * coeff * weight)
        for weight, decay, power in _integrated_exp(total_rate, 0, precision):
            for coeff, rate, degree in _decay_difference(gamma, noise_rate):
                terms[decay, power, rate, degree].append(colored * weight * coeff)
    if white and span != inf:
        for weight, decay, power in _integrated_exp(
            early_rate + late_rate, 0, precision
        ):
            terms[decay, power, late_rate, 0].append(white * weight)
    summation = sum if precision else fsum
    return tuple((summation(weights), *key) for key, weights in terms.items())


def _time_order(predecessors, unknown):
    """Complete a partial order, branching only at covariance edges."""
    if not unknown:
        yield predecessors
        return
    left, right = unknown[0]
    if left in predecessors[right] or right in predecessors[left]:
        yield from _time_order(predecessors, unknown[1:])
        return
    for early, late in ((left, right), (right, left)):
        before = predecessors[early] | {early}
        oriented = [
            prior | before if j == late or late in prior else prior
            for j, prior in enumerate(predecessors)
        ]
        yield from _time_order(oriented, unknown[1:])


def _close_predecessors(count, edges):
    predecessors = [set() for _ in range(count)]
    for early, late in edges:
        predecessors[late].add(early)
    for k in range(count):
        for prior in predecessors:
            if k in prior:
                prior.update(predecessors[k])
    return predecessors


def _time_components(legs, predecessors):
    """Independent branches are integrated separately."""
    pending = set(legs)
    while pending:
        component = {min(pending)}
        while True:
            grown = component | {
                i
                for i in pending
                if any(i in predecessors[j] or j in predecessors[i] for j in component)
            }
            if grown == component:
                break
            component = grown
        pending.difference_update(component)
        ordered = tuple(sorted(component))
        masks = tuple(
            sum(1 << k for k, prior in enumerate(ordered) if prior in predecessors[i])
            for i in ordered
        )
        yield ordered, masks


@lru_cache(maxsize=4096)
def _time_regions(count, n_bounds, strict, c_pairs):
    """Partition at fixed cuts, then orient the C edges inside each panel."""
    fixed = {~j: 2 * j for j in range(n_bounds)}
    candidates = []
    for leg in range(count):
        lo = max((fixed[a] for a, b in strict if a < 0 and b == leg), default=0)
        hi = min(
            (fixed[b] for a, b in strict if b < 0 and a == leg),
            default=2 * n_bounds - 2,
        )
        candidates.append(
            tuple(slot for slot in range(n_bounds - 1) if lo < 2 * slot + 1 < hi)
        )
    regions = []
    for slots in product(*candidates):
        keys = {**fixed, **{leg: 2 * slot + 1 for leg, slot in enumerate(slots)}}
        if any(
            keys[early] > keys[late] or (early < 0 and keys[early] == keys[late])
            for early, late in strict
        ):
            continue
        causal = [(a, b) for a, b in strict if keys[a] == keys[b]]
        predecessors = _close_predecessors(count, causal)
        if any(j in prior for j, prior in enumerate(predecessors)):
            continue
        unoriented = tuple(
            sorted(
                {
                    (min(a, b), max(a, b))
                    for a, b in c_pairs
                    if a >= 0 and b >= 0 and a != b and slots[a] == slots[b]
                }
            )
        )
        for oriented in _time_order(predecessors, unoriented):
            reverse = tuple(
                keys[a] > keys[b] or (a >= 0 and b >= 0 and b in oriented[a])
                for a, b in c_pairs
            )
            groups = []
            for slot in sorted(set(slots)):
                legs = tuple(j for j, s in enumerate(slots) if s == slot)
                groups.extend(
                    (slot, vertices, masks)
                    for vertices, masks in _time_components(legs, oriented)
                )
            regions.append((slots, reverse, tuple(groups)))
    return tuple(regions)


def _pair_terms(size, early, late, origins, basis, stationary):
    """Express one covariance factor in the coordinates of a simplex."""
    if not stationary and early < 0 and origins[early] <= 0:
        return ()
    early_age = origins[early]
    delay = origins[late] - early_age
    zero_powers = (0,) * size
    if early == late:
        basis = ((c, a, p, 0, 0) for c, a, p, _, q in basis if q == 0)
    terms = []
    for coefficient, age_decay, age_power, lag_decay, lag_power in basis:
        if coefficient == 0:
            continue
        rates = [0] * size
        if early >= 0:
            rates[early] += lag_decay - age_decay
        if late >= 0:
            rates[late] -= lag_decay
        offset = -age_decay * early_age - lag_decay * delay
        lag_terms = shifted_power(lag_power, delay)
        age_terms = (
            shifted_power(age_power, early_age)
            if early >= 0
            else ((0, (early_age**age_power if age_power else 1)),)
        )
        for (p, a), (q, b) in product(age_terms, lag_terms):
            for n_early in range(q + 1):
                n_late = q - n_early
                if (early < 0 and n_early) or (late < 0 and n_late):
                    continue
                powers = list(zero_powers)
                if early >= 0:
                    powers[early] += p + n_early
                if late >= 0:
                    powers[late] += n_late
                weight = coefficient * a * b * comb(q, n_early) * (-1) ** n_early
                if weight:
                    terms.append(
                        (
                            weight,
                            tuple(rates),
                            tuple(powers),
                            offset,
                            age_power + lag_power - sum(powers),
                        )
                    )
    return tuple(terms)


@lru_cache(maxsize=16384)
def _decimal_exp(value, precision):
    return Decimal(value).exp()


def _time_factors(bounds, count, strict, r_edges, c_edges, precision):
    """Rate and power factors on each causal panel."""
    span = bounds[-1] - bounds[0]
    stationary = span == inf
    pairs = tuple((left, right) for left, right, *_ in c_edges)
    for slots, reverse, panel_groups in _time_regions(
        count, len(bounds), strict, pairs
    ):
        origins = {i: bounds[slot] for i, slot in enumerate(slots)}
        origins.update({~j: time for j, time in enumerate(bounds)})
        rates = [0] * count
        r_offset = []
        for early, late, gamma in r_edges:
            if early >= 0:
                rates[early] += gamma
            if late >= 0:
                rates[late] -= gamma
            r_offset.append(-gamma * (origins[late] - origins[early]))
        factors = []
        for reversed_pair, edge in zip(reverse, c_edges):
            left, right, left_rate, right_rate, noise_rate, colored, white = edge
            if reversed_pair:
                left, right, left_rate, right_rate = right, left, right_rate, left_rate
            early_rate, late_rate = (
                (right_rate, left_rate) if stationary else (left_rate, right_rate)
            )
            basis = _covariance_basis(
                early_rate, late_rate, noise_rate, colored, white, span, precision
            )
            terms = _pair_terms(count, left, right, origins, basis, stationary)
            if not terms:
                break
            factors.append(terms)
        else:
            groups = [
                (bounds[slot + 1] - bounds[slot], legs, masks)
                for slot, legs, masks in panel_groups
            ]
            yield rates, r_offset, factors, groups


def _expand_time_monomials(rates, r_offset, factors, groups, summation):
    for terms in product(*factors):
        coefficient = prod(term[0] for term in terms)
        offset = summation((*r_offset, *(term[3] for term in terms)))
        power = sum(term[4] for term in terms)
        blocks = tuple(
            (
                width,
                tuple(summation((rates[v], *(t[1][v] for t in terms))) for v in legs),
                tuple(sum(t[2][v] for t in terms) for v in legs),
                masks,
            )
            for width, legs, masks in groups
        )
        yield coefficient, offset, power, blocks


def _fold_exp_monomials(rates, r_offset, factors, groups):
    """Combine matching exponent vectors before multiplying the next factor."""
    terms = {(sum(r_offset), *rates): 1}
    for factor in factors:
        additions = tuple(
            ((offset, *delta), coefficient)
            for coefficient, delta, _, offset, _ in factor
        )
        combined = defaultdict(int)
        for base, weight in terms.items():
            for delta, coefficient in additions:
                key = tuple(a + b for a, b in zip(base, delta))
                combined[key] += weight * coefficient
        terms = combined
    blocks = [(width, legs, (0,) * len(legs), masks) for width, legs, masks in groups]
    for (offset, *rates), weight in terms.items():
        yield (
            weight,
            offset,
            0,
            tuple(
                (width, tuple(rates[v] for v in legs), powers, masks)
                for width, legs, powers, masks in blocks
            ),
        )


@lru_cache(maxsize=128)
def _common_upper_monomials(count, strict, r_edges, c_edges, precision, fold_exp):
    """Compile dimensionless time factors once for equal-time sweeps."""
    zero, one = (Decimal(0), Decimal(1)) if precision else (0.0, 1.0)
    summation = sum if precision else fsum
    terms = defaultdict(list)
    raw_weight = zero
    for rates, offsets, factors, groups in _time_factors(
        (zero, one), count, strict, r_edges, c_edges, precision
    ):
        if fold_exp and all(
            not any(powers) and extra == 0
            for factor in factors
            for _, _, powers, _, extra in factor
        ):
            rows = _fold_exp_monomials(rates, offsets, factors, groups)
            if factors:
                # Keep a bound for the absolute sum before coalescing.
                # Each original exponential has nonpositive physical
                # exponent on its panel, hence the horizon**count bound.
                raw_weight += prod(
                    sum(abs(row[0]) for row in factor) for factor in factors
                )
        else:
            rows = _expand_time_monomials(rates, offsets, factors, groups, summation)
        for coefficient, offset, power, blocks in rows:
            terms[offset, power, blocks].append(coefficient)
    return tuple(
        (summation(weights), *key) for key, weights in terms.items()
    ), raw_weight


def _sum_time_regions(
    bounds, count, strict, r_edges, c_edges, precision=0, *, fold_exp=False
):
    """Integrate one component assignment; cache geometry-free time factors."""
    summation = sum if precision else fsum
    scaled = len(bounds) == 2 and bounds[0] == 0 and bounds[1] != inf
    if scaled:
        scale = bounds[1]
        monomials, raw_weight = _common_upper_monomials(
            count, strict, r_edges, c_edges, precision, fold_exp
        )
        roundoff_scale = raw_weight * scale**count if raw_weight else 0
    else:
        scale = 1
        roundoff_scale = 0
        monomials = (
            term
            for factors in _time_factors(
                bounds, count, strict, r_edges, c_edges, precision
            )
            for term in _expand_time_monomials(*factors, summation)
        )

    total = []
    for coefficient, offset, power, blocks in monomials:
        if not coefficient:
            continue
        offsets = [offset * scale]
        values = [coefficient * scale**power]
        for width, rates, powers, masks in blocks:
            value, shift = poset_integral(
                width * scale, rates, powers, masks, precision
            )
            values.append(value)
            offsets.append(shift)
        log_scale = summation(offsets)
        damping = _decimal_exp(log_scale, precision) if precision else exp(log_scale)
        total.append(prod(values) * damping)
    return summation(total), max(roundoff_scale, summation(map(abs, total)))


@lru_cache(maxsize=4096)
def _temporal_integral(bounds, count, strict, r_edges, c_edges):
    """Retry ill-conditioned finite-start sums with guard digits."""
    # Each finite-start colored leg contributes O(T**2). Several such
    # legs on a short common horizon can cancel many leading digits of
    # the expanded monomials. Start with guard digits in this regime.
    short_colored_window = (
        len(bounds) == 2
        and bounds[0] == 0
        and len(c_edges) >= 3
        and all(bounds[1] * rate <= 4 for _, _, rate in r_edges)
        and all(colored and not white for _, _, _, _, _, colored, white in c_edges)
        and bounds[1] * max(max(a, b, noise) for _, _, a, b, noise, _, _ in c_edges)
        <= 4
    )
    if not short_colored_window:
        try:
            value, magnitude = _sum_time_regions(
                bounds, count, strict, r_edges, c_edges
            )
        except (OverflowError, ZeroDivisionError):
            pass
        else:
            if isfinite(magnitude) and (
                magnitude == 0 or abs(value) * 256 >= magnitude
            ):
                return value

    retry_precisions = (48, 96, 192, 384, 768, 1536, 3072)
    if short_colored_window:
        retry_precisions = (36, *retry_precisions)
    for precision in retry_precisions:
        with localcontext() as context:
            context.prec = precision
            result, magnitude = _sum_time_regions(
                tuple(map(Decimal, bounds)),
                count,
                strict,
                tuple((a, b, Decimal(float(g))) for a, b, g in r_edges),
                tuple(
                    (a, b, *(Decimal(float(v)) for v in params))
                    for a, b, *params in c_edges
                ),
                precision,
                fold_exp=short_colored_window,
            )
            if not magnitude or abs(result) * 10 ** (precision - 20) >= magnitude:
                return float(result)
    raise ArithmeticError("method='exact' could not resolve covariance cancellation")


@lru_cache(maxsize=1024)
def _stationary_covariance(gamma, factor, noise_rate, sigma2):
    """Fixed-time C at stationarity without intermediate overflow/underflow."""
    with localcontext() as context:
        context.prec = 48
        rates = tuple(Decimal(float(rate)) for rate in gamma)
        color = Decimal(float(factor))
        width = Decimal(float(noise_rate))
        values = np.diag([float(color / (rate * (rate + width))) for rate in rates])
        if sigma2 is not None:
            for a, left in enumerate(rates):
                for b, right in enumerate(rates):
                    values[a, b] += float(Decimal(float(sigma2[a][b])) / (left + right))
        return values


def _component_pairs(left, right, count):
    if left is None or right is None:
        return tuple((i, i) for i in range(count))
    return ((left, right),)


def require_exact_covariance(covariance, direct, *, initial_covariance=None):
    """Admission shared by the exact path and workflow validation."""
    from .workflow.closed_forms import ClosedFormC

    if initial_covariance is not None:
        raise ValueError("method='exact' does not support an initial covariance")
    if (
        type(covariance) is not ClosedFormC
        or covariance.temporal_kind != "exponential"
        or not direct
        or not isfinite(1 / covariance.sigma_t)
    ):
        raise ValueError(
            "method='exact' requires direct built-in OU/white covariance and static diagonal response"
        )


def integrate_exact(
    integrand, final, cache, t_min, direction, positions, integrate_over, external_times
):
    """Execute exact OU integration through the same diagram admission as GL."""
    from .evaluate import (
        PropagatorCache,
        _finite_or_raise,
        _isotropic_C,
        _real_or_raise,
    )
    from .workflow.closed_forms import _separation
    from .workflow.specs import _StaticIsoR, _StaticMatR

    covariance = getattr(cache, "c_value_fn", None)
    r_time = getattr(getattr(cache, "model", None), "R_time", None)
    source = getattr(cache, "_source", cache)
    require_exact_covariance(
        covariance, getattr(cache, "_closed_form_only", False),
        initial_covariance=getattr(cache.model, "c_initial", None),
    )
    if (
        type(source) is not PropagatorCache
        or type(r_time) not in (_StaticIsoR, _StaticMatR)
        or any(
            name in vars(source)
            for name in ("C_at_batch", "C_value", "R_matrix_batch", "R_time_batch")
        )
    ):
        raise ValueError(
            "method='exact' requires a native cache with static diagonal response"
        )
    gamma = covariance.gamma
    r_rates = (
        (r_time.gamma,) * len(gamma)
        if type(r_time) is _StaticIsoR
        else tuple(r_time.gamma_arr)
    )
    if r_rates != gamma:
        raise ValueError(
            "method='exact' requires matching covariance and response rates"
        )
    if integrand.dynamic_coupling is not None:
        raise ValueError("method='exact' requires static couplings")
    lower = integrand._admit(cache, t_min, positions, direction)
    swept, fixed, ceiling = integrand._split_externals(
        integrate_over, final, external_times
    )
    stationary = final == inf
    if stationary and (
        min(gamma) <= 0 or swept or any(t != inf for t in fixed.values())
    ):
        raise ValueError(
            "method='exact' at infinity requires positive decay rates and all externals fixed at infinity"
        )
    if not stationary and not all(
        isfinite(t) for t in (lower, final, ceiling, *fixed.values())
    ):
        raise ValueError(
            "method='exact' requires finite time bounds for a finite final time"
        )
    if lower != covariance.t_min:
        raise ValueError(
            "method='exact' requires matching covariance and integration origins"
        )

    spatial = integrand.spatial
    if spatial.time_lags:
        raise ValueError("method='exact' does not integrate fixed time lags; use gauss_legendre")
    free = (*spatial.time_integration_vars, *swept)
    count = len(free)
    ends = {
        name: 0.0 if stationary else time - covariance.t_min
        for name, time in fixed.items()
    }
    start = lower - covariance.t_min
    stop = inf if stationary else ceiling - covariance.t_min
    sweep_end = final - covariance.t_min
    bounds = tuple(
        sorted({start, stop, *ends.values(), *([sweep_end] if swept else [])})
    )
    times = {name: i for i, name in enumerate(free)}
    times.update({name: ~bounds.index(end) for name, end in ends.items()})
    aliases = dict(spatial.equal_time_aliases)

    def at(name):
        return times[aliases.get(name, name)]

    strict_edges = [(at(early), at(late)) for early, late in spatial.time_orderings]
    if stationary:
        strict_edges = [(later, earlier) for earlier, later in strict_edges]
    strict_edges.extend((~bounds.index(start), i) for i in range(count))
    strict_edges.extend((i, ~bounds.index(stop)) for i in range(count))
    strict_edges.extend((times[name], ~bounds.index(sweep_end)) for name in swept)
    if any(early == late for early, late in strict_edges):
        return 0.0, 0.0
    strict = tuple(sorted(set(strict_edges)))

    group_x = integrand._resolve_group_x(spatial, positions, direction)
    c_positions = integrand._c_position_pairs(group_x)
    c_geometry = []
    for (left, right, il, ir), (xl, xr) in zip(spatial.c_propagators, c_positions):
        separation = _separation(np.atleast_1d([xl]), np.atleast_1d([xr]))[0]
        factor = covariance.lam * covariance.spatial_factor(float(separation))
        if il is None and ir is None:
            # Validate the index-free promise, including unequal-rate and
            # off-diagonal white-noise cases.
            sample = covariance(xl, covariance.t_min + 1, xr, covariance.t_min + 1)
            _isotropic_C(np.diag(sample) if cache.model.diag_C else sample)
            if len(set(gamma)) != 1:
                raise ValueError(
                    "index-free exact covariance requires equal component rates"
                )
        c_geometry.append((at(left), at(right), il, ir, factor, xl, xr))
    r_objects = integrand._kept_r_propagator_objects()
    names = [name for name, _ in integrand.diagram_term.propagator_indices]
    resolve = integrand._resolve_component
    values = []
    for idx in np.ndindex(integrand.coupling_array.shape):
        coefficient = integrand.coupling_array[idx]
        if not coefficient:
            continue
        coefficient = _real_or_raise(coefficient, integrand._e_psi, where=" (coupling)")
        labels = {**integrand.fixed_indices, **dict(zip(names, idx))}
        r_edges = []
        for r in r_objects:
            left = resolve(r.index_left, labels)
            right = resolve(r.index_right, labels)
            if not cache.model.iso_R and left != right:
                break
            rate = gamma[0 if cache.model.iso_R else left]
            early, late = at(r.spatial_right), at(r.spatial_left)
            r_edges.append((late, early, rate) if stationary else (early, late, rate))
        else:
            choices = []
            for left, right, il, ir, factor, xl, xr in c_geometry:
                i, j = resolve(il, labels), resolve(ir, labels)
                entries = _component_pairs(i, j, len(gamma))
                entries = tuple(
                    (a, b) for a, b in entries if a == b or not cache.model.diag_C
                )
                isotropic = il is None and ir is None
                if left < 0 and right < 0:
                    if stationary:
                        matrix = _stationary_covariance(
                            gamma, factor, 1 / covariance.sigma_t, covariance.sigma2
                        )
                    else:
                        matrix = covariance(
                            xl,
                            bounds[~left] + covariance.t_min,
                            xr,
                            bounds[~right] + covariance.t_min,
                        )
                    if isotropic:
                        coefficient *= _isotropic_C(
                            np.diag(matrix) if cache.model.diag_C else matrix
                        )
                    else:
                        coefficient *= fsum(matrix[a, b] for a, b in entries)
                    continue
                if isotropic:
                    white = (
                        fsum(covariance.sigma2[a][a] for a in range(len(gamma)))
                        / len(gamma)
                        if covariance.sigma2 is not None
                        else 0.0
                    )
                    if white:
                        choices.append(
                            [
                                (
                                    1.0,
                                    (
                                        left,
                                        right,
                                        gamma[0],
                                        gamma[0],
                                        1 / covariance.sigma_t,
                                        factor,
                                        white,
                                    ),
                                )
                            ]
                        )
                    elif factor:
                        choices.append(
                            [
                                (
                                    factor,
                                    (
                                        left,
                                        right,
                                        gamma[0],
                                        gamma[0],
                                        1 / covariance.sigma_t,
                                        1.0,
                                        0.0,
                                    ),
                                )
                            ]
                        )
                    else:
                        choices.append([])
                    continue
                pair_choices = []
                for a, b in entries:
                    colored = factor if a == b else 0.0
                    white = (
                        covariance.sigma2[a][b]
                        if covariance.sigma2 is not None
                        else 0.0
                    )
                    if colored and not white:
                        # Spatial/amplitude factors stay outside the time
                        # cache and are reused across position sweeps.
                        pair_choices.append(
                            (
                                colored,
                                (
                                    left,
                                    right,
                                    gamma[a],
                                    gamma[b],
                                    1 / covariance.sigma_t,
                                    1.0,
                                    0.0,
                                ),
                            )
                        )
                    elif colored or white:
                        pair_choices.append(
                            (
                                1.0,
                                (
                                    left,
                                    right,
                                    gamma[a],
                                    gamma[b],
                                    1 / covariance.sigma_t,
                                    colored,
                                    white,
                                ),
                            )
                        )
                choices.append(pair_choices)
            if not coefficient:
                continue
            for selected in product(*choices):
                weight = prod(c[0] for c in selected)
                c_edges = tuple(c[1] for c in selected)
                value = _temporal_integral(
                    bounds, count, strict, tuple(r_edges), c_edges
                )
                values.append(coefficient * weight * value)
    return _finite_or_raise(fsum(values), " (exact)"), 0.0
