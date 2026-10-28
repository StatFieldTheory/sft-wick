"""Exact references for the opt-in contraction backends.

The static integrand is ``sum_a c(a) * prod_k K_k(s, a)``.  These helpers
enumerate the component assignments with the ordered fold's semantics in
``Fraction`` arithmetic, sharing no code with the backends.  Index-free C
factors and R traces are float input transforms that the ordered fold and
the backends compute with the same NumPy call; the reference applies the
same transform to obtain its inputs.

``bound`` is the accuracy contract of :mod:`sft_wick._dense_contraction`,
``1.01 (m + q + A T) u S``.  Its scale ``S`` is always built from the summed
coefficient ``c(a)`` that the ordered fold multiplies, never from the
magnitudes of separate monomials.  ``m`` counts the factors of one product
and ``q`` the rationals among them (the prefactor included) that have no
exact float64 value: each is converted to float64 before it is multiplied,
one rounding more than the products.  Against :func:`dense_coefficients`
the coefficient is a float64 factor, so ``q = 0``; against
:func:`monomial_coefficients` the rationals are exact fractions.
"""

from fractions import Fraction

import numpy as np

from sft_wick.expressions import ImaginaryUnit, Product, Rational, Sum, Symbol

U = Fraction(1, 2**53)


def dense_coefficients(coeff, e_psi):
    """``{assignment: (c, |c|)}`` of the rotated real part, zero entries skipped."""
    table = {}
    for index in np.ndindex(*coeff.shape):
        value = coeff[index]
        if value == 0:
            continue
        rotated = Fraction((complex(value) * (1j**e_psi)).real)
        table[index] = (rotated, abs(rotated))
    return table


def _complex(value):
    z = complex(value)
    return Fraction(z.real), Fraction(z.imag)


def _times(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def monomial_coefficients(term, values, fixed=None):
    """``{assignment: (c, |c|)}`` with ``c = sum_t c_t`` from the original factors.

    Each monomial is multiplied out exactly, including the response phase
    ``(-i)**n_response``, the rational prefactor and the observable rotation
    ``i**E_psi``; every rotated monomial must be real.  The scale is ``|c|``,
    the magnitude of the coefficient the ordered fold uses, so a backend
    whose error grows with ``sum_t |c_t|`` fails the bound when monomials
    cancel.
    """
    names = [name for name, _ in term.propagator_indices]
    shape = tuple(dim for _, dim in term.propagator_indices)
    prefactor = _complex([1, -1j, -1, 1j][term.n_response % 4])
    prefactor = _times(
        prefactor,
        (
            Fraction(
                term.rational_prefactor.numerator, term.rational_prefactor.denominator
            ),
            Fraction(0),
        ),
    )
    prefactor = _times(prefactor, _complex(1j**term.n_external_response))
    expression = term.coupling_sum
    monomials = expression.terms if type(expression) is Sum else (expression,)
    table = {}
    for index in np.ndindex(*shape):
        labels = {**(fixed or {}), **dict(zip(names, index))}
        total = magnitude = Fraction(0)
        for monomial in monomials:
            value = prefactor
            stack = [monomial]
            while stack:
                node = stack.pop()
                if type(node) is Product:
                    stack.extend(node.factors)
                elif type(node) is Symbol:
                    array = values[node.name]
                    value = _times(
                        value, _complex(array[tuple(labels[i] for i in node.indices)])
                    )
                elif type(node) is Rational:
                    value = _times(
                        value, (Fraction(node.numerator, node.denominator), Fraction(0))
                    )
                elif type(node) is ImaginaryUnit:
                    value = _times(value, (Fraction(0), Fraction(1)))
                else:
                    raise TypeError(type(node))
            assert value[1] == 0, "every rotated monomial must be real"
            total += value[0]
            magnitude += abs(value[0])
        if magnitude:
            table[index] = (total, abs(total))
    return table


def kernel_product(sample, labels, c_batches, r_product, jacobians, r_batches):
    """Exact product of every propagator factor at one sample and assignment."""
    from sft_wick.evaluate import _isotropic_C

    value = Fraction(float(r_product[sample])) * Fraction(float(jacobians[sample]))
    for array, left, right, support in r_batches:
        if left and right:
            a, b = labels[left], labels[right]
            if not support[a, b]:
                return Fraction(0)
            value *= Fraction(float(array[sample, a, b]))
        else:
            value *= Fraction(float(np.einsum("iaa->i", array)[sample]))
    for array, left, right in c_batches:
        if left is None and right is None:
            value *= Fraction(float(_isotropic_C(array, batched=True)[sample]))
        elif array.ndim == 3:
            value *= Fraction(float(array[sample, labels[left], labels[right]]))
        else:
            a, b = labels[left], labels[right]
            value *= Fraction(float(array[sample, a])) if a == b else Fraction(0)
    return value


def exact_sum(table, prop_idx, fixed, c_batches, r_product, jacobians, r_batches):
    """Per sample: the exact sum and ``S = sum |terms|``."""
    names = [name for name, _ in prop_idx]
    exact, scale = [], []
    for sample in range(r_product.size):
        total = magnitude = Fraction(0)
        for index, (value, size) in table.items():
            labels = {**fixed, **dict(zip(names, index))}
            kernel = kernel_product(
                sample, labels, c_batches, r_product, jacobians, r_batches
            )
            total += value * kernel
            magnitude += size * abs(kernel)
        exact.append(total)
        scale.append(magnitude)
    return exact, scale


def bound(scale, factors, assignments, terms=1, *, inexact):
    """The documented error bound ``1.01 (m + q + A T) u S`` for one sample.

    ``factors`` is ``m``, ``inexact`` is ``q``, ``assignments`` is ``A`` and
    ``terms`` is ``T``.
    """
    return Fraction(101, 100) * (factors + inexact + assignments * terms) * U * scale


def errors_in_units(values, exact, scale):
    """``max |value - exact| / (u * S)`` over the samples (0 where S = 0)."""
    worst = Fraction(0)
    for value, reference, size in zip(values, exact, scale):
        error = abs(Fraction(float(value)) - reference)
        if size:
            worst = max(worst, error / (U * size))
        else:
            assert error == 0
    return float(worst)


def assert_within_bound(values, exact, scale, factors, assignments, terms=1, *, inexact):
    assert len(values) == len(exact)
    for value, reference, size in zip(values, exact, scale):
        error = abs(Fraction(float(value)) - reference)
        limit = bound(size, factors, assignments, terms, inexact=inexact)
        assert error <= limit, (float(value), float(reference), float(error), float(limit))


def assert_relative(values, exact, tolerance):
    """``|value - exact| <= tolerance * |exact|`` at every sample."""
    for value, reference in zip(values, exact):
        error = abs(Fraction(float(value)) - reference)
        assert error <= Fraction(tolerance) * abs(reference), (
            float(value),
            float(reference),
        )
