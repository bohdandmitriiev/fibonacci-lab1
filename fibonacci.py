"""Fibonacci numbers: several implementations with different complexity.

The Fibonacci sequence is defined as:

    F(0) = 0
    F(1) = 1
    F(n) = F(n - 1) + F(n - 2)    for n >= 2

so it starts with 0, 1, 1, 2, 3, 5, 8, 13, 21, ...
"""

from typing import Any


def _validate_index(value: Any, name: str = "n") -> int:
    """Check that ``value`` is a non-negative integer and return it.

    ``bool`` is a subclass of ``int`` in Python, but ``True`` or ``False``
    is clearly not a meaningful index, so booleans are rejected explicitly.

    Raises:
        ValueError: if ``value`` is not an ``int`` or is negative.
    """
    # Reject booleans and any non-integer type (float, str, None, ...).
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    # Negative indices are not defined for the classic sequence.
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def fib_iterative(n: int) -> int:
    """Return F(n) using a simple loop.

    Time complexity: O(n) additions. Memory: O(1) extra numbers.
    """
    _validate_index(n)

    # Keep only the two latest values: F(i) and F(i + 1).
    current, following = 0, 1
    for _ in range(n):
        # Shift the window one step forward along the sequence.
        current, following = following, current + following
    return current


def fib_recursive(n: int) -> int:
    """Return F(n) with the naive recursive definition.

    This version is intentionally slow: it recomputes the same values over
    and over, so the number of calls grows exponentially (about 1.6 ** n).
    It exists only to compare with the faster implementations.

    Time complexity: O(2 ** n) (more precisely O(phi ** n)). Memory: O(n) stack.
    """
    _validate_index(n)
    return _fib_recursive_unchecked(n)


def _fib_recursive_unchecked(n: int) -> int:
    """Naive recursion without re-validating ``n`` on every call."""
    # Base cases: F(0) = 0 and F(1) = 1.
    if n < 2:
        return n
    # Recursive case: the definition applied directly.
    return _fib_recursive_unchecked(n - 1) + _fib_recursive_unchecked(n - 2)
