"""Fibonacci numbers: several implementations with different complexity.

The Fibonacci sequence is defined as:

    F(0) = 0
    F(1) = 1
    F(n) = F(n - 1) + F(n - 2)    for n >= 2

so it starts with 0, 1, 1, 2, 3, 5, 8, 13, 21, ...

The module can be imported (see the ``fib_*`` functions) or run as a program:

    python fibonacci.py 10                          # first 10 numbers
    python fibonacci.py --nth 50                    # F(50)
    python fibonacci.py --nth 90 --method doubling  # choose the algorithm
"""

import argparse
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


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


# Memoized recursion is warmed up in steps of this size so that the recursion
# depth never exceeds it (Python's default recursion limit is about 1000).
_MEMO_WARMUP_STEP = 200


def fib_memo(n: int) -> int:
    """Return F(n) using recursion with memoization (top-down dynamic programming).

    Every F(k) is computed once and stored in a dictionary, so repeated
    sub-problems are answered from the cache instead of being recomputed.
    The cache is local to one call, which keeps measurements independent.

    Time complexity: O(n) additions. Memory: O(n) for the cache.
    """
    _validate_index(n)

    # Cache of already known values; the base cases are stored up front.
    cache: Dict[int, int] = {0: 0, 1: 1}

    def compute(k: int) -> int:
        # Return the cached value if this sub-problem was already solved.
        if k in cache:
            return cache[k]
        # Otherwise solve it recursively and remember the answer.
        cache[k] = compute(k - 1) + compute(k - 2)
        return cache[k]

    # Fill the cache bottom-up in steps. Each call below recurses at most
    # _MEMO_WARMUP_STEP levels deep, so large n do not hit RecursionError.
    for checkpoint in range(_MEMO_WARMUP_STEP, n, _MEMO_WARMUP_STEP):
        compute(checkpoint)
    return compute(n)


def _fib_pair(n: int) -> Tuple[int, int]:
    """Return the pair (F(n), F(n + 1)) using the fast doubling identities."""
    # Base case: (F(0), F(1)).
    if n == 0:
        return 0, 1
    # Solve the problem for half of n (integer division).
    half_value, half_next = _fib_pair(n // 2)
    # Doubling identities, with k = n // 2:
    #   F(2k)     = F(k) * (2 * F(k + 1) - F(k))
    #   F(2k + 1) = F(k) ** 2 + F(k + 1) ** 2
    even = half_value * (2 * half_next - half_value)
    odd = half_value * half_value + half_next * half_next
    if n % 2 == 0:
        # n = 2k  ->  (F(2k), F(2k + 1))
        return even, odd
    # n = 2k + 1  ->  (F(2k + 1), F(2k + 2)) where F(2k + 2) = F(2k) + F(2k + 1)
    return odd, even + odd


def fib_fast_doubling(n: int) -> int:
    """Return F(n) with the fast doubling method.

    The problem size is halved on every step, so only about log2(n) steps
    are needed instead of n.

    Time complexity: O(log n) arithmetic operations (on large integers).
    Memory: O(log n) for the recursion stack.
    """
    _validate_index(n)
    return _fib_pair(n)[0]


def fib_sequence(count: int) -> Iterator[int]:
    """Yield the first ``count`` Fibonacci numbers: F(0), F(1), ..., F(count - 1).

    This is a generator, so numbers are produced lazily one by one and the
    whole list is never stored in memory.

    Raises:
        ValueError: if ``count`` is not a non-negative integer. Because this
            is a generator, the error is raised on the first ``next()`` call.
    """
    _validate_index(count, "count")

    current, following = 0, 1
    for _ in range(count):
        # Hand out the current number, then move to the next pair.
        yield current
        current, following = following, current + following


# Maps the value of the --method option to the function that implements it.
METHODS: Dict[str, Callable[[int], int]] = {
    "iterative": fib_iterative,
    "recursive": fib_recursive,
    "memo": fib_memo,
    "doubling": fib_fast_doubling,
}

# Naive recursion becomes very slow above this value; the CLI warns about it.
_RECURSIVE_WARNING_THRESHOLD = 35


def _non_negative_int(text: str) -> int:
    """argparse "type" function: convert text to a non-negative integer.

    Raises:
        argparse.ArgumentTypeError: with a friendly message for bad input.
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{text}' is not an integer, please enter a whole number >= 0"
        ) from None
    if value < 0:
        raise argparse.ArgumentTypeError(
            f"{value} is negative, please enter a whole number >= 0"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="fibonacci.py",
        description="Compute Fibonacci numbers with different algorithms.",
    )
    parser.add_argument(
        "count",
        nargs="?",
        type=_non_negative_int,
        help="print the first COUNT Fibonacci numbers (starting from F(0))",
    )
    parser.add_argument(
        "--nth",
        type=_non_negative_int,
        metavar="N",
        help="print only the N-th Fibonacci number F(N)",
    )
    parser.add_argument(
        "--method",
        choices=list(METHODS),
        default="iterative",
        help="algorithm to use (default: %(default)s)",
    )
    return parser


def _ask_count(parser: argparse.ArgumentParser) -> int:
    """Interactive mode: ask the user how many numbers to print."""
    try:
        text = input("How many Fibonacci numbers do you want to see? ")
    except EOFError:
        # No input is available (for example, closed stdin).
        parser.error("no input was given")
    try:
        return _non_negative_int(text.strip())
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))


def _first_numbers(count: int, method: str) -> List[int]:
    """Return F(0) .. F(count - 1) computed with the chosen method."""
    if method == "iterative":
        # The generator is the natural and fastest way to get a prefix.
        return list(fib_sequence(count))
    # Other methods compute every number independently.
    return [METHODS[method](index) for index in range(count)]


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command line interface and return the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Printing numbers with thousands of digits is allowed on purpose.
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)

    # The two modes exclude each other.
    if args.count is not None and args.nth is not None:
        parser.error("use either COUNT or --nth, not both")

    # Warn (but do not stop) when the slow algorithm gets a large input.
    size = args.nth if args.nth is not None else args.count
    if args.method == "recursive" and size is not None and size > _RECURSIVE_WARNING_THRESHOLD:
        print("Warning: the naive recursive method is very slow for large values.",
              file=sys.stderr)

    try:
        if args.nth is not None:
            # Mode 1: a single number F(N).
            print(f"F({args.nth}) = {METHODS[args.method](args.nth)}")
        else:
            # Mode 2: the first COUNT numbers; ask interactively if not given.
            count = args.count if args.count is not None else _ask_count(parser)
            numbers = _first_numbers(count, args.method)
            print(f"First {count} Fibonacci numbers:")
            print(" ".join(str(number) for number in numbers))
    except ValueError as error:
        # Safety net: library validation errors become friendly messages.
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
