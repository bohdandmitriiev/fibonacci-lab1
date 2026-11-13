"""Fibonacci numbers: several implementations.

    python fibonacci.py 10                          # first 10 numbers
    python fibonacci.py --nth 50                    # F(50)
    python fibonacci.py --nth 90 --method doubling  # choose the algorithm
"""

import argparse
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


def _validate_index(value: Any, name: str = "n") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def fib_iterative(n: int) -> int:
    """F(n) with a loop: O(n)."""
    _validate_index(n)
    current, following = 0, 1
    for _ in range(n):
        current, following = following, current + following
    return current


def fib_recursive(n: int) -> int:
    """F(n) with naive recursion: O(2**n), for comparison only."""
    _validate_index(n)
    return _fib_recursive_unchecked(n)


def _fib_recursive_unchecked(n: int) -> int:
    if n < 2:
        return n
    return _fib_recursive_unchecked(n - 1) + _fib_recursive_unchecked(n - 2)


_MEMO_WARMUP_STEP = 200


def fib_memo(n: int) -> int:
    """F(n) with memoized recursion: O(n)."""
    _validate_index(n)
    cache: Dict[int, int] = {0: 0, 1: 1}

    def compute(k: int) -> int:
        if k in cache:
            return cache[k]
        cache[k] = compute(k - 1) + compute(k - 2)
        return cache[k]

    for checkpoint in range(_MEMO_WARMUP_STEP, n, _MEMO_WARMUP_STEP):
        compute(checkpoint)
    return compute(n)


def _fib_pair(n: int) -> Tuple[int, int]:
    if n == 0:
        return 0, 1
    half_value, half_next = _fib_pair(n // 2)
    even = half_value * (2 * half_next - half_value)
    odd = half_value * half_value + half_next * half_next
    if n % 2 == 0:
        return even, odd
    return odd, even + odd


def fib_fast_doubling(n: int) -> int:
    """F(n) with fast doubling: O(log n)."""
    _validate_index(n)
    return _fib_pair(n)[0]


def fib_sequence(count: int) -> Iterator[int]:
    """Yield the first ``count`` Fibonacci numbers."""
    _validate_index(count, "count")
    current, following = 0, 1
    for _ in range(count):
        yield current
        current, following = following, current + following


METHODS: Dict[str, Callable[[int], int]] = {
    "iterative": fib_iterative,
    "recursive": fib_recursive,
    "memo": fib_memo,
    "doubling": fib_fast_doubling,
}

_RECURSIVE_WARNING_THRESHOLD = 35


def _non_negative_int(text: str) -> int:
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
    try:
        text = input("How many Fibonacci numbers do you want to see? ")
    except EOFError:
        parser.error("no input was given")
    try:
        return _non_negative_int(text.strip())
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))


def _first_numbers(count: int, method: str) -> List[int]:
    if method == "iterative":
        return list(fib_sequence(count))
    return [METHODS[method](index) for index in range(count)]


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)

    if args.count is not None and args.nth is not None:
        parser.error("use either COUNT or --nth, not both")

    size = args.nth if args.nth is not None else args.count
    if args.method == "recursive" and size is not None and size > _RECURSIVE_WARNING_THRESHOLD:
        print("Warning: the naive recursive method is very slow for large values.",
              file=sys.stderr)

    try:
        if args.nth is not None:
            print(f"F({args.nth}) = {METHODS[args.method](args.nth)}")
        else:
            count = args.count if args.count is not None else _ask_count(parser)
            numbers = _first_numbers(count, args.method)
            print(f"First {count} Fibonacci numbers:")
            print(" ".join(str(number) for number in numbers))
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
