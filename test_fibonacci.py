"""Unit tests for fibonacci.py (run with: python -m unittest -v)."""

import contextlib
import io
import unittest

import fibonacci
from fibonacci import (
    fib_fast_doubling,
    fib_iterative,
    fib_memo,
    fib_recursive,
    fib_sequence,
)

# All single-number implementations, so each test can loop over them.
ALL_METHODS = {
    "iterative": fib_iterative,
    "recursive": fib_recursive,
    "memo": fib_memo,
    "doubling": fib_fast_doubling,
}

# Well-known reference values: n -> F(n).
KNOWN_VALUES = {
    0: 0,
    1: 1,
    2: 1,
    10: 55,
    20: 6765,
}

# Values that are too large for the naive recursive method to compute quickly.
LARGE_KNOWN_VALUES = {
    50: 12586269025,
    90: 2880067194370816120,
}


class KnownValuesTest(unittest.TestCase):
    """Every method must return the reference values."""

    def test_small_known_values(self) -> None:
        for name, function in ALL_METHODS.items():
            for n, expected in KNOWN_VALUES.items():
                with self.subTest(method=name, n=n):
                    self.assertEqual(function(n), expected)

    def test_large_known_values(self) -> None:
        # The naive recursion is skipped here: F(50) would take hours.
        for name in ("iterative", "memo", "doubling"):
            for n, expected in LARGE_KNOWN_VALUES.items():
                with self.subTest(method=name, n=n):
                    self.assertEqual(ALL_METHODS[name](n), expected)

    def test_very_large_index(self) -> None:
        # Python integers do not overflow; the fast methods must agree.
        # The value 3000 also exceeds the default recursion limit for memo
        # without the warm-up, so this checks that part as well.
        expected = fib_iterative(3000)
        self.assertEqual(fib_fast_doubling(3000), expected)
        self.assertEqual(fib_memo(3000), expected)


class MethodsAgreeTest(unittest.TestCase):
    """All implementations must give identical results."""

    def test_all_methods_agree_for_0_to_25(self) -> None:
        for n in range(26):
            results = {name: function(n) for name, function in ALL_METHODS.items()}
            with self.subTest(n=n):
                self.assertEqual(len(set(results.values())), 1, results)

    def test_recurrence_relation_holds(self) -> None:
        # F(n) = F(n - 1) + F(n - 2) must be true for the fast method too.
        for n in range(2, 200):
            with self.subTest(n=n):
                self.assertEqual(
                    fib_fast_doubling(n),
                    fib_fast_doubling(n - 1) + fib_fast_doubling(n - 2),
                )


class SequenceTest(unittest.TestCase):
    """Tests for the fib_sequence generator."""

    def test_first_ten_numbers(self) -> None:
        self.assertEqual(
            list(fib_sequence(10)), [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        )

    def test_empty_and_single(self) -> None:
        self.assertEqual(list(fib_sequence(0)), [])
        self.assertEqual(list(fib_sequence(1)), [0])

    def test_matches_single_number_function(self) -> None:
        self.assertEqual(
            list(fib_sequence(30)), [fib_iterative(n) for n in range(30)]
        )

    def test_is_lazy_generator(self) -> None:
        generator = fib_sequence(5)
        self.assertEqual(next(generator), 0)
        self.assertEqual(next(generator), 1)

    def test_invalid_count_raises_value_error(self) -> None:
        # A generator runs its body on the first next(), so we consume it.
        for bad_count in (-1, 2.5, "3"):
            with self.subTest(count=bad_count):
                with self.assertRaises(ValueError):
                    list(fib_sequence(bad_count))


class ValidationTest(unittest.TestCase):
    """Invalid arguments must raise ValueError in every method."""

    # Negative number, float, string, boolean and None are all rejected.
    BAD_VALUES = [-1, 3.5, "x", True, False, None]

    def test_invalid_values_raise_value_error(self) -> None:
        for name, function in ALL_METHODS.items():
            for bad_value in self.BAD_VALUES:
                with self.subTest(method=name, value=bad_value):
                    with self.assertRaises(ValueError):
                        function(bad_value)  # type: ignore[arg-type]


class CommandLineTest(unittest.TestCase):
    """Tests for the argparse based command line interface."""

    @staticmethod
    def run_cli(*arguments: str):
        """Run main() and return (exit_code, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = fibonacci.main(list(arguments))
            except SystemExit as exit_request:
                # argparse calls sys.exit(2) on errors.
                code = exit_request.code
        return code, out.getvalue(), err.getvalue()

    def test_print_first_numbers(self) -> None:
        code, out, _ = self.run_cli("10")
        self.assertEqual(code, 0)
        self.assertIn("0 1 1 2 3 5 8 13 21 34", out)

    def test_nth_with_each_method(self) -> None:
        for method in ("iterative", "memo", "doubling"):
            with self.subTest(method=method):
                code, out, _ = self.run_cli("--nth", "50", "--method", method)
                self.assertEqual(code, 0)
                self.assertIn("12586269025", out)

    def test_bad_input_gives_exit_code_2(self) -> None:
        for arguments in (("-5",), ("abc",), ("--nth", "x"), ("3", "--nth", "4")):
            with self.subTest(arguments=arguments):
                code, out, err = self.run_cli(*arguments)
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                self.assertIn("error", err)


if __name__ == "__main__":
    unittest.main()
