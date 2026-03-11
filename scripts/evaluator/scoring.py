"""Scoring module for comparing numeric values and computing scores."""

import re
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OverallGrade(Enum):
    """Overall grade for a solution."""
    PASS = "pass"       # Meets required digit threshold
    PARTIAL = "partial" # Some digits match but below threshold
    WRONG = "wrong"     # No significant digit match
    INVALID = "invalid" # Could not parse/compare values


@dataclass
class Score:
    """Score for comparing expected vs actual values."""
    grade: OverallGrade
    matching_digits: int
    required_digits: int

    def __bool__(self) -> bool:
        """Return True if grade is PASS."""
        return self.grade == OverallGrade.PASS

    @property
    def passed(self) -> bool:
        """Return True if solution passed."""
        return self.grade == OverallGrade.PASS


# Default scoring settings
DEFAULT_REQUIRED_DIGITS = 20


def compute_score(
    expected: str,
    actual: str,
    required_digits: int = DEFAULT_REQUIRED_DIGITS
) -> Score:
    """
    Compute score by comparing expected and actual numeric values.

    Args:
        expected: Expected numeric value as string
        actual: Actual numeric value from solution as string
        required_digits: Number of significant digits required to pass

    Returns:
        Score with grade, matching digit count, and required threshold
    """
    exp_normalized = _parse_numeric_string(expected)
    act_normalized = _parse_numeric_string(actual)

    if exp_normalized is None or act_normalized is None:
        return Score(
            grade=OverallGrade.INVALID,
            matching_digits=0,
            required_digits=required_digits
        )

    matching = _count_matching_digits(exp_normalized, act_normalized)

    if matching >= required_digits:
        grade = OverallGrade.PASS
    elif matching > 0:
        grade = OverallGrade.PARTIAL
    else:
        grade = OverallGrade.WRONG

    return Score(
        grade=grade,
        matching_digits=matching,
        required_digits=required_digits
    )


def _parse_numeric_string(value: str) -> Optional[str]:
    """Parse a numeric string, handling various formats."""
    if not value:
        return None

    value = value.strip()

    # Handle special values
    if value.lower() in ['+inf', '-inf', 'nan', 'none', 'null']:
        return None

    # Skip complex numbers
    if 'j' in value.lower():
        return None

    # Handle multi-line output - find first numeric line
    lines = value.split('\n')
    for line in lines:
        line = line.strip()
        if re.match(r'^-?\d', line):
            value = line
            break

    # Validate it's a parseable number (without truncation that breaks scientific notation)
    if not re.match(r'^-?\d+\.?\d*(e[+-]?\d+)?$', value, re.IGNORECASE):
        return None

    return value


def _count_matching_digits(expected: str, actual: str) -> int:
    """Count how many significant digits match between two numeric strings."""
    expected = expected.strip()
    actual = actual.strip()

    # Check sign
    exp_negative = expected.startswith('-')
    act_negative = actual.startswith('-')

    if exp_negative != act_negative:
        return 0

    # Remove sign
    expected = expected.lstrip('-+')
    actual = actual.lstrip('-+')

    def to_decimal_str(s: str) -> tuple[str, int]:
        """Convert to (digits, decimal_position) format."""
        s = s.lower()
        if 'e' in s:
            mantissa, exp = s.split('e')
            exp = int(exp)
        else:
            mantissa = s
            exp = 0

        if '.' in mantissa:
            int_part, frac_part = mantissa.split('.')
            dec_pos = len(int_part) + exp
            digits = int_part + frac_part
        else:
            dec_pos = len(mantissa) + exp
            digits = mantissa

        stripped = digits.lstrip('0') or '0'
        dec_pos -= (len(digits) - len(stripped))
        return stripped, dec_pos

    try:
        exp_digits, exp_dec = to_decimal_str(expected)
        act_digits, act_dec = to_decimal_str(actual)
    except (ValueError, IndexError):
        return 0

    # Decimal position must match (same order of magnitude)
    if exp_dec != act_dec:
        # Values may still be close across a rounding boundary
        # (e.g. 0.9999999 vs 1.0000001). Fall back to relative-error estimate.
        return _matching_digits_from_relative_error(expected, actual)

    # Count matching digits from the left
    matching = 0
    for i in range(min(len(exp_digits), len(act_digits))):
        if exp_digits[i] == act_digits[i]:
            matching += 1
        else:
            break

    return matching


def _matching_digits_from_relative_error(expected: str, actual: str) -> int:
    """Estimate matching significant digits via relative error using Decimal."""
    try:
        exp_val = Decimal(expected)
        act_val = Decimal(actual)
        if exp_val == 0:
            return 0
        rel_err = abs((exp_val - act_val) / exp_val)
        if rel_err == 0:
            return 50  # Effectively exact match
        # Number of matching significant digits ≈ -log10(relative_error)
        # Use Decimal.ln() / ln(10)
        import math
        digits = -float(rel_err.ln()) / math.log(10)
        return max(0, int(digits))
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return 0
