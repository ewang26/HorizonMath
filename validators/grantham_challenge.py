"""HorizonMath adapter for Grantham's quadratic Frobenius challenge."""

from typing import Any

from conjectures import verify_grantham_challenge

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_grantham_challenge, solution)
