"""HorizonMath adapter for skew-Hadamard matrices of order 356."""

from typing import Any

from conjectures import verify_skew_hadamard_356

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_skew_hadamard_356, solution)
