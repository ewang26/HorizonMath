"""HorizonMath adapter for the two-dimensional complex Jacobian conjecture."""

from typing import Any

from conjectures import verify_complex_jacobian

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_complex_jacobian, solution)
