"""HorizonMath adapter for symmetric conference matrices of order 86."""

from typing import Any

from conjectures import verify_symmetric_conference_matrix_86

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_symmetric_conference_matrix_86, solution)
