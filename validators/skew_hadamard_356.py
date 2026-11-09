"""Construction verifier for skew-Hadamard matrices of order 356."""

from typing import Any

from ._construction_combinatorics import verify_skew_hadamard_356

from ._conjecture_adapter import exact_witness_validator, validate_with_core
from .utils import ValidationResult


@exact_witness_validator
def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_skew_hadamard_356, solution)
