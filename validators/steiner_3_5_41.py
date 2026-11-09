"""Construction verifier for Steiner systems S(3,5,41)."""

from typing import Any

from ._construction_combinatorics import verify_steiner_3_5_41

from ._conjecture_adapter import exact_witness_validator, validate_with_core
from .utils import ValidationResult


@exact_witness_validator
def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_steiner_3_5_41, solution)
