"""Construction verifier for RSHCD(196,-1)."""

from typing import Any

from ._construction_combinatorics import verify_rshcd_196_type_minus

from ._conjecture_adapter import exact_witness_validator, validate_with_core
from .utils import ValidationResult


@exact_witness_validator
def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_rshcd_196_type_minus, solution)
