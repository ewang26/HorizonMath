"""Construction verifier for the standard Baillie-PSW primality test."""

from typing import Any

from ._construction_number_theory import verify_bpsw_standard

from ._conjecture_adapter import exact_witness_validator, validate_with_core
from .utils import ValidationResult


@exact_witness_validator
def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_bpsw_standard, solution)
