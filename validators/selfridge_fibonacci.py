"""Construction verifier for the Weak Selfridge/Fibonacci challenge."""

from typing import Any

from ._construction_number_theory import verify_weak_selfridge_fibonacci

from ._conjecture_adapter import exact_witness_validator, validate_with_core
from .utils import ValidationResult


@exact_witness_validator
def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_weak_selfridge_fibonacci, solution)
