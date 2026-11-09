"""Construction verifier for Wall-Sun-Sun primes."""

from typing import Any

from ._construction_number_theory import verify_wall_sun_sun

from ._conjecture_adapter import exact_witness_validator, validate_with_core
from .utils import ValidationResult


@exact_witness_validator
def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_wall_sun_sun, solution)
