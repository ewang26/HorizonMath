"""HorizonMath adapter for Wall-Sun-Sun primes."""

from typing import Any

from conjectures import verify_wall_sun_sun

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_wall_sun_sun, solution)
