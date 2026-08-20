"""HorizonMath adapter for the standard Baillie-PSW primality test."""

from typing import Any

from conjectures import verify_bpsw_standard

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_bpsw_standard, solution)
