"""HorizonMath adapter for RSHCD(196,-1)."""

from typing import Any

from conjectures import verify_rshcd_196_type_minus

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_rshcd_196_type_minus, solution)
