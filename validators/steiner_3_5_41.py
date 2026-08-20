"""HorizonMath adapter for Steiner systems S(3,5,41)."""

from typing import Any

from conjectures import verify_steiner_3_5_41

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_steiner_3_5_41, solution)
