"""HorizonMath adapter for Steiner systems S(3,6,46)."""

from typing import Any

from conjectures import verify_steiner_3_6_46

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_steiner_3_6_46, solution)
