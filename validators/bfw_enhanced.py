"""HorizonMath adapter for the Baillie-Fiori-Wagstaff strengthened test."""

from typing import Any

from conjectures import verify_bfw_enhanced

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_bfw_enhanced, solution)
