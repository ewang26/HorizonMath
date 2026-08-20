"""HorizonMath adapter for Seymour's second-neighborhood conjecture."""

from typing import Any

from conjectures import verify_seymour_second_neighborhood

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_seymour_second_neighborhood, solution)
