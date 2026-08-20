"""HorizonMath adapter for the finite-magma E677/E255 problem."""

from typing import Any

from conjectures import verify_finite_magma_e677_not_e255

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_finite_magma_e677_not_e255, solution)
