"""HorizonMath adapter for the dual finite-magma implication."""

from typing import Any

from conjectures import verify_dual_finite_magma

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_dual_finite_magma, solution)
