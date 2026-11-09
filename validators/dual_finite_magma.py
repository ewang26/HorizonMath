"""Construction verifier for the dual finite-magma implication."""

from typing import Any

from ._construction_combinatorics import verify_dual_finite_magma

from ._conjecture_adapter import exact_witness_validator, validate_with_core
from .utils import ValidationResult


@exact_witness_validator
def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_dual_finite_magma, solution)
