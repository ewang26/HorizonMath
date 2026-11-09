"""Construction verifier for cocyclic Hadamard matrices of order 188."""

from typing import Any

from ._construction_combinatorics import verify_cocyclic_hadamard_188

from ._conjecture_adapter import exact_witness_validator, validate_with_core
from .utils import ValidationResult


@exact_witness_validator
def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_cocyclic_hadamard_188, solution)
