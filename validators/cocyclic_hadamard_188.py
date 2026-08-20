"""HorizonMath adapter for cocyclic Hadamard matrices of order 188."""

from typing import Any

from conjectures import verify_cocyclic_hadamard_188

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_cocyclic_hadamard_188, solution)
