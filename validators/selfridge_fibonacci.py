"""HorizonMath adapter for the Weak Selfridge/Fibonacci challenge."""

from typing import Any

from conjectures import verify_weak_selfridge_fibonacci

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_weak_selfridge_fibonacci, solution)
