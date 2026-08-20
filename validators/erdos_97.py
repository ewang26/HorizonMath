"""HorizonMath adapter for Erdős Problem 97."""

from typing import Any

from conjectures import verify_erdos_97

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_erdos_97, solution)
