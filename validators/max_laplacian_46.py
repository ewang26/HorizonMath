"""HorizonMath adapter for open maximum-Laplacian Bound 46."""

from typing import Any

from conjectures import verify_max_laplacian_46

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_max_laplacian_46, solution)
