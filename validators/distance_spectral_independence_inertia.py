"""HorizonMath adapter for the distance-spectral inertia conjecture."""

from typing import Any

from conjectures import verify_distance_spectral_independence_inertia

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_distance_spectral_independence_inertia, solution)
