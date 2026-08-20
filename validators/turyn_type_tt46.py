"""HorizonMath adapter for Turyn-type sequences TT(46)."""

from typing import Any

from conjectures import verify_turyn_type_tt46

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_turyn_type_tt46, solution)
