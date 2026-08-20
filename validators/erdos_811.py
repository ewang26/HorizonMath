"""HorizonMath adapter for the finite Erdős 811 coloring challenge."""

from typing import Any

from conjectures import verify_erdos_811

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_erdos_811, solution)
