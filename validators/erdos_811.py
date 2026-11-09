"""Construction verifier for the finite Erdős 811 coloring challenge."""

from typing import Any

from ._construction_combinatorics import verify_erdos_811

from ._conjecture_adapter import exact_witness_validator, validate_with_core
from .utils import ValidationResult


@exact_witness_validator
def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_erdos_811, solution)
