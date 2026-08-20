"""HorizonMath adapter for Euler's sum-of-powers problem at k=6."""

from functools import partial
from typing import Any

from conjectures import verify_euler_sum_of_powers

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


_verify_k6 = partial(verify_euler_sum_of_powers, k=6)


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(_verify_k6, solution)
