"""Bridge exact conjecture validators into the HorizonMath evaluator contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from conjectures import VerificationError

from .utils import ValidationResult, failure


def validate_with_core(
    core_validator: Callable[[Any], Any], solution: Any
) -> ValidationResult:
    """Run a core conjecture validator and translate its result faithfully."""
    try:
        result = core_validator(solution)
    except VerificationError as exc:
        return failure(f"Invalid witness: {exc}")
    except Exception as exc:
        return failure(f"Validation error: {exc}")

    normalized = result.normalized_witness
    metrics = dict(normalized) if isinstance(normalized, Mapping) else {}
    return ValidationResult(
        valid=bool(result.valid),
        message=str(result.reason),
        metrics=metrics,
    )
