"""Bridge exact conjecture validators into the evaluator contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ._construction_common import VerificationError

from .utils import ValidationResult, failure


def exact_witness_validator(validator: Callable[[Any], ValidationResult]):
    """Mark validators that accept exact witness encodings beyond plain JSON."""
    validator.exact_witness_transport = True
    return validator


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
