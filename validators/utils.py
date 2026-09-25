#!/usr/bin/env python3
"""
Common utilities for validators.

Provides shared functionality for precision arithmetic, parsing,
and validation result formatting.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Union


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    valid: bool
    message: str
    metrics: dict

    def to_dict(self) -> dict:
        return {
            'valid': self.valid,
            'message': self.message,
            'metrics': self.metrics
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def load_solution(solution_arg: str) -> Any:
    """
    Load a solution from a JSON file or JSON string.

    Args:
        solution_arg: Either a path to a JSON file or a JSON string

    Returns:
        Parsed solution object
    """
    path = Path(solution_arg)
    if path.exists() and path.suffix == '.json':
        with open(path) as f:
            return json.load(f)
    else:
        try:
            return json.loads(solution_arg)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse solution: {solution_arg}")


def parse_rational(value: Union[str, int, float, list]) -> Fraction:
    """
    Parse a value as a rational number.

    Accepts:
        - Integer or float
        - String like "3/4" or "1.5"
        - List [numerator, denominator]

    Returns:
        Fraction object
    """
    if isinstance(value, (int, float)):
        return Fraction(value).limit_denominator(10**15)
    elif isinstance(value, str):
        if '/' in value:
            num, denom = value.split('/')
            return Fraction(int(num.strip()), int(denom.strip()))
        else:
            return Fraction(value).limit_denominator(10**15)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        return Fraction(int(value[0]), int(value[1]))
    else:
        raise ValueError(f"Cannot parse as rational: {value}")


def parse_integer(value: Union[str, int]) -> int:
    """
    Parse a value as an integer, handling large numbers.

    Args:
        value: String or integer representation

    Returns:
        Python integer
    """
    if isinstance(value, int):
        return value
    elif isinstance(value, str):
        return int(value.strip())
    else:
        raise ValueError(f"Cannot parse as integer: {value}")


def gcd(*args: int) -> int:
    """Compute GCD of multiple integers."""
    from math import gcd as math_gcd
    from functools import reduce
    return reduce(math_gcd, [abs(x) for x in args])


def output_result(result: ValidationResult) -> None:
    """Output validation result and exit with appropriate code."""
    print(result.to_json())
    sys.exit(0 if result.valid else 1)


def success(message: str, **metrics) -> ValidationResult:
    """Create a successful validation result."""
    return ValidationResult(valid=True, message=message, metrics=metrics)


def failure(message: str, **metrics) -> ValidationResult:
    """Create a failed validation result."""
    return ValidationResult(valid=False, message=message, metrics=metrics)


def sage_not_found_message() -> str:
    """Standard SageMath-not-found message for validators."""
    return (
        "SageMath not found. Install SageMath and ensure `sage` is on PATH, "
        "or set SAGE_CMD to the Sage executable."
    )


def _resolve_sage_command() -> list[str] | None:
    """Resolve the Sage executable command, optionally from SAGE_CMD."""
    override = os.environ.get("SAGE_CMD", "").strip()
    if override:
        parts = shlex.split(override)
        if parts:
            return parts

    sage_path = shutil.which("sage")
    if sage_path:
        return [sage_path]

    return None


def run_sage_script(script_path: Union[str, Path], timeout: int) -> subprocess.CompletedProcess[str]:
    """Run a Sage script file and return the completed subprocess result."""
    sage_cmd = _resolve_sage_command()
    if sage_cmd is None:
        raise FileNotFoundError(sage_not_found_message())

    return subprocess.run(
        [*sage_cmd, str(script_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
