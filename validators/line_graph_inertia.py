"""HorizonMath adapter for the line-graph inertia conjecture."""

from typing import Any

from conjectures import verify_line_graph_inertia

from ._conjecture_adapter import validate_with_core
from .utils import ValidationResult


def validate(solution: Any) -> ValidationResult:
    return validate_with_core(verify_line_graph_inertia, solution)
