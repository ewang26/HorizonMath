"""Tests for evaluation summary handling of indeterminate compliance checks."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from evaluate_responses import compute_summary


def test_compliance_indeterminate_is_not_counted_as_passed_or_failed():
    evaluations = [
        {
            "mode": "numeric",
            "solvability": 0,
            "success": True,
            "matching_digits": 100,
        },
        {
            "mode": "numeric",
            "solvability": 1,
            "success": False,
            "matching_digits": 100,
            "error_type": "compliance_indeterminate",
        },
        {
            "mode": "numeric",
            "solvability": 1,
            "success": False,
            "matching_digits": 3,
            "error_type": "comparison",
        },
    ]

    summary = compute_summary(evaluations, config={}, duration_seconds=0)

    assert summary["total_problems"] == 3
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["indeterminate"] == 1
