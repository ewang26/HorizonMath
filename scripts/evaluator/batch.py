"""Batch evaluation module for processing multiple LLM outputs."""

import json
import math
from pathlib import Path
from typing import Iterator, Optional

from .code_extraction import extract_proposed_solution, ExtractionStatus
from .sandbox import execute_sandboxed, ExecutionStatus
from .scoring import compute_score, OverallGrade, DEFAULT_REQUIRED_DIGITS
from .results import EvaluationResult, BatchResults


def load_llm_outputs(jsonl_path: Path) -> Iterator[dict]:
    """
    Load LLM outputs from a JSONL file.

    Each line should be a JSON object with at least:
    - problem_id: Problem ID
    - response: LLM response text

    May also include:
    - provider: Model provider
    - model: Model name
    - timestamp: When the response was generated

    Args:
        jsonl_path: Path to JSONL file

    Yields:
        Dict for each line in the file
    """
    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")


def evaluate_single(
    llm_output: str,
    problem: dict,
    problem_index: int,
    required_digits: int = DEFAULT_REQUIRED_DIGITS,
    timeout: int = 300,
) -> EvaluationResult:
    """
    Evaluate a single LLM output against a problem.

    Args:
        llm_output: Raw LLM response text
        problem: Problem dict with 'id', 'numeric_value', etc.
        problem_index: Index of the problem
        required_digits: Number of digits required to pass
        timeout: Execution timeout in seconds

    Returns:
        EvaluationResult with all evaluation details
    """
    problem_id = problem.get('id', f'problem_{problem_index}')
    problem_title = problem.get('title', f'Problem {problem_index}')
    expected_value = problem.get('numeric_value')

    # Per-problem required_digits override (e.g. to reject known-insufficient conjectures)
    if 'required_digits' in problem:
        required_digits = max(required_digits, problem['required_digits'])

    # Cap required_digits at the available precision in the ground truth.
    # For low-precision values, require one fewer digit than available
    # since the last digit typically has uncertainty.
    # For high-precision values, keep the default (e.g. 20).
    if expected_value:
        available = _significant_digits(expected_value)
        if available < required_digits:
            required_digits = max(available - 1, 1)

    # Check for multi-point evaluation (test_points supply the ground truth)
    test_points = problem.get('test_points')

    # Check if problem has ground truth (either numeric_value or test_points)
    if not expected_value and not test_points:
        return EvaluationResult(
            problem_id=problem_id,
            problem_index=problem_index,
            problem_title=problem_title,
            success=False,
            error_stage="scoring",
            error_message="Problem has no numeric_value for comparison",
        )

    # Step 1: Extract code
    extraction = extract_proposed_solution(llm_output)
    if not extraction:
        return EvaluationResult(
            problem_id=problem_id,
            problem_index=problem_index,
            problem_title=problem_title,
            extraction=extraction,
            success=False,
            error_stage="extraction",
            error_message=extraction.error_message or "Could not extract proposed_solution()",
            expected_value=_truncate(expected_value),
        )

    if test_points:
        # Multi-point evaluation mode
        execution = execute_sandboxed(extraction.code, timeout=timeout, test_points=test_points)
        if not execution:
            return EvaluationResult(
                problem_id=problem_id,
                problem_index=problem_index,
                problem_title=problem_title,
                extraction=extraction,
                execution=execution,
                success=False,
                error_stage="execution",
                error_message=execution.error_message or "Execution failed",
                expected_value=_truncate(expected_value),
            )

        # Parse JSON results list
        try:
            results = json.loads(execution.output)
        except (json.JSONDecodeError, TypeError):
            return EvaluationResult(
                problem_id=problem_id,
                problem_index=problem_index,
                problem_title=problem_title,
                extraction=extraction,
                execution=execution,
                success=False,
                error_stage="execution",
                error_message="Multi-point evaluation returned invalid JSON",
                expected_value=_truncate(expected_value),
            )

        # Score each test point, take minimum matching_digits.
        # Per-test-point, cap required digits based on available precision
        # (minus 1 for last-digit uncertainty, same rule as the global cap).
        min_digits = math.inf
        all_passed = True
        for result_str, tp in zip(results, test_points):
            if result_str is None:
                min_digits = 0
                all_passed = False
                break
            point_score = compute_score(tp["expected"], result_str, required_digits)
            point_digits = point_score.matching_digits
            tp_available = _significant_digits(tp["expected"])
            tp_required = max(tp_available - 1, 1) if tp_available < required_digits else required_digits
            if point_digits < tp_required:
                all_passed = False
            min_digits = min(min_digits, point_digits)

        min_digits = int(min_digits) if math.isfinite(min_digits) else 0
        success = all_passed

        from .scoring import Score
        score = Score(
            grade=OverallGrade.PASS if success else (OverallGrade.PARTIAL if min_digits > 0 else OverallGrade.WRONG),
            matching_digits=min_digits,
            required_digits=required_digits,
        )

        return EvaluationResult(
            problem_id=problem_id,
            problem_index=problem_index,
            problem_title=problem_title,
            extraction=extraction,
            execution=execution,
            score=score,
            success=success,
            error_stage=None if success else "scoring",
            error_message=None if success else f"Only {min_digits} digits match at worst test point (required: {required_digits})",
            expected_value=_truncate(expected_value),
            actual_value=_truncate(execution.output) if execution.output else None,
            matching_digits=min_digits,
        )

    # Single-point evaluation mode (original behavior)

    # Step 2: Execute code
    execution = execute_sandboxed(extraction.code, timeout=timeout)
    if not execution:
        return EvaluationResult(
            problem_id=problem_id,
            problem_index=problem_index,
            problem_title=problem_title,
            extraction=extraction,
            execution=execution,
            success=False,
            error_stage="execution",
            error_message=execution.error_message or "Execution failed",
            expected_value=_truncate(expected_value),
        )

    # Step 3: Score result
    score = compute_score(expected_value, execution.output, required_digits)
    success = score.passed

    return EvaluationResult(
        problem_id=problem_id,
        problem_index=problem_index,
        problem_title=problem_title,
        extraction=extraction,
        execution=execution,
        score=score,
        success=success,
        error_stage=None if success else "scoring",
        error_message=None if success else f"Only {score.matching_digits} digits match (required: {required_digits})",
        expected_value=_truncate(expected_value),
        actual_value=_truncate(execution.output) if execution.output else None,
        matching_digits=score.matching_digits,
    )


def evaluate_batch(
    jsonl_path: Path,
    problems: list[dict],
    required_digits: int = DEFAULT_REQUIRED_DIGITS,
    timeout: int = 300,
    problem_indices: Optional[set[int]] = None,
    verbose: bool = False,
) -> BatchResults:
    """
    Evaluate a batch of LLM outputs against problems.

    The JSONL file should have one entry per problem, in order.
    Each entry should have a 'response' field with the LLM output.

    Args:
        jsonl_path: Path to JSONL file with LLM outputs
        problems: List of problem dicts
        required_digits: Number of digits required to pass
        timeout: Execution timeout in seconds per problem
        problem_indices: Optional set of indices to evaluate (None = all)
        verbose: Print progress

    Returns:
        BatchResults with all evaluation results
    """
    batch_results = BatchResults()

    # Try to extract model/provider from first entry
    outputs = list(load_llm_outputs(jsonl_path))

    if outputs:
        first = outputs[0]
        batch_results.model = first.get('model', '')
        batch_results.provider = first.get('provider', '')

    # Build id -> problem mapping
    id_to_problem: dict[str, tuple[int, dict]] = {}
    for i, p in enumerate(problems):
        id_to_problem[p['id']] = (i, p)

    # Evaluate each output
    for entry in outputs:
        problem_id = entry.get('problem_id', '')
        response = entry.get('response', '')

        if problem_id not in id_to_problem:
            if verbose:
                print(f"  Warning: Unknown problem ID: {problem_id}")
            continue

        problem_index, problem = id_to_problem[problem_id]

        # Skip if not in requested indices
        if problem_indices is not None and problem_index not in problem_indices:
            continue

        if verbose:
            print(f"  [{problem_index:2d}] {problem_id[:60]}...")

        result = evaluate_single(
            llm_output=response,
            problem=problem,
            problem_index=problem_index,
            required_digits=required_digits,
            timeout=timeout,
        )

        batch_results.add_result(result)

        if verbose:
            status = "PASS" if result.success else "FAIL"
            if result.score is not None:
                print(f"       {status} ({result.score.matching_digits} digits)")
            else:
                print(f"       {status} ({result.error_stage}: {result.error_message})")

    return batch_results


def _significant_digits(value: str) -> int:
    """Count the number of significant digits in a numeric string."""
    value = value.strip().lstrip('-+')
    # Remove exponent
    if 'e' in value.lower():
        value = value.lower().split('e')[0]
    # Remove decimal point and leading zeros
    value = value.replace('.', '')
    value = value.lstrip('0') or '0'
    # Trailing zeros after decimal are significant, but for integers
    # the total digit count is what matters
    return len(value)


def _truncate(value: Optional[str], max_len: int = 80) -> Optional[str]:
    """Truncate a string for display."""
    if value is None:
        return None
    if len(value) <= max_len:
        return value
    return value[:max_len] + "..."
