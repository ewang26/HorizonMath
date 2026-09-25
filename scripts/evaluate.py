#!/usr/bin/env python3
"""
Evaluation harness for OpenMath benchmark.

This script evaluates LLM-generated solutions against:
1. Ground-truth numeric values (numeric mode)
2. Validators with baseline comparison (benchmark mode)

Usage:
    # Numeric evaluation (default for ground_truth_computable problems)
    python scripts/evaluate.py --llm-output <file> --problem-index <index>

    # Benchmark evaluation (for benchmark_best_known problems)
    python scripts/evaluate.py --llm-output <file> --problem-index <index> --mode benchmark

    # Auto-detect mode based on problem's evaluation_mode
    python scripts/evaluate.py --llm-output <file> --problem-index <index> --mode auto

    # List problems by mode
    python scripts/evaluate.py --list-problems --mode benchmark
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from evaluator import extract_proposed_solution, execute_sandboxed, evaluate_single

# Default precision settings
DEFAULT_PRECISION_DIGITS = 20  # Number of digits that must match
EXECUTION_TIMEOUT = 300  # 5 minutes
VALIDATOR_TIMEOUT = 600  # 10 minutes


def _run_validator(validator_func, solution):
    """Run a validator function in a subprocess-friendly way."""
    return validator_func(solution)


def run_validator_with_timeout(validator_func, solution, timeout=VALIDATOR_TIMEOUT):
    """
    Run a validator with a timeout using a subprocess.

    Returns a ValidationResult on success, or a failure ValidationResult on timeout.
    """
    from validators import failure

    executor = ProcessPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_run_validator, validator_func, solution)
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        return failure(f"Validator timed out after {timeout}s")
    except Exception as e:
        return failure(f"Validator subprocess error: {type(e).__name__}: {e}")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


@dataclass
class EvaluationResult:
    """Result of evaluating a single problem (numeric mode)."""
    problem_index: int
    problem_title: str
    success: bool
    error_type: Optional[str] = None  # 'extraction', 'execution', 'comparison'
    error_message: Optional[str] = None
    expected_value: Optional[str] = None
    actual_value: Optional[str] = None
    matching_digits: Optional[int] = None


@dataclass
class BenchmarkEvaluationResult:
    """Result of evaluating a benchmark problem."""
    problem_id: str
    problem_index: int
    problem_title: str
    valid: bool                          # Validator passed
    validator_message: str
    validator_metrics: dict
    baseline_comparison: Optional[dict]  # BaselineComparison.to_dict()
    error_type: Optional[str] = None     # 'extraction', 'validation', 'no_validator'
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def load_problems(data_path: Path) -> list[dict]:
    """Load problems from the JSON file."""
    with open(data_path) as f:
        return json.load(f)


def get_evaluatable_problems(problems: list[dict]) -> list[tuple[int, dict]]:
    """Return list of (index, problem) for problems that can be evaluated numerically."""
    return [
        (i, p) for i, p in enumerate(problems)
        if p.get('evaluation_mode') == 'ground_truth_computable'
    ]


def get_benchmark_problems(problems: list[dict]) -> list[tuple[int, dict]]:
    """Return list of (index, problem) for benchmark_best_known problems."""
    return [
        (i, p) for i, p in enumerate(problems)
        if p.get('evaluation_mode') == 'benchmark_best_known'
    ]


def get_construction_problems(problems: list[dict]) -> list[tuple[int, dict]]:
    """Return list of (index, problem) for new_construction problems."""
    return [
        (i, p) for i, p in enumerate(problems)
        if p.get('evaluation_mode') == 'new_construction'
    ]


def evaluate_problem(
    problem: dict,
    problem_index: int,
    llm_output: str,
    required_digits: int = DEFAULT_PRECISION_DIGITS
) -> EvaluationResult:
    """Evaluate an LLM's solution for a single problem (numeric mode).

    Uses evaluate_single() from the evaluator module and converts the result
    to the local EvaluationResult format for backward compatibility.
    """
    title = problem.get('id', f'problem_{problem_index}')

    # Use the evaluator module's evaluate_single function
    evaluator_result = evaluate_single(
        llm_output=llm_output,
        problem=problem,
        problem_index=problem_index,
        required_digits=required_digits,
        timeout=EXECUTION_TIMEOUT,
    )

    # Convert evaluator's EvaluationResult to local EvaluationResult format
    # Map error_stage to error_type (local uses different naming)
    error_type = evaluator_result.error_stage
    if error_type == 'scoring':
        # Check if it's a no_ground_truth case or a comparison failure
        if evaluator_result.error_message and 'no numeric_value' in evaluator_result.error_message:
            error_type = 'no_ground_truth'
        else:
            error_type = 'comparison'

    # Get matching_digits from score if available
    matching_digits = None
    if evaluator_result.score is not None:
        matching_digits = evaluator_result.score.matching_digits

    return EvaluationResult(
        problem_index=problem_index,
        problem_title=title,
        success=evaluator_result.success,
        error_type=error_type if not evaluator_result.success else None,
        error_message=evaluator_result.error_message,
        expected_value=evaluator_result.expected_value,
        actual_value=evaluator_result.actual_value,
        matching_digits=matching_digits
    )


def evaluate_benchmark_problem(
    problem: dict,
    problem_index: int,
    llm_output: str,
    baselines: dict[str, dict]
) -> BenchmarkEvaluationResult:
    """
    Evaluate a benchmark problem using validators and baseline comparison.

    The LLM outputs a proposed_solution() Python function that returns a
    construction (dict/object). We extract and execute this function using
    the evaluator module, then pass the result to the problem-specific validator.

    Args:
        problem: Problem dict from problems_full.json
        problem_index: Index of the problem
        llm_output: Raw LLM output text containing proposed_solution() function
        baselines: Dict of baseline data keyed by problem_id

    Returns:
        BenchmarkEvaluationResult
    """
    # Import here to avoid circular imports - use relative imports
    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from validator_registry import get_validator, has_validator
    from baseline_comparator import compare_against_baseline

    problem_id = problem.get('id', f'problem_{problem_index}')
    title = problem.get('id', f'problem_{problem_index}')

    # Check if validator exists
    if not has_validator(problem_id):
        return BenchmarkEvaluationResult(
            problem_id=problem_id,
            problem_index=problem_index,
            problem_title=title,
            valid=False,
            validator_message="",
            validator_metrics={},
            baseline_comparison=None,
            error_type='no_validator',
            error_message=f"No validator found for problem {problem_id}"
        )

    # Extract proposed_solution() function from LLM output using evaluator module
    extraction = extract_proposed_solution(llm_output)
    if not extraction:
        return BenchmarkEvaluationResult(
            problem_id=problem_id,
            problem_index=problem_index,
            problem_title=title,
            valid=False,
            validator_message="",
            validator_metrics={},
            baseline_comparison=None,
            error_type='extraction',
            error_message=extraction.error_message or 'Could not extract proposed_solution() function from LLM output'
        )

    # Execute the function to get the construction (returns JSON) using evaluator module
    execution = execute_sandboxed(extraction.code, timeout=EXECUTION_TIMEOUT, return_json=True)
    if not execution:
        return BenchmarkEvaluationResult(
            problem_id=problem_id,
            problem_index=problem_index,
            problem_title=title,
            valid=False,
            validator_message="",
            validator_metrics={},
            baseline_comparison=None,
            error_type='execution',
            error_message=execution.error_message or 'Execution failed'
        )

    # Parse the JSON result
    try:
        solution = json.loads(execution.output)
    except json.JSONDecodeError as e:
        return BenchmarkEvaluationResult(
            problem_id=problem_id,
            problem_index=problem_index,
            problem_title=title,
            valid=False,
            validator_message="",
            validator_metrics={},
            baseline_comparison=None,
            error_type='execution',
            error_message=f'proposed_solution() did not return valid JSON: {e}'
        )

    # Run validator with timeout
    validator = get_validator(problem_id)
    try:
        validation_result = run_validator_with_timeout(validator, solution)
    except Exception as e:
        return BenchmarkEvaluationResult(
            problem_id=problem_id,
            problem_index=problem_index,
            problem_title=title,
            valid=False,
            validator_message="",
            validator_metrics={},
            baseline_comparison=None,
            error_type='validation',
            error_message=f"Validator error: {type(e).__name__}: {e}"
        )

    # Compare against baseline if validation passed
    baseline_comparison = None
    if validation_result.valid and validation_result.metrics:
        comparison = compare_against_baseline(
            problem_id,
            validation_result.metrics,
            baselines
        )
        baseline_comparison = comparison.to_dict()

    return BenchmarkEvaluationResult(
        problem_id=problem_id,
        problem_index=problem_index,
        problem_title=title,
        valid=validation_result.valid,
        validator_message=validation_result.message,
        validator_metrics=validation_result.metrics,
        baseline_comparison=baseline_comparison
    )


@dataclass
class ConstructionEvaluationResult:
    """Result of evaluating a construction problem (pass/fail, no baseline)."""
    problem_id: str
    problem_index: int
    problem_title: str
    valid: bool
    validator_message: str
    error_type: Optional[str] = None     # 'extraction', 'execution', 'validation', 'no_validator'
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def evaluate_construction_problem(
    problem: dict,
    problem_index: int,
    llm_output: str,
) -> ConstructionEvaluationResult:
    """
    Evaluate a construction problem using validators (pass/fail, no baseline comparison).

    Same extraction/execution/validation pipeline as evaluate_benchmark_problem()
    but skips the baseline comparison step entirely.
    """
    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from validator_registry import get_validator, has_validator

    problem_id = problem.get('id', f'problem_{problem_index}')
    title = problem.get('id', f'problem_{problem_index}')

    if not has_validator(problem_id):
        return ConstructionEvaluationResult(
            problem_id=problem_id,
            problem_index=problem_index,
            problem_title=title,
            valid=False,
            validator_message="",
            error_type='no_validator',
            error_message=f"No validator found for problem {problem_id}"
        )

    extraction = extract_proposed_solution(llm_output)
    if not extraction:
        return ConstructionEvaluationResult(
            problem_id=problem_id,
            problem_index=problem_index,
            problem_title=title,
            valid=False,
            validator_message="",
            error_type='extraction',
            error_message=extraction.error_message or 'Could not extract proposed_solution() function from LLM output'
        )

    execution = execute_sandboxed(extraction.code, timeout=EXECUTION_TIMEOUT, return_json=True)
    if not execution:
        return ConstructionEvaluationResult(
            problem_id=problem_id,
            problem_index=problem_index,
            problem_title=title,
            valid=False,
            validator_message="",
            error_type='execution',
            error_message=execution.error_message or 'Execution failed'
        )

    try:
        solution = json.loads(execution.output)
    except json.JSONDecodeError as e:
        return ConstructionEvaluationResult(
            problem_id=problem_id,
            problem_index=problem_index,
            problem_title=title,
            valid=False,
            validator_message="",
            error_type='execution',
            error_message=f'proposed_solution() did not return valid JSON: {e}'
        )

    validator = get_validator(problem_id)
    try:
        validation_result = run_validator_with_timeout(validator, solution)
    except Exception as e:
        return ConstructionEvaluationResult(
            problem_id=problem_id,
            problem_index=problem_index,
            problem_title=title,
            valid=False,
            validator_message="",
            error_type='validation',
            error_message=f"Validator error: {type(e).__name__}: {e}"
        )

    return ConstructionEvaluationResult(
        problem_id=problem_id,
        problem_index=problem_index,
        problem_title=title,
        valid=validation_result.valid,
        validator_message=validation_result.message,
    )


def print_construction_result(result: ConstructionEvaluationResult, verbose: bool = False) -> None:
    """Print construction evaluation result in a readable format."""
    status = "✓ VALID" if result.valid else "✗ INVALID"
    print(f"{status} [{result.problem_index:2d}] {result.problem_title}")
    print(f"      Problem ID: {result.problem_id}")

    if result.error_type:
        print(f"      Error: {result.error_type} - {result.error_message}")
    else:
        print(f"      Message: {result.validator_message}")

    if verbose:
        print(f"\n      Full result:\n{result.to_json()}")


def print_result(result: EvaluationResult, verbose: bool = False) -> None:
    """Print numeric evaluation result in a readable format."""
    status = "✓ PASS" if result.success else "✗ FAIL"
    print(f"{status} [{result.problem_index:2d}] {result.problem_title}")

    if not result.success:
        print(f"      Error: {result.error_type} - {result.error_message}")

    if verbose and result.expected_value:
        print(f"      Expected: {result.expected_value}")
    if verbose and result.actual_value:
        print(f"      Actual:   {result.actual_value}")
    if result.matching_digits is not None:
        print(f"      Matching digits: {result.matching_digits}")


def print_benchmark_result(result: BenchmarkEvaluationResult, verbose: bool = False) -> None:
    """Print benchmark evaluation result in a readable format."""
    status = "✓ VALID" if result.valid else "✗ INVALID"
    print(f"{status} [{result.problem_index:2d}] {result.problem_title}")
    print(f"      Problem ID: {result.problem_id}")

    if result.error_type:
        print(f"      Error: {result.error_type} - {result.error_message}")
    else:
        print(f"      Message: {result.validator_message}")

        if result.validator_metrics:
            metrics_str = ", ".join(f"{k}={v}" for k, v in result.validator_metrics.items())
            print(f"      Metrics: {metrics_str}")

        if result.baseline_comparison:
            bc = result.baseline_comparison
            print(f"      Baseline: {bc['result']}")
            if bc['achieved_value'] is not None:
                print(f"        Achieved: {bc['achieved_value']}")
            if bc['baseline_value'] is not None:
                print(f"        Baseline: {bc['baseline_value']} ({bc['direction']})")
            if bc['improvement_percent'] is not None:
                sign = "+" if bc['improvement_percent'] > 0 else ""
                print(f"        Improvement: {sign}{bc['improvement_percent']}%")

    if verbose:
        print(f"\n      Full result:\n{result.to_json()}")


def determine_mode(problem: dict, explicit_mode: str) -> str:
    """Determine evaluation mode based on problem and explicit setting."""
    if explicit_mode in ('numeric', 'benchmark', 'construction'):
        return explicit_mode

    # Auto-detect from problem's evaluation_mode
    eval_mode = problem.get('evaluation_mode', '')
    if eval_mode == 'benchmark_best_known':
        return 'benchmark'
    if eval_mode == 'new_construction':
        return 'construction'
    return 'numeric'


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LLM solutions for OpenMath benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate numeric problem (single)
  python scripts/evaluate.py --llm-output solution.txt --problem-index 0

  # Batch evaluate all problems from JSONL
  python scripts/evaluate.py --batch results/openai_o1.jsonl

  # Batch evaluate with custom output path
  python scripts/evaluate.py --batch results/openai_o1.jsonl --output results/o1_eval

  # Evaluate benchmark problem
  python scripts/evaluate.py --llm-output construction.json --problem-index 45 --mode benchmark

  # Auto-detect mode
  python scripts/evaluate.py --llm-output output.txt --problem-index 45 --mode auto

  # List benchmark problems
  python scripts/evaluate.py --list-problems --mode benchmark
        """
    )
    # Batch mode arguments
    parser.add_argument('--batch', type=str, metavar='JSONL_FILE',
                        help='Batch evaluate from JSONL file (one LLM output per line)')
    parser.add_argument('--output', type=str, metavar='OUTPUT_PATH',
                        help='Output path for batch results (default: input path with .eval suffix)')

    # Single problem mode arguments
    parser.add_argument('--llm-output', type=str, help='LLM output (file path or raw string)')
    parser.add_argument('--problem-index', type=int, help='Problem index to evaluate')
    parser.add_argument('--problem-id', type=str, help='Problem ID to evaluate (for benchmark mode)')
    parser.add_argument('--problem-title', type=str, help='Problem title to evaluate')
    parser.add_argument('--mode', type=str, choices=['numeric', 'benchmark', 'construction', 'auto'], default='auto',
                        help='Evaluation mode (default: auto)')
    parser.add_argument('--required-digits', type=int, default=DEFAULT_PRECISION_DIGITS,
                        help=f'Number of matching digits required for numeric mode (default: {DEFAULT_PRECISION_DIGITS})')
    parser.add_argument('--data-file', type=str, default='data/problems_full.json',
                        help='Path to problems JSON file')
    parser.add_argument('--baselines-file', type=str, default='data/baselines.json',
                        help='Path to baselines JSON file')
    parser.add_argument('--list-problems', action='store_true', help='List evaluatable problems')
    parser.add_argument('--json', action='store_true', help='Output result as JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / args.data_file
    baselines_path = project_root / args.baselines_file

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    problems = load_problems(data_path)

    # List problems mode
    if args.list_problems:
        if args.mode == 'benchmark':
            benchmark_probs = get_benchmark_problems(problems)
            print(f"Benchmark problems ({len(benchmark_probs)} total):\n")
            for idx, prob in benchmark_probs:
                print(f"  [{idx:2d}] {prob['id']}")
        elif args.mode == 'construction':
            construction_probs = get_construction_problems(problems)
            print(f"Construction problems ({len(construction_probs)} total):\n")
            for idx, prob in construction_probs:
                print(f"  [{idx:2d}] {prob['id']}")
        else:
            evaluatable = get_evaluatable_problems(problems)
            benchmark_probs = get_benchmark_problems(problems)
            construction_probs = get_construction_problems(problems)
            print(f"Numeric evaluatable problems ({len(evaluatable)} of {len(problems)}):\n")
            for idx, prob in evaluatable:
                print(f"  [{idx:2d}] {prob['id']}")
            print(f"\nBenchmark problems: {len(benchmark_probs)} problems")
            print(f"Construction problems: {len(construction_probs)} problems")
        return

    # Batch evaluation mode
    if args.batch:
        from evaluator import evaluate_batch

        batch_path = Path(args.batch)
        if not batch_path.exists():
            print(f"Error: Batch file not found: {batch_path}", file=sys.stderr)
            sys.exit(1)

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            # Default: same as input but without .jsonl extension
            output_path = batch_path.with_suffix('')

        # Get evaluatable problem indices (ground_truth_computable only)
        evaluatable_indices = set(i for i, _ in get_evaluatable_problems(problems))

        print(f"Batch evaluating: {batch_path}")
        print(f"Problems to evaluate: {len(evaluatable_indices)} of {len(problems)}")
        print(f"Required digits: {args.required_digits}")
        print()

        batch_results = evaluate_batch(
            jsonl_path=batch_path,
            problems=problems,
            required_digits=args.required_digits,
            timeout=EXECUTION_TIMEOUT,
            problem_indices=evaluatable_indices,
            verbose=args.verbose,
        )

        # Save results
        jsonl_path, summary_path = batch_results.save(output_path)

        print()
        print(f"Results saved to: {jsonl_path}")
        print(f"Summary saved to: {summary_path}")
        print()

        # Print summary
        summary = batch_results.summary()
        print("=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Total evaluated: {summary['total']}")
        print(f"Passed: {summary['passed']} ({summary['pass_rate']})")
        print(f"Failed: {summary['failed']}")

        if summary['by_error_stage']:
            print("\nFailures by stage:")
            for stage, count in sorted(summary['by_error_stage'].items()):
                print(f"  {stage}: {count}")

        sys.exit(0 if summary['failed'] == 0 else 1)

    # Need LLM output for single-problem evaluation
    if not args.llm_output:
        parser.print_help()
        sys.exit(1)

    # Load LLM output
    llm_output_path = Path(args.llm_output)
    if llm_output_path.exists():
        llm_output = llm_output_path.read_text()
    else:
        llm_output = args.llm_output

    # Find problem by index, ID, or title
    problem_index = args.problem_index

    if problem_index is None and args.problem_id:
        # Find by problem ID
        for i, p in enumerate(problems):
            if p.get('id') == args.problem_id:
                problem_index = i
                break
        if problem_index is None:
            print(f"Error: Problem not found with ID '{args.problem_id}'", file=sys.stderr)
            sys.exit(1)

    if problem_index is None and args.problem_title:
        for i, p in enumerate(problems):
            if args.problem_title.lower() in p.get('id', '').lower():
                problem_index = i
                break
        if problem_index is None:
            print(f"Error: Problem not found with ID containing '{args.problem_title}'", file=sys.stderr)
            sys.exit(1)

    if problem_index is None:
        print("Error: Must specify --problem-index, --problem-id, or --problem-title", file=sys.stderr)
        sys.exit(1)

    if problem_index < 0 or problem_index >= len(problems):
        print(f"Error: Problem index {problem_index} out of range (0-{len(problems)-1})", file=sys.stderr)
        sys.exit(1)

    problem = problems[problem_index]
    mode = determine_mode(problem, args.mode)

    # Evaluate based on mode
    if mode == 'construction':
        result = evaluate_construction_problem(problem, problem_index, llm_output)

        if args.json:
            print(result.to_json())
        else:
            print_construction_result(result, verbose=args.verbose)

        sys.exit(0 if result.valid else 1)

    elif mode == 'benchmark':
        # Load baselines
        script_dir = Path(__file__).parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        from baseline_comparator import load_baselines
        baselines = load_baselines(baselines_path) if baselines_path.exists() else {}

        result = evaluate_benchmark_problem(problem, problem_index, llm_output, baselines)

        if args.json:
            print(result.to_json())
        else:
            print_benchmark_result(result, verbose=args.verbose)

        sys.exit(0 if result.valid else 1)

    else:
        # Numeric mode
        result = evaluate_problem(problem, problem_index, llm_output, args.required_digits)

        if args.json:
            print(json.dumps(asdict(result), indent=2))
        else:
            print_result(result, verbose=args.verbose)

        sys.exit(0 if result.success else 1)


if __name__ == '__main__':
    main()
