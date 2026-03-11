#!/usr/bin/env python3
"""
Evaluate saved LLM responses for OpenMath benchmark problems.

Reads responses.jsonl from a results directory (produced by run_benchmark.py)
and evaluates each response against ground truth, validators, and baselines.

Usage:
    # Evaluate all responses in a results directory
    uv run scripts/evaluate_responses.py results/openrouter_openai-gpt-5.2_20260205_143022/

    # Re-evaluate (overwrites previous evaluation results)
    uv run scripts/evaluate_responses.py results/openrouter_openai-gpt-5.2_20260205_143022/ --force

    # Custom data/baselines files
    uv run scripts/evaluate_responses.py results/my_run/ --data-file data/problems_full.json --baselines-file data/baselines.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Load .env and map GEMINI_API_KEY -> GOOGLE_API_KEY for compliance checker
_project_root = Path(__file__).parent.parent
from dotenv import load_dotenv
load_dotenv(_project_root / ".env")
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# Add scripts directory to path for imports
_script_dir = Path(__file__).parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from evaluate import (
    evaluate_problem,
    evaluate_benchmark_problem,
    evaluate_construction_problem,
    determine_mode,
    load_problems,
)
from evaluator import extract_proposed_solution, check_solution_compliance
from baseline_comparator import load_baselines


def _sanitize_for_json(obj):
    """Recursively convert numpy/non-native types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    # numpy.bool_ is a subclass of int but not bool — check bool first
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    return obj


def evaluate_response(
    problem: dict,
    problem_index: int,
    response: str,
    baselines: dict[str, dict],
) -> dict:
    """Evaluate an LLM response for a problem."""
    mode = determine_mode(problem, "auto")

    if mode == "construction":
        result = evaluate_construction_problem(problem, problem_index, response)
        return {
            "problem_id": result.problem_id,
            "problem_index": result.problem_index,
            "problem_title": result.problem_title,
            "mode": "construction",
            "solvability": problem.get("solvability"),
            "valid": result.valid,
            "validator_message": result.validator_message,
            "error_type": result.error_type,
            "error_message": result.error_message,
        }
    elif mode == "benchmark":
        result = evaluate_benchmark_problem(problem, problem_index, response, baselines)
        return {
            "problem_id": result.problem_id,
            "problem_index": result.problem_index,
            "problem_title": result.problem_title,
            "mode": "benchmark",
            "solvability": problem.get("solvability"),
            "valid": result.valid,
            "validator_message": result.validator_message,
            "validator_metrics": result.validator_metrics,
            "baseline_comparison": result.baseline_comparison,
            "error_type": result.error_type,
            "error_message": result.error_message,
        }
    else:
        result = evaluate_problem(problem, problem_index, response)
        eval_dict = {
            "problem_id": problem.get("id", f"problem_{problem_index}"),
            "problem_index": result.problem_index,
            "problem_title": result.problem_title,
            "mode": "numeric",
            "solvability": problem.get("solvability"),
            "success": result.success,
            "matching_digits": result.matching_digits,
            "expected_value": result.expected_value,
            "actual_value": result.actual_value,
            "error_type": result.error_type,
            "error_message": result.error_message,
        }

        # Run compliance check on passing numeric solutions
        if result.success:
            extraction = extract_proposed_solution(response)
            if extraction and extraction.code:
                compliance = check_solution_compliance(
                    extraction.code, problem_prompt=problem.get("prompt", "")
                )
                eval_dict["compliance_check"] = True
                eval_dict["compliance_passed"] = compliance.compliant
                eval_dict["compliance_reason"] = compliance.reason
                if not compliance.compliant:
                    eval_dict["success"] = False
                    eval_dict["error_type"] = "compliance"
                    eval_dict["error_message"] = compliance.reason

        return eval_dict


def format_summary_line(index: int, total: int, problem: dict, eval_result: dict) -> str:
    """Format a one-line summary string for a problem result."""
    problem_id = problem["id"]
    mode = eval_result.get("mode", "unknown")

    if mode == "numeric":
        digits = eval_result.get("matching_digits")
        if eval_result.get("success"):
            compliance_str = ""
            if eval_result.get("compliance_check"):
                compliance_str = ", compliant" if eval_result.get("compliance_passed") else ""
            return f"[{index + 1}/{total}] {problem_id} — PASS ({digits} digits{compliance_str})"
        else:
            error_type = eval_result.get("error_type", "unknown")
            if error_type == "compliance":
                reason = eval_result.get("compliance_reason", "")
                return f"[{index + 1}/{total}] {problem_id} — FAIL (compliance: {reason[:60]})"
            digits_str = f", {digits} digits" if digits is not None else ""
            return f"[{index + 1}/{total}] {problem_id} — FAIL ({error_type}{digits_str})"
    elif mode == "construction":
        if eval_result.get("valid"):
            return f"[{index + 1}/{total}] {problem_id} — VALID"
        else:
            error_type = eval_result.get("error_type") or "validation"
            return f"[{index + 1}/{total}] {problem_id} — INVALID ({error_type})"
    else:
        if eval_result.get("valid"):
            bc = eval_result.get("baseline_comparison")
            bc_result = bc.get("result", "no_baseline") if bc else "no_baseline"
            if bc_result in ("beats_baseline", "no_baseline"):
                return f"[{index + 1}/{total}] {problem_id} — VALID ({bc_result})"
            elif bc_result == "matches_baseline":
                return f"[{index + 1}/{total}] {problem_id} — VALID ({bc_result}) — not counted as passed"
            else:
                return f"[{index + 1}/{total}] {problem_id} — VALID (below_baseline)"
        else:
            error_type = eval_result.get("error_type") or "validation"
            return f"[{index + 1}/{total}] {problem_id} — INVALID ({error_type})"


def print_github_response_group(
    index: int,
    total: int,
    problem: dict,
    eval_result: dict,
    response: str,
) -> None:
    """Print model response in a collapsed GitHub Actions log group."""
    summary_line = format_summary_line(index, total, problem, eval_result)
    print(f"::group::{summary_line} — model response")
    print(response)
    print("::endgroup::")


def print_progress(
    index: int,
    total: int,
    problem: dict,
    eval_result: dict,
) -> None:
    """Print progress for a single problem."""
    problem_id = problem["id"]
    mode = eval_result.get("mode", "unknown")

    print(f"\n[{index + 1}/{total}] {problem_id} ({mode})")

    if mode == "numeric":
        digits = eval_result.get("matching_digits")
        if eval_result.get("success"):
            compliance_str = ""
            if eval_result.get("compliance_check"):
                compliance_str = ", compliant" if eval_result.get("compliance_passed") else ""
            print(f"        ✓ PASS - {digits} matching digits{compliance_str}")
        else:
            error_type = eval_result.get("error_type", "unknown")
            if error_type == "compliance":
                reason = eval_result.get("compliance_reason", "")
                print(f"        ✗ FAIL - compliance: {reason[:80]}")
            else:
                error_msg = eval_result.get("error_message", "")
                digits_str = f" ({digits} digits)" if digits is not None else ""
                print(f"        ✗ FAIL - {error_type}: {error_msg[:60]}{digits_str}")
    elif mode == "construction":
        if eval_result.get("valid"):
            print(f"        ✓ VALID")
        else:
            error_type = eval_result.get("error_type") or "validation"
            error_msg = eval_result.get("error_message") or eval_result.get("validator_message") or ""
            print(f"        ✗ INVALID - {error_type}: {error_msg[:60]}")
    else:
        if eval_result.get("valid"):
            bc = eval_result.get("baseline_comparison")
            bc_result = bc.get("result", "no_baseline") if bc else "no_baseline"
            improvement = bc.get("improvement_percent") if bc else None
            if bc_result in ("beats_baseline", "no_baseline"):
                if improvement is not None:
                    sign = "+" if improvement > 0 else ""
                    print(f"        ✓ VALID - {bc_result} ({sign}{improvement}%)")
                else:
                    print(f"        ✓ VALID - {bc_result}")
            else:
                if improvement is not None:
                    print(f"        ~ VALID - {bc_result} ({improvement}%) — not counted as passed")
                else:
                    print(f"        ~ VALID - {bc_result} — not counted as passed")
        else:
            error_type = eval_result.get("error_type") or "validation"
            error_msg = eval_result.get("error_message") or eval_result.get("validator_message") or ""
            print(f"        ✗ INVALID - {error_type}: {error_msg[:60]}")


def compute_summary(
    evaluations: list[dict],
    config: dict,
    duration_seconds: float,
) -> dict:
    """Compute summary statistics from evaluation results."""
    total = len(evaluations)

    # Count passes
    passed = 0
    by_mode = {
        "ground_truth_computable": {"total": 0, "passed": 0},
        "benchmark_best_known": {"total": 0, "passed": 0},
        "new_construction": {"total": 0, "passed": 0},
    }
    by_error_type = {}
    benchmark_results = {
        "beats_baseline": 0,
        "matches_baseline": 0,
        "below_baseline": 0,
        "no_baseline": 0,
    }
    all_numeric_digits = []  # matching digits for all numeric problems
    perfect_matches = 0
    by_solvability = {}  # solvability level -> {"total": N, "passed": N}

    for eval_result in evaluations:
        mode = eval_result.get("mode", "unknown")

        # Track solvability breakdown
        solv = eval_result.get("solvability")
        if solv is not None:
            if solv not in by_solvability:
                by_solvability[solv] = {"total": 0, "passed": 0}
            by_solvability[solv]["total"] += 1

        if mode == "numeric":
            by_mode["ground_truth_computable"]["total"] += 1
            digits = eval_result.get("matching_digits")
            if digits is not None:
                all_numeric_digits.append(digits)
                if digits >= 10:
                    perfect_matches += 1
            if eval_result.get("success"):
                passed += 1
                by_mode["ground_truth_computable"]["passed"] += 1
                if solv is not None:
                    by_solvability[solv]["passed"] += 1
            else:
                error_type = eval_result.get("error_type", "unknown")
                by_error_type[error_type] = by_error_type.get(error_type, 0) + 1

        elif mode == "construction":
            by_mode["new_construction"]["total"] += 1
            if eval_result.get("valid"):
                passed += 1
                by_mode["new_construction"]["passed"] += 1
                if solv is not None:
                    by_solvability[solv]["passed"] += 1
            else:
                error_type = eval_result.get("error_type", "unknown")
                by_error_type[error_type] = by_error_type.get(error_type, 0) + 1

        elif mode == "benchmark":
            by_mode["benchmark_best_known"]["total"] += 1
            if eval_result.get("valid"):
                bc = eval_result.get("baseline_comparison")
                bc_result = bc.get("result", "no_baseline") if bc else "no_baseline"
                if bc_result in ("beats_baseline", "no_baseline"):
                    passed += 1
                    by_mode["benchmark_best_known"]["passed"] += 1
                    if solv is not None:
                        by_solvability[solv]["passed"] += 1
                elif bc_result == "matches_baseline":
                    pass  # valid but not counted as passed
                else:
                    benchmark_results["valid_below_baseline"] = benchmark_results.get("valid_below_baseline", 0) + 1
                if bc_result in benchmark_results:
                    benchmark_results[bc_result] += 1
            else:
                error_type = eval_result.get("error_type", "unknown")
                by_error_type[error_type] = by_error_type.get(error_type, 0) + 1

        else:
            # Runtime errors or unrecognized modes
            error_type = eval_result.get("error_type", "unknown")
            by_error_type[error_type] = by_error_type.get(error_type, 0) + 1

    # Compute pass rates
    pass_rate = f"{(passed / total * 100):.1f}%" if total > 0 else "0.0%"

    for mode_key in by_mode:
        mode_total = by_mode[mode_key]["total"]
        mode_passed = by_mode[mode_key]["passed"]
        by_mode[mode_key]["pass_rate"] = f"{(mode_passed / mode_total * 100):.1f}%" if mode_total > 0 else "0.0%"

    # Compute solvability pass rates
    for solv_key in by_solvability:
        s = by_solvability[solv_key]
        s["pass_rate"] = f"{(s['passed'] / s['total'] * 100):.1f}%" if s["total"] > 0 else "0.0%"
    # Sort by solvability level (numeric keys)
    by_solvability_sorted = dict(sorted(by_solvability.items(), key=lambda x: x[0]))

    avg_digits = sum(all_numeric_digits) / len(all_numeric_digits) if all_numeric_digits else 0.0

    return {
        "run_id": config.get("run_id", ""),
        "timestamp": config.get("timestamp", ""),
        "provider": config.get("provider", ""),
        "model": config.get("model", ""),
        "total_problems": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": pass_rate,
        "by_evaluation_mode": by_mode,
        "by_error_type": by_error_type,
        "by_solvability": by_solvability_sorted,
        "benchmark_results": benchmark_results,
        "numeric_results": {
            "avg_matching_digits": round(avg_digits, 1),
            "perfect_matches": perfect_matches,
            "total_scored": len(all_numeric_digits),
        },
        "duration_seconds": round(duration_seconds, 1),
    }


def print_summary(summary: dict, output_dir: Path) -> None:
    """Print final summary to console."""
    print("\n" + "═" * 58)
    print("                    EVALUATION SUMMARY")
    print("═" * 58)

    total = summary["total_problems"]
    passed = summary["passed"]
    failed = summary["failed"]
    pass_rate = summary["pass_rate"]

    print(f"Total: {total} | Passed: {passed} ({pass_rate}) | Failed: {failed}")

    print("\nBy Mode:")
    for mode_name, mode_stats in summary["by_evaluation_mode"].items():
        mode_total = mode_stats["total"]
        mode_passed = mode_stats["passed"]
        mode_rate = mode_stats["pass_rate"]
        if mode_total > 0:
            print(f"  {mode_name}: {mode_passed}/{mode_total}  ({mode_rate})")

    by_solv = summary.get("by_solvability", {})
    if by_solv:
        solv_labels = {0: "calibration", 1: "likely solvable", 2: "challenging", 3: "possibly unsolvable"}
        print("\nBy Solvability:")
        for level, stats in sorted(by_solv.items(), key=lambda x: int(x[0])):
            s_total = stats["total"]
            s_passed = stats["passed"]
            s_rate = stats["pass_rate"]
            label = solv_labels.get(int(level), f"level {level}")
            if s_total > 0:
                print(f"  {level} ({label}): {s_passed}/{s_total}  ({s_rate})")

    numeric = summary["numeric_results"]
    if numeric["total_scored"] > 0:
        print(f"\nNumeric Problems:")
        print(f"  Avg matching digits (all): {numeric['avg_matching_digits']}")
        print(f"  Perfect (10+ digits): {numeric['perfect_matches']}/{numeric['total_scored']}")

    bench = summary["benchmark_results"]
    if any(v > 0 for v in bench.values()):
        print(f"\nBenchmark Problems:")
        print(f"  Beats baseline:       {bench['beats_baseline']}")
        print(f"  Matches baseline:     {bench['matches_baseline']}")
        print(f"  Below baseline:       {bench['below_baseline']}")
        if bench.get("valid_below_baseline"):
            print(f"  Valid but suboptimal: {bench['valid_below_baseline']} (not counted as passed)")

    errors = summary["by_error_type"]
    if errors:
        print(f"\nErrors by Type:")
        error_strs = [f"{k}: {v}" for k, v in errors.items()]
        print(f"  {' | '.join(error_strs)}")

    # Duration
    duration = summary["duration_seconds"]
    mins = int(duration // 60)
    secs = int(duration % 60)
    print(f"\nDuration: {mins}m {secs}s")
    print(f"Output: {output_dir}/")
    print("═" * 58)


def load_responses(responses_path: Path) -> list[dict]:
    """Load responses from a JSONL file."""
    responses = []
    with open(responses_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                responses.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return responses


def main():
    import os

    parser = argparse.ArgumentParser(
        description="Evaluate saved LLM responses for OpenMath benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to results directory containing responses.jsonl",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/problems_full.json",
        help="Path to problems JSON file",
    )
    parser.add_argument(
        "--baselines-file",
        type=str,
        default="data/baselines.json",
        help="Path to baselines JSON file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing evaluation results",
    )
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    results_dir = Path(args.results_dir)
    data_path = project_root / args.data_file
    baselines_path = project_root / args.baselines_file

    responses_path = results_dir / "responses.jsonl"
    evaluations_path = results_dir / "evaluation.jsonl"
    summary_path = results_dir / "summary.json"

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    if not responses_path.exists():
        print(f"Error: No responses.jsonl found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    if evaluations_path.exists() and not args.force:
        print(f"Error: evaluation.jsonl already exists in {results_dir}", file=sys.stderr)
        print("Use --force to overwrite existing evaluation results.")
        sys.exit(1)

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # Load problems and baselines
    all_problems = load_problems(data_path)
    baselines = load_baselines(baselines_path) if baselines_path.exists() else {}

    # Build mapping from problem id to (index, problem)
    problem_by_id = {p["id"]: (i, p) for i, p in enumerate(all_problems)}

    # Load config if available
    config = {}
    config_path = results_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Load responses
    responses = load_responses(responses_path)
    total = len(responses)

    print(f"Results directory: {results_dir}")
    print(f"Responses to evaluate: {total}")
    if config:
        print(f"Provider: {config.get('provider', 'unknown')}")
        print(f"Model: {config.get('model', 'unknown')}")
    print()

    # Auto-detect GitHub Actions for collapsible log groups
    is_github_actions = os.getenv("GITHUB_ACTIONS") == "true"

    # Load any pre-existing error entries from generation phase,
    # dropping stale errors for problems that were later successfully generated
    response_problem_ids = {r.get("problem_id") for r in responses}
    generation_errors = []
    if evaluations_path.exists():
        with open(evaluations_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        if entry.get("error_type") == "runtime":
                            if entry.get("problem_id") not in response_problem_ids:
                                generation_errors.append(entry)
                    except json.JSONDecodeError:
                        pass
        if generation_errors:
            print(f"Loaded {len(generation_errors)} generation error(s) from previous run")

    # Evaluate each response
    start_time = datetime.now()
    evaluations = list(generation_errors)

    # Write generation errors first, then evaluation results
    with open(evaluations_path, "w") as f:
        for entry in generation_errors:
            f.write(json.dumps(entry) + "\n")

    for idx, response_data in enumerate(responses):
        problem_id = response_data.get("problem_id")
        response_text = response_data.get("response", "")

        if problem_id not in problem_by_id:
            print(f"  Warning: Unknown problem ID: {problem_id}, skipping")
            continue

        problem_index, problem = problem_by_id[problem_id]

        # Evaluate
        eval_result = evaluate_response(
            problem, problem_index, response_text, baselines
        )
        evaluations.append(eval_result)

        # Save evaluation incrementally
        with open(evaluations_path, "a") as f:
            f.write(json.dumps(_sanitize_for_json(eval_result)) + "\n")

        # Print progress
        print_progress(idx, total, problem, eval_result)
        if is_github_actions:
            print_github_response_group(
                idx, total, problem, eval_result, response_text,
            )
        sys.stdout.flush()

    # Compute and save summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    summary = compute_summary(evaluations, config, duration)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print_summary(summary, results_dir)
    sys.exit(0)


if __name__ == "__main__":
    main()
