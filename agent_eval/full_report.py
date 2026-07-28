"""Aggregate completed Modal agent batches into one auditable evaluation report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agent_eval.modal_runner import (
    LOCAL_RESULTS_ROOT,
    PROBLEMS_PATH,
    get_state_volume,
    read_volume_file,
    volume_file_exists,
)

TOKEN_FIELDS = (
    "total_tokens",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)
COMPLIANCE_ERRORS = {"compliance", "compliance_indeterminate"}


def read_json(volume, path: str) -> dict[str, Any]:
    return json.loads(read_volume_file(volume, path))


def read_jsonl(volume, path: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in read_volume_file(volume, path).decode().splitlines()
        if line.strip()
    ]


def passed_after_all_gates(evaluation: dict[str, Any]) -> bool:
    mode = evaluation.get("mode")
    if mode == "numeric":
        return bool(evaluation.get("success"))
    if mode == "construction":
        return bool(evaluation.get("valid"))
    if mode == "benchmark":
        if not evaluation.get("valid"):
            return False
        comparison = evaluation.get("baseline_comparison") or {}
        result = comparison.get("result", "no_baseline")
        return result in {"beats_baseline", "no_baseline"}
    return False


def passed_before_compliance(evaluation: dict[str, Any]) -> bool:
    mode = evaluation.get("mode")
    pass_key = "success" if mode == "numeric" else "valid"
    return bool(evaluation.get(pass_key)) or (
        evaluation.get("error_type") in COMPLIANCE_ERRORS
    )


def sum_usage(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = {field: 0 for field in TOKEN_FIELDS}
    for record in records:
        usage = ((record.get("usage") or {}).get("total") or {})
        for field in TOKEN_FIELDS:
            totals[field] += int(usage.get(field) or 0)
    return totals


def add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {field: left[field] + right[field] for field in TOKEN_FIELDS}


def aggregate(run_ids: list[str], require_count: int | None) -> tuple[dict, list[dict]]:
    volume = get_state_volume()
    problems = json.loads(PROBLEMS_PATH.read_text())
    problem_by_id = {problem["id"]: problem for problem in problems}

    combined: dict[str, dict[str, Any]] = {}
    generation_usage = {field: 0 for field in TOKEN_FIELDS}
    review_usage = {field: 0 for field in TOKEN_FIELDS}
    review_rounds_with_usage = 0
    run_summaries = []

    for run_id in run_ids:
        prefix = f"/runs/{run_id}"
        required = (
            "responses.jsonl",
            "compliance.jsonl",
            "evaluations.jsonl",
            "runner_status.json",
            "scoring_status.json",
        )
        missing = [
            name
            for name in required
            if not volume_file_exists(volume, f"{prefix}/{name}")
        ]
        if missing:
            raise FileNotFoundError(
                f"Run {run_id} is missing completed artifacts: {missing}"
            )

        responses = read_jsonl(volume, f"{prefix}/responses.jsonl")
        compliance = read_jsonl(volume, f"{prefix}/compliance.jsonl")
        evaluations = read_jsonl(volume, f"{prefix}/evaluations.jsonl")
        runner = read_json(volume, f"{prefix}/runner_status.json")
        scoring = read_json(volume, f"{prefix}/scoring_status.json")
        if scoring.get("status") != "completed":
            raise RuntimeError(
                f"Run {run_id} scoring is {scoring.get('status')!r}, not completed"
            )

        response_by_id = {item["problem_id"]: item for item in responses}
        compliance_by_id = {item["problem_id"]: item for item in compliance}
        evaluation_by_id = {item["problem_id"]: item for item in evaluations}
        problem_ids = set(response_by_id)
        if problem_ids != set(compliance_by_id) or problem_ids != set(evaluation_by_id):
            raise ValueError(f"Run {run_id} artifacts cover different problem sets")

        overlap = problem_ids & set(combined)
        if overlap:
            raise ValueError(
                f"Duplicate problems across authoritative runs: {sorted(overlap)}"
            )

        generation_usage = add_usage(
            generation_usage,
            sum_usage(responses),
        )
        review_rounds = [
            review_round
            for item in compliance
            for review_round in item.get("rounds", [])
        ]
        review_usage = add_usage(review_usage, sum_usage(review_rounds))
        review_rounds_with_usage += sum(
            bool(item.get("usage"))
            for item in review_rounds
        )

        for problem_id in problem_ids:
            evaluation = evaluation_by_id[problem_id]
            combined[problem_id] = {
                "problem_id": problem_id,
                "problem_index": evaluation["problem_index"],
                "evaluation_mode": problem_by_id[problem_id].get(
                    "evaluation_mode"
                ),
                "solvability": problem_by_id[problem_id].get("solvability"),
                "run_id": run_id,
                "generation_status": response_by_id[problem_id].get("status"),
                "permissibility": compliance_by_id[problem_id],
                "evaluation": evaluation,
                "raw_validator_pass": passed_before_compliance(evaluation),
                "final_pass": passed_after_all_gates(evaluation),
            }

        run_summaries.append(
            {
                "run_id": run_id,
                "problems": len(problem_ids),
                "generation_counts": runner.get("counts"),
                "permissibility_counts": dict(
                    Counter(item["status"] for item in compliance)
                ),
                "scoring": scoring,
            }
        )

    ordered = sorted(combined.values(), key=lambda item: item["problem_index"])
    if require_count is not None and len(ordered) != require_count:
        raise ValueError(
            f"Expected {require_count} unique problems, found {len(ordered)}"
        )
    indices = [item["problem_index"] for item in ordered]
    if require_count is not None and indices != list(range(require_count)):
        raise ValueError("Problem indices do not provide exact contiguous coverage")

    final_passes = [item for item in ordered if item["final_pass"]]
    raw_passes = [item for item in ordered if item["raw_validator_pass"]]
    indeterminate = [
        item
        for item in ordered
        if item["evaluation"].get("error_type") == "compliance_indeterminate"
    ]
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in ("numeric", "benchmark", "construction"):
        group = [
            item
            for item in ordered
            if item["evaluation"].get("mode") == mode
        ]
        passed = sum(item["final_pass"] for item in group)
        by_mode[mode] = {
            "total": len(group),
            "raw_validator_passed": sum(
                item["raw_validator_pass"] for item in group
            ),
            "passed": passed,
            "pass_rate": passed / len(group) if group else 0.0,
        }

    by_solvability: dict[str, dict[str, Any]] = {}
    levels = sorted(
        {item["solvability"] for item in ordered},
        key=lambda value: (value is None, str(value)),
    )
    for level in levels:
        group = [item for item in ordered if item["solvability"] == level]
        passed = sum(item["final_pass"] for item in group)
        by_solvability[str(level)] = {
            "total": len(group),
            "raw_validator_passed": sum(
                item["raw_validator_pass"] for item in group
            ),
            "passed": passed,
            "pass_rate": passed / len(group) if group else 0.0,
        }

    compliance_counts = Counter(
        item["permissibility"]["status"] for item in ordered
    )
    generation_counts = Counter(item["generation_status"] for item in ordered)
    error_counts = Counter()
    benchmark_results = Counter()
    for item in ordered:
        evaluation = item["evaluation"]
        if evaluation.get("mode") == "benchmark":
            comparison = evaluation.get("baseline_comparison") or {}
            if evaluation.get("valid") or evaluation.get("error_type") in COMPLIANCE_ERRORS:
                benchmark_results[comparison.get("result", "no_baseline")] += 1
        if item["final_pass"]:
            continue
        error_type = evaluation.get("error_type") or "validator_rejected"
        if (
            evaluation.get("mode") == "benchmark"
            and evaluation.get("valid")
        ):
            comparison = evaluation.get("baseline_comparison") or {}
            error_type = comparison.get("result", "below_baseline")
        error_counts[error_type] += 1

    total_usage = add_usage(generation_usage, review_usage)
    summary = {
        "schema_version": 1,
        "run_ids": run_ids,
        "coverage": {
            "total_problems": len(ordered),
            "first_index": min(indices) if indices else None,
            "last_index": max(indices) if indices else None,
            "generation_status": dict(generation_counts),
        },
        "results": {
            "raw_validator_passed": len(raw_passes),
            "passed_after_permissibility_and_baseline": len(final_passes),
            "failed": len(ordered) - len(final_passes) - len(indeterminate),
            "indeterminate": len(indeterminate),
            "pass_rate": len(final_passes) / len(ordered) if ordered else 0.0,
            "pass_problem_ids": [
                item["problem_id"] for item in final_passes
            ],
        },
        "permissibility": dict(compliance_counts),
        "by_mode": by_mode,
        "by_solvability": by_solvability,
        "benchmark_comparisons": dict(benchmark_results),
        "failure_reasons": dict(error_counts),
        "tokens": {
            "generation": generation_usage,
            "permissibility_review": review_usage,
            "combined": total_usage,
            "review_rounds_with_usage": review_rounds_with_usage,
            "expected_review_rounds": len(ordered) * 3,
        },
        "runs": run_summaries,
    }
    return summary, ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--require-count", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    summary, per_problem = aggregate(args.run_id, args.require_count)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else LOCAL_RESULTS_ROOT / "modal_codex_sol_xhigh_full"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    with (output_dir / "per_problem.jsonl").open("w") as handle:
        for item in per_problem:
            handle.write(json.dumps(item, sort_keys=True, default=str) + "\n")
    with (output_dir / "per_problem.csv").open("w", newline="") as handle:
        fieldnames = (
            "problem_index",
            "problem_id",
            "evaluation_mode",
            "mode",
            "solvability",
            "generation_status",
            "permissibility_status",
            "raw_validator_pass",
            "final_pass",
            "matching_digits",
            "benchmark_result",
            "error_type",
            "error_message",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in per_problem:
            evaluation = item["evaluation"]
            comparison = evaluation.get("baseline_comparison") or {}
            writer.writerow(
                {
                    "problem_index": item["problem_index"],
                    "problem_id": item["problem_id"],
                    "evaluation_mode": item["evaluation_mode"],
                    "mode": evaluation.get("mode"),
                    "solvability": item["solvability"],
                    "generation_status": item["generation_status"],
                    "permissibility_status": item["permissibility"]["status"],
                    "raw_validator_pass": item["raw_validator_pass"],
                    "final_pass": item["final_pass"],
                    "matching_digits": evaluation.get("matching_digits"),
                    "benchmark_result": comparison.get("result"),
                    "error_type": evaluation.get("error_type"),
                    "error_message": evaluation.get("error_message"),
                }
            )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"Full report: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
