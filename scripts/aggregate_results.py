#!/usr/bin/env python3
"""
Aggregate evaluation results across multiple results directories.

Merges evaluation.jsonl files from split runs into a single combined report,
deduplicating by problem_id (latest directory wins by lexicographic sort).

Usage:
    # Aggregate all GPT 5.2 Pro results
    uv run scripts/aggregate_results.py results/openai_gpt-5.2-pro_*/

    # Aggregate specific directories
    uv run scripts/aggregate_results.py results/dir1/ results/dir2/

    # Save combined output to a file
    uv run scripts/aggregate_results.py results/openai_gpt-5.2-pro_*/ -o results/gpt-5.2-pro_combined/
"""

import argparse
import json
import sys
from pathlib import Path


def load_evaluations(results_dirs: list[Path]) -> dict[str, tuple[dict, str]]:
    """Load evaluations from multiple dirs, deduplicating by problem_id.

    Dirs are processed in sorted order so later (newer) dirs overwrite earlier ones.

    Returns:
        Dict mapping problem_id -> (eval_dict, source_dir_name)
    """
    evals = {}
    for d in sorted(results_dirs):
        evl_path = d / "evaluation.jsonl"
        if not evl_path.exists():
            print(f"  Warning: no evaluation.jsonl in {d}, skipping", file=sys.stderr)
            continue
        count = 0
        for line in open(evl_path):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            pid = entry.get("problem_id")
            if pid:
                evals[pid] = (entry, d.name)
                count += 1
        print(f"  Loaded {count} evaluations from {d.name}", file=sys.stderr)
    return evals


def compute_summary(evaluations: list[dict]) -> dict:
    """Compute summary statistics from evaluation results."""
    total = len(evaluations)

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
    all_numeric_digits = []
    perfect_matches = 0

    for e in evaluations:
        mode = e.get("mode", "unknown")

        if mode == "numeric":
            by_mode["ground_truth_computable"]["total"] += 1
            digits = e.get("matching_digits")
            if digits is not None:
                all_numeric_digits.append(digits)
                if digits >= 10:
                    perfect_matches += 1
            if e.get("success"):
                passed += 1
                by_mode["ground_truth_computable"]["passed"] += 1
            else:
                et = e.get("error_type", "unknown")
                by_error_type[et] = by_error_type.get(et, 0) + 1

        elif mode == "construction":
            by_mode["new_construction"]["total"] += 1
            if e.get("valid"):
                passed += 1
                by_mode["new_construction"]["passed"] += 1
            else:
                et = e.get("error_type", "unknown")
                by_error_type[et] = by_error_type.get(et, 0) + 1

        elif mode == "benchmark":
            by_mode["benchmark_best_known"]["total"] += 1
            if e.get("valid"):
                bc = e.get("baseline_comparison", {})
                bc_r = bc.get("result", "no_baseline") if bc else "no_baseline"
                if bc_r in ("beats_baseline", "no_baseline"):
                    passed += 1
                    by_mode["benchmark_best_known"]["passed"] += 1
                if bc_r in benchmark_results:
                    benchmark_results[bc_r] += 1
            else:
                et = e.get("error_type", "unknown")
                by_error_type[et] = by_error_type.get(et, 0) + 1

        else:
            et = e.get("error_type", "unknown")
            by_error_type[et] = by_error_type.get(et, 0) + 1

    for mode_key in by_mode:
        mt = by_mode[mode_key]["total"]
        mp = by_mode[mode_key]["passed"]
        by_mode[mode_key]["pass_rate"] = f"{(mp / mt * 100):.1f}%" if mt > 0 else "0.0%"

    avg_digits = sum(all_numeric_digits) / len(all_numeric_digits) if all_numeric_digits else 0.0

    return {
        "total_problems": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": f"{(passed / total * 100):.1f}%" if total > 0 else "0.0%",
        "by_evaluation_mode": by_mode,
        "by_error_type": by_error_type,
        "benchmark_results": benchmark_results,
        "numeric_results": {
            "avg_matching_digits": round(avg_digits, 1),
            "perfect_matches_10plus": perfect_matches,
            "total_scored": len(all_numeric_digits),
        },
    }


def format_problem_line(e: dict) -> str:
    """Format a single problem result as a one-line string."""
    pid = e.get("problem_id", "?")
    mode = e.get("mode", "unknown")

    if mode == "numeric":
        digits = e.get("matching_digits")
        if e.get("success"):
            return f"  PASS  {pid} ({digits} digits)"
        else:
            err = e.get("error_type", "")
            d_str = f", {digits} digits" if digits is not None else ""
            return f"  FAIL  {pid} ({err}{d_str})"
    elif mode == "construction":
        if e.get("valid"):
            return f"  PASS  {pid}"
        else:
            err = e.get("error_type", "") or "invalid"
            return f"  FAIL  {pid} ({err})"
    elif mode == "benchmark":
        bc = e.get("baseline_comparison", {})
        bc_r = bc.get("result", "?") if bc else "?"
        imp = bc.get("improvement_percent") if bc else None
        imp_str = f", {'+' if imp and imp > 0 else ''}{imp}%" if imp is not None else ""
        if e.get("valid") and bc_r in ("beats_baseline", "no_baseline"):
            return f"  PASS  {pid} ({bc_r}{imp_str})"
        elif e.get("valid"):
            return f"  ~     {pid} (valid, {bc_r}{imp_str})"
        else:
            err = e.get("error_type", "") or "invalid"
            return f"  FAIL  {pid} ({err})"
    else:
        err = e.get("error_type", "unknown")
        return f"  FAIL  {pid} ({err})"


def print_report(evaluations: list[dict], summary: dict, source_dirs: list[Path]) -> None:
    """Print the full aggregated report."""
    total = summary["total_problems"]
    passed = summary["passed"]
    failed = summary["failed"]
    rate = summary["pass_rate"]

    print("=" * 60)
    print("           AGGREGATED EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Source directories: {len(source_dirs)}")
    print(f"Total: {total} | Passed: {passed} ({rate}) | Failed: {failed}")

    print("\nBy Mode:")
    for mode_name, mode_stats in summary["by_evaluation_mode"].items():
        mt = mode_stats["total"]
        mp = mode_stats["passed"]
        mr = mode_stats["pass_rate"]
        if mt > 0:
            print(f"  {mode_name}: {mp}/{mt} ({mr})")

    numeric = summary["numeric_results"]
    if numeric["total_scored"] > 0:
        print(f"\nNumeric Problems:")
        print(f"  Avg matching digits (all): {numeric['avg_matching_digits']}")
        print(f"  Perfect (10+ digits): {numeric['perfect_matches_10plus']}/{numeric['total_scored']}")

    bench = summary["benchmark_results"]
    if any(v > 0 for v in bench.values()):
        print(f"\nBenchmark Problems:")
        print(f"  Beats baseline:   {bench['beats_baseline']}")
        print(f"  Matches baseline: {bench['matches_baseline']}")
        print(f"  Below baseline:   {bench['below_baseline']}")
        print(f"  No baseline:      {bench['no_baseline']}")

    errors = summary["by_error_type"]
    if errors:
        print(f"\nErrors by Type:")
        for k, v in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("PER-PROBLEM RESULTS")
    print("=" * 60)
    for e in sorted(evaluations, key=lambda x: x.get("problem_id", "")):
        print(format_problem_line(e))
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results across multiple results directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_dirs",
        nargs="+",
        type=str,
        help="Paths to results directories containing evaluation.jsonl",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Save combined evaluation.jsonl and summary.json to this directory",
    )
    args = parser.parse_args()

    dirs = [Path(d) for d in args.results_dirs]

    # Validate
    for d in dirs:
        if not d.is_dir():
            print(f"Error: {d} is not a directory", file=sys.stderr)
            sys.exit(1)

    print(f"Aggregating {len(dirs)} directories...", file=sys.stderr)
    evals_by_id = load_evaluations(dirs)
    evaluations = [entry for entry, _source in evals_by_id.values()]

    if not evaluations:
        print("Error: No evaluations found", file=sys.stderr)
        sys.exit(1)

    summary = compute_summary(evaluations)
    print_report(evaluations, summary, dirs)

    # Save if output dir specified
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        eval_path = out / "evaluation.jsonl"
        with open(eval_path, "w") as f:
            for e in sorted(evaluations, key=lambda x: x.get("problem_id", "")):
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        summary_path = out / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")

        # Also save provenance: which dir each problem came from
        provenance_path = out / "provenance.json"
        provenance = {pid: source for pid, (_entry, source) in sorted(evals_by_id.items())}
        with open(provenance_path, "w") as f:
            json.dump(provenance, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print(f"\nSaved to {out}/:", file=sys.stderr)
        print(f"  {eval_path.name} ({len(evaluations)} problems)", file=sys.stderr)
        print(f"  {summary_path.name}", file=sys.stderr)
        print(f"  {provenance_path.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
