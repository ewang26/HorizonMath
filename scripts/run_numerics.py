"""
Run numerical solution scripts and update problems_full.json with computed values.

This script:
1. Loads problems from problems_full.json
2. For each ground_truth_computable problem, executes the corresponding script in numerics/
3. Updates problems_full.json with numeric_value fields

Script filenames match problem IDs directly (e.g., "w4_watson_integral" -> "w4_watson_integral.py").

By default, runs 4 scripts in parallel to utilize multiple CPU cores.
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def id_to_script_name(problem_id: str) -> str:
    """Convert problem ID to script filename (ID is already the script name)."""
    return problem_id


def run_numeric_script(args: tuple) -> tuple[int, str, str | None, str | None]:
    """
    Run a numeric script and return results.

    Args:
        args: Tuple of (problem_index, script_path, python_exe, timeout)

    Returns:
        Tuple of (problem_index, script_name, numeric_value or None, error or None)
    """
    problem_index, script_path, python_exe, timeout = args
    script_path = Path(script_path)

    try:
        result = subprocess.run(
            [python_exe, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return (problem_index, script_path.name, result.stdout.strip(), None)
        else:
            return (problem_index, script_path.name, None, f"Error: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        return (problem_index, script_path.name, None, f"Timeout after {timeout}s")
    except Exception as e:
        return (problem_index, script_path.name, None, f"Exception: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run numeric scripts and update problems_full.json")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes, just show what would be done")
    parser.add_argument("--problem", type=int, help="Run only problem at index N")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per script in seconds (default: 600)")
    parser.add_argument("--force", action="store_true", help="Re-run even if numeric_value already exists")
    parser.add_argument("--parallel", type=int, default=4, help="Number of scripts to run in parallel (default: 4)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    problems_file = project_root / "data" / "problems_full.json"
    numerics_dir = project_root / "numerics"

    # Use the venv python
    venv_python = project_root / ".venv" / "bin" / "python"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable

    # Load problems
    with open(problems_file) as f:
        problems = json.load(f)

    print(f"Loaded {len(problems)} problems from {problems_file}")
    print(f"Using Python: {python_exe}")
    print()

    # Filter to ground_truth_computable problems
    computable_indices = [
        i for i, p in enumerate(problems)
        if p.get("evaluation_mode") == "ground_truth_computable"
    ]
    print(f"Found {len(computable_indices)} ground_truth_computable problems")
    print()

    # If specific problem requested, filter to just that one
    if args.problem is not None:
        if args.problem not in computable_indices:
            print(f"Problem {args.problem} is not ground_truth_computable", file=sys.stderr)
            sys.exit(1)
        computable_indices = [args.problem]

    # Build list of tasks to run
    tasks_to_run = []
    updated_count = 0
    skipped_count = 0
    failed_count = 0

    for i in computable_indices:
        problem = problems[i]
        problem_id = problem.get("id", f"problem_{i}")
        title = problem_id

        # Skip if already has numeric_value (unless --force)
        if problem.get("numeric_value") and not args.force:
            print(f"[{i}] {title}")
            print(f"  Already has numeric_value, skipping (use --force to recompute)")
            print()
            skipped_count += 1
            continue

        # Generate script name from problem ID
        script_base = id_to_script_name(problem_id)
        script_name = f"{script_base}.py"
        script_path = numerics_dir / script_name

        if not script_path.exists():
            print(f"[{i}] {title}")
            print(f"  Script: {script_name}")
            print(f"  Script not found, skipping")
            print()
            failed_count += 1
            continue

        if args.dry_run:
            print(f"[{i}] {title}")
            print(f"  Script: {script_name}")
            print(f"  Would run: {script_path}")
            print()
            continue

        tasks_to_run.append((i, str(script_path), python_exe, args.timeout, title))

    # Run tasks in parallel
    if tasks_to_run and not args.dry_run:
        print(f"Running {len(tasks_to_run)} scripts with {args.parallel} workers...")
        print()

        # Prepare args for worker function (without title)
        worker_args = [(i, path, exe, timeout) for i, path, exe, timeout, _ in tasks_to_run]
        title_map = {i: title for i, _, _, _, title in tasks_to_run}

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(run_numeric_script, arg): arg[0] for arg in worker_args}

            for future in as_completed(futures):
                problem_index, script_name, numeric_value, error = future.result()
                title = title_map[problem_index]
                print(f"[{problem_index}] {title}")
                print(f"  Script: {script_name}")

                if numeric_value:
                    problems[problem_index]["numeric_value"] = numeric_value
                    problems[problem_index].pop("numeric_error", None)
                    updated_count += 1
                    display_value = numeric_value[:50] + "..." if len(numeric_value) > 50 else numeric_value
                    print(f"  Result: {display_value}")
                else:
                    problems[problem_index]["numeric_value"] = None
                    problems[problem_index]["numeric_error"] = error or "Script execution failed"
                    failed_count += 1
                    print(f"  Failed: {error}")

                print()

    # Save results
    if not args.dry_run and updated_count > 0:
        with open(problems_file, "w") as f:
            json.dump(problems, f, indent=2)
        print(f"Updated {problems_file}")

    # Summary
    print(f"\nSummary:")
    print(f"  Updated: {updated_count}")
    print(f"  Skipped (already computed): {skipped_count}")
    print(f"  Failed: {failed_count}")

    total_with_values = sum(1 for p in problems if p.get("numeric_value"))
    total_computable = sum(1 for p in problems if p.get("evaluation_mode") == "ground_truth_computable")
    print(f"  Total with numeric_value: {total_with_values}/{total_computable} ground_truth_computable problems")


if __name__ == "__main__":
    main()
