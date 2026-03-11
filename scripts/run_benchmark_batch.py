#!/usr/bin/env python3
"""
Generate LLM responses for OpenMath benchmark using the OpenAI Batch API.

The Batch API provides higher rate limits and 50% cost discount compared to
the standard API. Batches complete within 24 hours.

Workflow:
    1. Submit: creates JSONL input, uploads to OpenAI, starts batch
    2. Status: checks batch progress
    3. Download: retrieves results and converts to responses.jsonl format

Usage:
    # Submit a new batch run (all problems)
    uv run scripts/run_benchmark_batch.py submit

    # Submit for specific model
    uv run scripts/run_benchmark_batch.py submit --model gpt-5.4-pro

    # Submit a single problem
    uv run scripts/run_benchmark_batch.py submit --problem diff_basis_upper

    # Submit and wait for completion
    uv run scripts/run_benchmark_batch.py submit --wait

    # Check batch status
    uv run scripts/run_benchmark_batch.py status results/batch_gpt-5.4-pro_20260308_120000/

    # Download results when batch is complete
    uv run scripts/run_benchmark_batch.py download results/batch_gpt-5.4-pro_20260308_120000/

    # Then evaluate as usual:
    uv run scripts/evaluate_responses.py results/batch_gpt-5.4-pro_20260308_120000/
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")

import openai

# Add scripts directory to path for imports
_script_dir = Path(__file__).parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from evaluate import load_problems
from run_benchmark import SYSTEM_MESSAGES, DEFAULT_OPENAI_MODEL


def build_batch_input(problems: list[dict], model: str) -> list[dict]:
    """Build Batch API JSONL entries for each problem."""
    entries = []
    for problem in problems:
        eval_mode = problem.get("evaluation_mode", "ground_truth_computable")
        system_message = SYSTEM_MESSAGES.get(eval_mode, SYSTEM_MESSAGES["ground_truth_computable"])

        entry = {
            "custom_id": problem["id"],
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model,
                "instructions": system_message,
                "input": [{"role": "user", "content": problem["prompt"]}],
                "max_output_tokens": 125000,
                "reasoning": {"effort": "high", "summary": "detailed"},
            },
        }
        entries.append(entry)
    return entries


def filter_problems(problems: list[dict], args) -> list[dict]:
    """Filter problems by --problem or --range flags."""
    if args.problem:
        matching = [p for p in problems if p["id"] == args.problem]
        if not matching:
            matching = [p for p in problems if args.problem in p["id"]]
        if not matching:
            print(f"Error: No problem found matching '{args.problem}'", file=sys.stderr)
            sys.exit(1)
        return matching

    if args.range:
        parts = args.range.split("-")
        if len(parts) != 2:
            print(f"Error: --range must be in format 'start-end'", file=sys.stderr)
            sys.exit(1)
        range_start, range_end = int(parts[0]), int(parts[1])
        return problems[range_start:range_end + 1]

    return problems


def cmd_submit(args):
    """Submit a new batch to the OpenAI Batch API."""
    project_root = Path(__file__).parent.parent
    data_path = project_root / args.data_file
    results_base = project_root / "results"

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    all_problems = load_problems(data_path)
    problems = filter_problems(all_problems, args)

    model = args.model
    print(f"Model: {model}")
    print(f"Problems: {len(problems)}")

    # Build batch JSONL entries
    entries = build_batch_input(problems, model)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "-")
    folder_name = f"batch_{safe_model}_{timestamp}"
    output_dir = results_base / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write batch input file
    batch_input_path = output_dir / "batch_input.jsonl"
    with open(batch_input_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"Batch input: {batch_input_path} ({len(entries)} requests)")

    # Save prompts.jsonl for reference
    prompts_path = output_dir / "prompts.jsonl"
    with open(prompts_path, "w") as f:
        for problem in problems:
            eval_mode = problem.get("evaluation_mode", "ground_truth_computable")
            system_message = SYSTEM_MESSAGES.get(eval_mode, SYSTEM_MESSAGES["ground_truth_computable"])
            f.write(json.dumps({
                "problem_id": problem["id"],
                "system_message": system_message,
                "prompt": problem["prompt"],
            }) + "\n")

    # Upload file to OpenAI
    client = openai.OpenAI()
    print("Uploading batch input file...")
    with open(batch_input_path, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    print(f"Uploaded file: {uploaded_file.id}")

    # Create batch
    print("Creating batch...")
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    print(f"Batch created: {batch.id}")
    print(f"Status: {batch.status}")

    # Save config
    config = {
        "batch_id": batch.id,
        "input_file_id": uploaded_file.id,
        "model": model,
        "num_problems": len(problems),
        "timestamp": datetime.now().isoformat(),
        "problems_file": args.data_file,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nOutput directory: {output_dir}")
    print(f"\nCheck status:")
    print(f"  uv run scripts/run_benchmark_batch.py status {output_dir}")
    print(f"\nDownload results when complete:")
    print(f"  uv run scripts/run_benchmark_batch.py download {output_dir}")

    if args.wait:
        print(f"\n--wait specified, polling until completion...")
        _poll_until_done(client, batch.id, output_dir)


def cmd_status(args):
    """Check the status of a batch run."""
    output_dir = Path(args.run_dir)
    config_path = output_dir / "config.json"
    if not config_path.exists():
        print(f"Error: config.json not found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    client = openai.OpenAI()
    batch = client.batches.retrieve(config["batch_id"])

    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    if batch.request_counts:
        print(f"Completed: {batch.request_counts.completed}/{batch.request_counts.total}")
        print(f"Failed: {batch.request_counts.failed}")
    if batch.output_file_id:
        print(f"Output file: {batch.output_file_id}")
    if batch.error_file_id:
        print(f"Error file: {batch.error_file_id}")

    if batch.status == "completed":
        print(f"\nBatch is complete! Download results with:")
        print(f"  uv run scripts/run_benchmark_batch.py download {output_dir}")


def cmd_download(args):
    """Download batch results and convert to responses.jsonl format."""
    output_dir = Path(args.run_dir)
    config_path = output_dir / "config.json"
    if not config_path.exists():
        print(f"Error: config.json not found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    client = openai.OpenAI()
    batch = client.batches.retrieve(config["batch_id"])

    if batch.status != "completed":
        print(f"Batch status is '{batch.status}', not 'completed'.")
        if batch.status == "in_progress" and batch.request_counts:
            print(f"Progress: {batch.request_counts.completed}/{batch.request_counts.total}")
        sys.exit(1)

    if not batch.output_file_id:
        print("Error: No output file available", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading results from {batch.output_file_id}...")
    file_response = client.files.content(batch.output_file_id)
    raw_output = file_response.text

    # Save raw batch output
    raw_path = output_dir / "batch_output.jsonl"
    with open(raw_path, "w") as f:
        f.write(raw_output)
    print(f"Raw batch output saved to {raw_path}")

    # Download error file if present
    if batch.error_file_id:
        print(f"Downloading error file from {batch.error_file_id}...")
        error_response = client.files.content(batch.error_file_id)
        error_path = output_dir / "batch_errors.jsonl"
        with open(error_path, "w") as f:
            f.write(error_response.text)
        print(f"Error file saved to {error_path}")

    # Load problem data for index mapping
    project_root = Path(__file__).parent.parent
    data_path = project_root / config.get("problems_file", "data/problems_full.json")
    all_problems = load_problems(data_path)
    problem_id_to_index = {p["id"]: i for i, p in enumerate(all_problems)}

    # Convert to responses.jsonl format
    responses_path = output_dir / "responses.jsonl"
    num_success = 0
    num_error = 0

    with open(responses_path, "w") as f:
        for line in raw_output.strip().split("\n"):
            if not line.strip():
                continue
            result = json.loads(line)
            custom_id = result["custom_id"]
            problem_index = problem_id_to_index.get(custom_id, -1)

            if result.get("error"):
                num_error += 1
                print(f"  ERROR {custom_id}: {result['error']}")
                error_data = {
                    "problem_id": custom_id,
                    "problem_index": problem_index,
                    "title": custom_id,
                    "provider": "openai",
                    "model": config["model"],
                    "response": "",
                    "error": result["error"],
                    "timestamp": datetime.now().isoformat(),
                }
                f.write(json.dumps(error_data) + "\n")
                continue

            # Extract response content from the batch result
            resp_body = result["response"]["body"]
            output_text = ""
            reasoning_details = []
            usage = None

            for item in resp_body.get("output", []):
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            output_text += content.get("text", "")
                elif item.get("type") == "reasoning":
                    reasoning_details.append(item)

            if resp_body.get("usage"):
                u = resp_body["usage"]
                usage = {
                    "input_tokens": u.get("input_tokens", 0),
                    "output_tokens": u.get("output_tokens", 0),
                    "total_tokens": u.get("total_tokens", 0),
                }

            response_data = {
                "problem_id": custom_id,
                "problem_index": problem_index,
                "title": custom_id,
                "provider": "openai",
                "model": config["model"],
                "response": output_text,
                "timestamp": datetime.now().isoformat(),
            }
            if reasoning_details:
                response_data["reasoning_details"] = reasoning_details
            if usage:
                response_data["usage"] = usage

            f.write(json.dumps(response_data) + "\n")
            num_success += 1

    print(f"\nConverted {num_success} responses to {responses_path}")
    if num_error:
        print(f"  ({num_error} errors)")

    # Update config with completion info
    config["completed_at"] = datetime.now().isoformat()
    config["num_success"] = num_success
    config["num_error"] = num_error
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTo evaluate results, run:")
    print(f"  uv run scripts/evaluate_responses.py {output_dir}")


def _poll_until_done(client, batch_id: str, output_dir: Path):
    """Poll batch status until complete, then trigger download."""
    poll_interval = 60
    while True:
        batch = client.batches.retrieve(batch_id)
        now = datetime.now().strftime("%H:%M:%S")
        if batch.request_counts:
            print(f"[{now}] Status: {batch.status} — "
                  f"{batch.request_counts.completed}/{batch.request_counts.total} complete, "
                  f"{batch.request_counts.failed} failed")
        else:
            print(f"[{now}] Status: {batch.status}")

        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(poll_interval)

    if batch.status == "completed":
        print("\nBatch completed! Downloading results...")
        dl_args = argparse.Namespace(run_dir=str(output_dir))
        cmd_download(dl_args)
    else:
        print(f"\nBatch ended with status: {batch.status}")
        if batch.error_file_id:
            error_response = client.files.content(batch.error_file_id)
            print(f"Errors:\n{error_response.text[:2000]}")


def main():
    parser = argparse.ArgumentParser(
        description="OpenMath benchmark via OpenAI Batch API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Submit
    sub = subparsers.add_parser("submit", help="Submit a new batch")
    sub.add_argument("--model", default="gpt-5.4-pro", help="Model (default: gpt-5.4-pro)")
    sub.add_argument("--problem", type=str, help="Run only a single problem by ID")
    sub.add_argument("--range", type=str, help="Range of problem indices, e.g. '10-25'")
    sub.add_argument("--data-file", default="data/problems_full.json", help="Problems JSON file")
    sub.add_argument("--wait", action="store_true", help="Poll and auto-download when complete")

    # Status
    sub = subparsers.add_parser("status", help="Check batch status")
    sub.add_argument("run_dir", help="Path to the batch run directory")

    # Download
    sub = subparsers.add_parser("download", help="Download completed batch results")
    sub.add_argument("run_dir", help="Path to the batch run directory")

    args = parser.parse_args()

    if args.command == "submit":
        cmd_submit(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "download":
        cmd_download(args)


if __name__ == "__main__":
    main()
