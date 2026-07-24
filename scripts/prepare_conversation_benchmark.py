#!/usr/bin/env python3
"""Prepare exact prompts for OpenAI-hosted, repository-free Codex cloud tasks.

This script deliberately does not call a model API or a local agent. It creates a
machine-checkable manifest for the Codex conversation orchestrator described in
the repository's root AGENTS.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent_benchmark import build_conversation_agent_prompt
from benchmark_prompts import SYSTEM_MESSAGES, system_messages_sha256
from conversation_benchmark import (
    REQUIRED_EXECUTION_LOCATION,
    REQUIRED_RUNTIME_KIND,
    RUNTIME_CONTRACT_VERSION,
)
from run_agent_benchmark import load_problems, select_problems


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare verbatim HorizonMath prompts for independent OpenAI-hosted "
            "Codex cloud containers; no GitHub agent repository is used"
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--problem", help="One exact ID or unique substring")
    selection.add_argument(
        "--range",
        dest="index_range",
        help="Inclusive 0-based range START-END",
    )
    parser.add_argument(
        "--data-file",
        default="data/problems_full.json",
        help="Problem dataset path relative to the repository root",
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max", "ultra"],
        default="ultra",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory; defaults to a timestamped results folder",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / args.data_file
    problems = load_problems(data_path)
    selected = select_problems(
        problems,
        problem_query=args.problem,
        index_range=args.index_range,
    )

    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model).strip("-")
        output_dir = (
            project_root
            / "results"
            / f"codex-conversation_{safe_model}_{timestamp}"
        )
        output_dir.mkdir(parents=True)

    run_id = uuid.uuid4().hex[:12]
    prompt_records = []
    for problem_index, problem in selected:
        mode = problem.get("evaluation_mode", "ground_truth_computable")
        system_message = SYSTEM_MESSAGES.get(
            mode, SYSTEM_MESSAGES["ground_truth_computable"]
        )
        problem_prompt = problem["prompt"]
        agent_task_prompt = build_conversation_agent_prompt(
            problem_id=problem["id"],
            system_message=system_message,
            problem_prompt=problem_prompt,
        )
        prompt_records.append(
            {
                "problem_id": problem["id"],
                "problem_index": problem_index,
                "system_message": system_message,
                "system_message_sha256": hashlib.sha256(
                    system_message.encode("utf-8")
                ).hexdigest(),
                "prompt": problem_prompt,
                "prompt_sha256": hashlib.sha256(
                    problem_prompt.encode("utf-8")
                ).hexdigest(),
                "agent_task_prompt": agent_task_prompt,
                "agent_task_prompt_sha256": hashlib.sha256(
                    agent_task_prompt.encode("utf-8")
                ).hexdigest(),
            }
        )

    config = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "codex-conversation",
        "execution_surface": (
            "repository-free tasks in OpenAI-hosted Codex cloud containers"
        ),
        "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
        "required_execution_location": REQUIRED_EXECUTION_LOCATION,
        "required_runtime_kind": REQUIRED_RUNTIME_KIND,
        "local_agent_execution_forbidden": True,
        "runtime_evidence_file": "cloud_runtime.jsonl",
        "prepared_state": "not_evaluable_until_cloud_runtime_is_confirmed",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "problems_file": args.data_file,
        "output_dir": str(output_dir),
        "selected_problem_ids": [
            problem["id"] for _, problem in selected
        ],
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "system_messages_sha256": system_messages_sha256(),
        "prompt_delivery": (
            "canonical system message and problem statement copied verbatim"
        ),
        "repository_attachment": "none",
        "primary_agent_may_spawn_subagents": True,
        "require_compliance": True,
        "attempts_per_problem": 1,
    }
    write_jsonl(output_dir / "prompts.jsonl", prompt_records)
    config["prompts_jsonl_sha256"] = hashlib.sha256(
        (output_dir / "prompts.jsonl").read_bytes()
    ).hexdigest()
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "responses.jsonl").touch()
    (output_dir / "cloud_runtime.jsonl").touch()

    print(f"Prepared {len(prompt_records)} exact conversation prompt(s)")
    print(f"Output: {output_dir}")
    print(
        "Codex must now create one OpenAI-hosted, repository-free Codex cloud "
        "task per record."
    )
    print(
        "A projectless task with hostId='local' is LOCAL and must be rejected."
    )


if __name__ == "__main__":
    main()
