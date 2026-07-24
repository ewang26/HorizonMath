#!/usr/bin/env python3
"""Run HorizonMath problems as isolated Codex cloud coding-agent tasks.

This is Phase 1 of the existing benchmark pipeline. It writes responses.jsonl
compatible with scripts/evaluate_responses.py, which remains the trusted Phase 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from agent_benchmark import (
    CloudTaskError,
    CodexCloudClient,
    IsolationError,
    answer_path,
    build_agent_prompt,
    extract_new_file_from_diff,
    repository_identity,
    validate_isolated_repository,
)
from benchmark_prompts import SYSTEM_MESSAGES, system_messages_sha256
from create_agent_workspace import AGENTS_MD, GITIGNORE, README_MD


EXPECTED_AGENT_FILES = {
    ".gitignore": GITIGNORE,
    "AGENTS.md": AGENTS_MD,
    "README.md": README_MD,
    "answers/.gitkeep": "",
}
ALLOWED_AGENT_HISTORY_PATHS = set(EXPECTED_AGENT_FILES) | {"answers"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def run_git(workspace: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise IsolationError(
            f"Git preflight failed in {workspace}: "
            f"{(result.stderr or result.stdout).strip()[:500]}"
        )
    return result.stdout.strip()


def read_git_blob(workspace: Path, object_spec: str) -> bytes:
    """Read one git blob without normalizing whitespace or encoding."""

    result = subprocess.run(
        ["git", "cat-file", "blob", object_spec],
        cwd=workspace,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise IsolationError(
            f"Git preflight could not read {object_spec!r}: "
            f"{result.stderr.decode(errors='replace').strip()[:500]}"
        )
    return result.stdout


def preflight_agent_workspace(
    workspace: Path,
    *,
    declared_repo_url: str,
    branch: str,
    verify_remote: bool = True,
) -> str:
    """Verify local and remote independent history against the exact safe seed."""

    workspace = workspace.expanduser().resolve()
    if not (workspace / ".git").exists():
        raise IsolationError(
            f"Agent workspace is not a git repository: {workspace}. "
            "Create it with scripts/create_agent_workspace.py."
        )
    if run_git(workspace, "status", "--porcelain"):
        raise IsolationError("Agent workspace must have a clean working tree")

    origin = run_git(workspace, "remote", "get-url", "origin")
    safe_origin = validate_isolated_repository(origin)
    if repository_identity(safe_origin) != repository_identity(declared_repo_url):
        raise IsolationError(
            f"--agent-repo-url identifies {declared_repo_url!r}, but the local "
            f"agent project's origin is {safe_origin!r}"
        )

    run_git(workspace, "check-ref-format", "--branch", branch)
    local_ref = f"refs/heads/{branch}"
    local_commit = run_git(workspace, "rev-parse", "--verify", local_ref)
    if verify_remote:
        # Codex cloud clones the GitHub repository, not this local checkout. Fetch
        # every remote branch and tag so the preflight covers everything an agent
        # could inspect through git, then require the selected branch to be identical.
        run_git(
            workspace,
            "fetch",
            "--quiet",
            "--prune",
            "origin",
            "+refs/heads/*:refs/remotes/origin/*",
            "+refs/tags/*:refs/tags/*",
        )
        remote_ref = f"refs/remotes/origin/{branch}"
        remote_commit = run_git(workspace, "rev-parse", "--verify", remote_ref)
        if remote_commit != local_commit:
            raise IsolationError(
                f"Local branch {branch!r} is {local_commit[:12]}, but the cloud "
                f"repository branch is {remote_commit[:12]}. Fetch/review it and "
                "make the verifier-free commits identical before running."
            )

    history_rows = run_git(workspace, "rev-list", "--objects", "--all").splitlines()
    historical_paths = set()
    for row in history_rows:
        if " " not in row:
            continue
        historical_path = row.split(" ", 1)[1].strip()
        if historical_path:
            historical_paths.add(historical_path)
    unexpected = sorted(historical_paths - ALLOWED_AGENT_HISTORY_PATHS)
    if unexpected:
        preview = ", ".join(unexpected[:8])
        raise IsolationError(
            "Agent repository history contains paths outside the verifier-free "
            f"allowlist: {preview}"
        )
    reachable_objects = set(run_git(workspace, "rev-list", "--all").splitlines())
    if reachable_objects != {local_commit}:
        raise IsolationError(
            "Agent repository must contain exactly the single verifier-free seed "
            "commit across all local and remote refs"
        )

    tracked_files = set(
        run_git(workspace, "ls-tree", "-r", "--name-only", local_commit).splitlines()
    )
    if tracked_files != set(EXPECTED_AGENT_FILES):
        raise IsolationError(
            "Agent repository HEAD does not match the exact verifier-free file set"
        )
    for path, expected_content in EXPECTED_AGENT_FILES.items():
        actual_content = read_git_blob(workspace, f"{local_commit}:{path}")
        if actual_content != expected_content.encode():
            raise IsolationError(
                f"Agent repository seed file {path!r} differs from the trusted template"
            )
    return safe_origin


def load_problems(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError("Problems file must contain a JSON list")
    return value


def select_problems(
    problems: list[dict],
    *,
    problem_query: str | None,
    index_range: str | None,
) -> list[tuple[int, dict]]:
    selected = list(enumerate(problems))
    if problem_query:
        exact = [item for item in selected if item[1]["id"] == problem_query]
        matches = exact or [
            item for item in selected if problem_query in item[1]["id"]
        ]
        if not matches:
            raise ValueError(f"No problem found matching {problem_query!r}")
        if not exact and len(matches) > 1:
            preview = ", ".join(problem["id"] for _, problem in matches[:8])
            raise ValueError(
                f"--problem substring {problem_query!r} is not unique; "
                f"matches: {preview}"
            )
        selected = matches
    if index_range:
        match = re.fullmatch(r"(\d+)-(\d+)", index_range)
        if not match:
            raise ValueError("--range must use inclusive 0-based form START-END")
        start, end = map(int, match.groups())
        if start > end or end >= len(selected):
            raise ValueError(
                f"--range {index_range} is outside 0-{len(selected) - 1}"
            )
        selected = selected[start : end + 1]
    return selected


def latest_task_events(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for event in load_jsonl(path):
        problem_id = event.get("problem_id")
        if problem_id:
            latest[problem_id] = event
    return latest


def create_results_dir(base: Path, model_label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_label).strip("-")
    output = base / f"codex-cloud_{safe_label or 'default'}_{timestamp}"
    output.mkdir(parents=True)
    return output


def generation_error(problem_index: int, problem: dict, message: str) -> dict:
    return {
        "problem_id": problem["id"],
        "problem_index": problem_index,
        "problem_title": problem["id"],
        "mode": problem.get("evaluation_mode", "unknown"),
        "error_type": "runtime",
        "error_message": message,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Codex cloud coding agents on HorizonMath without exposing "
            "trusted evaluators"
        )
    )
    parser.add_argument("--env", required=True, help="Codex cloud environment ID or unique label")
    parser.add_argument("--branch", default="main", help="Branch in the isolated agent repository")
    parser.add_argument(
        "--agent-workspace",
        type=Path,
        required=True,
        help="Local checkout of the separate verifier-free cloud repository",
    )
    parser.add_argument(
        "--agent-repo-url",
        required=True,
        help="GitHub URL of the separate verifier-free repository (recorded in config)",
    )
    parser.add_argument(
        "--confirm-agent-internet-off",
        action="store_true",
        help="Confirm that agent-phase internet access is Off in this cloud environment",
    )
    parser.add_argument(
        "--confirm-environment-isolated",
        action="store_true",
        help=(
            "Confirm that --env is bound to the verifier-free repository and "
            "its setup, maintenance, and cached container contain no benchmark "
            "evaluators or private artifacts"
        ),
    )
    parser.add_argument(
        "--confirm-goal-tools-available",
        action="store_true",
        help=(
            "Confirm that the cloud runtime exposes Codex goal lifecycle tools "
            "to the agent (the cloud CLI cannot verify or audit their use)"
        ),
    )
    parser.add_argument(
        "--confirm-live-canary",
        action="store_true",
        help=(
            "Confirm that one problem has completed end-to-end in this exact "
            "cloud environment before launching a multi-problem run"
        ),
    )
    parser.add_argument("--problem", help="Run one problem by ID or unique substring")
    parser.add_argument("--range", dest="index_range", help="Inclusive 0-based range START-END")
    parser.add_argument(
        "--data-file",
        default="data/problems_full.json",
        help="Trusted problem dataset path relative to HorizonMath",
    )
    parser.add_argument("--parallel", type=int, default=5, help="Maximum active cloud tasks")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Seconds between status checks",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=6 * 60 * 60,
        help="Maximum seconds to wait for one cloud task",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=120.0,
        help="Maximum seconds for each Codex CLI command",
    )
    parser.add_argument("--max-status-errors", type=int, default=5)
    parser.add_argument(
        "--model-label",
        default="codex-cloud-default",
        help=(
            "Results metadata label only; model selection is controlled by the "
            "Codex cloud environment/account"
        ),
    )
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--resume", type=Path, help="Resume an existing results directory")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Submit new tasks for problems whose prior cloud task failed",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save exact cloud prompts without submitting tasks",
    )
    args = parser.parse_args()

    if args.parallel < 1:
        parser.error("--parallel must be at least 1")
    if args.poll_interval <= 0 or args.task_timeout <= 0:
        parser.error("--poll-interval and --task-timeout must be positive")
    if not args.debug and not args.confirm_agent_internet_off:
        parser.error(
            "--confirm-agent-internet-off is required: verifier isolation cannot "
            "be guaranteed when the agent can fetch the public HorizonMath repository"
        )
    if not args.debug and not args.confirm_environment_isolated:
        parser.error(
            "--confirm-environment-isolated is required: the Codex cloud CLI "
            "does not expose repository binding or cached setup state for verification"
        )
    if not args.debug and not args.confirm_goal_tools_available:
        parser.error(
            "--confirm-goal-tools-available is required: goal creation is an "
            "explicit benchmark requirement but is not exposed in cloud CLI telemetry"
        )

    project_root = Path(__file__).resolve().parent.parent
    try:
        declared_repo = validate_isolated_repository(args.agent_repo_url)
        origin = preflight_agent_workspace(
            args.agent_workspace,
            declared_repo_url=declared_repo,
            branch=args.branch,
            verify_remote=not args.debug,
        )
        all_problems = load_problems(project_root / args.data_file)
        selected = select_problems(
            all_problems,
            problem_query=args.problem,
            index_range=args.index_range,
        )
    except (IsolationError, ValueError, OSError) as exc:
        parser.error(str(exc))
    if not args.debug and len(selected) > 1 and not args.confirm_live_canary:
        parser.error(
            "--confirm-live-canary is required for multi-problem runs: first "
            "complete and evaluate one problem in this exact cloud environment"
        )

    if args.resume:
        output_dir = args.resume.expanduser().resolve()
        if not output_dir.is_dir():
            parser.error(f"Resume directory not found: {output_dir}")
        with (output_dir / "config.json").open(encoding="utf-8") as handle:
            old_config = json.load(handle)
        dataset_sha256 = hashlib.sha256(
            (project_root / args.data_file).read_bytes()
        ).hexdigest()
        selected_problem_ids = [problem["id"] for _, problem in selected]
        expected = {
            "environment": args.env,
            "branch": args.branch,
            "model": args.model_label,
            "problems_file": args.data_file,
            "dataset_sha256": dataset_sha256,
            "system_messages_sha256": system_messages_sha256(),
            "selected_problem_ids": selected_problem_ids,
        }
        for key, value in expected.items():
            if old_config.get(key) != value:
                parser.error(
                    f"Resume configuration mismatch for {key}: "
                    f"{old_config.get(key)!r} != {value!r}"
                )
        old_repo_identity = old_config.get("agent_repo_identity")
        if not old_repo_identity and old_config.get("agent_repo_url"):
            old_repo_identity = repository_identity(old_config["agent_repo_url"])
        current_repo_identity = repository_identity(origin)
        if old_repo_identity != current_repo_identity:
            parser.error(
                "Resume configuration mismatch for agent repository identity: "
                f"{old_repo_identity!r} != {current_repo_identity!r}"
            )
        run_id = old_config["run_id"]
    else:
        output_dir = create_results_dir(project_root / "results", args.model_label)
        run_id = uuid.uuid4().hex[:12]
        dataset_sha256 = hashlib.sha256(
            (project_root / args.data_file).read_bytes()
        ).hexdigest()
        config = {
            "run_id": run_id,
            "timestamp": utc_now(),
            "provider": "codex-cloud",
            "model": args.model_label,
            "problems_file": args.data_file,
            "output_dir": str(output_dir),
            "environment": args.env,
            "branch": args.branch,
            "agent_repo_url": origin,
            "agent_repo_identity": repository_identity(origin),
            "agent_workspace": str(args.agent_workspace.expanduser().resolve()),
            "environment_isolation": (
                "repository binding and setup/cache contents operator confirmed"
            ),
            "agent_internet_access": (
                "off (operator confirmed)"
                if args.confirm_agent_internet_off
                else "not checked (debug-only prompt generation)"
            ),
            "goal_setting": (
                "required agent tool call (availability operator confirmed; "
                "lifecycle not exposed by cloud CLI)"
                if args.confirm_goal_tools_available
                else "requested in prompt only (debug prompt generation)"
            ),
            "live_canary": (
                "operator confirmed"
                if args.confirm_live_canary
                else "this run contains one problem"
            ),
            "attempts_per_problem": 1,
            "dataset_sha256": dataset_sha256,
            "system_messages_sha256": system_messages_sha256(),
            "selected_problem_ids": [problem["id"] for _, problem in selected],
            "verifier_isolation": (
                "separate allowlisted git history; no agent-phase internet; "
                "trusted evaluation after cloud execution"
            ),
        }
        (output_dir / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )

    prompts_path = output_dir / "prompts.jsonl"
    tasks_path = output_dir / "cloud_tasks.jsonl"
    responses_path = output_dir / "responses.jsonl"
    generation_errors_path = output_dir / "generation_errors.jsonl"
    responses_path.touch(exist_ok=True)

    completed = {
        row.get("problem_id")
        for row in load_jsonl(responses_path)
        if row.get("problem_id")
    }
    existing_prompts = {
        row.get("problem_id")
        for row in load_jsonl(prompts_path)
        if row.get("problem_id")
    }
    task_events = latest_task_events(tasks_path)
    problem_by_id = {problem["id"]: (index, problem) for index, problem in selected}

    queue: deque[tuple[int, dict, str, str]] = deque()
    active: dict[str, dict] = {}
    for index, problem in selected:
        problem_id = problem["id"]
        if problem_id in completed:
            continue
        previous = task_events.get(problem_id)
        if previous and previous.get("event") == "submitted":
            active[problem_id] = previous
            continue
        if previous and previous.get("event") == "error" and not args.retry_errors:
            continue
        mode = problem.get("evaluation_mode", "ground_truth_computable")
        system_message = SYSTEM_MESSAGES.get(
            mode, SYSTEM_MESSAGES["ground_truth_computable"]
        )
        output_path = answer_path(run_id, problem_id)
        cloud_prompt = build_agent_prompt(
            problem_id=problem_id,
            system_message=system_message,
            problem_prompt=problem["prompt"],
            output_path=output_path,
        )
        queue.append((index, problem, output_path, cloud_prompt))
        if problem_id not in existing_prompts:
            append_jsonl(
                prompts_path,
                {
                    "problem_id": problem_id,
                    "system_message": system_message,
                    "prompt": problem["prompt"],
                    "agent_task_prompt": cloud_prompt,
                    "answer_path": output_path,
                },
            )
            existing_prompts.add(problem_id)

    print(f"Codex cloud environment: {args.env}")
    print(f"Isolated repository: {origin}")
    print(f"Problems selected: {len(selected)}")
    print(f"Already completed: {len(completed & problem_by_id.keys())}")
    print(f"Output: {output_dir}")

    if args.debug:
        print(f"Debug complete: exact prompts saved to {prompts_path}")
        return

    client = CodexCloudClient(
        codex_binary=args.codex_binary,
        cwd=args.agent_workspace.expanduser().resolve(),
        command_timeout=args.command_timeout,
    )
    try:
        client.preflight()
    except CloudTaskError as exc:
        parser.error(str(exc))

    status_errors: dict[str, int] = {}
    diff_errors: dict[str, int] = {}
    try:
        while queue or active:
            while queue and len(active) < args.parallel:
                index, problem, output_path, cloud_prompt = queue.popleft()
                problem_id = problem["id"]
                try:
                    submission = client.submit(
                        prompt=cloud_prompt,
                        environment=args.env,
                        branch=args.branch,
                    )
                except CloudTaskError as exc:
                    message = f"Cloud task submission failed: {exc}"
                    append_jsonl(
                        tasks_path,
                        {
                            "event": "error",
                            "problem_id": problem_id,
                            "timestamp": utc_now(),
                            "error": message,
                        },
                    )
                    append_jsonl(
                        generation_errors_path,
                        generation_error(index, problem, message),
                    )
                    print(f"ERROR {problem_id}: {message}")
                    continue

                event = {
                    "event": "submitted",
                    "problem_id": problem_id,
                    "problem_index": index,
                    "task_id": submission.task_id,
                    "task_url": submission.url,
                    "submitted_at": utc_now(),
                    "answer_path": output_path,
                }
                append_jsonl(tasks_path, event)
                active[problem_id] = event
                print(f"SUBMITTED {problem_id}: {submission.url}")

            finished: list[str] = []
            for problem_id, task in list(active.items()):
                index, problem = problem_by_id[problem_id]
                try:
                    state = client.status(task["task_id"])
                    status_errors[problem_id] = 0
                except CloudTaskError as exc:
                    count = status_errors.get(problem_id, 0) + 1
                    status_errors[problem_id] = count
                    print(
                        f"STATUS RETRY {problem_id} ({count}/{args.max_status_errors}): {exc}"
                    )
                    if count < args.max_status_errors:
                        continue
                    state = "ERROR"
                    task = {**task, "status_error": str(exc)}

                submitted_at = datetime.fromisoformat(task["submitted_at"])
                elapsed = (datetime.now(timezone.utc) - submitted_at).total_seconds()
                if state == "PENDING" and elapsed <= args.task_timeout:
                    continue

                if state in {"READY", "APPLIED"}:
                    try:
                        diff = client.diff(task["task_id"])
                    except CloudTaskError as exc:
                        count = diff_errors.get(problem_id, 0) + 1
                        diff_errors[problem_id] = count
                        print(
                            f"DIFF RETRY {problem_id} "
                            f"({count}/{args.max_status_errors}): {exc}"
                        )
                        if count < args.max_status_errors:
                            continue
                        state = "ERROR"
                        task = {
                            **task,
                            "result_error": (
                                "Cloud task completed, but its diff could not be "
                                f"retrieved after {count} attempts: {exc}"
                            ),
                        }
                    else:
                        diff_errors[problem_id] = 0
                        try:
                            response = extract_new_file_from_diff(
                                diff, task["answer_path"]
                            )
                        except CloudTaskError as exc:
                            # A retrieved diff that violates the answer contract is
                            # deterministic and should fail immediately.
                            state = "ERROR"
                            task = {
                                **task,
                                "result_error": f"Rejected cloud diff: {exc}",
                            }

                    if state in {"READY", "APPLIED"}:
                        response_record = {
                            "problem_id": problem_id,
                            "problem_index": index,
                            "title": problem_id,
                            "provider": "codex-cloud",
                            "model": args.model_label,
                            "response": response,
                            "timestamp": utc_now(),
                            "cloud_task_id": task["task_id"],
                            "cloud_task_url": task["task_url"],
                            "cloud_environment": args.env,
                        }
                        append_jsonl(responses_path, response_record)
                        append_jsonl(
                            tasks_path,
                            {
                                **task,
                                "event": "completed",
                                "completed_at": utc_now(),
                                "cloud_state": state,
                            },
                        )
                        completed.add(problem_id)
                        finished.append(problem_id)
                        print(f"COMPLETED {problem_id}")
                        continue

                if state == "PENDING":
                    reason = (
                        f"Cloud task timed out after {elapsed:.0f}s and may still "
                        f"be running: {task['task_url']}"
                    )
                else:
                    reason = (
                        task.get("result_error")
                        or task.get("status_error")
                        or "Codex cloud task entered ERROR state"
                    )
                append_jsonl(
                    tasks_path,
                    {
                        **task,
                        "event": "error",
                        "failed_at": utc_now(),
                        "cloud_state": state,
                        "error": reason,
                    },
                )
                append_jsonl(
                    generation_errors_path,
                    generation_error(index, problem, reason),
                )
                finished.append(problem_id)
                print(f"ERROR {problem_id}: {reason}")

            for problem_id in finished:
                active.pop(problem_id, None)

            if queue or active:
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Resume with --resume {output_dir}", file=sys.stderr)
        raise SystemExit(130)

    generated = len(completed & problem_by_id.keys())
    print(f"\nResponses generated: {generated}/{len(selected)}")
    print("Trusted evaluation command:")
    print(f"  uv run scripts/evaluate_responses.py {output_dir}")


if __name__ == "__main__":
    main()
