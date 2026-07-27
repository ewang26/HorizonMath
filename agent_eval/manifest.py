"""Build and validate verifier-free run manifests."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_eval.config import (
    DEFAULT_CONCURRENCY,
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    PERMISSIBILITY_EFFORT,
    PERMISSIBILITY_MODEL,
    PERMISSIBILITY_ROUNDS,
    PROBLEM_TIMEOUT_SECONDS,
)

SAFE_PROBLEM_KEYS = {
    "problem_id",
    "problem_index",
    "prompt",
    "evaluation_mode",
    "developer_instructions",
    "prompt_sha256",
}

FORBIDDEN_SOURCE_KEYS = {
    "numeric_value",
    "test_points",
    "expected",
    "expected_value",
    "baseline",
    "baselines",
    "validator",
    "validator_path",
    "source_note",
    "source_url",
}
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def validate_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Unsafe {label}: {value!r}")
    return value


def load_problems(path: Path) -> list[dict[str, Any]]:
    problems = json.loads(path.read_text())
    if not isinstance(problems, list):
        raise ValueError("Problems file must contain a JSON list")
    return problems


def select_indices(
    problem_count: int,
    *,
    start: int = 0,
    count: int | None = None,
    indices: Iterable[int] | None = None,
) -> list[int]:
    if indices is not None:
        selected = list(indices)
    else:
        stop = problem_count if count is None else start + count
        selected = list(range(start, min(stop, problem_count)))

    if not selected:
        raise ValueError("No problems selected")
    if len(set(selected)) != len(selected):
        raise ValueError("Problem indices must be unique")
    if min(selected) < 0 or max(selected) >= problem_count:
        raise ValueError(f"Problem index out of range 0..{problem_count - 1}")
    return selected


def build_manifest(
    problems: list[dict[str, Any]],
    indices: Iterable[int],
    *,
    run_id: str,
    developer_instructions_by_mode: dict[str, str],
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout_seconds: int = PROBLEM_TIMEOUT_SECONDS,
    permissibility_rubric: str,
) -> dict[str, Any]:
    if timeout_seconds != PROBLEM_TIMEOUT_SECONDS:
        raise ValueError(
            f"Agent problem timeout must be exactly {PROBLEM_TIMEOUT_SECONDS} seconds"
        )
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if not isinstance(permissibility_rubric, str) or not permissibility_rubric:
        raise ValueError("Permissibility rubric must be a non-empty string")

    safe_problems: list[dict[str, Any]] = []
    for index in indices:
        source = problems[index]
        problem_id = validate_identifier(source.get("id"), "problem id")
        prompt = source.get("prompt")
        evaluation_mode = source.get("evaluation_mode", "ground_truth_computable")
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError(f"Problem {index} has no valid id")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"Problem {problem_id} has no valid prompt")
        instructions = developer_instructions_by_mode.get(
            evaluation_mode,
            developer_instructions_by_mode["ground_truth_computable"],
        )
        safe_problems.append(
            {
                "problem_id": problem_id,
                "problem_index": index,
                "prompt": prompt,
                "evaluation_mode": evaluation_mode,
                "developer_instructions": instructions,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "provider": "codex-chatgpt-subscription",
        "model": model,
        "reasoning_effort": effort,
        "concurrency": concurrency,
        "problem_timeout_seconds": timeout_seconds,
        "permissibility": {
            "model": PERMISSIBILITY_MODEL,
            "reasoning_effort": PERMISSIBILITY_EFFORT,
            "rounds": PERMISSIBILITY_ROUNDS,
            "rubric": permissibility_rubric,
        },
        "problems": safe_problems,
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    if manifest.get("model") != DEFAULT_MODEL:
        raise ValueError(f"Model must be pinned to {DEFAULT_MODEL}")
    if manifest.get("reasoning_effort") != DEFAULT_EFFORT:
        raise ValueError(f"Reasoning effort must be pinned to {DEFAULT_EFFORT}")
    if manifest.get("problem_timeout_seconds") != PROBLEM_TIMEOUT_SECONDS:
        raise ValueError("Problem timeout must be exactly three hours")
    validate_identifier(manifest.get("run_id"), "run id")
    permissibility = manifest.get("permissibility")
    if not isinstance(permissibility, dict):
        raise ValueError("Manifest has no permissibility configuration")
    if permissibility.get("model") != PERMISSIBILITY_MODEL:
        raise ValueError("Unexpected permissibility model")
    if permissibility.get("reasoning_effort") != PERMISSIBILITY_EFFORT:
        raise ValueError("Unexpected permissibility reasoning effort")
    if permissibility.get("rounds") != PERMISSIBILITY_ROUNDS:
        raise ValueError("Unexpected permissibility round count")
    rubric = permissibility.get("rubric")
    if not isinstance(rubric, str) or not rubric:
        raise ValueError("Manifest has no permissibility rubric")

    problems = manifest.get("problems")
    if not isinstance(problems, list) or not problems:
        raise ValueError("Manifest has no problems")

    for problem in problems:
        if not isinstance(problem, dict):
            raise ValueError("Each manifest problem must be an object")
        unexpected = set(problem) - SAFE_PROBLEM_KEYS
        if unexpected:
            raise ValueError(f"Unsafe manifest keys: {sorted(unexpected)}")
        forbidden = set(problem) & FORBIDDEN_SOURCE_KEYS
        if forbidden:
            raise ValueError(f"Ground-truth keys leaked into manifest: {sorted(forbidden)}")
        validate_identifier(problem.get("problem_id"), "problem id")


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    validate_manifest(manifest)
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
