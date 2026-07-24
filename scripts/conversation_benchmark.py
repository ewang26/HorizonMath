"""Fail-closed provenance checks for repository-free Codex cloud runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROVIDER = "codex-conversation"
RUNTIME_CONTRACT_VERSION = 1
REQUIRED_EXECUTION_LOCATION = "openai-hosted"
REQUIRED_RUNTIME_KIND = "codex-cloud-container"
LOCAL_HOST_IDS = frozenset(
    {"local", "localhost", "127.0.0.1", "::1", "this-device"}
)


class ConversationRunError(ValueError):
    """Raised when a conversation run lacks required cloud-runtime proof."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl_strict(path: Path) -> list[dict]:
    """Load JSONL without silently dropping malformed provenance records."""

    if not path.exists():
        raise ConversationRunError(f"Required file is missing: {path.name}")
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConversationRunError(
                    f"{path.name}:{line_number} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ConversationRunError(
                    f"{path.name}:{line_number} must contain a JSON object"
                )
            records.append(record)
    return records


def _unique_by_problem(records: list[dict], *, source: str) -> dict[str, dict]:
    by_problem: dict[str, dict] = {}
    for record in records:
        problem_id = record.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id:
            raise ConversationRunError(
                f"{source} contains a record without a non-empty problem_id"
            )
        if problem_id in by_problem:
            raise ConversationRunError(
                f"{source} contains duplicate records for {problem_id}"
            )
        by_problem[problem_id] = record
    return by_problem


def _require_equal(
    record: dict,
    field: str,
    expected: object,
    *,
    problem_id: str,
    source: str = "cloud_runtime.jsonl",
) -> None:
    actual = record.get(field)
    if actual != expected:
        raise ConversationRunError(
            f"{problem_id}: {source} field {field!r} must be "
            f"{expected!r}, got {actual!r}"
        )


def _validate_confirmed_cloud_runtime(
    evidence: dict,
    *,
    problem_id: str,
    config: dict,
) -> None:
    _require_equal(
        evidence,
        "execution_location",
        REQUIRED_EXECUTION_LOCATION,
        problem_id=problem_id,
    )
    _require_equal(
        evidence,
        "runtime_kind",
        REQUIRED_RUNTIME_KIND,
        problem_id=problem_id,
    )
    _require_equal(
        evidence,
        "repository_attachment",
        "none",
        problem_id=problem_id,
    )
    _require_equal(
        evidence,
        "local_execution_used",
        False,
        problem_id=problem_id,
    )
    _require_equal(
        evidence,
        "model",
        config.get("model"),
        problem_id=problem_id,
    )
    _require_equal(
        evidence,
        "reasoning_effort",
        config.get("reasoning_effort"),
        problem_id=problem_id,
    )
    _require_equal(
        evidence,
        "creation_confirmation_source",
        "task-creation-response",
        problem_id=problem_id,
    )

    host_id = evidence.get("host_id")
    if not isinstance(host_id, str) or not host_id.strip():
        raise ConversationRunError(
            f"{problem_id}: cloud runtime host_id was not confirmed"
        )
    if host_id.strip().lower() in LOCAL_HOST_IDS:
        raise ConversationRunError(
            f"{problem_id}: host_id={host_id!r} is local; model inference in the "
            "cloud does not make local tool execution a Codex cloud run"
        )
    for field in ("cloud_environment_id", "task_id"):
        value = evidence.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ConversationRunError(
                f"{problem_id}: cloud runtime field {field!r} was not confirmed"
            )


def validate_conversation_cloud_run(
    *,
    results_dir: Path,
    config: dict,
    responses: list[dict],
    generation_errors: list[dict],
) -> None:
    """Require per-problem proof that scored agents ran in Codex cloud containers."""

    if config.get("provider") != PROVIDER:
        return
    if config.get("runtime_contract_version") != RUNTIME_CONTRACT_VERSION:
        raise ConversationRunError(
            "config.json does not declare the current Codex cloud runtime contract"
        )
    if config.get("required_execution_location") != REQUIRED_EXECUTION_LOCATION:
        raise ConversationRunError(
            "config.json does not require OpenAI-hosted execution"
        )
    if config.get("required_runtime_kind") != REQUIRED_RUNTIME_KIND:
        raise ConversationRunError(
            "config.json does not require a Codex cloud container"
        )
    if config.get("local_agent_execution_forbidden") is not True:
        raise ConversationRunError(
            "config.json must explicitly forbid local scored-agent execution"
        )
    prompts_path = results_dir / "prompts.jsonl"
    expected_prompts_hash = config.get("prompts_jsonl_sha256")
    if not isinstance(expected_prompts_hash, str) or not expected_prompts_hash:
        raise ConversationRunError(
            "config.json does not bind the prepared prompts.jsonl manifest"
        )
    if not prompts_path.exists():
        raise ConversationRunError("Required file is missing: prompts.jsonl")
    actual_prompts_hash = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    if actual_prompts_hash != expected_prompts_hash:
        raise ConversationRunError(
            "prompts.jsonl differs from the manifest prepared in config.json"
        )

    selected = config.get("selected_problem_ids")
    if not isinstance(selected, list) or not selected:
        raise ConversationRunError(
            "config.json must list the selected problem IDs"
        )

    prompts = _unique_by_problem(
        load_jsonl_strict(prompts_path),
        source="prompts.jsonl",
    )
    evidence_by_problem = _unique_by_problem(
        load_jsonl_strict(results_dir / "cloud_runtime.jsonl"),
        source="cloud_runtime.jsonl",
    )
    responses_by_problem = _unique_by_problem(
        responses,
        source="responses.jsonl",
    )
    errors_by_problem = _unique_by_problem(
        generation_errors,
        source="generation_errors.jsonl",
    )

    selected_set = set(selected)
    if set(prompts) != selected_set:
        raise ConversationRunError(
            "prompts.jsonl problem IDs do not exactly match config.json"
        )
    if set(evidence_by_problem) != selected_set:
        missing = sorted(selected_set - set(evidence_by_problem))
        unexpected = sorted(set(evidence_by_problem) - selected_set)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ConversationRunError(
            "cloud_runtime.jsonl must contain exactly one terminal record per "
            f"selected problem ({'; '.join(details)})"
        )

    for problem_id in selected:
        prompt = prompts[problem_id]
        evidence = evidence_by_problem[problem_id]
        outcome = evidence.get("terminal_outcome")
        problem_index = prompt.get("problem_index")
        _require_equal(
            evidence,
            "problem_index",
            problem_index,
            problem_id=problem_id,
        )
        expected_prompt_hash = sha256_text(prompt.get("agent_task_prompt", ""))
        _require_equal(
            evidence,
            "agent_task_prompt_sha256",
            expected_prompt_hash,
            problem_id=problem_id,
        )
        _require_equal(
            evidence,
            "local_execution_used",
            False,
            problem_id=problem_id,
        )
        _require_equal(
            evidence,
            "model",
            config.get("model"),
            problem_id=problem_id,
        )
        _require_equal(
            evidence,
            "reasoning_effort",
            config.get("reasoning_effort"),
            problem_id=problem_id,
        )

        if outcome == "completed":
            if problem_id not in responses_by_problem:
                raise ConversationRunError(
                    f"{problem_id}: completed cloud record has no response"
                )
            if problem_id in errors_by_problem:
                raise ConversationRunError(
                    f"{problem_id}: completed cloud record also has a generation error"
                )
            _validate_confirmed_cloud_runtime(
                evidence,
                problem_id=problem_id,
                config=config,
            )
            response = responses_by_problem[problem_id]
            _require_equal(
                response,
                "provider",
                PROVIDER,
                problem_id=problem_id,
                source="responses.jsonl",
            )
            _require_equal(
                response,
                "model",
                config.get("model"),
                problem_id=problem_id,
                source="responses.jsonl",
            )
            _require_equal(
                response,
                "problem_index",
                problem_index,
                problem_id=problem_id,
                source="responses.jsonl",
            )
            _require_equal(
                evidence,
                "response_source",
                "primary-final-response",
                problem_id=problem_id,
            )
            _require_equal(
                evidence,
                "response_sha256",
                sha256_text(response.get("response", "")),
                problem_id=problem_id,
            )
        elif outcome == "generation_error":
            if problem_id not in errors_by_problem:
                raise ConversationRunError(
                    f"{problem_id}: generation_error cloud record has no error outcome"
                )
            if problem_id in responses_by_problem:
                raise ConversationRunError(
                    f"{problem_id}: generation_error cloud record also has a response"
                )
            task_created = evidence.get("task_created")
            if task_created is True:
                _validate_confirmed_cloud_runtime(
                    evidence,
                    problem_id=problem_id,
                    config=config,
                )
            elif task_created is False:
                _require_equal(
                    evidence,
                    "error_stage",
                    "task_creation",
                    problem_id=problem_id,
                )
                _require_equal(
                    evidence,
                    "attempted_runtime_kind",
                    REQUIRED_RUNTIME_KIND,
                    problem_id=problem_id,
                )
            else:
                raise ConversationRunError(
                    f"{problem_id}: generation error must declare task_created"
                )
        else:
            raise ConversationRunError(
                f"{problem_id}: terminal_outcome must be 'completed' or "
                "'generation_error'"
            )
