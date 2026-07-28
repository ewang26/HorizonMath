import json
import sys
import tomllib
from pathlib import Path

import pytest

from agent_eval.cloud_scorer import candidate_payload
from agent_eval.config import (
    AGENT_PERMISSION_PROFILE,
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    PERMISSIBILITY_EFFORT,
    PERMISSIBILITY_MODEL,
    PERMISSIBILITY_ROUNDS,
    PROBLEM_TIMEOUT_SECONDS,
    codex_config_toml,
    sandbox_timeout_seconds,
)
from agent_eval.manifest import build_manifest, validate_manifest
from agent_eval.modal_runner import (
    developer_instructions,
    is_missing_volume_path_error,
    permissibility_rubric,
)
from agent_eval.full_report import (
    passed_after_all_gates,
    passed_before_compliance,
)
from agent_eval.runtime.worker import (
    aggregate_permissibility_rounds,
    extract_proposed_solution_code,
    initial_turn_prompt,
    parse_permissibility_response,
    validate_manifest as validate_runtime_manifest,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_prompts import SYSTEM_MESSAGES
from evaluate_responses import apply_compliance_gate
from evaluator.sandbox import strip_expected_values
from run_benchmark import SYSTEM_MESSAGES as SINGLE_SHOT_SYSTEM_MESSAGES


def sample_problem() -> dict:
    return {
        "id": "hidden_answer_test",
        "prompt": "Return a proposed_solution function.",
        "evaluation_mode": "ground_truth_computable",
        "numeric_value": "123456789.987654321",
        "test_points": [{"args": ["1"], "expected": "999999999"}],
        "source_url": "https://example.invalid/answer",
    }


def test_fresh_agent_prompts_match_single_shot_byte_for_byte():
    assert SYSTEM_MESSAGES == SINGLE_SHOT_SYSTEM_MESSAGES
    assert developer_instructions() == SINGLE_SHOT_SYSTEM_MESSAGES
    problem = sample_problem()
    assert initial_turn_prompt(problem) == problem["prompt"]

    worker_source = (
        Path(__file__).resolve().parents[1]
        / "agent_eval"
        / "runtime"
        / "worker.py"
    ).read_text()
    assert "TURN_SUFFIX" not in worker_source
    assert "AGENTS_TEXT" not in worker_source


def test_manifest_contains_only_agent_safe_problem_fields():
    manifest = build_manifest(
        [sample_problem()],
        [0],
        run_id="test-run",
        developer_instructions_by_mode=SYSTEM_MESSAGES,
        permissibility_rubric=permissibility_rubric(),
    )
    encoded = json.dumps(manifest)
    problem = manifest["problems"][0]

    assert manifest["model"] == DEFAULT_MODEL
    assert manifest["reasoning_effort"] == DEFAULT_EFFORT
    assert manifest["problem_timeout_seconds"] == 10800
    assert manifest["permissibility"]["model"] == PERMISSIBILITY_MODEL
    assert manifest["permissibility"]["reasoning_effort"] == PERMISSIBILITY_EFFORT
    assert manifest["permissibility"]["rounds"] == PERMISSIBILITY_ROUNDS
    assert manifest["permissibility"]["rubric"] == permissibility_rubric()
    assert set(problem) == {
        "problem_id",
        "problem_index",
        "prompt",
        "evaluation_mode",
        "developer_instructions",
        "prompt_sha256",
    }
    assert "123456789.987654321" not in encoded
    assert "999999999" not in encoded
    assert "example.invalid" not in encoded
    validate_runtime_manifest(manifest)


def test_manifest_rejects_wrong_model_effort_or_timeout():
    manifest = build_manifest(
        [sample_problem()],
        [0],
        run_id="test-run",
        developer_instructions_by_mode=SYSTEM_MESSAGES,
        permissibility_rubric=permissibility_rubric(),
    )
    for key, bad_value in (
        ("model", "not-sol"),
        ("reasoning_effort", "high"),
        ("problem_timeout_seconds", 7200),
    ):
        changed = dict(manifest)
        changed[key] = bad_value
        with pytest.raises(ValueError):
            validate_manifest(changed)


def test_manifest_rejects_path_traversal_identifiers():
    with pytest.raises(ValueError):
        build_manifest(
            [sample_problem()],
            [0],
            run_id="../../escape",
            developer_instructions_by_mode=SYSTEM_MESSAGES,
            permissibility_rubric=permissibility_rubric(),
        )
    problem = sample_problem()
    problem["id"] = "../answer"
    with pytest.raises(ValueError):
        build_manifest(
            [problem],
            [0],
            run_id="safe-run",
            developer_instructions_by_mode=SYSTEM_MESSAGES,
            permissibility_rubric=permissibility_rubric(),
        )


def test_codex_config_denies_secrets_and_uses_seccomp_network_boundary():
    config = codex_config_toml()
    parsed = tomllib.loads(config)
    assert f'default_permissions = "{AGENT_PERMISSION_PROFILE}"' in config
    assert 'web_search = "disabled"' in config
    assert '"/codex-home" = "deny"' in config
    assert '"/state" = "deny"' in config
    assert '"/opt/horizonmath_agent_runtime" = "deny"' in config
    assert "enabled = true" in config
    assert '":workspace_roots"' in config
    assert '"." = "write"' in config
    assert parsed["permissions"][AGENT_PERMISSION_PROFILE]["network"]["enabled"] is True

    seccomp_source = (
        Path(__file__).resolve().parents[1]
        / "agent_eval"
        / "runtime"
        / "sandbox_shell.c"
    ).read_text()
    for syscall in ("socket", "connect", "sendto", "recvfrom"):
        assert f"SCMP_SYS({syscall})" in seccomp_source
    assert "PR_SET_NO_NEW_PRIVS" in seccomp_source


def test_modal_auth_is_interactive_ephemeral_and_never_copied_from_local_disk():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "agent_eval" / "modal_runner.py").read_text()
    entrypoint = (
        repo_root / "agent_eval" / "runtime" / "entrypoint.py"
    ).read_text()

    assert "_resolve_codex_bin" in entrypoint
    assert '"login", "--device-auth"' in entrypoint
    assert "prepare_ephemeral_codex_home" in entrypoint
    assert "AUTH_VOLUME_NAME" not in launcher
    assert "local_auth_path" not in launcher
    assert "seed_codex_home" not in launcher
    assert 'str(REMOTE_AUTH_ROOT):' not in launcher
    assert '"auth_persisted": False' in launcher


def test_first_ten_batch_fits_inside_modal_lifetime():
    assert sandbox_timeout_seconds(10, 4, PROBLEM_TIMEOUT_SECONDS) == 37800
    with pytest.raises(ValueError):
        sandbox_timeout_seconds(113, 4, PROBLEM_TIMEOUT_SECONDS)


def test_expected_values_are_removed_from_both_execution_paths():
    points = [{"args": ["1.25"], "expected": "TOP_SECRET_GROUND_TRUTH"}]
    assert strip_expected_values(points) == [{"args": ["1.25"]}]

    payload = candidate_payload(
        "def proposed_solution(x): return x",
        precision_dps=110,
        return_json=False,
        test_points=points,
    )
    encoded = json.dumps(payload)
    assert payload["points"] == [{"args": ["1.25"]}]
    assert "TOP_SECRET_GROUND_TRUTH" not in encoded


def test_problem_timeout_is_exactly_three_hours():
    assert PROBLEM_TIMEOUT_SECONDS == 3 * 60 * 60


@pytest.mark.parametrize(
    "message",
    [
        "not found",
        "No such file",
        "Path does not exist",
    ],
)
def test_modal_missing_volume_errors_are_recognized(message):
    assert is_missing_volume_path_error(RuntimeError(message))


def test_permissibility_review_extracts_and_parses_strict_json():
    response = """
Draft:
```python
def proposed_solution():
    return 1
```
Final:
```python
def proposed_solution():
    return 2
```
"""
    assert "return 2" in extract_proposed_solution_code(response)
    assert parse_permissibility_response(
        '```json\n{"compliant": true, "reason": "exact closed form"}\n```'
    ) == {
        "compliant": True,
        "reason": "exact closed form",
    }


def test_permissibility_vote_requires_a_strict_majority():
    decision = aggregate_permissibility_rounds(
        [
            {"compliant": True, "reason": "valid"},
            {"compliant": None, "error": "timeout"},
            {"compliant": None, "error": "parse error"},
        ]
    )
    assert decision["status"] == "indeterminate"
    assert decision["compliant"] is None
    assert decision["votes"] == {
        "compliant": 1,
        "non_compliant": 0,
        "indeterminate": 2,
        "total": 3,
    }


def test_worker_runs_terra_review_before_marking_batch_complete():
    worker_source = (
        Path(__file__).resolve().parents[1]
        / "agent_eval"
        / "runtime"
        / "worker.py"
    ).read_text()
    reviewing = worker_source.index('runner_status["status"] = "reviewing"')
    review_call = worker_source.index("compliance_status = await review_batch")
    completed = worker_source.index('"status": "completed"', review_call)
    assert reviewing < review_call < completed
    assert 'PERMISSIBILITY_MODEL = "gpt-5.6-terra"' in worker_source
    assert 'PERMISSIBILITY_EFFORT = "high"' in worker_source


def test_launcher_starts_separate_automatic_trusted_scorer():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "agent_eval" / "modal_runner.py").read_text()
    scorer = (repo_root / "agent_eval" / "trusted_scorer.py").read_text()

    assert "deploy_and_spawn(run_id)" in launcher
    assert '"scoring_status.json"' in launcher
    assert "watch_and_score" in scorer
    assert "score_problem.starmap" in scorer
    assert "block_network=True" in scorer
    assert "restrict_modal_access=True" in scorer
    assert "single_use_containers=True" in scorer
    assert 'LOCAL_REPO_ROOT / "validators"' in scorer
    assert "candidate_runtime" in scorer


@pytest.mark.parametrize("pass_key", ["success", "valid"])
def test_permissibility_gates_every_passing_evaluation_mode(pass_key):
    evaluation = {pass_key: True, "error_type": None, "error_message": None}
    gated = apply_compliance_gate(
        evaluation,
        pass_key=pass_key,
        response="```python\ndef proposed_solution():\n    return 1\n```",
        problem_prompt="Return an exact expression.",
        precomputed_compliance={
            "status": "non_compliant",
            "compliant": False,
            "reason": "Uses a forbidden numerical procedure.",
            "provider": "codex-chatgpt-subscription",
            "model": "gpt-5.6-terra",
        },
    )
    assert gated[pass_key] is False
    assert gated["error_type"] == "compliance"
    assert gated["compliance_passed"] is False


def test_full_report_uses_strict_benchmark_and_raw_compliance_semantics():
    assert passed_after_all_gates(
        {
            "mode": "benchmark",
            "valid": True,
            "baseline_comparison": {"result": "beats_baseline"},
        }
    )
    assert not passed_after_all_gates(
        {
            "mode": "benchmark",
            "valid": True,
            "baseline_comparison": {"result": "matches_baseline"},
        }
    )
    assert passed_before_compliance(
        {
            "mode": "numeric",
            "success": False,
            "error_type": "compliance",
        }
    )
