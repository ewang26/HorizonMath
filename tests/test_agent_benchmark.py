"""Tests for the isolated Codex cloud coding-agent pipeline."""

from __future__ import annotations

import json
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from agent_benchmark import (  # noqa: E402
    CloudTaskError,
    CodexCloudClient,
    IsolationError,
    TaskSubmission,
    answer_path,
    build_agent_prompt,
    extract_new_file_from_diff,
    parse_task_state,
    parse_task_submission,
    repository_identity,
    validate_isolated_repository,
)
from create_agent_workspace import create_workspace  # noqa: E402
from benchmark_prompts import system_messages_sha256  # noqa: E402
import evaluator.sandbox as sandbox_module  # noqa: E402
from evaluator.sandbox import ExecutionStatus, execute_sandboxed  # noqa: E402
import evaluate_responses  # noqa: E402
import run_agent_benchmark  # noqa: E402
from run_agent_benchmark import preflight_agent_workspace  # noqa: E402


def test_parse_cloud_submission_and_states():
    submission = parse_task_submission(
        "warning\nhttps://chatgpt.com/codex/tasks/task_i_abc123\n"
    )
    assert submission.task_id == "task_i_abc123"
    assert submission.url.endswith("/task_i_abc123")
    assert parse_task_state("[PENDING] Working") == "PENDING"
    assert parse_task_state("\x1b[32m[READY]\x1b[0m Done") == "READY"


def test_repository_identity_redacts_and_normalizes():
    https = "https://token@example.com/owner/agent-repo.git?secret=yes"
    ssh = "git@example.com:owner/agent-repo.git"
    assert repository_identity(https) == "example.com/owner/agent-repo"
    assert repository_identity(ssh) == "example.com/owner/agent-repo"


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/ewang26/HorizonMath",
        "git@github.com:ewang26/HorizonMath.git",
    ],
)
def test_trusted_repository_is_rejected_as_agent_workspace(url):
    with pytest.raises(IsolationError, match="exposes validators"):
        validate_isolated_repository(url)


def test_agent_prompt_contains_only_explicit_public_inputs():
    system_message = "SYSTEM SENTINEL\n\n  preserve whitespace α"
    problem_prompt = "PUBLIC PROBLEM SENTINEL\n\n```python\nreturn 1\n```"
    prompt = build_agent_prompt(
        problem_id="sample_problem",
        system_message=system_message,
        problem_prompt=problem_prompt,
        output_path="answers/run/sample_problem.md",
    )
    assert (
        f"<benchmark_system_message>\n{system_message}\n"
        "</benchmark_system_message>"
    ) in prompt
    assert f"<problem>\n{problem_prompt}\n</problem>" in prompt
    assert "numeric_value" not in prompt
    assert "test_points" not in prompt
    assert "use Codex goal setting" in prompt
    assert "answers/run/sample_problem.md" in prompt


def test_codex_cloud_agent_runs_always_require_compliance():
    assert evaluate_responses.requires_compliance({"provider": "codex-cloud"})
    assert not evaluate_responses.requires_compliance({"provider": "openai"})
    assert evaluate_responses.requires_compliance(
        {"provider": "openai", "require_compliance": True}
    )


def test_answer_path_rejects_traversal():
    assert answer_path("abc123", "problem_1") == "answers/abc123/problem_1.md"
    with pytest.raises(ValueError):
        answer_path("abc123", "../validators")


def test_extracts_exact_new_answer_file():
    diff = """diff --git a/answers/run/problem.md b/answers/run/problem.md
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/answers/run/problem.md
@@ -0,0 +1,4 @@
+Here is the result.
+
+def proposed_solution():
+    return 42
"""
    result = extract_new_file_from_diff(diff, "answers/run/problem.md")
    assert result == "Here is the result.\n\ndef proposed_solution():\n    return 42\n"


def test_rejects_extra_or_modified_cloud_files():
    extra = """diff --git a/answers/run/problem.md b/answers/run/problem.md
new file mode 100644
--- /dev/null
+++ b/answers/run/problem.md
@@ -0,0 +1 @@
+answer
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-old
+new
"""
    with pytest.raises(CloudTaskError, match="exactly one"):
        extract_new_file_from_diff(extra, "answers/run/problem.md")

    modified = """diff --git a/answers/run/problem.md b/answers/run/problem.md
--- a/answers/run/problem.md
+++ b/answers/run/problem.md
@@ -1 +1 @@
-old
+new
"""
    with pytest.raises(CloudTaskError, match="newly created"):
        extract_new_file_from_diff(modified, "answers/run/problem.md")


def test_cloud_client_submits_prompt_over_stdin(tmp_path):
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="https://chatgpt.com/codex/tasks/task_i_123\n",
        stderr="",
    )
    with patch("agent_benchmark.subprocess.run", return_value=completed) as run:
        client = CodexCloudClient(cwd=tmp_path)
        submission = client.submit(
            prompt="large exact prompt",
            environment="env-1",
            branch="main",
            model="gpt-5.6-sol",
            reasoning_effort="ultra",
        )
    assert submission.task_id == "task_i_123"
    command = run.call_args.args[0]
    assert command[-1] == "-"
    assert command[1:3] == ["cloud", "exec"]
    assert 'model="gpt-5.6-sol"' in command
    assert 'model_reasoning_effort="ultra"' in command
    assert run.call_args.kwargs["input"] == "large exact prompt"


def test_workspace_preflight_checks_entire_git_history(tmp_path):
    workspace = tmp_path / "agent"
    create_workspace(workspace, initialize_git=True)
    answer = workspace / "answers" / "run" / "problem.md"
    answer.parent.mkdir()
    answer.write_text("answer\n", encoding="utf-8")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "answers/run/" in status
    answer.unlink()
    answer.parent.rmdir()

    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/example/hm-agent.git"],
        cwd=workspace,
        check=True,
    )
    origin = preflight_agent_workspace(
        workspace,
        declared_repo_url="https://github.com/example/hm-agent",
        branch="main",
        verify_remote=False,
    )
    assert origin == "https://github.com/example/hm-agent.git"

    # A forbidden file remains detectable even after it is removed from HEAD.
    forbidden = workspace / "validators.py"
    forbidden.write_text("secret = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", "validators.py"], cwd=workspace, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "bad history",
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    forbidden.unlink()
    subprocess.run(["git", "add", "-u"], cwd=workspace, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "remove hidden file",
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    with pytest.raises(IsolationError, match="validators.py"):
        preflight_agent_workspace(
            workspace,
            declared_repo_url="https://github.com/example/hm-agent",
            branch="main",
            verify_remote=False,
        )


def test_workspace_preflight_rejects_remote_branch_drift(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", remote], check=True, capture_output=True)
    workspace = tmp_path / "agent"
    create_workspace(workspace, initialize_git=True)
    subprocess.run(
        ["git", "remote", "add", "origin", str(remote)],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "push", "-u", "origin", "main"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    preflight_agent_workspace(
        workspace,
        declared_repo_url=str(remote),
        branch="main",
    )

    attacker = tmp_path / "attacker"
    subprocess.run(
        ["git", "clone", "--branch", "main", str(remote), attacker],
        check=True,
        capture_output=True,
    )
    forbidden = attacker / "validators.py"
    forbidden.write_text("secret = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", "validators.py"], cwd=attacker, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "remote drift",
        ],
        cwd=attacker,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "push", "origin", "main"],
        cwd=attacker,
        check=True,
        capture_output=True,
    )

    with pytest.raises(IsolationError, match="cloud repository branch"):
        preflight_agent_workspace(
            workspace,
            declared_repo_url=str(remote),
            branch="main",
        )


def test_workspace_preflight_compares_seed_bytes_exactly(tmp_path):
    workspace = tmp_path / "agent"
    create_workspace(workspace, initialize_git=False)
    readme = workspace / "README.md"
    readme.write_text(readme.read_text() + " \n", encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "-f", "."], cwd=workspace, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "altered seed",
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    repo_url = "https://github.com/example/hm-agent.git"
    subprocess.run(
        ["git", "remote", "add", "origin", repo_url],
        cwd=workspace,
        check=True,
    )
    with pytest.raises(IsolationError, match="differs from the trusted template"):
        preflight_agent_workspace(
            workspace,
            declared_repo_url=repo_url,
            branch="main",
            verify_remote=False,
        )


def test_end_to_end_runner_emits_existing_response_schema(tmp_path, monkeypatch):
    workspace = tmp_path / "agent"
    create_workspace(workspace, initialize_git=True)
    repo_url = "https://github.com/example/hm-agent.git"
    subprocess.run(
        ["git", "remote", "add", "origin", repo_url],
        cwd=workspace,
        check=True,
    )
    results_dir = tmp_path / "results"

    class FakeCloudClient:
        prompt = ""
        diff_calls = 0

        def __init__(self, **_kwargs):
            pass

        def preflight(self):
            return None

        def submit(self, *, prompt, environment, branch, model, reasoning_effort):
            assert environment == "test-env"
            assert branch == "main"
            assert model == "gpt-5.6-sol"
            assert reasoning_effort == "ultra"
            self.prompt = prompt
            return TaskSubmission(
                task_id="task_i_test",
                url="https://chatgpt.com/codex/tasks/task_i_test",
            )

        def status(self, task_id):
            assert task_id == "task_i_test"
            return "READY"

        def diff(self, task_id):
            assert task_id == "task_i_test"
            type(self).diff_calls += 1
            if type(self).diff_calls == 1:
                raise CloudTaskError("temporary retrieval failure")
            path = re.search(r"`(answers/[^`]+\.md)`", self.prompt).group(1)
            return f"""diff --git a/{path} b/{path}
new file mode 100644
--- /dev/null
+++ b/{path}
@@ -0,0 +1,2 @@
+def proposed_solution():
+    return 42
"""

    monkeypatch.setattr(run_agent_benchmark, "CodexCloudClient", FakeCloudClient)
    monkeypatch.setattr(
        run_agent_benchmark,
        "preflight_agent_workspace",
        lambda *_args, **_kwargs: repo_url,
    )
    monkeypatch.setattr(
        run_agent_benchmark,
        "create_results_dir",
        lambda _base, _label: (results_dir.mkdir(), results_dir)[1],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agent_benchmark.py",
            "--env",
            "test-env",
            "--agent-workspace",
            str(workspace),
            "--agent-repo-url",
            repo_url,
            "--confirm-agent-internet-off",
            "--confirm-environment-isolated",
            "--confirm-goal-tools-available",
            "--problem",
            "w4_watson_integral",
            "--poll-interval",
            "0.001",
        ],
    )
    run_agent_benchmark.main()

    config = json.loads((results_dir / "config.json").read_text())
    assert config["model"] == "gpt-5.6-sol"
    assert config["reasoning_effort"] == "ultra"
    assert config["resolved_model_confirmation"] == (
        "not exposed by the current codex cloud CLI"
    )
    prompt_record = json.loads((results_dir / "prompts.jsonl").read_text())
    assert prompt_record["system_message_sha256"] == hashlib.sha256(
        prompt_record["system_message"].encode("utf-8")
    ).hexdigest()
    assert prompt_record["problem_prompt_sha256"] == hashlib.sha256(
        prompt_record["prompt"].encode("utf-8")
    ).hexdigest()
    assert prompt_record["agent_task_prompt_sha256"] == hashlib.sha256(
        prompt_record["agent_task_prompt"].encode("utf-8")
    ).hexdigest()
    response = json.loads((results_dir / "responses.jsonl").read_text())
    assert response["provider"] == "codex-cloud"
    assert response["model"] == "gpt-5.6-sol"
    assert response["reasoning_effort"] == "ultra"
    assert response["problem_id"] == "w4_watson_integral"
    assert response["response"] == "def proposed_solution():\n    return 42\n"
    assert FakeCloudClient.diff_calls == 2
    events = [
        json.loads(line)
        for line in (results_dir / "cloud_tasks.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["submitted", "completed"]
    assert events[0]["requested_model"] == "gpt-5.6-sol"
    assert events[0]["requested_reasoning_effort"] == "ultra"


def test_secure_execution_blocks_trusted_checkout():
    safe = execute_sandboxed(
        "def proposed_solution():\n    return mp.pi",
        timeout=10,
    )
    assert safe.status == ExecutionStatus.SUCCESS

    data_path = (ROOT / "data" / "problems_full.json").resolve()
    escape = execute_sandboxed(
        "def proposed_solution():\n"
        f"    return open({str(data_path)!r}).read()\n",
        timeout=10,
    )
    assert escape.status == ExecutionStatus.RUNTIME_ERROR
    assert "Operation not permitted" in (escape.error_message or "")


def test_secure_execution_supports_numpy_answers():
    result = execute_sandboxed(
        "import numpy as np\n"
        "def proposed_solution():\n"
        "    return np.array([2, 3]).prod()\n",
        timeout=10,
    )
    assert result.status == ExecutionStatus.SUCCESS
    assert result.output == "6"


def test_secure_execution_bounds_captured_output(monkeypatch):
    monkeypatch.setattr(sandbox_module, "MAX_OUTPUT_BYTES", 1024)
    result = execute_sandboxed(
        "def proposed_solution():\n"
        "    print('x' * 2048)\n"
        "    return 1\n",
        timeout=10,
    )
    assert result.status == ExecutionStatus.RUNTIME_ERROR
    assert "output exceeded" in (result.error_message or "")


def test_hidden_expected_values_are_not_embedded_in_answer_process():
    result = execute_sandboxed(
        "def proposed_solution(x):\n"
        "    return any(\n"
        "        'EXPECTED_SENTINEL' in repr(value)\n"
        "        for name, value in globals().items()\n"
        "        if name != '__builtins__'\n"
        "    )\n",
        timeout=10,
        test_points=[
            {"args": ["2"], "expected": "EXPECTED_SENTINEL"},
            {"args": ["3"], "expected": "EXPECTED_SENTINEL"},
        ],
    )
    assert result.status == ExecutionStatus.SUCCESS
    assert json.loads(result.output) == ["False", "False"]


def test_evaluator_handles_all_generation_failures_and_deduplicates(
    tmp_path, monkeypatch
):
    results_dir = tmp_path / "failed-run"
    results_dir.mkdir()
    error_path = results_dir / "generation_errors.jsonl"
    errors = [
        {
            "problem_id": "w4_watson_integral",
            "problem_index": 0,
            "problem_title": "w4_watson_integral",
            "mode": "ground_truth_computable",
            "error_type": "runtime",
            "error_message": "first attempt",
        },
        {
            "problem_id": "w4_watson_integral",
            "problem_index": 0,
            "problem_title": "w4_watson_integral",
            "mode": "ground_truth_computable",
            "error_type": "runtime",
            "error_message": "latest attempt",
        },
    ]
    error_path.write_text(
        "".join(json.dumps(error) + "\n" for error in errors),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_responses.py", str(results_dir)],
    )
    with pytest.raises(SystemExit) as exit_info:
        evaluate_responses.main()
    assert exit_info.value.code == 0
    evaluations = [
        json.loads(line)
        for line in (results_dir / "evaluation.jsonl").read_text().splitlines()
    ]
    assert len(evaluations) == 1
    assert evaluations[0]["error_message"] == "latest attempt"


def test_resume_rejects_changed_dataset(tmp_path, monkeypatch):
    workspace = tmp_path / "agent"
    create_workspace(workspace, initialize_git=True)
    repo_url = "https://github.com/example/hm-agent.git"
    subprocess.run(
        ["git", "remote", "add", "origin", repo_url],
        cwd=workspace,
        check=True,
    )
    data_file = tmp_path / "problems.json"
    problem = {
        "id": "sample_problem",
        "prompt": "Original public prompt",
        "evaluation_mode": "ground_truth_computable",
    }
    data_file.write_text(json.dumps([problem]), encoding="utf-8")
    results_dir = tmp_path / "results"
    monkeypatch.setattr(
        run_agent_benchmark,
        "preflight_agent_workspace",
        lambda *_args, **_kwargs: repo_url,
    )
    monkeypatch.setattr(
        run_agent_benchmark,
        "create_results_dir",
        lambda _base, _label: (results_dir.mkdir(), results_dir)[1],
    )
    base_argv = [
        "run_agent_benchmark.py",
        "--debug",
        "--env",
        "test-env",
        "--agent-workspace",
        str(workspace),
        "--agent-repo-url",
        repo_url,
        "--data-file",
        str(data_file),
    ]
    monkeypatch.setattr(sys, "argv", base_argv)
    run_agent_benchmark.main()

    problem["prompt"] = "Changed public prompt"
    data_file.write_text(json.dumps([problem]), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [*base_argv, "--resume", str(results_dir)],
    )
    with pytest.raises(SystemExit) as exit_info:
        run_agent_benchmark.main()
    assert exit_info.value.code == 2


def test_multi_problem_run_requires_live_canary(tmp_path, monkeypatch):
    workspace = tmp_path / "agent"
    create_workspace(workspace, initialize_git=True)
    repo_url = "https://github.com/example/hm-agent.git"
    subprocess.run(
        ["git", "remote", "add", "origin", repo_url],
        cwd=workspace,
        check=True,
    )
    data_file = tmp_path / "problems.json"
    data_file.write_text(
        json.dumps(
            [
                {
                    "id": f"sample_{index}",
                    "prompt": f"Problem {index}",
                    "evaluation_mode": "ground_truth_computable",
                }
                for index in range(2)
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_agent_benchmark,
        "preflight_agent_workspace",
        lambda *_args, **_kwargs: repo_url,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agent_benchmark.py",
            "--env",
            "test-env",
            "--agent-workspace",
            str(workspace),
            "--agent-repo-url",
            repo_url,
            "--data-file",
            str(data_file),
            "--confirm-agent-internet-off",
            "--confirm-environment-isolated",
            "--confirm-goal-tools-available",
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        run_agent_benchmark.main()
    assert exit_info.value.code == 2


def test_evaluator_uses_and_validates_configured_dataset(tmp_path, monkeypatch):
    results_dir = tmp_path / "custom-run"
    results_dir.mkdir()
    data_file = tmp_path / "custom-problems.json"
    custom_problem = {
        "id": "custom_problem",
        "prompt": "Custom prompt",
        "evaluation_mode": "ground_truth_computable",
        "numeric_value": "1",
    }
    data_file.write_text(json.dumps([custom_problem]), encoding="utf-8")
    (results_dir / "generation_errors.jsonl").write_text(
        json.dumps(
            {
                "problem_id": "custom_problem",
                "problem_index": 0,
                "problem_title": "custom_problem",
                "mode": "ground_truth_computable",
                "error_type": "runtime",
                "error_message": "cloud failure",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (results_dir / "config.json").write_text(
        json.dumps(
            {
                "problems_file": str(data_file),
                "dataset_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
                "system_messages_sha256": system_messages_sha256(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_responses.py", str(results_dir)],
    )
    with pytest.raises(SystemExit) as exit_info:
        evaluate_responses.main()
    assert exit_info.value.code == 0

    data_file.write_text(json.dumps([{**custom_problem, "prompt": "changed"}]))
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_responses.py", str(results_dir), "--force"],
    )
    with pytest.raises(SystemExit) as exit_info:
        evaluate_responses.main()
    assert exit_info.value.code == 1


@pytest.mark.parametrize("duplicate", [False, True])
def test_evaluator_rejects_incomplete_or_duplicate_agent_runs(
    tmp_path, monkeypatch, duplicate
):
    results_dir = tmp_path / ("duplicate-run" if duplicate else "incomplete-run")
    results_dir.mkdir()
    (results_dir / "config.json").write_text(
        json.dumps({"selected_problem_ids": ["w4_watson_integral"]}),
        encoding="utf-8",
    )
    responses = []
    if duplicate:
        response = {
            "problem_id": "w4_watson_integral",
            "response": "def proposed_solution():\n    return 1\n",
        }
        responses = [response, response]
    (results_dir / "responses.jsonl").write_text(
        "".join(json.dumps(response) + "\n" for response in responses),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_responses.py", str(results_dir)],
    )
    with pytest.raises(SystemExit) as exit_info:
        evaluate_responses.main()
    assert exit_info.value.code == 1
