"""Verifier-free runtime that drives parallel Codex threads inside one Modal Sandbox."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai_codex import ApprovalMode, AsyncCodex, CodexConfig
from openai_codex.client import _resolve_codex_bin

AUTH_ROOT = Path("/codex-home")
STATE_ROOT = Path("/state")
WORK_ROOT = Path("/workspaces")
FORBIDDEN_ROOT = Path("/forbidden")
PERMISSION_PROFILE = "horizonmath-agent"

ALLOWED_MANIFEST_PROBLEM_KEYS = {
    "problem_id",
    "problem_index",
    "prompt",
    "evaluation_mode",
    "developer_instructions",
    "prompt_sha256",
}
FORBIDDEN_TERMS = {
    "numeric_value",
    "test_points",
    "expected_value",
    "baseline",
    "validator",
    "source_url",
}
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
PERMISSIBILITY_MODEL = "gpt-5.6-terra"
PERMISSIBILITY_EFFORT = "high"
PERMISSIBILITY_ROUNDS = 3
PERMISSIBILITY_CONCURRENCY = 8
PERMISSIBILITY_TIMEOUT_SECONDS = 20 * 60
REVIEWER_DEVELOPER_INSTRUCTIONS = (
    "You are a strict mathematical permissibility reviewer. Apply the supplied "
    "rubric to the submitted proposed_solution. Return only the requested JSON "
    "object, without markdown or additional commentary."
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def sync_mount(path: Path) -> None:
    subprocess.run(["sync", str(path)], check=True, timeout=120)


def validate_runtime() -> None:
    auth_path = AUTH_ROOT / "auth.json"
    config_path = AUTH_ROOT / "config.toml"
    if not auth_path.is_file() or not config_path.is_file():
        raise RuntimeError("Ephemeral Codex device auth/config is unavailable")
    auth = json.loads(auth_path.read_text())
    if auth.get("auth_mode") != "chatgpt":
        raise RuntimeError("Codex must use ChatGPT-managed authentication")
    if not ((auth.get("tokens") or {}).get("refresh_token")):
        raise RuntimeError("Codex auth has no refresh token")
    if os.getenv("OPENAI_API_KEY") or os.getenv("CODEX_API_KEY"):
        raise RuntimeError("API-key authentication is forbidden for this evaluation")


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    if manifest.get("model") != "gpt-5.6-sol":
        raise ValueError("Unexpected model")
    if manifest.get("reasoning_effort") != "xhigh":
        raise ValueError("Unexpected reasoning effort")
    if manifest.get("problem_timeout_seconds") != 10800:
        raise ValueError("Unexpected problem timeout")
    if not IDENTIFIER_RE.fullmatch(str(manifest.get("run_id", ""))):
        raise ValueError("Unsafe run id")
    permissibility = manifest.get("permissibility")
    if not isinstance(permissibility, dict):
        raise ValueError("Manifest has no permissibility configuration")
    if permissibility.get("model") != PERMISSIBILITY_MODEL:
        raise ValueError("Unexpected permissibility model")
    if permissibility.get("reasoning_effort") != PERMISSIBILITY_EFFORT:
        raise ValueError("Unexpected permissibility effort")
    if permissibility.get("rounds") != PERMISSIBILITY_ROUNDS:
        raise ValueError("Unexpected permissibility rounds")
    if not isinstance(permissibility.get("rubric"), str):
        raise ValueError("Manifest has no permissibility rubric")
    problems = manifest.get("problems")
    if not isinstance(problems, list) or not problems:
        raise ValueError("Manifest has no problems")
    for problem in problems:
        if set(problem) != ALLOWED_MANIFEST_PROBLEM_KEYS:
            raise ValueError(f"Unsafe problem manifest keys: {sorted(set(problem))}")
        if set(problem) & FORBIDDEN_TERMS:
            raise ValueError("Manifest includes verifier or ground-truth fields")
        if not IDENTIFIER_RE.fullmatch(str(problem.get("problem_id", ""))):
            raise ValueError("Unsafe problem id")


def run_sandbox_self_test(codex_bin: Path, workspace: Path) -> dict[str, Any]:
    """Prove the nested Codex sandbox can write locally but not read protected mounts."""

    probe = r"""
set -eu
test ! -r /codex-home/auth.json
test ! -r /state/.controller-canary
test ! -r /forbidden/canary.txt
printf 'workspace-ok\n' > sandbox-self-test.txt
test -s sandbox-self-test.txt
if python - <<'PY'
import socket
try:
    socket.create_connection(("1.1.1.1", 443), timeout=1)
except OSError:
    raise SystemExit(1)
PY
then
  echo "network unexpectedly available" >&2
  exit 91
fi
"""
    started = time.monotonic()
    result = subprocess.run(
        [
            str(codex_bin),
            "sandbox",
            "--permission-profile",
            PERMISSION_PROFILE,
            "--cd",
            str(workspace),
            "--",
            "bash",
            "-lc",
            probe,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ.copy(),
        check=False,
    )
    report = {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "duration_ms": int((time.monotonic() - started) * 1000),
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-4000:],
    }
    if result.returncode != 0:
        raise RuntimeError(f"Codex sandbox self-test failed: {report}")
    return report


def workspace_for(run_id: str, problem: dict[str, Any]) -> Path:
    safe_id = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in problem["problem_id"]
    )
    return WORK_ROOT / run_id / f"{problem['problem_index']:03d}_{safe_id}"


def initial_turn_prompt(problem: dict[str, Any]) -> str:
    """Return the dataset prompt verbatim for a fresh agent thread."""

    return problem["prompt"]


def prepare_workspace(run_id: str, problem: dict[str, Any]) -> Path:
    workspace = workspace_for(run_id, problem)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    (workspace / "problem.md").write_text(problem["prompt"] + "\n")
    (workspace / ".tmp").mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.name", "HorizonMath Agent"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "agent-eval@invalid"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(["git", "add", "problem.md"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "Initialize verifier-free problem workspace"],
        cwd=workspace,
        check=True,
    )
    return workspace


def serialize_usage(usage: Any) -> Any:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    return str(usage)


def serialize_items(items: list[Any]) -> list[Any]:
    serialized = []
    for item in items:
        if hasattr(item, "model_dump"):
            serialized.append(item.model_dump(mode="json"))
        else:
            serialized.append(str(item))
    return serialized


def extract_proposed_solution_code(response: str) -> str:
    """Extract the last Python code block containing proposed_solution."""

    blocks = re.findall(r"```(?:python|py|python3)?\s*\n?(.*?)```", response, re.DOTALL)
    candidates = [block.strip() for block in blocks if "def proposed_solution" in block]
    if candidates:
        return candidates[-1]
    if "def proposed_solution" in response:
        return response.strip()
    return ""


def parse_permissibility_response(text: str) -> dict[str, Any]:
    """Parse one strict reviewer response."""

    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```\w*\n?", "", value)
        value = re.sub(r"\n?```$", "", value).strip()
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Reviewer response is not a JSON object")
    if not isinstance(payload.get("compliant"), bool):
        raise ValueError("Reviewer response has no boolean compliant field")
    if not isinstance(payload.get("reason"), str) or not payload["reason"].strip():
        raise ValueError("Reviewer response has no non-empty reason")
    return {
        "compliant": payload["compliant"],
        "reason": payload["reason"].strip(),
    }


def aggregate_permissibility_rounds(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply a strict-majority decision without treating errors as passes."""

    compliant_count = sum(item.get("compliant") is True for item in rounds)
    non_compliant_count = sum(item.get("compliant") is False for item in rounds)
    indeterminate_count = len(rounds) - compliant_count - non_compliant_count
    if compliant_count > len(rounds) / 2:
        compliant: bool | None = True
        reason = next(item["reason"] for item in rounds if item.get("compliant") is True)
        status = "compliant"
    elif non_compliant_count > len(rounds) / 2:
        compliant = False
        reason = next(item["reason"] for item in rounds if item.get("compliant") is False)
        status = "non_compliant"
    else:
        compliant = None
        status = "indeterminate"
        errors = [item.get("error") for item in rounds if item.get("error")]
        reason = "No strict permissibility majority was reached."
        if errors:
            reason += f" {errors[0]}"
    return {
        "status": status,
        "compliant": compliant,
        "reason": reason,
        "votes": {
            "compliant": compliant_count,
            "non_compliant": non_compliant_count,
            "indeterminate": indeterminate_count,
            "total": len(rounds),
        },
    }


def permissibility_prompt(manifest: dict[str, Any], problem: dict[str, Any], code: str) -> str:
    problem_context = (
        "The problem being solved is described below. Pay close attention to any "
        "problem-specific restrictions — these are additional rules that MUST be "
        "enforced on top of the general rules above.\n\n"
        f"**Problem description:**\n{problem['prompt']}\n\n"
    )
    return manifest["permissibility"]["rubric"].format(
        code=code,
        problem_context=problem_context,
    )


def review_workspace(run_id: str, problem_index: int, round_index: int) -> Path:
    workspace = (
        WORK_ROOT
        / run_id
        / "_permissibility"
        / f"{problem_index:03d}"
        / f"round_{round_index}"
    )
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


async def run_permissibility_round(
    codex: AsyncCodex,
    manifest: dict[str, Any],
    problem: dict[str, Any],
    code: str,
    round_index: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        workspace = review_workspace(
            manifest["run_id"],
            problem["problem_index"],
            round_index,
        )
        started = time.monotonic()
        thread = None
        turn = None
        try:
            thread = await codex.thread_start(
                approval_mode=ApprovalMode.deny_all,
                cwd=str(workspace),
                developer_instructions=REVIEWER_DEVELOPER_INSTRUCTIONS,
                ephemeral=True,
                model=PERMISSIBILITY_MODEL,
            )
            turn = await thread.turn(
                permissibility_prompt(manifest, problem, code),
                approval_mode=ApprovalMode.deny_all,
                cwd=str(workspace),
                effort=PERMISSIBILITY_EFFORT,
                model=PERMISSIBILITY_MODEL,
            )
            result = await asyncio.wait_for(
                turn.run(),
                timeout=PERMISSIBILITY_TIMEOUT_SECONDS,
            )
            parsed = parse_permissibility_response(result.final_response or "")
            return {
                "round": round_index,
                **parsed,
                "thread_id": thread.id,
                "turn_id": turn.id,
                "duration_seconds": round(time.monotonic() - started, 3),
                "usage": serialize_usage(result.usage),
            }
        except TimeoutError:
            if turn is not None:
                try:
                    await turn.interrupt()
                except Exception:
                    pass
            return {
                "round": round_index,
                "compliant": None,
                "error": "Permissibility review timed out",
                "duration_seconds": round(time.monotonic() - started, 3),
            }
        except Exception as exc:
            return {
                "round": round_index,
                "compliant": None,
                "error": f"{type(exc).__name__}: {exc}",
                "duration_seconds": round(time.monotonic() - started, 3),
            }


async def review_problem(
    codex: AsyncCodex,
    manifest: dict[str, Any],
    problem: dict[str, Any],
    response: dict[str, Any],
    run_dir: Path,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    result_path = (
        run_dir
        / "compliance"
        / f"{problem['problem_index']:03d}_{problem['problem_id']}.json"
    )
    code = extract_proposed_solution_code(response.get("response", ""))
    if response.get("status") != "completed" or not code:
        result = {
            "schema_version": 1,
            "problem_id": problem["problem_id"],
            "problem_index": problem["problem_index"],
            "status": "indeterminate",
            "compliant": None,
            "reason": "No completed proposed_solution was available for review.",
            "provider": "codex-chatgpt-subscription",
            "model": PERMISSIBILITY_MODEL,
            "reasoning_effort": PERMISSIBILITY_EFFORT,
            "rounds": [],
        }
    else:
        rounds = await asyncio.gather(
            *[
                run_permissibility_round(
                    codex,
                    manifest,
                    problem,
                    code,
                    round_index,
                    semaphore,
                )
                for round_index in range(1, PERMISSIBILITY_ROUNDS + 1)
            ]
        )
        decision = aggregate_permissibility_rounds(rounds)
        result = {
            "schema_version": 1,
            "problem_id": problem["problem_id"],
            "problem_index": problem["problem_index"],
            **decision,
            "provider": "codex-chatgpt-subscription",
            "model": PERMISSIBILITY_MODEL,
            "reasoning_effort": PERMISSIBILITY_EFFORT,
            "rounds": rounds,
        }
    atomic_json(result_path, result)
    sync_mount(STATE_ROOT)
    print(
        json.dumps(
            {
                "event": "permissibility_finished",
                "problem_id": problem["problem_id"],
                "status": result["status"],
            }
        ),
        flush=True,
    )
    return result


async def review_batch(
    codex: AsyncCodex,
    manifest: dict[str, Any],
    responses: list[dict[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    started_at = utc_now()
    status_path = run_dir / "compliance_status.json"
    status = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "status": "running",
        "started_at": started_at,
        "provider": "codex-chatgpt-subscription",
        "model": PERMISSIBILITY_MODEL,
        "reasoning_effort": PERMISSIBILITY_EFFORT,
        "rounds_per_problem": PERMISSIBILITY_ROUNDS,
    }
    atomic_json(status_path, status)
    sync_mount(STATE_ROOT)

    response_by_id = {item["problem_id"]: item for item in responses}
    semaphore = asyncio.Semaphore(PERMISSIBILITY_CONCURRENCY)
    results = await asyncio.gather(
        *[
            review_problem(
                codex,
                manifest,
                problem,
                response_by_id.get(problem["problem_id"], {}),
                run_dir,
                semaphore,
            )
            for problem in manifest["problems"]
        ]
    )
    ordered = sorted(results, key=lambda item: item["problem_index"])
    with (run_dir / "compliance.jsonl").open("w") as handle:
        for result in ordered:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    counts = {
        state: sum(result["status"] == state for result in ordered)
        for state in ("compliant", "non_compliant", "indeterminate")
    }
    status.update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "counts": counts,
        }
    )
    atomic_json(status_path, status)
    sync_mount(STATE_ROOT)
    return status


async def solve_problem(
    codex: AsyncCodex,
    manifest: dict[str, Any],
    problem: dict[str, Any],
    run_dir: Path,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    problem_id = problem["problem_id"]
    result_path = run_dir / "problems" / f"{problem['problem_index']:03d}_{problem_id}.json"
    existing: dict[str, Any] | None = None
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("status") in {"completed", "timed_out"}:
            return existing

    async with semaphore:
        started_at = utc_now()
        started_monotonic = time.monotonic()
        workspace = prepare_workspace(manifest["run_id"], problem)
        status = {
            "schema_version": 1,
            "run_id": manifest["run_id"],
            "problem_id": problem_id,
            "problem_index": problem["problem_index"],
            "status": "starting",
            "started_at": started_at,
            "model": manifest["model"],
            "reasoning_effort": manifest["reasoning_effort"],
            "attempt": int((existing or {}).get("attempt", 0)) + 1,
        }
        atomic_json(result_path, status)
        sync_mount(STATE_ROOT)

        thread = None
        turn = None
        try:
            previous_thread_id = (existing or {}).get("thread_id")
            if previous_thread_id:
                thread = await codex.thread_resume(
                    previous_thread_id,
                    approval_mode=ApprovalMode.deny_all,
                    cwd=str(workspace),
                    developer_instructions=problem["developer_instructions"],
                    model=manifest["model"],
                )
                turn_prompt = (
                    "The previous worker was interrupted. Continue the same problem from "
                    "your retained conversation, recreate any needed local files, and finish "
                    "with the complete proposed_solution implementation."
                )
                resumed = True
            else:
                thread = await codex.thread_start(
                    approval_mode=ApprovalMode.deny_all,
                    cwd=str(workspace),
                    developer_instructions=problem["developer_instructions"],
                    ephemeral=False,
                    model=manifest["model"],
                )
                turn_prompt = initial_turn_prompt(problem)
                resumed = False
            status.update(
                {
                    "status": "running",
                    "thread_id": thread.id,
                    "resumed_thread": resumed,
                }
            )
            atomic_json(result_path, status)
            sync_mount(STATE_ROOT)

            turn = await thread.turn(
                turn_prompt,
                approval_mode=ApprovalMode.deny_all,
                cwd=str(workspace),
                effort=manifest["reasoning_effort"],
                model=manifest["model"],
            )
            status["turn_id"] = turn.id
            atomic_json(result_path, status)
            sync_mount(STATE_ROOT)

            result = await asyncio.wait_for(
                turn.run(),
                timeout=manifest["problem_timeout_seconds"],
            )
            response = result.final_response or ""
            solution_path = workspace / "solution.py"
            if solution_path.is_file() and "def proposed_solution" not in response:
                response = (
                    response.rstrip()
                    + "\n\n```python\n"
                    + solution_path.read_text()
                    + "\n```\n"
                )
            status.update(
                {
                    "status": "completed",
                    "completed_at": utc_now(),
                    "duration_seconds": round(time.monotonic() - started_monotonic, 3),
                    "response": response,
                    "usage": serialize_usage(result.usage),
                    "items": serialize_items(result.items),
                }
            )
        except TimeoutError:
            if turn is not None:
                try:
                    await turn.interrupt()
                except Exception:
                    pass
            status.update(
                {
                    "status": "timed_out",
                    "completed_at": utc_now(),
                    "duration_seconds": round(time.monotonic() - started_monotonic, 3),
                    "response": "",
                    "error": "Three-hour agent timeout reached",
                }
            )
        except Exception as exc:
            status.update(
                {
                    "status": "failed",
                    "completed_at": utc_now(),
                    "duration_seconds": round(time.monotonic() - started_monotonic, 3),
                    "response": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            atomic_json(result_path, status)
            sync_mount(STATE_ROOT)
        print(
            json.dumps(
                {
                    "event": "problem_finished",
                    "problem_id": problem_id,
                    "status": status["status"],
                    "duration_seconds": status.get("duration_seconds"),
                }
            ),
            flush=True,
        )
        return status


def response_entries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "problem_id": result["problem_id"],
            "problem_index": result["problem_index"],
            "provider": "codex-chatgpt-subscription",
            "model": result["model"],
            "reasoning_effort": result["reasoning_effort"],
            "response": result.get("response", ""),
            "status": result["status"],
            "thread_id": result.get("thread_id"),
            "turn_id": result.get("turn_id"),
            "duration_seconds": result.get("duration_seconds"),
            "usage": result.get("usage"),
        }
        for result in sorted(results, key=lambda item: item["problem_index"])
    ]


def write_responses(run_dir: Path, responses: list[dict[str, Any]]) -> None:
    with (run_dir / "responses.jsonl").open("w") as handle:
        for response in responses:
            handle.write(json.dumps(response, sort_keys=True) + "\n")
    sync_mount(STATE_ROOT)


def load_responses(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "responses.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No completed responses exist at {path}")
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


async def run(args: argparse.Namespace) -> int:
    os.environ["CODEX_HOME"] = str(AUTH_ROOT)
    os.environ["CODEX_SQLITE_HOME"] = str(AUTH_ROOT / "state")
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("CODEX_API_KEY", None)
    os.environ["TMPDIR"] = str(WORK_ROOT / ".controller-tmp")
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    FORBIDDEN_ROOT.mkdir(parents=True, exist_ok=True)
    (FORBIDDEN_ROOT / "canary.txt").write_text("HORIZONMATH_FORBIDDEN_CANARY\n")
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    (STATE_ROOT / ".controller-canary").write_text(
        "HORIZONMATH_STATE_CANARY\n"
    )

    validate_runtime()
    run_dir = STATE_ROOT / "runs" / args.run_id
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    if manifest["run_id"] != args.run_id:
        raise ValueError("Run id does not match manifest")

    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    probe_workspace = WORK_ROOT / args.run_id / "_sandbox_probe"
    probe_workspace.mkdir(parents=True, exist_ok=True)

    codex_config = CodexConfig(
        config_overrides=(
            'web_search="disabled"',
            'approval_policy="never"',
            f'default_permissions="{PERMISSION_PROFILE}"',
        ),
        cwd=str(probe_workspace),
        env=os.environ.copy(),
        client_name="horizonmath_modal_agent_eval",
        client_title="HorizonMath Modal Agent Evaluation",
        client_version="1.0.0",
    )
    codex_bin = _resolve_codex_bin(codex_config)
    self_test = run_sandbox_self_test(codex_bin, probe_workspace)
    atomic_json(run_dir / "sandbox_self_test.json", self_test)

    if args.review_only:
        responses = load_responses(run_dir)
        async with AsyncCodex(config=codex_config) as codex:
            account = await codex.account(refresh_token=False)
            account_dump = account.model_dump(mode="json")
            if not account_dump.get("account"):
                raise RuntimeError(
                    "Codex app-server did not report a logged-in account"
                )
            print(
                f"HORIZONMATH_REVIEW_STARTED {args.run_id}",
                flush=True,
            )
            await review_batch(codex, manifest, responses, run_dir)
            print(
                f"HORIZONMATH_REVIEW_COMPLETED {args.run_id}",
                flush=True,
            )
        return 0

    runner_status = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "running",
        "started_at": utc_now(),
        "pid": os.getpid(),
        "problem_count": len(manifest["problems"]),
        "concurrency": manifest["concurrency"],
        "model": manifest["model"],
        "reasoning_effort": manifest["reasoning_effort"],
        "sandbox_self_test": self_test,
    }
    atomic_json(run_dir / "runner_status.json", runner_status)
    sync_mount(STATE_ROOT)

    try:
        async with AsyncCodex(config=codex_config) as codex:
            account = await codex.account(refresh_token=False)
            account_dump = account.model_dump(mode="json")
            if not account_dump.get("account"):
                raise RuntimeError("Codex app-server did not report a logged-in account")

            print(
                f"HORIZONMATH_WORKER_STARTED {args.run_id}",
                flush=True,
            )
            semaphore = asyncio.Semaphore(manifest["concurrency"])
            results = await asyncio.gather(
                *[
                    solve_problem(codex, manifest, problem, run_dir, semaphore)
                    for problem in manifest["problems"]
                ]
            )
            responses = response_entries(results)
            write_responses(run_dir, responses)
            runner_status["status"] = "reviewing"
            atomic_json(run_dir / "runner_status.json", runner_status)
            sync_mount(STATE_ROOT)
            compliance_status = await review_batch(
                codex,
                manifest,
                responses,
                run_dir,
            )
            print(
                f"HORIZONMATH_REVIEW_COMPLETED {args.run_id}",
                flush=True,
            )
    except Exception:
        runner_status.update(
            {
                "status": "failed",
                "completed_at": utc_now(),
                "error": traceback.format_exc(),
            }
        )
        atomic_json(run_dir / "runner_status.json", runner_status)
        sync_mount(STATE_ROOT)
        raise

    ordered = sorted(results, key=lambda item: item["problem_index"])
    runner_status.update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "counts": {
                state: sum(result["status"] == state for result in ordered)
                for state in ("completed", "failed", "timed_out")
            },
            "permissibility_counts": compliance_status["counts"],
        }
    )
    atomic_json(run_dir / "runner_status.json", runner_status)
    sync_mount(STATE_ROOT)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
