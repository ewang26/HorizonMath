"""Deployed, verifier-bearing Modal scorer kept separate from coding agents."""

from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from agent_eval.config import APP_NAME, STATE_VOLUME_NAME

LOCAL_REPO_ROOT = Path(__file__).resolve().parents[1]
TRUSTED_ROOT = Path("/opt/horizonmath_trusted")
CANDIDATE_ROOT = Path("/opt/horizonmath_candidate_runtime")
REMOTE_STATE_ROOT = Path("/state")
SCORER_APP_NAME = f"{APP_NAME}-trusted-scorer"
SCORER_CONCURRENCY = 8
SCORER_WATCH_TIMEOUT_SECONDS = 24 * 60 * 60
SCORER_POLL_SECONDS = 30

app = modal.App()
state_volume = modal.Volume.from_name(
    STATE_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)

candidate_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "mpmath==1.3.0",
        "networkx==3.6.1",
        "numpy==2.4.1",
        "scipy==1.17.0",
        "sympy==1.14.0",
    )
    .add_local_dir(
        LOCAL_REPO_ROOT / "agent_eval" / "candidate_runtime",
        str(CANDIDATE_ROOT),
        copy=True,
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
)

trusted_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgmp-dev", "libmpfr-dev")
    .pip_install(
        "cysignals==1.12.6",
        "fpylll==0.6.4",
        "google-genai==1.60.0",
        "mpmath==1.3.0",
        "networkx==3.6.1",
        "numpy==2.4.1",
        "openai==2.15.0",
        "python-dotenv==1.2.1",
        "scipy==1.17.0",
        "sympy==1.14.0",
    )
    .add_local_dir(
        LOCAL_REPO_ROOT / "scripts",
        str(TRUSTED_ROOT / "scripts"),
        copy=True,
    )
    .add_local_dir(
        LOCAL_REPO_ROOT / "data",
        str(TRUSTED_ROOT / "data"),
        copy=True,
    )
    .add_local_dir(
        LOCAL_REPO_ROOT / "validators",
        str(TRUSTED_ROOT / "validators"),
        copy=True,
    )
    .add_local_dir(
        LOCAL_REPO_ROOT / "numerics",
        str(TRUSTED_ROOT / "numerics"),
        copy=True,
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


@app.function(
    image=candidate_image,
    timeout=6 * 60,
    cpu=(0.5, 2.0),
    memory=(1024, 4096),
    block_network=True,
    restrict_modal_access=True,
    max_containers=SCORER_CONCURRENCY,
    single_use_containers=True,
)
def execute_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute candidate code without benchmark data, validators, volumes, or network."""

    if str(CANDIDATE_ROOT) not in sys.path:
        sys.path.insert(0, str(CANDIDATE_ROOT))
    from runner import execute

    return execute(payload)


class RemoteCandidateExecutor:
    """Adapter from the benchmark evaluator to the isolated Modal function."""

    def __call__(
        self,
        code: str,
        timeout: int = 300,
        precision_dps: int = 110,
        return_json: bool = False,
        test_points: list[dict] | None = None,
    ):
        scripts_dir = TRUSTED_ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from evaluator.sandbox import ExecutionResult, ExecutionStatus

        points = None
        if test_points is not None:
            points = [
                {"args": list(point.get("args", []))}
                for point in test_points
            ]
        payload = {
            "code": code,
            "precision_dps": precision_dps,
            "return_json": return_json,
            "points": points,
        }
        started = time.monotonic()
        try:
            parsed = execute_candidate.remote(payload)
        except Exception as exc:
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                error_message=(
                    "Remote candidate execution failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                execution_time_ms=int((time.monotonic() - started) * 1000),
            )

        execution_time_ms = int((time.monotonic() - started) * 1000)
        if not isinstance(parsed, dict):
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                error_message="Remote candidate function returned no result object",
                execution_time_ms=execution_time_ms,
            )
        if parsed.get("status") != "success":
            error = parsed.get("error", "Unknown candidate error")
            status = (
                ExecutionStatus.SYNTAX_ERROR
                if "SyntaxError" in error
                else ExecutionStatus.RUNTIME_ERROR
            )
            return ExecutionResult(
                status=status,
                error_message=error,
                execution_time_ms=parsed.get(
                    "execution_time_ms",
                    execution_time_ms,
                ),
            )
        output = parsed.get("output")
        if test_points is not None or return_json:
            output = json.dumps(output)
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            output=str(output),
            execution_time_ms=parsed.get(
                "execution_time_ms",
                execution_time_ms,
            ),
        )


@app.function(
    image=trusted_image,
    timeout=20 * 60,
    cpu=(1.0, 4.0),
    memory=(2048, 8192),
    max_containers=SCORER_CONCURRENCY,
    single_use_containers=True,
)
def score_problem(
    response_entry: dict[str, Any],
    precomputed_compliance: dict[str, Any] | None,
) -> dict[str, Any]:
    """Score one response with validators that never enter the candidate container."""

    scripts_dir = TRUSTED_ROOT / "scripts"
    project_root = TRUSTED_ROOT
    for path in (scripts_dir, project_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from baseline_comparator import load_baselines
    from evaluate import load_problems
    from evaluate_responses import evaluate_response

    problems = load_problems(TRUSTED_ROOT / "data" / "problems_full.json")
    problem_by_id = {
        problem["id"]: (index, problem)
        for index, problem in enumerate(problems)
    }
    baselines = load_baselines(TRUSTED_ROOT / "data" / "baselines.json")
    problem_id = response_entry["problem_id"]
    problem_index, problem = problem_by_id[problem_id]
    return evaluate_response(
        problem,
        problem_index,
        response_entry.get("response", ""),
        baselines,
        executor=RemoteCandidateExecutor(),
        precomputed_compliance=precomputed_compliance,
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def evaluation_passed(evaluation: dict[str, Any]) -> bool:
    mode = evaluation.get("mode")
    if mode == "numeric":
        return bool(evaluation.get("success"))
    if mode == "construction":
        return bool(evaluation.get("valid"))
    if mode == "benchmark":
        if not evaluation.get("valid"):
            return False
        comparison = evaluation.get("baseline_comparison") or {}
        return comparison.get("result", "no_baseline") in {
            "beats_baseline",
            "no_baseline",
        }
    return False


@app.function(
    image=trusted_image,
    volumes={str(REMOTE_STATE_ROOT): state_volume},
    timeout=SCORER_WATCH_TIMEOUT_SECONDS,
    cpu=1.0,
    memory=2048,
)
def watch_and_score(run_id: str) -> dict[str, Any]:
    """Wait for generation/review completion, then run trusted scoring automatically."""

    run_dir = REMOTE_STATE_ROOT / "runs" / run_id
    status_path = run_dir / "scoring_status.json"
    status = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "waiting_for_generation",
        "started_at": utc_now(),
        "candidate_execution": "networkless-modal-function",
        "concurrency": SCORER_CONCURRENCY,
    }
    atomic_json(status_path, status)
    state_volume.commit()

    deadline = time.monotonic() + SCORER_WATCH_TIMEOUT_SECONDS - 5 * 60
    while True:
        state_volume.reload()
        runner_path = run_dir / "runner_status.json"
        responses_path = run_dir / "responses.jsonl"
        runner = read_json(runner_path) if runner_path.is_file() else {}
        runner_state = runner.get("status")
        if runner_state == "completed" and responses_path.is_file():
            break
        if runner_state in {"cancelled", "failed"}:
            status.update(
                {
                    "status": runner_state,
                    "completed_at": utc_now(),
                    "reason": (
                        "Generation did not complete; trusted scoring was not run."
                    ),
                }
            )
            atomic_json(status_path, status)
            state_volume.commit()
            return status
        if time.monotonic() >= deadline:
            status.update(
                {
                    "status": "timed_out",
                    "completed_at": utc_now(),
                    "reason": "Timed out waiting for generation and review completion.",
                }
            )
            atomic_json(status_path, status)
            state_volume.commit()
            return status
        time.sleep(SCORER_POLL_SECONDS)

    status["status"] = "scoring"
    status["scoring_started_at"] = utc_now()
    atomic_json(status_path, status)
    state_volume.commit()

    try:
        responses = read_jsonl(run_dir / "responses.jsonl")
        compliance_path = run_dir / "compliance.jsonl"
        permissibility = {
            item["problem_id"]: item
            for item in (
                read_jsonl(compliance_path)
                if compliance_path.is_file()
                else []
            )
        }
        inputs = [
            (response, permissibility.get(response["problem_id"]))
            for response in responses
        ]
        evaluations = list(
            score_problem.starmap(
                inputs,
                order_outputs=False,
                return_exceptions=False,
            )
        )
        evaluations.sort(key=lambda item: item["problem_index"])
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "completed_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        atomic_json(status_path, status)
        state_volume.commit()
        raise

    with (run_dir / "evaluations.jsonl").open("w") as handle:
        for evaluation in evaluations:
            handle.write(
                json.dumps(evaluation, sort_keys=True, default=str) + "\n"
            )
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "evaluated": len(evaluations),
        "passed": sum(evaluation_passed(item) for item in evaluations),
        "candidate_execution": "networkless-modal-function",
        "permissibility_source": "subscription-terra-modal",
        "completed_at": utc_now(),
    }
    atomic_json(run_dir / "evaluation_summary.json", summary)
    status.update(
        {
            "status": "completed",
            "completed_at": summary["completed_at"],
            "evaluated": summary["evaluated"],
            "passed": summary["passed"],
        }
    )
    atomic_json(status_path, status)
    state_volume.commit()
    return status


def deploy_and_spawn(run_id: str) -> str:
    """Deploy the trusted scorer app and start its detached watcher."""

    app.deploy(name=SCORER_APP_NAME)
    call = watch_and_score.spawn(run_id)
    return call.object_id
