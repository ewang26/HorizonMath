"""Trusted scorer whose candidate execution happens in a networkless Modal Sandbox."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import modal

from agent_eval.config import APP_NAME
from agent_eval.modal_runner import (
    LOCAL_RESULTS_ROOT,
    REPO_ROOT,
    get_state_volume,
    read_volume_file,
    volume_file_exists,
)

CANDIDATE_RUNTIME_PATH = REPO_ROOT / "agent_eval" / "candidate_runtime"
CANDIDATE_REMOTE_ROOT = Path("/opt/horizonmath_candidate_runtime")


def candidate_payload(
    code: str,
    *,
    precision_dps: int,
    return_json: bool,
    test_points: list[dict] | None,
) -> dict[str, Any]:
    """Build the only data sent to untrusted execution, excluding ground truth."""

    points = None
    if test_points is not None:
        points = [{"args": list(point.get("args", []))} for point in test_points]
    return {
        "code": code,
        "precision_dps": precision_dps,
        "return_json": return_json,
        "points": points,
    }


def candidate_image() -> modal.Image:
    """Image intentionally contains no benchmark data, answers, or validators."""

    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "mpmath==1.3.0",
            "networkx==3.6.1",
            "numpy==2.4.1",
            "scipy==1.17.0",
            "sympy==1.14.0",
        )
        .add_local_dir(
            CANDIDATE_RUNTIME_PATH,
            str(CANDIDATE_REMOTE_ROOT),
            copy=True,
        )
        .env(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
    )


class ModalCandidateExecutor:
    """Adapter matching scripts.evaluator.execute_sandboxed."""

    def __init__(self, app: modal.App) -> None:
        self.app = app
        self.image = candidate_image()

    def __call__(
        self,
        code: str,
        timeout: int = 300,
        precision_dps: int = 110,
        return_json: bool = False,
        test_points: list[dict] | None = None,
    ):
        scripts_dir = REPO_ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from evaluator.sandbox import ExecutionResult, ExecutionStatus

        # Ground truth never crosses the Sandbox boundary.
        payload = candidate_payload(
            code,
            precision_dps=precision_dps,
            return_json=return_json,
            test_points=test_points,
        )

        started = time.monotonic()
        sandbox = modal.Sandbox.create(
            "python",
            str(CANDIDATE_REMOTE_ROOT / "runner.py"),
            app=self.app,
            image=self.image,
            timeout=timeout + 60,
            cpu=(0.5, 2.0),
            memory=(1024, 4096),
            block_network=True,
        )
        try:
            sandbox.stdin.write(json.dumps(payload))
            sandbox.stdin.write_eof()
            sandbox.wait(raise_on_termination=False)
            stdout = sandbox.stdout.read()
            stderr = sandbox.stderr.read()
        except Exception as exc:
            try:
                sandbox.terminate()
            except Exception:
                pass
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                error_message=f"Modal candidate Sandbox failed: {type(exc).__name__}: {exc}",
                execution_time_ms=int((time.monotonic() - started) * 1000),
            )

        parsed = None
        for line in reversed(stdout.splitlines()):
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        execution_time_ms = int((time.monotonic() - started) * 1000)
        if not isinstance(parsed, dict):
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                error_message=(
                    "Candidate Sandbox returned no result JSON. "
                    f"stderr: {stderr[-2000:]}"
                ),
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
                execution_time_ms=parsed.get("execution_time_ms", execution_time_ms),
            )

        output = parsed.get("output")
        if test_points is not None or return_json:
            output = json.dumps(output)
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            output=str(output),
            execution_time_ms=parsed.get("execution_time_ms", execution_time_ms),
        )


def load_remote_responses(run_id: str) -> list[dict[str, Any]]:
    state_volume = get_state_volume()
    remote_path = f"/runs/{run_id}/responses.jsonl"
    if not volume_file_exists(state_volume, remote_path):
        raise FileNotFoundError(
            f"Run {run_id} has no completed responses.jsonl yet"
        )
    return [
        json.loads(line)
        for line in read_volume_file(state_volume, remote_path).decode().splitlines()
        if line.strip()
    ]


def score_run(args: argparse.Namespace) -> int:
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from baseline_comparator import load_baselines
    from evaluate import load_problems
    from evaluate_responses import evaluate_response

    problems = load_problems(REPO_ROOT / "data" / "problems_full.json")
    problem_by_id = {
        problem["id"]: (index, problem)
        for index, problem in enumerate(problems)
    }
    baselines = load_baselines(REPO_ROOT / "data" / "baselines.json")

    destination = LOCAL_RESULTS_ROOT / f"modal_{args.run_id}"
    destination.mkdir(parents=True, exist_ok=True)

    with modal.enable_output():
        app = modal.App.lookup(APP_NAME, create_if_missing=True)
        responses = load_remote_responses(args.run_id)
        executor = ModalCandidateExecutor(app)
        evaluations = []
        for response_entry in responses:
            problem_id = response_entry["problem_id"]
            problem_index, problem = problem_by_id[problem_id]
            evaluation = evaluate_response(
                problem,
                problem_index,
                response_entry.get("response", ""),
                baselines,
                executor=executor,
            )
            evaluations.append(evaluation)
            print(
                json.dumps(
                    {
                        "problem_id": problem_id,
                        "mode": evaluation.get("mode"),
                        "passed": evaluation.get(
                            "success",
                            evaluation.get("valid", False),
                        ),
                    }
                ),
                flush=True,
            )

    output_path = destination / "evaluations.jsonl"
    with output_path.open("w") as handle:
        for evaluation in evaluations:
            handle.write(json.dumps(evaluation, sort_keys=True, default=str) + "\n")
    summary = {
        "run_id": args.run_id,
        "evaluated": len(evaluations),
        "passed": sum(
            bool(item.get("success", item.get("valid", False)))
            for item in evaluations
        ),
        "candidate_execution": "networkless-modal-sandbox",
        "evaluations_path": str(output_path),
    }
    (destination / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    return score_run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
