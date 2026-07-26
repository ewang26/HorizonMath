"""Trusted local launcher for verifier-free Codex agent batches on Modal."""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import modal
from modal.types import FileEntryType

from agent_eval.config import (
    APP_NAME,
    DEFAULT_CONCURRENCY,
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    OPENAI_EGRESS_ALLOWLIST,
    PROBLEM_TIMEOUT_SECONDS,
    REMOTE_AUTH_ROOT,
    REMOTE_RUNTIME_ROOT,
    REMOTE_STATE_ROOT,
    STATE_VOLUME_NAME,
    sandbox_timeout_seconds,
)
from agent_eval.manifest import (
    build_manifest,
    load_problems,
    manifest_bytes,
    select_indices,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBLEMS_PATH = REPO_ROOT / "data" / "problems_full.json"
RUNTIME_PATH = REPO_ROOT / "agent_eval" / "runtime"
LOCAL_RESULTS_ROOT = REPO_ROOT / "results"

CODEX_SDK_VERSION = "0.144.4"


def run_id_now() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"codex_sol_xhigh_{stamp}"


def developer_instructions() -> dict[str, str]:
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from benchmark_prompts import SYSTEM_MESSAGES

    addendum = (
        "\n\nThis is a long-horizon coding-agent evaluation. You have a hard three-hour "
        "wall-clock limit. Use local computational tools extensively, but do not attempt "
        "internet access or access outside the assigned workspace. Finish with your strongest "
        "concrete proposed_solution implementation."
    )
    return {mode: text + addendum for mode, text in SYSTEM_MESSAGES.items()}


def agent_image() -> modal.Image:
    """Build an image that contains tools and the safe runtime, never the benchmark repo."""

    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install(
            "bash",
            "build-essential",
            "ca-certificates",
            "git",
            "jq",
            "libseccomp-dev",
            "ripgrep",
        )
        .pip_install(
            f"openai-codex=={CODEX_SDK_VERSION}",
            "mpmath==1.3.0",
            "networkx==3.6.1",
            "numpy==2.4.1",
            "scipy==1.17.0",
            "sympy==1.14.0",
        )
        .add_local_dir(
            RUNTIME_PATH,
            str(REMOTE_RUNTIME_ROOT),
            copy=True,
        )
        .run_commands(
            (
                "gcc -O2 -Wall -Wextra "
                f"{REMOTE_RUNTIME_ROOT / 'sandbox_shell.c'} "
                "-lseccomp -o /tmp/horizonmath-sandbox-shell"
            ),
            (
                "bash -c '"
                "mv /bin/bash /bin/bash.real && "
                "mv /bin/sh /bin/sh.real && "
                "cp /tmp/horizonmath-sandbox-shell /bin/bash && "
                "cp /tmp/horizonmath-sandbox-shell /bin/sh"
                "'"
            ),
        )
        .env(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
    )


def get_state_volume() -> modal.Volume:
    return modal.Volume.from_name(
        STATE_VOLUME_NAME,
        create_if_missing=True,
        version=2,
    )


def volume_file_exists(volume: modal.Volume, remote_path: str) -> bool:
    try:
        return bool(volume.listdir(remote_path))
    except Exception as exc:
        if "not found" in str(exc).lower() or "no such" in str(exc).lower():
            return False
        raise


def upload_manifest(
    state_volume: modal.Volume,
    manifest: dict[str, Any],
) -> str:
    remote_path = f"/runs/{manifest['run_id']}/manifest.json"
    payload = io.BytesIO(manifest_bytes(manifest))
    with state_volume.batch_upload(force=True) as batch:
        batch.put_file(payload, remote_path, mode=0o600)
    return remote_path


def write_local_launch(run_id: str, payload: dict[str, Any]) -> Path:
    run_dir = LOCAL_RESULTS_ROOT / f"modal_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "launch.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def read_volume_file(volume: modal.Volume, remote_path: str) -> bytes:
    return b"".join(volume.read_file(remote_path))


def stream_until_marker(
    sandbox: modal.Sandbox,
    marker: str,
) -> str:
    """Relay interactive device-auth output until the remote worker is ready."""

    captured: list[str] = []
    for chunk in sandbox.stdout:
        captured.append(chunk)
        print(chunk, end="", flush=True)
        if marker in chunk:
            return "".join(captured)
    stderr = sandbox.stderr.read()
    raise RuntimeError(
        f"Modal Sandbox exited before {marker!r}. "
        f"stdout={''.join(captured)[-4000:]!r} stderr={stderr[-4000:]!r}"
    )


def launch(args: argparse.Namespace) -> int:
    problems = load_problems(PROBLEMS_PATH)
    indices = select_indices(
        len(problems),
        start=args.start,
        count=args.count,
        indices=args.indices,
    )
    run_id = args.run_id or run_id_now()
    manifest = build_manifest(
        problems,
        indices,
        run_id=run_id,
        developer_instructions_by_mode=developer_instructions(),
        model=DEFAULT_MODEL,
        effort=DEFAULT_EFFORT,
        concurrency=args.concurrency,
        timeout_seconds=PROBLEM_TIMEOUT_SECONDS,
    )
    outer_timeout = sandbox_timeout_seconds(
        len(indices),
        args.concurrency,
        PROBLEM_TIMEOUT_SECONDS,
    )

    with modal.enable_output():
        app = modal.App.lookup(APP_NAME, create_if_missing=True)
        state_volume = get_state_volume()
        manifest_path = upload_manifest(state_volume, manifest)
        sandbox = modal.Sandbox.create(
            "python",
            str(REMOTE_RUNTIME_ROOT / "entrypoint.py"),
            "run",
            "--run-id",
            run_id,
            app=app,
            name=f"horizonmath-{run_id}"[:64],
            tags={
                "run_id": run_id,
                "model": DEFAULT_MODEL,
                "effort": DEFAULT_EFFORT,
            },
            image=agent_image(),
            env={
                "CODEX_HOME": str(REMOTE_AUTH_ROOT),
                "CODEX_SQLITE_HOME": str(REMOTE_AUTH_ROOT / "state"),
                "PYTHONUNBUFFERED": "1",
            },
            timeout=outer_timeout,
            cpu=(2.0, 8.0),
            memory=(4096, 16384),
            outbound_domain_allowlist=list(OPENAI_EGRESS_ALLOWLIST),
            volumes={
                str(REMOTE_STATE_ROOT): state_volume,
            },
            pty=True,
        )
        if sandbox.poll() is not None:
            raise RuntimeError("Modal Sandbox exited during startup")

    launch_record = {
        "schema_version": 2,
        "run_id": run_id,
        "launched_at": datetime.now(UTC).isoformat(),
        "status": "awaiting_device_auth",
        "sandbox_id": sandbox.object_id,
        "app_name": APP_NAME,
        "auth_mode": "ephemeral_device_auth",
        "auth_persisted": False,
        "state_volume": STATE_VOLUME_NAME,
        "manifest_path": manifest_path,
        "problem_indices": indices,
        "problem_ids": [problems[index]["id"] for index in indices],
        "model": DEFAULT_MODEL,
        "reasoning_effort": DEFAULT_EFFORT,
        "problem_timeout_seconds": PROBLEM_TIMEOUT_SECONDS,
        "concurrency": args.concurrency,
        "outer_sandbox_timeout_seconds": outer_timeout,
        "egress_allowlist": list(OPENAI_EGRESS_ALLOWLIST),
    }
    launch_path = write_local_launch(run_id, launch_record)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "sandbox_id": sandbox.object_id,
                "status": "awaiting_device_auth",
                "auth_mode": "ephemeral_device_auth",
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Local launch record: {launch_path}")
    try:
        stream_until_marker(sandbox, "HORIZONMATH_WORKER_STARTED")
    except BaseException:
        sandbox.terminate(wait=False)
        raise
    launch_record["status"] = "running"
    launch_record["worker_started_at"] = datetime.now(UTC).isoformat()
    write_local_launch(run_id, launch_record)
    sandbox.detach()
    print(json.dumps(launch_record, indent=2, sort_keys=True))
    return 0


def preflight(args: argparse.Namespace) -> int:
    with modal.enable_output():
        app = modal.App.lookup(APP_NAME, create_if_missing=True)
        sandbox = modal.Sandbox.create(
            "python",
            str(REMOTE_RUNTIME_ROOT / "entrypoint.py"),
            "preflight",
            app=app,
            name="horizonmath-codex-preflight",
            tags={"purpose": "preflight"},
            image=agent_image(),
            env={
                "CODEX_HOME": str(REMOTE_AUTH_ROOT),
                "CODEX_SQLITE_HOME": str(REMOTE_AUTH_ROOT / "state"),
                "PYTHONUNBUFFERED": "1",
            },
            timeout=15 * 60,
            cpu=1.0,
            memory=2048,
            outbound_domain_allowlist=list(OPENAI_EGRESS_ALLOWLIST),
            pty=True,
        )
        try:
            stdout = stream_until_marker(sandbox, "HORIZONMATH_PREFLIGHT_OK")
            sandbox.wait(raise_on_termination=False)
        except BaseException:
            sandbox.terminate(wait=False)
            raise

    report = None
    prefix = "HORIZONMATH_PREFLIGHT_REPORT "
    for line in stdout.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            report = json.loads(line.removeprefix(prefix))
        except json.JSONDecodeError:
            continue
    if not isinstance(report, dict) or not report.get("ok"):
        raise RuntimeError(
            "Remote preflight failed. "
            f"stdout={stdout[-4000:]!r}"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def find_local_launch(run_id: str) -> dict[str, Any] | None:
    path = LOCAL_RESULTS_ROOT / f"modal_{run_id}" / "launch.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def status(args: argparse.Namespace) -> int:
    launch_record = find_local_launch(args.run_id)
    sandbox_id = args.sandbox_id or (
        launch_record.get("sandbox_id") if launch_record else None
    )
    with modal.enable_output():
        state_volume = get_state_volume()
        result: dict[str, Any] = {"run_id": args.run_id}
        for name in (
            "preflight.json",
            "runner_status.json",
            "sandbox_self_test.json",
        ):
            remote_path = f"/runs/{args.run_id}/{name}"
            if volume_file_exists(state_volume, remote_path):
                result[name.removesuffix(".json")] = json.loads(
                    read_volume_file(state_volume, remote_path)
                )
        try:
            problem_entries = state_volume.listdir(
                f"/runs/{args.run_id}/problems",
                recursive=True,
            )
        except Exception as exc:
            if "not found" in str(exc).lower() or "no such" in str(exc).lower():
                problem_entries = []
            else:
                raise
        problem_statuses = []
        for entry in problem_entries:
            if entry.path.endswith(".json"):
                problem_statuses.append(
                    json.loads(read_volume_file(state_volume, entry.path))
                )
        result["problems"] = sorted(
            problem_statuses,
            key=lambda item: item["problem_index"],
        )
        if sandbox_id:
            sandbox = modal.Sandbox.from_id(sandbox_id)
            result["sandbox_id"] = sandbox_id
            result["sandbox_exit_code"] = sandbox.poll()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def download(args: argparse.Namespace) -> int:
    destination = (
        Path(args.destination)
        if args.destination
        else LOCAL_RESULTS_ROOT / f"modal_{args.run_id}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    with modal.enable_output():
        state_volume = get_state_volume()
        prefix = f"/runs/{args.run_id}"
        entries = state_volume.listdir(prefix, recursive=True)
        downloaded = []
        for entry in entries:
            if entry.type != FileEntryType.FILE:
                continue
            relative = PurePosixPath(entry.path).relative_to(PurePosixPath(prefix))
            local_path = destination / Path(*relative.parts)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(read_volume_file(state_volume, entry.path))
            downloaded.append(str(local_path))
    print(json.dumps({"run_id": args.run_id, "downloaded": downloaded}, indent=2))
    return 0


def parse_indices(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run verifier-free Codex coding-agent evaluation on Modal"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--run-id")
    launch_parser.add_argument("--start", type=int, default=0)
    launch_parser.add_argument("--count", type=int)
    launch_parser.add_argument(
        "--indices",
        type=parse_indices,
        help="Comma-separated zero-based problem indices",
    )
    launch_parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
    )
    launch_parser.set_defaults(func=launch)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.set_defaults(func=preflight)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--run-id", required=True)
    status_parser.add_argument("--sandbox-id")
    status_parser.set_defaults(func=status)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--run-id", required=True)
    download_parser.add_argument("--destination")
    download_parser.set_defaults(func=download)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "launch" and args.count is None and args.indices is None:
        parser.error("launch requires --count or --indices")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
