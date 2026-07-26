"""Interactive, ephemeral ChatGPT login followed by preflight and evaluation."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from openai_codex import CodexConfig
from openai_codex.client import _resolve_codex_bin

AUTH_ROOT = Path("/codex-home")
RUNTIME_ROOT = Path("/opt/horizonmath_agent_runtime")


def clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in tuple(env):
        upper = name.upper()
        if (
            "API_KEY" in upper
            or "ACCESS_TOKEN" in upper
            or name in {"CODEX_API_KEY", "OPENAI_API_KEY", "CODEX_ACCESS_TOKEN"}
        ):
            env.pop(name, None)
    env["CODEX_HOME"] = str(AUTH_ROOT)
    env["CODEX_SQLITE_HOME"] = str(AUTH_ROOT / "state")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def prepare_ephemeral_codex_home() -> None:
    """Create a fresh local Codex home; it is deliberately not a Volume mount."""

    AUTH_ROOT.mkdir(mode=0o700, parents=True, exist_ok=False)
    shutil.copyfile(RUNTIME_ROOT / "config.toml", AUTH_ROOT / "config.toml")
    (AUTH_ROOT / "config.toml").chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "preflight"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.mode == "run" and not args.run_id:
        parser.error("run requires --run-id")

    prepare_ephemeral_codex_home()
    env = clean_environment()
    print(
        "HORIZONMATH_DEVICE_AUTH_BEGIN\n"
        "Authorize this one-time OpenAI device login. The resulting Codex session "
        "exists only inside this ephemeral Modal Sandbox.",
        flush=True,
    )
    codex_bin = _resolve_codex_bin(
        CodexConfig(
            cwd="/",
            env=env,
            client_name="horizonmath_modal_device_auth",
            client_title="HorizonMath Modal Device Auth",
            client_version="1.0.0",
        )
    )
    subprocess.run(
        [str(codex_bin), "login", "--device-auth"],
        check=True,
        env=env,
    )
    print("HORIZONMATH_DEVICE_AUTH_OK", flush=True)

    preflight_command = [
        sys.executable,
        str(RUNTIME_ROOT / "preflight.py"),
    ]
    if args.run_id:
        preflight_command.extend(["--run-id", args.run_id])
    subprocess.run(preflight_command, check=True, env=env)
    print("HORIZONMATH_PREFLIGHT_OK", flush=True)

    if args.mode == "preflight":
        return 0

    os.execve(
        sys.executable,
        [
            sys.executable,
            str(RUNTIME_ROOT / "worker.py"),
            "--run-id",
            args.run_id,
        ],
        env,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
