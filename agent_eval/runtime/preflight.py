"""Remote preflight for auth, model availability, and nested sandbox enforcement."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from openai_codex import AsyncCodex, CodexConfig
from openai_codex.client import _resolve_codex_bin
from worker import (
    AUTH_ROOT,
    PERMISSION_PROFILE,
    STATE_ROOT,
    WORK_ROOT,
    atomic_json,
    run_sandbox_self_test,
    sync_mount,
    validate_runtime,
)


async def preflight(run_id: str | None = None) -> dict:
    os.environ["CODEX_HOME"] = str(AUTH_ROOT)
    os.environ["CODEX_SQLITE_HOME"] = str(AUTH_ROOT / "state")
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("CODEX_API_KEY", None)
    workspace = WORK_ROOT / "_preflight"
    workspace.mkdir(parents=True, exist_ok=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    (STATE_ROOT / ".controller-canary").write_text(
        "HORIZONMATH_STATE_CANARY\n"
    )

    validate_runtime()
    config = CodexConfig(
        config_overrides=(
            'web_search="disabled"',
            'approval_policy="never"',
            f'default_permissions="{PERMISSION_PROFILE}"',
        ),
        cwd=str(workspace),
        env=os.environ.copy(),
        client_name="horizonmath_modal_agent_eval_preflight",
        client_title="HorizonMath Modal Agent Evaluation Preflight",
        client_version="1.0.0",
    )
    codex_bin = _resolve_codex_bin(config)
    sandbox_report = run_sandbox_self_test(codex_bin, workspace)

    async with AsyncCodex(config=config) as codex:
        account = (await codex.account(refresh_token=False)).model_dump(mode="json")
        models = (await codex.models(include_hidden=True)).model_dump(mode="json")

    model_json = json.dumps(models)
    report = {
        "ok": True,
        "chatgpt_account_present": bool(account.get("account")),
        "gpt_5_6_sol_available": "gpt-5.6-sol" in model_json,
        "gpt_5_6_terra_available": "gpt-5.6-terra" in model_json,
        "high_advertised": "high" in model_json,
        "xhigh_advertised": "xhigh" in model_json,
        "sandbox_self_test": sandbox_report,
        "api_key_present": bool(
            os.getenv("OPENAI_API_KEY") or os.getenv("CODEX_API_KEY")
        ),
    }
    if not all(
        (
            report["chatgpt_account_present"],
            report["gpt_5_6_sol_available"],
            report["gpt_5_6_terra_available"],
            report["high_advertised"],
            report["xhigh_advertised"],
            report["sandbox_self_test"]["ok"],
            not report["api_key_present"],
        )
    ):
        raise RuntimeError(f"Preflight invariant failed: {report}")
    if run_id:
        atomic_json(STATE_ROOT / "runs" / run_id / "preflight.json", report)
        sync_mount(STATE_ROOT)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        report = asyncio.run(preflight(args.run_id))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        "HORIZONMATH_PREFLIGHT_REPORT "
        + json.dumps(report, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
