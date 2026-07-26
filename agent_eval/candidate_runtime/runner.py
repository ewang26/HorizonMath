"""Execute one candidate payload without verifiers, answers, volumes, or network."""

from __future__ import annotations

import json
import sys
import time
import traceback
from typing import Any


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    code = payload["code"]
    precision_dps = int(payload.get("precision_dps", 110))
    return_json = bool(payload.get("return_json", False))
    points = payload.get("points")

    from mpmath import mp

    mp.dps = precision_dps
    namespace: dict[str, Any] = {"__name__": "candidate_solution"}
    try:
        exec(compile(code, "<candidate>", "exec"), namespace, namespace)
        proposed_solution = namespace["proposed_solution"]
        if points is not None:
            outputs = []
            for point in points:
                args = [mp.mpf(arg) if isinstance(arg, str) else arg for arg in point["args"]]
                outputs.append(str(proposed_solution(*args)))
            output: Any = outputs
        else:
            value = proposed_solution()
            output = value if return_json else str(value)
            if return_json:
                json.dumps(output)
        return {
            "status": "success",
            "output": output,
            "execution_time_ms": int((time.monotonic() - started) * 1000),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
            "execution_time_ms": int((time.monotonic() - started) * 1000),
        }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        result = execute(payload)
    except Exception as exc:
        result = {
            "status": "runner_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
