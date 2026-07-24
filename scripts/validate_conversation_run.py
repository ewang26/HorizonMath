#!/usr/bin/env python3
"""Validate cloud-runtime provenance before evaluating a conversation run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from conversation_benchmark import (
    ConversationRunError,
    load_jsonl_strict,
    validate_conversation_cloud_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fail unless every scored response is proven to come from an "
            "OpenAI-hosted Codex cloud container"
        )
    )
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    try:
        config = json.loads((results_dir / "config.json").read_text())
        responses = load_jsonl_strict(results_dir / "responses.jsonl")
        errors_path = results_dir / "generation_errors.jsonl"
        generation_errors = (
            load_jsonl_strict(errors_path) if errors_path.exists() else []
        )
        validate_conversation_cloud_run(
            results_dir=results_dir,
            config=config,
            responses=responses,
            generation_errors=generation_errors,
        )
    except (OSError, json.JSONDecodeError, ConversationRunError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        "VALID: every scored response has matching OpenAI-hosted Codex cloud "
        "runtime evidence"
    )


if __name__ == "__main__":
    main()
