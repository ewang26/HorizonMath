#!/usr/bin/env python3
"""Quick connectivity test for the Anthropic (Claude) API.

Loads ANTHROPIC_API_KEY from the project-root .env (same as run_benchmark.py),
makes one cheap request, and reports success or a clear error.

Usage:
    uv run scripts/test_anthropic_api.py                       # default: claude-haiku-4-5
    uv run scripts/test_anthropic_api.py --model claude-fable-5
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")

import anthropic

# Cheap, fast model for a default connectivity check.
DEFAULT_TEST_MODEL = "claude-haiku-4-5"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        default=DEFAULT_TEST_MODEL,
        help=f"Claude model to test (default: {DEFAULT_TEST_MODEL}). Try claude-fable-5 for the most capable model.",
    )
    args = parser.parse_args()
    test_model = args.model

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("✗ ANTHROPIC_API_KEY is not set (add it to .env). Aborting.", file=sys.stderr)
        return 1

    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=test_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Reply with exactly: API OK"}],
        )
    except anthropic.AuthenticationError:
        print("✗ Authentication failed — the API key is invalid or revoked.", file=sys.stderr)
        return 1
    except anthropic.PermissionDeniedError:
        print(f"✗ Permission denied — key lacks access to {test_model}.", file=sys.stderr)
        return 1
    except anthropic.NotFoundError:
        print(f"✗ Model not found: {test_model!r}. Check the model ID.", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as e:
        # Fable 5 requires >=30-day data retention; a ZDR org gets a 400 here.
        print(f"✗ API error {e.status_code}: {e.message}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError:
        print("✗ Connection error — check your network.", file=sys.stderr)
        return 1

    if response.stop_reason == "refusal":
        # Fable 5 safety classifiers can decline a request (HTTP 200).
        print(f"✗ Request was refused by safety classifiers (model={response.model}).", file=sys.stderr)
        return 1

    text = next((b.text for b in response.content if b.type == "text"), "")
    print("✓ Anthropic API works.")
    print(f"  Model:    {response.model}")
    print(f"  Reply:    {text.strip()!r}")
    print(f"  Tokens:   in={response.usage.input_tokens}, out={response.usage.output_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
