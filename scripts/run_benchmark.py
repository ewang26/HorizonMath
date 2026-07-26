#!/usr/bin/env python3
"""
Generate LLM responses for OpenMath benchmark problems.

Prompts models and saves raw responses to a JSONL file. Evaluation is done
separately via evaluate_responses.py.

Usage:
    # Full benchmark (default - uses OpenRouter gpt-5.2 without tool use)
    uv run scripts/run_benchmark.py

    # Use OpenAI directly (with code execution tool)
    uv run scripts/run_benchmark.py --provider openai

    # Use Claude (with extended thinking)
    uv run scripts/run_benchmark.py --provider anthropic

    # Single problem by ID
    uv run scripts/run_benchmark.py --problem diff_basis_upper

    # Override provider/model
    uv run scripts/run_benchmark.py --provider gemini

    # Use GPT-5.6 Sol with max reasoning
    uv run scripts/run_benchmark.py --provider openai --model gpt-5.6-sol --reasoning-effort max

    # Resume interrupted run
    uv run scripts/run_benchmark.py --resume results/openrouter_openai-gpt-5.2_20260205_143022/

    # Then evaluate separately:
    uv run scripts/evaluate_responses.py results/openrouter_openai-gpt-5.2_20260205_143022/
"""

import argparse
import json
import os
import queue
import signal
import sys
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env file from project root
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")

# Handle GEMINI_API_KEY -> GOOGLE_API_KEY mapping
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

import anthropic
import openai
from google import genai
from google.genai import types

# Add scripts directory to path for imports
_script_dir = Path(__file__).parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from evaluate import load_problems


# Default models
DEFAULT_OPENAI_MODEL = "gpt-5.2"
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-5.2"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-6"
DEFAULT_API_TIMEOUT_MINUTES = 75
GPT_5_6_MODELS = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
DEFAULT_REASONING_EFFORT = "high"


def openai_reasoning_config(effort: str) -> dict[str, str]:
    """Build a Responses API reasoning configuration for the requested effort."""

    config = {"effort": effort}
    if effort != "none":
        config["summary"] = "detailed"
    return config

_SYSTEM_MESSAGE_BASE = (
    "You are a research mathematican whose goal is novel mathematical discovery. "
    "You will be presented with problems that are currently open and unsolved that "
    "you must solve. It is important to note that just because no solution is "
    "currently known, it does not mean that a solution cannot be found. You must "
    "reason deeply through the problem and propose a solution, even if uncertain. "
    "Do not refuse or raise exceptions\u2014you must always return a solution by "
    "writing a concrete function called \"proposed_solution()\" that implements a "
    "definite answer. The function must be named exactly \"proposed_solution.\" "
    "VERY IMPORTANT CRITERIA: Your solution must be a genuine closed-form symbolic "
    "expression. Numerical integration functions such as mp.quad(), mp.quadgl(), "
    "mp.quadts(), mpmath.quad(), scipy.integrate, or any other numerical integration "
    "routine are explicitly FORBIDDEN. Rewriting the defining integral and evaluating "
    "it numerically is not a solution and strictly unacceptable. No finite truncations "
    "of infinite series are allowed either, because even though a truncated series is "
    "technically finite, it does not qualify as a genuine closed-form expression."
)

SYSTEM_MESSAGES = {
    "ground_truth_computable": (
        _SYSTEM_MESSAGE_BASE + " For problems requesting closed-form expressions, "
        "you must return a symbolic expression, not a numerical solver.\n\n"
        "Your solution must include only constants, elementary functions, gamma, zeta, "
        "polylog, elliptic integrals, and hypergeometric functions. Functions or expressions "
        "of greater complexity are explicitly not permissible.\n\n"
        "Your solution must be a finite combination of:\n\n"
        "- Rational and algebraic numbers\n\n"
        "- Constants: pi, e, gamma (Euler-Mascheroni), G (Catalan's constant)\n\n"
        "- Elementary functions at algebraic arguments\n\n"
        "- Special functions: Gamma, zeta, polylogarithms, elliptic integrals, "
        "hypergeometric functions at algebraic or rational arguments\n\n"
        "INADMISSIBLE expressions or tools:\n\n"
        "- The defining integral/sum itself or equivalent reformulations\n\n"
        "- Unevaluated infinite series, products, or limits\n\n"
        "- Numerical approximations\n\n"
        "If your solution merely restates the mathematical definition as a computational "
        "procedure\u2014for example, computing a resultant by evaluating one polynomial at the "
        "roots of another, or computing a sum/product by doing a finite or truncated iteration "
        "over its terms\u2014it will be automatically rejected, even if it produces the correct "
        "numerical answer. Using any form of numerical approximation, numerical root-finding, "
        "or equivalent reformulations of the defining expression in your final solution will be "
        "instantly rejected. You must discover a genuinely simplified symbolic formula that goes "
        "beyond the definition of the problem."
    ),
    "benchmark_best_known": (
        _SYSTEM_MESSAGE_BASE + " You must find a result that is both valid and "
        "improves upon the best-known value."
    ),
    "new_construction": (
        _SYSTEM_MESSAGE_BASE + " You must construct a valid mathematical object "
        "satisfying the stated conditions."
    ),
}


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""
    run_id: str
    timestamp: str
    provider: str
    model: str
    problems_file: str
    output_dir: str
    reasoning_effort: str = DEFAULT_REASONING_EFFORT


def is_pro_model(model: str) -> bool:
    return "pro" in model.lower()


def parse_optional_timeout_minutes(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"none", "no", "false", "off", "inf", "infinite", "unlimited"}:
        return None

    try:
        minutes = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a number of minutes or 'none', got {value!r}"
        ) from exc

    if minutes < 0:
        raise argparse.ArgumentTypeError("--api-timeout-minutes cannot be negative")
    if minutes == 0:
        return None
    return minutes


def format_timeout_minutes(minutes: Optional[float]) -> str:
    if minutes is None:
        return "none"
    return f"{minutes:g} minutes"


def openai_response_to_result(response) -> dict:
    result = {"content": response.output_text}
    if response.usage:
        result["usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    reasoning_items = [
        item for item in response.output if getattr(item, "type", None) == "reasoning"
    ]
    if reasoning_items:
        result["reasoning_details"] = [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in reasoning_items
        ]
    return result


def call_openai(
    prompt: str,
    model: str = DEFAULT_OPENAI_MODEL,
    system_message: str = "",
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    timeout_seconds: Optional[float] = DEFAULT_API_TIMEOUT_MINUTES * 60,
) -> dict:
    """Call OpenAI Responses API without tools.

    Returns dict with 'content', 'usage', and optional 'reasoning_details'.
    """
    client = openai.OpenAI(timeout=timeout_seconds)
    kwargs = dict(
        model=model,
        instructions=system_message,
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=125000,
    )
    kwargs["reasoning"] = openai_reasoning_config(reasoning_effort)
    response = client.responses.create(**kwargs)
    return openai_response_to_result(response)


def call_openai_streaming(
    prompt: str,
    model: str = DEFAULT_OPENAI_MODEL,
    system_message: str = "",
    problem_id: str = "",
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    timeout_seconds: Optional[float] = DEFAULT_API_TIMEOUT_MINUTES * 60,
    verbose: bool = True,
) -> dict:
    """Like call_openai but streams to stdout with [HH:MM:SS]-prefixed lines for hang detection.

    Prints reasoning summary and answer lines as they arrive, then returns the
    same dict as call_openai: 'content', 'usage', optional 'reasoning_details'.
    """
    client = openai.OpenAI(timeout=timeout_seconds)

    def ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    stream = client.responses.create(
        model=model,
        instructions=system_message,
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=125000,
        reasoning=openai_reasoning_config(reasoning_effort),
        stream=True,
    )

    label = problem_id or "problem"
    if verbose:
        print(f"[{ts()}] ── {label} ── REASONING ──────────────────────────────", flush=True)
    else:
        print(f"[{ts()}] {label} — streaming response started", flush=True)

    line_buf = ""
    section = "reasoning"
    final_response = None

    for event in stream:
        if event.type == "response.reasoning_summary_text.delta" and verbose:
            line_buf += event.delta
            while "\n" in line_buf:
                line, line_buf = line_buf.split("\n", 1)
                print(f"[{ts()}] {line}", flush=True)
        elif event.type == "response.output_text.delta" and verbose:
            if section == "reasoning":
                if line_buf:
                    print(f"[{ts()}] {line_buf}", flush=True)
                    line_buf = ""
                section = "answer"
                print(f"[{ts()}] ── {label} ── ANSWER ─────────────────────────────────", flush=True)
            line_buf += event.delta
            while "\n" in line_buf:
                line, line_buf = line_buf.split("\n", 1)
                print(f"[{ts()}] {line}", flush=True)
        elif event.type == "response.completed":
            final_response = event.response

    if verbose:
        if line_buf:
            print(f"[{ts()}] {line_buf}", flush=True)
        print(f"[{ts()}] ── {label} ── DONE ───────────────────────────────────", flush=True)
    else:
        print(f"[{ts()}] {label} — streaming response completed", flush=True)

    if final_response is None:
        return {"content": ""}

    return openai_response_to_result(final_response)


def call_openai_with_code_execution(
    prompt: str,
    model: str = DEFAULT_OPENAI_MODEL,
    system_message: str = "",
    container_id: Optional[str] = None,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    timeout_seconds: Optional[float] = DEFAULT_API_TIMEOUT_MINUTES * 60,
) -> dict:
    """Call OpenAI with code interpreter tool enabled.

    Returns dict with 'content' and 'usage'.
    """
    client = openai.OpenAI(timeout=timeout_seconds)
    container = container_id if container_id else {"type": "auto"}
    response = client.responses.create(
        model=model,
        instructions=system_message,
        input=[{"role": "user", "content": prompt}],
        # tools=[{"type": "code_interpreter", "container": container}, {"type": "web_search_preview"}],  # code_interpreter temporarily disabled
        tools=[{"type": "web_search_preview"}],
        max_output_tokens=125000,
        reasoning=openai_reasoning_config(reasoning_effort),
    )
    result = {"content": response.output_text}
    if response.usage:
        result["usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return result


def call_openrouter(
    prompt: str,
    model: str = DEFAULT_OPENROUTER_MODEL,
    system_message: str = "",
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    timeout_seconds: Optional[float] = DEFAULT_API_TIMEOUT_MINUTES * 60,
) -> dict:
    """Call OpenRouter API (OpenAI-compatible, no tool use, reasoning enabled).

    Returns dict with 'content' and optional 'reasoning_details'.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=timeout_seconds,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        max_tokens=125000,
        extra_body={
            "reasoning": {
                "enabled": reasoning_effort != "none",
                "effort": reasoning_effort,
            }
        },
    )
    message = response.choices[0].message
    result = {"content": message.content}
    reasoning_details = getattr(message, "reasoning_details", None)
    if reasoning_details:
        result["reasoning_details"] = [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in reasoning_details
        ]
    reasoning_content = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
    if reasoning_content and "reasoning_details" not in result:
        result["reasoning_details"] = [{"type": "thinking", "thinking": reasoning_content}]
    if response.usage:
        result["usage"] = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return result


def call_gemini_with_code_execution(prompt: str, model: str = DEFAULT_GEMINI_MODEL, system_message: str = "") -> dict:
    """Call Gemini with code execution tool enabled.

    Returns dict with 'content' and optional 'usage'.
    """
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_message,
            tools=[
                types.Tool(code_execution=types.ToolCodeExecution()),
                types.Tool(google_search=types.GoogleSearch()),
            ],
            max_output_tokens=62500,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
        ),
    )
    result = {"content": response.text}
    if response.usage_metadata:
        result["usage"] = {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count,
        }
    return result


def call_anthropic(prompt: str, model: str = DEFAULT_ANTHROPIC_MODEL, system_message: str = "") -> dict:
    """Call Anthropic API with extended thinking.

    Returns dict with 'content', optional 'thinking', 'usage', and 'stop_reason'.
    """
    client = anthropic.Anthropic()
    # Stream: at max_tokens=32000 with always-on thinking, a Fable 5 turn can run
    # past the SDK's ~10-minute non-streaming ceiling, which otherwise raises
    # "Streaming is required for operations that may take longer than 10 minutes"
    # before any request is sent. get_final_message() returns the same Message.
    with client.messages.stream(
        model=model,
        # 128000 is Fable 5's output ceiling. Needed because always-on adaptive
        # thinking can consume tens of thousands of tokens before the answer;
        # a smaller cap truncates mid-thinking and yields an empty response.
        max_tokens=128000,
        # Adaptive thinking works across Opus 4.6/4.7/4.8 and Fable 5. The older
        # {"type": "enabled", "budget_tokens": N} form is rejected with a 400 on
        # Fable 5 and deprecated on Opus 4.6.
        thinking={"type": "adaptive"},
        system=system_message,
        messages=[
            {"role": "user", "content": prompt},
        ],
    ) as stream:
        response = stream.get_final_message()
    result = {"content": ""}
    thinking_parts = []
    text_parts = []
    for block in response.content:
        if block.type == "thinking":
            thinking_parts.append(block.thinking)
        elif block.type == "text":
            text_parts.append(block.text)
    result["content"] = "\n".join(text_parts)
    if thinking_parts:
        result["thinking"] = thinking_parts
    if response.usage:
        result["usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    if response.stop_reason == "max_tokens":
        result["truncated"] = True
        print(f"  ⚠ Anthropic response truncated (hit max_tokens={128000})")
    return result


def create_run_folder(provider: str, model: str, base_dir: Path) -> Path:
    """Create a timestamped output folder for the run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize model name for filesystem (replace / with -)
    safe_model = model.replace("/", "-")
    folder_name = f"{provider}_{safe_model}_{timestamp}"
    output_dir = base_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_completed_problems(responses_path: Path) -> set[str]:
    """Get set of problem IDs that have already been completed."""
    completed = set()
    if responses_path.exists():
        with open(responses_path) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "problem_id" in data:
                        completed.add(data["problem_id"])
                except json.JSONDecodeError:
                    continue
    return completed


def retry_api_call(fn, *args, max_retries=3, **kwargs):
    """Retry an API call with exponential backoff on transient errors."""
    from google.genai import errors as genai_errors

    backoff_seconds = [10, 30, 90]
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except (
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            anthropic.APIError,
            genai_errors.ServerError,
            genai_errors.ClientError,
            ConnectionError,
            TimeoutError,
        ) as e:
            last_exception = e
            if attempt < max_retries:
                delay = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
                print(f"    ⚠ API error (attempt {attempt + 1}/{max_retries + 1}): "
                      f"{type(e).__name__}: {str(e)[:80]}")
                print(f"    Retrying in {delay}s...")
                sys.stdout.flush()
                time.sleep(delay)
            else:
                raise last_exception


def call_openai_for_model(
    prompt: str,
    model: str,
    system_message: str,
    *,
    problem_id: str,
    container_id: Optional[str],
    stream: bool,
    reasoning_effort: str,
    timeout_seconds: Optional[float],
) -> dict:
    if is_pro_model(model):
        return retry_api_call(
            call_openai_streaming,
            prompt,
            model,
            system_message,
            problem_id=problem_id,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
            verbose=stream,
        )

    return retry_api_call(
        call_openai_with_code_execution,
        prompt,
        model,
        system_message,
        container_id=container_id,
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout_seconds,
    )


def run_single_problem(
    problem: dict,
    problem_index: int,
    provider: str,
    model: str,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    container_id: Optional[str] = None,
    stream: bool = False,
    api_timeout_seconds: Optional[float] = DEFAULT_API_TIMEOUT_MINUTES * 60,
    empty_retries: int = 1,
) -> dict:
    """Run LLM on a single problem and return the raw response."""
    prompt = problem["prompt"]
    eval_mode = problem.get("evaluation_mode", "ground_truth_computable")
    system_message = SYSTEM_MESSAGES.get(eval_mode, SYSTEM_MESSAGES["ground_truth_computable"])

    reasoning_details = None
    usage = None
    if provider == "openai":
        result = call_openai_for_model(
            prompt,
            model,
            system_message,
            problem_id=problem["id"],
            container_id=container_id,
            stream=stream,
            reasoning_effort=reasoning_effort,
            timeout_seconds=api_timeout_seconds,
        )
        response = result["content"]
        usage = result.get("usage")
        reasoning_details = result.get("reasoning_details")
    elif provider == "openrouter":
        result = retry_api_call(
            call_openrouter,
            prompt,
            model,
            system_message,
            reasoning_effort=reasoning_effort,
            timeout_seconds=api_timeout_seconds,
        )
        response = result["content"]
        reasoning_details = result.get("reasoning_details")
        usage = result.get("usage")
    elif provider == "gemini":
        result = retry_api_call(call_gemini_with_code_execution, prompt, model, system_message)
        response = result["content"]
        usage = result.get("usage")
    elif provider == "anthropic":
        result = retry_api_call(call_anthropic, prompt, model, system_message)
        response = result["content"]
        reasoning_details = result.get("thinking")
        usage = result.get("usage")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Retry on empty response up to empty_retries times
    for attempt in range(empty_retries):
        if response and response.strip():
            break
        print(f"    ⚠ Empty response for {problem['id']}, retrying (attempt {attempt + 1}/{empty_retries})...")
        sys.stdout.flush()
        if provider == "openai":
            result = call_openai_for_model(
                prompt,
                model,
                system_message,
                problem_id=problem["id"],
                container_id=container_id,
                stream=stream,
                reasoning_effort=reasoning_effort,
                timeout_seconds=api_timeout_seconds,
            )
            response = result["content"]
            usage = result.get("usage")
            reasoning_details = result.get("reasoning_details")
        elif provider == "openrouter":
            result = retry_api_call(
                call_openrouter,
                prompt,
                model,
                system_message,
                reasoning_effort=reasoning_effort,
                timeout_seconds=api_timeout_seconds,
            )
            response = result["content"]
            reasoning_details = result.get("reasoning_details")
            usage = result.get("usage")
        elif provider == "gemini":
            result = retry_api_call(call_gemini_with_code_execution, prompt, model, system_message)
            response = result["content"]
            usage = result.get("usage")
        elif provider == "anthropic":
            result = retry_api_call(call_anthropic, prompt, model, system_message)
            response = result["content"]
            reasoning_details = result.get("thinking")
            usage = result.get("usage")

    data = {
        "problem_id": problem["id"],
        "problem_index": problem_index,
        "title": problem["id"],
        "provider": provider,
        "model": model,
        "response": response,
        "timestamp": datetime.now().isoformat(),
    }
    if provider in {"openai", "openrouter"}:
        data["reasoning_effort"] = reasoning_effort
    if reasoning_details:
        data["reasoning_details"] = reasoning_details
    if usage:
        data["usage"] = usage
    return data




def print_generation_summary(
    num_generated: int,
    total: int,
    duration_seconds: float,
    output_dir: Path,
) -> None:
    """Print final generation summary to console."""
    print("\n" + "═" * 58)
    print("                   GENERATION SUMMARY")
    print("═" * 58)

    print(f"Responses generated: {num_generated}/{total}")

    mins = int(duration_seconds // 60)
    secs = int(duration_seconds % 60)
    print(f"Duration: {mins}m {secs}s")
    print(f"Output: {output_dir}/")
    print(f"\nTo evaluate results, run:")
    print(f"  uv run scripts/evaluate_responses.py {output_dir}")
    print("═" * 58)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM responses for OpenMath benchmark problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "openrouter", "gemini", "anthropic"],
        default="openrouter",
        help="LLM provider (default: openrouter)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "Model name. OpenAI GPT-5.6 options: "
            f"{', '.join(GPT_5_6_MODELS)}. Defaults: {DEFAULT_OPENROUTER_MODEL} "
            f"for OpenRouter, {DEFAULT_OPENAI_MODEL} for OpenAI, "
            f"{DEFAULT_GEMINI_MODEL} for Gemini, and {DEFAULT_ANTHROPIC_MODEL} "
            "for Anthropic"
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=DEFAULT_REASONING_EFFORT,
        help=(
            "Reasoning effort for OpenAI and OpenRouter requests "
            f"(default: {DEFAULT_REASONING_EFFORT})"
        ),
    )
    parser.add_argument(
        "--problem",
        type=str,
        help="Run only a single problem by ID (e.g., 041_diff_basis_upper)",
    )
    parser.add_argument(
        "--range",
        type=str,
        default=None,
        help="Range of problem indices to run, e.g. '10-25' (inclusive, 0-based)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        metavar="DIR",
        help="Resume from an interrupted run directory",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/problems_full.json",
        help="Path to problems JSON file",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of problems to process in parallel (default: 1)",
    )
    parser.add_argument(
        "--api-timeout-minutes",
        type=str,
        default=None,
        help=(
            "API timeout in minutes, or 'none' for no client timeout. Defaults to "
            "none for OpenAI pro models and 75 for other models. Can also be set "
            "with OPENMATH_API_TIMEOUT_MINUTES."
        ),
    )
    parser.add_argument(
        "--empty-retries",
        type=int,
        default=1,
        help="Number of times to retry when the model returns an empty response (default: 1)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save prompts locally without calling the model API",
    )
    args = parser.parse_args()

    # Determine model
    if args.model is None:
        model_defaults = {
            "openai": DEFAULT_OPENAI_MODEL,
            "openrouter": DEFAULT_OPENROUTER_MODEL,
            "gemini": DEFAULT_GEMINI_MODEL,
            "anthropic": DEFAULT_ANTHROPIC_MODEL,
        }
        args.model = model_defaults[args.provider]

    is_openai_pro = args.provider == "openai" and is_pro_model(args.model)

    try:
        if args.api_timeout_minutes is not None:
            api_timeout_minutes = parse_optional_timeout_minutes(args.api_timeout_minutes)
        elif os.getenv("OPENMATH_API_TIMEOUT_MINUTES") is not None:
            api_timeout_minutes = parse_optional_timeout_minutes(os.environ["OPENMATH_API_TIMEOUT_MINUTES"])
        else:
            api_timeout_minutes = None if is_openai_pro else DEFAULT_API_TIMEOUT_MINUTES
    except argparse.ArgumentTypeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    api_timeout_seconds = None if api_timeout_minutes is None else api_timeout_minutes * 60

    # Paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / args.data_file
    results_base = project_root / "results"

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # Load problems
    all_problems = load_problems(data_path)

    # Build mapping from problem id to original index in the full list
    problem_id_to_index = {p["id"]: i for i, p in enumerate(all_problems)}

    # Filter to single problem if specified
    problems = all_problems
    if args.problem:
        matching = [p for p in problems if p["id"] == args.problem]
        if not matching:
            # Try partial match
            matching = [p for p in problems if args.problem in p["id"]]
        if not matching:
            print(f"Error: No problem found matching '{args.problem}'", file=sys.stderr)
            print("Available problem IDs:")
            for p in problems[:10]:
                print(f"  {p['id']}")
            print("  ...")
            sys.exit(1)
        problems = matching

    if args.range:
        parts = args.range.split("-")
        if len(parts) != 2:
            print(f"Error: --range must be in format 'start-end', got '{args.range}'", file=sys.stderr)
            sys.exit(1)
        try:
            range_start, range_end = int(parts[0]), int(parts[1])
        except ValueError:
            print(f"Error: --range values must be integers, got '{args.range}'", file=sys.stderr)
            sys.exit(1)
        if range_start < 0 or range_end >= len(problems) or range_start > range_end:
            print(f"Error: --range {args.range} out of bounds (0-{len(problems) - 1})", file=sys.stderr)
            sys.exit(1)
        problems = problems[range_start:range_end + 1]

    # Handle resume
    if args.resume:
        output_dir = Path(args.resume)
        if not output_dir.exists():
            print(f"Error: Resume directory not found: {output_dir}", file=sys.stderr)
            sys.exit(1)

        # Load existing config
        config_path = output_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
            # Drop legacy fields that may exist in old config files
            valid_fields = {f.name for f in RunConfig.__dataclass_fields__.values()}
            config_data = {k: v for k, v in config_data.items() if k in valid_fields}
            config = RunConfig(**config_data)
            args.reasoning_effort = config.reasoning_effort
        else:
            # Create config from directory name
            run_id = str(uuid.uuid4())[:8]
            config = RunConfig(
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                provider=args.provider,
                model=args.model,
                problems_file=args.data_file,
                output_dir=str(output_dir),
                reasoning_effort=args.reasoning_effort,
            )

        completed = get_completed_problems(output_dir / "responses.jsonl")
        print(f"Resuming run with {len(completed)} completed problems")
    else:
        output_dir = create_run_folder(args.provider, args.model, results_base)
        run_id = str(uuid.uuid4())[:8]
        config = RunConfig(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            provider=args.provider,
            model=args.model,
            problems_file=args.data_file,
            output_dir=str(output_dir),
            reasoning_effort=args.reasoning_effort,
        )
        completed = set()

        # Save config
        with open(output_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)

    # File paths
    prompts_path = output_dir / "prompts.jsonl"
    responses_path = output_dir / "responses.jsonl"
    evaluations_path = output_dir / "evaluation.jsonl"

    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    if args.provider in {"openai", "openrouter"}:
        print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"Problems: {len(problems)}")
    print(f"Parallel: {args.parallel}")
    print(f"API timeout: {format_timeout_minutes(api_timeout_minutes)}")
    if args.provider == "openai" and is_pro_model(args.model):
        print("OpenAI pro mode: streaming")
    print(f"Output: {output_dir}")
    print()

    # Run benchmark
    start_time = datetime.now()
    num_generated = 0
    write_lock = threading.RLock()

    # SIGTERM handler — print status before GitHub Actions kills the process
    def handle_sigterm(signum, frame):
        print(f"\nSIGTERM received — {num_generated} responses saved to {responses_path}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    # Build list of problems to run (use original index from full problem list)
    pending = [
        (problem_id_to_index[problem["id"]], problem) for problem in problems
        if problem["id"] not in completed
    ]

    # Pre-create containers for OpenAI code interpreter to avoid per-request
    # container provisioning (which causes thundering-herd timeouts at startup).
    # Containers expire after 20 minutes of inactivity, so a keepalive thread
    # calls retrieve() on each container every 5 minutes to reset the idle timer.
    container_queue: Optional[queue.Queue] = None
    openai_containers: list[str] = []
    mgmt_client: Optional[openai.OpenAI] = None
    keepalive_stop: Optional[threading.Event] = None
    keepalive_thread: Optional[threading.Thread] = None
    # code_interpreter temporarily disabled — container lifecycle issues under investigation
    uses_code_interpreter = False
    # uses_code_interpreter = (
    #     args.provider == "openai"
    #     and not args.debug
    #     and not ("pro" in (args.model or ""))
    # )
    if uses_code_interpreter:
        mgmt_client = openai.OpenAI(timeout=60)
        print(f"Pre-creating {args.parallel} code interpreter containers...")
        sys.stdout.flush()
        for i in range(args.parallel):
            container = mgmt_client.containers.create(name=f"openmath-worker-{i}")
            openai_containers.append(container.id)
            print(f"  Container {i + 1}/{args.parallel}: {container.id}")
            sys.stdout.flush()
        container_queue = queue.Queue()
        for cid in openai_containers:
            container_queue.put(cid)
        print()

        # Start keepalive thread — retrieve() resets last_active_at on each container
        keepalive_stop = threading.Event()
        def _keepalive_containers():
            while not keepalive_stop.wait(timeout=5 * 60):
                for cid in openai_containers:
                    try:
                        mgmt_client.containers.retrieve(cid)
                    except Exception:
                        pass
        keepalive_thread = threading.Thread(
            target=_keepalive_containers, daemon=True, name="container-keepalive"
        )
        keepalive_thread.start()

    def process_problem(i: int, problem: dict) -> dict:
        """Run LLM on a single problem and save the response."""
        problem_id = problem["id"]
        eval_mode = problem.get("evaluation_mode", "ground_truth_computable")
        system_message = SYSTEM_MESSAGES.get(eval_mode, SYSTEM_MESSAGES["ground_truth_computable"])

        # Save prompt before calling the API
        with write_lock:
            with open(prompts_path, "a") as f:
                f.write(json.dumps({
                    "problem_id": problem_id,
                    "system_message": system_message,
                    "prompt": problem["prompt"],
                }) + "\n")
            print(f"[{i + 1}/{len(problems)}] {problem_id} — prompt saved")
            sys.stdout.flush()

        if args.debug:
            return {"problem_id": problem_id, "skipped": True}

        container_id = None
        if container_queue is not None:
            container_id = container_queue.get()
        try:
            response_data = run_single_problem(
                problem,
                i,
                args.provider,
                args.model,
                reasoning_effort=args.reasoning_effort,
                container_id=container_id,
                stream=(args.parallel == 1),
                api_timeout_seconds=api_timeout_seconds,
                empty_retries=args.empty_retries,
            )
        finally:
            if container_queue is not None and container_id is not None:
                container_queue.put(container_id)

        # Save response immediately (thread-safe)
        with write_lock:
            with open(responses_path, "a") as f:
                f.write(json.dumps(response_data) + "\n")
            print(f"  {problem_id} — response saved")
            sys.stdout.flush()

        return response_data

    try:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(process_problem, i, problem): problem["id"]
                for i, problem in pending
            }
            num_errors = 0
            try:
                for future in as_completed(futures):
                    problem_id = futures[future]
                    try:
                        future.result()
                        num_generated += 1
                        print(f"  Progress: {num_generated}/{len(pending)} responses saved ({num_errors} errors)")
                        sys.stdout.flush()
                    except Exception as e:
                        num_errors += 1
                        print(f"\n  {problem_id}")
                        print(f"        ERROR - {type(e).__name__}: {str(e)[:60]}")
                        sys.stdout.flush()

                        # Look up real index and title for this problem
                        err_index = problem_id_to_index.get(problem_id, -1)
                        err_problem = next((p for p in problems if p["id"] == problem_id), {})
                        eval_result = {
                            "problem_id": problem_id,
                            "problem_index": err_index,
                            "problem_title": err_problem.get("id", ""),
                            "mode": err_problem.get("evaluation_mode", "unknown"),
                            "error_type": "runtime",
                            "error_message": f"{type(e).__name__}: {str(e)}",
                        }
                        with write_lock:
                            with open(evaluations_path, "a") as f:
                                f.write(json.dumps(eval_result) + "\n")
            except KeyboardInterrupt:
                print("\n\nInterrupted. Progress saved. Resume with:")
                print(f"  uv run scripts/run_benchmark.py --resume {output_dir}")
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(130)
    finally:
        if keepalive_stop is not None:
            keepalive_stop.set()
        if keepalive_thread is not None:
            keepalive_thread.join(timeout=10)
        if openai_containers and mgmt_client is not None:
            print(f"\nCleaning up {len(openai_containers)} containers...")
            sys.stdout.flush()
            for cid in openai_containers:
                try:
                    mgmt_client.containers.delete(cid)
                except Exception:
                    pass

    # Print generation summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_generation_summary(num_generated, len(problems), duration, output_dir)
    sys.exit(0)


if __name__ == "__main__":
    main()
