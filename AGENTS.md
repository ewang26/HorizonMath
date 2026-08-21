# AGENTS.md

## Cursor Cloud specific instructions

HorizonMath is a Python math-benchmark project. Its core product is the **automatic
verification pipeline** that scores LLM-generated math solutions. There is no GUI and
no web service — everything runs as CLI scripts. See `README.md` for the full command
reference; the notes below only capture non-obvious, durable gotchas.

### Environment / tooling
- Package manager is **uv** (installed at `~/.local/bin/uv`, already on `PATH` via
  `~/.profile`/`~/.bashrc`). Dependencies come from `pyproject.toml` + `uv.lock`; the
  startup update script runs `uv sync`. Always invoke project code with `uv run ...`.
- Python is pinned to 3.12 (`.python-version`).
- There is **no linter** configured (no ruff/flake8/black/pylint). "Lint" is not a step
  in this repo; don't invent one.

### Testing
- Run the suite with `uv run pytest` (43 tests, ~5–10s). `tests/test_ramsey_baseline.py`
  is a standalone script, not a pytest module, so pytest does not collect it.

### Running the product
- Single solution: `uv run python scripts/evaluate.py --llm-output <file> --problem-id <id>`
  (mode auto-detected). Numeric (`ground_truth_computable`) and benchmark
  (`benchmark_best_known`) evaluation need **no API key**.
- Two-phase benchmark: `scripts/run_benchmark.py` (Phase 1, generate) then
  `scripts/evaluate_responses.py <run_dir>` (Phase 2, evaluate). Only Phase 1 and the
  compliance reviewer call external model APIs. Results land in `results/` (gitignored).

### Non-obvious gotchas
- **Numeric solutions must return the value as a string (or high-precision), not a bare
  Python `float`.** A `float` only preserves ~16 digits, so it fails the default
  20-digit match threshold. Returning the digits as a string passes (e.g. 38/38 digits).
- **Model API keys must be stored WITHOUT trailing whitespace/newline.** A trailing
  `\n` in `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`, etc.) corrupts the `Authorization`
  header: OpenRouter returns `{"error":"JSON parsing failed","code":400}` and the OpenAI
  SDK raises `APIConnectionError: Connection error`. `run_benchmark.py` reads the key raw
  (`os.getenv`) and does not strip it, so the secret itself must be clean. Provider keys
  the runner understands: `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`,
  `ANTHROPIC_API_KEY`.
- A few validators (elliptic-curve rank, inverse-Galois) shell out to **SageMath**, which
  is not installed here. They report a "sage not found" failure and honor `SAGE_CMD` if
  you install Sage separately. The rest of the evaluation pipeline works without Sage.
