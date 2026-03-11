#!/usr/bin/env bash
#
# Run the OpenMath benchmark inside a tmux session.
#
# This script launches response generation in a detached tmux session so that
# long-running model calls survive SSH disconnects and laptop lid closures.
# Once generation finishes, evaluation runs automatically in the same session.
#
# Usage:
#   # Full benchmark with defaults (OpenRouter, gpt-5.2)
#   ./scripts/tmux_run.sh
#
#   # Specify provider and model
#   ./scripts/tmux_run.sh --provider openai --model gpt-5.2-pro
#
#   # Single problem
#   ./scripts/tmux_run.sh --problem 041_diff_basis_upper
#
#   # Resume an interrupted run
#   ./scripts/tmux_run.sh --resume results/openrouter_openai-gpt-5.2_20260205_143022/
#
#   # All run_benchmark.py flags are forwarded as-is
#   ./scripts/tmux_run.sh --provider anthropic --parallel 10
#
#   # Run only generation (skip evaluation)
#   ./scripts/tmux_run.sh --phase generate --provider openai --model gpt-5.2
#
#   # Run only evaluation on existing results
#   ./scripts/tmux_run.sh --phase evaluate --resume results/openrouter_openai-gpt-5.2_20260205_143022/
#
#   # Explicitly run both phases (default)
#   ./scripts/tmux_run.sh --phase both
#
# Monitoring:
#   tmux attach -t openmath       # Attach to see live output
#   Ctrl-b d                      # Detach without stopping
#   tmux kill-session -t openmath # Abort the run
#
set -euo pipefail

# Parse --phase flag before forwarding remaining args
PHASE="both"
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)
            if [[ $# -lt 2 ]]; then
                echo "Error: --phase requires a value (generate, evaluate, or both)"
                exit 1
            fi
            PHASE="$2"
            shift 2
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$PHASE" != "generate" && "$PHASE" != "evaluate" && "$PHASE" != "both" ]]; then
    echo "Error: --phase must be 'generate', 'evaluate', or 'both' (got '$PHASE')"
    exit 1
fi

# For evaluate-only, require --resume so we know which results dir to use
if [[ "$PHASE" == "evaluate" ]]; then
    HAS_RESUME=false
    for arg in "${FORWARD_ARGS[@]}"; do
        if [[ "$arg" == "--resume" ]]; then
            HAS_RESUME=true
            break
        fi
    done
    if [[ "$HAS_RESUME" == false ]]; then
        echo "Error: --phase evaluate requires --resume <results_dir>"
        exit 1
    fi
fi

# Re-set positional parameters to the forwarded args
set -- "${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}"

SESSION_NAME="openmath"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Ensure results directory exists
mkdir -p "$PROJECT_ROOT/results"

# Check for existing session
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists."
    echo "Kill it and start a new run? [y/N]"
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$SESSION_NAME"
    else
        echo "Aborting. Attach to the existing session with: tmux attach -t $SESSION_NAME"
        exit 1
    fi
fi

# Write the inner runner script to a temp file so tmux can execute it cleanly.
# This avoids quoting issues with passing complex shell code via tmux new-session.
RUNNER=$(mktemp "$PROJECT_ROOT/results/.tmux_runner_XXXXXX.sh")
LOGFILE="$PROJECT_ROOT/results/tmux_run_$(date +%Y%m%d_%H%M%S).log"

cat > "$RUNNER" <<RUNNER_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"

# Source .env if present (for API keys)
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

LOGFILE="$LOGFILE"
PHASE="$PHASE"

{
echo "=========================================="
echo " OpenMath Benchmark — tmux session"
echo " Started: \$(date)"
echo " Phase: \$PHASE"
echo " Args: $*"
echo "=========================================="
echo ""

# Phase 1: Generate responses
if [ "\$PHASE" = "generate" ] || [ "\$PHASE" = "both" ]; then
    echo ">>> Phase 1: Generating responses..."
    uv run scripts/run_benchmark.py --parallel 5 $* || GEN_EXIT=\$?
    GEN_EXIT=\${GEN_EXIT:-0}

    if [ \$GEN_EXIT -ne 0 ] && [ \$GEN_EXIT -ne 130 ]; then
        echo ""
        echo "ERROR: Response generation failed with exit code \$GEN_EXIT"
    fi
fi

# Phase 2: Evaluate responses
if [ "\$PHASE" = "evaluate" ] || [ "\$PHASE" = "both" ]; then
    # Determine results directory
    if [ "\$PHASE" = "evaluate" ]; then
        # Extract --resume value from args
        RESULTS_DIR=""
        EVAL_ARGS=($*)
        for ((i=0; i<\${#EVAL_ARGS[@]}; i++)); do
            if [ "\${EVAL_ARGS[i]}" = "--resume" ] && [ \$((i+1)) -lt \${#EVAL_ARGS[@]} ]; then
                RESULTS_DIR="\${EVAL_ARGS[i+1]}"
                break
            fi
        done
    else
        # Find the most recent results directory
        RESULTS_DIR=\$(ls -td results/*/ 2>/dev/null | head -1)
    fi

    if [ -z "\$RESULTS_DIR" ] || [ ! -f "\$RESULTS_DIR/responses.jsonl" ]; then
        echo ""
        echo "WARNING: No responses.jsonl found in \${RESULTS_DIR:-<none>}. Skipping evaluation."
    else
        echo ""
        echo ">>> Phase 2: Evaluating responses in \$RESULTS_DIR ..."
        uv run scripts/evaluate_responses.py "\$RESULTS_DIR" --force || EVAL_EXIT=\$?
        EVAL_EXIT=\${EVAL_EXIT:-0}

        echo ""
        echo "=========================================="
        echo " Benchmark complete at \$(date)"
        echo " Results: \$RESULTS_DIR"
        if [ \$EVAL_EXIT -eq 0 ]; then
            echo " Status: SUCCESS"
        else
            echo " Status: Evaluation exited with code \$EVAL_EXIT"
        fi
        echo "=========================================="
    fi
fi

} 2>&1 | tee "\$LOGFILE"

# Clean up the runner script
rm -f "$RUNNER"

# Drop into a shell so the session stays open for inspection
exec bash
RUNNER_EOF

chmod +x "$RUNNER"

# Launch the tmux session
tmux new-session -d -s "$SESSION_NAME" "$RUNNER"

echo "Benchmark started in tmux session '$SESSION_NAME'"
echo ""
echo "  Attach:  tmux attach -t $SESSION_NAME"
echo "  Detach:  Ctrl-b d"
echo "  Log:     $LOGFILE"
echo "  Kill:    tmux kill-session -t $SESSION_NAME"
