#!/bin/bash

# ==============================================================================
# Script Name: 04_run_eval.sh
# Description:
#   Reproduces the LLM evaluation experiments (see 0014_decoder /
#   0015_decoder_qa4re). Iterates over a set of models/providers and both
#   prompt styles ("direct" / "qa4re"), executing 04_eval_llm.py for each.
#
# Usage:
#   1. Ensure OPENAI_API_KEY / GEMINI_API_KEY are set (e.g. via .env) for the
#      API-based models below.
#   2. Update MODELS to the providers/models you want to evaluate.
#   3. Run: ./scripts/04_run_eval.sh
#
# @Author: Mei Yoshikawa
# ==============================================================================

set -e

# 1. Experiment Configuration

# --- Models to evaluate ---
# Any mix of OpenAI model names, "gemini-*" model names, and local
# "provider/model-name" HF identifiers (run via vLLM unless USE_HF=1 below).
MODELS=(
    "gpt-4"
    "gemini-1.5-flash"
)

# Prompt styles to sweep for every model above.
PROMPT_STYLES=("direct" "qa4re")

# Set to 1 to force local models through the Hugging Face pipeline instead of vLLM.
USE_HF=0

# --- Paths ---
DATASETS="semmed pubmed"
SEEDS="22 24 42 57 75"
DATA_ROOT="./data"
OUTPUT_ROOT="./results/llm"
SCRIPT_PATH="./scripts/04_eval_llm.py"

# 2. Pre-flight Checks

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Evaluation script not found at $SCRIPT_PATH"
    echo "Please run this script from the project root."
    exit 1
fi

echo "========================================================"
echo " Starting LLM Evaluation Experiments"
echo " Data Root    : ${DATA_ROOT}"
echo " Output Root  : ${OUTPUT_ROOT}"
echo "========================================================"

# 3. Execution Loop

count=0

for MODEL in "${MODELS[@]}"; do
  for PROMPT_STYLE in "${PROMPT_STYLES[@]}"; do

    count=$((count + 1))

    echo ""
    echo "--------------------------------------------------------"
    echo " [Run #${count}] Configuration:"
    echo "   Model        : ${MODEL}"
    echo "   Prompt Style : ${PROMPT_STYLE}"
    echo "--------------------------------------------------------"

    HF_ARGS=()
    if [[ "$USE_HF" == "1" ]]; then
      HF_ARGS=(--use_hf)
    fi

    python "$SCRIPT_PATH" \
      --model "${MODEL}" \
      --prompt_style "${PROMPT_STYLE}" \
      --datasets ${DATASETS} \
      --seeds ${SEEDS} \
      --data_root "${DATA_ROOT}" \
      --output_root "${OUTPUT_ROOT}" \
      "${HF_ARGS[@]}"

  done
done

echo ""
echo "========================================================"
echo " All LLM evaluation experiments completed successfully."
echo " Results saved in: ${OUTPUT_ROOT}"
echo "========================================================"
