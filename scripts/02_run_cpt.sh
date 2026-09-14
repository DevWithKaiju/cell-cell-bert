#!/bin/bash

# ==============================================================================
# Script Name: 02_run_cpt.sh
# Description:
#   Reproduces the Continual Pre-training (CPT) experiments.
#   Iterates over the two entity-representation strategies ("replacement" /
#   "boundary") and, for each, both the mixed-corpus setup (50% of samples
#   reverted, special tokens registered) and the fully tag-free setup
#   (100% reverted, no special tokens) via --strip_tags_only.
#   Executes 02_cpt.py for each combination.
#
# Usage:
#   1. Ensure your virtual environment is activated.
#   2. Update DATA_PATH_REPLACEMENT / DB_PATH / DATA_PATH_BOUNDARY below.
#   3. Run: ./scripts/02_run_cpt.sh
#
# @Author: Mei Yoshikawa
# ==============================================================================

set -e

# 1. Experiment Configuration

# --- Data Sources ---
# "replacement": entities already replaced with [CELL0]/[CELL1] in data_path;
#   db_path is required to look up the original (unmasked) sentence.
# "boundary": entities wrapped with <E0>...</E0> in data_path; no db_path needed.
DATA_SOURCES=("replacement" "boundary")

# Whether to also run the fully tag-free variant (--strip_tags_only) for each
# data source, in addition to the default mixed-corpus setup.
STRIP_TAGS_OPTIONS=("false" "true")

# --- Paths ---
# [!] UPDATE THESE PATHS: point them at your preprocessed CPT corpora.
DATA_PATH_REPLACEMENT="./data/semmed/cpt_replacement.csv"
DB_PATH="./data/semmed/hit_cells.db"
DATA_PATH_BOUNDARY="./data/semmed/cpt_boundary.csv"

MODEL_NAME="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
OUTPUT_ROOT="./results/cpt"
SCRIPT_PATH="./scripts/02_cpt.py"

# --- Fixed Hyperparameters ---
LR="2e-5"
WEIGHT_DECAY="1e-4"
WARMUP_RATIO="0.1"
BATCH_SIZE=128
MAX_EPOCHS=2
SEED=42

# 2. Pre-flight Checks

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: CPT script not found at $SCRIPT_PATH"
    echo "Please run this script from the project root."
    exit 1
fi

echo "========================================================"
echo " Starting CPT Experiments"
echo " Model Name       : ${MODEL_NAME}"
echo " Output Root      : ${OUTPUT_ROOT}"
echo "========================================================"

# 3. Execution Loop

count=0

for DATA_SOURCE in "${DATA_SOURCES[@]}"; do
  if [[ "$DATA_SOURCE" == "replacement" ]]; then
    DATA_PATH="$DATA_PATH_REPLACEMENT"
    DB_ARGS=(--db_path "${DB_PATH}")
  else
    DATA_PATH="$DATA_PATH_BOUNDARY"
    DB_ARGS=()
  fi

  for STRIP_TAGS in "${STRIP_TAGS_OPTIONS[@]}"; do

    count=$((count + 1))

    EXP_NAME="${DATA_SOURCE}"
    STRIP_ARGS=()
    if [[ "$STRIP_TAGS" == "true" ]]; then
      EXP_NAME="${EXP_NAME}_notag"
      STRIP_ARGS=(--strip_tags_only)
    fi

    echo ""
    echo "--------------------------------------------------------"
    echo " [Run #${count}] Configuration:"
    echo "   Data Source     : ${DATA_SOURCE}"
    echo "   Strip Tags Only : ${STRIP_TAGS}"
    echo "   Exp Name        : ${EXP_NAME}"
    echo "--------------------------------------------------------"

    python "$SCRIPT_PATH" \
      --model_name "${MODEL_NAME}" \
      --output_root "${OUTPUT_ROOT}" \
      --exp_name "${EXP_NAME}" \
      --data_source "${DATA_SOURCE}" \
      --data_path "${DATA_PATH}" \
      "${DB_ARGS[@]}" \
      --lr "${LR}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --warmup_ratio "${WARMUP_RATIO}" \
      --batch_size "${BATCH_SIZE}" \
      --max_epochs "${MAX_EPOCHS}" \
      --seed "${SEED}" \
      "${STRIP_ARGS[@]}"

  done
done

echo ""
echo "========================================================"
echo " All CPT experiments completed successfully."
echo " Results saved in: ${OUTPUT_ROOT}"
echo "========================================================"
