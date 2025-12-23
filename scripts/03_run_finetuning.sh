#!/bin/bash

# ==============================================================================
# Script Name: 03_run_finetuning.sh
# Description:
#   Reproduces the fine-tuning experiments for Relation Extraction (RE).
#   It iterates over a grid of hyperparameters (Data Size, Seeds, LR, etc.)
#   and executes the training script (03_train.py).
#
# Usage:
#   1. Ensure your virtual environment is activated.
#   2. Run: ./scripts/03_run_finetuning.sh
# 
# @Author: Mei Yoshikawa
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# 1. Experiment Configuration

# --- Grid Search Parameters ---
# Data sizes to simulate low-resource settings
SIZES=(300)

# Random seeds for reproducibility
SEEDS=(22)

# Hyperparameters
LRS=("2e-5")
WARMUPS=("0.1")
BATCH_SIZES=(16)
WEIGHT_DECAYS=("1e-4")

# --- Fixed Settings ---
MAX_EPOCHS=5
TARGET_EPOCHS="3 5 10 20" # Epochs to evaluate and save
DROPOUT=0.1

# --- Model Settings ---
# Comparison grid: 2 (Base/CPT) * 2 (R/B) * 2 (ENT/CLS) = 8 variations
MODEL_TYPES=(
    "R-CLS-base" "B-ENT-CPT"
    "B-CLS-base" "B-CLS-CPT"
)

# Paths to Base Model and CPT Model
# [!] UPDATE THIS PATH: Set the path to your actual CPT model checkpoint
PATH_TO_BASE_MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
PATH_TO_CPT_MODEL="/workspace/250911_cells_knowledge_ext/11_compare_models/cpt/surround_entity/lr2e-05_wd0.0001_warmup0.1_half"

# --- Paths ---
DATA_ROOT="./data"
OUTPUT_ROOT="./results/ft"
SCRIPT_PATH="./scripts/02_train.py" # Adjusted to standard structure

# 2. Pre-flight Checks

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Training script not found at $SCRIPT_PATH"
    echo "Please run this script from the project root."
    exit 1
fi

echo "========================================================"
echo " Starting Grid Search Experiments"
echo " Data Root        : ${DATA_ROOT}"
echo " Output Root      : ${OUTPUT_ROOT}"
echo " Base Model       : ${PATH_TO_BASE_MODEL}"
echo " CPT Model        : ${PATH_TO_CPT_MODEL}"
echo "========================================================"

# 3. Execution Loop

count=0

for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
  # --- Logic to Switch Model Path based on Type ---
  if [[ "$MODEL_TYPE" == *"-CPT" ]]; then
      CURRENT_MODEL_NAME="$PATH_TO_CPT_MODEL"
      echo "[Config] Selected CPT Weights for $MODEL_TYPE"
  else
      CURRENT_MODEL_NAME="$PATH_TO_BASE_MODEL"
      echo "[Config] Selected BASE Weights for $MODEL_TYPE"
  fi

  for size in "${SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for lr in "${LRS[@]}"; do
        for warmup in "${WARMUPS[@]}"; do
          for batch_size in "${BATCH_SIZES[@]}"; do
            for wd in "${WEIGHT_DECAYS[@]}"; do
              
              count=$((count + 1))
              
              echo ""
              echo "--------------------------------------------------------"
              echo " [Run #${count}] Configuration:"
              echo "   Model Type   : ${MODEL_TYPE}"
              echo "   Weights Path : ${CURRENT_MODEL_NAME}"
              echo "   Size=${size}, Seed=${seed}, LR=${lr}"
              echo "--------------------------------------------------------"

              python "$SCRIPT_PATH" \
                --model_type "${MODEL_TYPE}" \
                --model_name "${CURRENT_MODEL_NAME}" \
                --seed "${seed}" \
                --size "${size}" \
                --data_root "${DATA_ROOT}" \
                --output_root "${OUTPUT_ROOT}" \
                --lr "${lr}" \
                --warmup_ratio "${warmup}" \
                --batch_size "${batch_size}" \
                --weight_decay "${wd}" \
                --max_epochs "${MAX_EPOCHS}" \
                --dropout "${DROPOUT}" \
                --target_epochs ${TARGET_EPOCHS}

            done
          done
        done
      done
    done
  done
done

echo ""
echo "========================================================"
echo " All experiments completed successfully."
echo " Results saved in: ${OUTPUT_ROOT}"
echo "========================================================"