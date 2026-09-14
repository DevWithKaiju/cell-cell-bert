#!/bin/bash

# ==============================================================================
# Script Name: 03_run_finetuning.sh
# Description:
#   Reproduces the fine-tuning experiments for Relation Extraction (RE).
#   It iterates over a grid of model types / hyperparameters and executes the
#   training script (03_train.py). --base_model_path and --cpt_model_path are
#   both always passed; scripts/03_train.py (via src/cc_bert/finetune.py)
#   picks whichever one applies based on the "-CPT" suffix of --model_type.
#
# Usage:
#   1. Ensure your virtual environment is activated.
#   2. Update PATH_TO_CPT_MODEL_REPLACEMENT / PATH_TO_CPT_MODEL_BOUNDARY below
#      (these should point at the output_dir of a scripts/02_cpt.py run).
#   3. Run: ./scripts/03_run_finetuning.sh
#
# @Author: Mei Yoshikawa
# ==============================================================================

set -e

# 1. Experiment Configuration

# --- Grid Search Parameters ---
# Data sizes to simulate low-resource settings.
# Paper values: 1350 (full SemMed train set, see 0002/0003_ft_*_models),
# and 270/540/810/1080 for the data-size sweep (see 0011/0012_ft_*_change_datasize).
SIZES=(1350)

# Random seeds for reproducibility. Paper values: 22 24 42 57 75.
SEEDS=(22)

# Hyperparameters (fixed across the paper's grid)
LRS=("2e-5")
WARMUPS=("0.1")
BATCH_SIZES=(16)
WEIGHT_DECAYS=("1e-4")

# --- Fixed Settings ---
MAX_EPOCHS=5
TARGET_EPOCHS="3 5 10 20" # Epochs to evaluate and save
DROPOUT=0.1

# --- Model Settings ---
# Full grid: 2 (Replacement/Boundary) * 3 (HYB/ENT/CLS) * 2 (base/CPT) = 12 variations
MODEL_TYPES=(
    "R-HYB-base" "R-HYB-CPT" "R-ENT-base" "R-ENT-CPT" "R-CLS-base" "R-CLS-CPT"
    "B-HYB-base" "B-HYB-CPT" "B-ENT-base" "B-ENT-CPT" "B-CLS-base" "B-CLS-CPT"
)

# Paths to Base Model and CPT Model
PATH_TO_BASE_MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# [!] UPDATE THESE PATHS: point them at your scripts/02_cpt.py output directories.
# "R-*" model types use replacement-strategy CPT weights, "B-*" use boundary-strategy.
PATH_TO_CPT_MODEL_REPLACEMENT="./results/cpt/replacement"
PATH_TO_CPT_MODEL_BOUNDARY="./results/cpt/boundary"

# --- Paths ---
DATA_ROOT="./data"
OUTPUT_ROOT="./results/ft"
SCRIPT_PATH="./scripts/03_train.py"

# 2. Pre-flight Checks

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Training script not found at $SCRIPT_PATH"
    echo "Please run this script from the project root."
    exit 1
fi

echo "========================================================"
echo " Starting Grid Search Experiments"
echo " Data Root                  : ${DATA_ROOT}"
echo " Output Root                : ${OUTPUT_ROOT}"
echo " Base Model                 : ${PATH_TO_BASE_MODEL}"
echo " CPT Model (Replacement)    : ${PATH_TO_CPT_MODEL_REPLACEMENT}"
echo " CPT Model (Boundary)       : ${PATH_TO_CPT_MODEL_BOUNDARY}"
echo "========================================================"

# 3. Execution Loop

count=0

for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
  # CPT weights differ by data strategy: R-* uses replacement, B-* uses boundary.
  if [[ "$MODEL_TYPE" == R-* ]]; then
      CURRENT_CPT_MODEL="$PATH_TO_CPT_MODEL_REPLACEMENT"
  else
      CURRENT_CPT_MODEL="$PATH_TO_CPT_MODEL_BOUNDARY"
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
              echo "   Size=${size}, Seed=${seed}, LR=${lr}"
              echo "--------------------------------------------------------"

              python "$SCRIPT_PATH" \
                --model_type "${MODEL_TYPE}" \
                --base_model_path "${PATH_TO_BASE_MODEL}" \
                --cpt_model_path "${CURRENT_CPT_MODEL}" \
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
echo " All fine-tuning experiments completed successfully."
echo " Results saved in: ${OUTPUT_ROOT}"
echo "========================================================"

# --------------------------------------------------------------------------
# Other base models (see 0004_ft_other_models): only R-CLS-base/B-CLS-base,
# swept across alternative base models, with --append_model_name to keep
# each base model's results in a distinct output directory.
# --------------------------------------------------------------------------
# OTHER_BASE_MODELS=("answerdotai/ModernBERT-base")
# for base_model in "${OTHER_BASE_MODELS[@]}"; do
#   for model_type in "R-CLS-base" "B-CLS-base"; do
#     python "$SCRIPT_PATH" \
#       --model_type "${model_type}" \
#       --base_model_path "${base_model}" \
#       --seed 22 --size 1350 \
#       --data_root "${DATA_ROOT}" --output_root "${OUTPUT_ROOT}" \
#       --lr 2e-5 --warmup_ratio 0.1 --batch_size 16 --weight_decay 1e-4 \
#       --max_epochs 5 --dropout 0.1 --target_epochs 3 5 \
#       --append_model_name
#   done
# done
