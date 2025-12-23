"""
scripts/03_train.py

This script serves as the main entry point for running fine-tuning experiments 
for Relation Extraction (RE). It handles:
    1. Dynamic selection of model architecture (R-BERT, Cell-RBERT, etc.).
    2. Data loading and dataset-specific preprocessing.
    3. execution of the training loop using Hugging Face Trainer.

Usage:
    # Option 1: Run via the wrapper shell script
    scripts/03_run_finetuning.sh

    # Option 2: Direct execution
    python scripts/03_train.py \
        --model_type R-ENT-base \
        --model_name "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

@Author: Mei Yoshikawa
"""

import sys
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the parent directory to sys.path to allow importing from 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.CCBERT.trainer import run_training
from src.CCBERT.utils import (
    apply_boundary_tags,
    normalize_boundary_tags,
    convert_boundary_to_replacement,
    normalize_replacement_tokens,
    clean_replacement_suffix
)

def parse_args():
    """
    Parse command-line arguments for experiment configuration.
    """
    parser = argparse.ArgumentParser(description="Run Relation Extraction Fine-tuning")
    
    # --- Experiment Identification ---
    parser.add_argument(
        "--model_type", 
        type=str, 
        required=True, 
        help="Experiment identifier (e.g., 'R-ENT-base', 'R-ENT-CPT'). "
             "Must start with a valid core architecture: 'B-ENT', 'R-ENT', 'R-CLS', or 'B-CLS'."
    )
    
    # --- Model Configuration ---
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        help="Path to the pretrained model or Hugging Face model ID."
    )
    
    # --- Data Settings ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--size", type=int, default=1400, help="Number of training samples to use.")
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory containing datasets.")
    parser.add_argument("--output_root", type=str, default="./results", help="Root directory for saving outputs.")

    # --- Training Hyperparameters ---
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training and evaluation batch size.")
    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum number of training epochs.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for the scheduler.")
    
    parser.add_argument(
        "--target_epochs", 
        type=int, 
        nargs='+', 
        default=[3, 5, 10, 20],
        help="List of specific epochs at which to evaluate and save checkpoints."
    )
    
    return parser.parse_args()

def get_preprocessing_fn(core_model_type, data_source):
    """
    Factory function to return the appropriate preprocessing function based on 
    the model architecture and the source dataset format.

    Args:
        core_model_type (str): The core architecture (e.g., 'R-ENT', 'B-ENT').
        data_source (str): The source dataset name ('semmed' or 'pubmed').

    Returns:
        callable: A function that processes a raw text string.
    """
    # Check if the model uses replacement tokens (e.g., [CELL0]) instead of boundary tags
    is_replacement = core_model_type in ["R-ENT", "R-CLS"]
    
    # --- Strategy for SemMed Data (Raw format: "context 【Entity】 context") ---
    if data_source == "semmed":
        def preprocess_semmed_boundary(text):
            # Convert 【】 to <E0> tags and normalize
            t = apply_boundary_tags(text)
            return normalize_boundary_tags(t)

        def preprocess_semmed_replacement(text):
            # Convert 【】 -> <E0> -> [CELL0]
            t = apply_boundary_tags(text)
            t = convert_boundary_to_replacement(text)
            t = clean_replacement_suffix(t)
            return normalize_replacement_tokens(t)
            
        return preprocess_semmed_replacement if is_replacement else preprocess_semmed_boundary

    # --- Strategy for PubMed Data (Raw format: "context <E0>Entity</E0> context") ---
    elif data_source == "pubmed":
        def preprocess_pubmed_boundary(text):
            # Already has <E0> tags, just normalize
            return normalize_boundary_tags(text)

        def preprocess_pubmed_replacement(text):
            # Convert <E0> -> [CELL0] directly
            t = convert_boundary_to_replacement(text)
            t = clean_replacement_suffix(t)
            return normalize_replacement_tokens(t)

        return preprocess_pubmed_replacement if is_replacement else preprocess_pubmed_boundary
    
    else:
        raise ValueError(f"Unknown data source: {data_source}")

def main():
    args = parse_args()

    # 1. Determine Core Architecture
    # Extract the core architecture (e.g., R-ENT) from the experiment identifier (e.g., R-ENT-base).
    valid_archs = ["B-ENT", "R-ENT", "R-CLS", "B-CLS"]
    core_type = None
    
    for arch in valid_archs:
        if args.model_type.startswith(arch):
            core_type = arch
            break

    # 2. Setup Output Directory
    # The folder name will be the full identifier (args.model_type)
    output_dir = os.path.join(
        args.output_root, 
        args.model_type, 
        f"lr{args.lr}_seed{args.seed}_sz{args.size}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f" Start Fine-tuning Experiment")
    print(f" Experiment ID  : {args.model_type}")
    print(f" Core Arch      : {core_type}")
    print(f" Model Path     : {args.model_name}")
    print(f" Config         : Seed={args.seed}, Size={args.size}, LR={args.lr}")
    print(f" Output Dir     : {output_dir}")
    print(f"{'='*60}\n")

    # 3. Load Data
    # Expected structure: data_root/{dataset_name}/{seed}/train.csv
    train_path = os.path.join(args.data_root, "semmed_raw", str(args.seed), "train.csv")
    test_path  = os.path.join(args.data_root, "semmed_raw", str(args.seed), "test.csv")
    pubmed_path = os.path.join(args.data_root, "pubmed", str(args.seed), "test.csv")

    df_train_full = pd.read_csv(train_path).head(args.size)
    df_test = pd.read_csv(test_path)
    df_pubmed = pd.read_csv(pubmed_path)

    # 4. Split Train/Eval
    df_train, df_eval = train_test_split(
        df_train_full, test_size=0.1, random_state=args.seed, stratify=df_train_full["label"]
    )

    # 5. Apply Preprocessing
    prep_semmed = get_preprocessing_fn(core_type, "semmed")
    prep_pubmed = get_preprocessing_fn(core_type, "pubmed")

    print("[Info] Applying preprocessing to datasets...")
    df_train["sentence"] = df_train["sentence"].apply(prep_semmed)
    df_eval["sentence"]  = df_eval["sentence"].apply(prep_semmed)
    df_test["sentence"]  = df_test["sentence"].apply(prep_semmed)
    
    df_pubmed["sentence"] = df_pubmed["sentence"].apply(prep_pubmed)

    # 6. Run Training
    run_training(
        df_train=df_train,
        df_eval=df_eval,
        df_test=df_test,
        df_pubmed=df_pubmed,
        model_type=core_type,
        model_name_or_path=args.model_name,
        output_dir=output_dir,
        num_labels=2,
        seed=args.seed,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_train_epochs=args.max_epochs,
        dropout_rate=args.dropout,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        target_epochs=args.target_epochs
    )

if __name__ == "__main__":
    main()