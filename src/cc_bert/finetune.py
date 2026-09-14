"""
finetune.py

Resolves a `--model_type` identifier (e.g. "R-HYB-CPT", "B-CLS-base") into a
concrete architecture, weights source, and preprocessed data files, then runs
fine-tuning for Relation Extraction via `trainer.run_training`.

Data-size sweeps are just a different `--size` value (number of training rows
to use) passed to `run_experiment` -- there is no separate code path for them.

@Author: Mei Yoshikawa
"""

import os
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from .trainer import run_training


def get_parser():
    parser = argparse.ArgumentParser(description="Run Relation Extraction Fine-tuning")

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Experiment identifier (e.g., 'R-HYB-base', 'B-CLS-CPT'). "
             "Must start with a valid core architecture: 'B-HYB', 'R-HYB', 'B-ENT', 'R-ENT', 'R-CLS', or 'B-CLS'. "
             "A '-CPT' suffix selects --cpt_model_path as the starting weights instead of --base_model_path."
    )

    parser.add_argument(
        "--base_model_path",
        type=str,
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        help="Path to the pretrained model or Hugging Face model ID."
    )

    parser.add_argument(
        "--cpt_model_path",
        type=str,
        default=None,
        help="Path to the CPT model (required when --model_type ends with '-CPT'). "
             "Typically the output_dir of a run of scripts/02_cpt.py."
    )

    parser.add_argument("--seed", type=int, default=22, help="Random seed for reproducibility.")
    parser.add_argument("--size", type=int, default=300, help="Number of training samples to use.")

    parser.add_argument("--data_root", type=str, default="./data", help="Root directory containing datasets.")
    parser.add_argument("--output_root", type=str, default="./results/ft", help="Root directory for saving outputs.")

    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training and evaluation batch size.")
    parser.add_argument("--max_epochs", type=int, default=5, help="Maximum number of training epochs.")
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

    parser.add_argument(
        "--save_best_model_weights",
        action="store_true",
        default=True,
        help="Keep the model weights for the best model."
    )

    return parser


def run_experiment(args, append_model_name=False):
    """
    Args:
        args: Parsed arguments from `get_parser()`.
        append_model_name (bool): If True, append a short form of --base_model_path to the
            output directory name (e.g. "R-CLS-base-ModernBERT"). Useful when sweeping the
            same model_type across multiple base models (see 0004_ft_other_models).
    """
    # 1. Determine Core Architecture and format suffix
    valid_archs = ["B-HYB", "R-HYB", "B-ENT", "R-ENT", "R-CLS", "B-CLS"]
    core_type = None

    for arch in valid_archs:
        if args.model_type.startswith(arch):
            core_type = arch
            break

    if not core_type:
        raise ValueError(f"Invalid model_type: {args.model_type}. Must start with one of {valid_archs}")

    if core_type.startswith("R-"):
        suffix = "replacement"
    elif core_type.startswith("B-"):
        suffix = "boundary"
    else:
        raise ValueError(f"Cannot determine data suffix for {core_type}")

    # Determine Model Path dynamically
    if args.model_type.endswith("-CPT"):
        if not args.cpt_model_path:
            raise ValueError("--cpt_model_path must be provided for CPT models")
        current_model_name = args.cpt_model_path
        print(f"[Config] Selected CPT Weights for {args.model_type}")
    else:
        current_model_name = args.base_model_path
        print(f"[Config] Selected BASE Weights for {args.model_type}")

    # Determine output directory
    if append_model_name:
        if "ModernBERT" in args.base_model_path:
            short_model_name = "ModernBERT"
        elif "bert" in args.base_model_path.lower():
            short_model_name = "BERT"
        else:
            short_model_name = args.base_model_path.split("/")[-1]
        model_dir_name = f"{args.model_type}-{short_model_name}"
    else:
        model_dir_name = args.model_type

    output_dir = os.path.join(args.output_root, model_dir_name, str(args.seed))
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Start Fine-tuning Experiment")
    print(f" Experiment ID  : {args.model_type}")
    print(f" Core Arch      : {core_type}")
    print(f" Data Suffix    : {suffix}")
    print(f" Model Path     : {current_model_name}")
    print(f" Config         : Seed={args.seed}, Size={args.size}, LR={args.lr}")
    print(f" Output Dir     : {output_dir}")
    print(f"{'='*60}\n")

    # 3. Load Data (preprocessed by scripts/01_preprocess.py)
    train_path = os.path.join(args.data_root, "semmed", str(args.seed), f"train_{suffix}.csv")
    test_path = os.path.join(args.data_root, "semmed", str(args.seed), f"test_{suffix}.csv")
    pubmed_path = os.path.join(args.data_root, "pubmed", str(args.seed), f"test_{suffix}.csv")

    print(f"[Info] Loading data from {train_path}...")
    df_train_full = pd.read_csv(train_path).head(args.size)
    df_test = pd.read_csv(test_path)
    df_pubmed = pd.read_csv(pubmed_path)

    # 4. Split Train/Eval
    df_train, df_eval = train_test_split(
        df_train_full, test_size=0.1, random_state=args.seed, stratify=df_train_full["label"]
    )

    print(f"[Info] Pre-processed {suffix} CSV loaded. Skipping dynamic preprocessing.")

    # 5. Run Training
    run_training(
        df_train=df_train,
        df_eval=df_eval,
        df_test=df_test,
        df_pubmed=df_pubmed,
        model_type=core_type,
        model_name_or_path=current_model_name,
        output_dir=output_dir,
        num_labels=2,
        seed=args.seed,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_train_epochs=args.max_epochs,
        dropout_rate=args.dropout,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        target_epochs=args.target_epochs,
        save_best_model_weights=args.save_best_model_weights
    )
