"""
experiment_runner.py

CLI entry point that runs one (model_type, seed) fine-tuning experiment. Reads
the data produced by scripts/01_reconstruct_dataset.py:
    data/semmed/<seed>/{train,test}_{boundary,replacement}.csv
    data/pubmed/<seed>/test_{boundary,replacement}.csv
each with columns: id, sentence, label.
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from .models import VALID_MODEL_TYPES
from .trainer import run_training


def get_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune a cell-cell relation extraction model"
    )

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help=f"One of {VALID_MODEL_TYPES}, optionally suffixed with '-CPT' to start from --cpt_model_path.",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        help="Base pretrained model used when model_type does not end in '-CPT'.",
    )
    parser.add_argument(
        "--cpt_model_path",
        type=str,
        default=None,
        help="Continued-pretraining checkpoint, required when model_type ends in '-CPT' "
        "(e.g. mizuno-group/ccbert-R-pretrained, or a local checkpoint from scripts/02_run_cpt.py).",
    )

    parser.add_argument("--seed", type=int, default=22, help="Random seed.")
    parser.add_argument(
        "--size",
        type=int,
        default=300,
        help="Number of SemMed training rows to use (head).",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Directory produced by 01_reconstruct_dataset.py.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results/ft",
        help="Directory to save outputs.",
    )

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument(
        "--target_epochs",
        type=int,
        nargs="+",
        default=[3, 5, 10, 20],
        help="Additional epoch checkpoints to evaluate (besides the early-stopped best model).",
    )
    parser.add_argument("--save_best_model_weights", action="store_true", default=True)

    return parser


def run_experiment(args):
    core_type = next(
        (t for t in VALID_MODEL_TYPES if args.model_type.startswith(t)), None
    )
    if core_type is None:
        raise ValueError(
            f"Invalid model_type: {args.model_type}. Must start with one of {VALID_MODEL_TYPES}"
        )
    suffix = "replacement" if core_type.startswith("R-") else "boundary"

    if args.model_type.endswith("-CPT"):
        if not args.cpt_model_path:
            raise ValueError("--cpt_model_path must be provided for CPT models")
        current_model_name = args.cpt_model_path
        print(f"[Config] Selected CPT weights for {args.model_type}")
    else:
        current_model_name = args.base_model_path
        print(f"[Config] Selected base weights for {args.model_type}")

    output_dir = os.path.join(args.output_root, args.model_type, str(args.seed))
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(" Start Fine-tuning Experiment")
    print(f" model_type    : {args.model_type} (core={core_type}, data={suffix})")
    print(f" model path    : {current_model_name}")
    print(f" seed={args.seed}  size={args.size}  lr={args.lr}")
    print(f" output_dir    : {output_dir}")
    print(f"{'=' * 60}\n")

    train_path = os.path.join(
        args.data_root, "semmed", str(args.seed), f"train_{suffix}.csv"
    )
    test_path = os.path.join(
        args.data_root, "semmed", str(args.seed), f"test_{suffix}.csv"
    )
    pubmed_path = os.path.join(
        args.data_root, "pubmed", str(args.seed), f"test_{suffix}.csv"
    )

    df_train_full = pd.read_csv(train_path).head(args.size)
    df_test = pd.read_csv(test_path)
    df_pubmed = pd.read_csv(pubmed_path)

    df_train, df_eval = train_test_split(
        df_train_full,
        test_size=0.1,
        random_state=args.seed,
        stratify=df_train_full["label"],
    )

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
        save_best_model_weights=args.save_best_model_weights,
    )


if __name__ == "__main__":
    run_experiment(get_parser().parse_args())
