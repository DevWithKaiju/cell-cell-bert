"""
scripts/02_run_cpt.py

Continued pretraining (CPT) of PubMedBERT with cell-entity markup, from
data/cpt/relations_resolved.csv (scripts/01_reconstruct_dataset.py). See
src/cc_bert/cpt.py for the reproducibility caveat on the "replacement"
strategy's plain-text half.

Example:
    python scripts/02_run_cpt.py --strategy replacement
    python scripts/02_run_cpt.py --strategy boundary
"""

import argparse
import os
import sys

sys.path.append(
    os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(__file__), ".."))
)

import pandas as pd

from src.cc_bert.cpt import build_cpt_corpus, run_cpt_training


def main():
    parser = argparse.ArgumentParser(
        description="Continued pretraining (CPT) of PubMedBERT with cell-entity markup."
    )
    parser.add_argument(
        "--strategy", type=str, required=True, choices=["replacement", "boundary"]
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    )
    parser.add_argument(
        "--relations_csv", type=str, default="data/cpt/relations_resolved.csv"
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revert_frac", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=2)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join("results", "cpt", args.strategy)

    df_resolved = pd.read_csv(args.relations_csv)
    df_corpus = build_cpt_corpus(
        df_resolved, args.strategy, revert_frac=args.revert_frac, revert_seed=args.seed
    )
    print(
        f"corpus rows: {len(df_corpus)} (revert_frac={args.revert_frac} -> the reverted half uses the "
        f"plain sentence reconstructed from the same PubMed abstract, not the original unpublished "
        f"hit_cells.db source -- see src/cc_bert/cpt.py)"
    )

    run_cpt_training(
        df_corpus,
        args.strategy,
        args.base_model_path,
        output_dir,
        seed=args.seed,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    main()
