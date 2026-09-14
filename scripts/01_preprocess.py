"""
scripts/01_preprocess.py

Description:
    Builds the per-seed, boundary-tagged and replacement-token fine-tuning/evaluation
    datasets from the raw PubMed and SemMed exports, via `src.cc_bert.preprocess`.

    Expects raw input files at:
        {data_dir}/pubmed/data.csv
        {data_dir}/semmed/data.csv

    Writes, for each strategy ("boundary" / "replacement"):
        {data_dir}/pubmed/{seed}/test_{strategy}.csv
        {data_dir}/semmed/{seed}/{train,test}_{strategy}.csv

Usage:
    python scripts/01_preprocess.py --seed 22 --data_dir ./data

@Author: Mei Yoshikawa
"""

import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.cc_bert.preprocess import run_preprocessing


def parse_args():
    parser = argparse.ArgumentParser(description="Build per-seed PubMed/SemMed fine-tuning datasets")
    parser.add_argument("--seed", type=int, default=22, help="Seed for directory name and train/test split")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root data directory")
    return parser.parse_args()


def main():
    args = parse_args()
    run_preprocessing(data_dir=args.data_dir, seed=args.seed)


if __name__ == "__main__":
    main()
