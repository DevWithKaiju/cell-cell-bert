"""
scripts/03_train.py

This script serves as the main entry point for running fine-tuning experiments
for Relation Extraction (RE). All resolution logic (architecture selection from
--model_type, data path resolution, CPT-vs-base weight selection) lives in
`src.cc_bert.finetune`; this script only parses CLI arguments and invokes it.

Supports the following core architectures (via --model_type prefix): B-HYB, R-HYB,
B-ENT, R-ENT, R-CLS, B-CLS. Append "-CPT" (e.g. "R-HYB-CPT") to start from
--cpt_model_path instead of --base_model_path. Data-size sweeps (see
scripts/03_run_finetuning.sh) are just different --size values -- there is no
separate script for them.

Usage:
    # Option 1: Run via the wrapper shell script (sweeps multiple configurations)
    scripts/03_run_finetuning.sh

    # Option 2: Direct execution
    python scripts/03_train.py \
        --model_type R-ENT-base \
        --base_model_path "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

@Author: Mei Yoshikawa
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.cc_bert.finetune import get_parser, run_experiment


def main():
    parser = get_parser()
    parser.add_argument(
        "--append_model_name", action="store_true",
        help="Append a short form of --base_model_path to the output directory name "
             "(useful when sweeping the same --model_type across multiple base models)."
    )
    args = parser.parse_args()
    run_experiment(args, append_model_name=args.append_model_name)


if __name__ == "__main__":
    main()
