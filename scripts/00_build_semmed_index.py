"""
scripts/00_build_semmed_index.py

Builds a local SQLite index of SemMedDB sentence text, restricted to the
pred_ids referenced by semmed_relations, from a local copy of the official
SemMedDB distribution (PREDICATION/SENTENCE/PREDICATION_AUX tables). SemMedDB
requires its own separate license/download -- see
https://lhncbc.nlm.nih.gov/ii/tools/SemRep_SemMedDB_SKR/SemMedDB_download.html

Example:
    python scripts/00_build_semmed_index.py \\
        --predication /path/to/PREDICATION.csv.gz \\
        --sentence /path/to/SENTENCE.csv.gz \\
        --predication_aux /path/to/PREDICATION_AUX.csv.gz
"""

import argparse
import os
import sys

sys.path.append(
    os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(__file__), ".."))
)

from src.cc_bert.reconstruct import load_hf_configs
from src.cc_bert.semmed_db import build_semmed_index


def main():
    parser = argparse.ArgumentParser(
        description="Build a local SemMedDB sentence-lookup index."
    )
    parser.add_argument(
        "--predication", type=str, required=True, help="Path to PREDICATION.csv[.gz]"
    )
    parser.add_argument(
        "--sentence", type=str, required=True, help="Path to SENTENCE.csv[.gz]"
    )
    parser.add_argument(
        "--predication_aux",
        type=str,
        required=True,
        help="Path to PREDICATION_AUX.csv[.gz]",
    )
    parser.add_argument("--index_db", type=str, default="data/raw/semmed_index.db")
    parser.add_argument(
        "--has_header",
        action="store_true",
        help="Set this if your CSV exports already have a header row with the official column names "
        "(PREDICATION_ID, SENTENCE_ID, ...); otherwise the default (headerless, NLM's documented "
        "VER43 column order) is assumed -- see semmed_db.py if your version differs.",
    )
    args = parser.parse_args()

    configs = load_hf_configs()
    target_pred_ids = set(configs["semmed_relations"]["pred_id"])
    print(f"restricting index to {len(target_pred_ids)} referenced pred_ids")

    build_semmed_index(
        args.predication,
        args.sentence,
        args.predication_aux,
        args.index_db,
        target_pred_ids=target_pred_ids,
        has_header=args.has_header,
    )


if __name__ == "__main__":
    main()
