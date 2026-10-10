"""
scripts/00_build_pubmed_index.py

Parses the downloaded baseline files (00_download_pubmed_baseline.py) into a
local SQLite index of pmid -> (title, abstract). By default, restricts the
index to only the pmids referenced by pubmed_relations/cpt_relations (fetched
from the Hub), which keeps both the run time and the index small.

Example:
    python scripts/00_build_pubmed_index.py
"""

import argparse
import os
import sys

sys.path.append(
    os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(__file__), ".."))
)

from src.cc_bert.pubmed_ftp import build_abstract_index
from src.cc_bert.reconstruct import load_hf_configs


def main():
    parser = argparse.ArgumentParser(
        description="Build a local pmid -> abstract SQLite index."
    )
    parser.add_argument("--xml_dir", type=str, default="data/raw/pubmed_baseline")
    parser.add_argument("--index_db", type=str, default="data/raw/pubmed_abstracts.db")
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Index every pmid found in the downloaded files instead of only the ones referenced "
        "by pubmed_relations/cpt_relations (slower, bigger index; mainly useful when --xml_dir "
        "only has a handful of files from a --limit smoke-test download).",
    )
    args = parser.parse_args()

    target_pmids = None
    if not args.no_filter:
        configs = load_hf_configs()
        target_pmids = set(configs["pubmed_relations"]["pmid"]) | set(
            configs["cpt_relations"]["pmid"]
        )
        print(f"restricting index to {len(target_pmids)} referenced pmids")

    build_abstract_index(args.xml_dir, args.index_db, target_pmids=target_pmids)


if __name__ == "__main__":
    main()
