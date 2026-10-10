"""
scripts/00_download_pubmed_baseline.py

Downloads the PubMed annual baseline from NLM (the same release the published
dataset's PubMed-derived offsets were computed against). ~40GB / 1000+ files
for a full year -- use --limit for a quick smoke test first.

Example:
    python scripts/00_download_pubmed_baseline.py --limit 5
    python scripts/00_download_pubmed_baseline.py
"""

import argparse
import os
import sys

sys.path.append(
    os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(__file__), ".."))
)

from src.cc_bert.pubmed_ftp import download_baseline


def main():
    parser = argparse.ArgumentParser(
        description="Download the PubMed annual baseline from NLM."
    )
    parser.add_argument("--dest_dir", type=str, default="data/raw/pubmed_baseline")
    parser.add_argument(
        "--year", type=str, default="25", help="2-digit baseline year code."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only download the first N baseline files -- use this for a quick smoke test "
        "before committing to the full download.",
    )
    args = parser.parse_args()

    paths = download_baseline(args.dest_dir, year=args.year, limit=args.limit)
    print(f"downloaded/verified {len(paths)} files in {args.dest_dir}")


if __name__ == "__main__":
    main()
