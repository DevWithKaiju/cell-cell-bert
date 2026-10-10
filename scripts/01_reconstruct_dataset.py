"""
scripts/01_reconstruct_dataset.py

Rebuilds all sentence-level training/eval data from the published offset-only
HF dataset (DevWithKaiju/cell-cell-communication) plus the local PubMed/SemMedDB
indexes built by the 00_* scripts:
    data/semmed/<seed>/{train,test}_{boundary,replacement}.csv   (id, sentence, label)
    data/pubmed/<seed>/test_{boundary,replacement}.csv            (id, sentence, label)
    data/cpt/relations_resolved.csv                                (id, pmid, sentence_plain,
                                                                      sentence_boundary, sentence_replacement)

Run the 00_* scripts first. Use --sample_check to spot-check a handful of
reconstructed sentences before committing to a full run.

Example:
    python scripts/01_reconstruct_dataset.py --sample_check 5
    python scripts/01_reconstruct_dataset.py
"""

import argparse
import os
import sys

sys.path.append(
    os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(__file__), ".."))
)

from src.cc_bert.pubmed_ftp import load_abstracts
from src.cc_bert.reconstruct import (
    PRESENTATIONS,
    load_hf_configs,
    reconstruct_cpt_relations,
    reconstruct_pubmed_like_relations,
    reconstruct_semmed_relations,
    verify_semmed_offsets,
    write_seed_splits_for_presentation,
)
from src.cc_bert.semmed_db import load_semmed_sentences


def _print_sample(df, n, label, columns):
    print(f"\n--- sample_check: {label} ({min(n, len(df))} of {len(df)} rows) ---")
    for _, row in df.head(n).iterrows():
        parts = "  ".join(f"{c}={row[c]!r}" for c in columns)
        print(f"  {parts}")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct sentence-level data from the published HF dataset."
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument(
        "--pubmed_index_db", type=str, default="data/raw/pubmed_abstracts.db"
    )
    parser.add_argument(
        "--semmed_index_db", type=str, default="data/raw/semmed_index.db"
    )
    parser.add_argument("--skip_pubmed", action="store_true")
    parser.add_argument("--skip_semmed", action="store_true")
    parser.add_argument("--skip_cpt", action="store_true")
    parser.add_argument(
        "--sample_check",
        type=int,
        default=0,
        help="Print this many reconstructed sentences per source/presentation and exit, instead of "
        "writing the full dataset -- use this to spot-check reconstruction before a full run.",
    )
    args = parser.parse_args()

    configs = load_hf_configs()

    if not args.skip_pubmed:
        print("\n=== PubMed ===")
        pmids = set(configs["pubmed_relations"]["pmid"])
        abst_of = load_abstracts(args.pubmed_index_db, pmids)
        print(f"pmid -> abstract resolved: {len(abst_of)} / {len(pmids)}")

        for presentation in PRESENTATIONS:
            df_resolved, report = reconstruct_pubmed_like_relations(
                configs["pubmed_relations"], abst_of, presentation
            )
            n_ok = sum(1 for _, status in report if status == "ok")
            print(f"[pubmed/{presentation}] resolved {n_ok} / {len(report)} rows")
            if args.sample_check:
                _print_sample(
                    df_resolved,
                    args.sample_check,
                    f"pubmed/{presentation}",
                    ["id", "sentence"],
                )
            else:
                write_seed_splits_for_presentation(
                    df_resolved,
                    configs["pubmed_seed_splits"],
                    args.data_root,
                    "pubmed",
                    presentation,
                )

    if not args.skip_semmed:
        print("\n=== SemMed ===")
        pred_ids = set(configs["semmed_relations"]["pred_id"])
        semmed_lookup = load_semmed_sentences(args.semmed_index_db, pred_ids)
        print(f"pred_id -> sentence resolved: {len(semmed_lookup)} / {len(pred_ids)}")

        n_checked, n_matched = verify_semmed_offsets(
            configs["semmed_relations"], semmed_lookup, sample_size=2000
        )
        if n_checked:
            print(
                f"offset verification (sample of {n_checked}): {n_matched}/{n_checked} match "
                f"({100 * n_matched / n_checked:.1f}%) -- a low rate usually means your local SemMedDB "
                f"copy is a different version than VER43_R"
            )

        for presentation in PRESENTATIONS:
            df_resolved, report = reconstruct_semmed_relations(
                configs["semmed_relations"], semmed_lookup, presentation
            )
            n_ok = sum(1 for _, status in report if status == "ok")
            print(f"[semmed/{presentation}] resolved {n_ok} / {len(report)} rows")
            if args.sample_check:
                _print_sample(
                    df_resolved,
                    args.sample_check,
                    f"semmed/{presentation}",
                    ["id", "sentence"],
                )
            else:
                write_seed_splits_for_presentation(
                    df_resolved,
                    configs["semmed_seed_splits"],
                    args.data_root,
                    "semmed",
                    presentation,
                )

    if not args.skip_cpt:
        print("\n=== CPT ===")
        pmids = set(configs["cpt_relations"]["pmid"])
        abst_of = load_abstracts(args.pubmed_index_db, pmids)
        print(f"pmid -> abstract resolved: {len(abst_of)} / {len(pmids)}")

        df_resolved, report = reconstruct_cpt_relations(
            configs["cpt_relations"], abst_of
        )
        n_ok = sum(1 for _, status in report if status == "ok")
        print(f"[cpt] resolved {n_ok} / {len(report)} rows")

        if args.sample_check:
            _print_sample(
                df_resolved,
                args.sample_check,
                "cpt",
                ["id", "sentence_plain", "sentence_boundary", "sentence_replacement"],
            )
        else:
            out_dir = os.path.join(args.data_root, "cpt")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "relations_resolved.csv")
            df_resolved.to_csv(out_path, index=False)
            print(f"wrote {out_path} ({len(df_resolved)} rows)")


if __name__ == "__main__":
    main()
