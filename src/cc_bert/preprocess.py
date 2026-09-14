"""
preprocess.py

Builds the per-seed, per-strategy fine-tuning/evaluation datasets consumed by
02_cpt.py / 03_train.py from the raw PubMed and SemMed exports.

For each strategy:
    - "boundary":    entities wrapped in <E0>...</E0> tags (see tagging.py)
    - "replacement": entities replaced with [CELL0]/[CELL1] tokens

PubMed data is balanced and split into a single evaluation set (no train split;
used only to measure generalization to naturally occurring sentences).
SemMed data is balanced and split into train/test sets used for fine-tuning.

@Author: Mei Yoshikawa
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from .tagging import apply_boundary_tags, normalize_boundary_tags, convert_to_replacement


def process_pubmed(data_dir, seed):
    """
    Builds the PubMed evaluation set (boundary and replacement variants) for a given seed.

    Reads `{data_dir}/pubmed/data.csv` (columns: Sentence, Selected_Term_A, Selected_Term_B,
    label, pmid, sent_id, ID_A, ID_B), restores <E0>/<E1> boundary tags around the annotated
    entity terms, balances labels (75 samples per class), and writes both representations to
    `{data_dir}/pubmed/{seed}/test_{boundary,replacement}.csv`.

    Args:
        data_dir (str): Root data directory.
        seed (int): Random seed controlling label-balanced sampling and used as the output subdirectory name.
    """
    raw_path = os.path.join(data_dir, "pubmed", "data.csv")
    if not os.path.exists(raw_path):
        print(f"[Warning] PubMed raw data not found at {raw_path}")
        return

    df = pd.read_csv(raw_path)

    # Restore boundary tags: [CELL0] -> <E0>Selected_Term_A</E0>, [CELL1] -> <E1>Selected_Term_B</E1>
    def restore_boundary(row):
        sent = str(row["Sentence"])
        term_a = str(row["Selected_Term_A"])
        term_b = str(row["Selected_Term_B"])
        sent = sent.replace("[CELL0]", f"<E0>{term_a}</E0>")
        sent = sent.replace("[CELL1]", f"<E1>{term_b}</E1>")
        return sent
    df["Sentence"] = df.apply(restore_boundary, axis=1).apply(normalize_boundary_tags)

    # Balance labels (75 each, shuffled by seed, restricted to the original 800 items)
    df_0 = df[df["label"] == 0].head(400).sample(n=75, random_state=seed)
    df_1 = df[df["label"] == 1].head(400).sample(n=75, random_state=seed)
    df_balanced = pd.concat([df_0, df_1]).sample(frac=1, random_state=seed)

    df_balanced["id"] = (
        df_balanced["pmid"].astype(str) + "-" +
        df_balanced["sent_id"].astype(str) + "-" +
        df_balanced["ID_A"].astype(str) + "-" +
        df_balanced["ID_B"].astype(str)
    )
    df_balanced = df_balanced.rename(columns={"Sentence": "sentence"})
    df_final = df_balanced[["id", "sentence", "label"]]

    out_dir = os.path.join(data_dir, "pubmed", str(seed))
    os.makedirs(out_dir, exist_ok=True)

    df_final_boundary = df_final.copy()
    df_final_replacement = df_final.copy()
    df_final_replacement["sentence"] = df_final_replacement["sentence"].apply(convert_to_replacement)

    df_final_boundary.to_csv(os.path.join(out_dir, "test_boundary.csv"), index=False)
    df_final_replacement.to_csv(os.path.join(out_dir, "test_replacement.csv"), index=False)
    print(f"[PubMed] Saved boundary and replacement test files to {out_dir}")


def process_semmed(data_dir, seed):
    """
    Builds the SemMed fine-tuning/test sets (boundary and replacement variants) for a given seed.

    Reads `{data_dir}/semmed/data.csv` (columns: not_cell, label, pred_id, ann_other_cleaned),
    filters to `not_cell == False`, balances labels (750 per class), applies boundary tags to
    the 【...】-annotated entities, splits into train (1350) / test (150), and writes both
    representations to `{data_dir}/semmed/{seed}/{train,test}_{boundary,replacement}.csv`.

    Args:
        data_dir (str): Root data directory.
        seed (int): Random seed controlling label-balanced sampling, the train/test split, and
            used as the output subdirectory name.
    """
    raw_path = os.path.join(data_dir, "semmed", "data.csv")
    if not os.path.exists(raw_path):
        print(f"[Warning] SemMed raw data not found at {raw_path}")
        return

    df = pd.read_csv(raw_path)

    # not_cell == False のもののみ抽出（文字列/bool どちらでも対応）
    df["not_cell_str"] = df["not_cell"].astype(str).str.strip().str.upper()
    df = df[df["not_cell_str"] == "FALSE"]
    print(f"[Debug] After not_cell == False filter, size is {len(df)}")

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df_0 = df[df["label"] == 0].head(750)
    df_1 = df[df["label"] == 1].head(750)
    print(f"[Debug] df_0 size: {len(df_0)}, df_1 size: {len(df_1)}")

    df_balanced = pd.concat([df_0, df_1])
    print(f"[Debug] balanced size: {len(df_balanced)}")

    if len(df_balanced) == 0:
        raise ValueError("Balanced dataframe is empty. Check filtering conditions.")

    df_balanced["id"] = df_balanced["pred_id"]
    df_balanced = df_balanced.rename(columns={"ann_other_cleaned": "sentence"})

    # 【】括弧を境界タグに変換し、出現順序を正規化する
    df_balanced["sentence"] = df_balanced["sentence"].apply(apply_boundary_tags).apply(normalize_boundary_tags)

    df_final_boundary = df_balanced[["id", "sentence", "label"]].copy()
    df_final_replacement = df_final_boundary.copy()
    df_final_replacement["sentence"] = df_final_replacement["sentence"].apply(convert_to_replacement)

    # 1350(train) と 150(test) に分割 (stratifyでラベル比率を維持)
    test_size = 150 if len(df_final_boundary) >= 1500 else 0.1

    df_train_boundary, df_test_boundary = train_test_split(
        df_final_boundary, test_size=test_size, stratify=df_final_boundary["label"], random_state=seed
    )
    df_train_replacement, df_test_replacement = train_test_split(
        df_final_replacement, test_size=test_size, stratify=df_final_replacement["label"], random_state=seed
    )

    out_dir = os.path.join(data_dir, "semmed", str(seed))
    os.makedirs(out_dir, exist_ok=True)

    df_train_boundary.to_csv(os.path.join(out_dir, "train_boundary.csv"), index=False)
    df_test_boundary.to_csv(os.path.join(out_dir, "test_boundary.csv"), index=False)
    df_train_replacement.to_csv(os.path.join(out_dir, "train_replacement.csv"), index=False)
    df_test_replacement.to_csv(os.path.join(out_dir, "test_replacement.csv"), index=False)
    print(f"[SemMed] Saved boundary and replacement files to {out_dir}")


def run_preprocessing(data_dir, seed):
    """Runs both the PubMed and SemMed preprocessing steps for a given seed."""
    print("Starting preprocessing...")
    print(f"Data Dir: {data_dir}")
    print(f"Seed    : {seed}")

    process_pubmed(data_dir, seed)
    process_semmed(data_dir, seed)

    print("Preprocessing completed successfully.")
