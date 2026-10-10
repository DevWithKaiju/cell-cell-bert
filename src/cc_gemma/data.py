"""
data.py

Data loading for decoder-based evaluation and LoRA SFT. Reads the sentence-level
CSVs written by scripts/01_reconstruct_dataset.py:
    data/<dataset>/<seed>/{train,test}_{boundary,replacement}.csv
with columns: id, sentence, label.
"""

from __future__ import annotations

import os

import pandas as pd
from datasets import Dataset

from .prompts import build_prompt

SUFFIX_BY_PRESENTATION = {"boundary": "boundary", "replacement": "replacement"}


def load_dataframe(
    data_root: str, dataset: str, seed: int, presentation: str, split: str
) -> pd.DataFrame:
    """Loads the given split ("train" / "test") as a DataFrame."""
    suffix = SUFFIX_BY_PRESENTATION[presentation]
    path = os.path.join(data_root, dataset, str(seed), f"{split}_{suffix}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected data file not found: {path}")

    df = pd.read_csv(path)
    missing = {"sentence", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["sentence", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(float).astype(int)
    bad = set(df["label"].unique()) - {0, 1}
    if bad:
        raise ValueError(f"{path} contains non-binary labels: {sorted(bad)}")
    return df


def build_prompt_dataset(
    df: pd.DataFrame, prompt_style: str, presentation: str, tokenizer
) -> Dataset:
    """Converts a DataFrame into a prompt-completion Dataset (plain strings) for
    TRL's SFTTrainer. Each record is {"prompt": <chat-template-rendered string>,
    "completion": "0" or "1"}.

    Plain strings (not the "conversational" messages format) are used because
    chat templates can render the final assistant-turn prefix differently
    between a prompt-only render (add_generation_prompt=True) and a full
    conversation render -- which would break TRL's prompt/completion prefix
    matching and mask out every training example. With plain strings, `prompt`
    is guaranteed to be an exact prefix of `prompt + completion`.
    """
    records = [
        {
            "prompt": tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": build_prompt(
                            prompt_style, presentation, row["sentence"]
                        ),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            ),
            "completion": str(int(row["label"])),
        }
        for _, row in df.iterrows()
    ]
    return Dataset.from_list(records)
