"""
utils.py

Text representation helpers (boundary tags <E0>...</E0> vs. replacement tokens
[CELL0]) and training helpers (checkpoint loading/cleanup, metric saving).

The two representations are built directly from the (entity_0_start, entity_0_end,
entity_1_start, entity_1_end) offsets published in the dataset -- see
reconstruct.py. The regex-based helpers below operate on text that already
contains <E0>/<E1> tags (produced by reconstruct.py), not on raw annotation
markup, since the public dataset ships character offsets rather than raw text.
"""

import glob
import os
import re
import shutil

import pandas as pd
import torch
from safetensors.torch import load_file as load_safetensors


def normalize_boundary_tags(sentence):
    """Ensures <E0> is the first-occurring tag, swapping <E0>/<E1> if needed."""
    m = re.search(r"<E[01]>", sentence)
    if not m or m.group() == "<E0>":
        return sentence
    tmp_tag, tmp_end = "<TMP_SWAP_TAG>", "</TMP_SWAP_TAG>"
    sentence = sentence.replace("<E0>", tmp_tag).replace("</E0>", tmp_end)
    sentence = sentence.replace("<E1>", "<E0>").replace("</E1>", "</E0>")
    return sentence.replace(tmp_tag, "<E1>").replace(tmp_end, "</E1>")


def convert_boundary_to_replacement(sentence):
    """Converts <E0>...</E0>/<E1>...</E1> tags to [CELL0]/[CELL1] tokens."""
    sentence = re.sub(r"<E0>.*?</E0>", "[CELL0]", sentence)
    return re.sub(r"<E1>.*?</E1>", "[CELL1]", sentence)


def normalize_replacement_tokens(sentence):
    """Ensures [CELL0] is the first-occurring token, swapping if needed."""
    m = re.search(r"\[CELL[01]\]", sentence)
    if not m or m.group() == "[CELL0]":
        return sentence
    tmp_token = "[TMP_CELL_TOKEN]"
    return (
        sentence.replace("[CELL0]", tmp_token)
        .replace("[CELL1]", "[CELL0]")
        .replace(tmp_token, "[CELL1]")
    )


def cleanup_checkpoints(output_dir, kept_checkpoints):
    """Removes checkpoint-* directories under output_dir except those listed."""
    for cp_path in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        if cp_path not in kept_checkpoints:
            shutil.rmtree(cp_path)


def load_model_weights(model, model_path, device=None):
    """Loads weights from a directory containing model.safetensors or
    pytorch_model.bin into `model`. Returns True on success."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    safetensors_path = os.path.join(model_path, "model.safetensors")
    bin_path = os.path.join(model_path, "pytorch_model.bin")

    state_dict = None
    if os.path.exists(safetensors_path):
        state_dict = load_safetensors(safetensors_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location=device)

    if state_dict is None:
        return False

    mapped_state_dict = {}
    for k, v in state_dict.items():
        if k.endswith(".gamma"):
            mapped_state_dict[k.replace(".gamma", ".weight")] = v
        elif k.endswith(".beta"):
            mapped_state_dict[k.replace(".beta", ".bias")] = v
        else:
            mapped_state_dict[k] = v
    model.load_state_dict(mapped_state_dict, strict=False)
    return True


def evaluate_and_save_metrics(trainer, dataset, output_dir, file_prefix, suffix_name):
    """Runs trainer.predict(dataset) and writes both metrics and raw predictions
    to CSV files under output_dir."""
    result = trainer.predict(dataset)

    pd.DataFrame([result.metrics]).to_csv(
        os.path.join(output_dir, f"{file_prefix}_metrics_{suffix_name}.csv"),
        index=False,
    )

    if len(result.predictions.shape) > 1:
        cols = [f"class_{i}" for i in range(result.predictions.shape[1])]
    else:
        cols = ["prediction"]
    pd.DataFrame(result.predictions, columns=cols).to_csv(
        os.path.join(output_dir, f"{file_prefix}_predictions_{suffix_name}.csv"),
        index=False,
    )
