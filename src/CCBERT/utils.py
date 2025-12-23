"""
utils.py

This module provides utility functions for:
1. Text Preprocessing: 
   - Strategy A: Boundary Marking (e.g., <E0>target</E0>)
   - Strategy B: Entity Replacement (e.g., [CELL0])
2. Training Helpers: 
   - Checkpoint management, model weight loading (safetensors/bin), and metric logging.

@Author: Mei Yoshikawa
"""

import glob
import os
import re
import shutil
from typing import List, Any, Optional

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file as load_safetensors


# 1. Preprocessing Strategies

# Strategy A: Boundary Marking (<E0>...<E0>)

def apply_boundary_tags(sentence):
    """
    Converts raw text with specific brackets to boundary tags.
    Input format:  "Effect of 【Drug A】 on 【Cell B】..."
    Output format: "Effect of <E0>Drug A</E0> on <E1>Cell B</E1>..."
    
    Tags are incremented as E0, E1... based on their appearance order.

    Args:
        sentence (str): The input text containing 【...】 formatted entities.

    Returns:
        str: Text with <En> tags.
    """
    counter = [0]
    # Matches content inside 【】 and optional suffixes like 's' or 'es'
    regex = r'【(.*?)】((?:es|s)?)'

    def replacer(match):
        inner_text = match.group(1)
        suffix = match.group(2)
        combined_text = inner_text + suffix
        count = counter[0]
        counter[0] += 1
        return f'<E{count}>{combined_text}</E{count}>'

    return re.sub(regex, replacer, sentence)

def normalize_boundary_tags(sentence):
    """
    Standardizes the order of entity tags so that <E0> always appears before <E1>.
    If <E1> appears first in the sentence, the tags are swapped.

    Args:
        sentence (str): Input text with <E0>/<E1> tags.

    Returns:
        str: Text where the first occurring entity is guaranteed to be <E0>.
    """
    m = re.search(r'<E[01]>', sentence)
    if not m:
        return sentence
    
    first_tag = m.group()
    if first_tag == '<E0>':
        return sentence
    
    # Swap E0 <-> E1 using a unique temporary placeholder to avoid collision
    tmp_tag = '<TMP_SWAP_TAG>'
    tmp_end = '</TMP_SWAP_TAG>'
    
    sentence = sentence.replace('<E0>', tmp_tag).replace('</E0>', tmp_end)
    sentence = sentence.replace('<E1>', '<E0>').replace('</E1>', '</E0>')
    sentence = sentence.replace(tmp_tag, '<E1>').replace(tmp_end, '</E1>')
    return sentence

# Strategy B: Entity Replacement ([CELL0])

def convert_boundary_to_replacement(sentence):
    """
    Converts boundary tags (<E0>...</E0>) to special replacement tokens ([CELL0]).
    
    Dependency:
        This function assumes `apply_boundary_tags` has already been applied.

    Args:
        sentence (str): Input text with <E0>/<E1> tags.

    Returns:
        str: Text with [CELL0]/[CELL1] tokens replacing the tagged spans.
    """
    sentence = re.sub(r'<E0>.*?</E0>', '[CELL0]', sentence)
    sentence = re.sub(r'<E1>.*?</E1>', '[CELL1]', sentence)
    return sentence

def normalize_replacement_tokens(sentence):
    """
    Standardizes the order of replacement tokens so that [CELL0] always appears before [CELL1].
    
    Args:
        sentence (str): Input text with [CELLn] tokens.

    Returns:
        str: Text where the first occurring token is [CELL0].
    """
    m = re.search(r'\[CELL[01]\]', sentence)
    if not m:
        return sentence
    
    if m.group() == '[CELL1]':
        tmp_token = '[TMP_CELL_TOKEN]'
        sentence = sentence.replace('[CELL0]', tmp_token)\
                           .replace('[CELL1]', '[CELL0]')\
                           .replace(tmp_token, '[CELL1]')
    return sentence

def clean_replacement_suffix(sentence):
    """
    Removes linguistic suffixes (like 's', 'es') that might remain immediately 
    following replacement tokens due to tokenization artifacts.
    
    Example: "[CELL0]s" -> "[CELL0]"

    Args:
        sentence (str): Input text.

    Returns:
        str: Cleaned text.
    """
    sentence = re.sub(r'\[CELL([01])\]s\b', r'[CELL\1]', sentence)
    sentence = re.sub(r'\[CELL([01])\]es\b', r'[CELL\1]', sentence)
    return sentence

# 2. Training & Evaluation Helpers

def cleanup_checkpoints(output_dir, kept_checkpoints):
    """
    Removes unnecessary checkpoint directories to save disk space.
    Only keeps checkpoints specified in `kept_checkpoints`.

    Args:
        output_dir (str): The directory containing checkpoints.
        kept_checkpoints (List[str]): List of checkpoint paths to preserve.
    """
    saved_checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    for cp_path in saved_checkpoints:
        if cp_path not in kept_checkpoints:
            shutil.rmtree(cp_path)

def load_model_weights(model, model_path, device=None):
    """
    Loads model weights from a directory, supporting both 'safetensors' (preferred) 
    and 'pytorch_model.bin' formats.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        model_path (str): Directory containing the weight files.
        device (str, optional): Device to map weights to. Defaults to "cpu".

    Returns:
        bool: True if weights were loaded successfully, False otherwise.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    safetensors_path = os.path.join(model_path, "model.safetensors")
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    
    state_dict = None
    if os.path.exists(safetensors_path):
        state_dict = load_safetensors(safetensors_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location=device)
    
    if state_dict is not None:
        model.load_state_dict(state_dict)
        return True
    return False

def evaluate_and_save_metrics(trainer, dataset, output_dir, file_prefix, suffix_name):
    """
    Runs prediction using the trainer and saves both metrics and raw predictions to CSV files.

    Args:
        trainer (Any): The Hugging Face Trainer instance.
        dataset (Any): The dataset to evaluate.
        output_dir (str): Directory to save the output CSV files.
        file_prefix (str): Prefix for the output filenames (e.g., 'test', 'eval').
        suffix_name (str): Suffix/ID for the output filenames (e.g., 'epoch1', 'best').
    """
    # Run prediction
    result = trainer.predict(dataset)
    
    # 1. Save Metrics (Accuracy, Loss, etc.)
    metrics_df = pd.DataFrame([result.metrics])
    metrics_path = os.path.join(output_dir, f"{file_prefix}_metrics_{suffix_name}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    # 2. Save Raw Predictions (Logits/Probabilities)
    # Automatically handles binary vs. multi-class output shapes
    if len(result.predictions.shape) > 1:
        cols = [f"class_{i}" for i in range(result.predictions.shape[1])]
    else:
        cols = ["prediction"]
        
    df_pred = pd.DataFrame(result.predictions, columns=cols)
    pred_path = os.path.join(output_dir, f"{file_prefix}_predictions_{suffix_name}.csv")
    df_pred.to_csv(pred_path, index=False)