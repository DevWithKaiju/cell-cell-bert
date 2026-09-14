"""
utils.py

Training & evaluation helper functions: checkpoint management, model weight
loading (safetensors/bin), and metric logging.

Entity-representation text preprocessing (boundary tags / replacement tokens)
lives in tagging.py.

@Author: Mei Yoshikawa
"""

import glob
import os
import shutil

import pandas as pd
import torch
from safetensors.torch import load_file as load_safetensors


# Training & Evaluation Helpers

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