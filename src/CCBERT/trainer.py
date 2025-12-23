"""
trainer.py

This module manages the training lifecycle of the Relation Extraction models.
It handles:
    1. Data preparation (Pandas DataFrame -> Hugging Face Dataset).
    2. Tokenization using the appropriate tokenizer from the factory function.
    3. Model training using the Hugging Face Trainer API.
    4. Evaluation of specific checkpoints (e.g., Epoch 3, 5, 10, 20) and the best model.
    5. Cleanup of intermediate checkpoints to save disk space.

It relies on:
    - src.models.get_model_and_tokenizer: For model initialization.
    - src.utils: For checkpoint management and evaluation helpers.

@Author: Mei Yoshikawa
"""

import os
import glob
import numpy as np
import pandas as pd
import evaluate
from datasets import Dataset
from transformers import (
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback,
    DataCollatorWithPadding
)

# Import from local modules
from .models import get_model_and_tokenizer
from .utils import (
    load_model_weights, 
    evaluate_and_save_metrics, 
    cleanup_checkpoints
)


def run_training(
    df_train,
    df_eval,
    df_test,
    df_pubmed,
    model_type,
    model_name_or_path,
    output_dir,
    num_labels,
    seed,
    learning_rate = 1e-5,
    warmup_ratio = 0.1,
    batch_size = 16,
    num_train_epochs = 20,
    weight_decay = 1e-4,
    dropout_rate = 0.1,
    target_epochs = None
):
    """
    Executes the full training and evaluation pipeline.

    Args:
        df_train (pd.DataFrame): Training data.
        df_eval (pd.DataFrame): Validation data for early stopping.
        df_test (pd.DataFrame): Test data 1 (e.g., SemMed).
        df_pubmed (pd.DataFrame): Test data 2 (e.g., PubMed/Hoge).
        model_type (str): "rbert", "cell-rbert", "cls", or "cls-boundary".
        model_name_or_path (str): Pretrained model name (e.g., PubMedBERT).
        output_dir (str): Directory to save results.
        num_labels (int): Number of classification labels.
        seed (int): Random seed.
        learning_rate (float): Learning rate.
        warmup_ratio (float): Warmup ratio.
        batch_size (int): Batch size.
        num_train_epochs (int): Max epochs.
        weight_decay (float): Weight decay.
        dropout_rate (float): Dropout rate.
    """
    
    # 1. Initialize Model & Tokenizer
    print(f"\n[Info] Initializing model: {model_type} (Base: {model_name_or_path})")
    
    model, tokenizer = get_model_and_tokenizer(
        model_type=model_type,
        model_name_or_path=model_name_or_path,
        num_labels=num_labels,
        dropout_rate=dropout_rate
    )

    # 2. Prepare Datasets
    print("[Info] Tokenizing datasets...")
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"], 
            truncation=True, 
            padding="max_length", 
            max_length=256
        )

    # Convert pandas -> HF Dataset -> Tokenized -> Torch Format
    datasets_map = {
        "train": df_train,
        "eval": df_eval,
        "test_semmed": df_test,
        "test_pubmed": df_pubmed
    }
    
    tokenized_datasets = {}
    
    for name, df in datasets_map.items():
        ds = Dataset.from_pandas(df)
        tokenized_ds = ds.map(tokenize_fn, batched=True)
        tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        tokenized_datasets[name] = tokenized_ds

    # 3. Setup Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=None,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        seed=seed,
        data_seed=seed,
        log_level="info",
        report_to="none",
        optim="schedule_free_radam"
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 4. Training
    print(f"[Info] Starting training in {output_dir}...")
    trainer.train()
    
    # Save Best Model
    best_model_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_dir)
    
    # Save Training Log
    log_df = pd.DataFrame(trainer.state.log_history)
    log_df.to_csv(os.path.join(output_dir, "training_log.csv"), index=False)

    # 5. Evaluation Loop (Target Epochs & Best Model)
    steps_per_epoch = len(tokenized_datasets["train"]) // batch_size
    
    # Find saved checkpoints
    saved_checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    checkpoint_map = {} 
    for cp_path in saved_checkpoints:
        try:
            step_num = int(cp_path.split("-")[-1])
            checkpoint_map[step_num] = cp_path
        except ValueError:
            continue

    kept_checkpoints = set()

    # Define datasets to evaluate
    eval_targets = [
        ("semmed", tokenized_datasets["test_semmed"]),
        ("pubmed", tokenized_datasets["test_pubmed"])
    ]

    print("\n[Info] evaluating target checkpoints...")

    # A. Evaluate Target Epochs
    for ep in target_epochs:
        target_step = ep * steps_per_epoch
        
        # Find the closest existing checkpoint step
        closest_step = None
        min_diff = float('inf')
        
        for step in checkpoint_map.keys():
            # Allow slight deviation (e.g. due to gradient accumulation)
            if abs(step - target_step) < (steps_per_epoch * 0.1) + 5: 
                if abs(step - target_step) < min_diff:
                    min_diff = abs(step - target_step)
                    closest_step = step
        
        if closest_step is None:
            print(f"  - Epoch {ep} (Step ~{target_step}) not reached.")
            continue
        
        cp_path = checkpoint_map[closest_step]
        kept_checkpoints.add(cp_path)
        
        # Load weights and evaluate
        if load_model_weights(model, cp_path):
            for name, ds in eval_targets:
                evaluate_and_save_metrics(trainer, ds, output_dir, file_prefix=name, suffix_name=f"ep{ep}")
        else:
            print(f"  - Failed to load weights from {cp_path}")

    # B. Evaluate Best Model
    if os.path.exists(best_model_dir):
        if load_model_weights(model, best_model_dir):
             for name, ds in eval_targets:
                evaluate_and_save_metrics(trainer, ds, output_dir, file_prefix=name, suffix_name="best")
    else:
        print("  - Best model directory not found.")

    # 6. Cleanup
    cleanup_checkpoints(output_dir, list(kept_checkpoints))
    print(f"[Info] Training finished. Results saved to {output_dir}")