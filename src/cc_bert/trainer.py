"""
trainer.py

Training lifecycle for the relation-extraction models in models.py:
    1. pandas DataFrame -> tokenized HF Dataset
    2. training via the HF Trainer API
    3. evaluation at specific epoch checkpoints (e.g. 3/5/10/20) and at the best
       (early-stopped) checkpoint
    4. checkpoint cleanup to save disk space
"""

import glob
import os

import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from .models import get_model_and_tokenizer
from .utils import cleanup_checkpoints, evaluate_and_save_metrics, load_model_weights


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
    learning_rate=1e-5,
    warmup_ratio=0.1,
    batch_size=16,
    num_train_epochs=20,
    weight_decay=1e-4,
    dropout_rate=0.1,
    target_epochs=None,
    save_best_model_weights=True,
):
    """Fine-tunes one (model_type, seed) run and evaluates it on both test sets.

    Args:
        df_train, df_eval: SemMed train/validation splits (columns: sentence, label).
        df_test: SemMed test split.
        df_pubmed: PubMed test-only split.
        model_type: one of cc_bert.models.VALID_MODEL_TYPES.
        model_name_or_path: base model id/path, or a CPT checkpoint.
        output_dir: directory to write checkpoints, logs, and evaluation CSVs.
        num_labels: number of classes (2).
        seed: random seed (also used as the Trainer/data seed).
        target_epochs: epoch numbers to evaluate in addition to the best checkpoint.
    """
    target_epochs = target_epochs or []

    print(f"\n[Info] Initializing model: {model_type} (Base: {model_name_or_path})")
    model, tokenizer = get_model_and_tokenizer(
        model_type=model_type,
        model_name_or_path=model_name_or_path,
        num_labels=num_labels,
        dropout_rate=dropout_rate,
    )

    for df in (df_train, df_eval, df_test, df_pubmed):
        df["label"] = df["label"].astype(int)

    print("[Info] Tokenizing datasets...")

    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"], truncation=True, padding="max_length", max_length=256
        )

    datasets_map = {
        "train": df_train,
        "eval": df_eval,
        "test_semmed": df_test,
        "test_pubmed": df_pubmed,
    }
    tokenized_datasets = {}
    for name, df in datasets_map.items():
        ds = Dataset.from_pandas(df)
        ds = ds.map(tokenize_fn, batched=True)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        tokenized_datasets[name] = ds

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=None if target_epochs else 2,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        data_seed=seed,
        log_level="info",
        report_to="none",
        optim="schedule_free_radam",
        lr_scheduler_type="constant",  # warmup_ratio has no effect with a constant schedule
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return metric.compute(predictions=np.argmax(logits, axis=-1), references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"[Info] Starting training in {output_dir}...")
    trainer.train()

    best_model_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_dir)
    pd.DataFrame(trainer.state.log_history).to_csv(
        os.path.join(output_dir, "training_log.csv"), index=False
    )

    steps_per_epoch = len(tokenized_datasets["train"]) // batch_size
    checkpoint_map = {}
    for cp_path in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        try:
            checkpoint_map[int(cp_path.split("-")[-1])] = cp_path
        except ValueError:
            continue

    eval_targets = [
        ("semmed", tokenized_datasets["test_semmed"]),
        ("pubmed", tokenized_datasets["test_pubmed"]),
    ]

    print("\n[Info] evaluating target checkpoints...")
    for ep in target_epochs:
        target_step = ep * steps_per_epoch
        closest_step, min_diff = None, float("inf")
        for step in checkpoint_map:
            if (
                abs(step - target_step) < (steps_per_epoch * 0.1) + 5
                and abs(step - target_step) < min_diff
            ):
                min_diff, closest_step = abs(step - target_step), step
        if closest_step is None:
            print(f"  - Epoch {ep} (Step ~{target_step}) not reached.")
            continue
        if load_model_weights(model, checkpoint_map[closest_step]):
            for name, ds in eval_targets:
                evaluate_and_save_metrics(
                    trainer, ds, output_dir, file_prefix=name, suffix_name=f"ep{ep}"
                )
        else:
            print(f"  - Failed to load weights from {checkpoint_map[closest_step]}")

    if os.path.exists(best_model_dir) and load_model_weights(model, best_model_dir):
        for name, ds in eval_targets:
            evaluate_and_save_metrics(
                trainer, ds, output_dir, file_prefix=name, suffix_name="best"
            )
    elif not os.path.exists(best_model_dir):
        print("  - Best model directory not found.")

    # All evaluations are done -- checkpoints can be removed to save disk space.
    cleanup_checkpoints(output_dir, [])

    print("[Info] Cleaning up huge model weights to save disk space...")
    weight_files = [
        "model.safetensors",
        "pytorch_model.bin",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ]
    if not save_best_model_weights:
        for f in weight_files:
            p = os.path.join(best_model_dir, f)
            if os.path.exists(p):
                os.remove(p)
                print(f"  - Removed best model weight: {p}")
    else:
        print(f"  - Kept best model weights in {best_model_dir}")

    print(f"[Info] Training finished. Results saved to {output_dir}")
