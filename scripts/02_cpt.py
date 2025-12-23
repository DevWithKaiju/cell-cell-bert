"""
train_cpt.py

Description:
    Main script for running Continual Pre-training (CPT) using Masked Language Modeling (MLM).
    It utilizes the PubMedBERT model and fine-tunes it on domain-specific biomedical text.
    
    This script handles:
    1. Dynamic data loading from mixed sources (DB/CSV) via `src.CCBERT.data_cpt`.
    2. Tokenizer adaptation with custom cell-related tokens.
    3. MLM training with schedule-free RAdam optimizer and Early Stopping.
    4. Comprehensive logging of training metrics and loss curves.

Usage:
    python 02_cpt.py --data_source semmed_db --data_path ./data.csv --db_path ./semmed.db

@Author: Mei Yoshikawa
"""

import sys
import os
import argparse
import pandas as pd
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# Add the project root to sys.path to import local modules
# Adjust the path relative to the script location if necessary
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.CCBERT.data_cpt import load_cpt_dataset

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Continual Pre-training (MLM) for Biomedical NLP")
    
    # Model and Output Settings
    parser.add_argument("--model_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
                        help="HuggingFace model identifier")
    parser.add_argument("--output_root", type=str, default="./results/cpt", help="Root directory for outputs")
    parser.add_argument("--exp_name", type=str, default="cpt_experiment", help="Experiment name for directory creation")
    
    # Data Paths
    parser.add_argument("--data_source", type=str, required=True, choices=["replacement", "boundary"],
                        help="Data loading strategy identifier")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the main CSV dataset")
    parser.add_argument("--db_path", type=str, default=None, help="Path to SQLite DB (required for 'replacement')")
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Per-device train batch size")
    parser.add_argument("--max_epochs", type=int, default=2, help="Maximum number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimization")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Linear warmup ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    output_dir = os.path.join(args.output_root, args.exp_name)
    logging_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Starting CPT Experiment: {args.exp_name} ===")

    # 1. Tokenizer Setup
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    
    # Add special tokens for cell entities and tags
    special_tokens = []
    if args.data_source == "replacement":
        # Strategy: Replacement (Entities are replaced with [CELL0], [CELL1])
        special_tokens = ['[CELL0]', '[CELL1]']
    elif args.data_source == "boundary":
        # Strategy: Boundary (Entities are wrapped with <E0>...</E0>)
        special_tokens = ['<E0>', '</E0>', '<E1>', '</E1>']
    
    if special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        print(f"Added special tokens for '{args.data_source}': {special_tokens}")

    # 2. Data Loading & Preprocessing
    dataset = load_cpt_dataset(
        data_source=args.data_source,
        data_path=args.data_path,
        db_path=args.db_path,
        seed=args.seed
    )
    
    def tokenize_function(examples):
        return tokenizer(
            examples["Cleaned_Sentence"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[c for c in dataset.column_names if c not in ["input_ids", "attention_mask", "token_type_ids"]]
    )

    # Split into Train (95%) and Eval (5%)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.05, seed=args.seed)
    print(f"Dataset Size - Train: {len(split_dataset['train'])}, Eval: {len(split_dataset['test'])}")
    
    # 3. Model Initialization
    model = BertForMaskedLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # 4. Trainer Configuration
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,      # Stop if no improvement for 5 eval steps
        early_stopping_threshold=0.001  # Minimum change to qualify as an improvement
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        overwrite_output_dir=True,
        
        # Training Parameters
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=args.num_workers,
        fp16=True,
        
        # Evaluation & Saving Strategy
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_steps=200,
        
        # Best Model Management
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        
        # Misc
        seed=args.seed,
        report_to="none",
        optim="schedule_free_radam"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
        callbacks=[early_stopping_callback]
    )
    
    # 5. Execution
    print("\n=== Starting Training Loop ===")
    train_result = trainer.train()
    
    # ---- Save Artifacts ----
    print(f"Saving model and tokenizer to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # ---- Save Logs & Metrics ----
    # 1. Final Training Metrics (Runtime, Epochs, etc.)
    metrics = train_result.metrics
    metrics["final_step"] = trainer.state.global_step
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Save metrics to a simple CSV for easy parsing
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "final_training_metrics.csv"), index=False)
    
    # 2. Full Training History (Loss curves)
    log_history_df = pd.DataFrame(trainer.state.log_history)
    log_history_df.to_csv(os.path.join(output_dir, "full_log_history.csv"), index=False)

    print(f"=== CPT Completed Successfully. Output: {output_dir} ===")

if __name__ == "__main__":
    main()