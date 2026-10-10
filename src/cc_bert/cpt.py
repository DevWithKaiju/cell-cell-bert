"""
cpt.py

Continued pretraining (CPT) of PubMedBERT with cell-entity markup, matching
the two pretrained checkpoints published on the Hub
(mizuno-group/ccbert-R-pretrained, mizuno-group/ccbert-B-pretrained).

Training mixes two kinds of examples for robustness:
    - "marked" sentences, with entities shown as [CELL0]/[CELL1] (replacement
      strategy) or <E0>.../<E1>.. tags (boundary strategy)
    - "plain" sentences, with no entity markup at all

IMPORTANT -- reproducibility caveat: in the original training run, for the
*replacement* strategy the plain-text half was sourced from a broader,
unpublished cell-mention corpus (not from data_for_annotation.csv itself).
That corpus is not published, so it cannot be reproduced here. This module
instead uses the plain sentence reconstructed directly from the same PubMed
abstract (available precisely because offsets are published) -- a reasonable
substitute serving the same purpose (exposing the model to unmarked text), but
not a bit-identical reproduction of the original CPT training data. The
*boundary* strategy has no such gap: the original "plain" half was always just
the same row's sentence with its tags stripped, which is exactly what
`sentence_plain` already is here.
"""

import os

import pandas as pd
from datasets import Dataset
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

STRATEGY_TOKENS = {
    "replacement": ["[CELL0]", "[CELL1]"],
    "boundary": ["<E0>", "</E0>", "<E1>", "</E1>"],
}
MARKED_COLUMN = {"replacement": "sentence_replacement", "boundary": "sentence_boundary"}


def build_cpt_corpus(df_resolved, strategy, revert_frac=0.5, revert_seed=42):
    """Builds the mixed plain/marked training corpus from
    data/cpt/relations_resolved.csv (see reconstruct.reconstruct_cpt_relations).

    `revert_frac` of rows (selected by `.sample(random_state=revert_seed)`, the
    same mechanism the original training run used) get the plain sentence;
    the rest get the marked sentence. Returns a DataFrame with one column,
    `sentence`, in the original row order.
    """
    marked_col = MARKED_COLUMN[strategy]
    df = df_resolved.copy()
    plain_idx = df.sample(frac=revert_frac, random_state=revert_seed).index
    df["sentence"] = df[marked_col]
    df.loc[plain_idx, "sentence"] = df.loc[plain_idx, "sentence_plain"]
    return df[["sentence"]].reset_index(drop=True)


def run_cpt_training(
    df_corpus,
    strategy,
    base_model_path,
    output_dir,
    seed=42,
    learning_rate=2e-5,
    weight_decay=1e-4,
    warmup_ratio=0.1,
    batch_size=128,
    gradient_accumulation_steps=4,
    max_epochs=2,
    max_length=128,
    eval_frac=0.05,
):
    """Runs masked-language-model continued pretraining on `df_corpus`
    (a DataFrame with a single `sentence` column, from build_cpt_corpus()).

    Saves the best (lowest eval_loss) checkpoint, tokenizer, and training log
    to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = BertTokenizerFast.from_pretrained(base_model_path)
    tokenizer.add_tokens(STRATEGY_TOKENS[strategy])

    dataset = Dataset.from_pandas(df_corpus)

    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["sentence"])
    split = tokenized.train_test_split(test_size=eval_frac, seed=seed)
    train_dataset, eval_dataset = split["train"], split["test"]
    print(f"[cpt] train={len(train_dataset)}  eval={len(eval_dataset)}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    model = BertForMaskedLM.from_pretrained(base_model_path)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=max_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="linear",
        logging_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        dataloader_num_workers=8,
        fp16=True,
        report_to="none",
        optim="schedule_free_radam",
        logging_dir=os.path.join(output_dir, "logs"),
        log_level="info",
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5, early_stopping_threshold=0.001
            )
        ],
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["final_step"] = trainer.state.global_step
    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_dir, "final_training_metrics.csv"), index=False
    )
    pd.DataFrame(trainer.state.log_history).to_csv(
        os.path.join(output_dir, "full_log_history.csv"), index=False
    )

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[cpt] saved best checkpoint and tokenizer to {output_dir}")
