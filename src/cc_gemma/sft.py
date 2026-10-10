"""
sft.py

LoRA SFT of a decoder model (default: Gemma) for cell-cell relation extraction
(binary classification posed as generation), evaluated by greedy generation on
the test sets.

Compares two axes:
    - presentation : boundary (<E0>name</E0>) vs replacement ([CELL0])
    - prompt_style  : fewshot vs qa4re

One run is one (presentation, prompt_style, seed) combination. After training,
it runs greedy generation on the semmed/pubmed test sets and writes results in
the same format as decoder_eval.py (columns: sentence, output, label), so the
same downstream analysis code can consume either.

Training uses TRL's SFTTrainer + peft (LoRA). Data is passed in prompt-completion
form (plain strings; the chat template is applied in data.py). completion_only_loss
masks out the prompt so only the "0"/"1" answer (+ eos) contributes to the loss;
truncation_mode="keep_end" ensures the answer is never truncated away if the
prompt exceeds max_length.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from .data import build_prompt_dataset, load_dataframe
from .prompts import build_prompt

DEFAULT_MODEL = "google/gemma-4-12B-it"
EVAL_DATASETS = ("semmed", "pubmed")
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LoRA SFT of a decoder model for cell-cell relation extraction"
    )

    parser.add_argument(
        "--presentation", type=str, required=True, choices=["boundary", "replacement"]
    )
    parser.add_argument(
        "--prompt_style", type=str, required=True, choices=["fewshot", "qa4re"]
    )
    parser.add_argument("--seed", type=int, default=22)

    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="HF model id or local path."
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_root", type=str, default="results/llm_lora")
    parser.add_argument(
        "--size", type=int, default=1350, help="Number of training rows to use (head)."
    )
    parser.add_argument(
        "--eval_frac",
        type=float,
        default=0.1,
        help="Fraction of train held out for eval loss.",
    )

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--no_grad_checkpointing", action="store_true")

    parser.add_argument(
        "--load_in_4bit", action="store_true", help="QLoRA (bitsandbytes 4bit)."
    )
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument(
        "--debug", action="store_true", help="Use only 32 train / 10 test rows."
    )
    return parser


def model_short_name(model: str) -> str:
    return model.rstrip("/").split("/")[-1].lower()


def run_tag(args) -> str:
    return f"{model_short_name(args.model)}_{args.presentation}_{args.prompt_style}"


def parse_prediction(text: str) -> int:
    """Converts generated text to 0/1. Returns -1 if unparsable (callers
    should then score it as incorrect, not silently skip it)."""
    text = str(text).lower().strip()
    if "output: 1" in text or "output: `1`" in text or "option: 1" in text:
        return 1
    if "output: 0" in text or "output: `0`" in text or "option: 0" in text:
        return 0
    numbers = re.findall(r"[01]", text)
    if numbers:
        return int(numbers[0])
    return -1


def _load_model_and_tokenizer(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "dtype": torch.bfloat16,
        "attn_implementation": "eager",  # Gemma uses attention soft-capping, which needs eager attention
        "device_map": "auto",
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    return model, tokenizer


@torch.no_grad()
def _generate_predictions(model, tokenizer, prompts, eval_batch_size, max_length):
    model.eval()
    device = next(model.parameters()).device
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs = []
    try:
        for start in range(0, len(prompts), eval_batch_size):
            batch = prompts[start : start + eval_batch_size]
            chat_texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch
            ]
            enc = tokenizer(
                chat_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            ).to(device)
            gen = model.generate(
                **enc,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = gen[:, enc["input_ids"].shape[1] :]
            outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = prev_side
    return outputs


def _evaluate_split(model, tokenizer, args, dataset, tag_dir):
    df = load_dataframe(args.data_root, dataset, args.seed, args.presentation, "test")
    if args.debug:
        df = df.head(10)

    prompts = [
        build_prompt(args.prompt_style, args.presentation, s) for s in df["sentence"]
    ]
    raw_outputs = _generate_predictions(
        model, tokenizer, prompts, args.eval_batch_size, args.max_length
    )

    results = pd.DataFrame(
        {
            "sentence": df["sentence"].values,
            "output": raw_outputs,
            "label": df["label"].astype(int).values,
        }
    )
    out_dir = os.path.join(tag_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(
        os.path.join(out_dir, f"results_{args.seed}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    y_true = results["label"].to_numpy(dtype=int)
    y_pred = results["output"].apply(parse_prediction).to_numpy(dtype=int, copy=True)
    anomalies = int((y_pred == -1).sum())
    mask = y_pred == -1
    y_pred[mask] = 1 - y_true[mask]  # unparsable output is scored as incorrect

    return {
        "dataset": dataset,
        "n": len(results),
        "anomalies": anomalies,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred))
        if len(set(y_true)) > 1
        else 0.0,
    }


def run_sft_experiment(args) -> None:
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    tag = run_tag(args)
    tag_dir = os.path.join(args.output_root, tag)
    adapter_dir = os.path.join(tag_dir, "adapter")
    train_dir = os.path.join(tag_dir, "train")
    os.makedirs(tag_dir, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(" LoRA SFT for cell-cell relation extraction")
    print(f"   model        : {args.model}")
    print(f"   presentation : {args.presentation}")
    print(f"   prompt_style : {args.prompt_style}")
    print(f"   seed         : {args.seed}")
    print(f"   output       : {tag_dir}")
    print(f"{'=' * 64}\n")

    model, tokenizer = _load_model_and_tokenizer(args)
    use_gc = not args.no_grad_checkpointing

    df_train_full = load_dataframe(
        args.data_root, "semmed", args.seed, args.presentation, "train"
    )
    df_train_full = df_train_full.head(32 if args.debug else args.size)
    df_train = df_train_full.sample(frac=1.0 - args.eval_frac, random_state=args.seed)
    df_eval = df_train_full.drop(df_train.index)
    print(f"[data] train={len(df_train)}  eval={len(df_eval)}")

    train_ds = build_prompt_dataset(
        df_train, args.prompt_style, args.presentation, tokenizer
    )
    eval_ds = build_prompt_dataset(
        df_eval, args.prompt_style, args.presentation, tokenizer
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    sft_config = SFTConfig(
        output_dir=train_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_ratio,  # a float < 1 is treated as a warmup ratio
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="epoch",
        prediction_loss_only=True,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        gradient_checkpointing=use_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        truncation_mode="keep_end",
        completion_only_loss=True,
        packing=False,
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()

    trainer.train()
    trainer.save_model(adapter_dir)
    pd.DataFrame(trainer.state.log_history).to_csv(
        os.path.join(tag_dir, f"training_log_{args.seed}.csv"), index=False
    )

    model = trainer.model
    if use_gc and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    metrics = {
        "tag": tag,
        "model": args.model,
        "presentation": args.presentation,
        "prompt_style": args.prompt_style,
        "seed": args.seed,
        "splits": {},
    }
    for dataset in EVAL_DATASETS:
        split_metrics = _evaluate_split(model, tokenizer, args, dataset, tag_dir)
        metrics["splits"][dataset] = split_metrics
        print(
            f"[eval] {dataset}: acc={split_metrics['accuracy']:.4f} f1={split_metrics['f1']:.4f} "
            f"mcc={split_metrics['mcc']:.4f} (anomalies={split_metrics['anomalies']}/{split_metrics['n']})"
        )

    with open(os.path.join(tag_dir, f"metrics_{args.seed}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[done] metrics -> {os.path.join(tag_dir, f'metrics_{args.seed}.json')}")
