"""
decoder_eval.py

Zero/few-shot LLM evaluation of cell-cell relation extraction: no training,
just prompting a model (OpenAI-compatible API, Gemini API, or a local model via
vLLM/transformers) and scoring its answer against the gold label.

Supports both prompt styles and both entity presentations from prompts.py.
Output format matches sft.py's generation-eval output (columns: sentence,
output, label), so the same downstream analysis code can consume either.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from tqdm import tqdm

from .data import load_dataframe
from .prompts import build_prompt

EVAL_DATASETS = ("semmed", "pubmed")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero/few-shot LLM evaluation for cell-cell relation extraction"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="OpenAI-compatible model name, 'gemini-...', or a HF model id/path "
        "(a '/' in the name is treated as a local/HF model, run via vLLM by default).",
    )
    parser.add_argument(
        "--prompt_style", type=str, default="fewshot", choices=["fewshot", "qa4re"]
    )
    parser.add_argument(
        "--presentation",
        type=str,
        default="replacement",
        choices=["replacement", "boundary"],
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=list(EVAL_DATASETS),
        choices=list(EVAL_DATASETS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[22, 24, 42, 57, 75])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_root", type=str, default="results/llm")
    parser.add_argument(
        "--use_hf",
        action="store_true",
        help="Use the transformers pipeline instead of vLLM for local models.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Only process the first 10 rows of each split.",
    )
    return parser


def _get_llm_prediction(prompt_body, model_name, client=None, hf_pipeline=None):
    try:
        if hf_pipeline is not None:
            messages = [
                {
                    "role": "user",
                    "content": f"You are a biomedical expert.\n\n{prompt_body}",
                }
            ]
            outputs = hf_pipeline(
                messages, max_new_tokens=10, temperature=0.0, do_sample=False
            )
            return outputs[0]["generated_text"][-1]["content"].strip()

        if model_name.startswith("gemini"):
            import time

            time.sleep(4.1)  # stay under the free-tier rate limit (15 RPM)
            for attempt in range(3):
                try:
                    response = client.generate_content(
                        prompt_body, generation_config={"temperature": 0.0}
                    )
                    return response.text.strip()
                except Exception as e:
                    if attempt < 2 and ("429" in str(e) or "quota" in str(e).lower()):
                        print("Rate limit hit, sleeping for 10 seconds...")
                        time.sleep(10)
                    else:
                        raise

        kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a biomedical expert."},
                {"role": "user", "content": prompt_body},
            ],
        }
        if "gpt-5" not in model_name.lower() and "o1" not in model_name.lower():
            kwargs["temperature"] = 0.0
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Error] Prediction failed: {e}")
        return "error"


def _load_client_and_backend(args):
    """Returns (client, hf_pipeline, vllm_model, tokenizer) -- exactly one of
    (client, hf_pipeline, vllm_model) is set depending on --model/--use_hf."""
    if "/" in args.model:
        if args.use_hf:
            import torch
            from transformers import pipeline

            print(f"[Info] Loading transformers pipeline for {args.model}...")
            hf_pipeline = pipeline(
                "text-generation",
                model=args.model,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
            return None, hf_pipeline, None, None

        from transformers import AutoTokenizer
        from vllm import LLM

        print(f"[Info] Loading vLLM model {args.model}...")
        vllm_args = {"model": args.model, "gpu_memory_utilization": 0.90}
        if "qwen" in args.model.lower():
            vllm_args["max_model_len"] = 32768
        return None, None, LLM(**vllm_args), AutoTokenizer.from_pretrained(args.model)

    if args.model.startswith("gemini"):
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(args.model), None, None, None

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    if base_url:
        return OpenAI(api_key=api_key or "EMPTY", base_url=base_url), None, None, None
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is missing.")
    return OpenAI(api_key=api_key), None, None, None


def _vllm_batch_predict(vllm_model, tokenizer, prompts, model_name):
    from vllm import SamplingParams

    chat_texts = []
    for p in prompts:
        messages = [{"role": "user", "content": f"You are a biomedical expert.\n\n{p}"}]
        try:
            chat_texts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        except ValueError:
            if getattr(tokenizer, "chat_template", None) is not None:
                raise
            # Fall back to a minimal, widely-applicable chat template for models that ship without one.
            if "mistral" in model_name.lower() or "mixtral" in model_name.lower():
                tokenizer.chat_template = (
                    "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}"
                    "{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}"
                    "{{ message['content'] + eos_token}}{% endif %}{% endfor %}"
                )
            else:
                tokenizer.chat_template = (
                    "{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + "
                    "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{% endfor %}"
                    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
                )
            chat_texts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

    outputs = vllm_model.generate(
        chat_texts, SamplingParams(temperature=0.0, max_tokens=10)
    )
    return [out.outputs[0].text.strip() for out in outputs]


def run_decoder_eval(args) -> None:
    client, hf_pipeline, vllm_model, tokenizer = _load_client_and_backend(args)
    model_dir = args.model.replace("/", "_")

    for dataset in args.datasets:
        output_dir = os.path.join(args.output_root, model_dir, dataset)
        os.makedirs(output_dir, exist_ok=True)

        for seed in args.seeds:
            df = load_dataframe(
                args.data_root, dataset, seed, args.presentation, "test"
            )
            if args.debug:
                df = df.head(10)

            prompts = [
                build_prompt(args.prompt_style, args.presentation, s)
                for s in df["sentence"]
            ]

            if vllm_model is not None:
                outputs = _vllm_batch_predict(
                    vllm_model, tokenizer, prompts, args.model
                )
            else:
                outputs = [
                    _get_llm_prediction(
                        p, args.model, client=client, hf_pipeline=hf_pipeline
                    )
                    for p in tqdm(prompts, desc=f"{dataset} seed {seed}")
                ]

            results = pd.DataFrame(
                {"sentence": df["sentence"], "output": outputs, "label": df["label"]}
            )
            output_path = os.path.join(output_dir, f"results_{seed}.csv")
            results.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"Saved results to {output_path}")
