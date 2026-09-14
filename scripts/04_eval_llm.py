"""
scripts/04_eval_llm.py

Description:
    Evaluates Large Language Models on the cell-cell relation classification task.
    All backend dispatch (OpenAI / Gemini / vLLM / local Hugging Face pipeline) and
    prompt logic lives in `src.cc_bert.eval_llm`; this script only parses CLI
    arguments and invokes it.

    Supported backends (dispatched from --model):
        - "provider/model-name" (contains "/"): local model via vLLM (default,
          batched) or Hugging Face pipeline (--use_hf)
        - "gemini-*": Gemini API (requires GEMINI_API_KEY)
        - anything else: OpenAI API, or any OpenAI-compatible endpoint via
          OPENAI_API_BASE (e.g. a self-hosted vLLM OpenAI server)

    Prompt styles (--prompt_style):
        - "direct": ask the model to output 0/1 directly
        - "qa4re": multiple-choice style prompt (Yin et al., QA4RE)

Usage:
    python scripts/04_eval_llm.py --model gpt-4 --prompt_style direct --seeds 42
    python scripts/04_eval_llm.py --model gemini-1.5-flash --prompt_style qa4re
    python scripts/04_eval_llm.py --model meta-llama/Llama-3-8B-Instruct --use_hf

@Author: Mei Yoshikawa
"""

import sys
import os
import argparse
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.cc_bert.eval_llm import run_evaluation, PROMPT_TEMPLATES

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM evaluation for relation extraction.")
    parser.add_argument("--model", type=str, default="gpt-4",
                         help="Model name (e.g., gpt-4, gemini-1.5-flash, meta-llama/Llama-3-8b-it).")
    parser.add_argument("--prompt_style", type=str, default="direct", choices=list(PROMPT_TEMPLATES.keys()),
                         help="Prompt format: 'direct' (output 0/1) or 'qa4re' (multiple-choice).")
    parser.add_argument("--datasets", type=str, nargs="+", default=["semmed", "pubmed"],
                         choices=["semmed", "pubmed"], help="Datasets to process.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[22, 24, 42, 57, 75],
                         help="List of random seeds to process.")
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory for input data.")
    parser.add_argument("--output_root", type=str, default="./results/llm", help="Directory to save output results.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (only process the first 10 rows).")
    parser.add_argument("--use_hf", action="store_true",
                         help="Use a local Hugging Face pipeline instead of vLLM for local models.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
