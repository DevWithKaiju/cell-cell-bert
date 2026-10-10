"""
scripts/04_run_decoder_eval.py

Zero/few-shot LLM evaluation (no training). Run scripts/01_reconstruct_dataset.py
first.

Example:
    python scripts/04_run_decoder_eval.py --model gpt-4.1 --prompt_style fewshot --presentation replacement
    python scripts/04_run_decoder_eval.py --model meta-llama/Llama-3.1-8B-Instruct --prompt_style qa4re
"""

import os
import sys

sys.path.append(
    os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(__file__), ".."))
)

from src.cc_gemma.decoder_eval import get_parser, run_decoder_eval

if __name__ == "__main__":
    run_decoder_eval(get_parser().parse_args())
