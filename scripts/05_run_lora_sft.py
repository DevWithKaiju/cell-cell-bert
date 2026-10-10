"""
scripts/05_run_lora_sft.py

LoRA SFT of a decoder model (default: Gemma) for cell-cell relation extraction.
Run scripts/01_reconstruct_dataset.py first.

Example:
    python scripts/05_run_lora_sft.py --presentation replacement --prompt_style fewshot --seed 22
"""

import os
import sys

sys.path.append(
    os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(__file__), ".."))
)

from src.cc_gemma.sft import get_parser, run_sft_experiment

if __name__ == "__main__":
    run_sft_experiment(get_parser().parse_args())
