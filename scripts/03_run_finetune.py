"""
scripts/03_run_finetune.py

Fine-tunes one (model_type, seed) cell-cell relation extraction model. Run
scripts/01_reconstruct_dataset.py first.

model_type is one of B-HYB, R-HYB, B-ENT, R-ENT, R-CLS, B-CLS, optionally
suffixed with "-CPT" to start from a continued-pretraining checkpoint (either
the published mizuno-group/ccbert-{R,B}-pretrained, or a local checkpoint from
scripts/02_run_cpt.py).

Example:
    python scripts/03_run_finetune.py --model_type R-HYB-CPT \\
        --cpt_model_path mizuno-group/ccbert-R-pretrained --seed 22
    python scripts/03_run_finetune.py --model_type B-CLS --seed 22
"""

import os
import sys

sys.path.append(
    os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(__file__), ".."))
)

from src.cc_bert.experiment_runner import get_parser, run_experiment

if __name__ == "__main__":
    run_experiment(get_parser().parse_args())
