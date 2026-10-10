# Defining and Evaluating Cell–Cell Relation Extraction from Biomedical Literature under Realistic Annotation Constraints

This is the official repository for our paper:

> **Defining and Evaluating Cell–Cell Relation Extraction from Biomedical Literature under Realistic Annotation Constraints**  
> Mei Yoshikawa, Tadahaya Mizuno†, Yohei Ohto, Hiromi Fujimoto, Hiroyuki Kusuhara  
> *bioRxiv*, 2025.  
> [[bioRxiv]](https://doi.org/10.64898/2025.12.01.691726)

---
## Abstract
Extracting cell–cell relations from biomedical literature is essential for understanding intercellular communication in immunity, inflammation, and tissue biology. However, cell–cell relation extraction has not been established as a standalone biomedical relation extraction task, and no benchmark corpus or systematic evaluation framework currently exists. Fully manual corpus construction is costly and difficult to scale, limiting literature-based analyses of cell–cell communication.

In this work, we define a sentence-level cell–cell relation extraction task and construct complementary manually annotated corpora under realistic annotation constraints. Rule-based literature mining is used solely as an **annotation accelerator** to identify candidate sentences, while all relation labels are assigned manually. In addition, an independently annotated PubMed corpus without rule-based filtering is constructed to evaluate robustness on naturally occurring sentence distributions. Using these resources, we systematically evaluate representative model configurations involving entity indication strategies, classification architectures, and continued pre-training.

Our results show that cell–cell relation extraction remains challenging under realistic conditions. While increasing training data size and combining entity-aware architectures with continued pre-training yield modest robustness improvements, performance on unfiltered PubMed sentences remains in the 70% accuracy range. Comparisons with general-purpose large language models further suggest that task complexity, rather than model class, is the primary limiting factor.

This repository provides the datasets, model implementations, and evaluation code used in our study, serving as a practical foundation for future work on literature-scale cell–cell relation extraction.

## Models
Pre-trained model weights are publicly available via Hugging Face.

We release multiple trained models corresponding to different
architectures.
Each model is provided as an independent Hugging Face repository and
organized as a collection.

**Hugging Face Model Collection**: https://huggingface.co/collections/mizuno-group/cell-cell-bert

Please refer to the model cards on Hugging Face for:
- model configuration
- intended usage
- citation information

This GitHub repository provides the data-reconstruction, training, and
evaluation code used in the study, so that results can be reproduced starting
from the **published dataset** (offsets only, no raw text -- see below) rather
than our private raw corpora.

## Dataset

The annotated (sentence, entity pair, label) data is published as
[`DevWithKaiju/cell-cell-communication`](https://huggingface.co/datasets/DevWithKaiju/cell-cell-communication)
on the Hugging Face Hub, as character offsets (no raw sentence text, for
licensing reasons). See that dataset's README for the exact schema; this
repository's `scripts/01_reconstruct_dataset.py` turns those offsets back into
sentence-level training/eval CSVs.

## Installation

```bash
git clone https://github.com/mizuno-group/cell-cell-bert.git
cd cell-cell-bert
uv sync
```

## Reproducing results from the published dataset

The offset-only dataset does not include raw sentence text (licensing), so
reconstructing it requires two external sources:
- the **PubMed annual baseline** (public, fetched directly by this code), for
  `pubmed_relations` and `cpt_relations`
- a local copy of **SemMedDB** (requires a separate license from NLM), for
  `semmed_relations`

```bash
# 1. PubMed: download the baseline and build a local abstract index
#    (--limit N for a quick smoke test; omit it for the full ~40GB download)
python scripts/00_download_pubmed_baseline.py --limit 5
python scripts/00_build_pubmed_index.py

# 2. SemMed: build a local sentence index from your own SemMedDB download
python scripts/00_build_semmed_index.py \
    --predication /path/to/PREDICATION.csv.gz \
    --sentence /path/to/SENTENCE.csv.gz \
    --predication_aux /path/to/PREDICATION_AUX.csv.gz

# 3. Reconstruct the sentence-level CSVs (spot-check first with --sample_check)
python scripts/01_reconstruct_dataset.py --sample_check 5
python scripts/01_reconstruct_dataset.py

# 4. Continued pretraining (optional -- the pretrained checkpoints are also
#    published directly, see "Models" above)
python scripts/02_run_cpt.py --strategy replacement

# 5. Fine-tuning
python scripts/03_run_finetune.py --model_type R-HYB-CPT \
    --cpt_model_path mizuno-group/ccbert-R-pretrained --seed 22

# 6. Zero/few-shot LLM evaluation
python scripts/04_run_decoder_eval.py --model gpt-4.1 --prompt_style fewshot

# 7. LoRA SFT of a decoder model
python scripts/05_run_lora_sft.py --presentation replacement --prompt_style fewshot --seed 22
```

See each module's docstring (`src/cc_bert/`, `src/cc_gemma/`) for details,
including the one reproducibility gap: the original CPT run for the
replacement-token models (`ccbert-R-*`) mixed in sentences from a broader,
unpublished corpus for half of its examples; that half cannot be reproduced
bit-for-bit from the published dataset alone (see `src/cc_bert/cpt.py`).

## Directory Structure
```
.
├── scripts/                  # CLI entry points (run these)
│   ├── 00_download_pubmed_baseline.py
│   ├── 00_build_pubmed_index.py
│   ├── 00_build_semmed_index.py
│   ├── 01_reconstruct_dataset.py
│   ├── 02_run_cpt.py
│   ├── 03_run_finetune.py
│   ├── 04_run_decoder_eval.py
│   └── 05_run_lora_sft.py
├── src/
│   ├── cc_bert/               # BERT-based models: reconstruction, CPT, fine-tuning
│   └── cc_gemma/               # decoder-based (LLM) prompts, zero/few-shot eval, LoRA SFT
├── notebooks/
│   └── make_figures.ipynb     # paper figure generation
├── data/                      # reconstructed data (written by 01_reconstruct_dataset.py)
├── results/                   # training/eval outputs
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

## Requirements
All dependencies are listed in the pyproject.toml file.  

## How to Cite
If you find this work useful for your research, please consider citing our paper:  

```
@article{Yoshikawa2025CCBERT,
  title   = {Designing Cell–Cell Relation Extraction Models: A Systematic Evaluation of Entity Representation and Pre-training Strategies},
  author  = {Yoshikawa Mei and Mizuno Tadahaya and Ohto Yohei and Fujimoto Hiromi and Kusuhara Hiroyuki},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.64898/2025.12.01.691726},
  url     = {[https://doi.org/10.64898/2025.12.01.691726](https://doi.org/10.64898/2025.12.01.691726)}
}
```
    
## License
This project is licensed under the MIT License.  
See the LICENSE file for details.  

## Authors
- [Mei Yoshikawa](https://github.com/DevWithKaiju)  
    - main contributor  
- [Tadahaya Mizuno](https://github.com/tadahayamiz)  
    - correspondence  

## Contact
- Mei Yoshikawa - 	lion.giraffe.may.2525[at]gmail.com
- Tadahaya Mizuno - tadahaya[at]gmail.com (lead contact)