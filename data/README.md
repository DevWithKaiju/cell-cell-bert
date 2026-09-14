# Dataset Description

This directory contains the datasets used for training (fine-tuning) and evaluating the Cell-Cell Relation Extraction models. The data is organized by source (`pubmed`, `semmed`) and split across multiple random seeds for robust experimentation.

Raw exports (`pubmed/data.csv`, `semmed/data.csv`) are turned into the per-seed files below by `scripts/01_preprocess.py`. Each strategy is written as a separate file:

* `boundary`: entities wrapped in `<E0>...</E0>` / `<E1>...</E1>` tags (used by the R-BERT-style `*-HYB`/`*-ENT` architectures).
* `replacement`: entities replaced with `[CELL0]`/`[CELL1]` tokens (used by the `*-CLS` architectures and the LLM evaluation scripts).

## Directory Structure

The data is organized as follows:

```text
data/
├── pubmed/                          # PubMed-based dataset (Evaluation)
│   ├── data.csv                     # Raw export (input to scripts/01_preprocess.py)
│   ├── 22/                          # Random Seed 22
│   │   ├── test_boundary.csv
│   │   └── test_replacement.csv
│   ├── 24/
│   ├── 42/
│   ├── 57/
│   └── 75/
└── semmed/                          # SemMed-based dataset (Fine-tuning & Evaluation)
    ├── data.csv                     # Raw export (input to scripts/01_preprocess.py)
    ├── 22/
    │   ├── train_boundary.csv
    │   ├── test_boundary.csv
    │   ├── train_replacement.csv
    │   └── test_replacement.csv
    ├── ...
    └── 75/
```

## Dataset Usage

* `semmed/{seed}/train_{strategy}.csv`: Used for Fine-tuning the models (CPT or Supervised Fine-tuning).
* `semmed/{seed}/test_{strategy}.csv`: Used for evaluating the model on SemMed-derived samples.
* `pubmed/{seed}/test_{strategy}.csv`: Used for evaluating the model on PubMed-derived samples (Generalization capability).
