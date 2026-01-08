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

This GitHub repository provides the datasets, training code,
and evaluation pipeline used in the study.

## Installation
*(Code release coming soon)*

Install the latest version directly from the GitHub repository:

```bash
pip install git+https://github.com/mizuno-group/ccbert.git
```

<!--
## Directory Structure
```
.
├── notebooks/            # example notebooks
│   └── usage_example.ipynb
├── src/
│   └── my_project/       # source codes
│       ├── init.py
│       ├── cli.py        # CLI entry point
│       └── core.py
├── tests/                # test codes
│   └── test_module.py
├── .gitignore
├── LICENSE               
├── pyproject.toml        
└── README.md             
```
-->

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