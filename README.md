# Designing Cell–Cell Relation Extraction Models: A Systematic Evaluation of Entity Representation and Pre-training Strategies
This is the official repository for our paper:

> **Designing Cell–Cell Relation Extraction Models: A Systematic Evaluation of Entity Representation and Pre-training Strategies**<br>
> Mei Yoshikawa, Tadahaya Miuzuno†, Yohei Ohto, Hiromi Fujimoto, Hiroyuki Kusuhara<br>
> *bioRxiv*, 2025.<br>
> [[bioRxiv]](https://doi.org/10.64898/2025.12.01.691726)

## Note
This repository is under construction and will be officially released by [Mizuno group](https://github.com/mizuno-group).  
Please contact tadahaya[at]gmail.com before publishing your paper using the contents of this repository.

## Abstract
Extracting cell–cell communication (CCC) from biomedical literature is critical for understanding intercellular interactions. While Large Language Models (LLMs) have advanced NLP, their generative nature raises reliability concerns for scientific fact extraction, highlighting the need for transparent, domain-adapted models.

This work presents the **first systematic evaluation of fundamental design choices** for biomedical relation extraction (BioRE), focusing on CCC. Using a newly constructed dataset derived from SemMedDB and an independent PubMed test set, we report the following key findings:

- **Systematic Analysis:** We evaluated core design decisions: entity indication strategies, model architectures, and the necessity of continued pre-training (CPT).
- **Design Insights:** CPT substantially improved entity-aware architectures but had minimal effect on CLS-only models. Furthermore, "replacement" strategies generalized better to natural text than "boundary marking."
- **Superior Performance:** Our best model achieved an out-of-distribution (OOD) accuracy of **0.757±0.009**, outperforming GPT-4o (0.715±0.012) while remaining lightweight and reproducible.

This repository provides the datasets, models, and design principles established in our study.

## Installation
*(Code release coming soon)*

Install the latest version directly from the GitHub repository:

```bash
pip install git+https://github.com/mizuno-group/CCBERT.git
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

## Requirements
All dependencies are listed in the pyproject.toml file.  

## Installation for Reproducing the Results
Clone this repository and install the required packages in editable mode. We recommend using a virtual environment.  

```bash
# Clone the repository
git clone {repository_URL}
cd {repository_name}

# Install dependencies
pip install -e .

```
-->

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