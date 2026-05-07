
# ADU

This repository contains the anonymous implementation of **ADU**, an attention-structured machine unlearning method for large language models. The code is released for anonymous peer review and is intended to reproduce the main unlearning, retention, and evaluation procedures described in the accompanying paper.

> **Anonymous review note.**  
> This repository has been anonymized for double-blind review. Author names, affiliations, personal paths, and non-anonymous project links are intentionally omitted. The repository will be de-anonymized after the review process if the paper is accepted.

## Overview

ADU is designed to suppress target knowledge in large language models while preserving general utility. The implementation contains three main components:

1. **Forget stage**: applies an attention-structured unlearning objective on target forget examples.
2. **Retain stage**: applies retention constraints on general-purpose examples to recover or preserve model utility after unlearning.
3. **Evaluation stage**: evaluates forgetting and retention using standard lm-eval tasks.

The current implementation focuses on WMDP-style machine unlearning experiments and MMLU-style retention evaluation.

## Repository Structure

```text
ADU/
├── README.md
├── unlearn/
│   ├── forget_unlearn.py      # Main ADU forget-stage training script
│   ├── forget_unlearn.sh      # Example launch script for the forget stage
│   ├── forget_utils.py        # Forget-stage data, prompt, attention, and loss utilities
│   └── utils.py               # Model loading and parameter-selection utilities
├── retain/
│   ├── retain_learn.py        # Main retain-stage training script
│   ├── retain_learn.sh        # Example launch script for the retain stage
│   ├── retain_utils.py        # Retain-stage prompt, attention, and loss utilities
│   └── utils.py               # Model loading and parameter-selection utilities
└── test/
    ├── test-wmdp.sh           # lm-eval script for WMDP-Bio and WMDP-Cyber
    └── test-mmlu-n-loglikelihood.sh
                              # lm-eval script for MMLU retention evaluation
````

## Installation

We recommend using a clean Python environment.

```bash
conda create -n adu python=3.10 -y
conda activate adu
```

Install the main dependencies:

```bash
pip install torch transformers datasets accelerate numpy tqdm sentencepiece protobuf
pip install lm-eval
```

Depending on the model family, additional packages such as `tokenizers`, `safetensors`, or model-specific dependencies may be required.

For Hugging Face models or datasets that require authentication, set your token locally:

```bash
export HF_TOKEN="<your_huggingface_token>"
```

