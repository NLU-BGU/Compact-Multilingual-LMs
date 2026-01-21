# Learning from Child-Directed Speech in Multilingual Scenarios

> A French-English Case Study

This repository contains code for pretraining and evaluating language models on child-directed speech (CDS) in multilingual settings.

---

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Pretraining](#1-pretraining)
  - [Fine-tuning](#2-fine-tuning)
  - [Evaluation](#3-evaluation)
- [Datasets](#datasets)
- [Citation](#citation)

---

## Overview

This research investigates whether language models trained on child-directed speech can develop linguistic competence efficiently in bilingual (English-French) scenarios. We implement and compare three model architectures:

- **BabyBERTa**: Compact RoBERTa-style model (~10M parameters)
- **LTG-BERT**: Winner of BabyLM Challenge with architectural innovations (~80M parameters)
- **T5**: Encoder-decoder architecture with span corruption (~60M parameters)

### Research Questions

1. Can models learn from developmentally-appropriate input (child-directed speech)?
2. How do different architectures compare when trained on limited data?
3. Does bilingual pretraining improve cross-lingual transfer?
4. What linguistic phenomena do models learn from CDS?

---

## Models

### BabyBERTa
- **Architecture**: RoBERTa-based masked language model
- **Parameters**: ~10M (256 hidden, 8 layers, 8 heads)
- **Vocabulary**: 8K BPE tokens
- **Training**: Masked LM with custom 90% mask, 10% random strategy

### LTG-BERT
- **Architecture**: Enhanced BERT with NormFormer and disentangled attention
- **Parameters**: ~80M (768 hidden, 12 layers, 12 heads)
- **Vocabulary**: 30K tokens
- **Innovations**: GeGLU activation, relative position encoding, span masking

### T5
- **Architecture**: Encoder-decoder transformer
- **Parameters**: ~60M (T5-small base)
- **Vocabulary**: 32K SentencePiece tokens
- **Training**: Span corruption objective (text-to-text)

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Anaconda or virtualenv

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/cds-multilingual-learning.git
cd cds-multilingual-learning

# 2. Create virtual environment
python -m venv venv

# 3. Activate the environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Create necessary directories
mkdir -p data/pre-training data/finetune data/evaluation
mkdir -p models/pretrained models/finetuned
mkdir -p results logs
```

### Alternative: Using Conda

```bash
# Create conda environment
conda create -n cds-research python=3.9
conda activate cds-research

# Install PyTorch (with CUDA if available)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
Research/
├── src/
│   ├── pretraining/           # Pretraining scripts
│   │   ├── pretrain_babyberta.py
│   │   ├── pretrain_ltgbert.py
│   │   ├── pretrain_t5.py
│   │   └── t5_mlm_collator.py
│   │
│   ├── finetuning/            # Fine-tuning scripts
│   │   ├── finetune_utils.py
│   │   ├── finetune_qa.py
│   │   └── finetune_nli.py
│   │
│   └── evaluation/            # Evaluation scripts
│       ├── evaluate_qa.py
│       ├── evaluate_nli.py
│       └── evaluate_grammar.py
│
├── scripts/
│   ├── pretraining/           # SLURM job scripts
│   ├── finetuning/
│   └── evaluation/
│
├── data/                      # Data directory
├── models/                    # Models directory
├── results/                   # Results directory
└── logs/                      # Logs directory
```

---

## Usage

### 1. Pretraining

Train language models on child-directed speech data.

#### BabyBERTa

**Using SLURM:**
```bash
bash scripts/pretraining/run_babyberta.sh \
    data/pre-training/CHILDES/2.5M/EN.txt \
    models/pretrained/babyberta-en \
    "42 51 71"
```

**Direct Python:**
```bash
python src/pretraining/pretrain_babyberta.py \
    --data_path data/pre-training/CHILDES/2.5M/EN.txt \
    --output_dir models/pretrained/babyberta-en \
    --seeds 42 51 71 \
    --batch_size 16 \
    --max_steps 260000
```

#### T5

**Using SLURM:**
```bash
bash scripts/pretraining/run_t5.sh \
    data/pre-training/CHILDES/2.5M/FR.txt \
    models/pretrained/t5-fr \
    "42"
```

**Direct Python:**
```bash
python src/pretraining/pretrain_t5.py \
    --data_path data/pre-training/CHILDES/2.5M/FR.txt \
    --output_dir models/pretrained/t5-fr \
    --seeds 42 \
    --batch_size 8 \
    --num_epochs 3
```

#### LTG-BERT

**Using SLURM:**
```bash
bash scripts/pretraining/run_ltgbert.sh \
    data/pre-training/CHILDES/2.5M/EN.txt \
    models/pretrained/ltgbert-en \
    "42"
```

**Direct Python:**
```bash
python src/pretraining/pretrain_ltgbert.py \
    --data_path data/pre-training/CHILDES/2.5M/EN.txt \
    --output_dir models/pretrained/ltgbert-en \
    --seeds 42 \
    --batch_size 8 \
    --max_steps 100000
```

---

### 2. Fine-tuning

Fine-tune pretrained models on downstream tasks.

#### Question Answering (QA)

**Extractive QA (BabyBERTa, LTG-BERT):**

Using SLURM:
```bash
bash scripts/finetuning/finetune_single_task.sh \
    models/pretrained/babyberta-en/checkpoint-260000 \
    qa \
    squad \
    models/finetuned/babyberta-squad
```

Direct Python:
```bash
python src/finetuning/finetune_qa.py \
    --model_checkpoint models/pretrained/babyberta-en/checkpoint-260000 \
    --dataset_name squad \
    --output_dir models/finetuned/babyberta-squad \
    --model_type extractive \
    --batch_size 8 \
    --num_epochs 3
```

**Generative QA (T5):**

Using SLURM:
```bash
bash scripts/finetuning/finetune_single_task.sh \
    models/pretrained/t5-fr/checkpoint-50000 \
    qa \
    fr-squad \
    models/finetuned/t5-squad-fr \
    generative
```

Direct Python:
```bash
python src/finetuning/finetune_qa.py \
    --model_checkpoint models/pretrained/t5-fr/checkpoint-50000 \
    --dataset_name fr-squad \
    --output_dir models/finetuned/t5-squad-fr \
    --model_type generative \
    --batch_size 4 \
    --num_epochs 3
```

#### Natural Language Inference (NLI)

**Classification (BabyBERTa, LTG-BERT):**

Using SLURM:
```bash
bash scripts/finetuning/finetune_single_task.sh \
    models/pretrained/babyberta-en/checkpoint-260000 \
    nli \
    xnli-en \
    models/finetuned/babyberta-xnli
```

Direct Python:
```bash
python src/finetuning/finetune_nli.py \
    --model_checkpoint models/pretrained/babyberta-en/checkpoint-260000 \
    --dataset_name xnli-en \
    --output_dir models/finetuned/babyberta-xnli \
    --model_type classification \
    --batch_size 16 \
    --num_epochs 3
```

**Generative (T5):**

Using SLURM:
```bash
bash scripts/finetuning/finetune_single_task.sh \
    models/pretrained/t5-fr/checkpoint-50000 \
    nli \
    xnli-fr \
    models/finetuned/t5-xnli-fr \
    generative
```

---

### 3. Evaluation

Evaluate trained models on test sets.

#### QA Evaluation

Using SLURM:
```bash
bash scripts/evaluation/evaluate_qa.sh \
    models/finetuned/babyberta-squad \
    data/finetune/SQuAD/EN/squad_dev.json \
    squad
```

Direct Python:
```bash
python src/evaluation/evaluate_qa.py \
    models/finetuned/babyberta-squad \
    --validation_data data/finetune/SQuAD/EN/squad_dev.json \
    --dataset_name squad \
    --batch_size 16
```

**Output files:**
- `squad_results.txt` - Human-readable results
- `squad_metrics.json` - JSON metrics
- `squad_predictions.json` - Model predictions

#### NLI Evaluation

Using SLURM:
```bash
bash scripts/evaluation/evaluate_nli.sh \
    models/finetuned/babyberta-xnli \
    data/finetune/XNLI/EN/xnli_dev.json \
    xnli-en
```

Direct Python:
```bash
python src/evaluation/evaluate_nli.py \
    models/finetuned/babyberta-xnli \
    data/finetune/XNLI/EN/xnli_dev.json \
    --dataset_name xnli-en \
    --batch_size 32
```

**Output files:**
- `xnli-en_results.txt` - Human-readable results
- `xnli-en_metrics.json` - JSON metrics
- `xnli-en_predictions.json` - Model predictions

#### Grammar Evaluation (BLiMP/CLAMS)

Evaluate grammatical knowledge without fine-tuning:

Using SLURM:
```bash
bash scripts/evaluation/evaluate_grammar.sh \
    models/pretrained/babyberta-en-seed42 \
    data/pre-training/CHILDES/2.5M/EN.txt \
    data/evaluation/blimp \
    "42 51 71"
```

Direct Python:
```bash
python src/evaluation/evaluate_grammar.py \
    --base_model_path models/pretrained/babyberta-en-seed42 \
    --training_data_path data/pre-training/CHILDES/2.5M/EN.txt \
    --data_dir data/evaluation/blimp \
    --seeds 42 51 71 \
    --output_dir results/grammar
```

**Output file:**
- `{model}_grammar_results_{timestamp}.txt` - Detailed results table with mean and std across seeds

---

## Datasets

### Pretraining Data

**CHILDES (Child Language Data Exchange System)**
- Format: Plain text, one sentence per line (`.txt`)
- English: 2.5M tokens
- French: 2.5M tokens
- Bilingual: 5M tokens (combined)

**Example:**
```
The cat is sleeping.
I want to play outside.
Can you help me?
```

### Fine-tuning Data

#### Question Answering

**Supported datasets:**
- `squad` - SQuAD v1.1 (English)
- `fr-squad` - French SQuAD
- `qamr-en` / `qamr-fr` - QAMR
- `qasrl-en` / `qasrl-fr` - QA-SRL

**Format (JSON):**
```json
{
  "data": [
    {
      "id": "example_001",
      "context": "The cat sat on the mat.",
      "question": "Where did the cat sit?",
      "answers": {
        "text": ["on the mat"],
        "answer_start": [16]
      }
    }
  ]
}
```

#### Natural Language Inference

**Supported datasets:**
- `xnli-en` - XNLI English
- `xnli-fr` - XNLI French
- `anli` - Adversarial NLI
- `mnli` - MultiNLI

**Format (JSON):**
```json
[
  {
    "premise": "A person is walking a dog.",
    "hypothesis": "An animal is being walked.",
    "label": 0
  }
]
```

**Labels:**
- 0: entailment
- 1: neutral
- 2: contradiction

### Evaluation Data

**Grammar Tests (BLiMP/CLAMS)**

**Format (TSV):**
```
True	The cat is sleeping.
False	The cat are sleeping.
True	Dogs like to run.
False	Dogs likes to run.
```

---

## Configuration

### Modifying Training Parameters

Edit the shell scripts or pass arguments directly:

```bash
python src/pretraining/pretrain_babyberta.py \
    --data_path data/my_data.txt \
    --output_dir models/my_model \
    --seeds 42 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --max_steps 500000 \
    --warmup_steps 50000 \
    --save_steps 100000
```

### SLURM Configuration

Modify SLURM parameters in shell scripts:

```bash
#SBATCH --partition=main
#SBATCH --time=48:00:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=64G
```

---

## Troubleshooting

### Common Issues

**1. Out of Memory**
- Reduce `--batch_size`
- Add `--gradient_accumulation_steps`

**2. Tokenizer Errors**
- The code automatically fixes tokenizer format issues
- Check `finetune_utils.py` for the `fix_tokenizer()` function

**3. CUDA Errors**
- Update CUDA drivers
- Check PyTorch CUDA compatibility

**4. Data Format Issues**
- Ensure correct format (see [Datasets](#datasets) section)
- Check file encoding (should be UTF-8)

---

## Citation

### Related Work

**BabyBERTa:**
```bibtex
@inproceedings{huebner-etal-2021-babyberta,
  title={BabyBERTa: Learning More Grammar With Small-Scale Child-Directed Language},
  author={Huebner, Philip A and Sulem, Elior and Fisher, Cynthia and Roth, Dan},
  booktitle={CoNLL},
  year={2021}
}
```

**LTG-BERT:**
```bibtex
@inproceedings{samuel-etal-2023-trained,
  title={Trained on 100 Million Words and Still in Shape: {BERT} Meets British National Corpus},
  author={Samuel, David and Kutuzov, Andrey and {\O}vrelid, Lilja and Velldal, Erik},
  booktitle={BabyLM Challenge},
  year={2023}
}
```

**T5:**
```bibtex
@article{raffel2020exploring,
  title={Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  author={Raffel, Colin and others},
  journal={JMLR},
  year={2020}
}
```

---

## 👥 Contact

- **Author**: Liel Binyamin
- **Advisor**: Dr. Elior Sulem
- **Institution**: Ben-Gurion University of the Negev

---

## 📄 License

This project is licensed under the MIT License.

---

**Last Updated**: October 2025
