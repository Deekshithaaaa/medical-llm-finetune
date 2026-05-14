# 🏥 Medical LLM Fine-tuning — Mistral 7B on MedQA

> Fine-tuning Mistral 7B on USMLE medical exam questions using QLoRA, achieving 8% accuracy improvement over the base model.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Mistral](https://img.shields.io/badge/Model-Mistral_7B-orange)
![QLoRA](https://img.shields.io/badge/Method-QLoRA-purple)
![Accuracy](https://img.shields.io/badge/Accuracy-54%25-brightgreen)
![Improvement](https://img.shields.io/badge/Improvement-%2B8%25-green)

🤗 **Model on HuggingFace:** https://huggingface.co/deekshitha-urs/medical-mistral-7b-finetuned

---

## 📌 What is this?

An end-to-end LLM fine-tuning pipeline that takes the general-purpose Mistral 7B model and specializes it for medical question answering using USMLE (United States Medical Licensing Exam) questions. Uses QLoRA to make 7B parameter fine-tuning possible on a free GPU.

## 📊 Results

| Model | Accuracy | Test Questions |
|-------|----------|----------------|
| Base Mistral 7B | 46% | 50 USMLE questions |
| Fine-tuned (ours) | **54%** | 50 USMLE questions |
| **Improvement** | **+8%** | — |

Training loss: **1.33 → 0.91** over 2 epochs

## 🏗️ Architecture
MedQA Dataset (10,178 questions)
↓
Instruction Formatting ([INST] template)
↓
QLoRA Fine-tuning (4-bit quantization + LoRA adapters)
↓
Fine-tuned Mistral 7B
↓
Evaluation vs Base Model (50 USMLE questions)

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Base Model | Mistral 7B v0.1 |
| Fine-tuning Method | QLoRA (4-bit + LoRA) |
| LoRA Rank | r=16, alpha=32 |
| Dataset | MedQA USMLE 4-options |
| Training Examples | 500 |
| Training Epochs | 2 |
| GPU | Kaggle T4 x2 (free) |
| Framework | HuggingFace PEFT + TRL |
| Evaluation | Exact match accuracy |

## 📁 Project Structure
medical-llm-finetune/
├── src/
│   ├── prepare_data.py    # Format MedQA into instruction format
│   ├── train.py           # QLoRA fine-tuning script
│   ├── evaluate.py        # Base vs fine-tuned comparison
│   └── explore_data.py    # Dataset exploration
├── evaluation/            # Evaluation results
├── data/                  # Formatted datasets
├── requirements.txt
└── README.md

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
```bash
python src/prepare_data.py
```

### 3. Fine-tune (requires GPU — run on Kaggle)
```bash
python src/train.py
```

### 4. Evaluate
```bash
python src/evaluate.py
```

## 🔍 Why QLoRA?

Full fine-tuning of Mistral 7B requires ~80GB VRAM. QLoRA makes it possible on a free 16GB GPU by:
- **4-bit quantization** — compresses model weights from 16-bit to 4-bit
- **LoRA adapters** — only trains 0.09% of parameters (6.8M out of 7.2B)
- **Result** — same quality as full fine-tuning at a fraction of the cost

## 📚 Dataset

**MedQA USMLE** — real United States Medical Licensing Exam questions:
- 10,178 training questions
- 1,273 test questions
- 4 multiple choice options per question
- Covers clinical medicine, pharmacology, pathology

## 🔮 Future Improvements
- Train on full 10,178 examples for higher accuracy
- Add cross-encoder re-ranking
- Experiment with larger LoRA rank (r=32, r=64)
- Fine-tune on specific medical specialties

## 👩‍💻 Author
**Deekshitha Adishesha Raje Urs**
[GitHub](https://github.com/Deekshithaaaa) | [LinkedIn](#) | [HuggingFace](https://huggingface.co/deekshitha-urs)
