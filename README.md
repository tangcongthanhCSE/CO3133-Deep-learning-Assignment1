# CO3133 — Deep Learning and Its Applications | Assignment 1

> **Classification on Image, Text, and Multimodal Data**  
> HCMUT – VNUHCM | Semester 2, 2025–2026 | Instructor: Le Thanh Sach

---

## 📋 Overview

This repository contains the implementation for **Assignment 1** of the Deep Learning course (CO3133).  
The assignment focuses on classification tasks across **three data modalities**:

| Modality | Task | Models Compared |
|----------|------|-----------------|
| 🖼️ Image | Image Classification | CNN vs. ViT |
| 📝 Text | Text Classification | RNN (LSTM) vs. Transformer |
| 🔗 Multimodal | Multimodal Classification | Zero-shot vs. Few-shot |

---

## 🏗️ Project Structure

```
BTL1/
├── configs/                    # Hyperparameter & experiment configs (YAML/JSON)
├── data/
│   ├── raw/                    # Original unprocessed datasets
│   │   ├── image/
│   │   ├── text/
│   │   └── multimodal/
│   └── processed/              # Cleaned & preprocessed data
│       ├── image/
│       ├── text/
│       └── multimodal/
├── docs/                       # GitHub Pages landing page
│   └── assets/
├── extensions/                 # Bonus work (40% of grade)
│   ├── interpretability/       # Grad-CAM, attention visualization, saliency
│   ├── error_analysis/         # Confusion analysis, hard examples
│   ├── augmentation/           # RandAugment, MixUp, CutMix, back-translation
│   ├── ensemble/               # CNN+ViT, RNN+Transformer ensembles
│   └── app_demo/               # Gradio / Streamlit demo app
├── notebooks/                  # Jupyter notebooks for EDA & experiments
│   ├── image/
│   ├── text/
│   └── multimodal/
├── outputs/
│   ├── checkpoints/            # Saved model weights
│   │   ├── image/
│   │   ├── text/
│   │   └── multimodal/
│   ├── logs/                   # TensorBoard / training logs
│   ├── figures/                # Generated plots & visualizations
│   └── results/                # Metrics, tables (CSV/JSON)
├── reports/                    # Final reports & presentation materials
│   ├── figures/
│   └── slides/
├── scripts/                    # Shell/Python runner scripts
├── src/                        # Main source code
│   ├── image/
│   │   ├── models/             # CNN & ViT model definitions
│   │   ├── data/               # Dataset & DataLoader
│   │   ├── training/           # Training loops & schedulers
│   │   └── evaluation/         # Metrics & evaluation logic
│   ├── text/
│   │   ├── models/             # RNN (LSTM) & Transformer definitions
│   │   ├── data/
│   │   ├── training/
│   │   └── evaluation/
│   ├── multimodal/
│   │   ├── models/             # Zero-shot & few-shot models (e.g. CLIP)
│   │   ├── data/
│   │   ├── training/
│   │   └── evaluation/
│   └── utils/                  # Shared utilities (logging, visualization, etc.)
├── tests/                      # Unit & integration tests
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA (recommended for GPU training)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd BTL1

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Datasets

| Modality | Dataset | Classes | Training Samples |
|----------|---------|---------|-----------------|
| Image | *TBD* | ≥ 5 | ≥ 5,000 |
| Text | *TBD* | ≥ 5 | ≥ 5,000 |
| Multimodal | *TBD* | ≥ 5 | Genuine image–text pairs |

> **Note:** Dataset selection must satisfy the constraints in Section 3 of the assignment specification.

---

## 🧪 Evaluation Metrics

- **Accuracy** (primary)
- **F1-score** (when classes are imbalanced)
- Precision, Recall, Confusion Matrix (supplementary)

---

## 📦 Deliverables

- [x] GitHub Pages landing page
- [ ] Report 1 — EDA, Dataset & DataLoader setup
- [ ] Final Report — Full training, evaluation, comparison & extensions
- [ ] Demo video
- [ ] Presentation video (YouTube)

---

## 📅 Deadlines

| Milestone | Deadline |
|-----------|----------|
| Report 1 (50%) | 23:59, 26 March 2026 |
| Final Report (100%) | 23:59, 06 April 2026 |

> ⚠️ Late submission: –20% per week after the deadline.

---

## 👥 Team

| Member | Student ID |
|--------|-----------|
| *Name 1* | *ID* |
| *Name 2* | *ID* |
| *Name 3* | *ID* |
| *Name 4* | *ID* |

---

## 📄 License

This project is for academic purposes only — HCMUT, CO3133.
