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
├── configs/                    # Hyperparameter & experiment configs (YAML)
│   ├── default.yaml
│   ├── image.yaml
│   ├── text.yaml
│   └── multimodal.yaml
├── data/raw/                   # Raw datasets (git-ignored, download separately)
│   ├── image/
│   ├── text/
│   └── multimodal/
├── eda/                        # Exploratory Data Analysis
│   ├── image/
│   │   ├── notebooks/          # EDA Jupyter notebooks for image dataset
│   │   └── images/             # EDA output plots & visualizations
│   ├── text/
│   │   ├── notebooks/
│   │   └── images/
│   └── multimodal/
│       ├── notebooks/
│       └── images/
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
│   │   ├── models/             # Zero-shot & few-shot models (CLIP)
│   │   ├── data/
│   │   ├── training/
│   │   └── evaluation/
│   └── utils/                  # Shared utilities
├── results/                    # Experiment result images & figures
│   ├── image/                  # Plots, confusion matrices for image models
│   ├── text/                   # Plots, confusion matrices for text models
│   └── multimodal/             # Plots for multimodal models
├── outputs/checkpoints/        # Saved model weights (git-ignored)
│   ├── image/
│   ├── text/
│   └── multimodal/
├── extensions/                 # Bonus work (40% of grade)
│   ├── app_demo/               # FastAPI web inference demo
│   ├── interpretability/       # Grad-CAM, attention visualization
│   ├── error_analysis/         # Confusion analysis, hard examples
│   ├── augmentation/           # RandAugment, MixUp, back-translation
│   └── ensemble/               # Model combination experiments
├── reports/slides/             # Presentation slides
├── docs/                       # GitHub Pages landing page
│   ├── index.html
│   └── assignment1.html
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

| Modality | Dataset | Classes | Models & Strategies |
|----------|---------|---------|-----------------|
| Image | Intel Image Classification | 6 | ResNet18 vs. ViT-B/16 |
| Text | Emotion Dataset | 6 | BiLSTM vs. DistilBERT |
| Multimodal | Custom Vision-Language | ≥ 5 | Zero-shot & Few-shot Inference |

> **Note:** Dataset selection satisfies the constraints in Section 3 of the assignment specification. The compiled PDF Report (`CO3133_CC01_PE_Report1_pdf/main.pdf`) is not tracked by Git (`.gitignore`) to avoid bloat, please generate it locally using `latexmk -pdf main.tex`.

---

## 🧪 Evaluation Metrics

- **Accuracy** (primary)
- **F1-score** (when classes are imbalanced)
- Precision, Recall, Confusion Matrix (supplementary)

---

## 📦 Deliverables

- [x] GitHub Pages landing page
- [x] Report 1 — EDA, Dataset & DataLoader setup
- [x] Final Report — Full training, evaluation, comparison & extensions
- [x] Demo video & Interactive Web Interface (`app_demo`)
- [x] Presentation video (YouTube)

---

## 📅 Deadlines

| Milestone | Deadline |
|-----------|----------|
| Report 1 (50%) | 23:59, 26 March 2026 |
| Final Report (100%) | 23:59, 06 April 2026 |

> ⚠️ Late submission: –20% per week after the deadline.

---

## 👥 Team

| Member |
|--------|-----------|
| **Tang Cong Thanh** | 
| **Nguyen Ho Quang Khai** |

---

## 📄 License

This project is for academic purposes only — HCMUT, CO3133.

## 💻 Web App Inference Demo
To boot the FastAPI backend server powering the HTML Frontend Dashboard, run the following:
```bash
python -m uvicorn extensions.app_demo.main:app --reload --port 8000
```
Then navigate to `127.0.0.1:8000` or open `docs/demo/app.js` locally.