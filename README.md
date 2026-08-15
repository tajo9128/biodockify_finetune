# 🧬 BioDockify.AI: Fine-Tuned Multi-Target Alzheimer's Deep Learning Stacked Ensemble

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model Accuracy](https://img.shields.io/badge/Ensemble%20Accuracy-91.48%25-success.svg)](https://github.com/tajo9128/biodockify_finetune)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.958-brightgreen.svg)](https://github.com/tajo9128/biodockify_finetune)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Checkpoint-orange.svg)](https://huggingface.co/tajo9128/alzheimers-ensemble-91pct)
[![Platform](https://img.shields.io/badge/BioDockify-ai.biodockify.com-purple.svg)](https://www.biodockify.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Principal Investigator:** Tajuddin Shaik (`tajo9128@gmail.com`)  
**Co-Authors:** Saravanan Ravindiran, Anbuselvi S., M. Sudhakar  
**Affiliation:** Doctoral Research Program | BioDockify Platform ([www.biodockify.com](https://www.biodockify.com))  
**Associated Research Repository:** [publication2](https://github.com/tajo9128/publication2) | **Subdomain:** [ai.biodockify.com](https://ai.biodockify.com)

---

## 📌 1. Scientific Overview & Doctoral Context

This repository hosts the **official fine-tuned, multi-target deep learning stacked ensemble model** developed for prospective virtual screening and lead optimization against Alzheimer's Disease (AD). 

The model was engineered as the **computational fine-tuning filter** for the PhD dissertation:
> *"Evaluation and Investigation of Phytochemical and Pharmacological Comparative Study of Alzheimer's Activity in Evolvulus alsinoides and Cinnamomum: Integration of Traditional Wet Lab Methods with AI-Driven Computational Analysis and Lead Formulation Optimization"*

### 🎯 Primary Biological Targets Evaluated
1. **Human Acetylcholinesterase (AChE)** — PDB ID: [`4EY7`](https://www.rcsb.org/structure/4EY7) (1.90 Å resolution, Catalytic Triad: Ser203, Trp86, Tyr337)
2. **Human Beta-Secretase 1 (BACE1)** — PDB ID: [`1FKN`](https://www.rcsb.org/structure/1FKN) (1.90 Å resolution, Catalytic Dyad: Asp32, Asp228)
3. **Human Glycogen Synthase Kinase-3β (GSK-3β)** — PDB ID: [`1Q41`](https://www.rcsb.org/structure/1Q41) (2.10 Å resolution, ATP Hinge: Val135, Lys85)

---

## 🧠 2. Stacked Ensemble Architecture & Mathematical Formulation

The architecture integrates two foundation chemical transformers with a tree-based latent classifier, stacked into a calibrated Meta-Learner Logistic Regression head.

```
                                  INPUT CHEMICAL STRUCTURE
                             (Canonical SMILES / 2D Topology)
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               ▼                            ▼                            ▼
      ┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
      │  BASE LEARNER 1  │         │  BASE LEARNER 2  │         │  BASE LEARNER 3  │
      │  MolFormer-XL    │         │  ChemBERTa-77M   │         │  Random Forest   │
      │  (Fine-Tuned)    │         │  (RoBERTa 10M)   │         │  (Latent Tree)   │
      │  Weight: 0.3425  │         │  Weight: 0.3149  │         │  Weight: 0.3425  │
      └────────┬─────────┘         └────────┬─────────┘         └────────┬─────────┘
               │                            │                            │
               ▼                            ▼                            ▼
             ŷ_M1                         ŷ_M2                         ŷ_M3
               │                            │                            │
               └────────────────────────────┼────────────────────────────┘
                                            │
                                            ▼
                           ┌──────────────────────────────────┐
                           │   STACKED META-LEARNER           │
                           │   (Logistic Regression Head)     │
                           │   β0 = -3.2863                   │
                           │   β = [2.4895, 1.7947, 1.8232]   │
                           └────────────────┬─────────────────┘
                                            │
                                            ▼
                           ┌──────────────────────────────────┐
                           │   OUTPUT CALIBRATED PROBABILITY  │
                           │   P(Active) ∈ [0.0, 1.0]         │
                           │   Accuracy: 91.48% | AUC: 0.958  │
                           └──────────────────────────────────┘
```

### 📐 Exact Mathematical Formulation

1. **Learned Soft Voting Weights (`ensemble_weights.npy`)**:
   $$\mathbf{w} = \begin{bmatrix} w_{\text{MolFormer}} \\ w_{\text{ChemBERTa}} \\ w_{\text{RandomForest}} \end{bmatrix} = \begin{bmatrix} 0.34254144 \\ 0.31491713 \\ 0.34254144 \end{bmatrix}$$

2. **Stacked Meta-Learner Logistic Formulation (`meta_learner_91pct.pkl`)**:
   $$\text{Logit}\big(P(\text{Active})\big) = \beta_0 + \beta_1 \cdot \hat{y}_{\text{MolFormer}} + \beta_2 \cdot \hat{y}_{\text{ChemBERTa}} + \beta_3 \cdot \hat{y}_{\text{RF}}$$
   $$\beta_0 = -3.28631565$$
   $$\vec{\beta} = \begin{bmatrix} 2.48950582, & 1.79471169, & 1.82323685 \end{bmatrix}$$
   $$P(\text{Active} \mid \vec{\hat{y}}) = \frac{1}{1 + \exp\Big(-\big(-3.2863 + 2.4895 \hat{y}_1 + 1.7947 \hat{y}_2 + 1.8232 \hat{y}_3\big)\Big)}$$

---

## 📊 3. 5-Fold Stratified Cross-Validation Benchmarks

Evaluated on the benchmark dataset ($N = 10,134$ curated bioactive & decoy records):

| Model Component | 5-Fold CV Accuracy | Sensitivity (Recall) | Specificity | Precision | ROC-AUC | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MolFormer-XL (Fine-Tuned)** | 90.12% | 89.40% | 90.80% | 91.20% | 0.938 | 0.9028 |
| **ChemBERTa-77M (RoBERTa 10M)** | 88.75% | 87.90% | 89.60% | 90.10% | 0.922 | 0.8898 |
| **Random Forest (Latent Embeddings)** | 89.30% | 88.60% | 90.00% | 90.50% | 0.929 | 0.8954 |
| **🏆 BioDockify Stacked Meta-Learner** | **91.48%** | **91.20%** | **91.75%** | **92.30%** | **0.958** | **0.9175** |

---

## 📁 4. Repository Contents & File Manifest

| File Name | File Size | Description & Scientific Utility |
| :--- | :---: | :--- |
| 📦 **`ensemble_91pct_complete.pkl`** | `13.4 MB` | Master serialized dictionary containing `{'weights', 'meta_learner', 'rf'}`. |
| ⚖️ **`meta_learner_91pct.pkl`** | `739 B` | Standalone `LogisticRegression` meta-learner with exact learned $\beta$ weights. |
| 🌲 **`molformer_rf_classifier.pkl`** | `13.4 MB` | Random Forest classifier trained on 768-dimensional MolFormer latent vectors. |
| 🌳 **`final_rf.pkl`** | `13.4 MB` | Calibrated Random Forest model for multi-target classification. |
| 🔢 **`ensemble_weights.npy`** | `152 B` | NumPy array storing the normalized base learner voting weights. |
| 📊 **`level3_training_data.csv`** | `510 KB` | 10,134 curated chemical structures with binary bioactivity labels (`smiles`, `active`). |
| 🔬 **`specialized_95pct_data.csv`** | `113 KB` | 1,539 multi-target active records annotated with specific target designations (`AChE`, `BACE1`, `GSK3b`). |
| 📄 **`README.txt`** | `484 B` | Original legacy model header card. |

---

## 💻 5. Quick-Start & Inference Code

### Prerequisites
```bash
pip install rdkit scikit-learn numpy pandas joblib
```

### Running Model Inference in Python
```python
import pickle
import numpy as np

# 1. Load the complete ensemble package
with open("ensemble_91pct_complete.pkl", "rb") as f:
    ensemble = pickle.load(f)

weights = ensemble["weights"]
meta_learner = ensemble["meta_learner"]
rf_model = ensemble["rf"]

print(f"[OK] Loaded BioDockify Stacked Ensemble (Learned Weights: {weights})")

# 2. Example: Evaluate Luteolin / Epicatechin Sub-Model Probabilities
# Suppose submodel forward passes output:
y_molformer = 0.92
y_chemberta = 0.89
y_rf = 0.88

# 3. Stacked Meta-Learner Prediction
z_input = np.array([[y_molformer, y_chemberta, y_rf]])
p_meta = meta_learner.predict_proba(z_input)[0, 1]

print(f"Sub-Model Predictions: MolFormer={y_molformer}, ChemBERTa={y_chemberta}, RF={y_rf}")
print(f"🏆 Final Stacked Meta-Learner Bioactivity Probability: {p_meta:.4f} ({(p_meta*100):.2f}%)")
```

---

## 🌐 6. BioDockify Ecosystem & Cloud Subdomain

This model powers the live web screening engine at:
- **Research Portal Subdomain:** [https://ai.biodockify.com](https://ai.biodockify.com)
- **Main Platform:** [https://www.biodockify.com](https://www.biodockify.com)
- **Full Virtual Screening Dossier:** [https://github.com/tajo9128/publication2](https://github.com/tajo9128/publication2)

---

## 📜 7. Citation & Academic BibTeX

If you utilize this model or dataset in your academic research, please cite:

```bibtex
@article{Shaik2026BioDockifyEnsemble,
  title={Receptor-Directed Deep Learning Stacked Ensemble and Network Pharmacology for Multi-Target Drug Discovery in Alzheimer's Disease},
  author={Shaik, Tajuddin and Ravindiran, Saravanan and S., Anbuselvi and Sudhakar, M.},
  journal={Journal of Biomolecular Structure and Dynamics},
  year={2026},
  publisher={Taylor \& Francis},
  url={https://github.com/tajo9128/biodockify_finetune}
}
```

---
*Maintained by Tajuddin Shaik (`tajo9128@gmail.com`). Licensed under the MIT Open-Access Research License.*
