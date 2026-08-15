# 🧬 BioDockify.AI: Multi-Target Alzheimer's Deep Learning Stacked Ensemble Training Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model Accuracy](https://img.shields.io/badge/Ensemble%20Accuracy-91.48%25%20--%2094.20%25-success.svg)](https://github.com/tajo9128/biodockify_finetune)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.958-brightgreen.svg)](https://github.com/tajo9128/biodockify_finetune)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Checkpoint-orange.svg)](https://huggingface.co/tajo9128/alzheimers-ensemble-91pct)
[![Platform](https://img.shields.io/badge/BioDockify-ai.biodockify.com-purple.svg)](https://www.biodockify.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Principal Investigator:** Tajuddin Shaik (`tajo9128@gmail.com`)  
**Co-Authors:** Saravanan Ravindiran, Anbuselvi S., M. Sudhakar  
**Affiliation:** Doctoral Research Program | BioDockify Platform ([www.biodockify.com](https://www.biodockify.com))  
**Associated Research Repository:** [publication2](https://github.com/tajo9128/publication2) | **Live Subdomain:** [ai.biodockify.com](https://ai.biodockify.com)

---

## 📌 1. Scientific Overview & Doctoral Scope

This repository provides the **complete, end-to-end training suite, Colab notebooks, target checkpoints, datasets, and serialized model binaries** for the multi-target deep learning stacked ensemble developed for:
> *"Evaluation and Investigation of Phytochemical and Pharmacological Comparative Study of Alzheimer's Activity in Evolvulus alsinoides and Cinnamomum: Integration of Traditional Wet Lab Methods with AI-Driven Computational Analysis and Lead Formulation Optimization"*

### 🎯 Primary Biological Targets Evaluated
1. **Human Acetylcholinesterase (AChE)** — PDB ID: [`4EY7`](https://www.rcsb.org/structure/4EY7) (1.90 Å, Catalytic Triad: Ser203, Trp86, Tyr337)
2. **Human Beta-Secretase 1 (BACE1)** — PDB ID: [`1FKN`](https://www.rcsb.org/structure/1FKN) (1.90 Å, Catalytic Dyad: Asp32, Asp228)
3. **Human Glycogen Synthase Kinase-3β (GSK-3β)** — PDB ID: [`1Q41`](https://www.rcsb.org/structure/1Q41) (2.10 Å, ATP Hinge: Val135, Lys85)

---

## 🧠 2. Stacked Ensemble Architecture & Training Protocol

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

### 📐 Exact Mathematical Formulations

1. **Learned Soft Voting Weights (`ensemble_weights.npy`)**:
   $$\mathbf{w} = \begin{bmatrix} w_{\text{MolFormer}} \\ w_{\text{ChemBERTa}} \\ w_{\text{RandomForest}} \end{bmatrix} = \begin{bmatrix} 0.34254144 \\ 0.31491713 \\ 0.34254144 \end{bmatrix}$$

2. **Stacked Meta-Learner Logistic Head (`meta_learner_91pct.pkl`)**:
   $$\text{Logit}\big(P(\text{Active})\big) = \beta_0 + \beta_1 \cdot \hat{y}_{\text{MolFormer}} + \beta_2 \cdot \hat{y}_{\text{ChemBERTa}} + \beta_3 \cdot \hat{y}_{\text{RF}}$$
   $$\beta_0 = -3.28631565, \quad \vec{\beta} = [2.48950582, \; 1.79471169, \; 1.82323685]$$
   $$P(\text{Active} \mid \vec{\hat{y}}) = \frac{1}{1 + \exp\Big(-\big(-3.2863 + 2.4895 \hat{y}_1 + 1.7947 \hat{y}_2 + 1.8232 \hat{y}_3\big)\Big)}$$

---

## 📂 3. Complete Repository Structure & File Directory

```
biodockify_finetune/
├── README.md                           # Master Academic Documentation & Benchmarks
├── README.txt                          # Legacy Model Card
├── inference_example.py                # Standalone Python inference script
│
├── notebooks/                          # 📓 Interactive Google Colab Training Notebooks
│   ├── Stacked_Ensemble.ipynb          # Full dual-stream transformer fine-tuning loop
│   ├── stacked_Ensemble_Tier_1.ipynb   # 4-bit quantized MolFormer-XL training
│   ├── ensemble.ipynb                  # Data harvesting & model calibration
│   ├── Tier_4_.ipynb                   # Multi-target 5-fold cross-validation
│   ├── amber.ipynb                     # OpenMM & AmberTools 100ns molecular dynamics
│   └── Copy_of_Stacked_Ensemble.ipynb  # Backup ensemble training pipeline
│
├── checkpoints/                        # 🎯 Target-Specific Checkpoints & Predictions
│   ├── checkpoint_AChE.csv             # Acetylcholinesterase prediction checkpoint (448 KB)
│   ├── checkpoint_BACE1.csv            # Beta-Secretase 1 prediction checkpoint (1.68 MB)
│   ├── checkpoint_GSK3b.csv            # Glycogen Synthase Kinase-3β checkpoint (448 KB)
│   ├── biodockify_stacked_results.csv  # Multi-model stacked score log (278 KB)
│   └── biodockify_meta_fold1.joblib    # Fold-1 Meta-Learner serialized binary
│
├── data/                               # 📊 Curated Training & Natural Product Libraries
│   ├── biodockify_global_plant_database.csv  # 3,205 natural product plant SMILES
│   ├── biodockify_plant_screen_results.csv   # Confidence distribution on plant library
│   ├── level3_training_data.csv              # 10,134 curated training structures
│   └── specialized_95pct_data.csv            # 1,539 multi-target active records
│
└── models/                             # 🧠 Trained Model Binaries (.pkl & .npy)
    ├── ensemble_91pct_complete.pkl     # Master package {'weights', 'meta_learner', 'rf'}
    ├── meta_learner_91pct.pkl          # Standalone Logistic Regression meta-learner
    ├── molformer_rf_classifier.pkl     # Random Forest on MolFormer-XL embeddings
    ├── final_rf.pkl                    # Calibrated multi-target Random Forest
    └── ensemble_weights.npy            # Learned soft voting weights
```

---

## 📊 4. 5-Fold Stratified Cross-Validation Benchmarks

| Model Component | 5-Fold CV Accuracy | Sensitivity (Recall) | Specificity | Precision | ROC-AUC | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MolFormer-XL (Fine-Tuned)** | 90.12% | 89.40% | 90.80% | 91.20% | 0.938 | 0.9028 |
| **ChemBERTa-77M (RoBERTa 10M)** | 88.75% | 87.90% | 89.60% | 90.10% | 0.922 | 0.8898 |
| **Random Forest (Latent Embeddings)** | 89.30% | 88.60% | 90.00% | 90.50% | 0.929 | 0.8954 |
| **🏆 BioDockify Stacked Meta-Learner** | **91.48% – 94.20%** | **91.20%** | **91.75%** | **92.30%** | **0.958** | **0.9175** |

---

## 💻 5. Quick-Start & Inference Code

```python
import pickle
import numpy as np

# 1. Load the pre-trained ensemble
with open("ensemble_91pct_complete.pkl", "rb") as f:
    ensemble = pickle.load(f)

weights = ensemble["weights"]
meta_learner = ensemble["meta_learner"]

print(f"[OK] BioDockify Stacked Ensemble Loaded (Weights: {weights})")

# 2. Predict for Luteolin / Epicatechin
z_input = np.array([[0.945, 0.910, 0.895]])
p_active = meta_learner.predict_proba(z_input)[0, 1]

print(f"🏆 Meta-Learner Predicted Probability: {p_active:.4f} ({(p_active*100):.2f}%)")
```

---

## 🌐 6. BioDockify Ecosystem & Cloud Subdomain

- **Live Research Portal:** [https://ai.biodockify.com](https://ai.biodockify.com)
- **Main Platform:** [https://www.biodockify.com](https://www.biodockify.com)
- **Full PhD Research Repository:** [https://github.com/tajo9128/publication2](https://github.com/tajo9128/publication2)

---

## 📜 7. Citation & Academic BibTeX

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
