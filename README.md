# 🧬 BioDockify.AI: Multi-Target Alzheimer's Deep Learning Stacked Ensemble (94.20% Accuracy)

<p align="center">
  <a href="https://huggingface.co/BioDockify/alzheimers-ensemble-94pct"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20Hub%20(94.20%25)-yellow?style=for-the-badge&logo=huggingface" alt="Hugging Face Model"></a>
  <a href="https://huggingface.co/datasets/BioDockify/alzheimers-multi-target-10k-dataset"><img src="https://img.shields.io/badge/📊%20Hugging%20Face-Dataset%20Hub%20(10k)-orange?style=for-the-badge&logo=huggingface" alt="Hugging Face Dataset"></a>
  <a href="https://ai.biodockify.com"><img src="https://img.shields.io/badge/Live%20AI%20Portal-ai.biodockify.com-8b5cf6?style=for-the-badge&logo=fastapi" alt="Live AI Portal"></a>
  <a href="https://www.biodockify.com"><img src="https://img.shields.io/badge/BioDockify%20Platform-www.biodockify.com-0284c7?style=for-the-badge" alt="BioDockify Platform"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/5--Fold%20CV%20Accuracy-94.20%25-success?style=flat-square" alt="Accuracy">
  <img src="https://img.shields.io/badge/ROC--AUC-0.958-brightgreen?style=flat-square" alt="ROC-AUC">
  <img src="https://img.shields.io/badge/F1--Score-0.9747-blue?style=flat-square" alt="F1">
  <img src="https://img.shields.io/badge/OECD%20Compliant-Yes-green?style=flat-square" alt="OECD">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License">
</p>

---

## 📌 Repository Overview

This repository contains the **official training code, 5-fold cross-validation checkpoints, and serialized model binaries** for the **94.20% Multi-Target Alzheimer's Deep Learning Stacked Ensemble**.

Developed as part of doctoral research at the **Faculty of Pharmacy, Bharath Institute of Higher Education and Research (BIHER)**, the model fuses two foundation chemical language transformers (**MolFormer-XL** and **ChemBERTa-77M**) with 2048-bit Morgan Fingerprints (ECFP4), a calibrated Random Forest, and a regularized Level-1 Stacking Meta-Learner.

### 🎯 Biological Targets Evaluated
1. **Human Acetylcholinesterase (AChE)** — PDB ID: `4EY7` (1.90 Å, Catalytic Triad: Ser203, Trp86, Tyr337)
2. **Human Beta-Secretase 1 (BACE1)** — PDB ID: `1FKN` (1.90 Å, Catalytic Dyad: Asp32, Asp228)
3. **Human Glycogen Synthase Kinase-3β (GSK-3β)** — PDB ID: `1Q41` (2.10 Å, ATP Hinge: Val135, Lys85)

---

## 🌐 Official Open-Science Resources & Hubs

* 🤗 **Hugging Face Model Hub:** [https://huggingface.co/BioDockify/alzheimers-ensemble-94pct](https://huggingface.co/BioDockify/alzheimers-ensemble-94pct)
* 📊 **Hugging Face Dataset Hub:** [https://huggingface.co/datasets/BioDockify/alzheimers-multi-target-10k-dataset](https://huggingface.co/datasets/BioDockify/alzheimers-multi-target-10k-dataset)
* 🏛️ **BioDockify Organization:** [https://huggingface.co/BioDockify](https://huggingface.co/BioDockify)
* 🧬 **Live AI Screening Portal:** [https://ai.biodockify.com](https://ai.biodockify.com)
* 📚 **PhD Publication 2 Suite:** [https://github.com/tajo9128/publication2](https://github.com/tajo9128/publication2)

---

## 📊 5-Fold Stratified Cross-Validation Benchmarks ($N = 10,134$)

| Model Architecture | Accuracy (%) | Precision | Recall (Sens.) | Specificity | F1-Score | AUC-ROC | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Random Forest** | 86.79% | 0.842 | 0.871 | 0.865 | 0.856 | 0.891 | 0.724 |
| **ChemBERTa-77M (Fine-Tuned)** | 88.86% | 0.870 | 0.895 | 0.882 | 0.882 | 0.912 | 0.768 |
| **MolFormer-XL (Fine-Tuned)** | 91.48% | 0.898 | 0.923 | 0.906 | 0.910 | 0.929 | 0.819 |
| 🏆 **BioDockify Stacked Ensemble** | **94.20%** | **0.944** | **0.938** | **0.946** | **0.975** | **0.958** | **0.884** |

---

## 📐 Mathematical Formulation & Stacking Mechanism

$$\text{Logit}\big(P(\text{Active})\big) = \beta_0 + \beta_1 \cdot \hat{y}_{\text{MolFormer}} + \beta_2 \cdot \hat{y}_{\text{ChemBERTa}} + \beta_3 \cdot \hat{y}_{\text{RF}}$$

$$\beta_0 = -3.28631565, \quad \vec{\beta} = [2.48950582, \; 1.79471169, \; 1.82323685]$$

$$P(\text{Active} \mid \vec{\hat{y}}) = \frac{1}{1 + \exp\Big(-\big(-3.2863 + 2.4895 \hat{y}_1 + 1.7947 \hat{y}_2 + 1.8232 \hat{y}_3\big)\Big)}$$

---

## 💻 Quick-Start Python Inference

### Option A: Load Directly from Hugging Face Hub

```python
from huggingface_hub import hf_hub_download
import pickle
import numpy as np

# Download and load the calibrated meta-learner
model_path = hf_hub_download(
    repo_id="BioDockify/alzheimers-ensemble-94pct",
    filename="models/trained_stacked_meta_learner.pkl"
)

with open(model_path, "rb") as f:
    meta_learner = pickle.load(f)

# Input sub-model predictions [MolFormer, ChemBERTa, Random Forest]
# Example for Luteolin (Evolvulus alsinoides)
z_input = np.array([[0.945, 0.910, 0.895]])
p_active = meta_learner.predict_proba(z_input)[0, 1]

print(f"Predicted Multi-Target Active Probability: {p_active:.4f} ({p_active*100:.2f}%)")
```

### Option B: Run Local Script
```bash
# Clone the repository
git clone https://github.com/tajo9128/biodockify_finetune.git
cd biodockify_finetune

# Run inference on reference phytochemicals
python inference_example.py
```

---

## 📁 Repository Structure

```
biodockify_finetune/
├── 📄 README.md                                                 # This Documentation
├── 📄 Paper_AI_1_Receptor_Directed_Deep_Learning_Ensemble.docx  # Full 18-page Manuscript (94.20% Ensemble)
├── 📄 inference_example.py                                      # Standalone Python inference script
├── 📁 notebooks/                                                # Google Colab Training Notebooks
│   ├── Stacked_Ensemble.ipynb                                   # Dual-Stream Transformer & Stacking Suite
│   ├── amber.ipynb                                              # Force field & MD preparation
│   ├── Tier_4_.ipynb                                            # 100ns MD & MM-PBSA Trajectory Pipeline
│   └── stacked_Ensemble_Tier_1.ipynb                            # Tier 1 ADMET & Filter Notebook
├── 📁 checkpoints/                                              # Target-Specific Model Weights & Predictions
│   ├── checkpoint_AChE.csv                                      # Human AChE (4EY7) Log
│   ├── checkpoint_BACE1.csv                                     # Human BACE1 (1FKN) Log
│   ├── checkpoint_GSK3b.csv                                     # Human GSK-3β (1Q41) Log
│   ├── biodockify_stacked_results.csv                           # 5-Fold CV Cross-Validation Outputs
│   └── biodockify_meta_fold1.joblib                             # Meta-Learner Binary
└── 📁 data/                                                     # Curated Chemical Space Datasets
    ├── biodockify_global_plant_database.csv                     # 3,205 Curated Natural Product Chemical Space
    ├── biodockify_plant_screen_results.csv                      # Plant Screening Prediction Logs
    ├── level3_training_data.csv                                 # Level-3 Training Data
    └── specialized_95pct_data.csv                               # High-Potency Benchmark Subset
```

---

## 👥 Research Team & Affiliations

* **Tajuddin Shaik¹,\*** *(Corresponding Author — `tajo9128@gmail.com`)*
* **Saravanan Ravindiran²**
* **Anbuselvi S.³**
* **M. Sudhakar⁴**

¹*Department of Pharmacognosy and Phytochemistry, Faculty of Pharmacy, Bharath Institute of Higher Education and Research, Chennai 600073, India*  
²*Department of Pharmacology, Faculty of Pharmacy, Bharath Institute of Higher Education and Research, Chennai 600073, India*  
³*Department of Pharmaceutical Chemistry, Faculty of Pharmacy, Bharath Institute of Higher Education and Research, Chennai 600073, India*  
⁴*Department of Pharmacology, Malla Reddy College of Pharmacy, Osmania University, Hyderabad 500100, India*

---

## 📜 Citation

If you use this model, dataset, or code in your research, please cite:

```bibtex
@article{Shaik2026BioDockifyEnsemble,
  title={Receptor-Directed Deep Learning Ensemble with Interpretable Mechanisms for Multi-Target Alzheimer's Drug Discovery: Targeting AChE, BACE1, and GSK-3β Inhibitors},
  author={Shaik, Tajuddin and Ravindiran, Saravanan and S., Anbuselvi and Sudhakar, M.},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  publisher={American Chemical Society},
  url={https://huggingface.co/BioDockify/alzheimers-ensemble-94pct}
}
```
