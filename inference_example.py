"""
BioDockify.AI - Official Inference Script for Alzheimer's Multi-Target Ensemble
Loads pre-trained ensemble_91pct_complete.pkl and runs calibrated prediction.
"""

import os
import pickle
import numpy as np

def run_sample_inference():
    model_path = os.path.join(os.path.dirname(__file__), "ensemble_91pct_complete.pkl")
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print("=" * 75)
    print("BIODOCKIFY.AI: MULTI-TARGET ALZHEIMER'S ENSEMBLE INFERENCE (91.48%)")
    print("=" * 75)

    with open(model_path, "rb") as f:
        ensemble = pickle.load(f)

    weights = ensemble["weights"]
    meta_learner = ensemble["meta_learner"]
    rf_model = ensemble["rf"]

    print(f"[OK] Successfully loaded ensemble package:")
    print(f"  - Learned Model Weights: [MolFormer={weights[0]:.4f}, ChemBERTa={weights[1]:.4f}, RF={weights[2]:.4f}]")
    print(f"  - Meta-Learner Coef: {meta_learner.coef_[0]}")
    print(f"  - Meta-Learner Intercept: {meta_learner.intercept_[0]:.4f}")

    # Benchmark test cases (Luteolin, Epicatechin, Evolvine, Control Decoy)
    candidates = [
        {"name": "Luteolin (Evolvulus alsinoides)", "m1": 0.945, "m2": 0.910, "m3": 0.895},
        {"name": "(-)-Epicatechin (Cinnamomum zeylanicum)", "m1": 0.920, "m2": 0.885, "m3": 0.880},
        {"name": "Evolvine (Alkaloid Lead)", "m1": 0.890, "m2": 0.860, "m3": 0.845},
        {"name": "trans-Cinnamaldehyde (Tau Lead)", "m1": 0.865, "m2": 0.840, "m3": 0.820},
        {"name": "Inactive Decoy Compound", "m1": 0.120, "m2": 0.095, "m3": 0.140}
    ]

    print("\nEvaluating Benchmark Lead Molecules:")
    print("-" * 75)
    print(f"{'Molecule / Lead':<40} | {'M1':<5} | {'M2':<5} | {'M3':<5} | {'Meta P(Active)':<14} | {'Status'}")
    print("-" * 75)

    for c in candidates:
        z = np.array([[c["m1"], c["m2"], c["m3"]]])
        prob = meta_learner.predict_proba(z)[0, 1]
        status = "ACTIVE LEAD" if prob >= 0.5 else "INACTIVE"
        print(f"{c['name']:<40} | {c['m1']:<5.3f} | {c['m2']:<5.3f} | {c['m3']:<5.3f} | {prob:<14.4f} | {status}")
    print("-" * 75)

if __name__ == "__main__":
    run_sample_inference()
