"""
train_multimodal_master.py

Advanced Multimodal Training Pipeline combining:
1. Clinical Demographics & TNM Staging
2. 3D Volumetric & Morphological CT Biomarkers
3. 3D Margin Texture & Attenuation Gradients

Evaluates 5-Fold Stratified Cross-Validation on:
- Cancer Staging (Stage I, II, III)
- Major Histological Subtypes (Adenocarcinoma vs. Squamous Cell)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from configs.config import Config


def main():
    master_csv = Config.PROCESSED_DIR / "clinical" / "multimodal_master_dataset.csv"
    if not master_csv.exists():
        import extract_radiomics_features
        extract_radiomics_features.main()

    df = pd.read_csv(master_csv)
    print("=" * 75)
    print(f"MULTIMODAL MASTER BENCHMARK ({len(df)} Patients | 23 Features)")
    print("=" * 75)

    feature_cols = [c for c in df.columns if c not in ["PatientID", "histology", "stage"]]
    X = df[feature_cols].fillna(0).values
    y_stage = df["stage"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # -------------------------------------------------------------
    # 1. CANCER STAGING (Multimodal)
    # -------------------------------------------------------------
    print("\n[1] Evaluating Cancer Staging (Stage I, II, III)...")
    clf_stage = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    scores_stage = cross_validate(
        clf_stage, X, y_stage, cv=skf,
        scoring=["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    )
    s_acc = scores_stage["test_accuracy"].mean() * 100
    s_f1 = scores_stage["test_f1_weighted"].mean()
    print(f"--> Multimodal Stage Accuracy : {s_acc:.2f}% (± {scores_stage['test_accuracy'].std()*100:.2f}%)")
    print(f"--> Multimodal Stage F1 Score : {s_f1:.4f}")

    # -------------------------------------------------------------
    # 2. MAJOR HISTOLOGY SUBTYPES (Adeno vs. Squamous)
    # -------------------------------------------------------------
    print("\n[2] Evaluating Histology Classification (Adenocarcinoma vs. Squamous)...")
    major_mask = df["histology"].isin([0, 2])
    df_major = df[major_mask].copy().reset_index(drop=True)
    X_major = df_major[feature_cols].fillna(0).values
    y_major = (df_major["histology"].values == 2).astype(int)

    skf_major = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Balanced ExtraTrees with probability tuning
    clf_hist = ExtraTreesClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=6,
        min_samples_split=4,
        random_state=42,
    )
    scores_hist = cross_validate(
        clf_hist, X_major, y_major, cv=skf_major,
        scoring=["accuracy", "f1_weighted", "roc_auc"]
    )
    h_acc = scores_hist["test_accuracy"].mean() * 100
    h_f1 = scores_hist["test_f1_weighted"].mean()
    h_auc = scores_hist["test_roc_auc"].mean()
    print(f"--> Histology Accuracy (ADC vs SCC): {h_acc:.2f}% (± {scores_hist['test_accuracy'].std()*100:.2f}%)")
    print(f"--> Histology F1 Score             : {h_f1:.4f}")
    print(f"--> Histology ROC-AUC              : {h_auc:.4f}")

    # -------------------------------------------------------------
    # 3. OVERALL COMBINED MULTITASK PERFORMANCE
    # -------------------------------------------------------------
    overall_acc = (s_acc + h_acc) / 2
    print("\n" + "=" * 75)
    print("OVERALL MULTITASK PERFORMANCE")
    print("=" * 75)
    print(f"Stage Accuracy             : {s_acc:.2f}%")
    print(f"Histology Accuracy (ADC/SCC): {h_acc:.2f}% (AUC: {h_auc:.4f})")
    print(f"Combined Mean Accuracy     : {overall_acc:.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    main()
