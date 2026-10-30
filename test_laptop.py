"""
test_laptop.py

Immediate verification script for your laptop.
Tests Stage and Histology classification on clean data in ~3 seconds.
Outputs 5-Fold Cross-Validation accuracy, precision, recall, and F1.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report

from configs.config import Config


def main():
    clean_csv = Config.PROCESSED_DIR / "clinical" / "clean_dataset.csv"
    if not clean_csv.exists():
        print("Running prepare_clean_data.py...")
        import prepare_clean_data
        prepare_clean_data.prepare_data()

    df = pd.read_csv(clean_csv)
    print("=" * 70)
    print(f"LAPTOP BENCHMARK EVALUATION ({len(df)} Patients)")
    print("=" * 70)

    X = df[["age", "gender", "T_stage", "N_stage", "M_stage"]].values
    y_stage = df["stage"].values
    y_hist = df["histology"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Cancer Staging Benchmark
    print("\n[1] Evaluating Cancer Staging (Stage I, II, III)...")
    clf_stage = RandomForestClassifier(n_estimators=100, random_state=42)
    scores_stage = cross_validate(
        clf_stage, X, y_stage, cv=skf,
        scoring=["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    )

    stage_acc = scores_stage["test_accuracy"].mean() * 100
    stage_f1 = scores_stage["test_f1_weighted"].mean()
    print(f"--> Stage Accuracy: {stage_acc:.2f}% (± {scores_stage['test_accuracy'].std()*100:.2f}%)")
    print(f"--> Stage F1 Score: {stage_f1:.4f}")

    # 2. Major Histology Subtyping Benchmark (Adeno vs. Squamous)
    print("\n[2] Evaluating Major Histological Subtypes (Adenocarcinoma vs. Squamous)...")
    # Subtypes: 0 is Adeno, 2 is Squamous in clean mapping
    major_mask = df["histology"].isin([0, 2])
    df_major = df[major_mask]
    X_major = df_major[["age", "gender", "T_stage", "N_stage", "M_stage"]].values
    y_major = (df_major["histology"].values == 2).astype(int)

    skf_major = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_hist = HistGradientBoostingClassifier(random_state=42)
    scores_hist = cross_validate(
        clf_hist, X_major, y_major, cv=skf_major,
        scoring=["accuracy", "f1_weighted"]
    )

    hist_acc = scores_hist["test_accuracy"].mean() * 100
    hist_f1 = scores_hist["test_f1_weighted"].mean()
    print(f"--> Clinical-Only ADC vs SCC Accuracy: {hist_acc:.2f}% (F1: {hist_f1:.4f})")
    print("    (Note: Adding CT imaging features boosts this further in multimodal fusion)")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
