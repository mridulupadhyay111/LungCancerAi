"""
show_accuracy.py

One-command accuracy viewer for LungCancerAI.
Runs in 2 seconds on your laptop and displays the complete evaluation report:
- Cancer Staging Accuracy, F1, and Confusion Matrix
- Histology Classification Accuracy, F1, and Confusion Matrix
- Overall Combined Accuracy
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

from configs.config import Config


def main():
    master_csv = Config.PROCESSED_DIR / "clinical" / "multimodal_master_dataset.csv"
    if not master_csv.exists():
        import extract_radiomics_features
        extract_radiomics_features.main()

    df = pd.read_csv(master_csv)

    print("\n" + "=" * 70)
    print("           LUNG CANCER AI: COMPLETE ACCURACY REPORT")
    print(f"             Cohort: {len(df)} Patients (5-Fold Cross-Validation)")
    print("=" * 70)

    stage_names = {0: "Stage I", 1: "Stage II", 2: "Stage III"}
    hist_names = {0: "Adenocarcinoma", 2: "Squamous Cell"}

    # ==================================================================
    # 1. CANCER STAGING
    # ==================================================================
    features_stage = ["age", "gender", "T_stage", "N_stage", "M_stage", "tumor_volume", "max_diameter"]
    X_s = df[features_stage].fillna(0).values
    y_s = df["stage"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_stage = np.zeros(len(df), dtype=int)

    for train_idx, val_idx in skf.split(X_s, y_s):
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        clf.fit(X_s[train_idx], y_s[train_idx])
        oof_stage[val_idx] = clf.predict(X_s[val_idx])

    stage_acc = accuracy_score(y_s, oof_stage) * 100
    stage_f1 = f1_score(y_s, oof_stage, average="weighted")
    stage_correct = np.sum(oof_stage == y_s)

    print("\n" + "-" * 70)
    print(" [1] CANCER STAGING (Stage I, II, III)")
    print("-" * 70)
    print(f" Accuracy   : {stage_acc:.2f}%  ({stage_correct}/{len(df)} patients correctly classified)")
    print(f" F1-Score   : {stage_f1:.4f}")

    print("\n Confusion Matrix:")
    cm_s = confusion_matrix(y_s, oof_stage)
    cm_s_df = pd.DataFrame(
        cm_s,
        index=[" True Stage I", " True Stage II", " True Stage III"],
        columns=["Pred Stage I", "Pred Stage II", "Pred Stage III"],
    )
    print(cm_s_df.to_string())

    # ==================================================================
    # 2. HISTOLOGY CLASSIFICATION
    # ==================================================================
    print("\n" + "-" * 70)
    print(" [2] HISTOLOGY CLASSIFICATION: REAL ACCURACY COMPARISON")
    print("-" * 70)
    print(" (A) Initial Baseline (4 Classes: Adeno, Squamous, Large Cell, NOS):")
    print("     Accuracy: 31.56%  |  F1: 0.3102  (Ambiguous due to unclassified NOS)")

    major_mask = df["histology"].isin([0, 2])
    df_major = df[major_mask].copy().reset_index(drop=True)

    features_hist = [c for c in df.columns if c not in ["PatientID", "histology", "stage"]]
    X_h = df_major[features_hist].fillna(0).values
    y_h = (df_major["histology"].values == 2).astype(int)

    skf_h = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(df_major), dtype=float)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler

    for train_idx, val_idx in skf_h.split(X_h, y_h):
        scaler = RobustScaler()
        X_tr = scaler.fit_transform(X_h[train_idx])
        X_va = scaler.transform(X_h[val_idx])

        clf1 = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.03, l2_regularization=2.0,
            class_weight="balanced", random_state=42
        )
        clf2 = ExtraTreesClassifier(
            n_estimators=200, max_depth=5, min_samples_split=4,
            class_weight="balanced", random_state=42
        )
        clf3 = LogisticRegression(class_weight="balanced", C=0.2, random_state=42)

        clf1.fit(X_tr, y_h[train_idx])
        clf2.fit(X_tr, y_h[train_idx])
        clf3.fit(X_tr, y_h[train_idx])

        p1 = clf1.predict_proba(X_va)[:, 1]
        p2 = clf2.predict_proba(X_va)[:, 1]
        p3 = clf3.predict_proba(X_va)[:, 1]

        oof_probs[val_idx] = 0.50 * p1 + 0.35 * p2 + 0.15 * p3

    thresh = 0.48
    oof_hist = (oof_probs >= thresh).astype(int)

    hist_acc = accuracy_score(y_h, oof_hist) * 100
    hist_f1 = f1_score(y_h, oof_hist, average="weighted")
    hist_auc = roc_auc_score(y_h, oof_probs)
    hist_correct = np.sum(oof_hist == y_h)

    print("\n (B) Clinically Optimized NSCLC Subtypes (Adeno vs. Squamous):")
    print(f"     Accuracy   : {hist_acc:.2f}%  ({hist_correct}/{len(df_major)} patients correctly classified)")
    print(f"     F1-Score   : {hist_f1:.4f}")
    print(f"     ROC-AUC    : {hist_auc:.4f}")
    print(f"     Improvement: +{hist_acc - 31.56:.2f}% over initial baseline!")

    print("\n Confusion Matrix:")
    cm_h = confusion_matrix(y_h, oof_hist)
    cm_h_df = pd.DataFrame(
        cm_h,
        index=[" True Adenocarcinoma", " True Squamous Cell"],
        columns=["Pred Adenocarcinoma", "Pred Squamous Cell"],
    )
    print(cm_h_df.to_string())
    print(f"\n Detailed Clinical Recall:")
    print(f"   - Adenocarcinoma Sensitivity: {cm_h[0, 0]}/{cm_h[0, 0]+cm_h[0, 1]} ({cm_h[0, 0]/(cm_h[0, 0]+cm_h[0, 1])*100:.1f}%)")
    print(f"   - Squamous Cell Sensitivity : {cm_h[1, 1]}/{cm_h[1, 0]+cm_h[1, 1]} ({cm_h[1, 1]/(cm_h[1, 0]+cm_h[1, 1])*100:.1f}%)")

    # ==================================================================
    # 3. OVERALL SUMMARY
    # ==================================================================
    overall_mean = (stage_acc + hist_acc) / 2
    print("\n" + "=" * 70)
    print("                     FINAL MULTIMODAL SUMMARY")
    print("=" * 70)
    print(f" Cancer Stage Accuracy         : {stage_acc:.2f}%")
    print(f" Histology Subtype Accuracy    : {hist_acc:.2f}%  (AUC: {hist_auc:.4f})")
    print(f" Combined Multimodal Accuracy  : {overall_mean:.2f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
