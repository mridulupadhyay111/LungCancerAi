"""
optimize_histology.py

Dedicated Histology Subtype Optimization Pipeline for NSCLC.
Extracts 3D texture heterogeneity, entropy, and spatial boundary markers.
Tunes balanced classifiers specifically to maximize Adeno vs. Squamous accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from configs.config import Config


def main():
    master_csv = Config.PROCESSED_DIR / "clinical" / "multimodal_master_dataset.csv"
    if not master_csv.exists():
        import extract_radiomics_features
        extract_radiomics_features.main()

    df = pd.read_csv(master_csv)

    print("=" * 70)
    print("HISTOLOGY SUBTYPE OPTIMIZATION (Adenocarcinoma vs. Squamous Cell)")
    print("=" * 70)

    # Focus on true biological subtypes (Adeno = 0, Squamous = 2)
    major_df = df[df["histology"].isin([0, 2])].copy().reset_index(drop=True)
    print(f"Total Cohort: {len(major_df)} Patients")
    print(f"Adenocarcinoma: {sum(major_df['histology'] == 0)} | Squamous Cell: {sum(major_df['histology'] == 2)}")

    feature_cols = [c for c in major_df.columns if c not in ["PatientID", "histology", "stage"]]
    X = major_df[feature_cols].fillna(0).values
    y = (major_df["histology"].values == 2).astype(int)

    # Rebalance weights: Adenocarcinoma gets 3.0x weight
    class_weights = {0: 3.0, 1: 1.0}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifiers = {
        "Balanced ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_depth=5,
            min_samples_split=4,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=3,
            class_weight="balanced",
            l2_regularization=2.0,
            learning_rate=0.03,
            random_state=42,
        ),
        "Balanced RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            max_depth=4,
            min_samples_split=4,
            random_state=42,
        ),
    }

    for name, clf in classifiers.items():
        oof_preds = np.zeros(len(y), dtype=int)
        oof_probs = np.zeros(len(y), dtype=float)

        for train_idx, val_idx in skf.split(X, y):
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[val_idx])[:, 1]
            oof_probs[val_idx] = probs
            oof_preds[val_idx] = (probs >= 0.50).astype(int)

        acc = accuracy_score(y, oof_preds) * 100
        f1 = f1_score(y, oof_preds, average="weighted")
        auc = roc_auc_score(y, oof_probs)
        cm = confusion_matrix(y, oof_preds)
        sens_adeno = cm[0, 0] / (cm[0, 0] + cm[0, 1]) * 100
        sens_squam = cm[1, 1] / (cm[1, 0] + cm[1, 1]) * 100

        print(f"\nModel: {name}")
        print(f"Overall Accuracy         : {acc:.2f}%")
        print(f"F1-Score                 : {f1:.4f} | ROC-AUC: {auc:.4f}")
        print(f"Adeno Recall (Caught)    : {sens_adeno:.1f}% ({cm[0, 0]}/{cm[0, 0]+cm[0, 1]})")
        print(f"Squamous Recall (Caught) : {sens_squam:.1f}% ({cm[1, 1]}/{cm[1, 0]+cm[1, 1]})")
        print(f"Balanced Mean Accuracy   : {(sens_adeno + sens_squam)/2:.2f}%")


if __name__ == "__main__":
    main()
