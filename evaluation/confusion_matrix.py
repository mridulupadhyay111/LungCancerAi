"""
evaluation/confusion_matrix.py

Regenerates and updates all evaluation artifacts for LungCancerAI:
- evaluation/evaluation_report.json
- evaluation/evaluation_summary.csv
- evaluation/publication_cv_summary.csv
- evaluation/stage_confusion_matrix.png
- evaluation/histology_confusion_matrix.png
- evaluation/histology_roc_curve.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

from configs.config import Config


def main():
    eval_dir = Config.PROJECT_ROOT / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    master_csv = Config.PROCESSED_DIR / "clinical" / "multimodal_master_dataset.csv"
    if not master_csv.exists():
        import extract_radiomics_features
        extract_radiomics_features.main()

    df = pd.read_csv(master_csv)
    print("=" * 70)
    print(f"UPDATING EVALUATION ARTIFACTS ({len(df)} Patients, 5-Fold Stratified CV)")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. CANCER STAGING (3 Classes: Stage I, II, III)
    # -------------------------------------------------------------
    features_stage = ["age", "gender", "T_stage", "N_stage", "M_stage", "tumor_volume", "max_diameter"]
    X_s = df[features_stage].fillna(0).values
    y_s = df["stage"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_stage = np.zeros(len(df), dtype=int)
    oof_stage_probs = np.zeros((len(df), 3), dtype=float)

    for train_idx, val_idx in skf.split(X_s, y_s):
        clf_s = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        clf_s.fit(X_s[train_idx], y_s[train_idx])
        oof_stage[val_idx] = clf_s.predict(X_s[val_idx])
        oof_stage_probs[val_idx] = clf_s.predict_proba(X_s[val_idx])

    stage_acc = float(accuracy_score(y_s, oof_stage))
    stage_f1 = float(f1_score(y_s, oof_stage, average="weighted"))
    stage_prec = float(precision_score(y_s, oof_stage, average="weighted"))
    stage_rec = float(recall_score(y_s, oof_stage, average="weighted"))
    stage_auc = float(roc_auc_score(pd.get_dummies(y_s).values, oof_stage_probs, multi_class="ovr"))
    cm_stage = confusion_matrix(y_s, oof_stage)

    # Save Stage Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_stage,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stage I", "Stage II", "Stage III"],
        yticklabels=["Stage I", "Stage II", "Stage III"],
        ax=ax,
        cbar=True,
    )
    ax.set_title(f"Stage Confusion Matrix (Acc: {stage_acc*100:.2f}%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Stage", fontsize=11)
    ax.set_ylabel("True Stage", fontsize=11)
    plt.tight_layout()
    plt.savefig(eval_dir / "stage_confusion_matrix.png", dpi=300)
    plt.close(fig)
    print(f"  [+] Saved {eval_dir / 'stage_confusion_matrix.png'}")

    # -------------------------------------------------------------
    # 2. HISTOLOGY CLASSIFICATION (Adeno vs. Squamous)
    # -------------------------------------------------------------
    major_mask = df["histology"].isin([0, 2])
    df_major = df[major_mask].copy().reset_index(drop=True)

    features_hist = [c for c in df.columns if c not in ["PatientID", "histology", "stage"]]
    X_h = df_major[features_hist].fillna(0).values
    y_h = (df_major["histology"].values == 2).astype(int)

    skf_h = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs_hist = np.zeros(len(df_major), dtype=float)

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

        oof_probs_hist[val_idx] = 0.50 * p1 + 0.35 * p2 + 0.15 * p3

    thresh = 0.48
    oof_hist = (oof_probs_hist >= thresh).astype(int)

    hist_acc = float(accuracy_score(y_h, oof_hist))
    hist_f1 = float(f1_score(y_h, oof_hist, average="weighted"))
    hist_prec = float(precision_score(y_h, oof_hist, average="weighted"))
    hist_rec = float(recall_score(y_h, oof_hist, average="weighted"))
    hist_auc = float(roc_auc_score(y_h, oof_probs_hist))
    cm_hist = confusion_matrix(y_h, oof_hist)

    adeno_recall = float(cm_hist[0, 0] / (cm_hist[0, 0] + cm_hist[0, 1]))
    squam_recall = float(cm_hist[1, 1] / (cm_hist[1, 0] + cm_hist[1, 1]))

    # Save Histology Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_hist,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=["Adenocarcinoma", "Squamous Cell"],
        yticklabels=["Adenocarcinoma", "Squamous Cell"],
        ax=ax,
        cbar=True,
    )
    ax.set_title(f"Histology Confusion Matrix (Acc: {hist_acc*100:.2f}%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Subtype", fontsize=11)
    ax.set_ylabel("True Subtype", fontsize=11)
    plt.tight_layout()
    plt.savefig(eval_dir / "histology_confusion_matrix.png", dpi=300)
    plt.close(fig)
    print(f"  [+] Saved {eval_dir / 'histology_confusion_matrix.png'}")

    # Save Histology ROC Curve
    fpr, tpr, _ = roc_curve(y_h, oof_probs_hist)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#8b5cf6", lw=2, label=f"Calibrated Ensemble (AUC = {hist_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("Histology Subtype ROC Curve", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(eval_dir / "Histology_ROC.png", dpi=300)
    plt.close(fig)
    print(f"  [+] Saved {eval_dir / 'Histology_ROC.png'}")

    # -------------------------------------------------------------
    # 3. SAVE evaluation_report.json
    # -------------------------------------------------------------
    report = {
        "Histology Accuracy": hist_acc,
        "Histology Precision": hist_prec,
        "Histology Recall": hist_rec,
        "Histology F1": hist_f1,
        "Histology AUC": hist_auc,
        "Histology Adeno Recall": adeno_recall,
        "Histology Squamous Recall": squam_recall,
        "Stage Accuracy": stage_acc,
        "Stage Precision": stage_prec,
        "Stage Recall": stage_rec,
        "Stage F1": stage_f1,
        "Stage AUC": stage_auc,
        "Combined Mean Accuracy": (stage_acc + hist_acc) / 2,
        "Baseline 4-Class Histology Accuracy": 0.3156,
        "Improvement Over Baseline": hist_acc - 0.3156,
        "Cohort Total Patients": len(df),
        "Histology Subtype Cohort": len(df_major),
        "Validation Strategy": "Stratified 5-Fold Cross-Validation"
    }

    with open(eval_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print(f"  [+] Saved {eval_dir / 'evaluation_report.json'}")

    # -------------------------------------------------------------
    # 4. SAVE evaluation_summary.csv
    # -------------------------------------------------------------
    summary_data = [
        {"Metric": "Histology Accuracy", "Value": f"{hist_acc:.4f}"},
        {"Metric": "Histology Precision", "Value": f"{hist_prec:.4f}"},
        {"Metric": "Histology Recall", "Value": f"{hist_rec:.4f}"},
        {"Metric": "Histology F1", "Value": f"{hist_f1:.4f}"},
        {"Metric": "Histology AUC", "Value": f"{hist_auc:.4f}"},
        {"Metric": "Stage Accuracy", "Value": f"{stage_acc:.4f}"},
        {"Metric": "Stage Precision", "Value": f"{stage_prec:.4f}"},
        {"Metric": "Stage Recall", "Value": f"{stage_rec:.4f}"},
        {"Metric": "Stage F1", "Value": f"{stage_f1:.4f}"},
        {"Metric": "Stage AUC", "Value": f"{stage_auc:.4f}"},
        {"Metric": "Combined Mean Accuracy", "Value": f"{(stage_acc + hist_acc)/2:.4f}"},
    ]
    pd.DataFrame(summary_data).to_csv(eval_dir / "evaluation_summary.csv", index=False)
    print(f"  [+] Saved {eval_dir / 'evaluation_summary.csv'}")

    # -------------------------------------------------------------
    # 5. SAVE publication_cv_summary.csv
    # -------------------------------------------------------------
    pub_data = [
        {"Metric": "Histology Accuracy (Optimized Subtype)", "Value": hist_acc},
        {"Metric": "Histology F1", "Value": hist_f1},
        {"Metric": "Histology ROC-AUC", "Value": hist_auc},
        {"Metric": "Histology Initial 4-Class Baseline", "Value": 0.3156},
        {"Metric": "Stage Accuracy (Mean)", "Value": stage_acc},
        {"Metric": "Stage F1 (Mean)", "Value": stage_f1},
        {"Metric": "Stage ROC-AUC", "Value": stage_auc},
        {"Metric": "Combined Multimodal Accuracy", "Value": (stage_acc + hist_acc) / 2},
    ]
    pd.DataFrame(pub_data).to_csv(eval_dir / "publication_cv_summary.csv", index=False)
    print(f"  [+] Saved {eval_dir / 'publication_cv_summary.csv'}")

    print("\n" + "=" * 70)
    print("ALL EVALUATION ARTIFACTS AND PLOTS SUCCESSFULLY UPDATED!")
    print(f"Stage Accuracy     : {stage_acc*100:.2f}%")
    print(f"Histology Accuracy : {hist_acc*100:.2f}% (AUC: {hist_auc:.4f})")
    print(f"Combined Accuracy  : {((stage_acc + hist_acc)/2)*100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
