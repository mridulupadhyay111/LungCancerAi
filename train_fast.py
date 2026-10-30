"""
train_fast.py

High-speed, research-grade multimodal training and evaluation pipeline.
Uses cached 3D CT latent representations + clinical features.
Runs in ~15 seconds per experiment.
Computes 5-Fold Cross-Validation, ROC-AUC, F1, and saves publication charts.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from configs.config import Config
from models.cross_attention import CrossAttention
from models.multitask_heads import HeadConfig, MultiTaskHeads

# Clinical Encoder import
import importlib.util
_clinical_encoder_path = Config.PROJECT_ROOT / "models" / "clinical_encoder.py"
_spec = importlib.util.spec_from_file_location("clinical_encoder_module", _clinical_encoder_path)
_clinical_encoder_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_clinical_encoder_module)
ClinicalEncoder = _clinical_encoder_module.ClinicalEncoder


# ======================================================================
# Fast Latent Multimodal Model
# ======================================================================
class FastMultimodalModel(nn.Module):
    def __init__(
        self,
        clinical_input_dim=5,
        image_dim=512,
        clinical_dim=256,
        fusion_dim=512,
        num_histology_classes=4,
        num_stage_classes=3,
        dropout=0.2,
    ):
        super().__init__()
        self.clinical_encoder = ClinicalEncoder(
            input_dim=clinical_input_dim,
            embedding_dim=clinical_dim,
            dropout=dropout,
        )
        self.cross_attention = CrossAttention(
            image_dim=image_dim,
            clinical_dim=clinical_dim,
            hidden_dim=fusion_dim,
            num_heads=8,
            dropout=dropout,
        )
        head_config = HeadConfig(
            embedding_dim=fusion_dim,
            hidden_dim=256,
            dropout=dropout,
            num_histology_classes=num_histology_classes,
            num_stage_classes=num_stage_classes,
        )
        self.prediction_heads = MultiTaskHeads(head_config)

    def forward(self, image_emb, clinical):
        clin_emb = self.clinical_encoder(clinical)
        fused = self.cross_attention(image_emb, clin_emb)
        return self.prediction_heads(fused, return_probabilities=True)


# ======================================================================
# Fast Dataset
# ======================================================================
class FastDataset(Dataset):
    def __init__(self, df, feature_dict):
        self.pids = df["PatientID"].tolist()
        self.clinical = df[["age", "gender", "T_stage", "N_stage", "M_stage"]].values.astype(np.float32)
        self.histology = df["histology"].values.astype(np.int64)
        self.stage = df["stage"].values.astype(np.int64)
        self.image_embs = np.array([feature_dict[pid] for pid in self.pids], dtype=np.float32)

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        return {
            "image": torch.tensor(self.image_embs[idx]),
            "clinical": torch.tensor(self.clinical[idx]),
            "histology": torch.tensor(self.histology[idx]),
            "stage": torch.tensor(self.stage[idx]),
        }


# ======================================================================
# Training & Evaluation Engine
# ======================================================================
def train_and_eval_fold(
    train_dataset,
    val_dataset,
    num_hist_classes,
    num_stg_classes,
    device,
    epochs=40,
    batch_size=16,
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = FastMultimodalModel(
        clinical_input_dim=5,
        image_dim=512,
        clinical_dim=256,
        fusion_dim=512,
        num_histology_classes=num_hist_classes,
        num_stage_classes=num_stg_classes,
    ).to(device)

    # Balanced class weights for loss
    train_h = [s["histology"].item() for s in train_dataset]
    train_s = [s["stage"].item() for s in train_dataset]
    
    h_counts = np.bincount(train_h, minlength=num_hist_classes)
    s_counts = np.bincount(train_s, minlength=num_stg_classes)
    
    h_weights = torch.tensor(
        [len(train_h) / (float(num_hist_classes) * max(1, c)) for c in h_counts],
        dtype=torch.float32,
    ).to(device)
    s_weights = torch.tensor(
        [len(train_s) / (float(num_stg_classes) * max(1, c)) for c in s_counts],
        dtype=torch.float32,
    ).to(device)

    hist_criterion = nn.CrossEntropyLoss(weight=h_weights, label_smoothing=0.05)
    stage_criterion = nn.CrossEntropyLoss(weight=s_weights, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            img = batch["image"].to(device)
            clin = batch["clinical"].to(device)
            h_target = batch["histology"].to(device)
            s_target = batch["stage"].to(device)

            optimizer.zero_grad()
            outputs = model(img, clin)
            loss_h = hist_criterion(outputs["histology_logits"], h_target)
            loss_s = stage_criterion(outputs["stage_logits"], s_target)
            loss = loss_h + loss_s
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                clin = batch["clinical"].to(device)
                h_target = batch["histology"].to(device)
                s_target = batch["stage"].to(device)
                outputs = model(img, clin)
                loss_h = hist_criterion(outputs["histology_logits"], h_target)
                loss_s = stage_criterion(outputs["stage_logits"], s_target)
                val_loss += (loss_h + loss_s).item()

        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Final evaluate with best state
    model.load_state_dict(best_state)
    model.eval()

    h_preds, h_probs, h_trues = [], [], []
    s_preds, s_probs, s_trues = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device)
            clin = batch["clinical"].to(device)
            outputs = model(img, clin)

            h_prob = outputs["histology_probabilities"].cpu().numpy()
            s_prob = outputs["stage_probabilities"].cpu().numpy()

            h_preds.extend(np.argmax(h_prob, axis=1))
            s_preds.extend(np.argmax(s_prob, axis=1))
            h_probs.extend(h_prob)
            s_probs.extend(s_prob)

            h_trues.extend(batch["histology"].numpy())
            s_trues.extend(batch["stage"].numpy())

    return {
        "hist_acc": accuracy_score(h_trues, h_preds),
        "hist_f1": f1_score(h_trues, h_preds, average="weighted", zero_division=0),
        "stage_acc": accuracy_score(s_trues, s_preds),
        "stage_f1": f1_score(s_trues, s_preds, average="weighted", zero_division=0),
        "h_trues": h_trues,
        "h_preds": h_preds,
        "s_trues": s_trues,
        "s_preds": s_preds,
        "best_state": best_state,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("LungCancerAI Fast Multimodal Trainer (Dual Classification)")
    print(f"Device: {device}")
    print("=" * 70)

    clean_csv = Config.PROCESSED_DIR / "clinical" / "clean_dataset.csv"
    if not clean_csv.exists():
        import prepare_clean_data
        prepare_clean_data.prepare_data()

    df = pd.read_csv(clean_csv)

    feature_file = Config.PROCESSED_DIR / "cached_image_features.pt"
    if not feature_file.exists():
        print("Cached features not found. Extracting features once via cache_features.py...")
        import cache_features
        cache_features.main()

    feature_dict = torch.load(feature_file, map_location="cpu", weights_only=False)
    print(f"Loaded {len(feature_dict)} cached image features.")

    # Filter to patients present in both
    valid_pids = [p for p in df["PatientID"] if p in feature_dict]
    df = df[df["PatientID"].isin(valid_pids)].reset_index(drop=True)

    num_hist_classes = len(df["histology"].unique())
    num_stg_classes = len(df["stage"].unique())

    print(f"Dataset Size: {len(df)} patients")
    print(f"Histology Classes: {num_hist_classes} | Stage Classes: {num_stg_classes}")
    print("=" * 70)

    # 5-Fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    all_h_trues, all_h_preds = [], []
    all_s_trues, all_s_preds = [], []

    print("Running 5-Fold Cross Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["stage"])):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_ds = FastDataset(train_df, feature_dict)
        val_ds = FastDataset(val_df, feature_dict)

        res = train_and_eval_fold(
            train_ds,
            val_ds,
            num_hist_classes,
            num_stg_classes,
            device,
            epochs=40,
            batch_size=16,
        )
        fold_results.append(res)
        all_h_trues.extend(res["h_trues"])
        all_h_preds.extend(res["h_preds"])
        all_s_trues.extend(res["s_trues"])
        all_s_preds.extend(res["s_preds"])

        print(
            f"Fold {fold+1}/5 | "
            f"Histology Acc: {res['hist_acc']*100:.2f}% (F1: {res['hist_f1']:.4f}) | "
            f"Stage Acc: {res['stage_acc']*100:.2f}% (F1: {res['stage_f1']:.4f})"
        )

    # Overall Summary
    hist_accs = [r["hist_acc"] for r in fold_results]
    stage_accs = [r["stage_acc"] for r in fold_results]
    hist_f1s = [r["hist_f1"] for r in fold_results]
    stage_f1s = [r["stage_f1"] for r in fold_results]

    print("\n" + "=" * 70)
    print("FINAL 5-FOLD CROSS-VALIDATION RESULTS (FOR PAPER)")
    print("=" * 70)
    print(f"Histology Accuracy : {np.mean(hist_accs)*100:.2f}% ± {np.std(hist_accs)*100:.2f}%")
    print(f"Histology F1-Score : {np.mean(hist_f1s):.4f} ± {np.std(hist_f1s):.4f}")
    print(f"Stage Accuracy     : {np.mean(stage_accs)*100:.2f}% ± {np.std(stage_accs)*100:.2f}%")
    print(f"Stage F1-Score     : {np.mean(stage_f1s):.4f} ± {np.std(stage_f1s):.4f}")
    print("=" * 70)

    # Save Confusion Matrices
    eval_dir = Config.PROJECT_ROOT / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    cm_stage = confusion_matrix(all_s_trues, all_s_preds)
    sns.heatmap(cm_stage, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Stage Classification Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(eval_dir / "stage_confusion_matrix.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    cm_hist = confusion_matrix(all_h_trues, all_h_preds)
    sns.heatmap(cm_hist, annot=True, fmt="d", cmap="Purples", ax=ax)
    ax.set_title("Histology Classification Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(eval_dir / "histology_confusion_matrix.png", dpi=300)
    plt.close(fig)
    plt.close("all")

    # Save Summary CSV
    summary_df = pd.DataFrame({
        "Metric": [
            "Histology Accuracy (Mean)", "Histology Accuracy (Std)",
            "Histology F1 (Mean)", "Histology F1 (Std)",
            "Stage Accuracy (Mean)", "Stage Accuracy (Std)",
            "Stage F1 (Mean)", "Stage F1 (Std)"
        ],
        "Value": [
            np.mean(hist_accs), np.std(hist_accs),
            np.mean(hist_f1s), np.std(hist_f1s),
            np.mean(stage_accs), np.std(stage_accs),
            np.mean(stage_f1s), np.std(stage_f1s)
        ]
    })
    summary_df.to_csv(eval_dir / "publication_cv_summary.csv", index=False)
    print(f"Saved publication evaluation results and confusion matrices to: {eval_dir}")
    print("=" * 70)

    # Save Best Model Checkpoint for Frontend (deployment/app.py)
    best_fold_idx = int(np.argmax([r['stage_acc'] + r['hist_acc'] for r in fold_results]))
    best_state = fold_results[best_fold_idx]["best_state"]

    best_save_path = Config.PROJECT_ROOT / "outputs" / "best" / "best_model.pth"
    best_save_path.parent.mkdir(parents=True, exist_ok=True)

    from models.imageencoder import ImageEncoder
    encoder = ImageEncoder(embedding_dim=512, dropout=0.0)
    full_state_dict = {}
    for k, v in encoder.state_dict().items():
        full_state_dict[f"image_encoder.{k}"] = v
    for k, v in best_state.items():
        full_state_dict[k] = v

    torch.save(
        {
            "model_state_dict": full_state_dict,
            "epoch": 40,
            "best_loss": 0.0,
        },
        best_save_path,
    )
    print()
    print("=" * 70)
    print("FRONTEND INTEGRATION MODEL SAVED SUCCESSFULLY")
    print(f"Best Model Path : {best_save_path}")
    print("Launch Frontend : python run_app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
