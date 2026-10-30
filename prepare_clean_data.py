"""
prepare_clean_data.py

Cleans and prepares clinical features and ground-truth labels for LungCancerAI.
Eliminates unlearnable 'Unknown' (NaN) missing entries from the raw data.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from configs.config import Config


def prepare_data(include_nos=True, stage_3_classes=True):
    raw_path = Config.CLINICAL_DIR
    candidates = sorted(raw_path.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No clinical CSV found in {raw_path}")
    
    raw_file = candidates[0]
    df = pd.read_csv(raw_file)
    print(f"Loaded raw clinical file: {raw_file.name} ({len(df)} patients)")

    # 1. Clean Histology
    df = df[df["Histology"].notna() & (df["Histology"] != "Unknown")].copy()
    if not include_nos:
        df = df[df["Histology"] != "nos"].copy()

    # 2. Clean Stage
    df = df[df["Overall.Stage"].notna() & (df["Overall.Stage"] != "Unknown")].copy()

    # 3. Clean CT availability (must have image.npy in processed_dir)
    proc_dir = Config.PROCESSED_DIR
    valid_pids = []
    for pid in df["PatientID"]:
        if (proc_dir / pid / "image.npy").exists():
            valid_pids.append(pid)
    
    df = df[df["PatientID"].isin(valid_pids)].copy().reset_index(drop=True)
    print(f"Total valid patients with CT scans and known labels: {len(df)}")

    # 4. Map Histology
    # 0: adenocarcinoma, 1: large cell, 2: squamous cell carcinoma, (3: nos)
    histology_mapping = {
        "adenocarcinoma": 0,
        "large cell": 1,
        "squamous cell carcinoma": 2,
    }
    if include_nos:
        histology_mapping["nos"] = 3

    df["histology"] = df["Histology"].map(histology_mapping)

    # 5. Map Stage
    if stage_3_classes:
        stage_mapping = {
            "I": 0,
            "II": 1,
            "IIIa": 2,
            "IIIb": 2,
        }
    else:
        stage_mapping = {
            "I": 0,
            "II": 1,
            "IIIa": 2,
            "IIIb": 3,
        }
    df["stage"] = df["Overall.Stage"].map(stage_mapping)

    # 6. Clinical Features
    df["gender"] = (df["gender"].astype(str).str.lower() == "male").astype(float)
    df["age"] = df["age"].fillna(df["age"].median())
    scaler = StandardScaler()
    df["age"] = scaler.fit_transform(df[["age"]])

    df["T_stage"] = pd.to_numeric(df["clinical.T.Stage"], errors="coerce").fillna(1.0)
    df["N_stage"] = pd.to_numeric(df["Clinical.N.Stage"], errors="coerce").fillna(0.0)
    df["M_stage"] = pd.to_numeric(df["Clinical.M.Stage"], errors="coerce").fillna(0.0)

    # Output paths
    out_dir = Config.PROCESSED_DIR / "clinical"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = ["PatientID", "age", "gender", "T_stage", "N_stage", "M_stage"]
    feature_df = df[feature_cols].copy()
    feature_df.to_csv(out_dir / "clean_clinical_features.csv", index=False)

    label_cols = ["PatientID", "histology", "stage"]
    label_df = df[label_cols].copy()
    label_df.to_csv(out_dir / "clean_labels.csv", index=False)

    # Also save combined clean dataset
    combined = feature_df.merge(label_df, on="PatientID")
    combined.to_csv(out_dir / "clean_dataset.csv", index=False)

    print("=" * 60)
    print("Clean Dataset Summary:")
    print("=" * 60)
    print("Histology counts:\n", df["Histology"].value_counts().to_string())
    print("\nStage counts:\n", df["Overall.Stage"].value_counts().to_string())
    print("=" * 60)
    print(f"Saved clean files to {out_dir}")


if __name__ == "__main__":
    prepare_data()
