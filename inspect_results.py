"""
inspect_results.py

Inspects the dataset and model predictions on real patients.
Shows you patient-by-patient Ground Truth vs Model Predictions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from configs.config import Config


def main():
    clean_csv = Config.PROCESSED_DIR / "clinical" / "clean_dataset.csv"
    if not clean_csv.exists():
        import prepare_clean_data
        prepare_clean_data.prepare_data()

    df = pd.read_csv(clean_csv)

    print("=" * 75)
    print("STEP 1: INSPECTING THE CLEANED DATASET (REAL PATIENT SAMPLES)")
    print("=" * 75)
    
    stage_names = {0: "Stage I", 1: "Stage II", 2: "Stage III"}
    hist_names = {0: "Adenocarcinoma", 1: "Large Cell", 2: "Squamous Cell", 3: "NOS"}

    sample_view = df[["PatientID", "age", "gender", "T_stage", "N_stage", "M_stage", "stage", "histology"]].copy()
    sample_view["Stage_Name"] = sample_view["stage"].map(stage_names)
    sample_view["Histology_Name"] = sample_view["histology"].map(hist_names)
    sample_view["Gender_Name"] = sample_view["gender"].map({1.0: "Male", 0.0: "Female"})

    print(sample_view[["PatientID", "Gender_Name", "T_stage", "N_stage", "Stage_Name", "Histology_Name"]].head(10).to_string(index=False))

    print("\n" + "=" * 75)
    print("STEP 2: RUNNING REAL HELD-OUT CROSS-VALIDATION PREDICTIONS")
    print("=" * 75)

    X = df[["age", "gender", "T_stage", "N_stage", "M_stage"]].values
    y_stage = df["stage"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store out-of-fold predictions
    oof_preds = np.zeros(len(df), dtype=int)
    
    for train_idx, val_idx in skf.split(X, y_stage):
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X[train_idx], y_stage[train_idx])
        oof_preds[val_idx] = clf.predict(X[val_idx])

    df["Predicted_Stage"] = oof_preds
    df["True_Stage_Name"] = df["stage"].map(stage_names)
    df["Pred_Stage_Name"] = df["Predicted_Stage"].map(stage_names)
    df["Correct"] = df["stage"] == df["Predicted_Stage"]

    print("\n--- 10 Real Patient Predictions on Strictly Held-Out Test Folds ---")
    inspect_cols = ["PatientID", "T_stage", "N_stage", "True_Stage_Name", "Pred_Stage_Name", "Correct"]
    print(df[inspect_cols].head(10).to_string(index=False))

    correct_count = df["Correct"].sum()
    total_count = len(df)
    print("\n" + "-" * 75)
    print(f"Total Correct: {correct_count} / {total_count} patients ({correct_count / total_count * 100:.2f}% Accuracy)")
    print("-" * 75)

    print("\n--- Stage Confusion Matrix (Rows = True Stage, Columns = Predicted Stage) ---")
    cm = confusion_matrix(df["stage"], df["Predicted_Stage"])
    cm_df = pd.DataFrame(
        cm,
        index=["True Stage I", "True Stage II", "True Stage III"],
        columns=["Pred Stage I", "Pred Stage II", "Pred Stage III"],
    )
    print(cm_df.to_string())
    print("=" * 75)


if __name__ == "__main__":
    main()
