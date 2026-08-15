"""
clinical_dataset.py

Research-grade clinical dataset for LungCancerAI.

Author: LungCancerAI
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset


class ClinicalDataset(Dataset):

    def __init__(
        self,
        csv_file,
        patient_ids: Optional[List[str]] = None,
        fit_scaler: bool = True,
    ):

        self.csv_file = Path(csv_file)

        if not self.csv_file.exists():
            raise FileNotFoundError(self.csv_file)

        self.df = pd.read_csv(self.csv_file)

        if patient_ids is not None:
            self.df = self.df[
                self.df["PatientID"].isin(patient_ids)
            ].reset_index(drop=True)

        # ----------------------------
        # Missing values
        # ----------------------------

        self.df["age"] = self.df["age"].fillna(
            self.df["age"].median()
        )

        categorical = [
            "clinical.T.Stage",
            "Clinical.N.Stage",
            "Clinical.M.Stage",
            "Overall.Stage",
            "Histology",
            "gender",
        ]

        for col in categorical:
            self.df[col] = self.df[col].fillna("Unknown")

        # ----------------------------
        # Encode categorical variables
        # ----------------------------

        self.encoders = {}

        for col in categorical:

            encoder = LabelEncoder()

            self.df[col] = encoder.fit_transform(
                self.df[col].astype(str)
            )

            self.encoders[col] = encoder

        # ----------------------------
        # Numerical Features
        # ----------------------------

        numerical = [
            "age",
        ]

        self.scaler = StandardScaler()

        if fit_scaler:
            self.df[numerical] = self.scaler.fit_transform(
                self.df[numerical]
            )
        else:
            self.df[numerical] = self.scaler.transform(
                self.df[numerical]
            )

        # ----------------------------
        # Feature Columns
        # ----------------------------

        self.feature_columns = [

            "age",

            "clinical.T.Stage",

            "Clinical.N.Stage",

            "Clinical.M.Stage",

            "Overall.Stage",

            "Histology",

            "gender"

        ]

    # --------------------------------------------------

    def __len__(self):

        return len(self.df)

    # --------------------------------------------------

    def __getitem__(self, index):

        row = self.df.iloc[index]

        features = torch.tensor(

            row[self.feature_columns].values.astype(
                np.float32
            )

        )

        label = torch.tensor(

            int(row["deadstatus.event"]),

            dtype=torch.long

        )

        survival = torch.tensor(

            float(row["Survival.time"]),

            dtype=torch.float32

        )

        return {

            "patient_id": row["PatientID"],

            "clinical": features,

            "label": label,

            "survival": survival

        }

    # --------------------------------------------------

    def get_patient_ids(self):

        return self.df["PatientID"].tolist()

    # --------------------------------------------------

    def feature_size(self):

        return len(self.feature_columns)

    # --------------------------------------------------

    def summary(self):

        print("=" * 60)

        print("Clinical Dataset")

        print("=" * 60)

        print("Patients :", len(self))

        print("Features :", len(self.feature_columns))

        print()

        print(self.feature_columns)

        print("=" * 60)


if __name__ == "__main__":

    dataset = ClinicalDataset(

        csv_file=r"D:\LungCancerAI\data\raw\clinical\NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"

    )

    dataset.summary()

    sample = dataset[0]

    print(sample)