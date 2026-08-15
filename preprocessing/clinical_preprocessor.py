from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import Config


class ClinicalPreprocessor:

    def __init__(self):

        self.input_csv = None

        self.output_dir = Config.PROCESSED_DIR / "clinical"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = StandardScaler()

        self.encoders = {}

    # --------------------------------------------------

    def load(self):

        if self.input_csv is None:
            clinical_dir = Config.CLINICAL_DIR
            candidates = sorted(clinical_dir.glob("*.csv"))
            if not candidates:
                raise FileNotFoundError(clinical_dir)
            self.input_csv = candidates[0]

        if not self.input_csv.exists():
            raise FileNotFoundError(self.input_csv)

        df = pd.read_csv(self.input_csv)

        print(f"Loaded {len(df)} patients from {self.input_csv.name}")

        return df

    # --------------------------------------------------

    def clean(self, df):

        df = df.drop_duplicates("PatientID")

        df["age"] = df["age"].fillna(df["age"].median())

        categorical = [

            "gender",
            "clinical.T.Stage",
            "Clinical.N.Stage",
            "Clinical.M.Stage",
            "Overall.Stage",
            "Histology"

        ]

        for c in categorical:

            df[c] = df[c].fillna("Unknown")

        df["Survival.time"] = df["Survival.time"].fillna(0)

        df["deadstatus.event"] = df["deadstatus.event"].fillna(0)

        return df

    # --------------------------------------------------

    def encode(self, df):

        columns = [

            "gender",
            "clinical.T.Stage",
            "Clinical.N.Stage",
            "Clinical.M.Stage",
            "Overall.Stage",
            "Histology"

        ]

        for c in columns:

            encoder = LabelEncoder()

            df[c] = encoder.fit_transform(df[c].astype(str))

            self.encoders[c] = encoder

        return df

    # --------------------------------------------------

    def normalize(self, df):

        df["age"] = self.scaler.fit_transform(

            df[["age"]]

        )

        return df

    # --------------------------------------------------

    def build_feature_table(self, df):

        features = pd.DataFrame({

            "PatientID": df["PatientID"],

            "age": df["age"],

            "gender": df["gender"],

            "T_stage": df["clinical.T.Stage"],

            "N_stage": df["Clinical.N.Stage"],

            "M_stage": df["Clinical.M.Stage"],

        })

        labels = pd.DataFrame({

            "PatientID": df["PatientID"],

            "histology": df["Histology"],

            "stage": df["Overall.Stage"],

            "survival": df["Survival.time"],

            "event": df["deadstatus.event"],

        })

        return features, labels

    # --------------------------------------------------

    def save(self, features, labels):

        features.to_csv(

            self.output_dir / "clinical_features.csv",

            index=False,

        )

        labels.to_csv(

            self.output_dir / "labels.csv",

            index=False,

        )

        with open(

            self.output_dir / "scaler.pkl",

            "wb",

        ) as f:

            pickle.dump(self.scaler, f)

        with open(

            self.output_dir / "label_encoders.pkl",

            "wb",

        ) as f:

            pickle.dump(self.encoders, f)

        feature_names = [

            "age",

            "gender",

            "T_stage",

            "N_stage",

            "M_stage"

        ]

        with open(

            self.output_dir / "feature_names.json",

            "w",

        ) as f:

            json.dump(

                feature_names,

                f,

                indent=4,

            )

        metadata = {

            "num_features": len(feature_names),

            "num_patients": len(features),

            "feature_names": feature_names,

        }

        with open(

            self.output_dir / "metadata.json",

            "w",

        ) as f:

            json.dump(

                metadata,

                f,

                indent=4,

            )

    # --------------------------------------------------

    def run(self):

        df = self.load()

        df = self.clean(df)

        df = self.encode(df)

        df = self.normalize(df)

        features, labels = self.build_feature_table(df)

        self.save(features, labels)

        print("=" * 60)
        print("Clinical preprocessing complete")
        print("=" * 60)
        print(f"Patients : {len(features)}")
        print(f"Features : {features.shape[1]-1}")
        print("=" * 60)


if __name__ == "__main__":

    ClinicalPreprocessor().run()