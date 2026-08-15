"""
evaluate.py

Research-grade Evaluation Script

Evaluates

1. Histology Classification
2. Stage Prediction
3. Survival Prediction

Outputs

- Accuracy
- Precision
- Recall
- F1
- ROC
- AUC
- Confusion Matrix
- MAE
- RMSE
- C-index

Author:
LungCancerAI
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    confusion_matrix,

    classification_report,

)

from dataset.multimodal_dataset import MultiModalDataset

from models.multimodal_model import MultiModalModel
# =====================================================
# Arguments
# =====================================================

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--model",

        required=True,

        type=str,

    )

    parser.add_argument(

        "--csv",

        required=True,

        type=str,

    )

    parser.add_argument(

        "--batch_size",

        default=8,

        type=int,

    )

    parser.add_argument(

        "--device",

        default="cuda",

        type=str,

    )

    parser.add_argument(

        "--output",

        default="evaluation",

        type=str,

    )

    return parser.parse_args()


# =====================================================
# Device
# =====================================================

def get_device(name):

    if name == "cuda" and torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# =====================================================
# Dataset
# =====================================================

def build_loader(args):

    dataset = MultiModalDataset(

        csv_file=args.csv,

        training=False,

    )

    loader = DataLoader(

        dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=4,

        pin_memory=True,

    )

    print()

    print("=" * 70)

    print("Evaluation Dataset")

    print("=" * 70)

    print(

        "Samples:",

        len(dataset),

    )

    print("=" * 70)

    return loader


# =====================================================
# Model
# =====================================================

def load_model(

    checkpoint,

    device,

):

    model = MultiModalModel()

    state = torch.load(

        checkpoint,

        map_location=device,

    )

    if "model_state_dict" in state:

        model.load_state_dict(

            state["model_state_dict"]

        )

    else:

        model.load_state_dict(state)

    model.to(device)

    model.eval()

    return model
# =====================================================
# Inference
# =====================================================

@torch.no_grad()

def inference(

    model,

    loader,

    device,

):

    histology_predictions = []

    stage_predictions = []

    survival_predictions = []

    histology_probabilities = []

    stage_probabilities = []

    histology_targets = []

    stage_targets = []

    survival_targets = []

    for batch in loader:

        images = batch["image"].to(device)

        clinical = batch["clinical"].to(device)

        outputs = model(

            images,

            clinical,

        )

        histology_logits = outputs["histology_logits"]

        stage_logits = outputs["stage_logits"]

        survival_output = outputs["survival_prediction"]

        histology_probability = torch.softmax(

            histology_logits,

            dim=1,

        )

        stage_probability = torch.softmax(

            stage_logits,

            dim=1,

        )

        histology_prediction = torch.argmax(

            histology_probability,

            dim=1,

        )

        stage_prediction = torch.argmax(

            stage_probability,

            dim=1,

        )

        histology_predictions.extend(

            histology_prediction.cpu().numpy()

        )

        stage_predictions.extend(

            stage_prediction.cpu().numpy()

        )

        survival_predictions.extend(

            survival_output.squeeze().cpu().numpy()

        )

        histology_probabilities.extend(

            histology_probability.cpu().numpy()

        )

        stage_probabilities.extend(

            stage_probability.cpu().numpy()

        )

        histology_targets.extend(

            batch["histology"].numpy()

        )

        stage_targets.extend(

            batch["stage"].numpy()

        )

        survival_targets.extend(

            batch["survival"].numpy()

        )

    return {

        "histology_prediction":

            np.array(histology_predictions),

        "stage_prediction":

            np.array(stage_predictions),

        "survival_prediction":

            np.array(survival_predictions),

        "histology_probability":

            np.array(histology_probabilities),

        "stage_probability":

            np.array(stage_probabilities),

        "histology_target":

            np.array(histology_targets),

        "stage_target":

            np.array(stage_targets),

        "survival_target":

            np.array(survival_targets),

    }
# =====================================================
# Classification Metrics
# =====================================================

def classification_metrics(

    target,

    prediction,

):

    metrics = {}

    metrics["accuracy"] = accuracy_score(

        target,

        prediction,

    )

    metrics["precision"] = precision_score(

        target,

        prediction,

        average="weighted",

        zero_division=0,

    )

    metrics["recall"] = recall_score(

        target,

        prediction,

        average="weighted",

        zero_division=0,

    )

    metrics["f1"] = f1_score(

        target,

        prediction,

        average="weighted",

        zero_division=0,

    )

    metrics["confusion_matrix"] = confusion_matrix(

        target,

        prediction,

    )

    metrics["classification_report"] = classification_report(

        target,

        prediction,

        zero_division=0,

        output_dict=True,

    )

    return metrics


# =====================================================
# ROC-AUC
# =====================================================

def multiclass_auc(

    target,

    probability,

):

    try:

        auc = roc_auc_score(

            target,

            probability,

            multi_class="ovr",

        )

    except Exception:

        auc = None

    return auc


# =====================================================
# Print Metrics
# =====================================================

def print_metrics(

    name,

    metrics,

):

    print()

    print("=" * 70)

    print(name)

    print("=" * 70)

    print(

        f"Accuracy : {metrics['accuracy']:.4f}"

    )

    print(

        f"Precision: {metrics['precision']:.4f}"

    )

    print(

        f"Recall   : {metrics['recall']:.4f}"

    )

    print(

        f"F1 Score : {metrics['f1']:.4f}"

    )

    print("=" * 70)
# =====================================================
# Survival Metrics
# =====================================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def survival_metrics(

    target,

    prediction,

):

    metrics = {}

    metrics["mae"] = mean_absolute_error(

        target,

        prediction,

    )

    metrics["rmse"] = np.sqrt(

        mean_squared_error(

            target,

            prediction,

        )

    )

    return metrics


# =====================================================
# Concordance Index
# =====================================================

def concordance_index(

    target,

    prediction,

):

    try:

        from lifelines.utils import concordance_index

        score = concordance_index(

            target,

            -prediction,

        )

    except Exception:

        score = None

    return score


# =====================================================
# Print Survival Metrics
# =====================================================

def print_survival_metrics(

    metrics,

):

    print()

    print("=" * 70)

    print("Survival Prediction")

    print("=" * 70)

    print(

        f"MAE  : {metrics['mae']:.4f}"

    )

    print(

        f"RMSE : {metrics['rmse']:.4f}"

    )

    print("=" * 70)
# =====================================================
# Visualization
# =====================================================

import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


# =====================================================
# Confusion Matrix
# =====================================================

def save_confusion_matrix(

    matrix,

    labels,

    filename,

):

    fig, ax = plt.subplots(

        figsize=(8, 8)

    )

    display = ConfusionMatrixDisplay(

        confusion_matrix=matrix,

        display_labels=labels,

    )

    display.plot(

        ax=ax,

        cmap="Blues",

        colorbar=False,

    )

    plt.tight_layout()

    plt.savefig(

        filename,

        dpi=300,

    )

    plt.close(fig)


# =====================================================
# ROC Curves
# =====================================================

def save_multiclass_roc(

    targets,

    probabilities,

    output_dir,

    title,

):

    from sklearn.preprocessing import label_binarize

    n_classes = probabilities.shape[1]

    binary_target = label_binarize(

        targets,

        classes=np.arange(n_classes),

    )

    plt.figure(

        figsize=(8, 8)

    )

    for i in range(n_classes):

        RocCurveDisplay.from_predictions(

            binary_target[:, i],

            probabilities[:, i],

            name=f"Class {i}",

        )

    plt.title(title)

    plt.savefig(

        Path(output_dir)

        / f"{title}.png",

        dpi=300,

        bbox_inches="tight",

    )

    plt.close()


# =====================================================
# Precision Recall Curve
# =====================================================

def save_precision_recall(

    targets,

    probabilities,

    output_dir,

    title,

):

    from sklearn.metrics import PrecisionRecallDisplay

    from sklearn.preprocessing import label_binarize

    binary_target = label_binarize(

        targets,

        classes=np.arange(

            probabilities.shape[1]

        ),

    )

    plt.figure(

        figsize=(8, 8)

    )

    for i in range(

        probabilities.shape[1]

    ):

        PrecisionRecallDisplay.from_predictions(

            binary_target[:, i],

            probabilities[:, i],

            name=f"Class {i}",

        )

    plt.title(title)

    plt.savefig(

        Path(output_dir)

        / f"{title}_PR.png",

        dpi=300,

        bbox_inches="tight",

    )

    plt.close()
# =====================================================
# Create Output Directory
# =====================================================

import json
import csv


def create_output_directory(

    output_path,

):

    output_dir = Path(output_path)

    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    return output_dir


# =====================================================
# Save JSON Report
# =====================================================

def save_json_report(

    report,

    output_dir,

):

    file = output_dir / "evaluation_report.json"

    with open(

        file,

        "w",

    ) as f:

        json.dump(

            report,

            f,

            indent=4,

        )

    print()

    print(

        f"JSON report saved to {file}"

    )


# =====================================================
# Save CSV Report
# =====================================================

def save_csv_report(

    report,

    output_dir,

):

    file = output_dir / "evaluation_summary.csv"

    with open(

        file,

        "w",

        newline="",

    ) as f:

        writer = csv.writer(f)

        writer.writerow(

            [

                "Metric",

                "Value",

            ]

        )

        for key, value in report.items():

            if isinstance(

                value,

                (int, float),

            ):

                writer.writerow(

                    [

                        key,

                        value,

                    ]

                )

    print(

        f"CSV report saved to {file}"

    )
# =====================================================
# Complete Evaluation
# =====================================================

def evaluate(

    outputs,

    output_dir,

):

    histology = classification_metrics(

        outputs["histology_target"],

        outputs["histology_prediction"],

    )

    stage = classification_metrics(

        outputs["stage_target"],

        outputs["stage_prediction"],

    )

    survival = survival_metrics(

        outputs["survival_target"],

        outputs["survival_prediction"],

    )

    histology_auc = multiclass_auc(

        outputs["histology_target"],

        outputs["histology_probability"],

    )

    stage_auc = multiclass_auc(

        outputs["stage_target"],

        outputs["stage_probability"],

    )

    survival["c_index"] = concordance_index(

        outputs["survival_target"],

        outputs["survival_prediction"],

    )

    report = {

        "Histology Accuracy":

            histology["accuracy"],

        "Histology Precision":

            histology["precision"],

        "Histology Recall":

            histology["recall"],

        "Histology F1":

            histology["f1"],

        "Histology AUC":

            histology_auc,

        "Stage Accuracy":

            stage["accuracy"],

        "Stage Precision":

            stage["precision"],

        "Stage Recall":

            stage["recall"],

        "Stage F1":

            stage["f1"],

        "Stage AUC":

            stage_auc,

        "Survival MAE":

            survival["mae"],

        "Survival RMSE":

            survival["rmse"],

        "Concordance Index":

            survival["c_index"],

    }

    print_metrics(

        "Histology Classification",

        histology,

    )

    print_metrics(

        "Stage Classification",

        stage,

    )

    print_survival_metrics(

        survival,

    )

    save_confusion_matrix(

        histology["confusion_matrix"],

        np.unique(

            outputs["histology_target"]

        ),

        output_dir

        / "histology_confusion_matrix.png",

    )

    save_confusion_matrix(

        stage["confusion_matrix"],

        np.unique(

            outputs["stage_target"]

        ),

        output_dir

        / "stage_confusion_matrix.png",

    )

    save_multiclass_roc(

        outputs["histology_target"],

        outputs["histology_probability"],

        output_dir,

        "Histology_ROC",

    )

    save_multiclass_roc(

        outputs["stage_target"],

        outputs["stage_probability"],

        output_dir,

        "Stage_ROC",

    )

    save_precision_recall(

        outputs["histology_target"],

        outputs["histology_probability"],

        output_dir,

        "Histology",

    )

    save_precision_recall(

        outputs["stage_target"],

        outputs["stage_probability"],

        output_dir,

        "Stage",

    )

    save_json_report(

        report,

        output_dir,

    )

    save_csv_report(

        report,

        output_dir,

    )

    return report
# =====================================================
# Main Evaluation Pipeline
# =====================================================

def run_evaluation(args):

    device = get_device(

        args.device,

    )

    print("=" * 70)

    print("Evaluation Started")

    print("=" * 70)

    output_dir = create_output_directory(

        args.output,

    )

    loader = build_loader(

        args,

    )

    model = load_model(

        args.model,

        device,

    )

    outputs = inference(

        model,

        loader,

        device,

    )

    report = evaluate(

        outputs,

        output_dir,

    )

    print()

    print("=" * 70)

    print("Final Results")

    print("=" * 70)

    for key, value in report.items():

        print(

            f"{key:30s}: {value}"

        )

    print("=" * 70)

    return report
# =====================================================
# Utilities
# =====================================================

import time


def print_runtime(

    start_time,

):

    elapsed = time.time() - start_time

    print()

    print("=" * 70)

    print(

        f"Evaluation completed in {elapsed:.2f} seconds"

    )

    print("=" * 70)


# =====================================================
# Main
# =====================================================

def main():

    start = time.time()

    args = parse_arguments()

    try:

        run_evaluation(

            args,

        )

        print_runtime(

            start,

        )

        print()

        print("=" * 70)

        print("Evaluation Completed Successfully")

        print("=" * 70)

    except Exception as error:

        print()

        print("=" * 70)

        print("Evaluation Failed")

        print("=" * 70)

        print(error)

        raise


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":

    main()                                