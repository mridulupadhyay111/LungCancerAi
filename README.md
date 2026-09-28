# 🫁 LungCancerAI: Multimodal Explainable AI for Lung Cancer Diagnosis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-Medical%20AI-5C2D91.svg)](https://monai.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**LungCancerAI** is an easy-to-understand, research-grade Artificial Intelligence framework designed for automated **Non-Small Cell Lung Cancer (NSCLC)** diagnosis. By integrating **3D Volumetric CT Scans** with **Patient Clinical Records**, it simultaneously predicts cancer subtype, tumor stage, and survival expectancy—backed by transparent visual explanations.

---

## 📌 At a Glance: 3 Core Diagnostic Tasks

1. **Histological Subtyping**: Identifies cancer classification (e.g., Adenocarcinoma vs. Squamous Cell Carcinoma).
2. **Cancer Staging**: Predicts disease progression following the TNM staging system (Stage I, II, III, or IV).
3. **Survival Expectancy**: Estimates overall survival duration in days/years via regression.

---

## 📊 Dataset Explained Simply

The framework is built around the public **NSCLC-Radiomics** dataset (422 patient cases), combining two complementary data sources:

| Modality | Description | Format / Processing |
| :--- | :--- | :--- |
| 🖼️ **3D Volumetric CT Scans** | 3D thoracic CT imaging capturing lung anatomical structures and tumor volumes. | Normalised HU range ($-1000$ to $+400$), resampled to $1.0\text{mm}$ isotropic voxels ($128 \times 128 \times 128$ grid). |
| 📋 **Clinical Radiomics Data** | Patient demographics and diagnostic indicators. | `Age`, `Gender`, `T_Stage`, `N_Stage`, `M_Stage`, `Histology`, `Survival_Time`. |

---

## 🏗️ System Architecture in Simple Terms

The model operates like a collaborative team of specialized neural networks:

```
 ┌──────────────────────────┐             ┌──────────────────────────┐
 │  3D CT Scan (128x128x128)│             │ Clinical & Tabular Data  │
 └────────────┬─────────────┘             └────────────┬─────────────┘
              │                                        │
              ▼                                        ▼
 ┌──────────────────────────┐             ┌──────────────────────────┐
 │ 3D Image Encoder         │             │ Clinical MLP Encoder     │
 │ (3D ResNet / MedicalNet) │             │ (Dense Layers + Dropout) │
 └────────────┬─────────────┘             └────────────┬─────────────┘
              │ 512-dim Feature Vector                 │ 256-dim Feature Vector
              └───────────────────┬────────────────────┘
                                  ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │               Multi-Head Cross-Attention Fusion Block             │
 │    (Allows visual 3D features to query clinical context vectors)  │
 └────────────────────────────────┬──────────────────────────────────┘
                                  │ Fused 512-dim Vector
                                  ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │                    Multi-Task Prediction Heads                    │
 ├───────────────────┬───────────────────┬───────────────────────────┤
 │ 1. Histology Subtype│ 2. TNM Cancer Stage│ 3. Overall Survival Time  │
 └───────────────────┴───────────────────┴───────────────────────────┘
                                  │
                                  ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │                     Explainable AI (XAI) Layer                    │
 ├───────────────────────────────────┬───────────────────────────────┤
 │ 3D Grad-CAM (Visual CT Heatmaps) │ SHAP (Clinical Feature Ranks) │
 └───────────────────────────────────┴───────────────────────────────┘
                                  │
                                  ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │              Interactive Streamlit Web Dashboard UI               │
 └───────────────────────────────────────────────────────────────────┘
```

### Key Architectural Steps:
1. **3D Image Encoder**: Extracts 3D spatial feature maps from CT volumes into a 512-dimensional image vector.
2. **Clinical Encoder**: Passes clinical variables (Age, Gender, TNM) through a deep MLP to yield a 256-dimensional clinical embedding.
3. **Cross-Attention Fusion**: Uses multi-head cross-attention so image tokens dynamically interact with clinical context.
4. **Multi-Task Heads**: Predicts cancer subtype, stage, and survival time concurrently using dedicated output heads.
5. **Dual Explainability (XAI)**:
   - **3D Grad-CAM**: Generates 3D spatial heatmap overlays on CT slices highlighting tumor activation regions.
   - **SHAP Analysis**: Ranks clinical features based on their contribution to the diagnosis.

---

## 💡 Explainable AI (XAI) Features

- **Visual Attribution (3D Grad-CAM)**: Enables real-time slice-by-slice inspection of CT scans across Axial, Coronal, and Sagittal views with colored attention overlays.
- **Clinical Feature Attribution (SHAP)**: Provides quantitative bar graphs demonstrating how specific clinical parameters influenced the diagnostic output.

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/mridulupadhyay111/LungCancerAi.git
cd LungCancerAi

# Create virtual environment and install requirements
python -m venv .venv
.venv\Scripts\activate  # On Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Model Training
```bash
python train.py --config config/config.yaml
```

### 3. Model Evaluation
```bash
python evaluate.py --model checkpoints/best_model.pth
```

### 4. Interactive Web UI
```bash
python run_app.py
```
*Open `http://localhost:8501` in your browser to interact with the model, explore 3D CT volumes, and view XAI heatmaps.*

---

## 📂 Repository Structure

- `config/` - System configuration files (`config.yaml`, `model.yaml`, `dataset.yaml`, `train.yaml`).
- `datasets/` - MONAI preprocessors and PyTorch multimodal dataset loaders.
- `models/` - 3D ResNet encoder, MLP clinical encoder, Cross-Attention fusion, and multi-task prediction heads.
- `explainability/` - 3D Grad-CAM slice visualization and SHAP clinical feature engines.
- `deployment/` - Streamlit application UI and multi-planar slice viewer.
- `train.py` & `evaluate.py` - Pipelines for model training and evaluation metrics computation.

---

## 📄 License & Acknowledgments

- **License**: [MIT License](LICENSE)
- **Built With**: [PyTorch](https://pytorch.org/), [MONAI](https://monai.io/), [Streamlit](https://streamlit.io/)
- **Dataset**: Public NSCLC-Radiomics dataset hosted on [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/).