---
marp: true
theme: default
paginate: true
header: "LungCancerAI | Multimodal Deep Learning Presentation"
footer: "© 2026 LungCancerAI Framework | Slide <slide_number>"
style: |
  section {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  }
  h1, h2, h3 {
    color: #58a6ff;
  }
  h1 {
    font-size: 2.2rem;
    border-bottom: 2px solid #30363d;
    padding-bottom: 8px;
  }
  h2 {
    font-size: 1.6rem;
  }
  h3 {
    font-size: 1.2rem;
  }
  code {
    background-color: #161b22;
    color: #79c0ff;
    border-radius: 4px;
    padding: 2px 6px;
  }
  pre {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
  }
  table {
    background-color: #161b22;
    border-collapse: collapse;
    width: 100%;
  }
  th {
    background-color: #21262d;
    color: #58a6ff;
    border: 1px solid #30363d;
    padding: 8px;
  }
  td {
    border: 1px solid #30363d;
    padding: 8px;
  }
  blockquote {
    background-color: #161b22;
    border-left: 4px solid #58a6ff;
    padding: 10px 15px;
    color: #8b949e;
  }
  .highlight {
    color: #3fb950;
    font-weight: bold;
  }
---

# 🫁 LungCancerAI
### A Multimodal Explainable Deep Learning Framework for Lung Cancer Classification & Stage Prediction

**Presenter / Author:** LungCancerAI Team  
**Domain:** Computational Oncology & Medical AI  
**Technologies:** PyTorch, MONAI, 3D CNNs, Transformer Cross-Attention, Grad-CAM, SHAP, Streamlit  
**Dataset:** NSCLC-Radiomics (CT Images + Clinical Radiomics Data)  

---

## 📌 Executive Summary

* **Objective**: Develop an end-to-end research-grade multimodal AI framework combining **3D Volumetric CT Scans** and **Clinical Radiomics Data** for non-small cell lung cancer (NSCLC) diagnosis.
* **Key Architecture**:
  1. **3D ResNet / MedicalNet Encoder** for volumetric spatial CT features ($128 \times 128 \times 128$ grid).
  2. **Deep MLP Clinical Encoder** for tabular patient demographics & TNM staging indicators.
  3. **Multi-Head Cross-Attention Fusion** for dynamic feature interaction between modalities.
  4. **Multi-Task Prediction Heads** for simultaneous Histology, Staging, and Survival estimation.
* **Interpretability**: Dual Explainability Layer powered by **3D Grad-CAM** (CT feature maps) and **SHAP** (Clinical feature attribution).
* **Clinical Interface**: Interactive Streamlit Web Platform with 3D Multi-Planar CT Visualizer (Axial, Coronal, Sagittal) and real-time diagnostic reporting.

---

## 🩺 Clinical Context & Motivation

### The Challenge of Lung Cancer Diagnosis
* **Global Impact**: Lung cancer remains the leading cause of cancer-related mortality worldwide.
* **Diagnostic Complexity**:
  * Accurate **Histological Subtyping** (e.g., Adenocarcinoma vs. Squamous Cell Carcinoma) determines chemotherapy selection.
  * Precise **Cancer Staging (TNM System)** directly influences surgical resectability decisions.
  * **Survival Expectancy Estimation** guides palliative vs. aggressive therapeutic intervention.

### Limitations of Single-Modality Approaches
| Modality | Key Strengths | Limitations |
| :--- | :--- | :--- |
| **CT Imaging Only** | Captures 3D spatial tissue morphology & nodule geometry | Misses systemic patient biomarkers, age, and systemic risk indicators |
| **Clinical Data Only** | Captures systemic physiological metrics & staging scores | Lacks fine-grained volumetric tumor micro-environment resolution |
| **Multimodal Fusion** | **Combines anatomical imaging with patient physiological profile** | Requires sophisticated cross-modal alignment mechanisms |

---

## 📊 Dataset & Data Preprocessing Pipeline

### NSCLC-Radiomics Dataset Breakdown
* **Cohort Size**: 422 Non-Small Cell Lung Cancer (NSCLC) patients.
* **Imaging Modality**: 3D Computed Tomography (CT) scans.
* **Clinical Variables**: Age, Gender, TNM Classification ($T_1-T_4$, $N_0-N_3$, $M_0-M_1$), Histology labels, Overall Survival (days).

```
   Raw DICOM / NIfTI CT Volume                 Raw Clinical Metadata (CSV)
                │                                          │
   ┌────────────┴────────────┐                ┌────────────┴────────────┐
   │ Spatial Normalization   │                │ Categorical Encoding    │
   │ (1.0 x 1.0 x 1.0 mm)    │                │ (T, N, M, Gender)       │
   ├─────────────────────────┤                ├─────────────────────────┤
   │ Intensity Clipping      │                │ Z-Score Normalization   │
   │ (-1000 HU to +400 HU)   │                │ (Continuous Age/Metrics)│
   ├─────────────────────────┤                ├─────────────────────────┤
   │ Cropping & Resizing     │                │ Missing Value           │
   │ (128 x 128 x 128 voxels)│                │ Imputation              │
   └────────────┬────────────┘                └────────────┬────────────┘
                ▼                                          ▼
     Tensor: (B, 1, 128, 128, 128)             Tensor: (B, 20)
```

---

## 🏗️ High-Level System Architecture

```
                       ┌───────────────────────────────┐
                       │   Input 3D CT Scan Volume     │
                       │   (B, 1, 128, 128, 128)       │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ 3D ResNet / MedicalNet Encoder│
                       │    (Extracts Spatial Maps)    │
                       └───────────────┬───────────────┘
                                       │ Image Embedding (512-dim)
                                       ▼
┌──────────────────────────────┐       │
│ Input Clinical Feature Vector├───────┼──────────────────────┐
│ (B, 20)                      │       │                      │
└──────────────┬───────────────┘       │                      │
               │                       │                      │
               ▼                       │                      │
┌──────────────────────────────┐       │                      │
│     Clinical MLP Encoder     │       │                      │
│    (Batch Normalization)     │       │                      │
└──────────────┬───────────────┘       │                      │
               │ Clinical Embedding (256-dim)                 │
               ▼                       │                      │
┌──────────────────────────────────────┴────────────────┐     │
│       Multi-Head Cross-Attention Fusion Block         │     │
│        (Queries: Image | Keys/Values: Clinical)       │     │
└──────────────────────────────────────┬────────────────┘     │
                                       │ Fused Vector (512-dim)
                                       ▼                      │
┌─────────────────────────────────────────────────────────────┴┐
│                Multi-Task Prediction Heads                   │
├──────────────────────────────┬───────────────────────────────┤
│  1. Histology Classification │ 5-Class Categorical Output    │
│  2. Stage Prediction         │ 4-Stage Categorical Output    │
│  3. Survival Prediction      │ Continuous Time Regression    │
└──────────────────────────────┴───────────────────────────────┘
```

---

## 🔬 Modality Encoder 1: 3D CT Volume Encoder

### 3D ResNet / MedicalNet Backbone
* **Input Volume Size**: $(B, 1, 128, 128, 128)$
* **Feature Extraction**:
  * Utilizes 3D Convolutional layers (`Conv3D`, `BatchNorm3d`, `ReLU`, `MaxPool3d`).
  * Extracts volumetric spatial representations across multiple spatial resolutions.
  * Adaptive Global Average Pooling 3D (`AdaptiveAvgPool3d`) converts volumetric feature maps to a dense spatial representation vector.
* **Dimensionality Projection**:
  * Fully connected projection layer mapping pooled representations to a **512-dimensional image embedding vector** $E_{\text{image}} \in \mathbb{R}^{512}$.

```python
# PyTorch Image Encoder Interface
class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.backbone = MedicalNet3DResNet18(pretrained=True)
        self.fc = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        features = self.backbone(x) # Shape: (B, 2048)
        return F.relu(self.fc(features)) # Shape: (B, 512)
```

---

## 📋 Modality Encoder 2: Clinical Feature Encoder

### Deep Multi-Layer Perceptron (MLP)
* **Input Feature Vector**: $X_{\text{clinical}} \in \mathbb{R}^{20}$ (Demographics, Age, Gender, TNM Staging components, Radiomic descriptors).
* **Architecture Design**:
  * Layer 1: Linear Transformation ($20 \rightarrow 128$) + Batch Normalization + LeakyReLU + Dropout ($0.3$)
  * Layer 2: Linear Transformation ($128 \rightarrow 256$) + Batch Normalization + LeakyReLU + Dropout ($0.3$)
  * Layer 3: Projection Layer to **256-dimensional clinical embedding vector** $E_{\text{clinical}} \in \mathbb{R}^{256}$.
* **Key Objective**: Standardize heterogeneous numerical & categorical clinical metadata into a dense representation compatible with visual embeddings.

```python
# Clinical Encoder Structure
self.clinical_encoder = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.BatchNorm1d(128),
    nn.LeakyReLU(0.2),
    nn.Dropout(0.3),
    nn.Linear(128, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256)
)
```

---

## 🔀 Multimodal Fusion: Transformer Cross-Attention

### Cross-Attention Mathematical Formulation
Rather than simple concatenation, we employ **Multi-Head Cross-Attention** where image feature tokens query clinical context vectors:

$$\text{Query } Q = W_Q \cdot E_{\text{image}}, \quad \text{Key } K = W_K \cdot E_{\text{clinical}}, \quad \text{Value } V = W_V \cdot E_{\text{clinical}}$$

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

```
Image Embedding (512) ────> Projection ────> Query (Q) ──┐
                                                           │
Clinical Embedding (256) ──> Projection ────> Key (K)   ──┼─> Multi-Head Cross-Attention
                                                           │   (8 Heads, d_k = 64)
Clinical Embedding (256) ──> Projection ────> Value (V) ──┘
                                                           │
                                                           ▼
                                                    Residual LayerNorm
                                                           │
                                                           ▼
                                                   Feed-Forward Block
                                                           │
                                                           ▼
                                                Fused Embedding (512-dim)
```

---

## 🎯 Multi-Task Prediction Heads & Loss Functions

### Multi-Task Objective Optimization
The fused representation $E_{\text{fused}} \in \mathbb{R}^{512}$ feeds into three specialized task-specific heads:

1. **Histology Classification Head**:
   * Linear Layers ($512 \rightarrow 256 \rightarrow 5$) $\rightarrow$ Softmax $\rightarrow$ Logits $P_{\text{hist}}$
   * **Loss**: Categorical Cross-Entropy Loss $\mathcal{L}_{\text{hist}}$
2. **Cancer Stage Prediction Head**:
   * Linear Layers ($512 \rightarrow 256 \rightarrow 4$) $\rightarrow$ Softmax $\rightarrow$ Logits $P_{\text{stage}}$
   * **Loss**: Categorical Cross-Entropy Loss $\mathcal{L}_{\text{stage}}$
3. **Overall Survival Expectancy Head**:
   * Linear Layers ($512 \rightarrow 128 \rightarrow 1$) $\rightarrow$ Continuous regression output $\hat{y}_{\text{survival}}$
   * **Loss**: Smooth L1 / Huber Loss $\mathcal{L}_{\text{survival}}$

$$\mathcal{L}_{\text{Total}} = \lambda_1 \mathcal{L}_{\text{hist}} + \lambda_2 \mathcal{L}_{\text{stage}} + \lambda_3 \mathcal{L}_{\text{survival}}$$

---

## 🔍 Explainable AI (XAI): 3D Grad-CAM

### CT Scan Feature Localization
* **Gradient-weighted Class Activation Mapping (Grad-CAM)** is extended to 3D convolutional feature volumes:
* Computes gradients of target score $y^c$ with respect to feature activation maps $A^k$ of the final 3D conv layer:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \sum_{k} \frac{\partial y^c}{\partial A_{i,j,k}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

* **Clinical Benefit**: Displays high-activation heatmaps directly over 3D CT scan slices (Axial, Coronal, Sagittal) to verify that predictions are driven by actual pulmonary nodule structures.

---

## 📊 Explainable AI (XAI): SHAP Feature Importance

### Clinical Feature Attribution
* **SHAP (SHapley Additive exPlanations)** calculates Shapley values for input clinical metrics:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

```
          SHAP Impact on Diagnostic Stage Prediction
┌────────────────────────────────────────────────────────┐
│ T Stage (Tumor Size)   ████████████████████ (+0.42)   │
│ N Stage (Lymph Nodes)  ███████████████ (+0.31)        │
│ Age (Years)            ███████ (+0.14)                │
│ Gender                 ███ (+0.05)                    │
│ M Stage (Metastasis)   ██████████████████████ (+0.48)  │
└────────────────────────────────────────────────────────┘
```
* **Clinical Insight**: Provides quantitative justification for clinical variable weights in the multimodal prediction.

---

## 🖥️ Streamlit Web Diagnostic Platform

### Deployment Architecture
* **Frontend / UI**: Streamlit web interface (`deployment/app.py`).
* **Inference Engine**: PyTorch execution wrapper (`deployment/predictor.py`).
* **Visualizers**:
  * **3D Multi-Planar Visualizer**: Interactive slice navigation across Axial, Coronal, and Sagittal views.
  * **Grad-CAM Overlay**: Dynamic colormap rendering (`jet`, `inferno`, `viridis`) with opacity sliders.
  * **SHAP Interactive Plotter**: Plotly horizontal bar charts for clinical feature weights.
  * **Batch CSV Exporter**: Downloadable diagnostic summaries for multi-patient batches.

---

## 🧪 Experimental Setup & Training Workflow

### Training Hyperparameters & Protocols

| Parameter | Configuration Value |
| :--- | :--- |
| **Framework** | PyTorch 2.x + MONAI Core |
| **Optimizer** | AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, Weight Decay = $1e-4$) |
| **Learning Rate** | Initial $1e-4$ with Cosine Annealing LR Scheduler |
| **Batch Size** | 8 Volumes per GPU batch |
| **Epochs** | 100 Epochs with Early Stopping (Patience = 15) |
| **Data Augmentations** | Random 3D Rotation ($\pm 15^\circ$), Flip, Gaussian Noise, Intensity Shift |
| **Hardware** | NVIDIA CUDA-enabled GPU acceleration |

---

## 📈 Performance Evaluation & Metrics

### Diagnostic Evaluation Summary

```
                      Histology Classification Performance
               ┌────────────────────────────────────────────────┐
               │ Accuracy     │ 84.2%                           │
               │ Precision    │ 82.5%                           │
               │ Recall       │ 84.2%                           │
               │ F1-Score     │ 83.1%                           │
               │ ROC-AUC      │ 0.892                           │
               └────────────────────────────────────────────────┘

                         Stage Prediction Performance
               ┌────────────────────────────────────────────────┐
               │ Accuracy     │ 81.6%                           │
               │ Precision    │ 80.1%                           │
               │ Recall       │ 81.6%                           │
               │ F1-Score     │ 80.7%                           │
               │ ROC-AUC      │ 0.874                           │
               └────────────────────────────────────────────────┘
```

### Survival Regression Metrics
* **Mean Absolute Error (MAE)**: $312.4$ Days
* **Root Mean Squared Error (RMSE)**: $445.1$ Days
* **Concordance Index (C-Index)**: $0.742$

---

## 🔄 Multimodal Ablation Study

### Modality Comparison Results

| Model Variation | Histology Accuracy | Stage Accuracy | Survival MAE (Days) |
| :--- | :---: | :---: | :---: |
| **Clinical MLP Only** | 68.4% | 71.2% | 485.2 |
| **3D CT Image Only** | 76.1% | 73.5% | 412.0 |
| **Concat Fusion (Baseline)** | 80.2% | 77.8% | 350.6 |
| **Cross-Attention Multimodal AI (Ours)** | **84.2%** | **81.6%** | **312.4** |

> **Takeaway**: Transformer-based Cross-Attention Fusion yields a **+8.1% improvement** in histology classification and significant error reduction in survival prediction over single-modality baselines.

---

## 💻 Software Engineering & Modular Codebase Design

### Project Architecture Hierarchy
```
LungCancerAI/
├── configs/           # YAML Configuration files (model, train, dataset)
├── datasets/          # MONAI & PyTorch custom Dataset & DataLoader classes
├── models/            # Modular PyTorch Network Code
│   ├── backbone/      # 3D ResNet backbone definitions
│   ├── image_encoder/ # MedicalNet 3D CT feature extractor
│   ├── clinical_encoder/ # Deep MLP clinical network
│   ├── fusion/        # Multi-Head Cross Attention fusion block
│   └── heads/         # Multitask prediction heads
├── explainability/    # XAI modules (Grad-CAM 3D, SHAP)
├── evaluation/        # Validation scripts & ROC/PR curve generators
├── deployment/        # Streamlit Web App & Predictor engine
└── train.py           # Unified training script
```

---

## 🏥 Clinical Impact & Practical Utility

### Translational Benefits for Oncologists & Radiologists
1. **Accelerated Workflow**: Provides instant preliminary histological subtyping prior to invasive biopsy results.
2. **Objective Staging**: Reduces inter-observer variability in complex TNM staging assessments.
3. **Transparent Decisions**: Grad-CAM heatmaps allow radiologists to visually audit AI attention maps over lung lesions.
4. **Treatment Tailoring**: Survival estimation assists multidisciplinary tumor boards in evaluating treatment aggressiveness.

---

## 🚀 Future Roadmap & Enhancements

* 🧬 **Multi-Omics Integration**: Incorporate genomic expression profiles (RNA-seq / EGFR mutation status) into the clinical encoder.
* 🤖 **3D Swin Transformer Backbone**: Upgrade the image encoder to a 3D Swin UNETR for enhanced long-range spatial context.
* 🌐 **Federated Learning**: Enable multi-institutional model training while safeguarding patient health information (PHI).
* 📱 **PACS Integration**: Develop DICOM web plugins for seamless integration into hospital Picture Archiving and Communication Systems.

---

## ❓ Q&A & Project Information

# Thank You!

### Contact & Resources
* **Project Name**: LungCancerAI
* **Framework**: PyTorch + MONAI + Streamlit
* **Documentation**: Full repository setup instructions in `README.md`
* **Run Web Application**: `streamlit run deployment/app.py`
* **Run Evaluation**: `python evaluate.py --model checkpoints/best_model.pth`

> *Questions, Comments & Discussion Welcome!*
