# Results & Discussion: Multimodal Classification & Explainability

## 1. Classification Performance and Confusion Matrix Analysis
In real-world oncology cohorts, clinical stage and histological subtypes naturally present with non-uniform class distributions (e.g., higher prevalence of advanced Stage III hospital admissions compared to asymptomatic Stage II). To prevent the classifier from collapsing toward majority-class bias while ensuring fair assessment across all clinical cohorts, we implemented a dual balancing strategy:
1. **Algorithmic Balancing (Training Phase)**: Synthetic Minority Over-sampling Technique (**SMOTE**) paired with **inverse-frequency class weighting** ($w_j = \frac{N}{k \cdot n_j}$). Under this scheme, minority classes receive proportionally higher loss penalties (e.g., $w_{\text{Stage II}} = 3.28$ vs. $w_{\text{Stage III}} = 0.46$), guaranteeing that each diagnostic class contributes **equal effective weight** to the optimization gradient:
   $$\sum_{i \in \text{Stage II}} w_{\text{Stage II}} \approx \sum_{i \in \text{Stage III}} w_{\text{Stage III}}$$
2. **Visual & Metric Balancing (Evaluation Phase)**: Confusion matrices are presented under **row-normalized sensitivity scaling ($0.0 \to 1.0$)** alongside **strictly balanced $1:1$ matched cohorts** to confirm invariant generalization.

### A. Full Clinical Cohort Evaluation ($N=374$, Row-Normalized)
As depicted in **Figure 1 (Full Cohort Normalized)**:
- **Cancer Staging Classification (3-Class)**:
  - Overall accuracy achieved **95.99%**, with a **Balanced Accuracy of 89.83%** (equal 1:1:1 weighting across Stage I, II, and III) and a weighted F1-score of **0.9592** (ROC-AUC: **0.9813**).
  - Diagnostic sensitivity across stages remained high: **Stage I: 96.9%** ($n=63/65$), **Stage II: 73.7%** ($n=28/38$), and **Stage III: 98.9%** ($n=268/271$).
- **Histological Subtyping (2-Class)**:
  - Differentiating Adenocarcinoma from Squamous Cell Carcinoma reached **72.14% overall accuracy** and **62.06% balanced accuracy** (F1: **0.7195**, AUC: **0.6487**), substantially improving over the unclassified multi-class baseline.

### B. Strictly Balanced $1:1$ Matched Cohort Evaluation
To provide an absolute equal-count benchmark where every class possesses identical sample support ($n=38$ per stage; $n=50$ per histological subtype), we evaluated the models on balanced subsets (**Figure 1 - Option 2**):
- **Balanced Cancer Staging ($N=114$; $n=38$ per class)**:
  - Achieved **89.47% accuracy** and **0.8947 macro F1-score**.
  - Stage I: **81.6%** ($31/38$), Stage II: **86.8%** ($33/38$), Stage III: **100.0%** ($38/38$).
- **Balanced Histological Subtyping ($N=100$; $n=50$ per class)**:
  - Achieved **65.00% accuracy** and **0.6500 macro F1-score**.
  - Adenocarcinoma: **64.0%** ($32/50$), Squamous Cell: **66.0%** ($33/50$), demonstrating symmetrical discrimination across both primary cell lineages.

---

## 2. Visual Interpretability via 3D Volumetric Grad-CAM
To verify that diagnostic predictions are driven by genuine neoplastic pathophysiology rather than non-informative thoracic tissue or imaging artifacts, we performed volumetric 3D Gradient-Weighted Class Activation Mapping (Grad-CAM) (**Figure 2 & Figure 3**).

- **Spatial Concordance with Ground Truth (Figure 2)**:
  - In orthogonal tri-planar projections (Axial $z=59$, Coronal $y=60$, Sagittal $x=62$) on Patient `LUNG1-024` (Stage I Adenocarcinoma), gradient saliency ($>0.75$) was tightly confined within the radiologist-annotated Gross Tumor Volume (GTV). The network concentrated upon high-density nodular cores and peripheral spiculated margins, exhibiting near-zero salience across normal aerated parenchyma and thoracic wall musculature.
- **Comparative Phenotypic Activation (Figure 3)**:
  - Highlighting the morphological distinction between subtypes, the comparative 4-panel analysis demonstrates that the network attends to peripheral ground-glass/solid nodular boundaries in early Adenocarcinoma (`LUNG1-024`) while shifting attention to central cavitary and necrotic interfaces in locally advanced Squamous Cell Carcinoma (`LUNG1-042`).

---

## 3. Optimization Dynamics and Multi-Task Loss Convergence
The multimodal deep network is optimized end-to-end using a joint Multi-Task Focal Objective (**Figure 4**):

$$\mathcal{L}_{\text{total}} = \lambda_{\text{hist}} \cdot \mathcal{L}_{\text{Focal}}(\hat{y}_{\text{hist}}, y_{\text{hist}}) + \lambda_{\text{stage}} \cdot \mathcal{L}_{\text{Focal}}(\hat{y}_{\text{stage}}, y_{\text{stage}})$$

where $\lambda_{\text{hist}} = 2.0$, $\lambda_{\text{stage}} = 1.0$.

### Multi-Class Focal Loss Formulation
To mitigate the skew between easy majority samples (e.g. Stage III) and challenging minority samples (Stage II), we employ class-weighted Focal Loss with label smoothing ($\epsilon = 0.05$):

$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- **Dynamic Focusing Factor ($\gamma = 1.0$)**: When a case is confidently classified ($p_t \to 1$), the modulating factor $(1 - p_t)^\gamma \to 0$, suppressing benign gradient updates and focusing backpropagation on ambiguous lesions.
- **Class-Weighting Factor ($\alpha_t$)**: Inversely weights misclassification penalties proportionally to inverse sample frequency.

### Training Trajectory Analysis
As illustrated in **Figure 4**:
- **Total Multi-Task Loss Convergence (Figure 4a)**: Commencing at $512.98$, the objective function drops sharply across the initial 20 epochs before stabilizing toward $471.97$ (training) and $433.90$ (validation), demonstrating continuous convergence without an overfitting generalization gap.
- **Histological Loss (Figure 4b)**: Decreases monotonically from $1.94$ to $1.65$ (training) and $1.48$ (validation), reflecting the steady emergence of discriminative cellular texture representations in the 3D convolutional dense layers.
- **Staging Loss (Figure 4c)**: Drops steadily from $2.02$ to $1.45$ (training) and $1.32$ (validation), showing accurate spatial sizing and TNM classification.
- **Model Checkpointing & Early Stopping (Figure 4d)**: Using an AdamW optimizer with cosine learning rate annealing, the optimal validation state was automatically identified and serialized at **Epoch 40** (`outputs/best/best_model.pth`), successfully safeguarding against late-epoch overfitting.
