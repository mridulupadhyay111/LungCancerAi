"""
deployment/app.py

LungCancerAI - Modern Research-Grade Medical Diagnostic Web Application.
Features Upload CT Scan (NIfTI, NumPy, PNG/JPG), Custom Clinical Data Entry & CSV Upload,
Multimodal AI Prediction from trained PyTorch model, 3D CT Multi-Planar Visualizer,
Grad-CAM Heatmap Explainability, SHAP Clinical Feature Importance, Benchmark Analytics,
and Batch CSV Exporter.
"""

from __future__ import annotations

import json
from pathlib import Path

# Force headless Matplotlib backend to prevent GUI thread blocking on Windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import streamlit as st
from deployment.predictor import LungCancerPredictor

# ==============================================================================
# Page Configuration & Custom CSS
# ==============================================================================

st.set_page_config(
    page_title="LungCancerAI Multimodal Diagnostic Platform",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Medical Theme)
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
        padding: 24px;
        border-radius: 14px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .metric-card {
        background-color: #1E232A;
        border: 1px solid #2D343F;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    .prediction-box {
        background: #181E24;
        border-left: 5px solid #00D2FF;
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E232A;
        border-radius: 8px;
        padding: 10px 20px;
        color: #B0B8C4;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# Model Initialization
# ==============================================================================

@st.cache_resource
def get_predictor():
    try:
        return LungCancerPredictor()
    except Exception as e:
        st.error(f"Failed to load prediction engine: {e}")
        return None

predictor = get_predictor()

# Initialize session state for uploaded / selected data
if "uploaded_vol_3d" not in st.session_state:
    st.session_state.uploaded_vol_3d = None
if "clinical_vec" not in st.session_state:
    st.session_state.clinical_vec = None
if "current_pred" not in st.session_state:
    st.session_state.current_pred = None
if "last_input_key" not in st.session_state:
    st.session_state.last_input_key = None

# Auto-initialize default sample patient data on first load if available
if predictor and len(predictor.patient_ids) > 0 and st.session_state.uploaded_vol_3d is None:
    try:
        init_pid = predictor.patient_ids[0]
        sample = predictor.get_patient_data(init_pid)
        st.session_state.uploaded_vol_3d = np.squeeze(sample["image"].numpy())
        st.session_state.clinical_vec = sample["clinical"].numpy()
        st.session_state.current_pred = predictor.predict(init_pid)
        st.session_state.last_input_key = f"sample_{init_pid}"
    except Exception:
        pass

# ==============================================================================
# Header & Sidebar Controls
# ==============================================================================

st.markdown(
    """
    <div class="main-header">
        <h1 style="margin:0; font-size: 30px;">🫁 LungCancerAI Diagnostic & Explainability Platform</h1>
        <p style="margin:4px 0 0 0; color: #A0B2C6; font-size: 14px;">
            Research-Grade Multimodal Deep Learning for CT Scan & Clinical Diagnostic Prediction with Grad-CAM & SHAP Interpretability
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("⚙️ Control Panel")
    
    if predictor:
        st.success(f"System Ready ({predictor.device.type.upper()})")
        st.caption(f"Loaded Model: `best_model.pth`")
        st.caption(f"Dataset Size: {len(predictor.patient_ids)} Records")
    else:
        st.error("Model Engine Offline")

    st.markdown("---")
    st.subheader("Data Input Mode")
    
    input_mode = st.radio(
        "Select Input Source",
        [
            "👤 Select Sample Patient Record",
            "📁 Upload CT Scan Only",
            "📋 Input Clinical Data Only",
            "🧬 Upload Both (Multimodal CT + Clinical)"
        ],
        index=0 if (predictor and len(predictor.patient_ids) > 0) else 1,
    )

# ==============================================================================
# Data Selection / Upload Processing Logic
# ==============================================================================

vol_3d = st.session_state.uploaded_vol_3d
clinical_vec = st.session_state.clinical_vec
patient_identifier = "Custom_Patient"

if "Upload CT Scan" in input_mode:
    st.sidebar.markdown("### Upload CT Volume / Slice")
    uploaded_ct_file = st.sidebar.file_uploader(
        "Choose CT Scan File (.nii, .nii.gz, .npy, .png, .jpg)",
        type=["nii", "gz", "npy", "npz", "png", "jpg", "jpeg"],
        key="ct_only_uploader"
    )
    if uploaded_ct_file is not None and predictor:
        file_id = f"{uploaded_ct_file.name}_{uploaded_ct_file.size}"
        input_key = f"upload_ct_{file_id}"
        if st.session_state.last_input_key != input_key:
            try:
                vol_3d = predictor.preprocess_ct_file(uploaded_ct_file)
                st.session_state.uploaded_vol_3d = vol_3d
                if st.session_state.clinical_vec is None:
                    st.session_state.clinical_vec = predictor.format_clinical_vector()
                st.session_state.last_input_key = input_key
                st.session_state.current_pred = predictor.predict_raw(
                    vol_3d, st.session_state.clinical_vec, patient_id=uploaded_ct_file.name
                )
                st.sidebar.success(f"Loaded CT: {uploaded_ct_file.name}")
            except Exception as e:
                st.sidebar.error(f"Error reading CT file: {e}")
    if predictor and clinical_vec is None:
        clinical_vec = predictor.format_clinical_vector(age=62, gender=1, t_stage=2, n_stage=1, m_stage=0)

elif "Input Clinical Data Only" in input_mode:
    st.sidebar.markdown("### Clinical Feature Entry")
    c_age = st.sidebar.number_input("Age (Years)", 18, 100, 65, key="clin_age")
    c_gender = st.sidebar.selectbox("Gender", ["Male", "Female"], index=0, key="clin_gender")
    c_t = st.sidebar.slider("T Stage (Tumor Size)", 1, 4, 2, key="clin_t")
    c_n = st.sidebar.slider("N Stage (Node Spread)", 0, 3, 1, key="clin_n")
    c_m = st.sidebar.selectbox("M Stage (Metastasis)", [0, 1], index=0, key="clin_m")
    
    gender_val = 1 if c_gender == "Male" else 0
    if predictor:
        clinical_vec = predictor.format_clinical_vector(age=c_age, gender=gender_val, t_stage=c_t, n_stage=c_n, m_stage=c_m)
        input_key = f"clin_{c_age}_{c_gender}_{c_t}_{c_n}_{c_m}"
        if st.session_state.last_input_key != input_key:
            st.session_state.last_input_key = input_key
            st.session_state.clinical_vec = clinical_vec
            if st.session_state.uploaded_vol_3d is None:
                st.session_state.uploaded_vol_3d = np.zeros((128, 128, 128), dtype=np.float32)
            st.session_state.current_pred = predictor.predict_raw(
                st.session_state.uploaded_vol_3d, clinical_vec, patient_id="Custom_Clinical"
            )

elif "Upload Both" in input_mode:
    st.sidebar.markdown("### 1. Upload CT Scan")
    uploaded_ct_file = st.sidebar.file_uploader(
        "Choose CT File (.nii, .npy, .png)",
        type=["nii", "gz", "npy", "npz", "png", "jpg", "jpeg"],
        key="multi_ct"
    )
    st.sidebar.markdown("### 2. Enter Clinical Features")
    c_age = st.sidebar.number_input("Age", 18, 100, 62, key="multi_age")
    c_gender = st.sidebar.selectbox("Gender", ["Male", "Female"], key="multi_gender")
    c_t = st.sidebar.slider("T Stage", 1, 4, 2, key="multi_t")
    c_n = st.sidebar.slider("N Stage", 0, 3, 1, key="multi_n")
    c_m = st.sidebar.selectbox("M Stage", [0, 1], key="multi_m")
    
    gender_val = 1 if c_gender == "Male" else 0
    if predictor:
        clinical_vec = predictor.format_clinical_vector(age=c_age, gender=gender_val, t_stage=c_t, n_stage=c_n, m_stage=c_m)
        file_id = f"{uploaded_ct_file.name}_{uploaded_ct_file.size}" if uploaded_ct_file is not None else "nofile"
        input_key = f"multi_{file_id}_{c_age}_{c_gender}_{c_t}_{c_n}_{c_m}"
        if st.session_state.last_input_key != input_key:
            st.session_state.last_input_key = input_key
            if uploaded_ct_file is not None:
                try:
                    vol_3d = predictor.preprocess_ct_file(uploaded_ct_file)
                    st.session_state.uploaded_vol_3d = vol_3d
                    st.sidebar.success(f"CT Loaded: {uploaded_ct_file.name}")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
            if st.session_state.uploaded_vol_3d is None:
                st.session_state.uploaded_vol_3d = np.zeros((128, 128, 128), dtype=np.float32)
            st.session_state.clinical_vec = clinical_vec
            st.session_state.current_pred = predictor.predict_raw(
                st.session_state.uploaded_vol_3d, clinical_vec, patient_id="Multimodal_Patient"
            )

else:  # Select Sample Patient Record
    if predictor and len(predictor.patient_ids) > 0:
        selected_patient = st.sidebar.selectbox("Select Patient ID", predictor.patient_ids, index=0, key="sample_select")
        patient_identifier = selected_patient
        input_key = f"sample_{selected_patient}"
        if st.session_state.last_input_key != input_key:
            st.session_state.last_input_key = input_key
            sample = predictor.get_patient_data(selected_patient)
            vol_3d = np.squeeze(sample["image"].numpy())
            clinical_vec = sample["clinical"].numpy()
            st.session_state.uploaded_vol_3d = vol_3d
            st.session_state.clinical_vec = clinical_vec
            st.session_state.current_pred = predictor.predict(selected_patient)

# Update active volume and clinical vector in session state
if vol_3d is not None:
    st.session_state.uploaded_vol_3d = vol_3d
if clinical_vec is not None:
    st.session_state.clinical_vec = clinical_vec

# ==============================================================================
# Content Tabs
# ==============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Patient Diagnosis & 3D CT Visualizer",
    "🔍 Explainability (Grad-CAM & SHAP)",
    "📊 Benchmark Analytics",
    "⚡ Batch CSV Exporter"
])

if predictor:

    # --------------------------------------------------------------------------
    # TAB 1: Patient Diagnosis & 3D CT Visualizer
    # --------------------------------------------------------------------------
    with tab1:
        col_left, col_right = st.columns([1.2, 1.0])
        
        with col_left:
            st.subheader("🖼️ 3D CT Multi-Planar Visualizer")
            
            if st.session_state.uploaded_vol_3d is not None and st.session_state.uploaded_vol_3d.max() > 0:
                c1, c2, c3 = st.columns(3)
                with c1:
                    ax_val = st.slider("Axial Slice", 0.0, 1.0, 0.5, 0.02, key="slice_ax")
                with c2:
                    co_val = st.slider("Coronal Slice", 0.0, 1.0, 0.5, 0.02, key="slice_co")
                with c3:
                    sa_val = st.slider("Sagittal Slice", 0.0, 1.0, 0.5, 0.02, key="slice_sa")
                    
                slice_data = predictor.get_slices_from_volume(
                    st.session_state.uploaded_vol_3d,
                    axial_pct=ax_val,
                    coronal_pct=co_val,
                    sagittal_pct=sa_val
                )
                
                # Render 3 Cross-Sections
                fig_ct, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#0E1117")
                titles = ["Axial View", "Coronal View", "Sagittal View"]
                keys = ["axial", "coronal", "sagittal"]
                
                for ax, title, key in zip(axes, titles, keys):
                    img, _ = slice_data[key]
                    ax.imshow(img, cmap="gray")
                    ax.set_title(title, color="white", fontsize=11)
                    ax.axis("off")
                    
                plt.tight_layout()
                st.pyplot(fig_ct)
                plt.close(fig_ct)
            else:
                st.info("ℹ️ No CT scan uploaded or synthetic zero volume active. Upload a CT scan file (.nii, .npy, .png) to view multi-planar slices.")

        with col_right:
            st.subheader("🚀 Model Diagnostic Prediction")
            
            if st.button("RUN MULTIMODAL AI PREDICTION", type="primary", use_container_width=True, key="run_pred_btn"):
                if st.session_state.uploaded_vol_3d is None:
                    st.session_state.uploaded_vol_3d = np.zeros((128, 128, 128), dtype=np.float32)
                if st.session_state.clinical_vec is None:
                    st.session_state.clinical_vec = predictor.format_clinical_vector()

                with st.spinner("Analyzing CT features & clinical indicators..."):
                    pred = predictor.predict_raw(
                        st.session_state.uploaded_vol_3d,
                        st.session_state.clinical_vec,
                        patient_id=patient_identifier
                    )
                    st.session_state.current_pred = pred

            if st.session_state.current_pred is not None:
                pred = st.session_state.current_pred
                
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Predicted Histology", pred["histology_label"])
                with m2:
                    st.metric("Predicted Stage", pred["stage_label"])
                with m3:
                    st.metric("Survival Expectancy", f"{pred['survival_years']} Yrs ({pred['survival_days']} days)")
                st.markdown('</div>', unsafe_allow_html=True)

                # Histology Confidence Plot
                st.write("#### 🧬 Histology Subtype Probabilities")
                labels_h = pred.get("histology_classes", ["Unknown", "Adeno", "Large Cell", "NOS", "Squamous"])[:len(pred['histology_probs'])]
                if HAS_PLOTLY:
                    fig_h = px.bar(
                        x=labels_h,
                        y=pred['histology_probs'],
                        labels={'x': 'Histology Subtype', 'y': 'Probability'},
                        color=pred['histology_probs'],
                        color_continuous_scale="Viridis",
                        text=[f"{p*100:.1f}%" for p in pred['histology_probs']]
                    )
                    fig_h.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
                    st.plotly_chart(fig_h, use_container_width=True)
                else:
                    fig_h, ax_h = plt.subplots(figsize=(6, 2.5), facecolor="#0E1117")
                    ax_h.set_facecolor("#181E24")
                    ax_h.bar(labels_h, pred['histology_probs'], color="#00D2FF")
                    ax_h.tick_params(colors="white")
                    ax_h.set_ylabel("Probability", color="white")
                    st.pyplot(fig_h)
                    plt.close(fig_h)

    # --------------------------------------------------------------------------
    # TAB 2: Explainability Center (Grad-CAM & SHAP)
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("🔍 AI Interpretability & Explainability Center")
        st.write("Understand why the model made its diagnostic predictions through visual feature attribution.")
        
        ex_col1, ex_col2 = st.columns(2)
        
        with ex_col1:
            st.markdown("### 🎯 Grad-CAM 3D/2D Heatmap")
            st.write("Visualizes high-activation CT scan regions driving the neural network decision.")
            
            task_choice = st.selectbox("Explain Target Task", ["histology", "stage"], key="grad_task")
            plane_choice = st.selectbox("View Plane", ["axial", "coronal", "sagittal"], key="grad_plane")
            slice_pct = st.slider("Slice Position", 0.0, 1.0, 0.5, 0.02, key="grad_slice")
            cmap_choice = st.selectbox("Colormap", ["jet", "inferno", "viridis", "magma"], index=0, key="grad_cmap")
            alpha_val = st.slider("Heatmap Opacity", 0.1, 1.0, 0.55, key="grad_alpha")

            if st.session_state.uploaded_vol_3d is not None and st.session_state.uploaded_vol_3d.max() > 0:
                ct_slice, heatmap = predictor.generate_gradcam_heatmap(
                    st.session_state.uploaded_vol_3d,
                    clinical_np=st.session_state.clinical_vec,
                    task=task_choice,
                    plane=plane_choice,
                    slice_pct=slice_pct
                )
                
                fig_g, ax_g = plt.subplots(figsize=(6, 5), facecolor="#0E1117")
                ax_g.imshow(ct_slice, cmap="gray")
                ax_g.imshow(heatmap, cmap=cmap_choice, alpha=alpha_val)
                ax_g.set_title(f"Grad-CAM Attention ({task_choice.capitalize()} - {plane_choice.capitalize()})", color="white")
                ax_g.axis("off")
                st.pyplot(fig_g)
                plt.close(fig_g)
            else:
                st.warning("Upload a 3D CT scan to compute PyTorch Grad-CAM heatmaps.")

        with ex_col2:
            st.markdown("### 📊 SHAP Clinical Feature Attribution")
            st.write("Measures impact of clinical parameters (Age, Stage, Gender) on patient outcome.")
            
            if st.session_state.clinical_vec is not None:
                shap_data = predictor.get_shap_feature_importance(st.session_state.clinical_vec)
                df_shap = pd.DataFrame(shap_data, columns=["Feature", "Value", "SHAP Contribution"])
                
                if HAS_PLOTLY:
                    fig_shap = px.bar(
                        df_shap,
                        x="SHAP Contribution",
                        y="Feature",
                        orientation="h",
                        color="SHAP Contribution",
                        color_continuous_scale="RdBu_r",
                        title="Clinical Feature Contribution (SHAP Value)"
                    )
                    fig_shap.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
                    st.plotly_chart(fig_shap, use_container_width=True)
                else:
                    fig_s, ax_s = plt.subplots(figsize=(6, 3.5), facecolor="#0E1117")
                    ax_s.set_facecolor("#181E24")
                    ax_s.barh(df_shap["Feature"], df_shap["SHAP Contribution"], color="#00D2FF")
                    ax_s.tick_params(colors="white")
                    ax_s.set_xlabel("SHAP Contribution", color="white")
                    st.pyplot(fig_s)
                    plt.close(fig_s)
                
                st.dataframe(df_shap, use_container_width=True)
                st.info("💡 Positive SHAP values increase predicted risk / advanced stage; negative values indicate lower risk.")

    # --------------------------------------------------------------------------
    # TAB 3: Benchmark Analytics
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("📈 Model Performance & Validation Benchmarks")
        eval_path = Path("evaluation") / "evaluation_report.json"
        
        if eval_path.exists():
            with open(eval_path, "r") as f:
                eval_data = json.load(f)
                
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("Histology Accuracy", f"{eval_data.get('metrics', {}).get('histology', {}).get('accuracy', 0.3422)*100:.1f}%")
            with b2:
                st.metric("Stage Accuracy", f"{eval_data.get('metrics', {}).get('stage', {}).get('accuracy', 0.4169)*100:.1f}%")
            with b3:
                st.metric("Survival MAE", f"{eval_data.get('metrics', {}).get('survival', {}).get('mae', 944.35):.1f} Days")
            with b4:
                st.metric("Evaluated Patients", f"{eval_data.get('num_samples', 415)}")
                
            st.json(eval_data)
        else:
            st.info("Evaluation metrics report available at `evaluation/evaluation_report.json`.")

    # --------------------------------------------------------------------------
    # TAB 4: Batch Export
    # --------------------------------------------------------------------------
    with tab4:
        st.subheader("⚡ Batch Diagnostic Predictions & Report Exporter")
        st.write("Run predictions across dataset records and export structured CSV reports.")
        
        if len(predictor.patient_ids) > 0:
            num_patients = st.slider("Select Batch Size", 5, len(predictor.patient_ids), min(20, len(predictor.patient_ids)), key="batch_slider")
            
            if st.button("⚡ Run Batch Predictions", type="primary", key="batch_btn"):
                results = []
                progress_bar = st.progress(0)
                
                for idx in range(num_patients):
                    pid = predictor.patient_ids[idx]
                    res = predictor.predict(pid)
                    results.append({
                        "PatientID": res["patient_id"],
                        "Predicted Histology": res["histology_label"],
                        "Predicted Stage": res["stage_label"],
                        "Survival (Days)": res["survival_days"],
                        "Survival (Years)": res["survival_years"]
                    })
                    progress_bar.progress((idx + 1) / num_patients)
                    
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                csv_data = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download CSV Diagnostic Report",
                    data=csv_data,
                    file_name="LungCancerAI_batch_predictions.csv",
                    mime="text/csv",
                    key="dl_batch_csv"
                )

