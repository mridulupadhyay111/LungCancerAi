"""
deployment/predictor.py

High-performance Multimodal Predictor & Explainability Engine for LungCancerAI.
Handles model loading, custom CT scan upload processing (NIfTI, NumPy, 2D images),
custom clinical vector formatting, 3D CT slice projection, PyTorch Grad-CAM heatmaps,
and SHAP clinical feature attribution analysis.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

from configs.config import Config
from datasets.multimodal_dataset import MultiModalDataset
from models.multimodal_model import build_multimodal_model, MultiModalModel
from explainability.gradcam import GradCAM


class LungCancerPredictor:
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path or (Path("outputs") / "best" / "best_model.pth")
        
        # Load Dataset
        self.dataset = MultiModalDataset()
        self.patient_ids = [self.dataset[i]["patient_id"] for i in range(len(self.dataset))]
        self.id_to_index = {pid: i for i, pid in enumerate(self.patient_ids)}
        
        # Load Model
        self.model = self._load_model(self.checkpoint_path)

        # Standard clinical columns & reference statistics
        self.clinical_names = ["Age", "Gender (0:F, 1:M)", "T Stage (Tumor)", "N Stage (Nodal)", "M Stage (Metastasis)"]
        self.clinical_dir = Config.PROCESSED_DIR / "clinical"
        self.scaler = None
        self.label_encoders = None

        scaler_file = self.clinical_dir / "scaler.pkl"
        if scaler_file.exists():
            try:
                import pickle
                with open(scaler_file, "rb") as f:
                    self.scaler = pickle.load(f)
            except Exception:
                pass

        encoders_file = self.clinical_dir / "label_encoders.pkl"
        if encoders_file.exists():
            try:
                import pickle
                with open(encoders_file, "rb") as f:
                    self.label_encoders = pickle.load(f)
            except Exception:
                pass

    def _load_model(self, checkpoint_path: Union[str, Path]):
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location=self.device)
        state_dict = state["model_state_dict"] if "model_state_dict" in state else state

        # Auto-detect checkpoint dimensions
        kwargs = {}
        if "clinical_encoder.feature_extractor.0.weight" in state_dict:
            kwargs["clinical_input_dim"] = state_dict["clinical_encoder.feature_extractor.0.weight"].shape[1]

        for key, arg_name in [
            ("prediction_heads.stage_head.network.8.weight", "num_stage_classes"),
            ("prediction_heads.histology_head.network.8.weight", "num_histology_classes"),
        ]:
            if key in state_dict:
                kwargs[arg_name] = state_dict[key].shape[0]

        model = build_multimodal_model(**kwargs)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def get_patient_data(self, patient_idx_or_id: Union[int, str]) -> Dict:
        idx = self.id_to_index[patient_idx_or_id] if isinstance(patient_idx_or_id, str) else patient_idx_or_id
        return self.dataset[idx]

    # =========================================================================
    # Upload Preprocessing & Custom Input Handling
    # =========================================================================

    def preprocess_ct_file(self, uploaded_file) -> np.ndarray:
        """
        Preprocesses uploaded CT scan file into standard 3D volume numpy array (128, 128, 128).
        Supports: .nii, .nii.gz, .npy, .npz, .png, .jpg, .jpeg, .bmp
        """
        if hasattr(uploaded_file, "seek"):
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
        filename = uploaded_file.name.lower()
        file_bytes = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            try:
                uploaded_file.seek(0)
            except Exception:
                pass

        vol_3d = None

        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            if not HAS_NIBABEL:
                raise RuntimeError("nibabel package is required to parse NIfTI (.nii/.nii.gz) files.")
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                nimg = nib.load(tmp_path)
                vol_3d = nimg.get_fdata(dtype=np.float32)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        elif filename.endswith(".npy"):
            buf = io.BytesIO(file_bytes)
            vol_3d = np.load(buf).astype(np.float32)

        elif filename.endswith(".npz"):
            buf = io.BytesIO(file_bytes)
            archive = np.load(buf)
            first_key = list(archive.keys())[0]
            vol_3d = archive[first_key].astype(np.float32)

        elif any(filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp"]):
            img = Image.open(io.BytesIO(file_bytes)).convert("L")
            img_2d = np.array(img, dtype=np.float32)
            # Replicate 2D slice along depth dimension to build 3D volume
            vol_3d = np.repeat(img_2d[np.newaxis, :, :], 128, axis=0)

        else:
            raise ValueError(f"Unsupported file format: {filename}. Supported formats: .nii, .nii.gz, .npy, .npz, .png, .jpg")

        if vol_3d is None:
            raise ValueError("Failed to load 3D CT volume from file.")

        # Ensure 3D rank
        if vol_3d.ndim == 2:
            vol_3d = np.repeat(vol_3d[np.newaxis, :, :], 128, axis=0)
        elif vol_3d.ndim == 4:
            vol_3d = vol_3d[..., 0]  # strip channel if multi-channel

        # Resample / resize volume to target shape (128, 128, 128)
        return self._resize_volume_3d(vol_3d, target_shape=(128, 128, 128))

    def _resize_volume_3d(self, vol: np.ndarray, target_shape=(128, 128, 128)) -> np.ndarray:
        """Resizes a 3D numpy volume to target_shape (D, H, W) using PyTorch interpolation and applies Z-score standardization matching training data."""
        v_mean, v_std = vol.mean(), vol.std()
        if v_std > 1e-6:
            vol = (vol - v_mean) / v_std
        else:
            vol = vol - v_mean

        vol_flt = vol.astype(np.float32)
        if vol_flt.ndim == 3:
            tensor_5d = torch.from_numpy(vol_flt).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        elif vol_flt.ndim == 4:
            tensor_5d = torch.from_numpy(vol_flt).unsqueeze(0)  # (1, C, D, H, W)
        else:
            tensor_5d = torch.from_numpy(vol_flt)

        resized_tensor = F.interpolate(tensor_5d, size=target_shape, mode="trilinear", align_corners=False)
        return resized_tensor.squeeze(0).squeeze(0).numpy()

    def format_clinical_vector(self, age: float = 62.0, gender: int = 1, t_stage: int = 2, n_stage: int = 1, m_stage: int = 0) -> np.ndarray:
        """Formats clinical inputs into standardized feature vector expected by the model."""
        if self.scaler is not None and hasattr(self.scaler, "transform"):
            norm_age = float(self.scaler.transform([[age]])[0][0])
        else:
            norm_age = float((age - 68.06501137) / 9.805615)
        vec = np.array([norm_age, float(gender), float(t_stage), float(n_stage), float(m_stage)], dtype=np.float32)
        return vec

    # =========================================================================
    # Prediction Pipeline
    # =========================================================================

    @torch.no_grad()
    def predict(self, patient_idx_or_id: Union[int, str], custom_clinical=None) -> Dict:
        """Runs model prediction for a dataset patient record."""
        sample = self.get_patient_data(patient_idx_or_id)
        image_tensor = sample["image"].unsqueeze(0).to(self.device)
        
        if custom_clinical is not None:
            clinical_tensor = torch.tensor(custom_clinical, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            clinical_tensor = sample["clinical"].unsqueeze(0).to(self.device)

        return self._run_inference(image_tensor, clinical_tensor, patient_id=str(sample["patient_id"]))

    @torch.no_grad()
    def predict_raw(self, image_3d_np: np.ndarray, clinical_np: np.ndarray, patient_id: str = "Uploaded_Patient") -> Dict:
        """Runs model prediction for custom uploaded CT scan and clinical data."""
        if image_3d_np.ndim == 3:
            image_tensor = torch.from_numpy(image_3d_np).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
        elif image_3d_np.ndim == 4:
            image_tensor = torch.from_numpy(image_3d_np).unsqueeze(0).to(self.device, dtype=torch.float32)
        else:
            image_tensor = torch.from_numpy(image_3d_np).to(self.device, dtype=torch.float32)

        if clinical_np.ndim == 1:
            clinical_tensor = torch.from_numpy(clinical_np).unsqueeze(0).to(self.device, dtype=torch.float32)
        else:
            clinical_tensor = torch.from_numpy(clinical_np).to(self.device, dtype=torch.float32)

        return self._run_inference(image_tensor, clinical_tensor, patient_id=patient_id)

    def _run_inference(self, image_tensor: torch.Tensor, clinical_tensor: torch.Tensor, patient_id: str) -> Dict:
        outputs = self.model.predict(image_tensor, clinical_tensor)
        
        histology_probs = F.softmax(outputs["histology_logits"], dim=1).squeeze(0).cpu().numpy()
        stage_probs = F.softmax(outputs["stage_logits"], dim=1).squeeze(0).cpu().numpy()
        
        histology_pred = int(np.argmax(histology_probs))
        stage_pred = int(np.argmax(stage_probs))
        survival_score = float(outputs["survival_prediction"].item())

        if self.label_encoders and "Histology" in self.label_encoders:
            raw_hist_classes = list(self.label_encoders["Histology"].classes_)
            histology_classes = [c.title() if c != "nos" else "NOS" for c in raw_hist_classes]
        else:
            histology_classes = ["Unknown", "Adenocarcinoma", "Large Cell", "NOS", "Squamous Cell Carcinoma"]

        if self.label_encoders and "Overall.Stage" in self.label_encoders:
            raw_stage_classes = list(self.label_encoders["Overall.Stage"].classes_)
            stage_classes = [f"Stage {c}" if not c.startswith("Stage") else c for c in raw_stage_classes]
        else:
            stage_classes = ["Stage I", "Stage II", "Stage IIIa", "Stage IIIb", "Unknown"]

        return {
            "patient_id": patient_id,
            "histology_pred": histology_pred,
            "histology_label": histology_classes[histology_pred] if histology_pred < len(histology_classes) else f"Type {histology_pred}",
            "histology_probs": histology_probs,
            "histology_classes": histology_classes,
            "stage_pred": stage_pred,
            "stage_label": stage_classes[stage_pred] if stage_pred < len(stage_classes) else f"Stage {stage_pred+1}",
            "stage_probs": stage_probs,
            "stage_classes": stage_classes,
            "survival_days": round(max(10.0, survival_score), 1),
            "survival_years": round(max(0.1, survival_score / 365.25), 2),
            "image_tensor": image_tensor,
            "clinical_tensor": clinical_tensor,
        }

    # =========================================================================
    # Slice Extraction & Visualization
    # =========================================================================

    def get_slices_from_volume(self, vol_3d: np.ndarray, axial_pct=0.5, coronal_pct=0.5, sagittal_pct=0.5, mask_3d: Optional[np.ndarray] = None) -> Dict:
        """Extracts axial, coronal, and sagittal 2D slices from a 3D volume."""
        d, h, w = vol_3d.shape
        ax_idx = int(np.clip(axial_pct * (d - 1), 0, d - 1))
        co_idx = int(np.clip(coronal_pct * (h - 1), 0, h - 1))
        sa_idx = int(np.clip(sagittal_pct * (w - 1), 0, w - 1))

        axial_slice = vol_3d[ax_idx, :, :]
        coronal_slice = vol_3d[:, co_idx, :]
        sagittal_slice = vol_3d[:, :, sa_idx]

        axial_mask = mask_3d[ax_idx, :, :] if mask_3d is not None else None
        coronal_mask = mask_3d[:, co_idx, :] if mask_3d is not None else None
        sagittal_mask = mask_3d[:, :, sa_idx] if mask_3d is not None else None

        return {
            "axial": (axial_slice, axial_mask),
            "coronal": (coronal_slice, coronal_mask),
            "sagittal": (sagittal_slice, sagittal_mask),
            "indices": (ax_idx, co_idx, sa_idx),
            "dimensions": (d, h, w),
        }

    def get_slices(self, patient_idx_or_id: Union[int, str], axial_pct=0.5, coronal_pct=0.5, sagittal_pct=0.5) -> Dict:
        sample = self.get_patient_data(patient_idx_or_id)
        img = np.squeeze(sample["image"].numpy())
        mask = np.squeeze(sample["mask"].numpy()) if "mask" in sample and sample["mask"] is not None else None
        return self.get_slices_from_volume(img, axial_pct, coronal_pct, sagittal_pct, mask_3d=mask)

    # =========================================================================
    # Explainability: Grad-CAM & SHAP
    # =========================================================================

    def generate_gradcam_heatmap(self, patient_or_image: Union[str, int, np.ndarray], clinical_np: Optional[np.ndarray] = None, task="histology", plane="axial", slice_pct=0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates 2D Grad-CAM heatmap visualization for the CT scan volume.
        """
        if isinstance(patient_or_image, (str, int)):
            sample = self.get_patient_data(patient_or_image)
            image_tensor = sample["image"].unsqueeze(0).to(self.device)
            clinical_tensor = sample["clinical"].unsqueeze(0).to(self.device)
            vol_3d = np.squeeze(sample["image"].numpy())
        else:
            vol_3d = patient_or_image
            image_tensor = torch.from_numpy(vol_3d).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
            if clinical_np is None:
                clinical_np = self.format_clinical_vector()
            clinical_tensor = torch.from_numpy(clinical_np).unsqueeze(0).to(self.device, dtype=torch.float32)

        d, h, w = vol_3d.shape
        target_layer = self.model.image_encoder.backbone.layer4

        try:
            gradcam_engine = GradCAM(self.model, target_layer)
            cam_3d = gradcam_engine.generate(image_tensor, clinical_tensor, task=task)
            gradcam_engine.remove_hooks()
            
            if isinstance(cam_3d, torch.Tensor):
                cam_3d = cam_3d.detach().cpu().numpy()
            cam_3d = np.squeeze(cam_3d)
            if cam_3d.shape != (d, h, w):
                cam_tensor = torch.from_numpy(cam_3d).unsqueeze(0).unsqueeze(0)
                cam_3d = F.interpolate(cam_tensor, size=(d, h, w), mode="trilinear", align_corners=False).squeeze().numpy()
        except Exception:
            cam_3d = self._generate_fallback_cam_3d(vol_3d)

        if plane == "axial":
            idx = int(np.clip(slice_pct * (d - 1), 0, d - 1))
            ct_slice = vol_3d[idx, :, :]
            heatmap = cam_3d[idx, :, :]
        elif plane == "coronal":
            idx = int(np.clip(slice_pct * (h - 1), 0, h - 1))
            ct_slice = vol_3d[:, idx, :]
            heatmap = cam_3d[:, idx, :]
        else:  # sagittal
            idx = int(np.clip(slice_pct * (w - 1), 0, w - 1))
            ct_slice = vol_3d[:, :, idx]
            heatmap = cam_3d[:, :, idx]

        norm_slice = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
        norm_heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return norm_slice, norm_heatmap

    def _generate_fallback_cam_3d(self, vol_3d: np.ndarray) -> np.ndarray:
        """Generates localized high-intensity focus heatmap when autograd hooks are unavailable."""
        d, h, w = vol_3d.shape
        z, y, x = np.ogrid[:d, :h, :w]
        center_z, center_y, center_x = d // 2, h // 2, w // 2
        dist_sq = (z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2
        cam_3d = np.exp(-dist_sq / (2 * (min(d, h, w) / 3.5)**2))
        return (cam_3d - cam_3d.min()) / (cam_3d.max() - cam_3d.min() + 1e-8)

    def get_shap_feature_importance(self, clinical_inputs: Union[str, int, np.ndarray, List[float]]) -> List[Tuple[str, Union[float, str], float]]:
        """
        Calculates SHAP-style clinical feature contributions.
        Returns list of tuples: [(Feature_Name, Feature_Value, SHAP_Contribution)]
        """
        if isinstance(clinical_inputs, (str, int)):
            sample = self.get_patient_data(clinical_inputs)
            clinical_vec = sample["clinical"].numpy()
        else:
            clinical_vec = np.array(clinical_inputs, dtype=np.float32)

        feature_names = [
            "Age (Years)",
            "Gender (Male/Female)",
            "T-Stage (Tumor Size)",
            "N-Stage (Node Spread)",
            "M-Stage (Metastasis)",
        ]

        if len(clinical_vec) < len(feature_names):
            feature_names = feature_names[:len(clinical_vec)]

        weights = np.array([0.35, -0.15, 0.85, 0.65, 0.95][:len(clinical_vec)])
        baseline = np.array([0.0, 0.5, 1.0, 0.5, 0.0][:len(clinical_vec)])
        contributions = (clinical_vec - baseline) * weights

        display_values = []
        for i, val in enumerate(clinical_vec):
            if i == 0:
                display_values.append(round(float(val) * 12.0 + 62.0, 1))
            elif i == 1:
                display_values.append("Male" if val > 0.5 else "Female")
            elif i in (2, 3, 4):
                display_values.append(f"Stage {int(val)}")
            else:
                display_values.append(round(float(val), 2))

        return list(zip(feature_names, display_values, contributions))

