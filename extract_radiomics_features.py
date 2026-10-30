"""
extract_radiomics_features.py

Extracts high-yield 3D volumetric, intensity, morphological, and spatial
biomarkers from CT scans and tumor masks for all clean patients.
Runs in ~30 seconds on CPU.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.stats import skew, kurtosis
from tqdm import tqdm

from configs.config import Config


def extract_patient_features(image, mask):
    if mask is None or np.sum(mask > 0.5) == 0:
        return None

    mask_bool = mask > 0.5
    tumor_voxels = image[mask_bool]
    
    # 1. Volume & Geometric Dimensions
    vol = float(np.sum(mask_bool))
    coords = np.where(mask_bool)
    z_span = float(coords[0].max() - coords[0].min() + 1)
    y_span = float(coords[1].max() - coords[1].min() + 1)
    x_span = float(coords[2].max() - coords[2].min() + 1)
    max_diam = float(np.sqrt(z_span**2 + y_span**2 + x_span**2))
    
    # Sphericity / compactness approximation
    # Radius of sphere of same volume: r = (3*V / (4*pi))^(1/3)
    eq_radius = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
    sphericity = (2.0 * eq_radius) / max(1.0, max_diam)

    # 2. Location (Normalized Centroid: Central vs Peripheral)
    d, h, w = image.shape
    c_z = float(np.mean(coords[0])) / max(1.0, d)
    c_y = float(np.mean(coords[1])) / max(1.0, h)
    c_x = float(np.mean(coords[2])) / max(1.0, w)
    # Distance from center of lung axial plane (0.5, 0.5)
    central_dist = float(np.sqrt((c_y - 0.5)**2 + (c_x - 0.5)**2))

    # 3. Intensity Distribution (Hounsfield Units)
    mean_val = float(np.mean(tumor_voxels))
    std_val = float(np.std(tumor_voxels))
    min_val = float(np.min(tumor_voxels))
    max_val = float(np.max(tumor_voxels))
    q10, q25, q50, q75, q90 = np.percentile(tumor_voxels, [10, 25, 50, 75, 90])
    iqr = float(q75 - q25)
    skew_val = float(skew(tumor_voxels)) if len(tumor_voxels) > 10 else 0.0
    kurt_val = float(kurtosis(tumor_voxels)) if len(tumor_voxels) > 10 else 0.0

    # 4. Peri-Tumoral Margin Invasiveness (Ring around the tumor)
    struct = generate_binary_structure(3, 1)
    dilated = binary_dilation(mask_bool, structure=struct, iterations=2)
    margin_ring = dilated & (~mask_bool)
    if np.sum(margin_ring) > 0:
        margin_voxels = image[margin_ring]
        margin_mean = float(np.mean(margin_voxels))
        margin_std = float(np.std(margin_voxels))
        margin_ratio = mean_val / max(0.01, abs(margin_mean) + 0.01)
    else:
        margin_mean = 0.0
        margin_std = 0.0
        margin_ratio = 1.0

    return {
        "tumor_volume": vol,
        "max_diameter": max_diam,
        "sphericity": sphericity,
        "centroid_z": c_z,
        "centroid_dist": central_dist,
        "intensity_mean": mean_val,
        "intensity_std": std_val,
        "intensity_min": min_val,
        "intensity_max": max_val,
        "intensity_q25": float(q25),
        "intensity_q50": float(q50),
        "intensity_q75": float(q75),
        "intensity_iqr": iqr,
        "intensity_skew": skew_val,
        "intensity_kurt": kurt_val,
        "margin_mean": margin_mean,
        "margin_std": margin_std,
        "margin_ratio": margin_ratio,
    }


def main():
    print("=" * 70)
    print("3D CT Radiomic & Morphological Feature Extraction")
    print("=" * 70)

    clean_csv = Config.PROCESSED_DIR / "clinical" / "clean_dataset.csv"
    if not clean_csv.exists():
        import prepare_clean_data
        prepare_clean_data.prepare_data()

    df = pd.read_csv(clean_csv)
    pids = df["PatientID"].tolist()
    proc_dir = Config.PROCESSED_DIR

    records = []
    for pid in tqdm(pids, desc="Extracting 3D Biomarkers"):
        img_path = proc_dir / pid / "image.npy"
        mask_path = proc_dir / pid / "mask.npy"

        if not img_path.exists() or not mask_path.exists():
            continue

        img = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        feats = extract_patient_features(img, mask)
        if feats is not None:
            feats["PatientID"] = pid
            records.append(feats)

    rad_df = pd.DataFrame(records)
    out_path = Config.PROCESSED_DIR / "clinical" / "radiomic_features.csv"
    rad_df.to_csv(out_path, index=False)

    # Merge with clean dataset
    full_multimodal = df.merge(rad_df, on="PatientID")
    full_path = Config.PROCESSED_DIR / "clinical" / "multimodal_master_dataset.csv"
    full_multimodal.to_csv(full_path, index=False)

    print("=" * 70)
    print(f"Extracted {len(rad_df.columns)-1} 3D spatial & texture features for {len(rad_df)} patients.")
    print(f"Saved master multimodal dataset to: {full_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
