"""
cache_features.py

Extracts and caches 3D CT image embeddings for all clean patients.
Runs the heavy 3D convolution backbone ONCE, saving embeddings to disk.
Enables instant (10-second) training for all subsequent experiments.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from configs.config import Config
from models.imageencoder import ImageEncoder
from datasets.transforms import Resize3D


def crop_roi(image, mask, margin_ratio=0.20):
    if mask is None or not (mask > 0.5).any():
        return image
    nonzero = np.where(mask > 0.5)
    z_min, z_max = int(nonzero[0].min()), int(nonzero[0].max() + 1)
    y_min, y_max = int(nonzero[1].min()), int(nonzero[1].max() + 1)
    x_min, x_max = int(nonzero[2].min()), int(nonzero[2].max() + 1)

    z_margin = max(4, int(margin_ratio * (z_max - z_min)))
    y_margin = max(8, int(margin_ratio * (y_max - y_min)))
    x_margin = max(8, int(margin_ratio * (x_max - x_min)))

    z_min = max(0, z_min - z_margin)
    z_max = min(image.shape[0], z_max + z_margin)
    y_min = max(0, y_min - y_margin)
    y_max = min(image.shape[1], y_max + y_margin)
    x_min = max(0, x_min - x_margin)
    x_max = min(image.shape[2], x_max + x_margin)

    return image[z_min:z_max, y_min:y_max, x_min:x_max]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Feature Extraction & Caching Pipeline")
    print(f"Device: {device}")
    print("=" * 70)

    clean_csv = Config.PROCESSED_DIR / "clinical" / "clean_dataset.csv"
    if not clean_csv.exists():
        print("Clean dataset not found. Running prepare_clean_data.py first...")
        import prepare_clean_data
        prepare_clean_data.prepare_data()

    df = pd.read_csv(clean_csv)
    pids = df["PatientID"].tolist()
    print(f"Processing {len(pids)} clean patients...")

    # Load Image Encoder
    encoder = ImageEncoder(embedding_dim=512, dropout=0.0).to(device)
    
    # Check if a trained checkpoint exists to initialize weights, otherwise standard initialization
    best_ckpt = Config.PROJECT_ROOT / "outputs" / "best" / "best_model.pth"
    if best_ckpt.exists():
        try:
            state = torch.load(best_ckpt, map_location=device)
            state_dict = state["model_state_dict"] if "model_state_dict" in state else state
            encoder_dict = {
                k.replace("image_encoder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("image_encoder.")
            }
            if encoder_dict:
                encoder.load_state_dict(encoder_dict, strict=False)
                print("Loaded image encoder weights from existing checkpoint!")
        except Exception as e:
            print(f"Note: Using initialized encoder weights ({e})")

    encoder.eval()
    resize = Resize3D((128, 128, 128))

    feature_dict = {}
    proc_dir = Config.PROCESSED_DIR

    with torch.no_grad():
        for pid in tqdm(pids, desc="Extracting 3D Features"):
            img_path = proc_dir / pid / "image.npy"
            mask_path = proc_dir / pid / "mask.npy"

            image = np.load(img_path).astype(np.float32)
            if mask_path.exists():
                mask = np.load(mask_path).astype(np.float32)
                image = crop_roi(image, mask, margin_ratio=0.20)

            # Resize to (128, 128, 128)
            vol = resize(image)
            vol = vol.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 128, 128, 128)

            emb = encoder(vol).cpu().squeeze(0).numpy()  # (512,)
            feature_dict[pid] = emb

    out_file = Config.PROCESSED_DIR / "cached_image_features.pt"
    torch.save(feature_dict, out_file)
    print("=" * 70)
    print(f"Successfully cached {len(feature_dict)} embeddings to:")
    print(out_file)
    print("=" * 70)


if __name__ == "__main__":
    main()
