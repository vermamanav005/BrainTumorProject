import os, json, argparse
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Optional: force non-interactive backend for Windows
plt.switch_backend("Agg")

def compute_tumor_pct(seg_path):
    """Compute tumor voxel ratio from segmentation mask."""
    seg = nib.load(seg_path).get_fdata()
    tumor_voxels = np.count_nonzero(seg)
    total_voxels = np.prod(seg.shape)
    return round(100.0 * tumor_voxels / total_voxels, 3)

def save_preview(flair_path, seg_path, out_path):
    """Save mid-slice overlay of FLAIR with tumor mask."""
    flair = nib.load(flair_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()
    mid = flair.shape[2] // 2

    img = flair[:, :, mid]
    mask = seg[:, :, mid]

    # Normalize for visualization
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

    plt.figure(figsize=(5, 5))
    plt.imshow(img.T, cmap='gray', origin='lower')
    plt.imshow(np.ma.masked_where(mask == 0, mask).T, cmap='autumn', alpha=0.5, origin='lower')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=100)
    plt.close()

def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summary = []

    patient_dirs = sorted([d for d in os.listdir(in_dir) if d.startswith("BraTS")])
    print(f" Found {len(patient_dirs)} patients")

    for pid in tqdm(patient_dirs, desc="Inspecting patients"):
        pdir = os.path.join(in_dir, pid)
        entry = {"patient": pid}

        try:
            t1 = os.path.join(pdir, f"{pid}_t1.nii")
            t2 = os.path.join(pdir, f"{pid}_t2.nii")
            flair = os.path.join(pdir, f"{pid}_flair.nii")
            t1ce = os.path.join(pdir, f"{pid}_t1ce.nii")
            seg = os.path.join(pdir, f"{pid}_seg.nii")

            entry["has_t1"] = os.path.exists(t1)
            entry["has_t2"] = os.path.exists(t2)
            entry["has_flair"] = os.path.exists(flair)
            entry["has_t1ce"] = os.path.exists(t1ce)
            entry["has_seg"] = os.path.exists(seg)

            if os.path.exists(flair):
                shape = nib.load(flair).shape
                entry["shape"] = str(shape)
                entry["slices"] = shape[2]
            else:
                entry["shape"] = None
                entry["slices"] = None

            if os.path.exists(seg):
                entry["tumor_pct"] = compute_tumor_pct(seg)
                preview_path = os.path.join(out_dir, f"{pid}_preview.png")
                save_preview(flair, seg, preview_path)
            else:
                entry["tumor_pct"] = np.nan

        except Exception as e:
            entry["error"] = str(e)

        summary.append(entry)

    # Save metadata
    df = pd.DataFrame(summary)
    csv_path = os.path.join(out_dir, "summary.csv")
    json_path = os.path.join(out_dir, "summary.json")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    # Print summary stats
    print("\n Inspection Complete")
    print(f"Patients processed: {len(df)}")
    print(f"Patients missing seg: {df['has_seg'].eq(False).sum()}")
    print(f"Avg slices per volume: {df['slices'].dropna().mean():.1f}")
    print(f"Avg tumor % (with seg): {df['tumor_pct'].dropna().mean():.2f}")
    print(f"\nCSV saved at: {csv_path}")
    print(f"JSON saved at: {json_path}")
    print(f"Previews saved in: {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Inspect BraTS dataset and build summary CSV")
    p.add_argument("--in_dir", required=True, help="Path to BraTS2020_TrainingData")
    p.add_argument("--out_dir", required=True, help="Where to save summary + previews")
    args = p.parse_args()

    main(args.in_dir, args.out_dir)
