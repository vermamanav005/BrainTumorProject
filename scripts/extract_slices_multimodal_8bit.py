"""
extract_slices_multimodal_8bit.py
---------------------------------
Create high-quality 8-bit multimodal RGB PNG slices from BraTS 2020 volumes.

Modalities:
    R = T1CE,  G = T2,  B = FLAIR
Default: CLAHE on (disable with --no_clahe)

Outputs
--------
out_dir/
    train/normal/
    train/tumor/
    val/...
    test/...
    slices_metadata.csv
"""

import os, json, argparse, csv, cv2, nibabel as nib, glob
import numpy as np
from tqdm import tqdm
from pathlib import Path

# -------------------- utility functions -------------------- #

def clip_and_normalize(vol, low_q=0.5, high_q=99.5):
    """Clip volume to percentiles and scale to [0,1]."""
    v = vol[vol > 0]
    if len(v) == 0:
        return np.zeros_like(vol)
    lo, hi = np.percentile(v, [low_q, high_q])
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + 1e-8)
    return np.clip(vol, 0, 1)

def apply_clahe_to_slice(img2d, clipLimit=2.0, tileGridSize=(8,8)):
    """Apply CLAHE to a single 2-D image (uint8)."""
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img2d)

def resize_slice(img, size=256):
    """Resize keeping aspect (simple center crop/pad)."""
    h, w = img.shape[:2]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# -------------------- helper for modality path -------------------- #

def find_modality_path(patient_dir, pid, modality):
    """Find .nii or .nii.gz file for the given modality."""
    pattern = os.path.join(patient_dir, f"{pid}_{modality}.nii*")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return matches[0]

# -------------------- core processing -------------------- #

def process_patient(patient_dir, out_base, split, clahe=True, size=256,
                    tumor_thresh=0.005, writer=None):
    """
    Process one BraTS patient folder:
      - load T1CE, T2, FLAIR, seg
      - normalize and optionally apply CLAHE
      - create RGB fusion
      - save slices and write metadata
    """
    pid = os.path.basename(patient_dir)
    t1ce_p = find_modality_path(patient_dir, pid, "t1ce")
    t2_p   = find_modality_path(patient_dir, pid, "t2")
    fl_p   = find_modality_path(patient_dir, pid, "flair")
    seg_p  = find_modality_path(patient_dir, pid, "seg")

    if not (t1ce_p and t2_p and fl_p and seg_p):
        print(f"⚠️ Missing modality for {pid}, skipping.")
        return

    # Load all 4 volumes (we use 3)
    t1ce = nib.load(t1ce_p).get_fdata().astype(np.float32)
    t2   = nib.load(t2_p).get_fdata().astype(np.float32)
    fl   = nib.load(fl_p).get_fdata().astype(np.float32)
    seg  = nib.load(seg_p).get_fdata().astype(np.uint8)

    # normalize each modality
    t1ce = clip_and_normalize(t1ce)
    t2   = clip_and_normalize(t2)
    fl   = clip_and_normalize(fl)

    num_slices = t1ce.shape[2]
    normal_count, tumor_count = 0, 0

    for z in range(num_slices):
        mask = seg[:, :, z]
        tumor_area = (mask > 0).mean()

        # Skip empty slices (no brain)
        if t1ce[:, :, z].max() == 0 and t2[:, :, z].max() == 0 and fl[:, :, z].max() == 0:
            continue

        label = "tumor" if tumor_area >= tumor_thresh else "normal"

        # Prepare RGB slice
        r = (t1ce[:, :, z] * 255).astype(np.uint8)
        g = (t2[:, :, z] * 255).astype(np.uint8)
        b = (fl[:, :, z] * 255).astype(np.uint8)

        if clahe:
            r = apply_clahe_to_slice(r)
            g = apply_clahe_to_slice(g)
            b = apply_clahe_to_slice(b)

        rgb = np.stack([r, g, b], axis=-1)
        rgb = resize_slice(rgb, size)

        out_dir = os.path.join(out_base, split, label)
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, f"{pid}_slice_{z:03d}.png")
        cv2.imwrite(out_path, rgb, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        # Update metadata
        writer.writerow({
            "patient": pid,
            "split": split,
            "slice": z,
            "label": label,
            "tumor_pct": round(float(tumor_area * 100), 3),
            "mean_intensity": round(float(rgb.mean() / 255), 4),
            "clahe_used": int(clahe),
            "path": out_path
        })

        if label == "tumor":
            tumor_count += 1
        else:
            normal_count += 1

    return tumor_count, normal_count

# -------------------- main entry -------------------- #

def main(args):
    ensure_dir(args.out_dir)
    meta_path = os.path.join(args.out_dir, "slices_metadata.csv")

    with open(meta_path, "w", newline="") as csvfile:
        fieldnames = [
            "patient","split","slice","label",
            "tumor_pct","mean_intensity","clahe_used","path"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Load split lists
        splits = {}
        for s in ["train","val","test"]:
            fp = os.path.join(args.splits_dir, f"{s}_patients.json")
            with open(fp) as f:
                splits[s] = json.load(f)

        # Merge training + validation directories if both exist
        in_dirs = [args.in_dir]
        if args.val_dir and os.path.exists(args.val_dir):
            in_dirs.append(args.val_dir)

        for split, patient_ids in splits.items():
            print(f"\n Processing {split} ({len(patient_ids)} patients)")
            for data_root in in_dirs:
                for pid in tqdm(patient_ids, desc=f"{split.upper()}"):
                    pdir = os.path.join(data_root, pid)
                    if not os.path.exists(pdir):
                        continue
                    process_patient(
                        pdir, args.out_dir, split,
                        clahe=not args.no_clahe,
                        size=args.size,
                        tumor_thresh=args.tumor_thresh,
                        writer=writer
                    )

    print("\n✅ Extraction complete")
    print(f"Metadata CSV saved at: {meta_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Extract multimodal 8-bit RGB slices from BraTS 2020 volumes"
    )
    p.add_argument("--in_dir", required=True,
                   help="Path to BraTS training data root")
    p.add_argument("--val_dir", default=None,
                   help="Optional validation data root")
    p.add_argument("--splits_dir", required=True,
                   help="Folder containing train/val/test JSONs")
    p.add_argument("--out_dir", required=True,
                   help="Where to save PNGs and metadata CSV")
    p.add_argument("--size", type=int, default=256,
                   help="Output slice size (default 256)")
    p.add_argument("--tumor_thresh", type=float, default=0.005,
                   help="Min tumor ratio (0-1) to mark slice as tumor")
    p.add_argument("--no_clahe", action="store_true",
                   help="Disable CLAHE enhancement")
    args = p.parse_args()
    main(args)
