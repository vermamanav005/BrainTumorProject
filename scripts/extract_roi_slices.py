#!/usr/bin/env python3
"""
extract_roi_slices.py  (Stage 3A)
---------------------------------
Generate 256x256 tumor ROI crops using BraTS ground-truth segmentation masks.

Modalities combined into RGB:
    R = T1CE,  G = T2,  B = FLAIR

Outputs:
    out_dir/
        train/HGG/
        train/LGG/
        val/HGG/
        val/LGG/
        test/HGG/
        test/LGG/
    slices_roi_metadata.csv

Requires:
    --grades_csv : CSV mapping patient -> HGG/LGG
"""

import os, json, argparse, csv, glob
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
import pandas as pd

# -------------------- helpers -------------------- #

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def find_modality(patient_dir, pid, modality):
    """Locate .nii or .nii.gz for the modality."""
    for ext in [".nii.gz", ".nii"]:
        f = os.path.join(patient_dir, f"{pid}_{modality}{ext}")
        if os.path.exists(f):
            return f
    # fallback: wildcard search
    matches = glob.glob(os.path.join(patient_dir, f"{pid}_{modality}.nii*"))
    return matches[0] if matches else None

def clip_and_normalize(vol, low_q=0.5, high_q=99.5):
    """Clip non-zero intensities to percentiles and scale to [0,1]."""
    v = vol[vol > 0]
    if v.size == 0:
        return np.zeros_like(vol, dtype=np.float32)
    lo, hi = np.percentile(v, [low_q, high_q])
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + 1e-8)
    return np.clip(vol, 0, 1).astype(np.float32)

def load_rgb_and_seg(patient_dir, pid):
    """Load T1CE, T2, FLAIR, and seg volumes; return (H,W,D,3), seg(H,W,D)."""
    t1ce_p = find_modality(patient_dir, pid, "t1ce")
    t2_p   = find_modality(patient_dir, pid, "t2")
    fl_p   = find_modality(patient_dir, pid, "flair")
    seg_p  = find_modality(patient_dir, pid, "seg")
    if not (t1ce_p and t2_p and fl_p and seg_p):
        return None

    t1ce = clip_and_normalize(nib.load(t1ce_p).get_fdata().astype(np.float32))
    t2   = clip_and_normalize(nib.load(t2_p).get_fdata().astype(np.float32))
    fl   = clip_and_normalize(nib.load(fl_p).get_fdata().astype(np.float32))
    seg  = nib.load(seg_p).get_fdata().astype(np.uint8)

    if not (t1ce.shape == t2.shape == fl.shape == seg.shape):
        print(f" Shape mismatch for {pid}, skipping.")
        return None

    r = (t1ce * 255).astype(np.uint8)
    g = (t2 * 255).astype(np.uint8)
    b = (fl * 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)  # (H, W, D, 3)
    return rgb, seg

def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()

# -------------------- processing -------------------- #

def process_patient(patient_dir, pid, grade, split, out_base,
                    writer, min_area=300, pad=12, size=256):
    """Extract all ROI crops for a given patient."""
    loaded = load_rgb_and_seg(patient_dir, pid)
    if loaded is None:
        return 0
    rgb_vol, seg_vol = loaded
    H, W, D, _ = rgb_vol.shape
    count = 0

    for z in range(D):
        mask = seg_vol[:, :, z]
        if (mask > 0).sum() < min_area:
            continue
        bbox = bbox_from_mask(mask)
        if bbox is None:
            continue
        xmin, ymin, xmax, ymax = bbox
        xmin = max(0, xmin - pad)
        ymin = max(0, ymin - pad)
        xmax = min(W - 1, xmax + pad)
        ymax = min(H - 1, ymax + pad)

        crop = rgb_vol[ymin:ymax+1, xmin:xmax+1, z, :]
        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        out_dir = os.path.join(out_base, split, grade)
        ensure_dir(out_dir)
        fname = f"{pid}_slice_{z:03d}.png"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, crop_resized, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        writer.writerow({
            "patient": pid,
            "split": split,
            "slice": int(z),
            "grade": grade,
            "bbox": json.dumps([int(xmin), int(ymin), int(xmax), int(ymax)]),
            "mask_area_pct": round(float((mask > 0).mean() * 100.0), 4),
            "mean_intensity": round(float(crop_resized.mean() / 255.0), 4),
            "path": fpath
        })
        count += 1
    return count

# -------------------- main -------------------- #

def main(args):
    ensure_dir(args.out_dir)
    meta_path = os.path.join(args.out_dir, "slices_roi_metadata.csv")

    # load grade mapping
    grades = pd.read_csv(args.grades_csv)
    if not {"patient", "grade"} <= set(grades.columns):
        raise ValueError("grades_csv must have columns: patient, grade")
    grade_map = dict(zip(grades["patient"].astype(str), grades["grade"].astype(str)))

    # load splits
    splits = {}
    for s in ["train", "val", "test"]:
        f = os.path.join(args.splits_dir, f"{s}_patients.json")
        splits[s] = json.load(open(f)) if os.path.exists(f) else []

    with open(meta_path, "w", newline="") as csvfile:
        fieldnames = ["patient", "split", "slice", "grade", "bbox",
                      "mask_area_pct", "mean_intensity", "path"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        total = 0

        for split in ["train", "val", "test"]:
            pids = splits.get(split, [])
            print(f"\nProcessing {split} ({len(pids)} patients)")
            for pid in tqdm(pids, desc=split.upper()):
                found_dir = None
                for base in [args.in_dir, args.val_dir]:
                    if not base: continue
                    cand = os.path.join(base, pid)
                    if os.path.exists(cand):
                        found_dir = cand
                        break
                if not found_dir:
                    # try recursive search
                    matches = glob.glob(os.path.join(args.in_dir, "**", pid), recursive=True)
                    if matches:
                        found_dir = matches[0]
                if not found_dir:
                    continue

                grade = grade_map.get(pid)
                if grade is None:
                    continue

                total += process_patient(found_dir, pid, grade, split, args.out_dir,
                                         writer, args.min_area, args.padding, args.size)

    print(f"\n Done. Total ROI slices saved: {total}")
    print(f"Metadata CSV: {meta_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract ROI crops from BraTS volumes using seg masks")
    p.add_argument("--in_dir", required=True, help="Path to BraTS training data root")
    p.add_argument("--val_dir", default=None, help="Optional BraTS validation data root")
    p.add_argument("--splits_dir", required=True, help="Folder containing train/val/test JSONs")
    p.add_argument("--grades_csv", required=True, help="CSV mapping patient -> HGG/LGG")
    p.add_argument("--out_dir", required=True, help="Output folder for ROI crops")
    p.add_argument("--min_area", type=int, default=300, help="Min tumor area (pixels) to keep slice")
    p.add_argument("--padding", type=int, default=12, help="Padding (pixels) around ROI")
    p.add_argument("--size", type=int, default=256, help="Output crop size (square)")
    args = p.parse_args()
    main(args)
