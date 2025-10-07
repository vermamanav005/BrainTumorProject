import os, json, argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def make_splits(summary_csv, outdir, seed=42, train_frac=0.70, val_frac=0.15, test_frac=0.15, n_bins=5):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(summary_csv)

    # Sanity check
    if 'tumor_pct' not in df.columns:
        raise SystemExit("summary.csv must contain 'tumor_pct' column. Run inspect_dataset.py first.")

    # Remove patients with missing tumor_pct
    df = df.dropna(subset=['tumor_pct']).reset_index(drop=True)
    print(f" Using {len(df)} patients with valid tumor data")

    # Bin tumor percentages for stratified sampling
    df['tumor_bin'] = pd.qcut(df['tumor_pct'].fillna(0), q=n_bins, duplicates='drop').astype(str)

    ids = df['patient'].values
    y = df['tumor_bin'].values

    # Split train vs (val+test)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_frac), random_state=seed)
    train_idx, rest_idx = next(sss.split(ids, y))
    train_ids = ids[train_idx]
    rest_ids = ids[rest_idx]
    rest_y = y[rest_idx]

    # Split rest into val/test
    test_size = test_frac / (test_frac + val_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + 1)
    val_idx_rel, test_idx_rel = next(sss2.split(rest_ids, rest_y))
    val_ids = rest_ids[val_idx_rel]
    test_ids = rest_ids[test_idx_rel]

    # Save as JSON
    with open(os.path.join(outdir, "train_patients.json"), "w") as f:
        json.dump(list(map(str, train_ids)), f, indent=2)
    with open(os.path.join(outdir, "val_patients.json"), "w") as f:
        json.dump(list(map(str, val_ids)), f, indent=2)
    with open(os.path.join(outdir, "test_patients.json"), "w") as f:
        json.dump(list(map(str, test_ids)), f, indent=2)

    # Print stats
    print("\n Split Summary")
    print(f"Train: {len(train_ids)} patients")
    print(f"Val:   {len(val_ids)} patients")
    print(f"Test:  {len(test_ids)} patients")

    print(f"\n JSON files saved in: {outdir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Create stratified train/val/test patient splits from summary.csv")
    p.add_argument("--summary_csv", required=True, help="Path to outputs/inspect/summary.csv")
    p.add_argument("--outdir", required=True, help="Output folder for split JSONs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    make_splits(args.summary_csv, args.outdir, args.seed)
