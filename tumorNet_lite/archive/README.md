# 🗄️ tumorNet_lite Archive

This directory contains old notebook versions and deprecated files from the `tumorNet_lite/` working directory.

**These files are NOT needed for current workflow.**

---

## 📁 Contents

### `old_notebooks/`
Previous versions of notebooks (before fixes):
- `TumorNetLite.ipynb` - Original version (had bugs)
- `TumorNetLitev2.ipynb` - Second version (had bugs)
- `preprocessing.ipynb` - Old preprocessing (inconsistent)

**Use instead:**
- `02_train_tumornet_lite.ipynb` (replaces TumorNet notebooks)
- `preprocessing_FIXED.ipynb` (replaces preprocessing.ipynb)

### `old_images/`
Old training outputs and checkpoints:
- `tumornet_lite_best.pth` - Old checkpoint
- `tumornet_lite_best2.pth` - Old checkpoint
- `confusion_matrix.png` - Old visualization
- `confusion_matrix2.png` - Old visualization
- `training_history.png` - Old training curves
- `training_history2.png` - Old training curves

**New outputs saved to:** `../../results/` and `../../checkpoints/`

### Root Files:
- `experiment_runner.py` - CLI version (replaced by notebooks)

**Use instead:** Notebooks `02_*.ipynb`, `03_*.ipynb`, `04_*.ipynb`

---

## 🐛 Why These Were Replaced

All archived notebooks had one or more of these issues:
1. ❌ Preprocessing inconsistency
2. ❌ Data leakage (train/test contamination)
3. ❌ Model weight carryover (not training from scratch)
4. ❌ Incomplete reproducibility (missing seed settings)
5. ❌ Transform ordering issues
6. ❌ Hardcoded paths

**Fixed versions** have all these issues resolved.

---

## 📊 Current Workflow (Clean)

```
✅ 01_setup_and_config.ipynb       (setup)
✅ preprocessing_FIXED.ipynb       (preprocess)
✅ 02_train_tumornet_lite.ipynb    (train)
✅ 03_ablation_study.ipynb         (analyze)
✅ 04_baseline_comparison.ipynb    (compare)
```

---

## 🗑️ Can I Delete This?

**Yes**, but keeping it provides:
- Reference for comparing old vs new approaches
- Ability to recover old code if needed
- Understanding of what was fixed

**Disk usage:** ~200 MB (mostly old checkpoints)

---

**Last Updated:** December 1, 2025  
**Status:** Archived - use active notebooks instead
