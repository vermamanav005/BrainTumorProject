# 🗄️ Archive Directory

This directory contains old experiments, deprecated notebooks, and historical files.

**These files are kept for reference but are NOT needed for current workflow.**

---

## 📁 Contents

### `old_experiments/`
Old experimental versions before the clean implementation:
- `LightTumorNet/` - Early experimental version
- `Light_tumor_net_v2/` - Second experimental version  
- `basic/` - Basic classification experiments
- `test_notebooks/` - Testing and inference notebooks
- `notebooks/` - Miscellaneous verification notebooks

**Status:** Superseded by `tumorNet_lite/` notebooks

---

### `documentation/`
Outdated documentation files:
- `COMPLETE_FILE_MANIFEST.md` - Old file listing (outdated)

**Current docs:** See main directory for up-to-date documentation

---

## ❓ Why Keep This?

✅ **Reference** - Compare old vs new approaches  
✅ **Recovery** - Retrieve old code if needed  
✅ **History** - Track project evolution  
✅ **Learning** - See what didn't work

---

## 🗑️ Can I Delete This?

**Yes, but not recommended.**

These files don't affect current workflow but provide valuable reference. If disk space is critical, you can:

```bash
# Option 1: Keep archive (recommended)
# Just ignore it

# Option 2: Compress archive
tar -czf archive_backup.tar.gz archive/
rm -rf archive/

# Option 3: Delete completely (not recommended)
rm -rf archive/
```

---

## 📊 What to Use Instead

| Old File | Use This Instead |
|----------|------------------|
| `LightTumorNet/*` | `tumorNet_lite/02_train_tumornet_lite.ipynb` |
| `basic/preprocessing.ipynb` | `tumorNet_lite/preprocessing_FIXED.ipynb` |
| `test_notebooks/inference_*.ipynb` | Evaluate in training notebooks |
| Old documentation | `PROJECT_STRUCTURE.md`, `START_HERE.md` |

---

## 🔄 Archive Structure

```
archive/
├── old_experiments/
│   ├── LightTumorNet/
│   │   ├── light_tumor_net.ipynb
│   │   ├── model2.ipynb
│   │   ├── overfit_diagnostics.ipynb
│   │   └── *.pth (old checkpoints)
│   │
│   ├── Light_tumor_net_v2/
│   │   └── data.ipynb
│   │
│   ├── basic/
│   │   ├── BrainTumorClassification.ipynb
│   │   ├── preprocessing.ipynb
│   │   └── logs/
│   │
│   ├── test_notebooks/
│   │   ├── test.ipynb
│   │   └── inference_efficientnet_b0.ipynb
│   │
│   └── notebooks/
│       └── verify_roi_dataset.ipynb
│
└── documentation/
    └── COMPLETE_FILE_MANIFEST.md
```

---

**Last Updated:** December 1, 2025  
**Status:** Archived, safe to ignore  
**Disk Usage:** ~500 MB (model checkpoints)
