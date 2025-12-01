# 🧠 Brain Tumor Classification Project

**Clean, organized research project for brain tumor classification using deep learning.**

**✅ Windows 11 Compatible** - All paths and configurations updated for Windows!

---

## 📁 Project Structure

```
BrainTumorProject/
│
├── tumorNet_lite/              # ✅ MAIN WORKING DIRECTORY
│   ├── 01_setup_and_config.ipynb          # Setup & configuration
│   ├── 02_train_tumornet_lite.ipynb       # Main training
│   ├── 03_ablation_study.ipynb            # Component analysis
│   ├── 04_baseline_comparison.ipynb       # Model comparison
│   ├── preprocessing_FIXED.ipynb          # Data preprocessing
│   ├── TumorNetLitev2_FIXED.ipynb         # Alternative training notebook
│   │
│   ├── utils.py                           # Shared utility functions
│   ├── models.py                          # Model architectures
│   ├── requirements.txt                   # Python dependencies
│   │
│   ├── NOTEBOOK_EXECUTION_ORDER.md        # Execution guide
│   ├── docs/                              # Additional documentation
│   └── archive/                           # Old/deprecated files
│
├── config.yaml                 # ✅ CONFIGURATION FILE
│
├── BUGS_IDENTIFIED.md          # Bug analysis documentation
├── SUMMARY.md                  # Project summary
├── FIX_INSTRUCTIONS.md         # Implementation guide
├── README_QUICKSTART.md        # Quick start guide
├── README.md                   # Original project README
│
├── scripts/                    # Data processing scripts
│   ├── extract_roi_slices.py
│   ├── extract_slices_multimodal_8bit.py
│   ├── inspect_dataset.py
│   └── make_patient_splits.py
│
├── metadata/                   # Dataset metadata
│   └── grades.csv
│
├── outputs/                    # Processing outputs
│   ├── inspect/
│   └── splits/
│
└── archive/                    # Archived experiments
    ├── old_experiments/        # Old experiment folders
    └── documentation/          # Old documentation

```

---

## 🚀 Quick Start

### **1. Install Dependencies**
```cmd
REM Windows Command Prompt
cd tumorNet_lite
pip install -r requirements.txt
```

Or using PowerShell:
```powershell
# Windows PowerShell
Set-Location tumorNet_lite
pip install -r requirements.txt
```

### **2. Run Notebooks in Order**
```cmd
jupyter notebook
```

Open and execute:
1. `01_setup_and_config.ipynb` - Setup (2 min)
2. `preprocessing_FIXED.ipynb` - Preprocess data (5-10 min)
3. `02_train_tumornet_lite.ipynb` - Train main model (30-60 min)
4. `03_ablation_study.ipynb` - Component analysis (2-3 hours) [Optional]
5. `04_baseline_comparison.ipynb` - Compare models (3-5 hours) [Optional]

### **3. View Results**
Results will be saved in:
- `..\checkpoints\` - Model checkpoints
- `..\results\` - Metrics, plots, JSON files
- `..\logs\` - Training logs

**Note:** Windows uses backslashes, but Python handles forward slashes automatically.

---

## 📊 What This Project Does

✅ **Preprocesses brain tumor MRI images** with consistent pipeline  
✅ **Trains TumorNet-Lite** - novel lightweight architecture  
✅ **Ablation study** - validates component contributions  
✅ **Baseline comparison** - compares against ResNet, EfficientNet, MobileNets, DMFNet  
✅ **Generates publication-ready visualizations**  

---

## 🔧 Key Files

| File | Purpose |
|------|---------|
| `config.yaml` | All hyperparameters and paths |
| `utils.py` | Data loading, training, evaluation functions |
| `models.py` | TumorNet-Lite and baseline architectures |
| `NOTEBOOK_EXECUTION_ORDER.md` | Detailed execution guide |
| `BUGS_IDENTIFIED.md` | Bug analysis and fixes |
| `SUMMARY.md` | Executive summary |

---

## 📦 Output Directories

**Created automatically:**
- `preprocessed_canonical/` - Preprocessed images (train/val/test splits)
- `checkpoints/` - Saved model weights
- `results/` - Metrics, confusion matrices, ROC curves
- `logs/` - TensorBoard logs
- `figures/` - Publication-quality plots

---

## 🧹 Archived Content

Old experiments and deprecated files have been moved to `archive/`:
- `archive/old_experiments/` - Previous experimental notebooks
- `archive/documentation/` - Old documentation files
- `tumorNet_lite/archive/` - Old notebook versions

**These are kept for reference but not needed for current workflow.**

---

## 📚 Documentation

- **🪟 Windows Setup:** `WINDOWS_SETUP.md` ⭐ **START HERE for Windows 11!**
- **Quick Start:** `README_QUICKSTART.md`
- **Execution Order:** `tumorNet_lite/NOTEBOOK_EXECUTION_ORDER.md`
- **Bug Analysis:** `BUGS_IDENTIFIED.md`
- **Implementation Guide:** `FIX_INSTRUCTIONS.md`
- **Summary:** `SUMMARY.md`

---

## 🎯 Research Workflow

```
Setup → Preprocess → Train → Analyze → Compare → Publish
  ↓         ↓          ↓        ↓         ↓         ↓
 01.ipynb  preprocessing  02.ipynb  03.ipynb  04.ipynb  Results!
```

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{tumornetlite2025,
  title={TumorNet-Lite: Lightweight Deep Learning Framework for Brain Tumor Classification},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

---

## 🆘 Troubleshooting

**Issue:** "Module not found"  
**Solution:** `pip install -r tumorNet_lite\requirements.txt`

**Issue:** "Preprocessed data not found"  
**Solution:** Run `preprocessing_FIXED.ipynb` first

**Issue:** "Out of memory"  
**Solution:** Reduce `batch_size` in `config.yaml`

**Issue:** "num_workers error on Windows"  
**Solution:** Set `num_workers: 0` in `config.yaml` (required for Windows)

**Issue:** "Path not found"  
**Solution:** Use forward slashes (/) in paths, even on Windows

See `WINDOWS_SETUP.md` for more Windows-specific troubleshooting.

---

**Status:** ✅ Clean, organized, ready for research  
**Last Updated:** December 1, 2025
