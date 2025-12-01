# ⚡ START HERE - Quick Reference

**Last Updated:** December 1, 2025  
**Status:** ✅ Ready to use  
**Platform:** ✅ Windows 11 Compatible

---

## 🪟 Windows Users - Read This First!

**Complete Windows setup guide:** `WINDOWS_SETUP.md` ⭐

**Quick Windows setup:**
1. Install Python 3.8+ (check "Add to PATH")
2. Open Command Prompt or PowerShell
3. Follow commands below

---

## 🎯 Essential Files Only

### **📂 Working Directory: `tumorNet_lite/`**

#### **Run These Notebooks (in order):**
```
1. 01_setup_and_config.ipynb          ← Start here (2 min)
2. preprocessing_FIXED.ipynb          ← Preprocess data (5-10 min)
3. 02_train_tumornet_lite.ipynb       ← Train model (30-60 min)
4. 03_ablation_study.ipynb            ← Optional: Component analysis
5. 04_baseline_comparison.ipynb       ← Optional: Compare models
```

#### **Keep These Files (imported by notebooks):**
```
utils.py                              ← Helper functions
models.py                             ← Model architectures  
requirements.txt                      ← Dependencies
config.yaml                           ← Configuration (in parent dir)
```

---

## 🚀 Quick Start (3 Commands) - Windows 11

```cmd
cd tumorNet_lite
pip install -r requirements.txt
jupyter notebook
```

Or using PowerShell:
```powershell
Set-Location tumorNet_lite
pip install -r requirements.txt
jupyter notebook
```

Then open `01_setup_and_config.ipynb` and run all cells.

---

## 📋 What Each File Does

| File | Time | Purpose |
|------|------|---------|
| `01_setup_and_config.ipynb` | 2 min | Load config, set seeds, verify directories |
| `preprocessing_FIXED.ipynb` | 5-10 min | Clean data pipeline (run once) |
| `02_train_tumornet_lite.ipynb` | 30-60 min | Train your main model |
| `03_ablation_study.ipynb` | 2-3 hrs | Test component contributions |
| `04_baseline_comparison.ipynb` | 3-5 hrs | Compare 6 models |

---

## 🗄️ Ignore These (Archived)

- `archive/` folder - Old experiments (safe to ignore)
- `docs/` folder - Reference documentation
- Other files - Bug analysis, summaries, guides

---

## 📊 Output Locations

After running notebooks, find results in:
```
..\checkpoints\          ← Saved models (.pth files)
..\results\              ← Metrics, plots, JSON
..\preprocessed_canonical\   ← Preprocessed images
..\logs\                 ← Training logs
```

Note: Windows uses backslashes (\) but Python handles forward slashes (/) automatically.

---

## 💡 Tips

✅ **Run notebooks in order** - Each depends on previous  
✅ **Keep `utils.py` and `models.py`** - Notebooks import these  
✅ **Edit `config.yaml`** - Change hyperparameters here  
✅ **Check `NOTEBOOK_EXECUTION_ORDER.md`** - Detailed guide  

---

## 🆘 Troubleshooting

**Error: "Module not found"**
```cmd
pip install -r requirements.txt
```

**Error: "Preprocessed data not found"**
```cmd
REM Run preprocessing_FIXED.ipynb first
```

**Error: "Out of memory"**
```cmd
REM Edit config.yaml, reduce batch_size to 16 or 8
REM Windows note: Set num_workers: 0 in config.yaml
```

---

## 📚 Documentation Files

If you want more details:
- 🪟 `WINDOWS_SETUP.md` - Complete Windows 11 setup guide
- `NOTEBOOK_EXECUTION_ORDER.md` - Detailed execution guide
- `PROJECT_STRUCTURE.md` - Clean project structure
- `CLEANUP_SUMMARY.md` - What was cleaned
- `BUGS_IDENTIFIED.md` - What was fixed
- `README_QUICKSTART.md` - Extended quick start

---

## ✨ You're Ready!

**Windows Command Prompt:**
```cmd
cd tumorNet_lite
jupyter notebook 01_setup_and_config.ipynb
```

**Windows PowerShell:**
```powershell
Set-Location tumorNet_lite
jupyter notebook 01_setup_and_config.ipynb
```

**That's it!** Just run the notebooks in order. 🚀

---

**Questions?** Check `NOTEBOOK_EXECUTION_ORDER.md` for complete guide.
