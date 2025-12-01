# 📚 Research Notebook Execution Order

## Complete Workflow for Brain Tumor Classification Research

---

## 🎯 Quick Start (Minimum Required)

### **1. Setup & Configuration** (2 minutes)
```
📓 01_setup_and_config.ipynb
```
- Loads config.yaml
- Sets all random seeds
- Verifies directory structure
- Configures device (GPU/CPU)

### **2. Data Preprocessing** (5-10 minutes)
```
📓 preprocessing_FIXED.ipynb
```
- Loads raw MRI images
- Applies bilateral filter + COLORMAP_BONE
- Creates stratified splits (train/val/internal_test)
- Preserves held-out test set
- **Output:** `preprocessed_canonical/` directory

### **3. Train Main Model** (30-60 minutes)
```
📓 02_train_tumornet_lite.ipynb
```
- Trains TumorNet-Lite from scratch
- All bugs fixed (reproducibility, data leakage, etc.)
- Evaluates on internal test and held-out test
- **Output:** Model checkpoint + results

---

## 🔬 Complete Research Workflow (Recommended)

### **Phase 1: Setup** (7-12 minutes total)
1. **01_setup_and_config.ipynb** (2 min)
2. **preprocessing_FIXED.ipynb** (5-10 min)

### **Phase 2: Main Experiments** (30-60 minutes each)
3. **02_train_tumornet_lite.ipynb** - Your main model
4. **03_ablation_study.ipynb** - Component analysis
5. **04_baseline_comparison.ipynb** - Compare with other models

### **Phase 3: Analysis** (as needed)
6. **05_results_analysis.ipynb** - Statistical analysis & visualization

---

## 📋 Detailed Execution Order

### **Step 1: Initial Setup**
```bash
cd /Users/manavverma/Documents/desktop/TumorNetLite/BrainTumorProject/tumorNet_lite
jupyter notebook
```

Open and run in order:
1. `01_setup_and_config.ipynb` → Run all cells
2. `preprocessing_FIXED.ipynb` → Run all cells (creates preprocessed_canonical/)

**Verify:** Check that `../preprocessed_canonical/` exists with 4 subdirectories

---

### **Step 2: Main Training**
3. `02_train_tumornet_lite.ipynb` → Run all cells

**What it does:**
- Validates checkpoint doesn't exist (fresh training)
- Trains TumorNet-Lite with proper train/val/test protocol
- Generates confusion matrices, ROC curves, training curves
- Evaluates on held-out test set (first and only time)

**Output files:**
- `../checkpoints/tumornet_lite_main_YYYYMMDD_HHMMSS.pth`
- `../results/tumornet_lite_main_YYYYMMDD_HHMMSS_*.png`
- `../results/tumornet_lite_main_YYYYMMDD_HHMMSS_complete_results.json`

---

### **Step 3: Ablation Study** (Optional but recommended)
4. `03_ablation_study.ipynb` → Run all cells

**What it does:**
- Tests contribution of each TumorNet-Lite component
- Compares: Full model vs. simplified versions vs. baseline
- Generates comparison charts

**Expected pattern:** Full > Partial > Baseline (monotonic)

**Output:**
- `../results/ablation_study_YYYYMMDD_HHMMSS.json`
- `../results/ablation_comparison.png`

---

### **Step 4: Baseline Comparison** (Optional)
5. `04_baseline_comparison.ipynb` → Run all cells

**What it does:**
- Trains standard models: ResNet50, EfficientNet-B0, MobileNets, DMFNet
- Fair comparison (same data, same protocol)
- Compares accuracy, parameters, efficiency

**Output:**
- Individual checkpoints for each model
- Comparison table and visualizations
- `../results/baseline_comparison_YYYYMMDD_HHMMSS.json`

---

### **Step 5: Statistical Analysis** (Optional)
6. `05_results_analysis.ipynb` → Run all cells

**What it does:**
- Loads all experiment results
- Statistical significance tests
- Cross-validation analysis
- Publication-ready figures

---

## 📊 What Each Notebook Produces

| Notebook | Runtime | Outputs | Purpose |
|----------|---------|---------|---------|
| 01_setup_and_config | 2 min | Config validation | Environment setup |
| preprocessing_FIXED | 5-10 min | preprocessed_canonical/ | Clean, consistent data |
| 02_train_tumornet_lite | 30-60 min | Checkpoint + metrics | Main results |
| 03_ablation_study | 2-3 hours | Component analysis | Validate architecture |
| 04_baseline_comparison | 3-5 hours | Model comparison | State-of-art comparison |
| 05_results_analysis | 10-20 min | Statistical tests | Publication figures |

---

## ✅ Verification Checklist

**After Step 1 (Setup + Preprocessing):**
- [ ] `preprocessed_canonical/` exists
- [ ] Contains train/, val/, internal_test/, heldout_test/
- [ ] preprocessing_report.txt shows "✓ No data leakage"

**After Step 2 (Main Training):**
- [ ] Checkpoint saved in checkpoints/
- [ ] Training curves show smooth convergence
- [ ] Held-out test accuracy reported
- [ ] All PNG visualizations generated

**After Step 3 (Ablation):**
- [ ] Full model outperforms baseline
- [ ] Results show monotonic improvement
- [ ] Comparison chart saved

**After Step 4 (Baseline Comparison):**
- [ ] All models trained from scratch
- [ ] Fair comparison (same data, same protocol)
- [ ] TumorNet-Lite competitive or better

---

## 🚨 Common Issues & Solutions

### **"Preprocessed data not found"**
→ Run `preprocessing_FIXED.ipynb` first

### **"Checkpoint already exists"**
→ This ensures fresh training - delete old checkpoint or use new name

### **"Out of memory"**
→ Edit `config.yaml`: reduce `batch_size` to 16 or 8

### **"Import error: No module"**
→ Install: `pip install torch torchvision pyyaml scikit-learn matplotlib seaborn tqdm`

### **Results not reproducible**
→ Verify `config.yaml` has `deterministic: true`

---

## 🎓 For Publication

**Minimum required experiments:**
1. ✅ Main training (02_train_tumornet_lite.ipynb)
2. ✅ Ablation study (03_ablation_study.ipynb)
3. ✅ Baseline comparison (04_baseline_comparison.ipynb)

**Report these metrics:**
- Held-out test accuracy (primary metric)
- Per-class precision/recall/F1
- Mean AUC across classes
- Model parameters and inference time
- Statistical significance vs. baselines

**Include these figures:**
- Confusion matrix (held-out test)
- ROC curves (all classes)
- Training curves (loss + accuracy)
- Ablation comparison chart
- Baseline comparison table

---

## 💡 Tips

1. **Always run notebooks in order** - Each depends on previous ones
2. **Save notebooks frequently** - Training can take hours
3. **Monitor GPU memory** - Check with `nvidia-smi`
4. **Use different experiment names** - Timestamp automatically added
5. **Keep raw data unchanged** - Only modify through preprocessing notebook

---

## 📞 Quick Reference

**To restart from scratch:**
```bash
# Delete all outputs
rm -rf ../preprocessed_canonical ../checkpoints ../results

# Run notebooks 1-2 again
```

**To rerun just training:**
```bash
# Keep preprocessing, just retrain
# Delete checkpoints, then run notebook 2
rm ../checkpoints/*.pth
```

**To compare different hyperparameters:**
```bash
# Edit config.yaml, then rerun from notebook 2
```

---

**Ready to start? Open `01_setup_and_config.ipynb` and run all cells!** 🚀

---

**Last Updated:** December 1, 2025  
**Status:** All notebooks created and ready  
**Total Notebooks:** 6 (2 required, 4 optional)
