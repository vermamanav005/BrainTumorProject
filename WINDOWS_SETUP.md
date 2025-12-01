# 🪟 Windows 11 Setup Guide

**Complete setup instructions for running this project on Windows 11**

---

## ⚙️ Prerequisites

### 1. Python Installation
```cmd
REM Check if Python is installed
python --version

REM Should show Python 3.8 or higher
REM If not installed, download from: https://www.python.org/downloads/
```

**Important:** During Python installation, check "Add Python to PATH"

### 2. Git Installation (Optional)
Download from: https://git-scm.com/download/win

---

## 🚀 Quick Setup

### Option 1: Command Prompt
```cmd
REM 1. Navigate to project directory
cd C:\path\to\BrainTumorProject

REM 2. Go to working directory
cd tumorNet_lite

REM 3. Install dependencies
pip install -r requirements.txt

REM 4. Start Jupyter
jupyter notebook
```

### Option 2: PowerShell
```powershell
# 1. Navigate to project directory
Set-Location C:\path\to\BrainTumorProject

# 2. Go to working directory
Set-Location tumorNet_lite

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Jupyter
jupyter notebook
```

---

## 📝 Configuration for Windows

### Edit `config.yaml`

**Important Windows-specific settings:**

```yaml
paths:
  # Use forward slashes (/) even on Windows - Python handles it!
  # Option 1: Absolute path
  project_root: "C:/Users/YourName/BrainTumorProject"
  
  # Option 2: Relative path (recommended)
  project_root: "."

training:
  # Windows MUST use 0 workers (multiprocessing issues)
  num_workers: 0
  
  # Use GPU if available (requires CUDA installed)
  device: "cuda"  # or "cpu" if no GPU
```

---

## 🐍 Virtual Environment (Recommended)

### Using venv (Built-in)

**Command Prompt:**
```cmd
REM Create virtual environment
python -m venv venv

REM Activate it
venv\Scripts\activate

REM Install dependencies
pip install -r requirements.txt

REM Deactivate when done
deactivate
```

**PowerShell:**
```powershell
# Create virtual environment
python -m venv venv

# Activate it (may need to enable scripts first)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

### Using Conda (Alternative)
```cmd
REM Create conda environment
conda create -n tumor_net python=3.9

REM Activate it
conda activate tumor_net

REM Install dependencies
pip install -r requirements.txt

REM Deactivate when done
conda deactivate
```

---

## 🎮 GPU Setup (Optional but Recommended)

### Check if CUDA is available:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Install CUDA PyTorch (if not working):
Visit: https://pytorch.org/get-started/locally/

**Example for CUDA 11.8:**
```cmd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Example for CPU only:**
```cmd
pip3 install torch torchvision torchaudio
```

---

## 📁 Path Handling

### Windows Path Examples

**Absolute paths (forward slashes work!):**
```yaml
project_root: "C:/Users/YourName/Documents/BrainTumorProject"
raw_data: "C:/Users/YourName/Documents/BrainTumorProject/raw_data/Brain_Tumor_MRI_Dataset"
```

**Relative paths (recommended):**
```yaml
project_root: "."
raw_data: "raw_data/Brain_Tumor_MRI_Dataset"
preprocessed_data: "preprocessed_canonical"
```

### In Python/Notebooks

Python's `pathlib` and `os.path` handle Windows paths automatically:
```python
from pathlib import Path

# These all work on Windows:
path1 = Path("C:/Users/Name/file.txt")
path2 = Path("C:\\Users\\Name\\file.txt")
path3 = Path.home() / "Documents" / "file.txt"

# Convert to Windows path string:
str(path1)  # Automatically uses backslashes on Windows
```

---

## 🐛 Common Windows Issues

### Issue 1: "num_workers > 0 causes freeze"
**Solution:** Set `num_workers: 0` in `config.yaml`
```yaml
training:
  num_workers: 0  # Must be 0 on Windows
```

### Issue 2: "PowerShell script execution disabled"
**Solution:** Enable script execution
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue 3: "CUDA out of memory"
**Solutions:**
```yaml
# In config.yaml:
training:
  batch_size: 16  # Reduce from 32
  mixed_precision:
    enabled: true  # Enable for memory savings
```

### Issue 4: "FileNotFoundError with paths"
**Solution:** Use forward slashes (/) not backslashes (\\)
```python
# Good (works on all platforms)
path = "data/images/file.jpg"

# Bad (Windows-specific, can cause issues)
path = "data\\images\\file.jpg"
```

### Issue 5: "Permission denied when deleting files"
**Solution:** Close any programs using the files (Jupyter, Python)
```cmd
REM Force delete if needed
rmdir /s /q folder_name
```

### Issue 6: "Module 'cv2' not found"
**Solution:** Install OpenCV
```cmd
pip install opencv-python
```

---

## 🔥 Performance Tips for Windows

1. **Use GPU if available**
   - Install CUDA toolkit
   - Install CUDA-enabled PyTorch
   - Set `device: "cuda"` in config

2. **Optimize data loading**
   - Set `num_workers: 0` (required)
   - Set `pin_memory: true` if using GPU
   - Use SSD for data storage

3. **Close background apps**
   - More RAM available = larger batch sizes
   - Close browser tabs, other apps

4. **Use mixed precision training**
   ```yaml
   training:
     mixed_precision:
       enabled: true  # 2-3x faster on modern GPUs
   ```

---

## 📊 Verify Installation

Run this in Python/Jupyter to verify everything works:

```python
# Test imports
import torch
import torchvision
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("✓ All packages imported successfully!")

# Check PyTorch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Check paths
config_path = Path("../config.yaml")
print(f"\nConfig exists: {config_path.exists()}")

print("\n✓ Setup verification complete!")
```

---

## 📂 Expected Directory Structure

After setup, your project should look like:

```
C:\path\to\BrainTumorProject\
├── tumorNet_lite\              ← Your working directory
│   ├── 01_setup_and_config.ipynb
│   ├── 02_train_tumornet_lite.ipynb
│   ├── 03_ablation_study.ipynb
│   ├── 04_baseline_comparison.ipynb
│   ├── preprocessing_FIXED.ipynb
│   ├── utils.py
│   ├── models.py
│   └── requirements.txt
│
├── config.yaml                 ← Configuration file
├── raw_data\                   ← Your raw MRI images
│   └── Brain_Tumor_MRI_Dataset\
│       ├── Training\
│       └── Testing\
│
├── preprocessed_canonical\     ← Created by preprocessing
├── checkpoints\                ← Model checkpoints
├── results\                    ← Results and plots
└── logs\                       ← Training logs
```

---

## 🎯 Next Steps

1. ✅ Verify Python installation
2. ✅ Install dependencies (`pip install -r requirements.txt`)
3. ✅ Update `config.yaml` with your paths
4. ✅ Run verification script above
5. ✅ Open `START_HERE.md` for workflow
6. ✅ Start Jupyter: `jupyter notebook`
7. ✅ Run `01_setup_and_config.ipynb`

---

## 📞 Quick Commands Reference

### Command Prompt
```cmd
cd tumorNet_lite                    # Navigate to working directory
pip install -r requirements.txt    # Install dependencies
jupyter notebook                    # Start Jupyter
python -m venv venv                # Create virtual environment
venv\Scripts\activate              # Activate virtual environment
```

### PowerShell
```powershell
Set-Location tumorNet_lite         # Navigate to working directory
pip install -r requirements.txt    # Install dependencies
jupyter notebook                    # Start Jupyter
python -m venv venv                # Create virtual environment
.\venv\Scripts\Activate.ps1        # Activate virtual environment
```

---

## 🆘 Need Help?

1. Check `START_HERE.md` - Quick reference
2. Check `NOTEBOOK_EXECUTION_ORDER.md` - Detailed workflow
3. Check `BUGS_IDENTIFIED.md` - Known issues and fixes
4. Check PyTorch docs: https://pytorch.org/
5. Check Windows-Python guide: https://docs.python.org/3/using/windows.html

---

**Last Updated:** December 1, 2025  
**Tested on:** Windows 11  
**Python:** 3.8+  
**Status:** ✅ Ready for Windows!
