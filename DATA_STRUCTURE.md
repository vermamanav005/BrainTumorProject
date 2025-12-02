# 📁 Data Folder Structure - IMPORTANT

**Your data path:** `C:\Users\manav\Documents\GitHub\BrainTumorProject\BrainDataSet`

---

## ✅ Required Folder Structure

Your `BrainDataSet` folder **must** have this exact structure:

```
C:\Users\manav\Documents\GitHub\BrainTumorProject\
└── BrainDataSet\
    ├── Training\
    │   ├── glioma\
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── meningioma\
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── notumor\
    │   │   ├── image1.jpg
    │   │   └── ...
    │   └── pituitary\
    │       ├── image1.jpg
    │       └── ...
    │
    └── Testing\
        ├── glioma\
        │   ├── image1.jpg
        │   └── ...
        ├── meningioma\
        │   ├── image1.jpg
        │   └── ...
        ├── notumor\
        │   ├── image1.jpg
        │   └── ...
        └── pituitary\
            ├── image1.jpg
            └── ...
```

---

## 🔍 Verify Your Data Structure

### Option 1: File Explorer
1. Open `C:\Users\manav\Documents\GitHub\BrainTumorProject\BrainDataSet`
2. Check you see:
   - `Training` folder
   - `Testing` folder
3. Inside each, check you see:
   - `glioma` folder
   - `meningioma` folder
   - `notumor` folder
   - `pituitary` folder

### Option 2: Command Prompt
```cmd
cd C:\Users\manav\Documents\GitHub\BrainTumorProject\BrainDataSet
dir /B
```
Should show:
```
Testing
Training
```

Then check Training:
```cmd
dir Training /B
```
Should show:
```
glioma
meningioma
notumor
pituitary
```

### Option 3: Python Script
Run this in Python/Jupyter to verify:

```python
from pathlib import Path

data_path = Path("C:/Users/manav/Documents/GitHub/BrainTumorProject/BrainDataSet")

print("Checking data structure...\n")

# Check main folders
for split in ['Training', 'Testing']:
    split_path = data_path / split
    if split_path.exists():
        print(f"✓ {split}/ exists")
        
        # Check class folders
        for class_name in ['glioma', 'meningioma', 'notumor', 'pituitary']:
            class_path = split_path / class_name
            if class_path.exists():
                num_images = len(list(class_path.glob('*.jpg'))) + \
                           len(list(class_path.glob('*.png')))
                print(f"  ✓ {class_name}/ - {num_images} images")
            else:
                print(f"  ✗ {class_name}/ - MISSING!")
    else:
        print(f"✗ {split}/ - MISSING!")

print("\nIf all folders show ✓, you're ready to go!")
```

---

## ⚠️ Common Issues

### Issue: "Data not found" error
**Cause:** Folder structure doesn't match expected layout

**Solution:** 
1. Check folder names are exactly: `Training` and `Testing` (capital T)
2. Check class folders are exactly: `glioma`, `meningioma`, `notumor`, `pituitary` (lowercase)
3. Ensure images are inside class folders, not in Training/Testing directly

### Issue: "No images found"
**Cause:** Images are in wrong location

**Solution:**
Images should be:
```
BrainDataSet/Training/glioma/image1.jpg  ✓ Correct
BrainDataSet/Training/image1.jpg         ✗ Wrong
BrainDataSet/glioma/image1.jpg          ✗ Wrong
```

### Issue: "Class mismatch" error
**Cause:** Folder names don't match config

**Solution:**
In `config.yaml`, class names must match folder names EXACTLY:
```yaml
data:
  class_names: ['glioma', 'meningioma', 'notumor', 'pituitary']
```

---

## 📊 Expected Image Counts

Typical dataset sizes (adjust based on your data):
- **Training:** 2,500-3,000 images total
  - glioma: ~900 images
  - meningioma: ~900 images
  - pituitary: ~900 images
  - notumor: ~500 images

- **Testing:** 300-500 images total (held-out test set)

---

## 🔧 After Verifying Structure

Once your data structure is correct:

1. ✅ Data is at: `C:\Users\manav\Documents\GitHub\BrainTumorProject\BrainDataSet`
2. ✅ Structure matches above layout
3. ✅ Config file updated (already done)
4. ✅ Ready to run preprocessing!

**Next step:** Run `preprocessing_FIXED.ipynb` to create preprocessed data.

---

## 📝 Config File Settings (Already Set)

Your `config.yaml` is configured with:

```yaml
paths:
  project_root: "C:/Users/manav/Documents/GitHub/BrainTumorProject"
  raw_data: "C:/Users/manav/Documents/GitHub/BrainTumorProject/BrainDataSet"
  preprocessed_data: "C:/Users/manav/Documents/GitHub/BrainTumorProject/preprocessed_canonical"
```

**Note:** Use forward slashes (/) in config, even on Windows. Python handles conversion automatically.

---

**Status:** ✅ Configuration updated for your data path!  
**Next:** Verify folder structure, then run notebooks!
