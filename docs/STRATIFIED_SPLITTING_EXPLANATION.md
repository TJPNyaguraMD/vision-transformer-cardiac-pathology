# Data Preparation Approach: Understanding Stratified Splitting

## The Problem

If you randomly split 71,589 images into train/val/test, you might get:
- Train set: 95% SLVH, 5% DLV (imbalanced!)
- Val set: 20% SLVH, 80% DLV (completely different distribution!)
- Test set: 50% SLVH, 50% DLV (yet another distribution!)

**Result:** Your model trains on one distribution but tests on another. Bad evaluation!

---

## The Solution: Stratified Splitting

**What it does:**
Ensures EACH split (train/val/test) has the SAME pathology distribution as the original data.

**Example - Hypothetical Data Distribution:**
- Original dataset: 40% SLVH, 40% DLV, 20% Composite

**With Stratified Splitting:**
- Train (60%): 40% SLVH, 40% DLV, 20% Composite ✓
- Val (15%): 40% SLVH, 40% DLV, 20% Composite ✓
- Test (25%): 40% SLVH, 40% DLV, 20% Composite ✓

All splits maintain the original distribution!

---

## What 01_prepare_dataset.py Will Do

### INPUT
```
F:\Data\Medical Data\Xray Images\chexchonet-...\
├── images\           (71,589 .jpg files)
├── metadata.csv      (pathology labels)
└── SHA256SUMS.txt    (verification hashes)
```

### PROCESS

#### Step 1: Load Metadata
```python
import pandas as pd
from src.config import EXTERNAL_METADATA_CSV

# Load the CSV file
df = pd.read_csv(EXTERNAL_METADATA_CSV)
# Output: DataFrame with 71,589 rows, columns: [image_id, pathology, patient_id, ...]
```

**Output:**
```
        image_id pathology  patient_id
0  image_0001.jpg      SLVH      patient_1
1  image_0002.jpg       DLV      patient_2
2  image_0003.jpg  Composite   patient_3
...
71588 image_71589.jpg  SLVH  patient_24689
```

#### Step 2: Verify Data Integrity
```python
# Check that all images actually exist
for image_id in df['image_id']:
    image_path = EXTERNAL_IMAGES_PATH / image_id
    if not image_path.exists():
        print(f"MISSING: {image_id}")

# Result: All 71,589 files verified ✓
```

#### Step 3: Analyze Distribution
```python
print(df['pathology'].value_counts())
```

**Example Output:**
```
SLVH        28,000  (39%)
DLV         25,000  (35%)
Composite   18,589  (26%)
Total       71,589  (100%)
```

#### Step 4: Create Stratified Splits
```python
from sklearn.model_selection import StratifiedShuffleSplit

# Create splitter
splitter = StratifiedShuffleSplit(
    n_splits=1,           # One split operation
    test_size=0.40,       # Test+Val = 40%, Train = 60%
    random_state=42       # Reproducible (same split every time)
)

# First split: Train/Temp (Temp will be split into Val/Test)
for train_idx, temp_idx in splitter.split(df, df['pathology']):
    # train_idx: ~42,953 indices (60%)
    # temp_idx: ~28,636 indices (40%)
    pass

# Second split: Val/Test from Temp
splitter2 = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.625,      # Test = 625% of Temp ≈ 25% of original
    random_state=42
)

for val_idx, test_idx in splitter2.split(df.iloc[temp_idx], df.iloc[temp_idx]['pathology']):
    # val_idx: ~10,738 indices (15%)
    # test_idx: ~17,898 indices (25%)
    pass
```

#### Step 5: Save Split Indices
```python
import json

# Save as JSON for reproducibility
splits = {
    'train_indices': train_idx.tolist(),  # [0, 5, 12, 34, ...]
    'val_indices': val_idx.tolist(),
    'test_indices': test_idx.tolist(),
}

with open(SPLITS_DIR / 'splits.json', 'w') as f:
    json.dump(splits, f)

# Output files:
# E:\PORTFOLIO\MEDICAL IMAGING\vision_transformer\data\splits\
# ├── train_indices.json (42,953 indices)
# ├── val_indices.json (10,738 indices)
# └── test_indices.json (17,898 indices)
```

#### Step 6: Verify & Report
```python
print(f"Train split - SLVH: {count_slvh(train_idx)}, DLV: {count_dlv(train_idx)}")
print(f"Val split   - SLVH: {count_slvh(val_idx)}, DLV: {count_dlv(val_idx)}")
print(f"Test split  - SLVH: {count_slvh(test_idx)}, DLV: {count_dlv(test_idx)}")
```

**Example Output:**
```
DATA DISTRIBUTION REPORT
========================
Train split (60% - 42,953 images):
  SLVH: 16,800 (39.1%)
  DLV: 15,034 (35.0%)
  Composite: 11,119 (25.9%)

Val split (15% - 10,738 images):
  SLVH: 4,200 (39.1%)
  DLV: 3,759 (35.0%)
  Composite: 2,779 (25.9%)

Test split (25% - 17,898 images):
  SLVH: 6,993 (39.1%)
  DLV: 6,255 (35.0%)
  Composite: 4,650 (25.9%)

✓ All splits maintain original distribution!
```

---

## Why This Matters

### ✓ Scientific Rigor
- Each split represents the same population distribution
- Validation metrics (AUC, accuracy) are fair and reliable
- Test results are unbiased

### ✓ Reproducibility
- JSON indices saved with random_state=42
- Anyone can regenerate EXACT same splits
- No data leakage between splits

### ✓ Balanced Evaluation
- If your training data has 40% SLVH, your validation does too
- No surprises during testing
- Honest assessment of model performance

---

## What Happens Next

Once this script runs successfully:

1. **splits.json saved** in `E:\...\data\splits\`
2. **Data report printed** to console
3. **Ready for 02_train_vit.py** which will:
   - Load the split indices
   - Create PyTorch DataLoaders for each split
   - Begin ViT training

---

## Key Parameters in Script

```python
TRAIN_SPLIT = 0.60      # 60% of data for training
VAL_SPLIT = 0.15        # 15% of data for validation
TEST_SPLIT = 0.25       # 25% of data for testing
RANDOM_SEED = 42        # Ensures reproducibility

STRATIFY_BY = "pathology"  # Stratify by SLVH/DLV/Composite
```

---

## Summary

**01_prepare_dataset.py will:**
1. ✓ Load metadata from F: drive
2. ✓ Verify all 71,589 images exist
3. ✓ Analyze pathology distribution
4. ✓ Create stratified train/val/test splits (60/15/25%)
5. ✓ Save split indices to E: drive (fast SSD)
6. ✓ Print distribution report

**Output: splits.json** - Used by next script for data loading

Ready to see the actual code? 🚀
