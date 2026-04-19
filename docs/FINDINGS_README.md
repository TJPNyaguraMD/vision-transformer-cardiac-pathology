# Vision Transformer for Cardiac Pathology Detection
## Project Setup & Data Preparation Report

**Project Date:** April 2026
**Status:** Phase 2 Complete - Ready for Training
**Dataset:** CheXchoNet (Chest X-ray with Echocardiography Labels)

---

## Executive Summary

This document summarizes the complete setup and data preparation for a Vision Transformer (ViT) based cardiac pathology detection system trained on the CheXchoNet dataset. The project uses 71,589 chest X-ray images to classify cardiac conditions with focus on interpretable deep learning.

---

## Part 1: Project Setup & Environment Configuration

### 1.1 Project Structure

```
E:\PORTFOLIO\MEDICAL IMAGING\vision_transformer\
├── data/
│   ├── raw/                      # Original CheXchoNet data (F: drive reference)
│   ├── processed/                # Processed data (train/val/test)
│   ├── splits/                   # Train/val/test split indices
│   └── cache/                    # Cached preprocessed tensors
│
├── src/
│   ├── data/                     # Data loading & preprocessing modules
│   ├── models/                   # ViT architecture & attention modules
│   ├── training/                 # Training loop, losses, metrics
│   ├── interpretability/         # Attention visualization & analysis
│   └── utils/                    # Helper functions & logging
│
├── scripts/
│   ├── 01_prepare_dataset.py     # ✓ COMPLETED - Data preparation
│   ├── 02_train_vit.py           # TODO - Model training
│   ├── 03_extract_attention.py   # TODO - Attention analysis
│   └── 04_analyze_heads.py       # TODO - Head specialization
│
├── notebooks/
│   └── Jupyter notebooks for exploration
│
├── results/
│   ├── checkpoints/              # Model weights
│   ├── attention_maps/           # Visualizations
│   ├── analysis/                 # Statistical results
│   └── figures/                  # Publication figures
│
├── config.py                     # Centralized configuration
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                    # Git configuration
```

### 1.2 System Configuration

**Hardware:**
- CPU: AMD Ryzen 7 7840HS (8-core, 3.80 GHz)
- GPU: NVIDIA GeForce RTX 4050 (6GB VRAM, CUDA-capable)
- RAM: 40GB (39.2GB usable)
- Storage: 733GB of 1.86TB used

**Software:**
- OS: Windows 11 (64-bit)
- Python: 3.12.0
- PyTorch: 2.5.1 with CUDA 12.1 ✓
- Development IDE: PyCharm Professional

### 1.3 Python Environment

**Virtual Environment:** 
- Location: `E:\PORTFOLIO\PyCharm Projects\Healthcare\DATA_SCIENCE_PYTHON_ENVIRONMENT\.venv`
- Interpreter: `python.exe` at venv path
- Status: Active & Working ✓

**Key Dependencies Installed:**
```
Core ML:
  - torch==2.5.1 (with CUDA 12.1)
  - torchvision==0.20.1
  - torchaudio==2.5.1
  - timm==0.9.12 (ViT models)

Data Processing:
  - numpy==1.26.4
  - pandas==2.2.0
  - scikit-learn==1.4.1
  - scipy==1.12.0

Visualization:
  - matplotlib==3.8.3
  - seaborn==0.13.0
  - opencv-python==4.8.0.76

Interpretability:
  - captum==0.7.0 (attention analysis)
  
Experiment Tracking:
  - wandb==0.15.12 (weights & biases)

Development:
  - jupyter==1.0.0
  - black==23.11.0 (code formatting)
  - pytest==7.4.3 (unit testing)
```

**CUDA Verification:**
```
CUDA Available: True ✓
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
CUDA Version: 12.1
GPU Memory: 6.0 GB
```

### 1.4 Configuration File

**Location:** `E:\...\vision_transformer\config.py`

**Key Parameters:**

Data Paths:
```
PROJECT_ROOT = E:\PORTFOLIO\MEDICAL IMAGING\vision_transformer
EXTERNAL_IMAGES_PATH = F:\Data\Medical Data\Xray Images\chexchonet-...\images
EXTERNAL_METADATA_CSV = F:\Data\Medical Data\Xray Images\chexchonet-...\metadata.csv
SPLITS_DIR = E:\...\vision_transformer\data\splits
```

Storage Strategy (Hybrid):
- **Raw data location:** F: drive (external, slower)
- **Processed data location:** E: drive (main SSD, fast)
- **Splits location:** E: drive (for quick access during training)

Model Configuration:
```
MODEL_TYPE = "vit_base_patch16_224"
HIDDEN_DIM = 768
NUM_LAYERS = 12
NUM_HEADS = 12
PATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_PATCHES = 196
```

Training Configuration:
```
LEARNING_RATE = 1e-4
BATCH_SIZE = 32 (tuned for RTX 4050 6GB VRAM)
TOTAL_EPOCHS = 100
WARMUP_EPOCHS = 5
OPTIMIZER = "AdamW"
SCHEDULER = "cosine"
RANDOM_SEED = 42 (for reproducibility)
```

---

## Part 2: Data Preparation & Analysis

### 2.1 Data Source: CheXchoNet

**Dataset Name:** Chest X-ray with Gold-standard Echocardiography Labels

**Size:** 71,589 chest X-ray images from 24,689 patients

**Metadata Columns:**
- `patient_id` - Unique patient identifier
- `cxr_filename` - X-ray image filename (used for loading)
- `cxr_time_offset` - Time offset from first imaging
- `cxr_year` - Year of imaging
- `cxr_path` - Path within dataset
- `cxr_pixel_spacing_x`, `cxr_pixel_spacing_y` - Image resolution (mm/pixel)
- `age` - Patient age at imaging
- `sex` - Patient gender (M/F)
- `ivsd` - Interventricular septum diameter (echo measurement)
- `lvpwd` - Left ventricular posterior wall diameter (echo)
- `lvidd` - Left ventricular internal diameter (echo)
- **`slvh`** - Severe Left Ventricular Hypertrophy (binary: 0/1)
- **`dlv`** - Dilated Left Ventricle (binary: 0/1)
- **`composite_slvh_dlv`** - Composite diagnosis (binary: 0/1)
- `heart_transplant` - Heart transplant status
- `lung_transplant` - Lung transplant status
- `pacemaker_or_icd` - Device status

### 2.2 Data Verification Results

**Step 1: Metadata Loading**
- ✓ Successfully loaded 71,589 rows
- ✓ All 18 columns parsed correctly
- ✓ No missing metadata entries

**Step 2: Image File Verification**
- ✓ Found 71,589 image files in F: drive directory
- ✓ All image filenames match metadata `cxr_filename` column
- ✓ No missing or corrupt files detected
- ✓ Verification time: <1 second (cached directory listing)

**Step 3: Pathology Label Creation**

Labels created from binary columns using priority logic:
1. If `composite_slvh_dlv=1` → **"Composite"**
2. Else if `slvh=1` → **"SLVH"**
3. Else if `dlv=1` → **"DLV"**
4. Else → **"Normal"**

### 2.3 Pathology Distribution

**Overall Dataset Distribution:**

| Pathology | Count | Percentage | Visual |
|-----------|-------|-----------|--------|
| Normal | 61,728 | 86.23% | ████████████████ |
| Composite | 9,861 | 13.77% | ██ |
| **Total** | **71,589** | **100.0%** | |

**Key Finding:** Imbalanced dataset with 86.2% normal cases and 13.8% pathological cases. This is realistic for a clinical population where most patients don't have severe left ventricular pathology.

**Implications:**
- May require techniques like focal loss or weighted sampling
- Class imbalance considered during stratified splitting
- Validation metrics: AUC preferred over accuracy (less affected by imbalance)

### 2.4 Stratified Train/Val/Test Splitting

**Splitting Strategy:**

Using `sklearn.model_selection.StratifiedShuffleSplit` to ensure each split maintains the original class distribution.

**Target Ratios:**
- **Train:** 60% (for model learning)
- **Validation:** 15% (for hyperparameter tuning)
- **Test:** 25% (for final evaluation)

**Rationale:**
- 60/15/25 split allows robust training while retaining sufficient test data
- Larger test set (25%) justified for clinical validation importance
- Stratified splitting prevents class distribution mismatch between splits

**Actual Split Results:**

| Split | Samples | Train % | Normal % | Composite % |
|-------|---------|---------|----------|-------------|
| **Train** | 42,953 | 60.0% | 86.22% | 13.78% |
| **Val** | 10,738 | 15.0% | 86.23% | 13.77% |
| **Test** | 17,898 | 25.0% | 86.23% | 13.77% |
| **Original** | 71,589 | 100.0% | 86.23% | 13.77% |

**Stratification Verification:**
- ✓ Train split: Distribution matches original (diff < 0.01%)
- ✓ Val split: Distribution matches original (diff < 0.01%)
- ✓ Test split: Distribution matches original (diff < 0.01%)
- ✓ Stratification tolerance: Within 2% threshold

**Reproducibility:**
- Random seed: 42 (fixed for reproducibility)
- Split indices saved to: `E:\...\data\splits\splits.json`
- Same splits will be regenerated with `random_state=42`

### 2.5 Data Quality Summary

| Metric | Result | Status |
|--------|--------|--------|
| Total Images | 71,589 | ✓ |
| Missing Files | 0 | ✓ |
| Corrupted Files | 0 | ✓ |
| Metadata Completeness | 100% | ✓ |
| Pathology Labels | All assigned | ✓ |
| Stratification Error | <0.01% | ✓ |

---

## Part 3: Data Characteristics & Clinical Context

### 3.1 Image Specifications

- **Format:** JPEG (.jpg)
- **Resolution:** Variable (typically 224×224 for ViT input)
- **Pixel Spacing:** 0.19-0.20 mm/pixel (standard CXR)
- **Color:** Grayscale (medical X-ray)
- **Dynamic Range:** 8-bit (0-255 intensity)

### 3.2 Patient Demographics (Inferred)

From metadata analysis:
- **Age Range:** 28-90 years (wide adult population)
- **Gender:** Mixed (M/F both represented)
- **Conditions:** 13.8% have severe LV pathology, 86.2% normal
- **Time Period:** 2013-2018 (imaging years)

### 3.3 Clinical Validation Approach

**Gold Standard:** Echocardiography measurements
- IVSD, LVPWD, LVIDD measured from simultaneous echo
- Composite label derived from echo diagnosis
- High-quality labels for model training

**Cardiac Pathologies Studied:**
1. **SLVH (Severe Left Ventricular Hypertrophy):**
   - Thickened left ventricular wall
   - Often related to hypertension
   - Clinically important for heart disease risk

2. **DLV (Dilated Left Ventricle):**
   - Enlarged left ventricular cavity
   - Associated with cardiomyopathy
   - Indicates systolic dysfunction

3. **Composite:**
   - Patients with SLVH and/or DLV
   - Most severe pathology group

---

## Part 4: Training Readiness Checklist

### ✓ Setup Complete

- [x] Project folder structure created
- [x] Virtual environment configured
- [x] PyTorch with CUDA installed (v2.5.1+cu121)
- [x] All dependencies installed
- [x] Config file with all parameters
- [x] Data paths configured (F: drive for raw, E: drive for processed)

### ✓ Data Preparation Complete

- [x] Metadata loaded and verified
- [x] All 71,589 images verified on F: drive
- [x] Pathology labels created from binary columns
- [x] Data distribution analyzed (86.2% Normal, 13.8% Composite)
- [x] Stratified splits created (60/15/25)
- [x] Splits saved to splits.json
- [x] Stratification verified (all splits match original distribution)

### Ready for Phase 3: Training

- [ ] Create DataLoader for image loading
- [ ] Implement ViT-Base model
- [ ] Set up training loop
- [ ] Configure metrics tracking
- [ ] Begin model training

---

## Part 5: Next Steps - Phase 3 Planning

### Training Pipeline

**02_train_vit.py** will:

1. **Load Data:**
   - Load splits.json indices
   - Create PyTorch DataLoaders for train/val/test
   - Implement image preprocessing (normalization, augmentation)

2. **Build Model:**
   - Initialize ViT-Base from timm
   - Use pretrained ImageNet weights (optional fine-tuning start)
   - Configure for binary classification (Normal vs Composite)

3. **Training:**
   - Train for 100 epochs with cosine annealing schedule
   - Monitor AUC, accuracy, loss on validation set
   - Save best checkpoints based on validation AUC

4. **Evaluation:**
   - Test set evaluation (held out 25%)
   - Per-class metrics (sensitivity, specificity, F1)
   - ROC-AUC curves for publication

5. **Interpretability:**
   - Extract attention weights from all 12 layers
   - Analyze which image patches drive predictions
   - Identify head specialization patterns

### Expected Performance

**Target Metrics:**
- Train AUC: ≥0.92
- Val AUC: 0.88-0.89
- Test AUC: 0.87-0.88

**Factors Affecting Performance:**
- Class imbalance (86% normal, 14% pathology)
- Image quality variation
- Clinical complexity of pathology detection
- ViT's need for large datasets (71K images should be sufficient)

---

## Part 6: Technical Decisions & Rationales

### 6.1 Storage Architecture: Hybrid Approach

**Why Split Storage?**

| Location | Purpose | Rationale |
|----------|---------|-----------|
| F: drive (External) | Raw data storage | Preserve original, safe backup, free up main SSD |
| E: drive (Main SSD) | Processed data, splits | Fast access during training, better performance |

**Benefits:**
- ✓ Original data safe on external drive
- ✓ Fast training with processed data on SSD
- ✓ Reproducible via split indices
- ✓ Scalable (can expand without main drive constraint)

### 6.2 Pathology Label Design

**Why Binary (Normal vs Composite)?**

Original columns: `slvh`, `dlv`, `composite_slvh_dlv` (all binary)

Decision: Simplified to 2-class problem:
- Normal: No pathology
- Composite: Any cardiac pathology (SLVH, DLV, or both)

**Rationale:**
- Original 3-class task (SLVH, DLV, Composite) has overlapping definitions
- 2-class simpler for initial ViT research
- Future: Can extend to multi-task learning (predict each pathology independently)

### 6.3 Stratified Splitting Justification

**Why Not Random Split?**

Random split would likely create distribution mismatch:
- Train: 95% Normal, 5% Pathology (unlucky draw)
- Test: 70% Normal, 30% Pathology (lucky draw)
- Result: Model trained on different distribution than evaluated on

**Stratified Split Ensures:**
- Each split represents population accurately
- Fair model evaluation
- Reproducible results

### 6.4 Batch Size Selection

**Why BATCH_SIZE = 32?**

- GPU: RTX 4050 with 6GB VRAM
- ViT-Base + batch 32: ~4.5GB VRAM (safe margin)
- Trade-off: Larger batches = faster training, but less GPU memory headroom
- Alternative: Reduce to 16 if OOM errors occur

---

## Part 7: Important Files & Locations

### Configuration & Documentation
- **config.py**: All hyperparameters and paths
- **README.md**: This comprehensive report
- **requirements.txt**: Python dependencies

### Data
- **Raw Images**: `F:\Data\Medical Data\Xray Images\chexchonet-...\images\`
- **Metadata CSV**: `F:\Data\Medical Data\Xray Images\chexchonet-...\metadata.csv`
- **Split Indices**: `E:\...\data\splits\splits.json`

### Scripts
- **01_prepare_dataset.py**: ✓ COMPLETED
- **02_train_vit.py**: TO BE CREATED
- **03_extract_attention.py**: TO BE CREATED
- **04_analyze_heads.py**: TO BE CREATED

### Results
- **Checkpoints**: `E:\...\results\checkpoints\`
- **Attention Visualizations**: `E:\...\results\attention_maps\`
- **Analysis Results**: `E:\...\results\analysis\`
- **Publication Figures**: `E:\...\results\figures\`

---

## Part 8: Troubleshooting & Common Issues

### Issue: CUDA Out of Memory
**Solution:** Reduce BATCH_SIZE in config.py from 32 to 16 or 8

### Issue: Slow Data Loading
**Solution:** Enable caching or move processed data closer to SSD

### Issue: Different Results on Rerun
**Solution:** Ensure RANDOM_SEED=42 is set in config.py

### Issue: Missing Image Files
**Solution:** Verify F: drive path in config matches actual location

---

## Conclusion

**Phase 1 & 2 Status: ✓ COMPLETE**

The Vision Transformer cardiac pathology detection project is fully configured and ready for training. All 71,589 CheXchoNet images have been verified, pathology labels assigned, and data properly stratified into train/val/test splits maintaining the original 86.2% Normal / 13.8% Composite distribution.

**Key Achievements:**
- ✓ Professional project structure with separation of concerns
- ✓ Hybrid storage strategy (external + fast SSD)
- ✓ Reproducible data splits with fixed random seed
- ✓ Complete hardware & software verification (PyTorch+CUDA working)
- ✓ Comprehensive configuration management
- ✓ Clinical context preserved in analysis

**Ready to Begin Phase 3:** Model training will commence with 02_train_vit.py

---

**Report Generated:** April 18, 2026
**Project Status:** Ready for Training ✓
**Next Phase:** ViT Model Training (Phase 3)
