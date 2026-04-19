# ViT Cardiac Pathology Project Structure

## Directory Layout

```
vit-cardiac-pathology/
│
├── data/
│   ├── raw/
│   │   ├── images/                    # Your CheXchoNet images folder
│   │   ├── metadata.csv               # Metadata CSV file
│   │   └── metadata.txt               # Metadata text file
│   │
│   ├── processed/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels.csv
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── labels.csv
│   │   └── test/
│   │       ├── images/
│   │       └── labels.csv
│   │
│   ├── splits/
│   │   └── stratified_splits.json     # Train/val/test split indices
│   │
│   └── cache/
│       └── (preprocessed data, embeddings, etc.)
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # Configuration & hyperparameters
│   ├── constants.py                   # Dataset constants
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                 # CheXchoNet dataset class
│   │   ├── loader.py                  # DataLoader utilities
│   │   ├── preprocessing.py           # Image preprocessing & augmentation
│   │   └── splits.py                  # Stratified splitting logic
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vit.py                     # Vision Transformer architecture
│   │   └── attention_extractor.py     # Attention weight extraction
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Training loop
│   │   ├── losses.py                  # Loss functions
│   │   ├── metrics.py                 # Evaluation metrics (AUC, etc.)
│   │   └── callbacks.py               # Callbacks for logging, checkpointing
│   │
│   ├── interpretability/
│   │   ├── __init__.py
│   │   ├── attention_viz.py           # Attention visualization
│   │   ├── patch_analysis.py          # Patch-level analysis
│   │   ├── head_specialization.py     # Head specialization detection
│   │   └── anatomy_mapping.py         # Anatomy correlation analysis
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                  # Logging utilities
│       ├── io.py                      # File I/O helpers
│       └── plotting.py                # Visualization utilities
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_stratified_splitting.ipynb
│   ├── 03_data_loading_verification.ipynb
│   └── 04_vit_training.ipynb
│
├── scripts/
│   ├── 01_prepare_dataset.py          # Dataset preparation & splitting
│   ├── 02_train_vit.py                # Training script
│   ├── 03_extract_attention.py        # Attention weight extraction
│   ├── 04_analyze_heads.py            # Head specialization analysis
│   └── 05_generate_visualizations.py  # Generate publication-ready figures
│
├── results/
│   ├── checkpoints/
│   │   └── (model weights and training logs)
│   ├── attention_maps/
│   │   └── (visualized attention patterns)
│   ├── analysis/
│   │   └── (statistical analyses, head specialization results)
│   └── figures/
│       └── (publication-ready visualizations)
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loading.py
│   ├── test_preprocessing.py
│   └── test_model.py
│
├── requirements.txt                   # Python dependencies
├── .gitignore
├── README.md
├── setup.py
└── pyproject.toml
```

## Step-by-Step Setup Instructions

### Phase 1: Data Organization (YOUR CURRENT STEP)

**Goal:** Organize raw data into a clean structure ready for processing.

1. **Create the project folder structure** (we'll generate this next)
2. **Move your data into `data/raw/`:**
   - Copy your `images/` folder → `data/raw/images/`
   - Copy `metadata.csv` → `data/raw/metadata.csv`
   - Copy `metadata.txt` → `data/raw/metadata.txt`

3. **Verify data integrity:**
   - Count images in folder
   - Load CSV and inspect columns
   - Check for missing files/labels

### Phase 2: Stratified Data Splitting (NEXT STEP)

**Goal:** Create reproducible train/val/test splits (60/15/25%) stratified by pathology.

- Script: `01_prepare_dataset.py`
- Output: Organized data in `data/processed/` + split indices in `data/splits/`

### Phase 3: Data Loading Pipeline (STEP 3)

**Goal:** Build robust PyTorch DataLoaders with preprocessing.

- Implement `src/data/dataset.py` (CheXchoNet class)
- Implement `src/data/preprocessing.py` (normalization, augmentation)
- Verify loading in Jupyter notebook

### Phase 4: ViT Training Setup (STEP 4)

**Goal:** Train Vision Transformer baseline.

- Implement `src/models/vit.py` (ViT-Base configuration)
- Implement `src/training/trainer.py` (training loop)
- Run `02_train_vit.py` and reach target AUC (0.87-0.89)

### Phase 5: Attention Analysis (STEP 5)

**Goal:** Extract and analyze attention mechanisms.

- Implement `src/interpretability/` modules
- Extract attention weights for all predictions
- Visualize patch importance and head specialization

### Phase 6: Novel Research & Publication (STEP 6)

**Goal:** Generate novel insights and publication-ready figures.

## Key Configuration Details

### Dataset Constants (to implement)
- Image size: 224×224 pixels
- Patch size: 16 (196 patches total)
- Normalization: ImageNet statistics (or custom CXR statistics)
- Classes: SLVH, DLV, Composite (binary or multi-task)
- Train/Val/Test split: 60/15/25% (stratified by pathology)

### Training Hyperparameters (to confirm)
- Model: ViT-Base (12 layers, 12 heads, 768 dim)
- Learning rate: 1e-4
- Warmup epochs: 5
- Total epochs: 100
- Batch size: 32 (adjust based on GPU memory)
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR

### Hardware Requirements
- GPU: V100 or A100 (8GB+ VRAM)
- Training time: ~6-8 hours
- Storage: ~20-30GB (raw images + checkpoints)

## Python Dependencies (High-Level)

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0              # For ViT models
numpy
pandas
scikit-learn
matplotlib
seaborn
tensorboard
tqdm
```

---

## NEXT STEP: Create Project Folder Structure

Ready for me to generate the actual PyCharm project folders and initial files?
