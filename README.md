# Vision Transformer for Cardiac Pathology Detection

## Project Overview

This research project applies Vision Transformers (ViT) to detect cardiac pathologies from chest X-rays using the CheXchoNet dataset. The focus is on **interpretable AI** — understanding what the model attends to and why it makes specific predictions.

### Key Research Contributions
- Comprehensive ViT application to cardiac X-ray pathology
- Novel patch-level and head-level attention analysis
- Discovery of head specialization across different pathologies
- Clinical validation of ViT interpretability vs CNN methods

### Dataset
- **CheXchoNet**: 71,589 chest X-rays from 24,689 patients
- **Gold-standard labels**: Echocardiography validation
- **Binary tasks**: SLVH, DLV, Composite
- **Image size**: 224×224 pixels
- **Source**: PhysioNet (requires account)

---

## Project Structure

```
vision_transformer/
├── data/                          # Data storage
│   ├── raw/                       # Original CheXchoNet data
│   │   ├── images/               # CXR images
│   │   ├── metadata.csv          # Labels & metadata
│   │   └── metadata.txt          # Additional info
│   ├── processed/                # Train/val/test splits
│   ├── splits/                   # Split indices (JSON)
│   └── cache/                    # Cached preprocessed data
│
├── src/                           # Source code
│   ├── data/                      # Data loading & preprocessing
│   ├── models/                    # ViT architecture
│   ├── training/                  # Training loops & metrics
│   ├── interpretability/          # Attention analysis tools
│   └── utils/                     # Helper functions
│
├── scripts/                       # Executable scripts
│   ├── 01_prepare_dataset.py     # Data splitting
│   ├── 02_train_vit.py           # Training
│   ├── 03_extract_attention.py   # Attention extraction
│   ├── 04_analyze_heads.py       # Head specialization
│   └── 05_generate_visualizations.py
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_stratified_splitting.ipynb
│   ├── 03_data_loading_verification.ipynb
│   └── 04_vit_training.ipynb
│
├── results/                       # Model outputs
│   ├── checkpoints/              # Model weights
│   ├── attention_maps/           # Visualized attention
│   ├── analysis/                 # Statistical results
│   └── figures/                  # Publication figures
│
├── tests/                         # Unit tests
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md (this file)
```

---

## Setup Instructions

### 1. Create Project Structure

Run the setup script from **Windows Command Prompt** or **PowerShell**:

```bash
# Navigate to project directory
cd E:\PORTFOLIO\MEDICAL IMAGING\vision_transformer

# Run setup script (make sure setup_project.bat is in this directory)
setup_project.bat
```

This will create all necessary folders and `__init__.py` files.

### 2. Move Your Data

After running the setup script:

```
Copy your CheXchoNet data to:
- E:\PORTFOLIO\MEDICAL IMAGING\vision_transformer\data\raw\images\
- E:\PORTFOLIO\MEDICAL IMAGING\vision_transformer\data\raw\metadata.csv
- E:\PORTFOLIO\MEDICAL IMAGING\vision_transformer\data\raw\metadata.txt
```

### 3. Set Up Python Virtual Environment

Open **Command Prompt** and run:

```bash
# Navigate to project
cd E:\PORTFOLIO\MEDICAL IMAGING\vision_transformer

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure PyCharm

1. Open PyCharm
2. **File → Open** → Select `E:\PORTFOLIO\MEDICAL IMAGING\vision_transformer`
3. **File → Settings → Project → Python Interpreter**
4. Click gear icon → **Add**
5. Select **Existing Environment**
6. Navigate to: `E:\PORTFOLIO\MEDICAL IMAGING\vision_transformer\venv\Scripts\python.exe`
7. Click **OK**

---

## Research Pipeline

### Phase 1: Data Preparation (Current Phase)
- ✅ Organize raw data
- ⬜ Create stratified train/val/test splits (60/15/25%)
- ⬜ Implement data loading pipeline
- ⬜ Verify data integrity

**Next Step**: Run `scripts/01_prepare_dataset.py`

### Phase 2: Model Training
- ⬜ Train ViT-Base from scratch
- ⬜ Track metrics (AUC, accuracy, loss)
- ⬜ Save checkpoints
- **Target**: AUC ≥ 0.87

**Script**: `scripts/02_train_vit.py`

### Phase 3: Attention Analysis
- ⬜ Extract patch-level attention weights
- ⬜ Visualize attention patterns
- ⬜ Analyze head specialization
- ⬜ Correlate with cardiac anatomy

**Script**: `scripts/03_extract_attention.py`

### Phase 4: Novel Research & Insights
- ⬜ Discover head specialization patterns
- ⬜ Test clinical outcome prediction
- ⬜ Compare ViT vs CNN interpretability
- ⬜ Generate publication figures

**Script**: `scripts/04_analyze_heads.py`

### Phase 5: Validation & Publication
- ⬜ Radiologist validation study
- ⬜ Faithfulness testing (deletion/insertion)
- ⬜ Write manuscript
- ⬜ Submit to Medical Image Analysis

---

## Key Hyperparameters

### Model Configuration
```python
model_type: "vit_base_patch16_224"
num_heads: 12
hidden_dim: 768
num_layers: 12
patch_size: 16
image_size: 224
```

### Training Configuration
```python
learning_rate: 1e-4
batch_size: 32
num_epochs: 100
warmup_epochs: 5
weight_decay: 0.01
optimizer: "AdamW"
scheduler: "CosineAnnealingLR"
```

### Dataset Configuration
```python
train_split: 0.60
val_split: 0.15
test_split: 0.25
stratify_by: "pathology"  # SLVH, DLV, Composite
normalize: "ImageNet"     # or custom CXR statistics
```

---

## Expected Results

| Metric | Target | Notes |
|--------|--------|-------|
| **Train AUC** | 0.92+ | Should show good fit |
| **Val AUC** | 0.88-0.89 | Generalization target |
| **Test AUC** | 0.87-0.88 | Final performance |
| **Head Specialization** | 3-5 distinct patterns | Different heads focus on different anatomy |
| **Patch Correlation** | r > 0.7 with anatomy | Attention aligns with cardiac regions |

---

## Dependencies

**Core Libraries**:
- PyTorch 2.1.2 + CUDA support
- torchvision 0.16.2
- timm 0.9.12 (ViT models from Meta)

**Data & Processing**:
- NumPy, Pandas, scikit-learn

**Visualization**:
- Matplotlib, Seaborn, Pillow

**Training**:
- TensorBoard (monitoring), tqdm (progress bars)

See `requirements.txt` for complete list with versions.

---

## Usage Examples

### Running the Pipeline

```bash
# Activate virtual environment
venv\Scripts\activate

# Step 1: Prepare dataset
python scripts/01_prepare_dataset.py

# Step 2: Train ViT
python scripts/02_train_vit.py --epochs 100 --batch_size 32

# Step 3: Extract attention weights
python scripts/03_extract_attention.py --checkpoint results/checkpoints/best_model.pth

# Step 4: Analyze head specialization
python scripts/04_analyze_heads.py

# Step 5: Generate publication figures
python scripts/05_generate_visualizations.py
```

### Using Jupyter Notebooks

```bash
# Activate environment
venv\Scripts\activate

# Launch Jupyter
jupyter notebook

# Open notebooks in order:
# 1. 01_exploratory_data_analysis.ipynb
# 2. 02_stratified_splitting.ipynb
# 3. 03_data_loading_verification.ipynb
# 4. 04_vit_training.ipynb
```

---

## Development Workflow

1. **Create a branch** for new features:
   ```bash
   git checkout -b feature/attention-visualization
   ```

2. **Write code** following PEP 8 standards

3. **Test** your code:
   ```bash
   python -m pytest tests/
   ```

4. **Commit** with meaningful messages:
   ```bash
   git add .
   git commit -m "Add patch-level attention analysis"
   ```

5. **Push** and create pull request

---

## Troubleshooting

### Issue: Virtual environment not found
**Solution**: Ensure you're in the project directory when creating venv:
```bash
cd E:\PORTFOLIO\MEDICAL IMAGING\vision_transformer
python -m venv venv
```

### Issue: CUDA out of memory
**Solution**: Reduce batch size in config:
```python
batch_size: 16  # or 8 for smaller GPU
```

### Issue: Data not found
**Solution**: Verify file structure matches expected layout:
```
data/raw/
├── images/
├── metadata.csv
└── metadata.txt
```

### Issue: Import errors in PyCharm
**Solution**: 
1. Open **File → Settings → Project → Python Interpreter**
2. Verify interpreter points to `venv\Scripts\python.exe`
3. Reinstall packages: `pip install -r requirements.txt`

---

## Publication Timeline

- **Weeks 1-3**: Data preparation & ViT training
- **Weeks 4-6**: Attention analysis & visualization
- **Weeks 7-9**: Novel insights & head specialization
- **Weeks 10-12**: Clinical validation & expert review
- **Weeks 13-16**: Manuscript writing & submission

**Target Venue**: Medical Image Analysis (Tier-1)
**Expected Citations**: 20-40/year (high-impact)

---

## Key References

- Dosovitskiy et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- Chen et al. (2023). "CheXchoNet: Cardiac Pathology Detection from Chest X-rays"
- Simonyan & Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition"

---

## Contact & Support

For questions about this project, refer to:
- Project documentation in `results/analysis/`
- Jupyter notebooks in `notebooks/`
- Inline code comments in `src/`

---

## License

This project is for research and educational purposes.

---

**Last Updated**: April 2026
**Status**: Setup Phase ✓ | Data Preparation (Current)
