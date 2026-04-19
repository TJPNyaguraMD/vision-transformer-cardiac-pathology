# GitHub Setup Guide
## Vision Transformer for Cardiac Pathology Detection

This guide walks you through publishing your project on GitHub.

---

## Step 1: Create GitHub Account (if you don't have one)

1. Go to https://github.com
2. Click "Sign up"
3. Enter email, password, username
4. Verify email
5. Done!

**Recommended username:** Something professional
- ✓ `your-name-ai` 
- ✓ `your-name-ml`
- ✗ `coolDude123`

---

## Step 2: Create New Repository

1. Log in to GitHub
2. Click **"+"** icon (top right)
3. Select **"New repository"**
4. Fill in details:

**Repository name:**
```
vision-transformer-cardiac-pathology
```

**Description:**
```
Vision Transformer for detecting cardiac pathology (SLVH/DLV) from chest X-rays using the CheXchoNet dataset. Includes interpretable attention visualization.
```

**Public or Private:**
- Select **"Public"** (more impressive for hiring)

**Initialize with:**
- ✓ Add README.md (you'll replace this)
- ✓ Add .gitignore (choose Python)
- ✓ Choose license (MIT is good)

5. Click **"Create repository"**

---

## Step 3: Clone Repository to Your Computer

In PowerShell:

```powershell
cd C:\Users\YourName\Documents

git clone https://github.com/YOUR-USERNAME/vision-transformer-cardiac-pathology.git

cd vision-transformer-cardiac-pathology
```

Replace `YOUR-USERNAME` with your actual GitHub username!

---

## Step 4: Add Your Project Files

Copy these files into the repository folder:

**Essential files:**
```
vision-transformer-cardiac-pathology/
├── README.md                          (detailed one you'll create)
├── QUICK_START.md                     (quick start guide)
├── config.py
├── requirements.txt
├── LICENSE                            (GitHub created this)
├── .gitignore                         (GitHub created this)
│
├── scripts/
│   ├── 01_prepare_dataset.py
│   ├── 02_train_vit.py
│   ├── 02b_train_vit_improved.py
│   └── 03_extract_attention.py
│
├── docs/
│   ├── FINDINGS_README.md
│   ├── TRAINING_APPROACH_DETAILED.md
│   ├── TRAINING_RESULTS_README.md
│   └── ATTENTION_ANALYSIS_REPORT.md
│
└── images/
    └── example_attention_visualization.png
```

**Files to SKIP (don't upload):**
- ✗ data/ (too large, reference external)
- ✗ results/checkpoints/ (too large, link to download)
- ✗ .venv/ (Python virtual environment)
- ✗ __pycache__/ (Python cache)

---

## Step 5: Update .gitignore

Edit `.gitignore` to exclude large files:

```
# Virtual Environment
.venv/
venv/
ENV/

# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/

# Data (too large)
data/raw/
data/processed/
data/splits/
*.jpg
*.png

# Model checkpoints (too large)
results/checkpoints/
results/checkpoints_improved/
results/attention_maps/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

---

## Step 6: Create Main README

Create `README.md` in the repository root:

```markdown
# Vision Transformer for Cardiac Pathology Detection

![Status](https://img.shields.io/badge/Status-Proof%20of%20Concept-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)

Interpretable Vision Transformer for detecting cardiac pathology from chest X-rays using CheXchoNet dataset (71,589 images).

## Features

- ✅ Vision Transformer (ViT-Base) with ImageNet pretrained weights
- ✅ Class imbalance handling (Focal Loss implementation)
- ✅ Attention visualization for interpretability
- ✅ Stratified train/val/test splits (60/15/25)
- ✅ Complete documentation and analysis
- ✅ Reproducible results (fixed random seeds)

## Results

### Initial Training (CrossEntropyLoss)
- **Test AUC:** 0.7717
- **Test Accuracy:** 86.55%
- **Sensitivity:** 14.48% (low - class imbalance issue)
- **Specificity:** 98.06%

### Improved Training (Focal Loss) - In Progress
- **Expected AUC:** 0.82-0.85
- **Expected Sensitivity:** 60-70%
- **Expected Specificity:** 95-97%

## Quick Start

### Prerequisites
- Python 3.12
- NVIDIA GPU (6GB+ VRAM) or CPU
- 500GB storage for dataset

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/vision-transformer-cardiac-pathology.git
cd vision-transformer-cardiac-pathology

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. Download CheXchoNet from [PhysioNet](https://physionet.org/)
2. Extract to: `F:\Data\Medical Data\Xray Images\chexchonet-...\`
3. Run preparation:

```bash
python scripts/01_prepare_dataset.py
```

### Training

```bash
# Original training
python scripts/02_train_vit.py

# Improved training with Focal Loss
python scripts/02b_train_vit_improved.py
```

### Attention Extraction

```bash
python scripts/03_extract_attention.py
```

## Documentation

- **[PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)** - Complete project overview
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide
- **[TRAINING_RESULTS_README.md](docs/TRAINING_RESULTS_README.md)** - Detailed results
- **[ATTENTION_ANALYSIS_REPORT.md](docs/ATTENTION_ANALYSIS_REPORT.md)** - Interpretability findings

## Project Structure

```
vision-transformer-cardiac-pathology/
├── scripts/
│   ├── 01_prepare_dataset.py          # Data preparation
│   ├── 02_train_vit.py                # Standard training
│   ├── 02b_train_vit_improved.py      # Focal Loss training
│   └── 03_extract_attention.py        # Attention visualization
├── docs/
│   ├── FINDINGS_README.md
│   ├── TRAINING_APPROACH_DETAILED.md
│   ├── TRAINING_RESULTS_README.md
│   └── ATTENTION_ANALYSIS_REPORT.md
├── config.py                          # Configuration
├── requirements.txt                   # Dependencies
└── README.md                          # This file
```

## Key Features

### 1. Interpretable AI
- Attention visualization shows what model focuses on
- 40+ visualizations comparing normal vs pathology
- Layer-by-layer analysis (Layer 8 most interpretable)

### 2. Proper Data Handling
- Stratified splitting maintains class distribution
- Addresses class imbalance (86% normal, 14% pathology)
- Reproducible with fixed random seed

### 3. Medical Imaging Best Practices
- Transfer learning from ImageNet
- Appropriate metrics (AUC, sensitivity, specificity)
- Attention alignment with cardiac anatomy

## Results & Analysis

### Normal Cases (Healthy)
- Model: 91-98% confidence in "normal"
- Attention: Broad, distributed across chest
- Interpretation: "Nothing abnormal detected"

### Pathology Cases (Disease)
- Model: 14-39% confidence in "composite"
- Attention: Concentrated on cardiac region (when detected)
- Interpretation: "Abnormality detected in heart area"

### Attention Patterns
- **Red color:** Model focuses here (high attention)
- **Orange/Yellow:** Medium attention
- **Blue/White:** Model ignores this area

## Clinical Notes

⚠️ **Not for Clinical Use Yet**
- Sensitivity currently too low (14% vs. 70% needed)
- Improved training should address this (focal loss)
- Requires clinical validation before deployment

✅ **Good For**
- Research and development
- Portfolio demonstration
- Understanding medical AI
- Interpretability research

## Future Improvements

- [ ] Clinical validation with radiologists
- [ ] Multi-center testing
- [ ] Multi-task learning (SLVH + DLV separate)
- [ ] Ensemble with CNN models
- [ ] Mobile deployment app

## Hardware Requirements

**Minimum:**
- 4GB GPU VRAM
- 16GB RAM
- 500GB storage

**Recommended:**
- 6GB+ GPU (like RTX 4050)
- 40GB RAM
- 1TB SSD

**Training Time:**
- Data preparation: 30 min
- Model training: 6-8 hours
- Attention extraction: 2-3 hours

## References

- **ViT Paper:** [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **Focal Loss:** [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- **Dataset:** [CheXchoNet on PhysioNet](https://physionet.org/content/chexchonet/1.0.0/)
- **timm Library:** [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

## License

MIT License - See [LICENSE](LICENSE) file

## Contact & Questions

Questions about the project?
- Open an [Issue](https://github.com/YOUR-USERNAME/vision-transformer-cardiac-pathology/issues)
- Check documentation in `/docs/`
- Review code comments for details

## Citation

If you use this project, please cite:

```bibtex
@software{vit_cardiac_2026,
  title={Vision Transformer for Cardiac Pathology Detection},
  author={Your Name},
  year={2026},
  url={https://github.com/YOUR-USERNAME/vision-transformer-cardiac-pathology}
}
```

## Acknowledgments

- CheXchoNet dataset creators
- PyTorch and timm teams
- Vision Transformer paper authors

---

**Last Updated:** April 2026
**Status:** Active Development
**Next Phase:** Focal Loss training & clinical validation
```

---

## Step 7: Create QUICK_START.md

Create `QUICK_START.md` for impatient users:

```markdown
# Quick Start (5 Minutes)

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/vision-transformer-cardiac-pathology.git
cd vision-transformer-cardiac-pathology

python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt
```

## Prepare Data

1. Download CheXchoNet
2. Extract to `F:\Data\Medical Data\Xray Images\chexchonet-...\`
3. Run: `python scripts/01_prepare_dataset.py`

## Train Model

```bash
# Standard training (6-8 hours)
python scripts/02_train_vit.py

# Improved training with Focal Loss (8-10 hours)
python scripts/02b_train_vit_improved.py
```

## Extract Attention

```bash
python scripts/03_extract_attention.py
# Output: results/attention_maps/ (40 visualizations)
```

## Results

Check `results/` folder:
- `checkpoints/` - Trained model
- `attention_maps/` - Visualizations
- `logs/` - Training logs

## Full Documentation

See [README.md](README.md) for complete guide and [docs/](docs/) for detailed analysis.
```

---

## Step 8: Add Example Image

If you have one of your attention visualizations, add it:

```
vision-transformer-cardiac-pathology/
└── images/
    └── example_attention_layer8.png  (copy from results/attention_maps/)
```

Reference in README:
```markdown
![Example Attention Visualization](images/example_attention_layer8.png)
*Layer 8 attention visualization showing cardiac region focus (red = high attention)*
```

---

## Step 9: Upload to GitHub

In PowerShell in your repository folder:

```powershell
# Check status
git status

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Vision Transformer for cardiac pathology detection"

# Push to GitHub
git push origin main
```

Done! 🎉

---

## Step 10: Verify on GitHub

1. Go to https://github.com/YOUR-USERNAME/vision-transformer-cardiac-pathology
2. Check that files are there
3. README should display nicely
4. Share the link!

---

## Optional: Add Badges to README

```markdown
![Status](https://img.shields.io/badge/Status-Proof%20of%20Concept-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)
```

Generates nice colored badges at top of README!

---

## Optional: Enable GitHub Pages

Want a fancy project website?

1. Go to Repository → Settings → Pages
2. Select "Deploy from a branch"
3. Choose "main" branch
4. Click Save
5. Your README becomes a website!

---

## Troubleshooting

**"git command not found"**
- Install Git: https://git-scm.com/

**"Authentication failed"**
- Use personal access token instead of password
- Generate here: https://github.com/settings/tokens

**"File too large"**
- Git has 100MB limit
- Add to .gitignore
- Use Git LFS for large files

---

## Next Steps After Uploading

1. ✅ Repository is live
2. Share link on LinkedIn
3. Add to portfolio/CV
4. Run improved training
5. Update README with results
6. Push improved results to GitHub

---

**You're ready!** 🚀
```