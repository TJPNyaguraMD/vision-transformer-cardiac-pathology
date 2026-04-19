"""
Configuration file for ViT Cardiac Pathology Project
Update these values as needed for different experiments
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(r"E:\PORTFOLIO\MEDICAL IMAGING\vision_transformer")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
CACHE_DIR = DATA_DIR / "cache"

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
ATTENTION_MAPS_DIR = RESULTS_DIR / "attention_maps"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
FIGURES_DIR = RESULTS_DIR / "figures"

SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_NAME = "CheXchoNet"
DATASET_SIZE = 71589  # Total number of images
NUM_PATIENTS = 24689

# Image specifications
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 196

# Pathology classes
PATHOLOGIES = ["SLVH", "DLV", "Composite"]
NUM_CLASSES = 3

# Split configuration (stratified by pathology)
TRAIN_SPLIT = 0.60
VAL_SPLIT = 0.15
TEST_SPLIT = 0.25

# Normalization (ImageNet defaults)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Note: Can be updated with custom CXR statistics if available
# MEAN = [0.5]  # Grayscale CXR
# STD = [0.25]

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Vision Transformer (ViT-Base)
MODEL_TYPE = "vit_base_patch16_224"
HIDDEN_DIM = 768
NUM_LAYERS = 12
NUM_HEADS = 12
MLP_DIM = 3072
DROPOUT = 0.1
ATTENTION_DROPOUT = 0.0

# Pretrained weights (optional)
PRETRAINED = True  # Use ImageNet pretrained weights as initialization
PRETRAINED_MODEL = "vit_base_patch16_224_in21k"  # From timm

# Alternative: Train from scratch
# PRETRAINED = False

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Learning parameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
OPTIMIZER = "AdamW"

# Scheduling
SCHEDULER = "cosine"  # or "linear", "constant"
WARMUP_EPOCHS = 5
TOTAL_EPOCHS = 100

# Batch size
BATCH_SIZE = 32  # Adjust based on GPU memory (V100: 32, A100: 64)
NUM_WORKERS = 4  # Parallel data loading

# Data augmentation (during training)
USE_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    "random_crop": True,
    "random_flip": True,
    "random_rotation": 15,  # degrees
    "random_affine": True,
    "random_gamma": True,
    "mixup_alpha": 0.1,  # Mixup augmentation
}

# Loss function
LOSS_FUNCTION = "cross_entropy"  # or "focal_loss" for class imbalance
# If using focal loss:
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0

# ============================================================================
# TRAINING MONITORING
# ============================================================================

# Checkpointing
SAVE_CHECKPOINT_EVERY = 5  # epochs
KEEP_BEST_N_CHECKPOINTS = 3
EARLY_STOPPING_PATIENCE = 15  # epochs without improvement

# Metrics to track
TRACK_METRICS = ["loss", "accuracy", "auc", "f1_score", "sensitivity", "specificity"]

# Logging
LOG_EVERY_N_BATCHES = 100
USE_TENSORBOARD = True
TENSORBOARD_LOG_DIR = RESULTS_DIR / "logs"

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Validation during training
VALIDATE_EVERY_N_EPOCHS = 1
VALIDATION_BATCH_SIZE = 64

# Test time
TEST_BATCH_SIZE = 64
TEST_TIME_AUGMENTATION = False  # TTA for slightly better performance

# Metrics thresholds
OPTIMAL_THRESHOLD = 0.5  # Can be tuned on validation set

# ============================================================================
# ATTENTION ANALYSIS CONFIGURATION
# ============================================================================

# Which layers to extract attention from
EXTRACT_ATTENTION_LAYERS = list(range(12))  # All 12 layers

# Attention visualization
ATTENTION_VIZ_CONFIG = {
    "upsample_method": "bilinear",  # Upsample 14x14 patches to 224x224
    "normalize": True,  # Normalize attention weights per image
    "colormap": "viridis",  # Matplotlib colormap
    "alpha": 0.5,  # Overlay transparency
}

# Head specialization analysis
HEAD_SPECIALIZATION_CONFIG = {
    "n_clusters": 5,  # Number of attention patterns to discover
    "correlation_threshold": 0.7,  # For anatomy alignment
    "min_patch_frequency": 0.1,  # Min % of images where patch is attended to
}

# ============================================================================
# HARDWARE & COMPUTATION
# ============================================================================

DEVICE = "cuda"  # or "cpu" for debugging
GPU_ID = 0  # If multiple GPUs available
USE_MIXED_PRECISION = True  # AMP for faster training
GRADIENT_ACCUMULATION_STEPS = 1  # For larger effective batch sizes

# ============================================================================
# RANDOM SEED (for reproducibility)
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# PATHS TO INPUT DATA
# ============================================================================

# When running scripts, these are constructed from RAW_DATA_DIR
IMAGES_PATH = RAW_DATA_DIR / "images"
METADATA_CSV_PATH = RAW_DATA_DIR / "metadata.csv"
METADATA_TXT_PATH = RAW_DATA_DIR / "metadata.txt"

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

EXPERIMENT_NAME = "vit_cardiac_baseline"
EXPERIMENT_DESCRIPTION = "ViT-Base trained on CheXchoNet for cardiac pathology detection"

# Useful for ablation studies:
# EXPERIMENT_NAME = "vit_cardiac_no_pretrained"
# EXPERIMENT_NAME = "vit_cardiac_focal_loss"
# EXPERIMENT_NAME = "vit_cardiac_larger_batch"

# ============================================================================
# PUBLICATION & ANALYSIS
# ============================================================================

# For generating publication-quality figures
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"  # or "png", "svg"

# Head specialization analysis
ANALYZE_HEAD_CORRELATION_WITH = "pathology"  # Analyze which heads specialize in which pathologies
GENERATE_ATTENTION_HEATMAPS = True
GENERATE_PATCH_IMPORTANCE_MAPS = True

# Statistical testing
STATISTICAL_TEST = "t_test"  # For comparing head specialization across pathologies
SIGNIFICANCE_LEVEL = 0.05  # p-value threshold

# ============================================================================
# REPRODUCTION & VERSION CONTROL
# ============================================================================

# Record all hyperparameters used
SAVE_CONFIG_WITH_CHECKPOINT = True
CONFIG_VERSION = "1.0"