"""
02_train_vit.py

Vision Transformer Training Script for Cardiac Pathology Detection

This script:
1. Loads data splits from Phase 2
2. Creates PyTorch DataLoaders for train/val/test
3. Initializes ViT-Base model with ImageNet pretrained weights
4. Trains for 100 epochs with AdamW + cosine annealing
5. Tracks AUC, accuracy, loss metrics
6. Saves best model checkpoints
7. Evaluates on test set

Run: python scripts/02_train_vit.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import timm

# ============================================================================
# CONFIGURATION (Hardcoded for reliability)
# ============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
EXTERNAL_IMAGES_PATH = Path(r"F:\Data\Medical Data\Xray Images\chexchonet-a-chest-radiograph-dataset-with-gold-standard-echocardiography-labels-1.0.0\chexchonet-a-chest-radiograph-dataset-with-gold-standard-echocardiography-labels-1.0.0\images")
EXTERNAL_METADATA_CSV = Path(r"F:\Data\Medical Data\Xray Images\chexchonet-a-chest-radiograph-dataset-with-gold-standard-echocardiography-labels-1.0.0\chexchonet-a-chest-radiograph-dataset-with-gold-standard-echocardiography-labels-1.0.0\metadata.csv")
SPLITS_PATH = PROJECT_ROOT / "data" / "splits" / "splits.json"
CHECKPOINTS_DIR = PROJECT_ROOT / "results" / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"

# Training parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 5
RANDOM_SEED = 42
PATIENCE = 15  # Early stopping patience

# Model parameters
MODEL_NAME = "vit_base_patch16_224"
IMAGE_SIZE = 224
NUM_CLASSES = 2
PRETRAINED = True

# ============================================================================
# DATASET CLASS
# ============================================================================

class CheXchoNetDataset(Dataset):
    """PyTorch Dataset for CheXchoNet images"""
    
    def __init__(self, metadata_path, image_dir, indices, transform=None):
        """
        Args:
            metadata_path: Path to metadata.csv
            image_dir: Path to images directory
            indices: Array of indices to use from metadata
            transform: Optional transforms to apply
        """
        import pandas as pd
        
        self.df = pd.read_csv(metadata_path)
        self.df = self.df.iloc[indices].reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        
        # Create pathology labels
        self.labels = []
        for _, row in self.df.iterrows():
            if row['composite_slvh_dlv'] == 1:
                label = 1  # Composite (pathology)
            else:
                label = 0  # Normal
            self.labels.append(label)
        
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.df.iloc[idx]['cxr_filename']
        img_path = self.image_dir / img_name
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, torch.tensor(label, dtype=torch.long)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_transforms(augment=False):
    """Create image transforms"""
    if augment:
        return transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def load_splits(splits_path):
    """Load train/val/test splits from JSON"""
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    return (np.array(splits['train_indices']),
            np.array(splits['val_indices']),
            np.array(splits['test_indices']))


def compute_metrics(outputs, labels):
    """Compute AUC, accuracy, and other metrics"""
    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    preds = outputs.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    
    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return auc, acc, sensitivity, specificity


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    all_outputs = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        all_outputs.append(outputs.detach().cpu())
        all_labels.append(labels.cpu())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute epoch metrics
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    train_auc, train_acc, train_sens, train_spec = compute_metrics(all_outputs, all_labels)
    train_loss = total_loss / len(train_loader)

    return train_loss, train_auc, train_acc, train_sens, train_spec


def validate_epoch(model, val_loader, criterion, device):
    """Validate one epoch"""
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    # Compute epoch metrics
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    val_auc, val_acc, val_sens, val_spec = compute_metrics(all_outputs, all_labels)
    val_loss = total_loss / len(val_loader)

    return val_loss, val_auc, val_acc, val_sens, val_spec


def save_checkpoint(model, epoch, best_auc, checkpoint_dir):
    """Save model checkpoint"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch}_auc_{best_auc:.4f}.pth"
    torch.save(model.state_dict(), checkpoint_path)

    # Keep only best 3 checkpoints
    checkpoints = sorted(checkpoint_dir.glob("best_model_*.pth"))
    if len(checkpoints) > 3:
        for old_checkpoint in checkpoints[:-3]:
            old_checkpoint.unlink()

    return checkpoint_path


def load_best_checkpoint(checkpoint_dir):
    """Load best checkpoint"""
    checkpoints = sorted(checkpoint_dir.glob("best_model_*.pth"))
    if checkpoints:
        return checkpoints[-1]
    return None


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    print(f"\n{'='*80}")
    print("ViT CARDIAC PATHOLOGY - MODEL TRAINING")
    print(f"{'='*80}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    # Set seed
    set_seed(RANDOM_SEED)

    try:
        # Step 1: Load splits
        print(f"\n{'='*80}")
        print("STEP 1: Loading Data Splits")
        print(f"{'='*80}")
        train_indices, val_indices, test_indices = load_splits(SPLITS_PATH)
        print(f"✓ Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")

        # Step 2: Create datasets
        print(f"\n{'='*80}")
        print("STEP 2: Creating Datasets")
        print(f"{'='*80}")

        import pandas as pd

        train_transform = create_transforms(augment=True)
        val_transform = create_transforms(augment=False)

        train_dataset = CheXchoNetDataset(
            EXTERNAL_METADATA_CSV, EXTERNAL_IMAGES_PATH,
            train_indices, transform=train_transform
        )
        val_dataset = CheXchoNetDataset(
            EXTERNAL_METADATA_CSV, EXTERNAL_IMAGES_PATH,
            val_indices, transform=val_transform
        )
        test_dataset = CheXchoNetDataset(
            EXTERNAL_METADATA_CSV, EXTERNAL_IMAGES_PATH,
            test_indices, transform=val_transform
        )

        print(f"✓ Datasets created")

        # Step 3: Create dataloaders
        print(f"\n{'='*80}")
        print("STEP 3: Creating DataLoaders")
        print(f"{'='*80}")

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )

        print(f"✓ DataLoaders created")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        # Step 4: Initialize model
        print(f"\n{'='*80}")
        print("STEP 4: Initializing Model")
        print(f"{'='*80}")

        model = timm.create_model(MODEL_NAME, pretrained=PRETRAINED, num_classes=NUM_CLASSES)
        model = model.to(DEVICE)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model loaded: {MODEL_NAME}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Step 5: Setup optimizer and scheduler
        print(f"\n{'='*80}")
        print("STEP 5: Setting Up Optimizer & Scheduler")
        print(f"{'='*80}")

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        # Cosine annealing scheduler with warmup
        def warmup_cosine(epoch):
            if epoch < WARMUP_EPOCHS:
                return (epoch + 1) / WARMUP_EPOCHS
            else:
                progress = (epoch - WARMUP_EPOCHS) / (NUM_EPOCHS - WARMUP_EPOCHS)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)

        print(f"✓ Optimizer: AdamW (lr={LEARNING_RATE})")
        print(f"✓ Scheduler: Cosine annealing with {WARMUP_EPOCHS}-epoch warmup")

        # Step 6: Training loop
        print(f"\n{'='*80}")
        print("STEP 6: Training")
        print(f"{'='*80}\n")

        best_val_auc = 0
        patience_counter = 0

        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

            # Train
            train_loss, train_auc, train_acc, train_sens, train_spec = train_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )

            # Validate
            val_loss, val_auc, val_acc, val_sens, val_spec = validate_epoch(
                model, val_loader, criterion, DEVICE
            )

            # Step scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Print metrics
            print(f"  Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
            print(f"  LR: {current_lr:.2e}")

            # Save checkpoint if improved
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                save_checkpoint(model, epoch+1, best_val_auc, CHECKPOINTS_DIR)
                print(f"  ✓ New best AUC: {best_val_auc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  ⚠ No improvement for {PATIENCE} epochs. Stopping training.")
                    break

        # Step 7: Test evaluation
        print(f"\n{'='*80}")
        print("STEP 7: Test Set Evaluation")
        print(f"{'='*80}")

        # Load best checkpoint
        best_checkpoint = load_best_checkpoint(CHECKPOINTS_DIR)
        if best_checkpoint:
            model.load_state_dict(torch.load(best_checkpoint))
            print(f"✓ Loaded best checkpoint: {best_checkpoint.name}")

        # Evaluate on test set
        model.eval()
        test_loss = 0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing")
            for images, labels in pbar:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        # Compute test metrics
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        test_auc, test_acc, test_sens, test_spec = compute_metrics(all_outputs, all_labels)
        test_loss = test_loss / len(test_loader)

        print(f"\n{'='*80}")
        print("FINAL TEST RESULTS")
        print(f"{'='*80}")
        print(f"Test Loss:      {test_loss:.4f}")
        print(f"Test AUC:       {test_auc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"Test Sensitivity: {test_sens:.4f}")
        print(f"Test Specificity: {test_spec:.4f}")

        print(f"\n{'='*80}")
        print("✓ TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"Best model saved in: {CHECKPOINTS_DIR}")
        print(f"\nNext steps:")
        print(f"  1. Review test results above")
        print(f"  2. Run: python scripts/03_extract_attention.py")
        print(f"  3. Run: python scripts/04_analyze_heads.py")

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ ERROR DURING TRAINING")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()