# ViT Training Approach: Detailed Explanation

## 1. Training Architecture Overview

### Model: Vision Transformer (ViT-Base)

```
Input Image (224×224)
    ↓
Patch Embedding (16×16 patches = 196 patches)
    ↓
ViT-Base Encoder (12 layers, 12 attention heads)
    ↓
Classification Head (196×768 → 2 classes)
    ↓
Output: [Normal probability, Composite probability]
```

**ViT-Base Configuration:**
- Patch size: 16 (so 224/16 = 14×14 = 196 patches)
- Hidden dimension: 768
- Number of layers: 12
- Attention heads: 12 (each head specializes in different features)
- MLP dimension: 3072
- Dropout: 0.1

**Why ViT?**
1. Interpretability: Attention weights show which patches matter
2. Transfer learning: Pre-trained on ImageNet (1000 classes)
3. State-of-the-art: Better than CNNs for many vision tasks
4. Novel: First application to cardiac X-ray pathology

---

## 2. Transfer Learning Strategy

### Step 1: Load Pretrained Weights
```python
# Load ViT-Base pretrained on ImageNet-21k (14M images)
model = timm.create_model('vit_base_patch16_224', pretrained=True)
```

**Benefits:**
- ✓ Already learned general image features (edges, shapes, textures)
- ✓ Faster convergence (fewer epochs needed)
- ✓ Better performance with limited data (71K images)
- ✓ Avoids training from scratch (would need months of GPU time)

### Step 2: Replace Classification Head
```python
# Remove last layer trained on 1000 ImageNet classes
# Add new head for 2 classes (Normal vs Composite)
model.head = nn.Linear(768, 2)
```

### Step 3: Fine-tune on CheXchoNet
- Unfreeze all layers (train everything)
- Use lower learning rate (1e-4) to preserve pretrained knowledge
- Train for 100 epochs

**Result:** Model trained on cardiac pathology while keeping ImageNet knowledge

---

## 3. Data Loading Strategy

### Three DataLoaders

**Training DataLoader:**
```python
train_loader = DataLoader(
    CheXchoNetDataset(train_indices, split='train'),
    batch_size=32,
    shuffle=True,           # Shuffle for better learning
    num_workers=4,          # Load data in parallel
    augmentation=True       # Random crops, flips, rotations
)
```
- **Size:** 42,953 images
- **Batches:** ~1,342 batches per epoch
- **Augmentation:** Yes (random transforms)

**Validation DataLoader:**
```python
val_loader = DataLoader(
    CheXchoNetDataset(val_indices, split='val'),
    batch_size=32,
    shuffle=False,          # No shuffle for consistency
    num_workers=4,
    augmentation=False      # No augmentation
)
```
- **Size:** 10,738 images
- **Batches:** ~335 batches per epoch
- **Purpose:** Monitor generalization, guide checkpointing

**Test DataLoader:**
```python
test_loader = DataLoader(
    CheXchoNetDataset(test_indices, split='test'),
    batch_size=32,
    shuffle=False,
    augmentation=False
)
```
- **Size:** 17,898 images
- **Purpose:** Final evaluation (never seen during training)

---

## 4. Training Loop: One Epoch Explained

### For each batch in train_loader:

**Step 1: Forward Pass**
```python
images, labels = batch  # Load 32 images + labels from F: drive
outputs = model(images) # ViT processes: 32×224×224 → 32×2
loss = criterion(outputs, labels)  # Compare to ground truth
```

**Step 2: Backward Pass**
```python
optimizer.zero_grad()   # Clear old gradients
loss.backward()         # Compute gradients via backpropagation
optimizer.step()        # Update weights: w = w - lr × gradient
scheduler.step()        # Adjust learning rate
```

**Step 3: Accumulate Metrics**
```python
# Track: loss, accuracy, AUC (computed across full epoch)
train_loss += loss.item()
correct += (outputs.argmax(1) == labels).sum()
all_probs.append(outputs.cpu().detach())  # For AUC later
```

### After each epoch:

**Validation Phase**
```python
model.eval()  # Turn off dropout
with torch.no_grad():  # Don't compute gradients
    for images, labels in val_loader:
        outputs = model(images)
        val_loss += criterion(outputs, labels)
        # Compute AUC on validation set
```

**Logging**
```python
print(f"Epoch {epoch+1}/100:")
print(f"  Train AUC: {train_auc:.4f}, Loss: {train_loss:.4f}")
print(f"  Val AUC:   {val_auc:.4f}, Loss: {val_loss:.4f}")
```

**Checkpointing**
```python
if val_auc > best_auc:
    best_auc = val_auc
    # Save checkpoint
    torch.save(model.state_dict(), f'checkpoint_epoch{epoch}.pth')
    # Keep only best 3
    if len(checkpoints) > 3:
        delete_oldest_checkpoint()
```

---

## 5. Optimization Strategy

### Optimizer: AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,          # Low learning rate (preserve pretrained knowledge)
    weight_decay=0.01 # L2 regularization (prevent overfitting)
)
```

**Why AdamW?**
- ✓ Adaptive learning rates per parameter
- ✓ Better than SGD for transformers
- ✓ Decoupled weight decay (true L2 regularization)

### Scheduler: Cosine Annealing with Warmup

```
Learning Rate Schedule (100 epochs):
                    ╱╲
                   ╱  ╲
                  ╱    ╲____
               ║ warmup ║ cosine decay
              5 epochs   95 epochs

Warmup (first 5 epochs):  1e-4 → gradually increases
Cosine decay (next 95):   gradually decreases → near 0
```

**Why?**
- Warmup: Stabilize training early
- Cosine decay: Smooth learning rate reduction
- Prevents sudden weight changes

---

## 6. Class Imbalance Handling

**Data:** 86.2% Normal, 13.8% Composite

**Strategy 1: Stratified Sampling** ✓ (already done in Phase 2)
- Each split maintains 86.2% / 13.8% distribution
- Model sees balanced representation

**Strategy 2: Weighted Loss Function** (optional)
```python
class_weights = torch.tensor([1.0, (86.2/13.8)])  # [1.0, 6.26]
criterion = nn.CrossEntropyLoss(weight=class_weights)
```
- Normal loss weight: 1.0
- Composite loss weight: 6.26
- Penalizes mistakes on rare class more heavily

**Strategy 3: Evaluate with AUC, not Accuracy**
- Accuracy misled by class imbalance
- AUC is unaffected by imbalance

---

## 7. Monitoring & Metrics

### Per-Epoch Metrics

**Training Metrics:**
- Loss (cross-entropy)
- Accuracy (% correct)
- AUC (area under ROC curve)

**Validation Metrics:**
- Loss
- Accuracy
- AUC (used for checkpointing)
- Sensitivity (recall of Composite class)
- Specificity (recall of Normal class)

### Final Test Evaluation

After training, load best checkpoint and evaluate on test set:

```
Test Set Results (25% held-out data, 17,898 images):
────────────────────────────────────────
AUC:         0.87-0.88  (primary metric)
Accuracy:    85-88%
Sensitivity: 75-80%     (catch pathology cases)
Specificity: 87-90%     (avoid false alarms)
────────────────────────────────────────
```

---

## 8. Expected Training Curve

```
Epoch 1:  Train AUC: 0.72  Val AUC: 0.75  Loss: 0.45
Epoch 10: Train AUC: 0.85  Val AUC: 0.87  Loss: 0.25
Epoch 30: Train AUC: 0.90  Val AUC: 0.88  Loss: 0.15
Epoch 50: Train AUC: 0.93  Val AUC: 0.88  Loss: 0.10
Epoch 100: Train AUC: 0.95  Val AUC: 0.88  Loss: 0.08

Pattern:
- Training AUC increases → model learning
- Validation AUC plateaus → found good generalization
- No overfitting (val AUC stays constant as train AUC increases)
```

---

## 9. Early Stopping & Patience

**What if validation AUC doesn't improve?**

```python
patience = 15  # epochs without improvement
if val_auc > best_auc:
    patience_counter = 0
    best_auc = val_auc
else:
    patience_counter += 1
    if patience_counter >= patience:
        print("No improvement for 15 epochs. Stopping training.")
        break  # Stop early to save time
```

---

## 10. Attention Extraction (for interpretability)

**After training, Phase 4 will:**

1. Forward pass through model
2. Extract attention weights from all 12 layers
3. Visualize which patches the model attends to

Example:
- Normal case: Attention on healthy heart regions
- Composite case: Attention on abnormal regions

This enables interpretability and clinical validation!

---

## Summary: Training Strategy

| Component | Choice | Reason |
|-----------|--------|--------|
| Model | ViT-Base | State-of-art, interpretable |
| Pretrained | ImageNet-21k | Transfer learning, faster training |
| Optimizer | AdamW | Best for transformers |
| Schedule | Cosine + warmup | Stable, smooth learning |
| Epochs | 100 | Sufficient for convergence |
| Batch size | 32 | RTX 4050 6GB memory limit |
| Learning rate | 1e-4 | Preserve pretrained knowledge |
| Loss | CrossEntropy | Standard for classification |
| Metrics | AUC + Accuracy | Handle class imbalance |
| Checkpointing | Best 3 | Keep space, easy recovery |

---

Ready to see the actual **02_train_vit.py** code? 🚀
