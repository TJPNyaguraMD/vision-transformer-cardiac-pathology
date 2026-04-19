"""
03_extract_attention.py - FIXED VERSION (v3)

Vision Transformer Attention Extraction & Visualization
"""

import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import timm

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
EXTERNAL_IMAGES_PATH = Path(r"F:\Data\Medical Data\Xray Images\chexchonet-a-chest-radiograph-dataset-with-gold-standard-echocardiography-labels-1.0.0\chexchonet-a-chest-radiograph-dataset-with-gold-standard-echocardiography-labels-1.0.0\images")
EXTERNAL_METADATA_CSV = Path(r"F:\Data\Medical Data\Xray Images\chexchonet-a-chest-radiograph-dataset-with-gold-standard-echocardiography-labels-1.0.0\chexchonet-a-chest-radiograph-dataset-with-gold-standard-echocardiography-labels-1.0.0\metadata.csv")
CHECKPOINTS_DIR = PROJECT_ROOT / "results" / "checkpoints"
ATTENTION_OUTPUT_DIR = PROJECT_ROOT / "results" / "attention_maps"

# Model parameters
MODEL_NAME = "vit_base_patch16_224"
IMAGE_SIZE = 224
NUM_HEADS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Visualization parameters
NUM_SAMPLES_PER_CLASS = 5

# ============================================================================
# ATTENTION EXTRACTION
# ============================================================================

class AttentionExtractor:
    """Extract attention weights from ViT model"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.attention_weights = {}
        self.hooks = []

        # Register hooks for each attention layer
        for layer_idx, block in enumerate(self.model.blocks):
            hook = block.attn.register_forward_hook(
                self._get_attention_hook(layer_idx)
            )
            self.hooks.append(hook)

    def _get_attention_hook(self, layer_idx):
        def hook(module, input, output):
            self.attention_weights[layer_idx] = output.detach().cpu()
        return hook

    def extract(self, images):
        """Extract attention for a batch of images"""
        self.attention_weights = {}
        with torch.no_grad():
            outputs = self.model(images)
        return self.attention_weights, outputs

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_attention_map(attention_weights, layer_idx, head_idx):
    """Create visualization of attention for a specific layer and head"""
    # Get attention for specific head: (seq_len, seq_len)
    # seq_len = 197 (196 patches + 1 CLS token)
    attn = attention_weights[layer_idx][head_idx, :, :]  # (197, 197)

    # Skip CLS token (first row and column)
    attn = attn[1:, 1:]  # (196, 196)

    # Average attention across KEY dimension (dim=1)
    # This shows what each QUERY position attends to on average
    attn_avg = attn.mean(dim=1)  # (196,) - average attention for each query

    # Reshape to grid: 196 = 14 * 14
    attn_grid = attn_avg.reshape(14, 14)

    # Upsample to image size
    attn_map = F.interpolate(
        attn_grid.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    ).squeeze()

    # Normalize to [0, 1]
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    return attn_map.numpy()

def visualize_layer_attention(attention_weights, image, layer_idx, title, output_dir):
    """Visualize all 12 heads for a specific layer"""
    attn = attention_weights[layer_idx]  # (num_heads, seq_len, seq_len)
    num_heads = attn.shape[0]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    # Show original image
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    for head_idx in range(num_heads):
        attention_map = create_attention_map(attention_weights, layer_idx, head_idx)

        axes[head_idx].imshow(image_np, cmap='gray', alpha=0.6)
        axes[head_idx].imshow(attention_map, cmap='hot', alpha=0.4)
        axes[head_idx].set_title(f'Head {head_idx}')
        axes[head_idx].axis('off')

    fig.suptitle(f'{title} - Layer {layer_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / f"{title.replace(' ', '_')}_layer_{layer_idx}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return output_path

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_best_checkpoint(checkpoint_dir):
    """Load best trained model"""
    checkpoints = sorted(checkpoint_dir.glob("best_model_*.pth"))
    if checkpoints:
        return checkpoints[-1]
    return None

def get_sample_images(metadata_csv, num_normal=5, num_composite=5):
    """Get sample image paths"""
    import pandas as pd

    df = pd.read_csv(metadata_csv)

    # Create labels
    labels = []
    for _, row in df.iterrows():
        label = 1 if row['composite_slvh_dlv'] == 1 else 0
        labels.append(label)

    labels = np.array(labels)

    # Get indices
    normal_indices = np.where(labels == 0)[0]
    composite_indices = np.where(labels == 1)[0]

    normal_samples = np.random.choice(normal_indices, min(num_normal, len(normal_indices)), replace=False)
    composite_samples = np.random.choice(composite_indices, min(num_composite, len(composite_indices)), replace=False)

    normal_files = [df.iloc[idx]['cxr_filename'] for idx in normal_samples]
    composite_files = [df.iloc[idx]['cxr_filename'] for idx in composite_samples]

    return normal_files, composite_files

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"\n{'='*80}")
    print("ViT CARDIAC PATHOLOGY - ATTENTION EXTRACTION & VISUALIZATION")
    print(f"{'='*80}")

    try:
        # Step 1: Create output directory
        print(f"\n{'='*80}")
        print("STEP 1: Setting Up Output Directories")
        print(f"{'='*80}")

        ATTENTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directory: {ATTENTION_OUTPUT_DIR}")

        # Step 2: Load best model
        print(f"\n{'='*80}")
        print("STEP 2: Loading Trained Model")
        print(f"{'='*80}")

        best_checkpoint = load_best_checkpoint(CHECKPOINTS_DIR)
        if not best_checkpoint:
            print(f"✗ No checkpoint found in {CHECKPOINTS_DIR}")
            sys.exit(1)

        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
        model.load_state_dict(torch.load(best_checkpoint, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()

        print(f"✓ Model loaded: {best_checkpoint.name}")

        # Step 3: Create attention extractor
        print(f"\n{'='*80}")
        print("STEP 3: Setting Up Attention Extractor")
        print(f"{'='*80}")

        extractor = AttentionExtractor(model, DEVICE)
        print(f"✓ Extractor ready (12 layers, {NUM_HEADS} heads per layer)")

        # Step 4: Get sample images
        print(f"\n{'='*80}")
        print("STEP 4: Getting Sample Images")
        print(f"{'='*80}")

        normal_files, composite_files = get_sample_images(
            EXTERNAL_METADATA_CSV,
            num_normal=NUM_SAMPLES_PER_CLASS,
            num_composite=NUM_SAMPLES_PER_CLASS
        )

        print(f"✓ Found {len(normal_files)} normal samples")
        print(f"✓ Found {len(composite_files)} composite samples")

        # Setup image transform
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Step 5: Extract attention for normal cases
        print(f"\n{'='*80}")
        print("STEP 5: Extracting Attention for Normal Cases")
        print(f"{'='*80}")

        saved_count = 0
        for idx, filename in enumerate(normal_files):
            img_path = EXTERNAL_IMAGES_PATH / filename

            if not img_path.exists():
                print(f"  ⚠️  Image not found: {filename}")
                continue

            # Load and preprocess image
            image_pil = Image.open(img_path).convert('RGB')
            image_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)

            # Extract attention
            attn_weights, outputs = extractor.extract(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

            print(f"  [{idx+1}] {filename}")
            print(f"       Normal prob: {probs[0]:.3f}, Composite prob: {probs[1]:.3f}")

            # Create visualizations for selected layers
            for layer_idx in [2, 5, 8, 11]:
                title = f"Normal_case_{idx:02d}"
                path = visualize_layer_attention(attn_weights, image_pil, layer_idx,
                                        title, ATTENTION_OUTPUT_DIR)
                saved_count += 1

        # Step 6: Extract attention for composite cases
        print(f"\n{'='*80}")
        print("STEP 6: Extracting Attention for Composite Cases (Pathology)")
        print(f"{'='*80}")

        for idx, filename in enumerate(composite_files):
            img_path = EXTERNAL_IMAGES_PATH / filename

            if not img_path.exists():
                print(f"  ⚠️  Image not found: {filename}")
                continue

            # Load and preprocess image
            image_pil = Image.open(img_path).convert('RGB')
            image_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)

            # Extract attention
            attn_weights, outputs = extractor.extract(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

            print(f"  [{idx+1}] {filename}")
            print(f"       Normal prob: {probs[0]:.3f}, Composite prob: {probs[1]:.3f}")

            # Create visualizations for selected layers
            for layer_idx in [2, 5, 8, 11]:
                title = f"Composite_case_{idx:02d}"
                path = visualize_layer_attention(attn_weights, image_pil, layer_idx,
                                        title, ATTENTION_OUTPUT_DIR)
                saved_count += 1

        # Cleanup
        extractor.remove_hooks()

        # Step 7: Summary
        print(f"\n{'='*80}")
        print("STEP 7: Attention Extraction Complete")
        print(f"{'='*80}")

        print(f"""
Visualizations created:
  - Normal cases: {len(normal_files)} cases × 4 layers = {len(normal_files)*4} visualizations
  - Composite cases: {len(composite_files)} cases × 4 layers = {len(composite_files)*4} visualizations
  - Per visualization: 12 attention heads (3×4 grid)
  - Total saved: {saved_count} images
  
Output location: {ATTENTION_OUTPUT_DIR}

Key observations to look for:
  1. Do normal and composite cases attend to different regions?
  2. Which layers show strongest specialization?
  3. Do specific heads focus on cardiac regions?
  4. Is there evidence of pathology-related patterns?
        """)

        print(f"\n{'='*80}")
        print("✓ ATTENTION EXTRACTION COMPLETE!")
        print(f"{'='*80}")

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ ERROR DURING ATTENTION EXTRACTION")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()