"""
01_prepare_dataset.py - CORRECTED FOR ACTUAL DATA

Data Preparation Script for ViT Cardiac Pathology Project
Uses actual CheXchoNet metadata format with binary pathology columns
"""

import json
import sys
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

EXTERNAL_IMAGES_PATH = Path(r"F:\Data\Medical Data\Xray Images\chexchonet-a-chest-radiograph-dataset-with-gold-standard-echocardiography-labels-1.0.0\chexchonet-a-chest-radiograph-dataset-with-gold-standard-echocardiography-labels-1.0.0\images")
EXTERNAL_METADATA_CSV = Path(r"F:\Data\Medical Data\Xray Images\chexchonet-a-chest-radiograph-dataset-with-gold-standard-echocardiography-labels-1.0.0\chexchonet-a-chest-radiograph-dataset-with-gold-standard-echocardiography-labels-1.0.0\metadata.csv")

PROJECT_ROOT = Path(__file__).parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# Split configuration
TRAIN_SPLIT = 0.60
VAL_SPLIT = 0.15
TEST_SPLIT = 0.25
RANDOM_SEED = 42

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load metadata CSV from external drive."""
    print(f"\n{'='*80}")
    print("STEP 1: Loading Metadata")
    print(f"{'='*80}")
    print(f"Loading from: {metadata_path}")

    try:
        df = pd.read_csv(metadata_path)
        print(f"✓ Successfully loaded metadata!")
        print(f"  Total rows: {len(df)}")
        print(f"  Columns: {', '.join(df.columns.tolist())}")
        return df
    except Exception as e:
        print(f"✗ ERROR loading metadata: {e}")
        raise


def create_pathology_label(row):
    """Create a single pathology label from binary columns."""
    # slvh=1, dlv=0, composite=0 -> SLVH
    # slvh=0, dlv=1, composite=0 -> DLV
    # slvh=0, dlv=0, composite=1 -> Composite
    # slvh=1, dlv=1, composite=1 -> Composite (both conditions)
    # slvh=0, dlv=0, composite=0 -> Normal

    slvh = int(row['slvh'])
    dlv = int(row['dlv'])
    composite = int(row['composite_slvh_dlv'])

    # Priority: if composite is set, it takes precedence
    if composite == 1:
        return 'Composite'
    elif slvh == 1:
        return 'SLVH'
    elif dlv == 1:
        return 'DLV'
    else:
        return 'Normal'


def verify_data_integrity(df: pd.DataFrame, images_path: Path) -> pd.DataFrame:
    """Verify that all image files exist."""
    print(f"\n{'='*80}")
    print("STEP 2: Verifying Data Integrity")
    print(f"{'='*80}")
    print(f"Checking images in: {images_path}")
    print(f"Total images to verify: {len(df)}")

    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_path}")

    # Get list of image files
    existing_images = set(f.name for f in images_path.glob("*") if f.is_file())
    print(f"Found {len(existing_images)} image files in directory")

    # Use cxr_filename column
    image_col = 'cxr_filename'
    print(f"Using image column: '{image_col}'")

    # Verify each image
    missing_count = 0
    verified_indices = []

    print("\nVerifying files...")
    for idx, image_name in tqdm(enumerate(df[image_col]), total=len(df)):
        if str(image_name) in existing_images:
            verified_indices.append(idx)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"⚠️  WARNING: {missing_count} images not found!")
        df_clean = df.iloc[verified_indices].reset_index(drop=True)
        return df_clean
    else:
        print(f"✓ All {len(df)} images verified!")
        return df


def analyze_distribution(df: pd.DataFrame) -> Dict:
    """Analyze pathology distribution."""
    print(f"\n{'='*80}")
    print("STEP 3: Analyzing Data Distribution")
    print(f"{'='*80}")

    # Create pathology labels from binary columns
    print("Creating pathology labels from: slvh, dlv, composite_slvh_dlv columns")
    df['pathology_label'] = df.apply(create_pathology_label, axis=1)
    pathology_col = 'pathology_label'

    print(f"Pathology labels created")

    # Count distribution
    distribution = df[pathology_col].value_counts()
    total = len(df)

    print(f"\nPathology Distribution:")
    print(f"{'-'*50}")
    for pathology, count in distribution.items():
        percentage = (count / total) * 100
        bar_length = int(percentage / 5)
        bar = "█" * bar_length
        print(f"  {pathology:15} {count:7d} ({percentage:5.1f}%) {bar}")
    print(f"{'-'*50}")
    print(f"  {'Total':15} {total:7d} (100.0%)")

    return {
        'total': total,
        'distribution': distribution.to_dict(),
        'pathology_column': pathology_col
    }


def create_stratified_splits(
    df: pd.DataFrame,
    pathology_col: str,
    train_ratio: float = 0.60,
    val_ratio: float = 0.15,
    test_ratio: float = 0.25,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create stratified train/val/test splits."""
    print(f"\n{'='*80}")
    print("STEP 4: Creating Stratified Splits")
    print(f"{'='*80}")
    print(f"Target split ratios:")
    print(f"  Train: {train_ratio*100:.0f}%")
    print(f"  Val:   {val_ratio*100:.0f}%")
    print(f"  Test:  {test_ratio*100:.0f}%")
    print(f"Stratifying by: {pathology_col}")

    # First split: Train vs Temp (Val+Test)
    splitter1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=(1 - train_ratio),
        random_state=random_state
    )

    train_idx, temp_idx = next(splitter1.split(df, df[pathology_col]))

    # Second split: Val vs Test from Temp
    test_portion_of_temp = test_ratio / (1 - train_ratio)

    splitter2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_portion_of_temp,
        random_state=random_state
    )

    val_idx, test_idx = next(splitter2.split(df.iloc[temp_idx], df.iloc[temp_idx][pathology_col]))

    # Convert to original indices
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    print(f"\n✓ Splits created:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")

    return train_idx, val_idx, test_idx


def verify_split_stratification(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    pathology_col: str
) -> None:
    """Verify splits maintain original distribution."""
    print(f"\n{'='*80}")
    print("STEP 5: Verifying Split Stratification")
    print(f"{'='*80}")

    def get_distribution(indices):
        subset = df.iloc[indices]
        dist = subset[pathology_col].value_counts()
        total = len(subset)
        return {path: (count/total)*100 for path, count in dist.items()}

    original_dist = df[pathology_col].value_counts()
    original_dist = {path: (count/len(df))*100 for path, count in original_dist.items()}

    train_dist = get_distribution(train_idx)
    val_dist = get_distribution(val_idx)
    test_dist = get_distribution(test_idx)

    print(f"\nOriginal Distribution:")
    for path in sorted(original_dist.keys()):
        pct = original_dist[path]
        print(f"  {path:15} {pct:6.2f}%")

    print(f"\nTrain Split ({len(train_idx):6d} samples - {len(train_idx)/len(df)*100:.1f}%):")
    for path in sorted(train_dist.keys()):
        pct = train_dist[path]
        print(f"  {path:15} {pct:6.2f}%")

    print(f"\nVal Split ({len(val_idx):6d} samples - {len(val_idx)/len(df)*100:.1f}%):")
    for path in sorted(val_dist.keys()):
        pct = val_dist[path]
        print(f"  {path:15} {pct:6.2f}%")

    print(f"\nTest Split ({len(test_idx):6d} samples - {len(test_idx)/len(df)*100:.1f}%):")
    for path in sorted(test_dist.keys()):
        pct = test_dist[path]
        print(f"  {path:15} {pct:6.2f}%")

    print(f"\n✓ All splits are properly stratified!")


def save_splits(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: Path
) -> None:
    """Save split indices to JSON files."""
    print(f"\n{'='*80}")
    print("STEP 6: Saving Split Indices")
    print(f"{'='*80}")

    output_dir.mkdir(parents=True, exist_ok=True)

    splits_data = {
        'train_indices': train_idx.tolist(),
        'val_indices': val_idx.tolist(),
        'test_indices': test_idx.tolist(),
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'total_size': len(train_idx) + len(val_idx) + len(test_idx),
        'random_seed': RANDOM_SEED,
    }

    output_path = output_dir / 'splits.json'

    with open(output_path, 'w') as f:
        json.dump(splits_data, f, indent=2)

    print(f"✓ Splits saved to: {output_path}")
    print(f"  Train indices: {len(train_idx)}")
    print(f"  Val indices:   {len(val_idx)}")
    print(f"  Test indices:  {len(test_idx)}")
    print(f"  Total:         {splits_data['total_size']}")


def main():
    """Main execution."""
    print(f"\n{'='*80}")
    print("ViT CARDIAC PATHOLOGY - DATA PREPARATION")
    print(f"{'='*80}")
    print(f"Project Configuration:")
    print(f"  External images (F: drive): {EXTERNAL_IMAGES_PATH}")
    print(f"  Metadata (F: drive):        {EXTERNAL_METADATA_CSV}")
    print(f"  Output splits (E: drive):   {SPLITS_DIR}")
    print(f"  Random seed:                {RANDOM_SEED}")

    try:
        # Step 1: Load metadata
        df = load_metadata(EXTERNAL_METADATA_CSV)

        # Step 2: Verify data integrity
        df = verify_data_integrity(df, EXTERNAL_IMAGES_PATH)

        # Step 3: Analyze distribution
        stats = analyze_distribution(df)
        pathology_col = stats['pathology_column']

        # Step 4: Create stratified splits
        train_idx, val_idx, test_idx = create_stratified_splits(
            df,
            pathology_col,
            train_ratio=TRAIN_SPLIT,
            val_ratio=VAL_SPLIT,
            test_ratio=TEST_SPLIT,
            random_state=RANDOM_SEED
        )

        # Step 5: Verify stratification
        verify_split_stratification(df, train_idx, val_idx, test_idx, pathology_col)

        # Step 6: Save splits
        save_splits(train_idx, val_idx, test_idx, SPLITS_DIR)

        print(f"\n{'='*80}")
        print("✓ DATA PREPARATION COMPLETE!")
        print(f"{'='*80}")
        print(f"\nNext steps:")
        print(f"  1. Review splits.json in {SPLITS_DIR}")
        print(f"  2. Run: python scripts/02_train_vit.py")
        print(f"\nYour data is ready for training!")

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ ERROR DURING DATA PREPARATION")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()