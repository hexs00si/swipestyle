import os
import sys
import logging
import shutil
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple

import pandas as pd
from tqdm import tqdm
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import config
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Constants
MIN_IMAGES_EXPECTED = 100  # Safety threshold

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_image(filepath: Path) -> bool:
    """Verify image is not corrupt"""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except (IOError, SyntaxError) as e:
        logger.warning(f"Invalid image {filepath}: {e}")
        return False

def copy_file(args: Tuple[Path, Path]) -> Tuple[bool, Path]:
    """Thread-safe file copy with validation"""
    src, dst = args
    try:
        shutil.copy2(src, dst)
        return (validate_image(dst), dst)
    except Exception as e:
        logger.error(f"Failed to copy {src}: {e}")
        return (False, src)

def locate_dataset() -> Optional[Path]:
    """
    Locate the manually downloaded dataset
    """
    logger.info("Looking for manually downloaded dataset...")
    
    # Check common locations
    possible_locations = [
        RAW_DATA_DIR / "fashion-product-images-small",
        RAW_DATA_DIR / "fashion-dataset",
        RAW_DATA_DIR
    ]
    
    for location in possible_locations:
        # Check if this location has styles.csv and images directory
        styles_path = location / "styles.csv"
        images_dir = location / "images"
        
        if os.path.exists(styles_path) and os.path.exists(images_dir):
            logger.info(f"Found dataset at {location}")
            image_count = len(os.listdir(images_dir))
            logger.info(f"Found {image_count} image files")
            return location
    
    logger.error("Dataset not found in expected locations")
    return None

def prepare_dataset(dataset_dir: Path) -> Optional[Path]:
    """
    Prepare the dataset for use by organizing files and creating metadata
    with parallel processing and validation
    """
    logger.info("Preparing dataset...")
    
    # Create output directories
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    processed_images_dir = PROCESSED_DATA_DIR / "images"
    os.makedirs(processed_images_dir, exist_ok=True)
    
    # Check for styles.csv and images
    styles_path = dataset_dir / "styles.csv"
    original_images_dir = dataset_dir / "images"
    
    if not os.path.exists(styles_path):
        logger.error(f"Error: styles.csv not found at {styles_path}")
        return None
    
    if not os.path.exists(original_images_dir):
        logger.error(f"Error: images directory not found at {original_images_dir}")
        return None
    
    # Load styles metadata with error handling
    try:
        logger.info("Loading metadata...")
        styles_df = pd.read_csv(styles_path, on_bad_lines='skip')
    except Exception as e:
        logger.error(f"Error loading styles.csv: {e}")
        return None
    
    # Create a simpler metadata file with only the columns we need
    logger.info("Creating clean metadata...")
    
    # Filter out rows where the image doesn't exist
    existing_images = set(os.listdir(original_images_dir))
    styles_df['image_file'] = styles_df['id'].astype(str) + '.jpg'
    valid_df = styles_df[styles_df['image_file'].isin(existing_images)].copy()
    
    if len(valid_df) < MIN_IMAGES_EXPECTED:
        logger.warning(f"Only {len(valid_df)} valid images found (expected ≥{MIN_IMAGES_EXPECTED})")
    
    # Select and rename columns for clarity
    try:
        metadata_df = valid_df[[
            'id', 'gender', 'masterCategory', 'subCategory', 'articleType',
            'baseColour', 'season', 'year', 'usage', 'productDisplayName'
        ]].rename(columns={
            'masterCategory': 'master_category',
            'subCategory': 'sub_category',
            'articleType': 'article_type',
            'baseColour': 'base_color',
            'productDisplayName': 'product_name'
        })
    except KeyError as e:
        logger.error(f"Missing expected column in styles.csv: {e}")
        # Try to create metadata with available columns
        logger.info("Attempting to create metadata with available columns...")
        required_cols = ['id']
        available_cols = [col for col in required_cols if col in valid_df.columns]
        
        if 'id' not in available_cols:
            logger.error("Required column 'id' not found in styles.csv")
            return None
            
        # Create a minimal metadata dataframe
        metadata_df = valid_df[available_cols].copy()
        # Add default values for missing columns
        if 'gender' not in metadata_df:
            metadata_df['gender'] = 'Unknown'
        if 'master_category' not in metadata_df and 'masterCategory' in valid_df:
            metadata_df['master_category'] = valid_df['masterCategory']
        else:
            metadata_df['master_category'] = 'Unknown'
        if 'product_name' not in metadata_df and 'productDisplayName' in valid_df:
            metadata_df['product_name'] = valid_df['productDisplayName']
        else:
            metadata_df['product_name'] = 'Product ' + metadata_df['id'].astype(str)
            
    # Add filename column
    metadata_df['filename'] = metadata_df['id'].astype(str) + '.jpg'
    
    # Save cleaned metadata
    metadata_path = PROCESSED_DATA_DIR / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    logger.info(f"Saved metadata with {len(metadata_df)} items to {metadata_path}")
    
    # Copy images to the processed directory using parallel processing
    logger.info("Copying images to processed directory...")
    
    # Prepare copy tasks
    copy_tasks = []
    for _, row in metadata_df.iterrows():
        src_path = original_images_dir / f"{row['id']}.jpg"
        dst_path = processed_images_dir / f"{row['id']}.jpg"
        if os.path.exists(src_path):
            copy_tasks.append((src_path, dst_path))
    
    # Use parallel processing for copying
    num_workers = min(cpu_count(), 8)  # Limit to 8 cores max
    logger.info(f"Using {num_workers} workers for parallel processing")
    
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(copy_file, copy_tasks),
            total=len(copy_tasks),
            desc="Copying images"
        ))
    
    # Verify results
    successful_copies = sum(1 for result, _ in results if result)
    failed_copies = [(src, dst) for (result, _), (src, dst) in zip(results, copy_tasks) if not result]
    
    if failed_copies:
        logger.warning(f"{len(failed_copies)} images failed validation")
        for src, dst in failed_copies[:5]:  # Show first 5 failures
            logger.warning(f"Failed: {src} -> {dst}")
    
    logger.info(f"Copied {successful_copies} images to {processed_images_dir}")
    
    # Save version info for tracking
    version_info = {
        "dataset": "manually_downloaded",
        "processing_date": pd.Timestamp.now().isoformat(),
        "total_items": len(metadata_df),
        "valid_images": successful_copies,
        "columns": list(metadata_df.columns)
    }
    
    with open(PROCESSED_DATA_DIR / "version.json", 'w') as f:
        json.dump(version_info, f, indent=2)
    
    return metadata_path

if __name__ == "__main__":
    try:
        # Locate manually downloaded dataset
        dataset_dir = locate_dataset()
        
        if dataset_dir:
            # Prepare dataset
            metadata_path = prepare_dataset(dataset_dir)
            
            if metadata_path:
                logger.info("Dataset successfully processed!")
            else:
                logger.error("Failed to process dataset")
                sys.exit(1)
        else:
            logger.error("Could not locate manually downloaded dataset")
            logger.info("Please ensure the dataset is in data/raw/ with styles.csv and images/ folder")
            sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)