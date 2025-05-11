# scripts/extract_features.py
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DATA_DIR, DEPLOYMENT_DATA_DIR, SUBSET_SIZE, METADATA_CSV_PATH
from src.models.feature_extractor import FeatureExtractor
from src.utils.cloudinary_helper import CloudinaryHelper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_and_save_features(
    metadata_path=METADATA_CSV_PATH, 
    output_dir=DEPLOYMENT_DATA_DIR, 
    subset_size=SUBSET_SIZE, 
    upload_to_cloudinary=False
):
    """
    Extract features from fashion dataset
    
    Args:
        metadata_path: Path to metadata CSV file
        output_dir: Directory to save extracted features
        subset_size: Number of items to process (0 means all)
        upload_to_cloudinary: Whether to upload images to Cloudinary
    """
    logger.info("Starting feature extraction...")
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    logger.info(f"Loading metadata from {metadata_path}")
    metadata_df = pd.read_csv(metadata_path)
    total_items = len(metadata_df)
    logger.info(f"Loaded metadata with {total_items} items")
    
    # Apply subset if specified
    if subset_size > 0 and subset_size < total_items:
        metadata_df = metadata_df.sample(n=subset_size, random_state=42)
        logger.info(f"Selected subset of {subset_size} items")
    else:
        logger.info(f"Processing full dataset: {total_items} items")
    
    # Initialize components
    logger.info("Initializing feature extractor...")
    extractor = FeatureExtractor()
    
    # Initialize Cloudinary if requested
    cloudinary_helper = None
    if upload_to_cloudinary:
        try:
            cloudinary_helper = CloudinaryHelper()
            logger.info("Cloudinary initialized successfully")
        except Exception as e:
            logger.warning(f"Cloudinary initialization failed: {e}")
            logger.warning("Continuing without Cloudinary upload")
    
    # Initialize storage
    features_dict = {}
    metadata_dict = {}
    cloudinary_urls = {}
    failed_items = []
    
    # Process each item
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Extracting features"):
        product_id = str(row['id'])
        image_path = os.path.join(PROCESSED_DATA_DIR, "images", row['filename'])
        
        try:
            # Extract features
            features = extractor.extract_features(image_path)
            features_dict[product_id] = features.tolist()  # Convert numpy array to list for JSON
            
            # Upload to Cloudinary if enabled
            image_url = f"/image/{product_id}"  # Default local URL
            if cloudinary_helper:
                try:
                    cloud_url = cloudinary_helper.upload_image(
                        image_path, 
                        public_id=product_id,
                        folder="fashion-recommender"
                    )
                    cloudinary_urls[product_id] = cloud_url
                    image_url = cloud_url
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Cloudinary upload failed for {product_id}: {e}")
            
            # Store metadata
            metadata_dict[product_id] = {
                'id': product_id,
                'product_name': row.get('product_name', ''),
                'gender': row.get('gender', ''),
                'category': row.get('master_category', ''),
                'subcategory': row.get('sub_category', ''),
                'article_type': row.get('article_type', ''),
                'color': row.get('base_color', ''),
                'filename': row['filename'],
                'image_url': image_url
            }
            
            # Save checkpoint every 100 items
            if len(features_dict) % 100 == 0:
                save_checkpoint(features_dict, metadata_dict, cloudinary_urls, output_dir)
                
        except Exception as e:
            logger.error(f"Error processing product {product_id}: {e}")
            failed_items.append(product_id)
    
    # Save final results
    save_final_results(features_dict, metadata_dict, cloudinary_urls, output_dir)
    
    # Print summary
    elapsed_time = time.time() - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Successfully processed: {len(features_dict)} items")
    logger.info(f"Failed: {len(failed_items)} items")
    if failed_items:
        logger.info(f"Failed items: {failed_items[:10]}...")  # Show first 10 failed items
    
    return features_dict, metadata_dict, cloudinary_urls

def save_checkpoint(features_dict, metadata_dict, cloudinary_urls, output_dir):
    """Save intermediate checkpoint"""
    checkpoint_data = {
        'features': features_dict,
        'metadata': metadata_dict,
        'cloudinary_urls': cloudinary_urls
    }
    
    checkpoint_path = os.path.join(output_dir, f'checkpoint_{len(features_dict)}.json')
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f)
    
    logger.info(f"Checkpoint saved: {len(features_dict)} items processed")

def save_final_results(features_dict, metadata_dict, cloudinary_urls, output_dir):
    """Save final processed data in multiple formats"""
    # Save as JSON
    json_data = {
        'features': features_dict,
        'metadata': metadata_dict,
        'cloudinary_urls': cloudinary_urls
    }
    
    json_path = os.path.join(output_dir, 'fashion_data.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    
    # Save metadata separately
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f)
    
    # Save Cloudinary URLs
    if cloudinary_urls:
        urls_path = os.path.join(output_dir, 'cloudinary_urls.json')
        with open(urls_path, 'w') as f:
            json.dump(cloudinary_urls, f)
    
    # Save features as pickle (numpy arrays)
    features_np = {k: np.array(v) for k, v in features_dict.items()}
    pickle_path = os.path.join(output_dir, 'features.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(features_np, f)
    
    # Save metadata as pickle
    metadata_pickle_path = os.path.join(output_dir, 'metadata.pkl')
    with open(metadata_pickle_path, 'wb') as f:
        pickle.dump(metadata_dict, f)
    
    # File size info
    json_size = os.path.getsize(json_path) / (1024 * 1024)
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"JSON file size: {json_size:.2f} MB")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features for fashion recommendation")
    parser.add_argument("--metadata_path", type=str, default=str(METADATA_CSV_PATH),
                        help="Path to metadata CSV file")
    parser.add_argument("--output_dir", type=str, default=str(DEPLOYMENT_DATA_DIR),
                        help="Directory to save deployment files")
    parser.add_argument("--subset_size", type=int, default=SUBSET_SIZE,
                        help="Number of items to process (0 for all)")
    parser.add_argument("--upload_to_cloudinary", action="store_true",
                        help="Upload images to Cloudinary")
    
    args = parser.parse_args()
    
    extract_and_save_features(
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        subset_size=args.subset_size,
        upload_to_cloudinary=args.upload_to_cloudinary
    )