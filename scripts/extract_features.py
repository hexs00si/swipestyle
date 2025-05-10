import os
import sys
import json
import pickle
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    PROCESSED_DATA_DIR, 
    DEPLOYMENT_DATA_DIR, 
    SUBSET_SIZE, 
    METADATA_CSV_PATH
)
from src.models.feature_extractor import FeatureExtractor
from src.utils.cloudinary_helper import CloudinaryHelper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('feature_extraction.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingState:
    """Track the state of feature extraction process"""
    processed_ids: List[str]
    failed_ids: List[str]
    features_dict: Dict[str, List[float]]
    metadata_dict: Dict[str, Dict[str, Any]]
    cloudinary_urls: Dict[str, str]
    last_checkpoint: int
    
    def save(self, filepath: str):
        """Save state to file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f)
    
    @classmethod
    def load(cls, filepath: str) -> 'ProcessingState':
        """Load state from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

class FeatureExtractionPipeline:
    """Enhanced pipeline for feature extraction with resume capability"""
    
    def __init__(self, 
                 metadata_path: str = str(METADATA_CSV_PATH),
                 output_dir: str = str(DEPLOYMENT_DATA_DIR),
                 subset_size: int = SUBSET_SIZE,
                 batch_size: int = 10,
                 checkpoint_interval: int = 50):
        
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.extractor = None
        self.cloudinary_helper = None
        
        # State tracking
        self.state_file = self.output_dir / 'extraction_state.json'
        self.state = self._load_or_create_state()
    
    def _load_or_create_state(self) -> ProcessingState:
        """Load existing state or create new one"""
        if self.state_file.exists():
            logger.info(f"Loading existing state from {self.state_file}")
            return ProcessingState.load(self.state_file)
        else:
            return ProcessingState(
                processed_ids=[],
                failed_ids=[],
                features_dict={},
                metadata_dict={},
                cloudinary_urls={},
                last_checkpoint=0
            )
    
    def _initialize_components(self, use_cloudinary: bool = True):
        """Initialize ML models and cloud services"""
        # Initialize feature extractor
        logger.info("Initializing feature extractor...")
        self.extractor = FeatureExtractor()
        
        # Initialize Cloudinary
        if use_cloudinary:
            try:
                self.cloudinary_helper = CloudinaryHelper()
                logger.info("Cloudinary initialized successfully")
            except Exception as e:
                logger.warning(f"Cloudinary initialization failed: {e}")
                self.cloudinary_helper = None
    
    def _load_dataset(self) -> pd.DataFrame:
        """Load and prepare dataset"""
        # Load metadata
        logger.info(f"Loading metadata from {self.metadata_path}")
        metadata_df = pd.read_csv(self.metadata_path)
        
        # Filter out already processed items
        if self.state.processed_ids:
            logger.info(f"Filtering out {len(self.state.processed_ids)} already processed items")
            metadata_df = metadata_df[~metadata_df['id'].astype(str).isin(self.state.processed_ids)]
        
        # Sample if necessary
        if len(metadata_df) > self.subset_size:
            metadata_df = metadata_df.sample(n=self.subset_size, random_state=42)
            logger.info(f"Sampled {self.subset_size} items for processing")
        
        return metadata_df
    
    def _process_single_item(self, row: pd.Series) -> Tuple[str, Dict[str, Any]]:
        """Process a single fashion item"""
        product_id = str(row['id'])
        image_path = os.path.join(PROCESSED_DATA_DIR, "images", row['filename'])
        
        try:
            # Extract features
            features = self.extractor.extract_features(image_path)
            
            # Upload to Cloudinary if available
            image_url = f"/image/{product_id}"  # Default local URL
            if self.cloudinary_helper:
                try:
                    image_url = self.cloudinary_helper.upload_image(
                        image_path, 
                        public_id=product_id,
                        folder="fashion-recommender"
                    )
                    self.state.cloudinary_urls[product_id] = image_url
                except Exception as e:
                    logger.warning(f"Cloudinary upload failed for {product_id}: {e}")
            
            # Create metadata
            metadata = {
                'id': product_id,
                'product_name': row.get('product_name', row.get('productDisplayName', '')),
                'gender': row['gender'],
                'category': row.get('master_category', row.get('masterCategory', '')),
                'subcategory': row.get('sub_category', row.get('subCategory', '')),
                'article_type': row.get('article_type', row.get('articleType', '')),
                'color': row.get('base_color', row.get('baseColour', '')),
                'filename': row['filename'],
                'image_url': image_url
            }
            
            # Store results
            self.state.features_dict[product_id] = features.tolist()
            self.state.metadata_dict[product_id] = metadata
            self.state.processed_ids.append(product_id)
            
            return product_id, metadata
            
        except Exception as e:
            logger.error(f"Error processing product {product_id}: {e}")
            self.state.failed_ids.append(product_id)
            raise
    
    def _save_checkpoint(self):
        """Save current state as checkpoint"""
        self.state.last_checkpoint = len(self.state.processed_ids)
        self.state.save(self.state_file)
        
        # Save intermediate results
        checkpoint_data = {
            'features': self.state.features_dict,
            'metadata': self.state.metadata_dict,
            'cloudinary_urls': self.state.cloudinary_urls
        }
        
        checkpoint_path = self.output_dir / f'checkpoint_{self.state.last_checkpoint}.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
        
        logger.info(f"Checkpoint saved at {len(self.state.processed_ids)} items")
    
    def process_dataset(self, use_cloudinary: bool = True, parallel: bool = False):
        """Process the entire dataset"""
        start_time = time.time()
        
        # Initialize components
        self._initialize_components(use_cloudinary)
        
        # Load dataset
        metadata_df = self._load_dataset()
        
        if len(metadata_df) == 0:
            logger.info("No items to process")
            return
        
        logger.info(f"Processing {len(metadata_df)} items...")
        
        # Process items
        if parallel and self.cloudinary_helper is None:
            # Parallel processing (only for local feature extraction)
            self._process_parallel(metadata_df)
        else:
            # Sequential processing (required for Cloudinary uploads)
            self._process_sequential(metadata_df)
        
        # Save final results
        self._save_final_results()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Successfully processed: {len(self.state.processed_ids)} items")
        logger.info(f"Failed: {len(self.state.failed_ids)} items")
    
    def _process_sequential(self, metadata_df: pd.DataFrame):
        """Process items sequentially"""
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing"):
            try:
                self._process_single_item(row)
                
                # Save checkpoint periodically
                if len(self.state.processed_ids) % self.checkpoint_interval == 0:
                    self._save_checkpoint()
                    
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue
    
    def _process_parallel(self, metadata_df: pd.DataFrame):
        """Process items in parallel (for local extraction only)"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for idx, row in metadata_df.iterrows():
                future = executor.submit(self._process_single_item, row)
                futures.append(future)
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    result = future.result()
                    
                    # Save checkpoint periodically
                    if len(self.state.processed_ids) % self.checkpoint_interval == 0:
                        self._save_checkpoint()
                        
                except Exception as e:
                    logger.error(f"Error in parallel processing: {e}")
    
    def _save_final_results(self):
        """Save final processed data in multiple formats"""
        # Save as JSON
        json_data = {
            'features': self.state.features_dict,
            'metadata': self.state.metadata_dict,
            'cloudinary_urls': self.state.cloudinary_urls
        }
        
        json_path = self.output_dir / 'fashion_data.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f)
        
        # Save metadata separately
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.state.metadata_dict, f)
        
        # Save features as pickle
        features_np = {k: np.array(v) for k, v in self.state.features_dict.items()}
        pickle_path = self.output_dir / 'features.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(features_np, f)
        
        # Save metadata as pickle
        metadata_pickle_path = self.output_dir / 'metadata.pkl'
        with open(metadata_pickle_path, 'wb') as f:
            pickle.dump(self.state.metadata_dict, f)
        
        # Calculate and save statistics
        stats = {
            'total_processed': len(self.state.processed_ids),
            'total_failed': len(self.state.failed_ids),
            'feature_dimension': len(next(iter(self.state.features_dict.values()))) if self.state.features_dict else 0,
            'json_size_mb': os.path.getsize(json_path) / (1024 * 1024),
            'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        stats_path = self.output_dir / 'extraction_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"JSON file size: {stats['json_size_mb']:.2f} MB")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features for fashion recommendation")
    parser.add_argument("--metadata_path", type=str, default=str(METADATA_CSV_PATH),
                        help="Path to metadata CSV file")
    parser.add_argument("--output_dir", type=str, default=str(DEPLOYMENT_DATA_DIR),
                        help="Directory to save deployment files")
    parser.add_argument("--subset_size", type=int, default=SUBSET_SIZE,
                        help="Number of items to include in the subset")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for processing")
    parser.add_argument("--upload_to_cloudinary", action="store_true",
                        help="Whether to upload images to Cloudinary")
    parser.add_argument("--parallel", action="store_true",
                        help="Use parallel processing (only for local extraction)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = FeatureExtractionPipeline(
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        subset_size=args.subset_size,
        batch_size=args.batch_size
    )
    
    # Process dataset
    pipeline.process_dataset(
        use_cloudinary=args.upload_to_cloudinary,
        parallel=args.parallel and not args.upload_to_cloudinary
    )

if __name__ == "__main__":
    main()