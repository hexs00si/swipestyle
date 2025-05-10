import torch
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from PIL import Image
import os
import logging
from typing import Union, List, Optional
import gc
from contextlib import contextmanager

from src.config import MODEL_NAME

# Set up logging
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Feature extractor using DINOv2 model for fashion image embeddings
    Optimized for Apple Silicon (MPS) with proper memory management
    """
    
    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        """
        Initialize DINOv2 feature extractor
        
        Args:
            model_name: Name of the model to use from Hugging Face
            device: Specific device to use (optional)
        """
        logger.info(f"Initializing feature extractor with model: {model_name}")
        
        # Set device
        self.device = self._get_device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Get feature dimension
            self.feature_dim = self._get_feature_dimension()
            logger.info(f"Model loaded successfully. Feature dimension: {self.feature_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_device(self, preferred_device: Optional[str] = None) -> torch.device:
        """
        Get the best available device for computation
        
        Args:
            preferred_device: User-specified device preference
            
        Returns:
            torch.device: Selected device
        """
        if preferred_device:
            return torch.device(preferred_device)
            
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _get_feature_dimension(self) -> int:
        """
        Get the dimension of the feature vectors produced by the model
        
        Returns:
            int: Feature dimension
        """
        # Create a dummy input to get feature dimensions
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output = self.model(dummy_input)
            features = output.last_hidden_state[:, 0, :]
        return features.shape[-1]
    
    @contextmanager
    def _memory_efficient_context(self):
        """Context manager for memory-efficient operations on MPS"""
        try:
            yield
        finally:
            if self.device.type == "mps":
                torch.mps.empty_cache()
    
    def _preprocess_image(self, image: Union[str, Image.Image]) -> Image.Image:
        """
        Load and preprocess image
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            PIL.Image: Preprocessed image
        """
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be either a file path or a PIL Image")
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return image
    
    def extract_features(self, image_input: Union[str, Image.Image]) -> np.ndarray:
        """
        Extract features from a single image
        
        Args:
            image_input: Image path or PIL Image object
            
        Returns:
            np.ndarray: Feature vector
        """
        try:
            with self._memory_efficient_context():
                # Preprocess image
                image = self._preprocess_image(image_input)
                
                # Prepare input for model
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Extract features
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                # Get CLS token embedding
                features = outputs.last_hidden_state[:, 0, :]
                
                # Move to CPU and convert to numpy
                features = features.cpu().numpy().flatten()
                
                return features
                
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def batch_extract_features(self, 
                               image_paths: List[str], 
                               batch_size: int = 4,
                               show_progress: bool = True) -> np.ndarray:
        """
        Extract features from multiple images with memory-efficient batch processing
        
        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            np.ndarray: Array of feature vectors
        """
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, len(image_paths), batch_size), 
                                desc="Extracting features")
            except ImportError:
                iterator = range(0, len(image_paths), batch_size)
        else:
            iterator = range(0, len(image_paths), batch_size)
        
        features_list = []
        
        for i in iterator:
            batch_paths = image_paths[i:i+batch_size]
            
            with self._memory_efficient_context():
                batch_features = []
                
                for path in batch_paths:
                    try:
                        features = self.extract_features(path)
                        batch_features.append(features)
                    except Exception as e:
                        logger.warning(f"Failed to process {path}: {e}")
                        # Append zeros for failed images to maintain alignment
                        batch_features.append(np.zeros(self.feature_dim))
                
                if batch_features:
                    features_list.extend(batch_features)
                
                # Force garbage collection after each batch on MPS
                if self.device.type == "mps":
                    gc.collect()
        
        return np.array(features_list)
    
    def extract_features_from_dataset(self, 
                                      dataset_path: str,
                                      output_path: str,
                                      batch_size: int = 4) -> dict:
        """
        Extract features from an entire dataset and save them
        
        Args:
            dataset_path: Path to dataset images
            output_path: Path to save extracted features
            batch_size: Batch size for processing
            
        Returns:
            dict: Mapping of image names to features
        """
        # Get all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        image_files = [
            f for f in os.listdir(dataset_path)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
        
        image_paths = [os.path.join(dataset_path, f) for f in image_files]
        
        # Extract features
        features = self.batch_extract_features(image_paths, batch_size)
        
        # Create feature dictionary
        feature_dict = {
            os.path.splitext(img_file)[0]: feat
            for img_file, feat in zip(image_files, features)
        }
        
        # Save features
        np.save(output_path, feature_dict)
        logger.info(f"Saved features for {len(feature_dict)} images to {output_path}")
        
        return feature_dict