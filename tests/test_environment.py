# tests/test_environment.py
import sys
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import logging
import cloudinary
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import configuration
from src.config import CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET
from src.utils.cloudinary_helper import CloudinaryHelper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('environment_test.log')
    ]
)
logger = logging.getLogger(__name__)

def test_hardware_acceleration():
    """Test hardware acceleration options"""
    logger.info("\n=== Hardware Acceleration ===")
    
    acceleration_info = {
        "CUDA": torch.cuda.is_available(),
        "MPS": torch.backends.mps.is_available(),
        "MPS Built": torch.backends.mps.is_built()
    }
    
    for name, status in acceleration_info.items():
        logger.info(f"{name}: {status}")
    
    # Determine best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    return device

def test_tensor_operations(device):
    """Test basic tensor operations on target device"""
    logger.info("\n=== Tensor Operations Test ===")
    try:
        x = torch.rand(2, 3, device=device)
        y = torch.rand(2, 3, device=device)
        z = x + y
        logger.info(f"Tensor operations working correctly on {device}")
        logger.debug(f"Sample tensor result:\n{z}")
        return True
    except Exception as e:
        logger.error(f"Tensor operations error: {e}")
        return False

def test_image_processing():
    """Test image processing dependencies"""
    logger.info("\n=== Image Processing Test ===")
    try:
        # Test PIL
        img = Image.new('RGB', (100, 100))
        logger.info("PIL working correctly")
        
        # Test image processing pipeline
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(test_img)
        logger.info("Image conversion working")
        return True
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return False

def test_cloudinary():
    """Test Cloudinary configuration"""
    logger.info("\n=== Cloudinary Test ===")
    try:
        if all([CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET]):
            cloudinary.config(
                cloud_name=CLOUDINARY_CLOUD_NAME,
                api_key=CLOUDINARY_API_KEY,
                api_secret=CLOUDINARY_API_SECRET
            )
            
            # Test connection
            result = cloudinary.api.ping()
            logger.info(f"Cloudinary ping successful: {result}")
            
            # Test helper class
            helper = CloudinaryHelper()
            logger.info("CloudinaryHelper initialized")
            return True
        else:
            logger.warning("Cloudinary credentials not set. Skipping tests.")
            return True  # Not a critical failure
    except Exception as e:
        logger.error(f"Cloudinary error: {e}")
        return False

def test_fashion_model(device):
    """Test DINOv2 model for fashion recommendations"""
    logger.info("\n=== Fashion Model Test ===")
    try:
        from transformers import AutoModel, AutoImageProcessor
        
        # Load model and processor with progress logging
        logger.info("Loading DINOv2 model components...")
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
        logger.info("Model loaded successfully")
        
        # Test with fashion-relevant image size
        dummy_image = torch.rand(1, 3, 224, 224, device=device)
        
        # Run inference with timing
        with torch.no_grad():
            logger.info("Running inference...")
            inputs = {"pixel_values": dummy_image}
            outputs = model(**inputs)
            logger.info(f"Feature extraction successful. Output shape: {outputs.last_hidden_state.shape}")
        
        # Memory cleanup
        del model, processor, outputs
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("Memory management working")
        return True
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        return False

def test_environment():
    """Comprehensive environment test for fashion recommendation system"""
    logger.info("Starting comprehensive environment test...")
    
    # System and package info
    logger.info("\n=== System Information ===")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"NumPy: {np.__version__}")
    logger.info(f"Pandas: {pd.__version__}")
    
    # Run tests
    device = test_hardware_acceleration()
    tests = {
        "Tensor Operations": test_tensor_operations(device),
        "Image Processing": test_image_processing(),
        "Cloudinary": test_cloudinary(),
        "Fashion Model": test_fashion_model(device)
    }
    
    # Summary
    logger.info("\n=== Test Summary ===")
    all_passed = all(tests.values())
    for name, passed in tests.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{name}: {status}")
    
    if all_passed:
        logger.info("\nEnvironment test completed successfully!")
    else:
        logger.error("\nEnvironment test completed with failures!")
    
    return all_passed

if __name__ == "__main__":
    successful = test_environment()
    sys.exit(0 if successful else 1)