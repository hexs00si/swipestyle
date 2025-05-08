import os
import cloudinary
import cloudinary.uploader
import cloudinary.api
from PIL import Image
import io
import logging
from src.config import CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET

# Set up logging
logger = logging.getLogger(__name__)

class CloudinaryHelper:
    """Helper class for Cloudinary integration"""
    
    def __init__(self):
        """Initialize Cloudinary with credentials from environment variables"""
        # Validate credentials
        if not all([CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET]):
            raise ValueError(
                "Cloudinary credentials not provided. Set CLOUDINARY_CLOUD_NAME, "
                "CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET environment variables."
            )
        
        # Configure Cloudinary
        cloudinary.config(
            cloud_name=CLOUDINARY_CLOUD_NAME,
            api_key=CLOUDINARY_API_KEY,
            api_secret=CLOUDINARY_API_SECRET
        )
        
        logger.info("Cloudinary configured successfully")
    
    def upload_image(self, image_path, public_id=None, folder="fashion-recommender"):
        """
        Upload an image to Cloudinary
        
        Args:
            image_path: Path to the image file
            public_id: Optional public ID for the image (defaults to filename without extension)
            folder: Folder in Cloudinary to store the image
            
        Returns:
            Cloudinary URL for the uploaded image
        """
        if not public_id:
            # Use filename without extension as public_id
            public_id = os.path.splitext(os.path.basename(image_path))[0]
        
        # Upload the image
        result = cloudinary.uploader.upload(
            image_path,
            public_id=public_id,
            folder=folder,
            overwrite=True
        )
        
        return result['secure_url']
    
    def upload_image_data(self, image_data, public_id, folder="fashion-recommender"):
        """
        Upload image data to Cloudinary
        
        Args:
            image_data: PIL Image or bytes
            public_id: Public ID for the image
            folder: Folder in Cloudinary to store the image
            
        Returns:
            Cloudinary URL for the uploaded image
        """
        if isinstance(image_data, Image.Image):
            # Convert PIL Image to bytes
            buffer = io.BytesIO()
            image_data.save(buffer, format="JPEG")
            image_data = buffer.getvalue()
        
        # Upload the image
        result = cloudinary.uploader.upload(
            image_data,
            public_id=public_id,
            folder=folder,
            overwrite=True
        )
        
        return result['secure_url']
    
    def bulk_upload_directory(self, directory_path, folder="fashion-recommender"):
        """
        Upload all images in a directory to Cloudinary
        
        Args:
            directory_path: Path to the directory containing images
            folder: Folder in Cloudinary to store the images
            
        Returns:
            Dictionary mapping filenames to Cloudinary URLs
        """
        url_mapping = {}
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                image_path = os.path.join(directory_path, filename)
                public_id = os.path.splitext(filename)[0]
                
                try:
                    url = self.upload_image(image_path, public_id, folder)
                    url_mapping[filename] = url
                    logger.info(f"Uploaded {filename} to Cloudinary: {url}")
                except Exception as e:
                    logger.error(f"Error uploading {filename}: {e}")
        
        return url_mapping