import os
import io
import json
import pickle
import base64
import logging
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from src.config import FASHION_DATA_PATH, PROCESSED_DATA_DIR
from src.models.feature_extractor import FeatureExtractor
from src.models.recommendation import RecommendationEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Fashion Recommendation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and data
feature_extractor = None
recommendation_engine = None
metadata_dict = None
cloudinary_urls = {}

# Request/Response models
class ImageUploadRequest(BaseModel):
    image: str  # Base64 encoded image
    num_recommendations: Optional[int] = 5

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]

# Load models and data at startup
@app.on_event("startup")
async def startup_event():
    global feature_extractor, recommendation_engine, metadata_dict, cloudinary_urls
    
    try:
        # Load feature extractor
        logger.info("Loading feature extractor...")
        feature_extractor = FeatureExtractor()
        
        # Load recommendation data
        logger.info("Loading recommendation data...")
        if os.path.exists(FASHION_DATA_PATH):
            # Load from JSON
            with open(FASHION_DATA_PATH, 'r') as f:
                data = json.load(f)
                features_dict = {k: np.array(v) for k, v in data['features'].items()}
                metadata_dict = data['metadata']
                cloudinary_urls = data.get('cloudinary_urls', {})
            
            # Initialize recommendation engine
            recommendation_engine = RecommendationEngine(features_dict, metadata_dict)
            logger.info(f"Loaded {len(features_dict)} products")
        else:
            logger.error("No data file found!")
            
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

# Clean up MPS memory after operations
def cleanup_memory():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# API endpoints
@app.get("/")
async def root():
    """Basic API info"""
    return {
        "message": "Fashion Recommendation API",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend (POST)",
            "upload": "/upload-file (POST)",
            "image": "/image/{product_id}"
        }
    }

@app.get("/health")
async def health_check():
    """Check if API is healthy"""
    return {
        "status": "healthy" if feature_extractor and recommendation_engine else "unhealthy",
        "models_loaded": bool(feature_extractor and recommendation_engine),
        "products_count": len(metadata_dict) if metadata_dict else 0
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: ImageUploadRequest):
    """Get recommendations from base64 image"""
    if not feature_extractor or not recommendation_engine:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        
        # Extract features
        features = feature_extractor.extract_features(image)
        
        # Get recommendations
        recommendations = recommendation_engine.get_recommendations(
            features, 
            num_recommendations=request.num_recommendations
        )
        
        # Add image URLs
        for rec in recommendations:
            product_id = str(rec['id'])
            if product_id in cloudinary_urls:
                rec['image_url'] = cloudinary_urls[product_id]
            else:
                rec['image_url'] = f"/image/{product_id}"
        
        cleanup_memory()
        return {"recommendations": recommendations}
        
    except Exception as e:
        cleanup_memory()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-file", response_model=RecommendationResponse)
async def upload_file(file: UploadFile = File(...), num_recommendations: int = 5):
    """Get recommendations from uploaded file"""
    if not feature_extractor or not recommendation_engine:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Extract features
        features = feature_extractor.extract_features(image)
        
        # Get recommendations
        recommendations = recommendation_engine.get_recommendations(
            features, 
            num_recommendations=num_recommendations
        )
        
        # Add image URLs
        for rec in recommendations:
            product_id = str(rec['id'])
            if product_id in cloudinary_urls:
                rec['image_url'] = cloudinary_urls[product_id]
            else:
                rec['image_url'] = f"/image/{product_id}"
        
        cleanup_memory()
        return {"recommendations": recommendations}
        
    except Exception as e:
        cleanup_memory()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image/{product_id}")
async def get_product_image(product_id: str):
    """Get product image by ID"""
    # Check if we have a Cloudinary URL
    if product_id in cloudinary_urls:
        return RedirectResponse(url=cloudinary_urls[product_id])
    
    # Check local files
    if metadata_dict and product_id in metadata_dict:
        filename = metadata_dict[product_id].get('filename')
        if filename:
            image_path = os.path.join(PROCESSED_DATA_DIR, "images", filename)
            if os.path.exists(image_path):
                return FileResponse(image_path)
    
    raise HTTPException(status_code=404, detail="Image not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)