# src/config.py

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Base directory paths using Path for cross-platform compatibility
BASE_DIR = Path(__file__).resolve().parent.parent  # Gets the project root directory
DATA_DIR = BASE_DIR / "data"  # Path to data directory
RAW_DATA_DIR = DATA_DIR / "raw"  # Path to raw data
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # Path to processed data
DEPLOYMENT_DATA_DIR = DATA_DIR / "deployment"  # Path to deployment data

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(DEPLOYMENT_DATA_DIR, exist_ok=True)

# Kaggle settings - for dataset download
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# Cloudinary settings - for image hosting
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# API settings
API_HOST = os.getenv("API_HOST", "127.0.0.1")  # Default to localhost if not specified
API_PORT = int(os.getenv("API_PORT", "8000"))  # Convert string to integer

# UI settings
UI_PORT = int(os.getenv("UI_PORT", "8501"))  # Streamlit default port

# Model settings
MODEL_NAME = os.getenv("MODEL_NAME", "facebook/dinov2-small")  # Default model
SUBSET_SIZE = int(os.getenv("SUBSET_SIZE", "500"))  # Number of products to use

# File paths for important data files
METADATA_CSV_PATH = PROCESSED_DATA_DIR / "metadata.csv"
FEATURES_PATH = DEPLOYMENT_DATA_DIR / "features.pkl"
METADATA_PATH = DEPLOYMENT_DATA_DIR / "metadata.pkl"
FASHION_DATA_PATH = DEPLOYMENT_DATA_DIR / "fashion_data.json"
CLOUDINARY_URLS_PATH = DEPLOYMENT_DATA_DIR / "cloudinary_urls.json"