# src/models/text_embedder.py
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Union, Optional
import gc
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class TextEmbedder:
    """
    Text embedding using sentence-transformers for fashion item descriptions
    Optimized for Apple Silicon (MPS) with proper memory management
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initialize text embedder with sentence-transformers model
        
        Args:
            model_name: Name of the sentence-transformers model to use
            device: Specific device to use (optional)
        """
        logger.info(f"Initializing text embedder with model: {model_name}")
        
        # Set device with MPS support
        self.device = self._get_device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            
            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Text embedder initialized. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_device(self, preferred_device: Optional[str] = None) -> str:
        """
        Get the best available device for computation
        
        Args:
            preferred_device: User-specified device preference
            
        Returns:
            str: Selected device
        """
        if preferred_device:
            return preferred_device
            
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    @contextmanager
    def _memory_efficient_context(self):
        """Context manager for memory-efficient operations on MPS"""
        try:
            yield
        finally:
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
    
    def create_item_text(self, metadata: dict) -> str:
        """
        Create text description from item metadata
        
        Args:
            metadata: Dictionary containing item metadata
            
        Returns:
            str: Combined text description
        """
        # Extract relevant fields
        gender = metadata.get('gender', '').lower()
        category = metadata.get('master_category', '').lower()
        subcategory = metadata.get('sub_category', '').lower()
        article_type = metadata.get('article_type', '').lower()
        color = metadata.get('base_color', '').lower()
        usage = metadata.get('usage', '').lower()
        product_name = metadata.get('product_name', '').lower()
        
        # Create a natural text description
        parts = []
        
        # Add color if available
        if color and color != 'unknown':
            parts.append(color)
        
        # Add article type
        if article_type and article_type != 'unknown':
            parts.append(article_type)
        
        # Add gender if relevant
        if gender and gender != 'unknown' and gender != 'unisex':
            parts.append(f"for {gender}")
        
        # Add usage if available
        if usage and usage != 'unknown':
            parts.append(f"for {usage} wear")
        
        # Combine parts
        description = " ".join(parts)
        
        # Add product name if it provides additional info
        if product_name and product_name != 'unknown':
            description = f"{description}. {product_name}"
        
        return description.strip()
    
    def embed_text(self, text: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for text with memory-efficient batch processing
        
        Args:
            text: Single text or list of texts to embed
            batch_size: Number of texts to process at once (for memory efficiency)
            
        Returns:
            np.ndarray: Text embeddings
        """
        with self._memory_efficient_context():
            if isinstance(text, str):
                text = [text]
            
            # Process in batches for large inputs
            if len(text) > batch_size:
                embeddings_list = []
                
                for i in range(0, len(text), batch_size):
                    batch = text[i:i + batch_size]
                    batch_embeddings = self.model.encode(
                        batch, 
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                    embeddings_list.append(batch_embeddings)
                    
                    # Memory cleanup for MPS
                    if self.device == "mps" and i % (batch_size * 4) == 0:
                        gc.collect()
                
                embeddings = np.vstack(embeddings_list)
            else:
                embeddings = self.model.encode(
                    text, 
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            
            return embeddings
    
    def embed_metadata(self, metadata_dict: dict, batch_size: int = 32) -> dict:
        """
        Generate text embeddings for all items in metadata
        
        Args:
            metadata_dict: Dictionary of metadata for all items
            batch_size: Number of items to process at once
            
        Returns:
            dict: Dictionary mapping item IDs to text embeddings
        """
        logger.info("Generating text embeddings for all items...")
        
        # Create text descriptions
        texts = []
        item_ids = []
        
        for item_id, metadata in metadata_dict.items():
            text = self.create_item_text(metadata)
            texts.append(text)
            item_ids.append(item_id)
        
        # Generate embeddings with batch processing
        embeddings = self.embed_text(texts, batch_size=batch_size)
        
        # Create mapping
        text_embeddings = {
            item_id: embedding 
            for item_id, embedding in zip(item_ids, embeddings)
        }
        
        logger.info(f"Generated text embeddings for {len(text_embeddings)} items")
        return text_embeddings