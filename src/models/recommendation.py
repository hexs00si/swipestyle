# src/models/recommendation.py (complete updated version)
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import pickle
import json
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

class RecommendationEngine:
    """
    Recommendation engine using nearest neighbors to find similar items
    """
    def __init__(self, features_dict=None, metadata_dict=None):
        """
        Initialize the recommendation engine
        
        Args:
            features_dict: Dictionary mapping product IDs to feature vectors
            metadata_dict: Dictionary mapping product IDs to metadata
        """
        self.features_dict = features_dict
        self.metadata_dict = metadata_dict
        self.neighbors_model = None
        self.product_ids = None
        self.features_array = None
        
        # If features are provided, build the model
        if features_dict is not None:
            self.build_model()
    
    def build_model(self):
        """Build the nearest neighbors model from the features"""
        # Convert dictionary to arrays for nearest neighbors
        self.product_ids = list(self.features_dict.keys())
        
        # Handle both numpy arrays and lists (from JSON)
        if isinstance(next(iter(self.features_dict.values())), list):
            self.features_array = np.array([self.features_dict[pid] for pid in self.product_ids])
        else:
            self.features_array = np.array([self.features_dict[pid] for pid in self.product_ids])
        
        # Normalize features for cosine similarity
        # When using euclidean distance on normalized vectors, it's equivalent to cosine similarity
        self.features_array = normalize(self.features_array, norm='l2')
        
        # Create and fit nearest neighbors model
        self.neighbors_model = NearestNeighbors(
            n_neighbors=min(10, len(self.features_array)),
            algorithm='brute',
            metric='euclidean'  # Using euclidean on normalized vectors = cosine similarity
        )
        
        self.neighbors_model.fit(self.features_array)
        
        logger.info(f"Built recommendation model with {len(self.product_ids)} products")
        return self
    
    @classmethod
    def load(cls, data_path):
        """
        Load features and metadata from JSON or pickle files
        
        Args:
            data_path: Path to the data file (JSON or pickle)
            
        Returns:
            Initialized RecommendationEngine
        """
        if data_path.endswith('.json'):
            # Load from JSON
            with open(data_path, 'r') as f:
                data = json.load(f)
                features_dict = data['features']
                metadata_dict = data['metadata']
                
                logger.info(f"Loaded data from JSON file: {data_path}")
                logger.info(f"Features: {len(features_dict)} items")
                logger.info(f"Metadata: {len(metadata_dict)} items")
        elif data_path.endswith('.pkl'):
            # Load from pickle
            with open(data_path, 'rb') as f:
                features_dict = pickle.load(f)
            
            # Try to load metadata
            metadata_path = data_path.replace('features.pkl', 'metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata_dict = pickle.load(f)
                    
                logger.info(f"Loaded data from pickle files: {data_path}, {metadata_path}")
                logger.info(f"Features: {len(features_dict)} items")
                logger.info(f"Metadata: {len(metadata_dict)} items")
            else:
                metadata_dict = {}
                logger.warning(f"Metadata file not found: {metadata_path}")
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Create and return engine
        engine = cls(features_dict, metadata_dict)
        return engine
    
    def get_recommendations(self, query_features, num_recommendations=5):
        """
        Get recommendations for a query feature vector
        
        Args:
            query_features: Feature vector for the query image
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of recommendation dictionaries
        """
        if self.neighbors_model is None:
            raise ValueError("Nearest neighbors model not built")
        
        # Reshape and normalize query features
        if len(query_features.shape) == 1:
            query_features = query_features.reshape(1, -1)
        
        # Normalize query features to match the normalized database features
        query_features = normalize(query_features, norm='l2')
        
        # Get nearest neighbors
        distances, indices = self.neighbors_model.kneighbors(query_features)
        
        # Prepare recommendations
        recommendations = []
        for i, idx in enumerate(indices[0]):
            if i >= num_recommendations:
                break
                
            product_id = self.product_ids[idx]
            
            # Convert Euclidean distance on normalized vectors to cosine similarity
            # For normalized vectors: cosine_similarity = 1 - (euclidean_distance^2 / 2)
            euclidean_dist = distances[0][i]
            similarity = 1 - (euclidean_dist ** 2) / 2
            
            # Create recommendation with product info and similarity score
            recommendation = {
                'id': product_id,
                'similarity': float(max(0, similarity))  # Ensure non-negative
            }
            
            # Add metadata if available
            if self.metadata_dict is not None and product_id in self.metadata_dict:
                recommendation.update(self.metadata_dict[product_id])
            
            recommendations.append(recommendation)
        
        return recommendations