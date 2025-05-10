import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import pickle
import json
import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class Recommendation:
    """Data class for recommendation results"""
    product_id: str
    similarity: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.product_id,
            'similarity': self.similarity,
            **self.metadata
        }

class SimilarityMetric(ABC):
    """Abstract base class for similarity metrics"""
    
    @abstractmethod
    def calculate(self, query_features: np.ndarray, 
                  database_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate similarity between query and database features"""
        pass

class CosineSimilarity(SimilarityMetric):
    """Cosine similarity implementation"""
    
    def calculate(self, query_features: np.ndarray, 
                  database_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate cosine similarity"""
        # Normalize features
        query_norm = normalize(query_features.reshape(1, -1), norm='l2')
        db_norm = normalize(database_features, norm='l2')
        
        # Calculate cosine similarity
        similarities = np.dot(query_norm, db_norm.T).flatten()
        
        # Sort by similarity (descending)
        indices = np.argsort(similarities)[::-1]
        sorted_similarities = similarities[indices]
        
        return sorted_similarities, indices

class RecommendationEngine:
    """
    Enhanced recommendation engine using nearest neighbors
    """
    
    def __init__(self, 
                 features_dict: Optional[Dict[str, np.ndarray]] = None,
                 metadata_dict: Optional[Dict[str, Dict[str, Any]]] = None,
                 similarity_metric: str = 'cosine'):
        """
        Initialize the recommendation engine
        
        Args:
            features_dict: Dictionary mapping product IDs to feature vectors
            metadata_dict: Dictionary mapping product IDs to metadata
            similarity_metric: Metric to use for similarity ('cosine' or 'euclidean')
        """
        self.features_dict = features_dict
        self.metadata_dict = metadata_dict or {}
        self.neighbors_model = None
        self.product_ids = None
        self.features_array = None
        self.similarity_metric = similarity_metric
        
        # Statistics for monitoring
        self.stats = {
            'total_products': 0,
            'feature_dimension': 0,
            'average_similarity': 0.0
        }
        
        # Build model if data is provided
        if features_dict is not None:
            self.build_model()
    
    def build_model(self) -> None:
        """Build the nearest neighbors model from features"""
        if not self.features_dict:
            raise ValueError("No features provided to build model")
        
        logger.info("Building recommendation model...")
        
        # Convert dictionary to arrays
        self.product_ids = list(self.features_dict.keys())
        
        # Handle both numpy arrays and lists
        feature_values = list(self.features_dict.values())
        if isinstance(feature_values[0], list):
            self.features_array = np.array(feature_values)
        else:
            self.features_array = np.vstack(feature_values)
        
        # Normalize features for better similarity calculation
        self.features_array = normalize(self.features_array, norm='l2')
        
        # Update statistics
        self.stats['total_products'] = len(self.product_ids)
        self.stats['feature_dimension'] = self.features_array.shape[1]
        
        # Create KNN model
        self.neighbors_model = NearestNeighbors(
            n_neighbors=min(50, len(self.features_array)),  # Increased for re-ranking
            algorithm='brute' if len(self.features_array) < 1000 else 'ball_tree',
            metric=self.similarity_metric,
            n_jobs=-1  # Use all CPU cores
        )
        
        self.neighbors_model.fit(self.features_array)
        logger.info(f"Model built with {self.stats['total_products']} products")
    
    def get_recommendations(self,
                            query_features: np.ndarray,
                            num_recommendations: int = 5,
                            filters: Optional[Dict[str, Any]] = None,
                            diversity_weight: float = 0.0) -> List[Recommendation]:
        """
        Get recommendations with optional filtering and diversity
        
        Args:
            query_features: Feature vector for query image
            num_recommendations: Number of recommendations to return
            filters: Optional filters (e.g., {'category': 'dress', 'color': 'blue'})
            diversity_weight: Weight for diversity (0-1, higher = more diverse)
            
        Returns:
            List of Recommendation objects
        """
        if self.neighbors_model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Normalize query features
        query_features = normalize(query_features.reshape(1, -1), norm='l2')
        
        # Get more candidates than needed for filtering/diversity
        k_candidates = min(num_recommendations * 10, len(self.features_array))
        
        # Find nearest neighbors
        distances, indices = self.neighbors_model.kneighbors(
            query_features, 
            n_neighbors=k_candidates
        )
        
        # Convert distances to similarities
        if self.similarity_metric == 'cosine':
            similarities = 1 - distances[0]
        else:  # Euclidean distance
            # Normalize to 0-1 range
            max_dist = np.max(distances[0])
            similarities = 1 - (distances[0] / max_dist)
        
        # Create candidate recommendations
        candidates = []
        for i, idx in enumerate(indices[0]):
            product_id = self.product_ids[idx]
            metadata = self.metadata_dict.get(product_id, {})
            
            # Apply filters if provided
            if filters and not self._matches_filters(metadata, filters):
                continue
            
            candidates.append(Recommendation(
                product_id=product_id,
                similarity=float(similarities[i]),
                metadata=metadata
            ))
        
        # Apply diversity if requested
        if diversity_weight > 0:
            candidates = self._apply_diversity(candidates, num_recommendations, diversity_weight)
        
        # Return top recommendations
        return candidates[:num_recommendations]
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches all filters"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True
    
    def _apply_diversity(self, 
                         candidates: List[Recommendation], 
                         num_recommendations: int,
                         diversity_weight: float) -> List[Recommendation]:
        """Apply diversity to recommendations using MMR algorithm"""
        if not candidates:
            return []
        
        # Maximum Marginal Relevance (MMR) algorithm
        selected = [candidates[0]]  # Start with most similar
        remaining = candidates[1:]
        
        while len(selected) < num_recommendations and remaining:
            # Calculate MMR scores
            mmr_scores = []
            for candidate in remaining:
                # Relevance score (similarity to query)
                relevance = candidate.similarity
                
                # Diversity score (minimum similarity to selected items)
                diversity = min(
                    self._calculate_item_similarity(candidate, sel)
                    for sel in selected
                )
                
                # MMR score
                mmr = (1 - diversity_weight) * relevance - diversity_weight * diversity
                mmr_scores.append(mmr)
            
            # Select item with highest MMR score
            best_idx = np.argmax(mmr_scores)
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
        
        return selected
    
    def _calculate_item_similarity(self, item1: Recommendation, item2: Recommendation) -> float:
        """Calculate similarity between two items based on metadata"""
        # Simple example using category similarity
        if 'category' in item1.metadata and 'category' in item2.metadata:
            return 1.0 if item1.metadata['category'] == item2.metadata['category'] else 0.0
        return 0.0
    
    def get_item_features(self, product_id: str) -> Optional[np.ndarray]:
        """Get features for a specific product"""
        return self.features_dict.get(product_id)
    
    def find_similar_items(self, product_id: str, num_items: int = 5) -> List[Recommendation]:
        """Find items similar to a specific product"""
        features = self.get_item_features(product_id)
        if features is None:
            raise ValueError(f"Product ID {product_id} not found")
        
        # Exclude the query item from results
        recommendations = self.get_recommendations(features, num_items + 1)
        return [rec for rec in recommendations if rec.product_id != product_id][:num_items]
    
    def batch_recommendations(self, 
                              feature_list: List[np.ndarray], 
                              num_recommendations: int = 5) -> List[List[Recommendation]]:
        """Get recommendations for multiple queries efficiently"""
        results = []
        for features in feature_list:
            results.append(self.get_recommendations(features, num_recommendations))
        return results
    
    @classmethod
    def load(cls, data_path: str) -> 'RecommendationEngine':
        """Load recommendation engine from saved data"""
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
                features_dict = {k: np.array(v) for k, v in data['features'].items()}
                metadata_dict = data.get('metadata', {})
        else:  # Pickle format
            with open(data_path, 'rb') as f:
                features_dict = pickle.load(f)
            
            # Try to load metadata
            metadata_path = data_path.replace('features.pkl', 'metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata_dict = pickle.load(f)
            else:
                metadata_dict = {}
        
        return cls(features_dict, metadata_dict)
    
    def save(self, features_path: str, metadata_path: Optional[str] = None) -> None:
        """Save the recommendation engine data"""
        # Save features
        with open(features_path, 'wb') as f:
            pickle.dump(self.features_dict, f)
        
        # Save metadata if path provided
        if metadata_path and self.metadata_dict:
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata_dict, f)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the recommendation engine"""
        return self.stats.copy()