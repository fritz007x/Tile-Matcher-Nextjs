import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import cv2
from PIL import Image
import logging
import pickle
import time

# Try to import faiss, but provide graceful fallback if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("FAISS not available - using fallback matching methods only")
    FAISS_AVAILABLE = False

from .feature_extractors import get_feature_extractor, BaseFeatureExtractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    tile_id: str
    score: float
    method: str
    metadata: dict

class TileMatchingService:
    def __init__(self, methods: List[str] = ['sift', 'orb', 'kaze', 'vit']):
        """
        Initialize the tile matching service with specified feature extraction methods.
        
        Args:
            methods: List of methods to use for feature extraction. 
                    Options: 'sift', 'orb', 'kaze', 'vit'
        """
        self.methods = methods
        self.extractors = {method: get_feature_extractor(method) for method in methods}
        self.tile_features = {method: {} for method in methods}  # In-memory storage for tile features
        self.index = None
        self.tile_metadata = {}
        
    def add_tile(self, tile_id: str, image_path: str, metadata: Optional[dict] = None):
        """
        Add a tile to the matching service.
        
        Args:
            tile_id: Unique identifier for the tile
            image_path: Path to the tile image
            metadata: Optional metadata about the tile (e.g., SKU, model_name, collection_name)
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return False
        
        try:
            # Read and preprocess image
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
                
            # Store metadata
            self.tile_metadata[tile_id] = metadata or {}
            
            # Extract features with all methods
            for method, extractor in self.extractors.items():
                try:
                    features = extractor.extract(image)
                    self.tile_features[method][tile_id] = features
                    logger.info(f"Extracted {method} features for tile {tile_id}")
                except Exception as e:
                    logger.error(f"Error extracting {method} features for tile {tile_id}: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding tile {tile_id}: {str(e)}")
            return False
    
    def match_image(self, query_image: np.ndarray, top_k: int = 5) -> List[MatchResult]:
        """
        Match a query image against all stored tiles.
        
        Args:
            query_image: Input image as numpy array (BGR format)
            top_k: Number of top matches to return
            
        Returns:
            List of MatchResult objects sorted by score (highest first)
        """
        if query_image is None or query_image.size == 0:
            raise ValueError("Query image is empty or invalid")
        
        all_results = []
        
        # Process with each method
        for method, extractor in self.extractors.items():
            try:
                # Extract features from query image
                query_features = extractor.extract(query_image)
                
                # Compare with all stored tiles for this method
                for tile_id, tile_features in self.tile_features[method].items():
                    try:
                        score = extractor.match(query_features, tile_features)
                        all_results.append(MatchResult(
                            tile_id=tile_id,
                            score=score,
                            method=method,
                            metadata=self.tile_metadata.get(tile_id, {})
                        ))
                    except Exception as e:
                        logger.error(f"Error matching with {method} for tile {tile_id}: {str(e)}")
                        
            except Exception as e:
                logger.error(f"Error processing with {method}: {str(e)}")
        
        # Group results by tile_id and calculate weighted average score
        tile_scores = {}
        for result in all_results:
            if result.tile_id not in tile_scores:
                tile_scores[result.tile_id] = {
                    'scores': [],
                    'methods': [],
                    'metadata': result.metadata
                }
            tile_scores[result.tile_id]['scores'].append(result.score)
            tile_scores[result.tile_id]['methods'].append(result.method)
        
        # Calculate weighted average score for each tile
        final_results = []
        for tile_id, data in tile_scores.items():
            avg_score = np.mean(data['scores'])
            final_results.append(MatchResult(
                tile_id=tile_id,
                score=avg_score,
                method=", ".join(data['methods']),
                metadata=data['metadata']
            ))
        
        # Sort by score in descending order and return top_k results
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:top_k]
    
    def build_faiss_index(self):
        """
        Build a FAISS index for faster similarity search.
        This is particularly useful for large datasets.
        """
        # Check if FAISS is available
        if not FAISS_AVAILABLE:
            logger.warning("FAISS is not available - skipping index building")
            return
            
        # Only implemented for ViT features for now
        if 'vit' not in self.extractors:
            logger.warning("FAISS index is only supported for ViT features")
            return
            
        # Get all ViT features
        vit_features = []
        tile_ids = []
        extractor = self.extractors['vit']
        
        for tile_id, features in self.tile_features['vit'].items():
            if 'features' in features:
                vit_features.append(features['features'])
                tile_ids.append(tile_id)
        
        if not vit_features:
            logger.warning("No ViT features found for building FAISS index")
            return
            
        # Convert to numpy array
        vit_features = np.array(vit_features).astype('float32')
        
        # Build and train the index
        dimension = vit_features.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(vit_features)
        self.index_tile_ids = tile_ids
        
        logger.info(f"Built FAISS index with {len(tile_ids)} vectors")
    
    def search_with_faiss(self, query_features: np.ndarray, top_k: int = 5) -> List[MatchResult]:
        """
        Search for similar tiles using FAISS index.
        
        Args:
            query_features: Query features from ViT extractor
            top_k: Number of results to return
            
        Returns:
            List of MatchResult objects
        """
        # Check if FAISS is available
        if not FAISS_AVAILABLE:
            logger.warning("FAISS is not available - unable to perform index search")
            return []
            
        if self.index is None or 'vit' not in self.extractors:
            logger.warning("FAISS index not built or ViT extractor not available")
            return []
            
        if 'features' not in query_features:
            logger.error("Query features must be extracted with ViT extractor")
            return []
            
        # Convert query features to numpy array
        query_vector = query_features['features'].reshape(1, -1).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_vector, top_k)
        
        # Convert to MatchResult objects
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:  # No more results
                continue
                
            tile_id = self.index_tile_ids[idx]
            # Convert L2 distance to similarity score (higher is better)
            score = 1.0 / (1.0 + distance)
            
            results.append(MatchResult(
                tile_id=tile_id,
                score=score,
                method="vit_faiss",
                metadata=self.tile_metadata.get(tile_id, {})
            ))
        
        return results

def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load an image from file path"""
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Preprocess image for feature extraction"""
    # Resize while maintaining aspect ratio
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_size = (int(w * scale), int(h * scale))
    
    # Resize
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    
    # Pad to target size
    delta_w = target_size[1] - new_size[0]
    delta_h = target_size[0] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    color = [0, 0, 0]
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                               cv2.BORDER_CONSTANT, value=color)
    
    return padded
