from typing import Dict, List, Optional, Tuple
import os
import logging
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import hashlib
from fastapi import UploadFile

class TileMatchingService:
    def __init__(self, upload_dir: str = None):
        self.tiles: Dict[str, dict] = {}
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing TileMatchingService")
        self.methods = ['vit', 'color_hist', 'orb']
        
        # Set up uploads directory
        if upload_dir is None:
            # Default to backend/api/uploads relative to this file
            current_dir = Path(__file__).parent
            self.upload_dir = current_dir.parent.parent / 'api' / 'uploads'
        else:
            self.upload_dir = Path(upload_dir)
            
        # Create uploads directory if it doesn't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing images
        self._load_images_from_uploads()
        self.logger.info(f"Initialized TileMatchingService with {len(self.tiles)} tiles")

    def add_tile(self, tile_id: str, image_path: str, metadata: dict, image_data: bytes = None):
        """
        Add a new tile to the matching service.
        
        Args:
            tile_id: Unique identifier for the tile
            image_path: Path to the image file or identifier
            metadata: Additional metadata about the tile
            image_data: Optional raw image data (bytes) to store as base64
        """
        try:
            tile_data = {
                'id': tile_id,
                'image_path': image_path,
                'metadata': metadata or {}
            }
            
            # If image_data is provided, store it as base64 in metadata
            if image_data is not None:
                import base64
                tile_data['image_data'] = base64.b64encode(image_data).decode('utf-8')
            
            self.tiles[tile_id] = tile_data
            self.logger.info(f"Added tile {tile_id} with metadata: {metadata}")
            
        except Exception as e:
            self.logger.error(f"Error adding tile {tile_id}: {str(e)}")
            raise

    def _load_images_from_uploads(self):
        """Load all images from the uploads directory into the service."""
        try:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = [
                f for f in self.upload_dir.glob('*')
                if f.suffix.lower() in image_extensions and f.is_file()
            ]
            
            for img_path in image_files:
                try:
                    tile_id = hashlib.md5(str(img_path).encode()).hexdigest()
                    metadata = {
                        'filename': img_path.name,
                        'uploaded_at': img_path.stat().st_mtime,
                        'content_type': f'image/{img_path.suffix[1:].lower()}'  # e.g., 'image/jpg'
                    }
                    
                    # Read the image data
                    with open(img_path, 'rb') as f:
                        image_data = f.read()
                    
                    # Store both the path and the base64-encoded image data
                    self.add_tile(tile_id, str(img_path), metadata, image_data)
                except Exception as e:
                    self.logger.error(f"Error loading image {img_path}: {str(e)}")
                    
            self.logger.info(f"Loaded {len(image_files)} images from {self.upload_dir}")
            
        except Exception as e:
            self.logger.error(f"Error loading images from uploads: {str(e)}")
            
    def get_tile_image_data(self, tile_id: str) -> Optional[dict]:
        """
        Get the base64-encoded image data for a tile.
        
        Args:
            tile_id: The ID of the tile
            
        Returns:
            dict containing 'data' (base64 string) and 'content_type', or None if not found
        """
        tile = self.tiles.get(tile_id)
        if not tile:
            return None
            
        # If we have the image data already (from add_tile with image_data)
        if 'image_data' in tile:
            return {
                'data': tile['image_data'],
                'content_type': tile.get('metadata', {}).get('content_type', 'image/jpeg')
            }
            
        # Otherwise, try to read from the file system
        try:
            with open(tile['image_path'], 'rb') as f:
                image_data = f.read()
                
            import base64
            return {
                'data': base64.b64encode(image_data).decode('utf-8'),
                'content_type': tile.get('metadata', {}).get('content_type', 'image/jpeg')
            }
        except Exception as e:
            self.logger.error(f"Error reading image data for tile {tile_id}: {str(e)}")
            return None
    
    def _extract_features(self, image_path: str, method: str = 'color_hist') -> np.ndarray:
        """Extract features from an image using the specified method."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
                
            if method == 'color_hist':
                # Simple color histogram (3D - 8 bins per channel)
                hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                return cv2.normalize(hist, hist).flatten()
                
            elif method == 'orb':
                # ORB features
                orb = cv2.ORB_create()
                kp, des = orb.detectAndCompute(img, None)
                return des.flatten() if des is not None else np.array([])
                
            else:
                # Default to VIT (dummy implementation - would use a real model in production)
                return np.random.rand(768)  # Dummy 768-dim vector
                
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return np.array([])
    
    def _calculate_similarity(self, feat1: np.ndarray, feat2: np.ndarray, method: str = 'cosine') -> float:
        """Calculate similarity between two feature vectors."""
        if feat1.size == 0 or feat2.size == 0:
            return 0.0
            
        if method == 'cosine':
            # Cosine similarity
            return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-10)
        else:
            # Default to Euclidean distance (inverted to similarity)
            return 1.0 / (1.0 + np.linalg.norm(feat1 - feat2))
    
    def find_similar_tiles(self, query_image_path: str, top_k: int = 5, method: str = 'color_hist', threshold: float = 0.0) -> List[dict]:
        """
        Find similar tiles to the query image.
        
        Args:
            query_image_path: Path to the query image
            top_k: Maximum number of matches to return
            method: Feature extraction method to use ('color_hist', 'orb', 'vit')
            threshold: Minimum similarity score (0-1) for a match to be included
            
        Returns:
            List of matching tiles with similarity scores, sorted by score (highest first)
        """
        if not self.tiles:
            self.logger.warning("No tiles available for matching")
            return []
            
        try:
            # Extract features from query image
            query_features = self._extract_features(query_image_path, method)
            if query_features.size == 0:
                self.logger.error("Failed to extract features from query image")
                return []
            
            # Compare with all tiles
            similarities = []
            for tile_id, tile_data in self.tiles.items():
                tile_features = self._extract_features(tile_data['image_path'], method)
                if tile_features.size > 0:
                    score = float(self._calculate_similarity(query_features, tile_features))
                    if score >= threshold:
                        similarities.append((tile_id, score))
            
            # Sort by score (highest first) and take top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Prepare results
            results = []
            for tile_id, score in similarities[:top_k]:
                tile_data = self.tiles[tile_id].copy()
                results.append({
                    'id': tile_id,
                    'similarity': score,
                    **tile_data
                })
            
            self.logger.info(f"Found {len(results)} matches for {query_image_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in find_similar_tiles: {str(e)}", exc_info=True)
            return []

# Global instance of the matching service
_matching_service = None

def get_matching_service() -> TileMatchingService:
    """
    Dependency function to get the matching service instance (singleton).
    """
    global _matching_service
    if _matching_service is None:
        _matching_service = TileMatchingService()
    return _matching_service
