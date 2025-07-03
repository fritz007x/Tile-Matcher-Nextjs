from typing import Dict, List, Optional
import os
import logging
import numpy as np
import cv2
from pathlib import Path

class TileMatchingService:
    def __init__(self):
        self.tiles: Dict[str, dict] = {}
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized TileMatchingService")

    def add_tile(self, tile_id: str, image_path: str, metadata: dict):
        """
        Add or update a tile in the matching service.
        """
        try:
            # Read and process the image
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return
                
            # Store the tile metadata and path
            self.tiles[tile_id] = {
                'id': tile_id,
                'image_path': str(image_path),
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error adding tile {tile_id}: {str(e)}")
            raise

    def find_similar_tiles(self, query_image_path: str, top_k: int = 5) -> List[dict]:
        """
        Find similar tiles to the query image.
        Returns a list of matching tiles with similarity scores.
        """
        if not self.tiles:
            return []
            
        try:
            # For now, return a simple list of all tiles
            # In a real implementation, you would implement actual image matching logic here
            return [
                {
                    'tile_id': tile_id,
                    'similarity': 0.9,  # Dummy similarity score
                    **tile_data
                }
                for tile_id, tile_data in list(self.tiles.items())[:top_k]
            ]
            
        except Exception as e:
            self.logger.error(f"Error finding similar tiles: {str(e)}")
            return []

def get_matching_service() -> TileMatchingService:
    """
    Dependency function to get the matching service instance.
    """
    return TileMatchingService()
