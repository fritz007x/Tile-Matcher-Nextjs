"""
Utility functions for image data retrieval and processing.
"""

import base64
import logging
from typing import Optional, Dict, Any
from bson import ObjectId
from backend.models.tile import Tile

logger = logging.getLogger(__name__)

def validate_objectid(tile_id: str) -> bool:
    """Validate if a string is a valid MongoDB ObjectId."""
    try:
        ObjectId(tile_id)
        return True
    except Exception:
        return False

async def get_tile_image_data(tile_id: str, matching_service=None) -> Optional[Dict[str, Any]]:
    """
    Multi-layer image data retrieval with fallback mechanisms.
    
    Args:
        tile_id: The tile ID to lookup
        matching_service: Optional matching service instance
        
    Returns:
        Dict with tile_id, content_type, and data (base64) or None if not found
    """
    
    # First try: Database lookup for ObjectId-based tiles
    if validate_objectid(tile_id):
        try:
            tile = await Tile.get(tile_id)
            if tile and tile.image_data:
                b64_data = base64.b64encode(tile.image_data).decode('utf-8')
                return {
                    "tile_id": str(tile.id),
                    "content_type": tile.content_type or 'image/jpeg',
                    "data": b64_data
                }
        except Exception as e:
            logger.debug(f"Database lookup failed for tile {tile_id}: {e}")
    
    # Second try: Matching service lookup (for hash-based IDs)
    if matching_service and hasattr(matching_service, 'tiles') and tile_id in matching_service.tiles:
        try:
            tile_data = matching_service.tiles[tile_id]
            if 'image_data' in tile_data and tile_data['image_data']:
                return {
                    "tile_id": tile_id,
                    "content_type": tile_data.get('content_type', 'image/jpeg'),
                    "data": tile_data['image_data']  # Already base64 encoded
                }
        except Exception as e:
            logger.debug(f"Matching service lookup failed for tile {tile_id}: {e}")
    
    logger.warning(f"Image data not found for tile {tile_id}")
    return None

def ensure_image_format(img_format: str) -> str:
    """Ensure image format is supported, default to JPEG."""
    if img_format.lower() in ['jpeg', 'jpg', 'png', 'webp']:
        return img_format.lower()
    return 'jpeg'

def get_mime_type(img_format: str) -> str:
    """Get proper MIME type for image format."""
    format_map = {
        'jpeg': 'image/jpeg',
        'jpg': 'image/jpeg', 
        'png': 'image/png',
        'webp': 'image/webp'
    }
    return format_map.get(img_format.lower(), 'image/jpeg')