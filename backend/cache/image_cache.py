"""
Image caching system for thumbnails and Base64 data.
"""

import asyncio
import base64
import hashlib
import logging
from io import BytesIO
from typing import Optional, Dict, Any, Tuple
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

class ImageCache:
    """In-memory cache for image thumbnails and Base64 data."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.thumbnails: Dict[str, bytes] = {}
        self.base64_cache: Dict[str, str] = {}
        self._lock = asyncio.Lock()
    
    def _make_thumbnail_key(self, tile_id: str, width: int, height: int) -> str:
        """Create cache key for thumbnail."""
        return f"thumb_{tile_id}_{width}x{height}"
    
    def _make_base64_key(self, tile_id: str) -> str:
        """Create cache key for base64 data."""
        return f"b64_{tile_id}"
    
    async def get_thumbnail(self, tile_id: str, width: int, height: int) -> Optional[bytes]:
        """Get cached thumbnail."""
        key = self._make_thumbnail_key(tile_id, width, height)
        return self.thumbnails.get(key)
    
    async def set_thumbnail(self, tile_id: str, width: int, height: int, data: bytes):
        """Cache thumbnail data."""
        async with self._lock:
            if len(self.thumbnails) >= self.max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.thumbnails))
                del self.thumbnails[oldest_key]
            
            key = self._make_thumbnail_key(tile_id, width, height)
            self.thumbnails[key] = data
    
    async def get_base64_image(self, tile_id: str) -> Optional[str]:
        """Get cached base64 image data."""
        key = self._make_base64_key(tile_id)
        return self.base64_cache.get(key)
    
    async def set_base64_image(self, tile_id: str, data: str):
        """Cache base64 image data."""
        async with self._lock:
            if len(self.base64_cache) >= self.max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.base64_cache))
                del self.base64_cache[oldest_key]
            
            key = self._make_base64_key(tile_id)
            self.base64_cache[key] = data
    
    async def generate_and_cache_thumbnail(self, tile_id: str, image_data: bytes, 
                                         width: int = 200, height: int = 200) -> Optional[bytes]:
        """
        Generate thumbnail from image data and cache it.
        
        Args:
            tile_id: Unique identifier for the tile
            image_data: Raw image bytes
            width: Thumbnail width
            height: Thumbnail height
            
        Returns:
            Thumbnail bytes or None if generation failed
        """
        try:
            # Check cache first
            cached_thumbnail = await self.get_thumbnail(tile_id, width, height)
            if cached_thumbnail:
                return cached_thumbnail
            
            # Generate thumbnail
            img = Image.open(BytesIO(image_data))
            
            # Convert to RGB if necessary (for PNG with transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Generate thumbnail with high quality
            img.thumbnail((width, height), Image.Resampling.LANCZOS)
            
            # Save as JPEG with optimization
            buffered = BytesIO()
            img.save(buffered, format='JPEG', quality=85, optimize=True)
            thumbnail_data = buffered.getvalue()
            
            # Cache the result
            await self.set_thumbnail(tile_id, width, height, thumbnail_data)
            
            return thumbnail_data
            
        except Exception as e:
            logger.error(f"Failed to generate thumbnail for tile {tile_id}: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached data."""
        self.thumbnails.clear()
        self.base64_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "thumbnails_count": len(self.thumbnails),
            "base64_count": len(self.base64_cache),
            "total_items": len(self.thumbnails) + len(self.base64_cache)
        }

# Global cache instance
cache_manager = ImageCache()