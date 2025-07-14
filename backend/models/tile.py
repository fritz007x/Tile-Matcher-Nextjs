from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np

from beanie import Document, Indexed, PydanticObjectId
from pydantic import Field, BaseModel

class FeatureVector(BaseModel):
    """Stores feature vectors for different extraction methods."""
    color_hist: Optional[List[float]] = None
    orb: Optional[List[float]] = None
    vit: Optional[List[float]] = None
    clip: Optional[List[float]] = None
    ensemble: Optional[List[float]] = None

class Tile(Document):
    """
    Represents a single tile in the database with image data and feature vectors.
    """
    # Basic information
    sku: Indexed(str, unique=True)  # type: ignore
    model_name: Indexed(str)  # type: ignore
    collection_name: Indexed(str)  # type: ignore
    
    # Image data
    image_data: bytes = Field(..., description="Raw image binary data")
    image_path: Optional[str] = Field(default=None, description="File path for the image (optional)")
    content_type: str = Field(default="image/jpeg", description="MIME type of the image")
    
    # Feature vectors for different extraction methods
    features: FeatureVector = Field(default_factory=FeatureVector)
    
    # Metadata
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "tiles"  # MongoDB collection name
        use_state_management = True

    def update_timestamp(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()

    async def save(self, *args, **kwargs):
        """Save the document and ensure timestamps are updated."""
        self.update_timestamp()
        await super().save(*args, **kwargs)

    @classmethod
    async def get_by_sku(cls, sku: str) -> Optional['Tile']:
        """Get a tile by its SKU."""
        return await cls.find_one(cls.sku == sku)

    @classmethod
    async def search(
        cls,
        query: str = "",
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        limit: int = 10,
        skip: int = 0
    ) -> List['Tile']:
        """Search for tiles with optional filters."""
        query_filters = []
        
        if query:
            query_filters.append({
                "$or": [
                    {"sku": {"$regex": query, "$options": "i"}},
                    {"model_name": {"$regex": query, "$options": "i"}},
                    {"collection_name": {"$regex": query, "$options": "i"}},
                    {"description": {"$regex": query, "$options": "i"}}
                ]
            })
            
        if model_name:
            query_filters.append({"model_name": model_name})
            
        if collection_name:
            query_filters.append({"collection_name": collection_name})
        
        if query_filters:
            final_filter = {"$and": query_filters} if len(query_filters) > 1 else query_filters[0]
            return await cls.find(final_filter).skip(skip).limit(limit).to_list()
            
        return await cls.find().skip(skip).limit(limit).to_list()
