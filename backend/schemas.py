from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional

class TileBase(BaseModel):
    """Base schema for a tile."""
    sku: str
    model_name: str
    collection_name: str
    image_path: Optional[str] = None
    content_type: str = "image/jpeg"
    created_at: datetime
    updated_at: datetime
    description: Optional[str] = None

class TileCreate(TileBase):
    """Schema used for creating a new tile in the database."""
    pass

from bson import ObjectId

class TileResponse(TileBase):
    """Schema for returning a tile from the API."""
    id: str
    has_image_data: bool = False
    
    class Config:
        from_attributes = True  # Replaces orm_mode in Pydantic v2
        populate_by_name = True
        json_encoders = {
            ObjectId: str,
            'bytes': lambda v: v.decode('utf-8') if isinstance(v, bytes) else v
        }
    
    @classmethod
    def from_mongo(cls, data: dict) -> 'TileResponse':
        """Convert MongoDB document to TileResponse."""
        # Handle _id to id conversion
        if '_id' in data:
            if isinstance(data['_id'], ObjectId):
                data['id'] = str(data['_id'])
            else:
                data['id'] = str(data['_id'])
            # Remove the _id field after conversion
            data.pop('_id', None)
        
        # Handle image_data field
        if 'image_data' in data and data['image_data'] is not None:
            data['has_image_data'] = True
            # Don't include the actual binary data in the response by default
            data.pop('image_data', None)
        else:
            data['has_image_data'] = False
            
        return cls(**data)

class TileUpload(BaseModel):
    """Schema for uploading a new tile with metadata."""
    sku: str
    model_name: str
    collection_name: str

class TileSearch(BaseModel):
    """Schema for searching tiles."""
    sku: Optional[str] = None
    model_name: Optional[str] = None
    collection_name: Optional[str] = None
    description: Optional[str] = None
    created_after: Optional[datetime] = None
    limit: int = 20
    offset: int = 0

    class Config:
        json_schema_extra = {
            "example": {
                "sku": "TILE-123",
                "model_name": "Classic",
                "collection_name": "Heritage",
                "description": "marble look",
                "limit": 10,
                "offset": 0
            }
        }

class TileSearchResults(BaseModel):
    """Schema for search results."""
    results: List[TileResponse]
    total: int
    limit: int
    offset: int

class MatchResponse(BaseModel):
    """Schema for returning match results."""
    query_filename: str
    matches: List[TileResponse]
    scores: List[float]
