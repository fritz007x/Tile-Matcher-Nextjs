from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional

class TileBase(BaseModel):
    """Base schema for a tile."""
    sku: str
    model_name: str
    collection_name: str
    image_path: str
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
    
    class Config:
        from_attributes = True  # Replaces orm_mode in Pydantic v2
        populate_by_name = True
        json_encoders = {
            ObjectId: str
        }
        
    @classmethod
    def from_mongo(cls, data: dict):
        """Convert MongoDB document to Pydantic model."""
        if '_id' in data:
            data['id'] = str(data['_id'])
        return cls(**data)

class TileUpload(BaseModel):
    """Schema for uploading a new tile with metadata."""
    sku: str
    model_name: str
    collection_name: str

class MatchResponse(BaseModel):
    """Schema for returning match results."""
    query_filename: str
    matches: List[TileResponse]
    scores: List[float]
