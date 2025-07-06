from datetime import datetime
from typing import Optional

from beanie import Document, Indexed
from pydantic import Field

class Tile(Document):
    """
    Represents a single tile in the database.
    """
    sku: Indexed(str, unique=True) # type: ignore
    model_name: Indexed(str) # type: ignore
    collection_name: Indexed(str) # type: ignore
    image_path: str = Field(..., description="The path to the tile's image file on the server.")
    image_data: bytes = Field(..., description="Base64-encoded image data stored in the database")
    content_type: str = Field(default="image/jpeg", description="MIME type of the image")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "tiles" # The name of the MongoDB collection

    # This is a good practice to update the 'updated_at' timestamp on save
    async def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        await super().save(*args, **kwargs)
