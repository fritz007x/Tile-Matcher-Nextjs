from datetime import datetime

from beanie import Document, Indexed
from pydantic import EmailStr, Field

class User(Document):
    """Application user stored in MongoDB (Beanie)."""

    email: Indexed(EmailStr, unique=True)  # type: ignore
    name: str = Field(...)
    hashed_password: str = Field(..., exclude=True)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "users"

    async def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        await super().save(*args, **kwargs)
