from datetime import datetime, timedelta
import secrets

from beanie import Document, Indexed
from pydantic import Field
from beanie import PydanticObjectId

class PasswordResetToken(Document):
    """Password-reset token stored in MongoDB (Beanie)."""

    token: Indexed(str, unique=True)  # type: ignore
    user_id: PydanticObjectId = Field(...)
    expires_at: datetime = Field(...)
    used: bool = Field(default=False)

    class Settings:
        name = "password_reset_tokens"

    @classmethod
    def generate(cls, user_id: PydanticObjectId, ttl_minutes: int = 30) -> "PasswordResetToken":
        """Create a new token instance (unsaved)."""
        return cls(
            token=secrets.token_urlsafe(48),
            user_id=user_id,
            expires_at=datetime.utcnow() + timedelta(minutes=ttl_minutes),
        )

    def is_valid(self) -> bool:
        return not self.used and datetime.utcnow() < self.expires_at

    @classmethod
    async def cleanup_expired_tokens(cls) -> int:
        """Remove expired and used tokens from the database."""
        now = datetime.utcnow()
        expired_tokens = await cls.find(
            {"$or": [
                {"expires_at": {"$lt": now}},
                {"used": True}
            ]}
        ).to_list()
        
        if expired_tokens:
            for token in expired_tokens:
                await token.delete()
            return len(expired_tokens)
        return 0
