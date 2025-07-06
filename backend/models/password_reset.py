from datetime import datetime, timedelta
import secrets

from beanie import Document, Indexed
from pydantic import Field

class PasswordResetToken(Document):
    """Password-reset token stored in MongoDB (Beanie)."""

    token: Indexed(str, unique=True)  # type: ignore
    user_id: str = Field(...)
    expires_at: datetime = Field(...)
    used: bool = Field(default=False)

    class Settings:
        name = "password_reset_tokens"

    @classmethod
    def generate(cls, user_id: str, ttl_minutes: int = 30) -> "PasswordResetToken":
        """Create a new token instance (unsaved)."""
        return cls(
            token=secrets.token_urlsafe(48),
            user_id=user_id,
            expires_at=datetime.utcnow() + timedelta(minutes=ttl_minutes),
        )

    def is_valid(self) -> bool:
        return not self.used and datetime.utcnow() < self.expires_at
