from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Optional
import os
import logging
from jose import JWTError, jwt
from datetime import datetime

# Re-export the simple matching service
from backend.services.simple_matching import get_simple_matching_service, SimpleTileMatchingService
from backend.models.user import User

# For compatibility, create aliases
def get_matching_service():
    return get_simple_matching_service()

TileMatchingService = SimpleTileMatchingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Dependency to get the current authenticated user from JWT token.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Get secret key from environment
        SECRET_KEY = os.getenv("SECRET_KEY", "devsecretkey")
        ALGORITHM = "HS256"
        
        # Decode the JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Extract user information
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
            
        # Check token expiration
        exp: int = payload.get("exp")
        if exp is None or datetime.utcnow().timestamp() > exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    user = await User.find_one(User.id == user_id)
    if user is None:
        raise credentials_exception
        
    return user

async def get_current_user_optional(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[User]:
    """
    Optional dependency to get current user. Returns None if no valid token.
    """
    if not token:
        return None
        
    try:
        return await get_current_user(token)
    except HTTPException:
        return None
