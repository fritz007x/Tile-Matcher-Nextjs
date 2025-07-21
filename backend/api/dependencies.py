from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Optional
import os
import logging

# Re-export the simple matching service
from backend.services.simple_matching import get_simple_matching_service, SimpleTileMatchingService

# For compatibility, create aliases
def get_matching_service():
    return get_simple_matching_service()

TileMatchingService = SimpleTileMatchingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Dependency to get the current authenticated user.
    This is a placeholder - implement your actual authentication logic here.
    """
    # TODO: Implement actual token validation
    # This is a simplified example
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"username": "test_user"}  # Replace with actual user data
