from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Optional
import os
import logging

from ml.matching_service import TileMatchingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Global instance of the matching service
_matching_service = None

def get_matching_service() -> TileMatchingService:
    """
    Dependency that provides the TileMatchingService instance.
    Initializes the service on first call.
    """
    global _matching_service
    
    if _matching_service is None:
        try:
            # Initialize with all available methods
            _matching_service = TileMatchingService(methods=['sift', 'orb', 'kaze', 'vit'])
            logger.info("Initialized TileMatchingService")
            
            # Here you would typically load your catalog of tiles
            # Example:
            # for tile in catalog_tiles:
            #     _matching_service.add_tile(
            #         tile_id=tile['id'],
            #         image_path=tile['image_path'],
            #         metadata={
            #             'sku': tile.get('sku'),
            #             'model_name': tile.get('model_name'),
            #             'collection_name': tile.get('collection_name')
            #         }
            #     )
            # _matching_service.build_faiss_index()
            
        except Exception as e:
            logger.error(f"Failed to initialize TileMatchingService: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize matching service"
            )
    
    return _matching_service

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
