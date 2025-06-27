from fastapi import APIRouter, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from typing import List, Optional
import numpy as np
import cv2
import os
import logging
from PIL import Image
import io

from ml.matching_service import TileMatchingService, MatchResult, load_image
from ..dependencies import get_matching_service

router = APIRouter(prefix="/api/matching", tags=["matching"])

@router.post("/match", response_model=List[dict])
async def match_tile(
    file: UploadFile,
    top_k: int = 5,
    matching_service: TileMatchingService = Depends(get_matching_service)
):
    """
    Match an uploaded tile image with the catalog
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not read image"
            )
        
        # Match image
        results = matching_service.match_image(image, top_k=top_k)
        
        # Convert results to dict
        return [{
            "tile_id": r.tile_id,
            "score": float(r.score),
            "method": r.method,
            "metadata": r.metadata
        } for r in results]
        
    except Exception as e:
        logging.error(f"Error in match_tile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )

@router.get("/methods")
async def get_available_methods():
    """
    Get list of available matching methods
    """
    return {
        "methods": ["sift", "orb", "kaze", "vit"],
        "default_methods": ["sift", "orb", "kaze", "vit"]
    }
