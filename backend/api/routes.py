from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from typing import List, Optional
from datetime import datetime
import numpy as np
import cv2
import os
import logging

from backend.models.tile import Tile
from backend.services.matching import TileMatchingService, get_matching_service
from backend.schemas import TileResponse, MatchResponse, TileSearch, TileSearchResults

router = APIRouter(prefix="/api/matching", tags=["matching"])

@router.post("/upload", response_model=TileResponse)
async def upload_tile(
    file: UploadFile = File(...),
    sku: str = Form(...),
    model_name: str = Form(...),
    collection_name: str = Form(...),
    matching_service: TileMatchingService = Depends(get_matching_service)
):
    """
    Upload a new tile with metadata.
    If a tile with the same SKU already exists, it will be updated.
    """
    try:
        # Save the uploaded file
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)

        # Check if tile with this SKU already exists
        existing_tile = await Tile.find_one(Tile.sku == sku)
        
        if existing_tile:
            # Update existing tile
            existing_tile.model_name = model_name
            existing_tile.collection_name = collection_name
            existing_tile.image_path = file_path
            await existing_tile.save()
            tile = existing_tile
            
            # Update the tile in the matching service by re-adding it
            # This will replace the existing tile with the same ID
            matching_service.add_tile(
                tile_id=str(tile.id),
                image_path=file_path,
                metadata={
                    "sku": tile.sku,
                    "model_name": tile.model_name,
                    "collection_name": tile.collection_name
                }
            )
        else:
            # Create new tile
            new_tile = Tile(
                sku=sku,
                model_name=model_name,
                collection_name=collection_name,
                image_path=file_path
            )
            await new_tile.save()
            tile = new_tile
            
            # Add the tile to the matching service
            matching_service.add_tile(
                tile_id=str(tile.id),
                image_path=file_path,
                metadata={
                    "sku": tile.sku,
                    "model_name": tile.model_name,
                    "collection_name": tile.collection_name
                }
            )
        
        # Convert the tile to TileResponse and return
        return TileResponse.from_mongo(tile.dict(by_alias=True))

    except Exception as e:
        logging.exception("Error in upload_tile")
        # Clean up the uploaded file if something went wrong
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                logging.error(f"Failed to clean up file {file_path}: {cleanup_error}")
                
        error_detail = str(e)
        if "duplicate key error" in error_detail.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A tile with SKU '{sku}' already exists."
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error uploading tile: {error_detail}"
            )

@router.post("/match", response_model=List[dict])
async def match_tile(
    file: UploadFile = File(...),
    top_k: int = Form(5),
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
        logging.exception("Error in match_tile")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )

@router.get("/methods", response_model=List[str])
async def get_available_methods():
    """
    Get list of available matching methods
    """
    return ["color_histogram", "sift"]

@router.post("/search", response_model=TileSearchResults)
async def search_tiles(
    search: TileSearch,
    matching_service: TileMatchingService = Depends(get_matching_service)
):
    """
    Search for tiles based on various criteria.
    Supports searching by SKU, model name, collection name, and description.
    """
    try:
        # Build the query
        query = {}
        
        # Add text search if any text fields are provided
        if search.sku:
            query["sku"] = {"$regex": f".*{search.sku}.*", "$options": "i"}
        if search.model_name:
            query["model_name"] = {"$regex": f".*{search.model_name}.*", "$options": "i"}
        if search.collection_name:
            query["collection_name"] = {"$regex": f".*{search.collection_name}.*", "$options": "i"}
        if search.description:
            query["description"] = {"$regex": f".*{search.description}.*", "$options": "i"}
        if search.created_after:
            query["created_at"] = {"$gte": search.created_after}
        
        # Execute the query with pagination
        cursor = Tile.find(query).skip(search.offset).limit(search.limit)
        tiles = await cursor.to_list(length=search.limit)
        total = await Tile.find(query).count()
        
        # Convert to response models
        results = [TileResponse.from_mongo(tile.dict(by_alias=True)) for tile in tiles]
        
        return {
            "results": results,
            "total": total,
            "limit": search.limit,
            "offset": search.offset
        }
        
    except Exception as e:
        logging.exception("Error in search_tiles")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching tiles: {str(e)}"
        )
