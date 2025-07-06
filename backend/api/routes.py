from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request, status, Response
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional, Union, Dict, Any
from datetime import datetime
from pathlib import Path
import numpy as np
import tempfile
import os
import cv2
import logging
import base64
from io import BytesIO
from PIL import Image

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
        # Read the uploaded file content first
        file_content = await file.read()
        logging.info(f"Read {len(file_content)} bytes from uploaded file: {file.filename}")
        
        # Save the uploaded file to disk
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        # Check if tile with this SKU already exists
        existing_tile = await Tile.find_one(Tile.sku == sku)
        
        if existing_tile:
            # Update existing tile with new image data
            existing_tile.model_name = model_name
            existing_tile.collection_name = collection_name
            existing_tile.image_path = file_path
            existing_tile.image_data = file_content
            existing_tile.content_type = file.content_type or "image/jpeg"
            logging.info(f"Updating existing tile {existing_tile.id} with {len(file_content)} bytes of image data")
            await existing_tile.save()
            tile = existing_tile
            logging.info(f"Successfully updated tile {tile.id} in database")
            
            # Update the tile in the matching service
            matching_service.add_tile(
                tile_id=str(tile.id),
                image_path=file_path,
                metadata={
                    "sku": tile.sku,
                    "model_name": tile.model_name,
                    "collection_name": tile.collection_name,
                    "content_type": tile.content_type,
                    "uploaded_at": str(tile.updated_at)
                },
                image_data=file_content
            )
        else:
            # Create new tile with image data
            logging.info(f"Creating new tile with SKU: {sku}, image data: {len(file_content)} bytes")
            new_tile = Tile(
                sku=sku,
                model_name=model_name,
                collection_name=collection_name,
                image_path=file_path,
                image_data=file_content,
                content_type=file.content_type or "image/jpeg"
            )
            await new_tile.save()
            tile = new_tile
            logging.info(f"Successfully created new tile {tile.id} in database")
            
            # Add the tile to the matching service with the image data
            matching_service.add_tile(
                tile_id=str(tile.id),
                image_path=file_path,
                metadata={
                    "sku": tile.sku,
                    "model_name": tile.model_name,
                    "collection_name": tile.collection_name,
                    "content_type": tile.content_type,
                    "uploaded_at": str(tile.created_at)
                },
                image_data=file_content
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

@router.post("/match", response_model=MatchResponse)
async def match_tile(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    method: Optional[str] = Form('vit'),
    threshold: Optional[float] = Form(0.0),
    matching_service: TileMatchingService = Depends(get_matching_service)
):
    """
    Match an uploaded tile image against the catalog.
    
    Args:
        file: The image file to match
        top_k: Maximum number of matches to return (1-10)
        method: Feature extraction method ('color_hist', 'orb', 'vit')
        threshold: Minimum similarity score (0.0-1.0) for a match to be included
        
    Returns:
        MatchResponse with query filename, matches, and scores
    """
    temp_path = None
    try:
        # Debug logging
        logging.info(f"Match request - file: {file.filename}, top_k: {top_k}, method: {method}, threshold: {threshold}")
        
        # Validate parameters
        if top_k < 1 or top_k > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="top_k must be between 1 and 10"
            )
            
        if threshold < 0.0 or threshold > 1.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="threshold must be between 0.0 and 1.0"
            )
            
        if method not in matching_service.methods:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid method '{method}'. Available methods: {', '.join(matching_service.methods)}"
            )
        
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name
        
        logging.info(f"Processing image: {file.filename} (size: {os.path.getsize(temp_path)} bytes)")
        
        # Find similar tiles
        matches = matching_service.find_similar_tiles(
            query_image_path=temp_path,
            top_k=top_k,
            method=method,
            threshold=threshold
        )
        
        logging.info(f"Found {len(matches)} matches for {file.filename}")
        
        # Format the response according to MatchResponse schema
        formatted_matches = []
        for match in matches:
            match_data = {
                "id": match.get('id', ''),
                "sku": match.get('metadata', {}).get('sku', ''),
                "model_name": match.get('metadata', {}).get('model_name', ''),
                "collection_name": match.get('metadata', {}).get('collection_name', ''),
                "image_path": match.get('image_path', ''),
                "created_at": match.get('metadata', {}).get('uploaded_at', ''),
                "updated_at": match.get('metadata', {}).get('uploaded_at', ''),
                "description": None,
                "has_image_data": 'image_data' in match
            }
            
            # Include base64 image data if available
            if 'image_data' in match:
                match_data['image_data'] = match['image_data']
                match_data['content_type'] = match.get('metadata', {}).get('content_type', 'image/jpeg')
                
            formatted_matches.append(match_data)
        
        response = {
            "query_filename": file.filename,
            "matches": formatted_matches,
            "scores": [match.get('similarity', 0.0) for match in matches]
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error matching tile: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing the image: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logging.warning(f"Failed to delete temporary file {temp_path}: {str(e)}")

@router.get("/tile/{tile_id}/image")
async def get_tile_image(
    tile_id: str,
    matching_service: TileMatchingService = Depends(get_matching_service)
):
    """
    Get a tile image as a Base64-encoded string from the database
    """
    try:
        # Try to get the tile from the database first
        tile = await Tile.get(tile_id)
        if not tile or not tile.image_data:
            raise HTTPException(status_code=404, detail="Tile not found or has no image data")
        
        import base64
        return {
            "tile_id": str(tile.id),
            "content_type": tile.content_type,
            "data": base64.b64encode(tile.image_data).decode('utf-8')
        }
    except Exception as e:
        logging.error(f"Error serving image for tile {tile_id}: {str(e)}")
        # Fall back to the matching service if database retrieval fails
        try:
            image_data = matching_service.get_tile_image_data(tile_id)
            if not image_data:
                raise HTTPException(status_code=404, detail="Image not found")
            return {
                "tile_id": tile_id,
                "content_type": image_data.get('content_type', 'image/jpeg'),
                "data": image_data['data']
            }
        except Exception as e2:
            logging.error(f"Error serving image from matching service: {str(e2)}")
            raise HTTPException(status_code=500, detail="Error processing image data")

@router.get("/tile/{tile_id}/thumbnail")
async def get_tile_thumbnail(
    tile_id: str,
    width: int = 200,
    height: int = 200,
    matching_service: TileMatchingService = Depends(get_matching_service)
):
    """
    Get a thumbnail of a tile image as a Base64-encoded string from the database
    """
    try:
        # Try to get the tile from the database first
        tile = await Tile.get(tile_id)
        if not tile or not tile.image_data:
            raise HTTPException(status_code=404, detail="Tile not found or has no image data")
        
        # Create thumbnail from the database image data
        img = Image.open(BytesIO(tile.image_data))
        img.thumbnail((width, height))
        
        # Convert to base64
        buffered = BytesIO()
        img_format = tile.content_type.split('/')[-1] if tile.content_type else 'jpeg'
        if img_format.lower() not in ['jpeg', 'jpg', 'png']:
            img_format = 'jpeg'
        img.save(buffered, format=img_format)
        
        return {
            "tile_id": str(tile.id),
            "content_type": f"image/{img_format}",
            "data": base64.b64encode(buffered.getvalue()).decode('utf-8')
        }
    except Exception as e:
        logging.error(f"Error creating thumbnail for tile {tile_id}: {str(e)}")
        # Fall back to the matching service if database retrieval fails
        try:
            image_data = matching_service.get_tile_image_data(tile_id)
            if not image_data or 'data' not in image_data:
                raise HTTPException(status_code=404, detail="Image not found")
                
            # Decode base64 and create thumbnail
            img_bytes = base64.b64decode(image_data['data'])
            img = Image.open(BytesIO(img_bytes))
            img.thumbnail((width, height))
            
            # Convert back to base64
            buffered = BytesIO()
            img_format = image_data.get('content_type', 'image/jpeg').split('/')[-1]
            if img_format.lower() not in ['jpeg', 'jpg', 'png']:
                img_format = 'jpeg'
            img.save(buffered, format=img_format)
            
            return {
                "tile_id": tile_id,
                "content_type": f"image/{img_format}",
                "data": base64.b64encode(buffered.getvalue()).decode('utf-8')
            }
        except Exception as e2:
            logging.error(f"Error creating thumbnail from matching service: {str(e2)}")
            raise HTTPException(status_code=500, detail="Error processing thumbnail")

@router.get("/methods", response_model=List[str])
async def get_available_methods(
    matching_service: TileMatchingService = Depends(get_matching_service)
):
    """
    Get list of available matching methods
    """
    return list(matching_service.methods)

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
