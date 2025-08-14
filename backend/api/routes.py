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
import hashlib
import re
import uuid
import time
from io import BytesIO
from PIL import Image
from werkzeug.utils import secure_filename
from bson import ObjectId
from bson.errors import InvalidId

from backend.models.tile import Tile
from backend.services.simple_matching import SimpleTileMatchingService, get_simple_matching_service
from backend.api.dependencies import get_matching_service, TileMatchingService
from backend.schemas import TileResponse, MatchResponse, TileSearch, TileSearchResults
from backend.exceptions import ValidationError, FileProcessingError, DatabaseError, MatchingServiceError
from backend.utils import get_tile_image_data, ensure_image_format, get_mime_type, validate_objectid
from backend.cache.image_cache import cache_manager

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
    file_path = None  # Initialize file_path to prevent undefined variable error
    
    try:
        # Input validation and sanitization
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required"
            )
        
        # Validate and sanitize filename to prevent path traversal
        safe_filename = secure_filename(file.filename)
        if not safe_filename:
            # If secure_filename returns empty, generate a safe name
            file_extension = os.path.splitext(file.filename)[1].lower()
            safe_filename = f"{uuid.uuid4()}{file_extension}"
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Only image files are allowed"
            )
        
        # Validate and sanitize form inputs
        sku = _sanitize_input(sku.strip())
        model_name = _sanitize_input(model_name.strip())
        collection_name = _sanitize_input(collection_name.strip())
        
        if not sku or not model_name or not collection_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="SKU, model name, and collection name are required and cannot be empty"
            )
        
        # Validate SKU format (alphanumeric, hyphens, underscores only)
        if not re.match(r'^[a-zA-Z0-9_-]+$', sku):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="SKU can only contain letters, numbers, hyphens, and underscores"
            )
        
        # Read and validate file content
        file_content = await file.read()
        
        # Validate file size (10MB limit)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file is not allowed"
            )
        
        # Validate actual image content
        try:
            with Image.open(BytesIO(file_content)) as img:
                img.verify()  # Verify it's a valid image
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image file"
            )
        
        logging.info(f"Processing upload - SKU: {sku}, file: {safe_filename}, size: {len(file_content)} bytes")
        
        # Create upload directory securely
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Use content hash to generate unique filename and prevent duplicates
        content_hash = hashlib.sha256(file_content).hexdigest()[:16]
        file_extension = os.path.splitext(safe_filename)[1].lower()
        unique_filename = f"{content_hash}{file_extension}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        # Write file to disk
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Database transaction logic
        tile = None
        try:
            # Check if tile with this SKU already exists
            existing_tile = await Tile.find_one(Tile.sku == sku)
            
            if existing_tile:
                # Update existing tile
                existing_tile.model_name = model_name
                existing_tile.collection_name = collection_name
                existing_tile.image_path = file_path
                existing_tile.image_data = file_content
                existing_tile.content_type = file.content_type
                existing_tile.updated_at = datetime.now()
                
                await existing_tile.save()
                tile = existing_tile
                logging.info(f"Updated existing tile {tile.id}")
                
                # Update in matching service
                matching_service.add_tile(
                    image_data=file_content,
                    metadata={
                        "sku": tile.sku,
                        "model_name": tile.model_name,
                        "collection_name": tile.collection_name,
                        "content_type": tile.content_type,
                        "filename": safe_filename,
                        "uploaded_at": str(tile.updated_at)
                    },
                    tile_id=str(tile.id)
                )
            else:
                # Create new tile
                new_tile = Tile(
                    sku=sku,
                    model_name=model_name,
                    collection_name=collection_name,
                    image_path=file_path,
                    image_data=file_content,
                    content_type=file.content_type
                )
                
                await new_tile.save()
                tile = new_tile
                logging.info(f"Created new tile {tile.id}")
                
                # Add to matching service
                matching_service.add_tile(
                    image_data=file_content,
                    metadata={
                        "sku": tile.sku,
                        "model_name": tile.model_name,
                        "collection_name": tile.collection_name,
                        "content_type": tile.content_type,
                        "filename": safe_filename,
                        "uploaded_at": str(tile.created_at)
                    },
                    tile_id=str(tile.id)
                )
            
            return TileResponse.from_mongo(tile.dict(by_alias=True))
            
        except Exception as db_error:
            # If database operation fails, clean up
            logging.error(f"Database operation failed: {str(db_error)}")
            raise db_error
            
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        if file_path and os.path.exists(file_path):
            _safe_file_cleanup(file_path)
        raise
    except Exception as e:
        logging.exception("Unexpected error in upload_tile")
        
        # Clean up file on any error
        if file_path and os.path.exists(file_path):
            _safe_file_cleanup(file_path)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during upload"
        )


def _sanitize_input(input_str: str) -> str:
    """Sanitize input string to prevent injection attacks"""
    if not input_str:
        return ""
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\';\\]', '', input_str)
    return sanitized.strip()


def _safe_file_cleanup(file_path: str) -> None:
    """Safely remove a file without raising exceptions"""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logging.error(f"Failed to clean up file {file_path}: {e}")

@router.post("/match", response_model=MatchResponse)
async def match_tile(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    method: Optional[str] = Form('color_hist'),
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
    try:
        logging.info(f"Match request received - filename: {file.filename}, content_type: {file.content_type}")
        
        # Input validation
        if not file.filename:
            raise ValidationError("Filename is required", field="file")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise ValidationError("Invalid file type. Only image files are allowed", field="file")
        
        # Validate parameters
        if top_k < 1 or top_k > 10:
            raise ValidationError("top_k must be between 1 and 10", field="top_k")
            
        # Handle both percentage (0-100) and decimal (0.0-1.0) threshold formats
        if threshold > 1.0 and threshold <= 100.0:
            # Convert percentage to decimal
            threshold = threshold / 100.0
        elif threshold < 0.0 or threshold > 1.0:
            raise ValidationError("threshold must be between 0.0 and 1.0 (or 0 and 100 if using percentage)", field="threshold")
        
        # Validate method
        available_methods = matching_service.get_available_methods()
        if method not in available_methods:
            raise ValidationError(
                f"Invalid method '{method}'. Available methods: {', '.join(available_methods)}", 
                field="method"
            )
        
        # Read and validate file content
        contents = await file.read()
        
        # Validate file size (5MB limit for matching)
        MAX_MATCH_FILE_SIZE = 5 * 1024 * 1024  # 5MB
        if len(contents) > MAX_MATCH_FILE_SIZE:
            raise FileProcessingError(
                f"File too large for matching. Maximum size is {MAX_MATCH_FILE_SIZE // (1024*1024)}MB",
                file_name=file.filename
            )
        
        if len(contents) == 0:
            raise FileProcessingError("Empty file is not allowed", file_name=file.filename)
        
        # Validate actual image content
        try:
            with Image.open(BytesIO(contents)) as img:
                img.verify()  # Verify it's a valid image
        except Exception as e:
            raise FileProcessingError("Invalid or corrupted image file", file_name=file.filename)
        
        # Sanitize filename for logging
        safe_filename = secure_filename(file.filename) or "unknown.jpg"
        logging.info(f"Simple match request - file: {safe_filename}, top_k: {top_k}, method: {method}, threshold: {threshold}")
        
        # Use simplified matching service (just fetches from database)
        try:
            logging.info(f"Calling simple matching service with method={method}, top_k={top_k}, threshold={threshold}")
            matches = await matching_service.find_similar_tiles(
                query_image_data=contents,
                top_k=top_k,
                method=method,
                threshold=threshold
            )
            logging.info(f"Simple matching service returned {len(matches)} matches")
        except Exception as e:
            logging.error(f"Error in simple matching service: {str(e)}", exc_info=True)
            raise MatchingServiceError("Failed to process image matching", method=method)
        
        logging.info(f"Found {len(matches)} matches for {safe_filename}")
        
        # Simple formatting - the matching service already returns properly formatted data
        formatted_matches = []
        current_time = datetime.now()
        
        for match in matches:
            # Simple timestamp handling
            created_at = current_time
            if match.get('created_at'):
                try:
                    created_at = datetime.fromtimestamp(match['created_at'])
                except:
                    created_at = current_time
            
            # Extract data from the simple matching service response
            metadata = match.get('metadata', {})
            
            match_data = {
                "id": match.get('tile_id', ''),
                "sku": metadata.get('sku', ''),
                "model_name": metadata.get('model_name', ''),
                "collection_name": metadata.get('collection_name', ''),
                "image_path": metadata.get('image_path', ''),
                "created_at": created_at,
                "updated_at": created_at,
                "description": None,
                "has_image_data": match.get('has_image_data', False),
                "content_type": metadata.get('content_type', 'image/jpeg')
            }
            
            # Add image data if available
            if match.get('image_data'):
                match_data['image_data'] = match['image_data']
            
            formatted_matches.append(match_data)
        
        response = {
            "query_filename": safe_filename,
            "matches": formatted_matches,
            "scores": [float(match.get('similarity', 0.0)) for match in matches]
        }
        
        return response
        
    except (ValidationError, FileProcessingError, DatabaseError, MatchingServiceError):
        # Re-raise our custom exceptions to be handled by the global handler
        raise
    except Exception as e:
        logging.exception("Unexpected error in match_tile")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail={
                "message": "An unexpected error occurred while processing the image",
                "error_code": "UNEXPECTED_ERROR",
                "details": {}
            }
        )

@router.get("/tile/{tile_id}/image")
async def get_tile_image(
    tile_id: str,
    matching_service: TileMatchingService = Depends(get_matching_service)
):
    """
    Get a tile image as a Base64-encoded string from database or matching service
    """
    logging.info(f"Requesting image for tile_id: {tile_id}")
    
    # Check if tile_id is a valid ObjectId for database lookup
    is_valid_objectid = False
    try:
        ObjectId(tile_id)
        is_valid_objectid = True
        logging.debug(f"tile_id {tile_id} is a valid ObjectId")
    except InvalidId:
        logging.info(f"tile_id {tile_id} is not a valid ObjectId, skipping database lookup")
    
    # Try database lookup only if it's a valid ObjectId
    if is_valid_objectid:
        try:
            logging.debug(f"Attempting database lookup for tile {tile_id}")
            tile = await Tile.get(tile_id)
            if tile and tile.image_data:
                logging.info(f"Found tile in database: {tile.sku}")
                return {
                    "tile_id": str(tile.id),
                    "content_type": tile.content_type or "image/jpeg",
                    "data": base64.b64encode(tile.image_data).decode('utf-8')
                }
            elif tile:
                logging.warning(f"Tile {tile_id} found in database but no image_data")
            else:
                logging.debug(f"Tile {tile_id} not found in database")
        except Exception as e:
            logging.warning(f"Database lookup failed for tile {tile_id}: {str(e)}")
    
    # Always try matching service as fallback (primary source for hash-based IDs)
    try:
        logging.debug(f"Attempting matching service lookup for tile {tile_id}")
        image_data = matching_service.get_tile_image_data(tile_id)
        if image_data and 'data' in image_data:
            logging.info(f"Found image in matching service for tile {tile_id}")
            return {
                "tile_id": tile_id,
                "content_type": image_data.get('content_type', 'image/jpeg'),
                "data": image_data['data']
            }
        else:
            logging.warning(f"No image data returned from matching service for tile {tile_id}")
    except Exception as e:
        logging.error(f"Error serving image from matching service for tile {tile_id}: {str(e)}")
    
    # If both methods fail, return 404
    logging.error(f"Image not found for tile {tile_id} in either database or matching service")
    raise HTTPException(status_code=404, detail="Image not found")

@router.get("/tile/{tile_id}/thumbnail")
async def get_tile_thumbnail(
    tile_id: str,
    width: int = 200,
    height: int = 200,
    matching_service: TileMatchingService = Depends(get_matching_service)
):
    """
    Get a thumbnail of a tile image from database or matching service
    """
    logging.info(f"Requesting thumbnail for tile_id: {tile_id} (size: {width}x{height})")
    
    # Validate thumbnail dimensions
    if width <= 0 or height <= 0 or width > 1000 or height > 1000:
        raise HTTPException(status_code=400, detail="Invalid thumbnail dimensions")
    
    # Check if tile_id is a valid ObjectId for database lookup
    is_valid_objectid = validate_objectid(tile_id)
    if is_valid_objectid:
        logging.debug(f"tile_id {tile_id} is a valid ObjectId")
    else:
        logging.info(f"tile_id {tile_id} is not a valid ObjectId, skipping database lookup")
    
    # Try database lookup only if it's a valid ObjectId
    if is_valid_objectid:
        try:
            logging.debug(f"Attempting database lookup for thumbnail {tile_id}")
            tile = await Tile.get(ObjectId(tile_id))
            if tile and tile.image_data:
                logging.info(f"Found tile in database, creating thumbnail: {tile.sku}")
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
            elif tile:
                logging.warning(f"Tile {tile_id} found in database but no image_data")
            else:
                logging.debug(f"Tile {tile_id} not found in database")
        except Exception as e:
            logging.warning(f"Database lookup failed for thumbnail {tile_id}: {str(e)}")
    
    # If database lookup fails, return 404
    logging.error(f"Thumbnail not found for tile {tile_id} in database")
    raise HTTPException(status_code=404, detail="Thumbnail not found")

@router.get("/methods", response_model=List[str])
async def get_available_methods(
    matching_service: TileMatchingService = Depends(get_matching_service)
):
    """
    Get list of available matching methods
    """
    return matching_service.get_available_methods()

@router.get("/cache/stats")
async def get_cache_stats():
    """Get image cache statistics for monitoring."""
    return cache_manager.get_cache_stats()

@router.get("/debug/tiles")
async def debug_tiles():
    """Debug endpoint to check if we can fetch tiles from database."""
    try:
        tiles = await Tile.find().limit(5).to_list()
        return {
            "tile_count": len(tiles),
            "tiles": [{"id": str(tile.id), "sku": tile.sku, "model_name": tile.model_name} for tile in tiles]
        }
    except Exception as e:
        logging.error(f"Error fetching tiles for debug: {e}")
        return {"error": str(e), "tile_count": 0}

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
        # Log the actual error for debugging
        logging.error(f"Error searching tiles: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while searching tiles. Please try again."
        )
