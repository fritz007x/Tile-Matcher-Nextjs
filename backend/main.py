import os
import sys
# Add parent directory to Python path so we can import ml module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add project root to path

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import API routers
from backend.api import routes, auth
from backend.api.auth import public_router
from backend.api.dependencies import get_matching_service, get_current_user
from backend.db import init_db
from backend.exceptions import TileMatcherException, handle_tile_matcher_exception

# Initialize FastAPI app
app = FastAPI(
    title="Tile Matcher API",
    description="API for matching tile images with catalog using various computer vision algorithms",
    version="0.2.0",
    contact={
        "name": "Support",
        "email": "support@tilematcher.com"
    },
    license_info={
        "name": "Proprietary"
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(routes.router)
app.include_router(auth.router)
app.include_router(public_router)

# Models
class HealthCheck(BaseModel):
    status: str
    version: str
    environment: str

# Routes
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint that provides basic API information"""
    return {
        "message": "Tile Matcher API is running",
        "version": app.version,
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health", response_model=HealthCheck, tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": app.version,
        "environment": os.getenv("ENVIRONMENT", "development")
    }

# Error handlers
@app.exception_handler(TileMatcherException)
async def tile_matcher_exception_handler(request, exc: TileMatcherException):
    logger.error(f"TileMatcherException: {exc.message}", extra={
        "error_code": exc.error_code,
        "details": exc.details,
        "path": request.url.path
    })
    http_exc = handle_tile_matcher_exception(exc)
    return JSONResponse(
        status_code=http_exc.status_code,
        content=http_exc.detail
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.exception("Unhandled exception occurred", extra={
        "path": request.url.path,
        "method": request.method,
        "exception_type": type(exc).__name__
    })
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "An unexpected error occurred. Please try again.",
            "error_code": "INTERNAL_SERVER_ERROR",
            "details": {}
        }
    )

# Application startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application services on startup"""
    logger.info("Starting application initialization...")
    try:
        # Initialize the database connection
        await init_db()
        
        # Small delay to ensure Beanie is fully initialized
        import asyncio
        await asyncio.sleep(0.5)
        
        # Initialize the matching service (no await needed for simple service)
        get_matching_service()
        logger.info("Simple matching service successfully initialized.")

        logger.info("Application startup complete.")
    except Exception as e:
        logger.critical(f"An error occurred during application startup: {e}", exc_info=True)
        # Re-raising the exception will stop the application from starting
        # in a broken state, which is generally the desired behavior.
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENVIRONMENT") == "development"
    )
