import os
from fastapi import FastAPI, Depends, HTTPException, status
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
from api.endpoints import matching
from api.dependencies import get_matching_service, get_current_user

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
app.include_router(matching.router)

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
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.exception("Unhandled exception occurred")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

# Application startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application services on startup"""
    try:
        # Initialize the matching service
        matching_service = get_matching_service()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENVIRONMENT") == "development"
    )
