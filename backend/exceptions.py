"""Custom exceptions for the Tile Matcher application."""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class TileMatcherException(Exception):
    """Base exception for all Tile Matcher errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(TileMatcherException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field = field


class FileProcessingError(TileMatcherException):
    """Raised when file processing fails."""
    
    def __init__(self, message: str, file_name: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="FILE_PROCESSING_ERROR", **kwargs)
        self.file_name = file_name


class DatabaseError(TileMatcherException):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DATABASE_ERROR", **kwargs)
        self.operation = operation


class MatchingServiceError(TileMatcherException):
    """Raised when matching service operations fail."""
    
    def __init__(self, message: str, method: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="MATCHING_SERVICE_ERROR", **kwargs)
        self.method = method


class FeatureExtractionError(TileMatcherException):
    """Raised when feature extraction fails."""
    
    def __init__(self, message: str, extractor: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="FEATURE_EXTRACTION_ERROR", **kwargs)
        self.extractor = extractor


def create_http_exception(
    status_code: int,
    message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> HTTPException:
    """Create a standardized HTTP exception."""
    detail = {
        "message": message,
        "error_code": error_code,
        "details": details or {}
    }
    return HTTPException(status_code=status_code, detail=detail)


def handle_tile_matcher_exception(exc: TileMatcherException) -> HTTPException:
    """Convert TileMatcherException to HTTPException with appropriate status code."""
    
    if isinstance(exc, ValidationError):
        return create_http_exception(
            status.HTTP_400_BAD_REQUEST,
            exc.message,
            exc.error_code,
            {"field": getattr(exc, 'field', None)}
        )
    
    elif isinstance(exc, FileProcessingError):
        return create_http_exception(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            exc.message,
            exc.error_code,
            {"file_name": getattr(exc, 'file_name', None)}
        )
    
    elif isinstance(exc, DatabaseError):
        return create_http_exception(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "A database error occurred. Please try again.",
            exc.error_code,
            {"operation": getattr(exc, 'operation', None)}
        )
    
    elif isinstance(exc, (MatchingServiceError, FeatureExtractionError)):
        return create_http_exception(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "An error occurred during image processing. Please try again.",
            exc.error_code,
            {"method": getattr(exc, 'method', None)}
        )
    
    else:
        return create_http_exception(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            exc.message,
            exc.error_code,
            exc.details
        )