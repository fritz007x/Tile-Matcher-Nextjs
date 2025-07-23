from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr
from typing import Optional
import os
from datetime import datetime, timedelta
from jose import jwt
import logging

from backend.models.user import User
from backend.models.password_reset import PasswordResetToken
from backend.services.email_service import email_service
from backend.api.dependencies import get_current_user
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter(prefix="/api/auth", tags=["auth"])
# Separate router for generic /api routes like /api/token expected by frontend
public_router = APIRouter(prefix="/api", tags=["auth"])

ALGORITHM = "HS256"


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class TokenRequest(BaseModel):
    username: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    email: EmailStr
    name: str
    user_id: str

class RegisterResponse(BaseModel):
    message: str = "Registration successful"
    email: EmailStr
    name: str
    user_id: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ForgotPasswordResponse(BaseModel):
    message: str = "If an account with this email exists, we've sent password reset instructions."

class ResetPasswordRequest(BaseModel):
    token: str
    password: str

class ResetPasswordResponse(BaseModel):
    message: str = "Password has been reset successfully."

class UserResponse(BaseModel):
    user_id: str
    email: EmailStr
    name: str
    created_at: datetime
    updated_at: datetime


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register_user(payload: RegisterRequest):
    """Simple placeholder registration endpoint.

    This is NOT production-ready – it only echoes the request so the frontend
    can continue development without a 404.  Replace with real user creation
    logic and password hashing when integrating a database.
    """
    logger.info("Register request for %s", payload.email)

    # Check if user already exists
    normalized_email = payload.email.lower()
    existing = await User.find_one(User.email == normalized_email)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User with this email already exists")

    # Hash password
    hashed = pwd_context.hash(payload.password)

    # Create and save user
    user = User(email=normalized_email, name=payload.name, hashed_password=hashed)
    await user.insert()

    return RegisterResponse(email=user.email, name=user.name, user_id=str(user.id))


@public_router.post("/token", response_model=TokenResponse)
async def login_for_access_token(payload: TokenRequest):
    """Credential login compatible with NextAuth."""
    normalized_username = payload.username.lower()
    logger.info("Login attempt for %s", normalized_username)

    user = await User.find_one(User.email == normalized_username)
    if not user:
        logger.warning("User not found for email: %s", normalized_username)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    
    logger.info("Found user: %s, checking password...", user.email)
    password_valid = pwd_context.verify(payload.password, user.hashed_password)
    logger.info("Password verification result: %s", password_valid)
    
    if not password_valid:
        logger.warning("Invalid password for user: %s", user.email)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")

    SECRET_KEY = os.getenv("SECRET_KEY", "devsecretkey")
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode = {
        "sub": str(user.id),
        "email": user.email,
        "name": user.name,
        "exp": expire,
    }
    access_token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return TokenResponse(
        access_token=access_token,
        email=user.email,
        name=user.name,
        user_id=str(user.id),
    )


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
async def forgot_password(payload: ForgotPasswordRequest):
    """
    Send password reset email to user.
    
    Security note: Always returns success message regardless of whether 
    the email exists to prevent email enumeration attacks.
    """
    logger.info("Password reset request for email: %s", payload.email)
    
    try:
        # Normalize email and check if user exists
        normalized_email = payload.email.lower()
        user = await User.find_one(User.email == normalized_email)
        
        if user:
            # Generate reset token
            reset_token = PasswordResetToken.generate(user_id=user.id)
            await reset_token.insert()
            logger.info("Generated reset token for user %s: %s", user.email, reset_token.token[:8] + "...")
            
            # Send email
            email_sent = email_service.send_password_reset_email(
                to_email=user.email,
                reset_token=reset_token.token,
                user_name=user.name
            )
            
            if email_sent:
                logger.info("Password reset email sent successfully to: %s", user.email)
            else:
                logger.error("Failed to send password reset email to: %s", user.email)
                # Don't expose email sending failures to prevent enumeration
        else:
            logger.info("Password reset requested for non-existent email: %s", normalized_email)
            # Still simulate processing time to prevent timing attacks
            import asyncio
            await asyncio.sleep(0.5)
    
    except Exception as e:
        logger.error("Error processing password reset request: %s", e)
        # Don't expose internal errors
    
    # Always return success message for security
    return ForgotPasswordResponse()


@router.post("/reset-password", response_model=ResetPasswordResponse)
async def reset_password(payload: ResetPasswordRequest):
    """
    Reset user password using a valid reset token.
    """
    logger.info("Password reset attempt with token: %s", payload.token[:8] + "...")
    
    try:
        # Find and validate reset token
        reset_token = await PasswordResetToken.find_one(
            PasswordResetToken.token == payload.token
        )
        
        if not reset_token:
            logger.warning("Reset token not found in database: %s", payload.token[:8] + "...")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        logger.info("Found reset token for user_id: %s, expires_at: %s, used: %s", 
                   reset_token.user_id, reset_token.expires_at, reset_token.used)
        
        if not reset_token.is_valid():
            logger.warning("Reset token is invalid - expired: %s, used: %s, current_time: %s", 
                         reset_token.expires_at < datetime.utcnow(), 
                         reset_token.used, 
                         datetime.utcnow())
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Reset token has expired or has already been used"
            )
        
        # Find user
        user = await User.find_one(User.id == reset_token.user_id)
        if not user:
            logger.error("User not found for user_id: %s", reset_token.user_id)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User associated with this token no longer exists"
            )
        
        logger.info("Found user: %s", user.email)
        
        # Validate new password
        if len(payload.password) < 8:
            logger.warning("Password too short: %d characters", len(payload.password))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long"
            )
        
        # Hash new password and update user
        logger.info("Hashing new password for user: %s", user.email)
        hashed_password = pwd_context.hash(payload.password)
        logger.info("New password hash generated: %s", hashed_password[:20] + "...")
        
        # Update the user's password
        user.hashed_password = hashed_password
        user.updated_at = datetime.utcnow()
        
        # Save the user
        await user.save()
        logger.info("User password updated successfully in database")
        
        # Verify the update by re-fetching the user
        updated_user = await User.find_one(User.id == user.id)
        logger.info("Verification - stored hash: %s", updated_user.hashed_password[:20] + "...")
        
        # Test the new password immediately
        test_verify = pwd_context.verify(payload.password, updated_user.hashed_password)
        logger.info("Immediate password verification test: %s", test_verify)
        
        # Mark token as used
        reset_token.used = True
        await reset_token.save()
        logger.info("Reset token marked as used")
        
        logger.info("Password reset successful for user: %s", user.email)
        return ResetPasswordResponse()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error resetting password: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while resetting your password"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user information.
    """
    return UserResponse(
        user_id=str(current_user.id),
        email=current_user.email,
        name=current_user.name,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at
    )


@router.get("/debug/reset-tokens")
async def debug_reset_tokens():
    """
    DEBUG ENDPOINT: List all reset tokens (remove in production)
    """
    tokens = await PasswordResetToken.find_all().to_list()
    return {
        "total_tokens": len(tokens),
        "tokens": [
            {
                "token": token.token[:8] + "...",
                "user_id": str(token.user_id),
                "expires_at": token.expires_at,
                "used": token.used,
                "is_valid": token.is_valid()
            }
            for token in tokens
        ]
    }


@router.get("/debug/users")
async def debug_users():
    """
    DEBUG ENDPOINT: List all users (remove in production)
    """
    users = await User.find_all().to_list()
    return {
        "total_users": len(users),
        "users": [
            {
                "user_id": str(user.id),
                "email": user.email,
                "name": user.name,
                "password_hash": user.hashed_password[:20] + "...",
                "created_at": user.created_at,
                "updated_at": user.updated_at
            }
            for user in users
        ]
    }


@router.post("/debug/test-password")
async def debug_test_password(payload: dict):
    """
    DEBUG ENDPOINT: Test password verification (remove in production)
    """
    email = payload.get("email")
    password = payload.get("password")
    
    if not email or not password:
        return {"error": "email and password required"}
    
    user = await User.find_one(User.email == email.lower())
    if not user:
        return {"error": "User not found"}
    
    is_valid = pwd_context.verify(password, user.hashed_password)
    
    return {
        "email": user.email,
        "password_hash": user.hashed_password[:20] + "...",
        "password_valid": is_valid,
        "updated_at": user.updated_at
    }
