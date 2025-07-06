from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import Optional
import os
from datetime import datetime, timedelta
from jose import jwt
import logging

from backend.models.user import User
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
    if not user or not pwd_context.verify(payload.password, user.hashed_password):
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
