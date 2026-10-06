"""Authentication, password security, and session token services."""
import hmac
import hashlib
import secrets
import base64
import time
from typing import Optional
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.database import get_db
from app.models import User

# Secret key for signing authentication tokens
SECRET_KEY = "autooptml_production_secret_key_change_in_prod_random_salt"
TOKEN_EXPIRY_SECONDS = 86400 * 7  # 7 days

security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Hashes a password securely using PBKDF2-HMAC-SHA256 with a unique random salt."""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    )
    return f"{salt}${key.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against the stored salt$hash."""
    try:
        salt, stored_hash = hashed_password.split('$')
        computed_key = hashlib.pbkdf2_hmac(
            'sha256',
            plain_password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return hmac.compare_digest(computed_key.hex(), stored_hash)
    except Exception:
        return False


def create_access_token(user_id: int) -> str:
    """Creates a tamper-proof signed bearer token containing user_id and expiration."""
    expires_at = int(time.time()) + TOKEN_EXPIRY_SECONDS
    payload = f"{user_id}:{expires_at}"
    payload_b64 = base64.urlsafe_b64encode(payload.encode('utf-8')).decode('utf-8')
    
    signature = hmac.new(
        SECRET_KEY.encode('utf-8'),
        payload_b64.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return f"{payload_b64}.{signature}"


def verify_token(token: str) -> Optional[int]:
    """Verifies a signed token and returns the user_id if valid and unexpired."""
    try:
        parts = token.split('.')
        if len(parts) != 2:
            return None
        payload_b64, signature = parts
        
        expected_sig = hmac.new(
            SECRET_KEY.encode('utf-8'),
            payload_b64.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(expected_sig, signature):
            return None
            
        payload = base64.urlsafe_b64decode(payload_b64.encode('utf-8')).decode('utf-8')
        user_id_str, expires_at_str = payload.split(':')
        
        if int(expires_at_str) < time.time():
            return None  # Expired
            
        return int(user_id_str)
    except Exception:
        return None


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Enforces authentication and returns the currently logged-in user."""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = verify_token(credentials.credentials)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Retrieves current user if token is present and valid; otherwise returns None."""
    if not credentials or not credentials.credentials:
        return None
    user_id = verify_token(credentials.credentials)
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()


def init_default_demo_user(db: Session) -> User:
    """Seeds a ready-to-use demo user so reviewers can test immediately with 1 click."""
    demo_email = "demo@autooptml.com"
    user = db.query(User).filter(User.email == demo_email).first()
    if not user:
        user = User(
            username="DemoResearcher",
            email=demo_email,
            hashed_password=hash_password("demo123")
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user
