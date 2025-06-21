import traceback
import logging
from typing import Optional # Added Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from supabase import create_client, Client # ClientOptions and httpx import removed

# Assuming config.py is in the parent directory (almustashar_api)
from ..config import USERS_SUPABASE_URL, USERS_SUPABASE_KEY # Using the public/anon key for client-side token verification

logger = logging.getLogger(__name__)

# OAuth2 scheme for extracting the bearer token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # tokenUrl is nominal, actual validation is via Supabase

# Supabase client for auth (token verification)
supabase_auth_client: Client | None = None
try:
    if USERS_SUPABASE_URL and USERS_SUPABASE_KEY:
        # Simplified initialization without custom httpx client
        supabase_auth_client = create_client(
            USERS_SUPABASE_URL,
            USERS_SUPABASE_KEY
        )
        logger.info("Supabase client for auth initialized successfully (standard initialization).")
    else:
        logger.warning("Supabase URL or Key for auth not provided. Supabase auth client not initialized.")
except Exception as e:
    # Log the full exception traceback for better debugging if it still fails
    logger.critical(f"Failed to initialize Supabase client for auth: {e}", exc_info=True)
    supabase_auth_client = None # Ensure it's None on failure


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Dependency function to verify Supabase JWT.
    Raises HTTPException if token is invalid or expired.
    Returns the user object upon success.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
    )

    if not supabase_auth_client:
        logger.error("ERROR:    get_current_user called but Supabase auth client is not initialized.")
        raise credentials_exception # Or a 503 Service Unavailable

    try:
        # Verify the token using the Supabase client
        response = supabase_auth_client.auth.get_user(token)
        user = response.user

        if not user:
            logger.warning(f"Supabase token verification failed: No user returned for token: {token[:20]}...") # Log part of token
            raise credentials_exception

        if not hasattr(user, 'id') or not user.id:
            logger.warning(f"Supabase token verification failed: User object missing ID. User data: {user}")
            raise credentials_exception
        
        # logger.info(f"Supabase user authenticated: {user.id}") # Can be noisy, consider DEBUG level
        return user

    except HTTPException: # Re-raise if it's already an HTTPException (like credentials_exception)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during Supabase token verification: {e}")
        logger.error(traceback.format_exc())
        raise credentials_exception


async def get_current_admin_user(current_user = Depends(get_current_user)):
    """
    Dependency function that relies on get_current_user and then checks
    for admin privileges based on Supabase user data.
    (This logic needs to be adapted based on how admin roles are stored in your Supabase setup)
    """
    forbidden_exception = HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="User does not have admin privileges",
    )

    # Placeholder for admin check logic.
    # Example: Check a custom claim in user_metadata
    # is_admin = getattr(current_user, 'user_metadata', {}).get('role') == 'admin'
    
    # For now, let's assume this check needs to be implemented.
    # To make any endpoint using this work, you'd need to define how 'admin' is determined.
    # Defaulting to False for safety until implemented.
    is_admin = False 
    
    # Example: if you have a specific list of admin UIDs in an env var
    # ADMIN_UIDS_STR = os.getenv("ADMIN_UIDS", "")
    # ADMIN_UIDS = [uid.strip() for uid in ADMIN_UIDS_STR.split(',')]
    # if current_user.id in ADMIN_UIDS:
    #     is_admin = True

    if not is_admin:
        logger.warning(f"Admin access denied for user {current_user.id}. Admin check logic needs implementation or user lacks role.")
        raise forbidden_exception

    logger.info(f"Admin user confirmed: {current_user.id}")
    return current_user
