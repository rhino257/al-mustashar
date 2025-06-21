import logging
import os
import asyncio # Import asyncio
from fastapi import Depends, HTTPException, status
from supabase import create_client, Client as SupabaseClient
from gotrue.types import User # For type hinting current_user

from .auth import get_current_user # To get current_user.id
from ..config import USERS_SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY # For service client

logger = logging.getLogger(__name__)

supabase_rate_limit_client: SupabaseClient | None = None
rate_limit_client_initialized = False

try:
    if USERS_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        supabase_rate_limit_client = create_client(USERS_SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        rate_limit_client_initialized = True
        logger.info("Supabase client for rate limiting initialized successfully.")
    else:
        logger.error("Supabase URL or Service Role Key for rate limiting not provided in config. Rate limiting will be disabled or fail.")
except Exception as e:
    logger.critical(f"Failed to initialize Supabase client for rate limiting: {e}", exc_info=True)
    # supabase_rate_limit_client remains None

async def user_rate_limiter(
    current_user: User = Depends(get_current_user)
):
    """
    FastAPI dependency to enforce per-user rate limits by calling a Supabase RPC.
    """
    global rate_limit_client_initialized, supabase_rate_limit_client

    if not rate_limit_client_initialized or not supabase_rate_limit_client:
        logger.error(f"Rate limiter called for user {current_user.id}, but Supabase client for rate limiting is not initialized. Allowing request to proceed (fail open).")
        # In a stricter setup, you might raise HTTPException 503 here.
        # For now, failing open to not block users if rate limiting itself has an issue.
        return

    user_id = str(current_user.id) 

    try:
        # The RPC function defaults to p_limit_count = 10 and p_window_interval = '1 minute'
        # If you made the RPC parameters mandatory or want to pass them from Python:
        # response = await supabase_rate_limit_client.rpc(
        #     "check_rate_limit", 
        #     {"p_user_id": user_id, "p_limit_count": 10, "p_window_interval": "1 minute"}
        # ).execute()
        
        # Calling with only p_user_id, relying on RPC defaults for limit and window
        # Wrap the synchronous execute() call in asyncio.to_thread
        response = await asyncio.to_thread(
            supabase_rate_limit_client.rpc(
                "check_rate_limit", 
                {"p_user_id": user_id}
            ).execute
        )
        
        # For RPC functions returning a single scalar boolean, response.data is often the boolean directly.
        if response.data is False: # RPC returned FALSE, meaning limit exceeded
            logger.warning(f"Rate limit exceeded for user {user_id}. Denying request.")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too Many Requests. Please try again later.",
            )
        elif response.data is True: # RPC returned TRUE, meaning request is allowed
            # logger.debug(f"Rate limit check passed for user {user_id}")
            pass # Request allowed
        else:
            # This case covers if response.data is None or not a boolean,
            # which would be an unexpected response from our RPC.
            logger.error(f"Unexpected value from check_rate_limit RPC for user {user_id}. Response data: {response.data}. Response object: {response}. Allowing request (fail open).")

    except HTTPException: # Re-raise if it's already an HTTPException (like the 429)
        raise
    except Exception as e:
        logger.error(f"Exception during rate limit check for user {user_id}: {e}", exc_info=True)
        # Failing open during unexpected exceptions. Consider failing closed for higher security.
        # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error processing rate limit check.")
