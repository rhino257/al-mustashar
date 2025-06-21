import os
from typing import Optional, Dict, Any, List
from portkey_ai._vendor.openai import AsyncOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
import logging

logger = logging.getLogger(__name__)

class ProductionGeminiChatModel:
    """Production-ready Gemini model with fallback and load balancing via Portkey."""
    
    def __init__(
        self,
        portkey_api_key: Optional[str] = None,
        fallback_config_id: Optional[str] = None,
        loadbalance_config_id: Optional[str] = None,
        premium_config_id: Optional[str] = None, # Added for premium config
        gemini_pro_virtual_key: Optional[str] = None, # Added for dynamic premium
        environment: str = "production",
        target_config_id: Optional[str] = None, # New parameter
        target_metadata: Optional[Dict[str, Any]] = None # New parameter
    ):
        self.portkey_api_key = portkey_api_key or os.getenv("PORTKEY_API_KEY")
        self.environment = environment
        
        self.fallback_config_id = fallback_config_id or os.getenv("FALLBACK_CONFIG_ID")
        self.loadbalance_config_id = loadbalance_config_id or os.getenv("LOADBALANCE_CONFIG_ID")
        self.premium_config_id = premium_config_id or os.getenv("PREMIUM_CONFIG_ID") # Load premium config ID
        self.gemini_pro_virtual_key = gemini_pro_virtual_key or os.getenv("GEMINI_PRO_VIRTUAL_KEY", "gemini-pro-e6d800") # Default if not in env
        self.default_model_for_sdk = "gemini-1.5-flash-latest" # Default model for SDK calls, Portkey will override

        self.target_config_id = target_config_id # Store new parameter
        self.target_metadata = target_metadata # Store new parameter

        if not self.portkey_api_key:
            raise ValueError("Portkey API Key not found. Please set PORTKEY_API_KEY environment variable.")

        self.client = AsyncOpenAI(
            base_url=PORTKEY_GATEWAY_URL,
            api_key="dummy-key-not-used"  # Actual API key is passed in headers by Portkey
        )
        
        logger.info(f"Initialized ProductionGeminiChatModel for '{self.environment}' environment.")
        logger.info(f"Fallback Config ID: {self.fallback_config_id}")
        logger.info(f"Loadbalance Config ID: {self.loadbalance_config_id}")
        logger.info(f"Premium Config ID: {self.premium_config_id}")
        logger.info(f"Gemini Pro Virtual Key (for dynamic premium): {self.gemini_pro_virtual_key}")

    def get_headers_for_strategy(
        self, 
        strategy: str = "fallback", 
        priority: str = "normal",
        user_metadata_param: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get Portkey headers based on strategy, priority, and optional user metadata."""
        
        # Base metadata
        current_metadata = {
            "environment": self.environment,
            "priority": priority,
            "strategy_used": strategy # This might be overridden if target_config_id is used
        }

        if self.target_metadata: # Merge persona-specific metadata first
            current_metadata.update(self.target_metadata)

        if user_metadata_param: # Then merge user-specific metadata
            # Ensure _user is set if user_id is in user_metadata_param
            user_metadata_to_merge = user_metadata_param.copy()
            if "user_id" in user_metadata_to_merge and "_user" not in user_metadata_to_merge:
                user_metadata_to_merge["_user"] = user_metadata_to_merge["user_id"]
            current_metadata.update(user_metadata_to_merge)

        portkey_specific_headers = {
            "api_key": self.portkey_api_key, # This is the Portkey API Key
            "metadata": current_metadata
        }

        # If a specific target_config_id is provided (for a persona), use it directly.
        if self.target_config_id:
            portkey_specific_headers["config"] = self.target_config_id
            logger.debug(f"Using target_config_id (persona-specific): {self.target_config_id}")
            # Update strategy_used in metadata if it was set by target_metadata
            if self.target_metadata and "strategy_used" in self.target_metadata:
                 current_metadata["strategy_used"] = self.target_metadata["strategy_used"]
            elif "strategy_used" not in current_metadata: # if not set by target_metadata or original strategy
                 current_metadata["strategy_used"] = "persona_config" # Indicate persona config was used
        
        # Otherwise, use the strategy-based selection
        # elif strategy == "fallback" and self.fallback_config_id:
        #     portkey_specific_headers["config"] = self.fallback_config_id
        #     logger.debug(f"Using Fallback Config ID: {self.fallback_config_id}")
        # elif strategy == "loadbalance" and self.loadbalance_config_id:
        #     portkey_specific_headers["config"] = self.loadbalance_config_id
        #     logger.debug(f"Using Loadbalance Config ID: {self.loadbalance_config_id}")
        elif strategy == "premium": # Fallback and Loadbalance are commented out
            if self.premium_config_id:
                portkey_specific_headers["config"] = self.premium_config_id
                logger.debug(f"Using Premium Config ID: {self.premium_config_id}")
            elif self.gemini_pro_virtual_key: 
                portkey_specific_headers["config"] = {
                    "targets": [{
                        "virtual_key": self.gemini_pro_virtual_key,
                        "override_params": {"model": "gemini-1.5-pro-latest"}
                    }]
                }
                logger.debug(f"Using dynamic Premium strategy targeting virtual key: {self.gemini_pro_virtual_key}")
            else:
                logger.warning("Premium strategy selected, but neither PREMIUM_CONFIG_ID nor GEMINI_PRO_VIRTUAL_KEY is set. Defaulting to a non-configured state for premium.")
                # If fallback_config_id was intended here, it's now commented out.
                # Consider if an error should be raised or if Portkey handles no config being set.
                # For now, if premium isn't configured, no specific config ID will be set by this branch.
                # The self.target_config_id might still apply if set.
                # If neither target_config_id nor premium is configured, Portkey might use a default virtual key or error.
                if not self.target_config_id: # Only log error if no persona config is overriding
                    logger.error("Premium strategy failed: No Premium config and no target_config_id. Portkey might use default virtual key or error.")
        
        # Update metadata one last time in case strategy_used was changed
        portkey_specific_headers["metadata"] = current_metadata
        
        # createHeaders will structure these appropriately for Portkey
        return createHeaders(**portkey_specific_headers)
    
    async def chat_completion_with_tools(
        self,
        messages: list,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None, # Changed to Dict to match OpenAI SDK
        strategy: str = "fallback",
        priority: str = "normal",
        user_metadata: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None, # Added model_override
        **kwargs
    ):
        """Create chat completion with tools support and specified strategy."""
        
        headers = self.get_headers_for_strategy(strategy, priority, user_metadata_param=user_metadata)
        
        request_params: Dict[str, Any] = {
            "model": model_override if model_override else self.default_model_for_sdk, # Use override if provided
            "messages": messages,
            **kwargs  # Include other parameters like max_tokens, temperature, etc.
        }
        
        if tools:
            request_params["tools"] = tools
        if tool_choice:
            request_params["tool_choice"] = tool_choice
        
        try:
            logger.debug(f"Portkey Request Params: {request_params}")
            logger.debug(f"Portkey Request Params: {request_params}")
            logger.debug(f"Portkey Headers for chat.completions.create: {headers}") # Log the headers being passed

            response = await self.client.chat.completions.create(
                extra_headers=headers, # Use extra_headers to pass Portkey headers
                **request_params
            )
            
            logger.info(
                f"Chat completion with tools successful",
                extra={
                    "strategy_used": strategy,
                    "priority": priority,
                    "tools_count": len(tools) if tools else 0,
                    "tool_choice": tool_choice,
                    "portkey_id": getattr(response, 'id', None) # Portkey ID is often in response.id
                }
            )
            return response
            
        except Exception as e:
            logger.error(f"Chat completion with tools failed: {str(e)}", exc_info=True)
            raise

    async def astream_chat_completion_content(
        self,
        messages: list, # Expects list of dicts: [{"role": "user", "content": "..."}]
        strategy: str = "fallback",
        priority: str = "normal",
        user_metadata: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None, # Added model_override
        **kwargs
    ):
        """
        Streams chat completion content using the specified strategy.
        Yields content chunks (strings).
        """
        headers = self.get_headers_for_strategy(strategy, priority, user_metadata_param=user_metadata)
        
        request_params: Dict[str, Any] = {
            "model": model_override if model_override else self.default_model_for_sdk, # Use override if provided
            "messages": messages,
            "stream": True, # Enable streaming
            **kwargs 
        }
        
        try:
            logger.debug(f"Portkey Streaming Request Params: {request_params}")
            logger.debug(f"Portkey Headers for streaming chat.completions.create: {headers}")

            stream_response = await self.client.chat.completions.create(
                extra_headers=headers,
                **request_params
            )
            
            async for chunk in stream_response:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    content = delta.content
                    if content:
                        yield content
            
            logger.info(
                f"Streaming chat completion successful",
                extra={
                    "strategy_used": strategy,
                    "priority": priority,
                    # Portkey ID might not be available until stream is fully processed or in metadata
                }
            )
            
        except Exception as e:
            logger.error(f"Streaming chat completion failed: {str(e)}", exc_info=True)
            # Yield a user-friendly error message or re-raise
            yield f"Error during streaming: {str(e)}" # Or handle more gracefully
            # raise # Optionally re-raise
