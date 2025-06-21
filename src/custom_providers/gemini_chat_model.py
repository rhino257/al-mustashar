import asyncio
import json
import logging
import os
from portkey_ai import Portkey, PORTKEY_GATEWAY_URL, createHeaders # Added PORTKEY_GATEWAY_URL and createHeaders
from openai import AsyncOpenAI # Added AsyncOpenAI
from typing import Any, Dict, List, Optional, Type, Sequence, Union, Callable, Literal, AsyncIterator, Iterator, Tuple # Updated imports

# Use Pydantic V1 from langchain_core for compatibility with BaseChatModel
from langchain_core.pydantic_v1 import Field, SecretStr, BaseModel as LangchainPydanticV1BaseModel, root_validator
# Import Pydantic V2 BaseModel for tool schema type hinting
from pydantic import BaseModel as PydanticV2BaseModel
# PrivateAttr is a Pydantic V2 feature. For V1, we manage private fields by convention (underscore)
# and exclude them in dict() calls if necessary. So, PrivateAttr import is removed.

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    # generate_from_stream, # Not directly used in the new code, can be removed if not needed elsewhere
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ChatMessage, # Keep if used
    SystemMessage,
    ToolMessage, # Keep if used
    # FunctionMessage, # Keep if used, Gemini might use ToolMessage more
    ToolCall,
    InvalidToolCall,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
# from langchain_core.utils import get_from_dict_or_env # Not directly used in new code
from langchain_core.utils.function_calling import convert_to_openai_tool # Used as a helper
from langchain_core.runnables import Runnable # For type hinting bind_tools return
from langchain_core.runnables.config import RunnableConfig # For type hinting bind_tools if needed

import google.generativeai as genai
from google.generativeai import types as genai_types
from google.generativeai.types import HarmCategory, HarmBlockThreshold, SafetySettingDict # Keep for safety settings
from google.api_core import exceptions as google_exceptions # Keep for error handling

import tenacity # Keep for retries
from ratelimit import limits, sleep_and_retry, RateLimitException # Keep for rate limiting

logger = logging.getLogger(__name__)

# Default RPM limit, can be overridden by environment variable GEMINI_RPM
DEFAULT_GEMINI_RPM = 60 # As per original

class GeminiChatModel(BaseChatModel):
    """
    Custom ChatModel for Google Gemini, compliant with Pydantic V2 and Langchain tool binding.
    """
    # Standard Langchain configurable parameters, ensure these are Pydantic fields
    model_name: str = Field(default="gemini-1.5-flash-latest", alias="model")
    temperature: Optional[float] = Field(default=0.1) # Default from original, can be None
    top_p: Optional[float] = Field(default=None)
    top_k: Optional[int] = Field(default=None)
    max_output_tokens: Optional[int] = Field(default=None)
    candidate_count: Optional[int] = Field(default=None) # Gemini specific, usually 1 for chat
    stop_sequences: Optional[List[str]] = Field(default=None)
    
    safety_settings: Optional[List[Dict[str, Any]]] = Field(default=None)
    """
    A list of safety settings for the model. Each dictionary should have "category" and "threshold" keys.
    Example: [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]
    Category and threshold values can be strings (e.g., "HARM_CATEGORY_HARASSMENT") or
    the respective enum members (HarmCategory.HARM_CATEGORY_HARASSMENT).
    """

    # Your custom parameters
    generation_kwargs: Dict[str, Any] = Field(default_factory=dict)

    # Client and other internal state
    google_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")

    # Placeholder for the actual Gemini client instance
    # With Pydantic V1, fields starting with _ are 'private' by convention.
    _client: Any = Field(default=None, exclude=True)
    _last_system_instruction: Optional[str] = Field(default=None, exclude=True)
    # portkey_client: Any = Field(default=None, exclude=True) # Native Portkey SDK client
    async_openai_client_for_portkey: Optional[AsyncOpenAI] = Field(default=None, exclude=True) # OpenAI SDK client for Portkey

    # Internal state attributes will be set using object.__setattr__ in __init__
    # No class-level type hints for _client_initialized and _google_api_key_configured needed here

    # RPM limit from original, ensure it's handled if needed by decorators
    RPM_LIMIT: int = int(os.getenv("GEMINI_RPM", str(DEFAULT_GEMINI_RPM)))
    # rpm_limit_instance field removed as it caused issues and the decorator uses the class RPM_LIMIT

    use_portkey_for_generate: bool = Field(default=True) # Flag to control Portkey usage for _agenerate

    def __init__(self, **data: Any):
        logger.info("GeminiChatModel __init__ called") 
        super().__init__(**data)
        # Use object.__setattr__ to initialize internal flags,
        # bypassing Pydantic's __setattr__ for these specific assignments.
        object.__setattr__(self, '_client_initialized', False)
        object.__setattr__(self, '_google_api_key_configured', False)
        # Note: _client and _last_system_instruction are Pydantic Fields
        # and are handled by super().__init__(**data).
        # Initialization of _client can happen here or lazily in _initialize_client
        # For lazy init, ensure _initialize_client is called before any API interaction.

    def _initialize_client(self, system_instruction: Optional[str] = None) -> None:
        """
        Initializes the Gemini API client (genai.GenerativeModel).
        Re-initializes if the client is not set or if the system_instruction changes.
        """
        # Check if re-initialization is needed
        current_api_key_secret = self.google_api_key.get_secret_value() if self.google_api_key else os.environ.get("GOOGLE_API_KEY")
        
        # Simple check: if client exists, system instruction matches, and API key seems configured.
        # A more robust check might involve comparing the actual key if it could change.
        if self._client and self._client_initialized and self._last_system_instruction == system_instruction and self._google_api_key_configured:
            return

        api_key_str: Optional[str] = None
        if self.google_api_key:
            api_key_str = self.google_api_key.get_secret_value()
        else:
            api_key_str = os.environ.get("GOOGLE_API_KEY")

        if not api_key_str:
            raise ValueError(
                "Google API Key not found. Set GOOGLE_API_KEY environment variable "
                "or pass google_api_key (api_key alias) parameter."
            )

        try:
            # Configure genai if not already done or if key might have changed.
            # This is tricky with a global configure. Assuming SDK handles it.
            genai.configure(api_key=api_key_str)
            object.__setattr__(self, '_google_api_key_configured', True)
        except Exception as e:
            logger.error(f"Failed to configure Google API key: {e}")
            object.__setattr__(self, '_google_api_key_configured', False)
            raise

        # Prepare GenerationConfig for the model
        # These are defaults for the model instance. Per-call settings can override.
        gen_config_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
            "candidate_count": self.candidate_count if self.candidate_count is not None else 1, # Default to 1 for chat
            # stop_sequences are typically per-request via generation_kwargs or direct param
        }
        # Filter out None values to let Gemini use its defaults
        generation_config_at_model = genai_types.GenerationConfig(
            **{k: v for k, v in gen_config_params.items() if v is not None}
        )
        
        model_safety_settings = self._prepare_safety_settings(self.safety_settings)
        
        # Tools and tool_config from self.generation_kwargs can be set at model level if desired,
        # but Langchain's bind_tools typically applies them per-call by creating a new model instance.
        # For this _initialize_client, we'll use what's in self.generation_kwargs if present.
        model_level_tools = self.generation_kwargs.get("tools") # Expects Gemini formatted tools
        model_level_tool_config = self.generation_kwargs.get("tool_config")

        object.__setattr__(self, '_client', genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction if system_instruction else None,
            generation_config=generation_config_at_model,
            safety_settings=model_safety_settings,
            tools=model_level_tools,
            tool_config=model_level_tool_config,
        ))
        object.__setattr__(self, '_client_initialized', True)
        object.__setattr__(self, '_last_system_instruction', system_instruction)
        logger.info(f"Google Generative AI client initialized for model: {self.model_name}")

        # --- New Portkey Client Initialization ---
        portkey_api_key = os.environ.get("PORTKEY_API_KEY")
        google_virtual_key_id = os.environ.get("GOOGLE_VIRTUAL_KEY_ID")

        if not portkey_api_key:
            logger.error("PORTKEY_API_KEY environment variable not set.")
            # For debugging, let's log what was found, if anything
            logger.debug(f"Value of PORTKEY_API_KEY from env: {portkey_api_key}")
            raise ValueError("PORTKEY_API_KEY environment variable not set.")
        else:
            # Log a portion of the key for verification
            pk_key_display = f"{portkey_api_key[:5]}...{portkey_api_key[-4:]}" if len(portkey_api_key) > 9 else "Key too short to partially display"
            logger.info(f"PORTKEY_API_KEY loaded, partial view: {pk_key_display}")

        if not google_virtual_key_id:
            logger.error("GOOGLE_VIRTUAL_KEY_ID environment variable not set. This is required for Portkey calls to Gemini.")
            raise ValueError("GOOGLE_VIRTUAL_KEY_ID environment variable not set. This is required for Portkey calls to Gemini.")

        try:
            # The native Portkey SDK client is no longer needed as all calls will go through AsyncOpenAI client.
            # self.portkey_client = Portkey(
            #     api_key=portkey_api_key, # This is the Portkey API Key
            #     virtual_key=google_virtual_key_id # Set the virtual key for Google here
            # )
            # logger.info(f"Portkey client (native portkey-ai SDK) initialized. API Key: {portkey_api_key[:5]}..., Virtual Key: {google_virtual_key_id}")
            
            # Initialize OpenAI client configured for Portkey, as per Portkey support suggestion
            try:
                # Prepare Portkey config for Gemini-specific parameters
                gemini_override_params = {
                    "model": self.model_name, # Ensure model name is also in override_params
                }
                if self.top_k is not None:
                    gemini_override_params["top_k"] = self.top_k
                if self.candidate_count is not None:
                    gemini_override_params["candidate_count"] = self.candidate_count
                if self.safety_settings is not None:
                    # Convert safety settings to the format expected by Portkey/Gemini API
                    # This might require re-preparing them if they are Langchain enum objects
                    prepared_safety_settings = self._prepare_safety_settings(self.safety_settings)
                    if prepared_safety_settings:
                        # Convert enum values back to strings for JSON serialization if needed by Portkey
                        serializable_safety_settings = [
                            {"category": s["category"].name, "threshold": s["threshold"].name}
                            for s in prepared_safety_settings
                        ]
                        gemini_override_params["safety_settings"] = serializable_safety_settings

                portkey_config = {
                    "provider": "google",
                    "api_key": api_key_str, # Use the actual Google API key here for Portkey to pass through
                    "override_params": gemini_override_params
                }

                self._portkey_gemini_config = portkey_config # Store for per-request headers
                self.async_openai_client_for_portkey = AsyncOpenAI(
                    api_key="dummy",  # Dummy key as per Portkey's instruction
                    base_url=PORTKEY_GATEWAY_URL,
                    default_headers=createHeaders(
                        api_key=portkey_api_key, # Actual Portkey API Key
                        virtual_key=google_virtual_key_id,
                        config=self._portkey_gemini_config # Pass the config object here
                    )
                )
                logger.info(f"AsyncOpenAI client for Portkey initialized. Base URL: {self.async_openai_client_for_portkey.base_url}, Headers created with config (including Gemini override_params).")
            except Exception as e:
                logger.error(f"Failed to initialize AsyncOpenAI client for Portkey: {e}", exc_info=True)
                self.async_openai_client_for_portkey = None # Set to None on failure
                raise # Re-raise the exception to make it critical, as this is the primary client now

        except Exception as e:
            logger.error(f"Failed to initialize Portkey related clients: {e}")
            raise # Re-raise the exception to make it critical

    def _convert_messages_to_portkey_format(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        """
        Converts a list of LangChain BaseMessage objects to Portkey's (OpenAI-like) message format.
        System messages are included directly. Tool messages are ignored for now.
        """
        portkey_messages: List[Dict[str, Any]] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                portkey_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                portkey_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                # For now, ignoring tool_calls in AIMessage for Portkey conversion in this step
                if message.content:
                    portkey_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, ChatMessage):
                role = message.role.lower()
                if role in ["user", "assistant", "system"]:
                    portkey_messages.append({"role": role, "content": message.content})
                else:
                    logger.warning(f"Unsupported ChatMessage role '{role}' for Portkey. Converting to user.")
                    portkey_messages.append({"role": "user", "content": f"[{role.capitalize()} Note]: {message.content}"})
            # ToolMessage instances are ignored for now as per instructions for _agenerate
            elif isinstance(message, ToolMessage):
                logger.debug(f"Ignoring ToolMessage for Portkey conversion in _agenerate: {message.tool_call_id}")
            else:
                logger.warning(f"Unsupported message type for Portkey conversion: {type(message)}. Skipping.")
        return portkey_messages

    def _convert_messages_to_gemini_format(
        self, messages: List[BaseMessage]
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Converts a list of LangChain BaseMessage objects to Gemini's content format.
        Extracts the first system message as system_instruction.
        Handles ToolMessages by converting them to Gemini's function response format.
        """
        logger.debug(f"[_convert_messages_to_gemini_format] Input messages: {messages}")
        system_instruction: Optional[str] = None
        gemini_contents: List[Dict[str, Any]] = []
        
        processed_messages = []
        for i, message in enumerate(messages):
            if isinstance(message, SystemMessage):
                if system_instruction is None:
                    system_instruction = message.content
                else:
                    logger.warning("Multiple SystemMessages found. Only the first one is used as system_instruction. Subsequent ones are currently converted to user messages or ignored.")
                    # Optionally, convert subsequent system messages to user messages or handle as error
                    # For now, let's append as a user message to not lose the content, though this might not be ideal.
                    # gemini_contents.append({'role': 'user', 'parts': [{'text': f"[System Note: {message.content}]"}]})
                continue # SystemMessages are handled at model init or passed separately
            processed_messages.append(message)

        for message in processed_messages:
            if isinstance(message, HumanMessage):
                gemini_contents.append({'role': 'user', 'parts': [{'text': message.content}]})
            elif isinstance(message, AIMessage):
                if message.tool_calls: # AIMessage with tool calls
                    function_calls_parts = []
                    for tc in message.tool_calls:
                        function_calls_parts.append(
                            genai_types.Part(
                                function_call=genai_types.FunctionCall(name=tc["name"], args=tc["args"])
                            )
                        )
                    if function_calls_parts:
                         gemini_contents.append({'role': 'model', 'parts': function_calls_parts})
                    elif message.content: # If tool_calls was empty but content exists
                         gemini_contents.append({'role': 'model', 'parts': [{'text': message.content}]})
                    # If both are empty, it's an empty AIMessage, let it pass as empty model turn
                else: # Regular AIMessage
                    gemini_contents.append({'role': 'model', 'parts': [{'text': message.content}]})
            elif isinstance(message, ToolMessage): # Changed from FunctionMessage
                # Gemini expects a "function" role for tool responses (or "tool" role in some contexts)
                # The content should be a FunctionResponse part.
                # Ensure content is serializable if it's complex.
                # Langchain ToolMessage.content is typically a string (often JSON stringified).
                # Gemini's FunctionResponse expects 'name' (of the function called) and 'response' (dict with 'content').
                
                # We need to ensure the content is structured correctly for Gemini's FunctionResponse.
                # The 'response' field in FunctionResponse usually contains the actual data returned by the tool.
                # If ToolMessage.content is a JSON string, parse it.
                try:
                    tool_response_content = json.loads(message.content) if isinstance(message.content, str) else message.content
                except json.JSONDecodeError:
                    tool_response_content = message.content # Use as is if not JSON

                gemini_contents.append({
                    'role': 'function', # Or 'tool' - check Gemini docs for exact role name for tool responses
                    'parts': [
                        genai_types.Part(
                            function_response=genai_types.FunctionResponse(
                                name=message.name if hasattr(message, 'name') else message.tool_call_id, # tool_call_id is more standard
                                response={'content': tool_response_content} # Gemini expects response to be a dict, often with a 'content' key
                            )
                        )
                    ]
                })
            elif isinstance(message, ChatMessage): # General ChatMessage
                role = message.role.lower()
                if role == "user":
                    gemini_contents.append({'role': 'user', 'parts': [{'text': message.content}]})
                elif role == "model" or role == "assistant":
                    gemini_contents.append({'role': 'model', 'parts': [{'text': message.content}]})
                # Other roles like "tool" or "function" might need specific handling if not covered by ToolMessage
                else:
                    logger.warning(f"Unsupported ChatMessage role '{role}' being converted to user message for Gemini.")
                    gemini_contents.append({'role': 'user', 'parts': [{'text': f"[{role.capitalize()} Note: {message.content}]"}]})
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        
        logger.debug(f"[_convert_messages_to_gemini_format] Extracted system_instruction: '{system_instruction}'")
        logger.debug(f"[_convert_messages_to_gemini_format] Generated gemini_contents: {gemini_contents}")
        return system_instruction, gemini_contents

    def _prepare_safety_settings(self, settings_list: Optional[List[Dict[str, Any]]]) -> Optional[List[SafetySettingDict]]:
        # This method seems fine from the original, ensure HarmCategory and HarmBlockThreshold are correctly used.
        if not settings_list:
            return None
        
        converted_settings: List[SafetySettingDict] = []
        for setting_dict in settings_list:
            category_val = setting_dict.get("category")
            threshold_val = setting_dict.get("threshold")

            if not category_val or not threshold_val:
                logger.warning(f"Skipping malformed safety setting (missing category or threshold): {setting_dict}")
                continue

            final_category: Optional[HarmCategory] = None
            if isinstance(category_val, str):
                try:
                    final_category = getattr(HarmCategory, category_val.upper())
                except AttributeError: # Check if it's already a valid enum value string like "HARM_CATEGORY_HARASSMENT"
                    try:
                        final_category = HarmCategory[category_val.upper()]
                    except KeyError:
                        logger.warning(f"Invalid HarmCategory string: {category_val} in {setting_dict}. Valid names: {[e.name for e in HarmCategory]}")
                        continue
            elif isinstance(category_val, HarmCategory):
                final_category = category_val
            else:
                logger.warning(f"Unsupported HarmCategory type: {type(category_val)} in {setting_dict}")
                continue

            final_threshold: Optional[HarmBlockThreshold] = None
            if isinstance(threshold_val, str):
                try:
                    final_threshold = getattr(HarmBlockThreshold, threshold_val.upper())
                except AttributeError:
                    try:
                        final_threshold = HarmBlockThreshold[threshold_val.upper()]
                    except KeyError:
                        logger.warning(f"Invalid HarmBlockThreshold string: {threshold_val} in {setting_dict}. Valid names: {[e.name for e in HarmBlockThreshold]}")
                        continue
            elif isinstance(threshold_val, HarmBlockThreshold):
                final_threshold = threshold_val
            else:
                logger.warning(f"Unsupported HarmBlockThreshold type: {type(threshold_val)} in {setting_dict}")
                continue
            
            converted_settings.append(SafetySettingDict(category=final_category, threshold=final_threshold))
        return converted_settings if converted_settings else None

    def _create_chat_result(self, response: Union[genai_types.GenerateContentResponse, Any], from_portkey: bool = False) -> ChatResult:
        generations = []
        llm_output = {}

        if from_portkey:
            if not response or not hasattr(response, 'choices') or not response.choices:
                message_content = "No response or choices from Portkey/Gemini."
                finish_reason_str = "NO_CHOICES_PORTKEY"
                if hasattr(response, 'error') and response.error: # Check for Portkey error object
                    message_content = f"Portkey Error: {response.error.message if hasattr(response.error, 'message') else str(response.error)}"
                    finish_reason_str = "ERROR_PORTKEY"
                elif hasattr(response, 'message') and response.message: # Direct error message from Portkey
                    message_content = f"Portkey Error: {response.message}"
                    finish_reason_str = "ERROR_PORTKEY"

                generations.append(ChatGeneration(
                    message=AIMessage(content=message_content, additional_kwargs={"error": True}),
                    generation_info={"finish_reason": finish_reason_str}
                ))
            else:
                choice = response.choices[0]
                content = choice.message.content or ""
                # Map Portkey finish reasons if necessary, or use as is.
                # OpenAI finish reasons: "stop", "length", "content_filter", "tool_calls", "function_call"
                finish_reason_str = choice.finish_reason if choice.finish_reason else "UNKNOWN_PORTKEY"
                
                # For this step, we are not handling tool_calls via Portkey in _agenerate
                ai_message_kwargs = {}
                # if choice.message.tool_calls:
                #     # Convert Portkey/OpenAI tool_calls to Langchain ToolCall/InvalidToolCall
                #     # This logic will be added in a later step.
                #     pass

                generations.append(ChatGeneration(
                    message=AIMessage(content=content, **ai_message_kwargs),
                    generation_info={"finish_reason": finish_reason_str}
                ))

            if hasattr(response, 'usage') and response.usage:
                llm_output["token_usage"] = {
                    "prompt_token_count": response.usage.prompt_tokens,
                    "candidates_token_count": response.usage.completion_tokens, # Portkey uses completion_tokens
                    "total_token_count": response.usage.total_tokens,
                }
            if hasattr(response, 'id'):
                llm_output["portkey_id"] = response.id
            
            return ChatResult(generations=generations, llm_output=llm_output if llm_output else None)

        # Existing Google SDK response parsing
        elif isinstance(response, genai_types.GenerateContentResponse):
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                error_message = (
                    f"Prompt blocked due to {response.prompt_feedback.block_reason.name}. "
                    f"Safety ratings: {response.prompt_feedback.safety_ratings}"
                )
                ai_message = AIMessage(content=error_message, additional_kwargs={"blocked": True, "prompt_feedback": str(response.prompt_feedback)})
                generations.append(ChatGeneration(message=ai_message, generation_info={"finish_reason": "PROMPT_BLOCKED"}))
            elif not response.candidates:
                ai_message = AIMessage(content="No candidates returned from Gemini.", additional_kwargs={"error": True, "no_candidates": True})
                generations.append(ChatGeneration(message=ai_message, generation_info={"finish_reason": "NO_CANDIDATES"}))
            else:
                candidate = response.candidates[0]
                tool_calls = []
                invalid_tool_calls = []
            response_text = ""

            # Gemini API v1beta+ uses candidate.function_calls
            if hasattr(candidate, 'function_calls') and candidate.function_calls:
                for fc in candidate.function_calls:
                    try:
                        # Ensure args is a dict, as Langchain expects
                        args_dict = dict(fc.args) if fc.args else {}
                        tool_calls.append(
                            ToolCall(name=fc.name, args=args_dict, id=f"call_{fc.name.replace('-', '_')}_{os.urandom(4).hex()}")
                        )
                    except Exception as e:
                        invalid_tool_calls.append(
                            InvalidToolCall(name=fc.name, args=str(fc.args), id=f"call_{fc.name.replace('-', '_')}_{os.urandom(4).hex()}", error=str(e))
                        )
            # Fallback or alternative: check content.parts for function_call (older Gemini or specific cases)
            elif candidate.content and candidate.content.parts and hasattr(candidate.content.parts[0], 'function_call'):
                fc = candidate.content.parts[0].function_call
                try:
                    args_dict = dict(fc.args) if fc.args else {}
                    tool_calls.append(
                        ToolCall(name=fc.name, args=args_dict, id=f"call_{fc.name.replace('-', '_')}_{os.urandom(4).hex()}")
                    )
                except Exception as e:
                    invalid_tool_calls.append(
                        InvalidToolCall(name=fc.name, args=str(fc.args), id=f"call_{fc.name.replace('-', '_')}_{os.urandom(4).hex()}", error=str(e))
                    )
            
            # Extract text content if no tool calls or if text coexists
            if not tool_calls and not invalid_tool_calls:
                try:
                    if candidate.content and candidate.content.parts:
                        # Check if response_schema was used (this implies JSON output was expected)
                        # We need a way to know if response_schema was active for this call.
                        # For now, let's assume if the first part is not text, it might be structured JSON.
                        first_part = candidate.content.parts[0]
                        if hasattr(first_part, 'text') and first_part.text:
                             response_text = "".join(part.text for part in candidate.content.parts if hasattr(part, "text"))
                        else:
                            # If no text in parts, and JSON mode was likely active,
                            # Gemini might place the structured data directly in the part.
                            # The SDK might have already parsed it if response_schema was a Pydantic type.
                            # If response_schema was List[str], parts[0] might be the list.
                            # We need to serialize it back to a JSON string for StrOutputParser.
                            # This is a common pattern if the model respects response_schema.
                            # Note: This part is speculative and depends on exact SDK behavior with List[str] schema.
                            # If parts[0] is the actual list of strings:
                            if isinstance(first_part, list) or isinstance(first_part, dict): # Or check specific Gemini part type for JSON
                                response_text = json.dumps(first_part)
                                logger.info(f"Extracted JSON from structured part: {response_text}")
                            elif hasattr(response, 'text') and response.text: # Fallback to response.text
                                response_text = response.text
                            else: # If still nothing, it's genuinely empty or in an unexpected format
                                response_text = "" 
                                logger.info("No text content found in parts, and first part not recognized as direct JSON list/dict.")
                                
                    elif hasattr(response, 'text'): # Fallback if no parts
                        response_text = response.text
                except ValueError as e: # Likely content blocked
                    logger.warning(f"Error extracting text from response candidate: {e}. Candidate: {candidate}")
                    response_text = (
                        f"Content generation error (likely blocked). Finish reason: {candidate.finish_reason.name if candidate.finish_reason else 'UNKNOWN'}. "
                        f"Safety ratings: {candidate.safety_ratings if candidate.safety_ratings else 'N/A'}"
                    )
                    llm_output["blocked_reason"] = candidate.finish_reason.name if candidate.finish_reason else "BLOCKED_CONTENT"
                except Exception as e: # Catch other unexpected errors during extraction
                    logger.error(f"Unexpected error extracting content from candidate parts: {e}", exc_info=True)
                    response_text = ""


            ai_msg_kwargs = {}
            if tool_calls:
                ai_msg_kwargs["tool_calls"] = tool_calls
            if invalid_tool_calls:
                ai_msg_kwargs["invalid_tool_calls"] = invalid_tool_calls
            
            # If there was a block reason from safety ratings even with content/tool_calls
            if candidate.finish_reason and candidate.finish_reason.name in ["SAFETY", "RECITATION", "OTHER"]:
                 llm_output["blocked_reason"] = candidate.finish_reason.name
                 # Prepend warning to text if content exists
                 if response_text and not tool_calls: # Don't prepend if it's a tool call response
                     response_text = f"[WARNING: Response may be incomplete due to finish reason: {candidate.finish_reason.name}] {response_text}"
            
            final_finish_reason_google = "UNKNOWN"
            if candidate.finish_reason:
                final_finish_reason_google = candidate.finish_reason.name
            elif tool_calls: # If no explicit finish reason but tool calls exist
                final_finish_reason_google = "TOOL_CALLS"


            generations.append(ChatGeneration(
                message=AIMessage(content=response_text, **ai_msg_kwargs),
                generation_info={"finish_reason": final_finish_reason_google}
            ))

        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            llm_output["token_usage"] = {
                "prompt_token_count": response.usage_metadata.prompt_token_count,
                "candidates_token_count": response.usage_metadata.candidates_token_count, # Sum of tokens for all candidates
                "total_token_count": response.usage_metadata.total_token_count,
            }
        
        return ChatResult(generations=generations, llm_output=llm_output if llm_output else None)

    @sleep_and_retry
    @limits(calls=RPM_LIMIT, period=60) # Ensure RPM_LIMIT is accessible (class variable)
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=60),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception_type((
            google_exceptions.DeadlineExceeded,
            google_exceptions.ServiceUnavailable,
            google_exceptions.ResourceExhausted,
            google_exceptions.InternalServerError,
            RateLimitException
        )),
        reraise=True
    )
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Ensure clients are initialized. _initialize_client handles both.
        # For Google SDK, system_instruction is passed during _initialize_client if it's the first system message.
        # For Portkey, system message is part of the messages list.
        # We call _initialize_client without system_instruction here;
        # if Google SDK path is taken, _convert_messages_to_gemini_format will extract it
        # and _initialize_client will be called again with it if needed.
        self._initialize_client(system_instruction=None) 

        # Ensure clients are initialized. _initialize_client handles both.
        # For Google SDK, system_instruction is passed during _initialize_client if it's the first system message.
        # For Portkey, system message is part of the messages list.
        # We call _initialize_client without system_instruction here;
        # if Google SDK path is taken, _convert_messages_to_gemini_format will extract it
        # and _initialize_client will be called again with it if needed.
        self._initialize_client(system_instruction=None) 

        # Updated logic to use OpenAI SDK pattern with Portkey
        if self.use_portkey_for_generate and self.async_openai_client_for_portkey:
            logger.info("Using OpenAI SDK pattern with Portkey Gateway for _agenerate")
            
            portkey_formatted_messages = self._convert_messages_to_portkey_format(messages)

            # Dynamically use self.model_name for the model parameter
            effective_model_name_for_portkey_openai = self.model_name
            logger.info(f"Using model name for Portkey OpenAI SDK call: {effective_model_name_for_portkey_openai} (Original instance model_name: {self.model_name})")

            openai_params: Dict[str, Any] = {
                "model": effective_model_name_for_portkey_openai, 
                "messages": portkey_formatted_messages,
            }
            if self.temperature is not None:
                openai_params["temperature"] = self.temperature
            if self.max_output_tokens is not None:
                openai_params["max_tokens"] = self.max_output_tokens # OpenAI SDK uses max_tokens
            if self.top_p is not None:
                openai_params["top_p"] = self.top_p
            if stop:
                openai_params["stop"] = stop
            
            # Gemini-specific parameters (top_k, safety_settings, candidate_count) are now
            # handled via the `config` in `default_headers` during client initialization.
            # No `override_params` are used here directly in the `create` call.

            try:
                logger.debug(f"OpenAI SDK to Portkey: chat.completions.create params: {json.dumps(openai_params, indent=2)}")
                
                request_specific_headers = None
                run_config: Optional[RunnableConfig] = kwargs.get("config")
                if run_config and isinstance(run_config, dict): # Langchain passes config as dict
                    user_metadata_from_config = run_config.get("metadata")
                    if user_metadata_from_config and isinstance(user_metadata_from_config, dict):
                        user_id = user_metadata_from_config.get("user_id")
                        if user_id:
                            final_metadata_for_request = {**user_metadata_from_config}
                            if "_user" not in final_metadata_for_request:
                                final_metadata_for_request["_user"] = user_id
                            
                            pk_api_key_for_req = os.environ.get("PORTKEY_API_KEY")
                            google_virtual_key_for_req = os.environ.get("GOOGLE_VIRTUAL_KEY_ID")

                            if pk_api_key_for_req and google_virtual_key_for_req and hasattr(self, '_portkey_gemini_config'):
                                request_specific_headers = createHeaders(
                                    api_key=pk_api_key_for_req,
                                    virtual_key=google_virtual_key_for_req,
                                    metadata=final_metadata_for_request,
                                    config=self._portkey_gemini_config 
                                )
                                logger.info(f"Portkey _agenerate: Using request-specific headers for user_id: {user_id}")
                            else:
                                logger.warning("Portkey _agenerate: Missing API keys or Portkey config for request-specific headers.")

                if request_specific_headers:
                    pk_response = await self.async_openai_client_for_portkey.with_options(
                        headers=request_specific_headers
                    ).chat.completions.create(**openai_params)
                else:
                    pk_response = await self.async_openai_client_for_portkey.chat.completions.create(**openai_params)
                
                return self._create_chat_result(pk_response, from_portkey=True)
            except Exception as e:
                logger.error(f"OpenAI SDK pattern to Portkey API error during _agenerate: {e}", exc_info=True)
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"OpenAI SDK/Portkey Error: {str(e)}", additional_kwargs={"error": True}))])

        elif self.use_portkey_for_generate and not self.async_openai_client_for_portkey:
            logger.error("Portkey usage is enabled, but AsyncOpenAI client for Portkey is not initialized. Erroring out.")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Configuration error: AsyncOpenAI client for Portkey not available.", additional_kwargs={"error": True}))])
                
        else: # Original Google SDK path
            logger.info("Using Google SDK client for _agenerate (Portkey not used or not configured for OpenAI SDK pattern).")
            system_instruction, gemini_formatted_messages = self._convert_messages_to_gemini_format(messages)
            # Re-call _initialize_client if system_instruction was extracted and is different
            if system_instruction != self._last_system_instruction:
                self._initialize_client(system_instruction=system_instruction)

            # Prepare generation_config for the API call
            # Start with model defaults, override with call-specific kwargs
            gen_config_params = {
                "stop_sequences": stop, # Per-call stop sequences
                "candidate_count": self.candidate_count if self.candidate_count is not None else 1, # Use instance's candidate_count
            # These can be overridden by kwargs if passed directly to _agenerate
            "max_output_tokens": kwargs.get("max_output_tokens", self.max_output_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", self.top_k),
        }
        # Filter out None values to let Gemini use its defaults
        final_gen_config_params = {k: v for k, v in gen_config_params.items() if v is not None}
        
        # Merge model's generation_kwargs with per-call kwargs
        # Per-call kwargs (passed directly to _agenerate or via .bind().with_config()) take precedence
        merged_generation_kwargs = {**self.generation_kwargs, **kwargs}

        if "response_mime_type" in merged_generation_kwargs:
            final_gen_config_params["response_mime_type"] = merged_generation_kwargs["response_mime_type"]
        if "response_schema" in merged_generation_kwargs:
            final_gen_config_params["response_schema"] = merged_generation_kwargs["response_schema"]
            
        generation_config_for_call = genai_types.GenerationConfig(**final_gen_config_params)
        
        # Safety settings: from merged_generation_kwargs or instance settings
        effective_safety_settings_input = merged_generation_kwargs.get('safety_settings', self.safety_settings)
        final_safety_settings = self._prepare_safety_settings(effective_safety_settings_input)

        # Tools and tool_config from merged_generation_kwargs
        gemini_tools_for_call = merged_generation_kwargs.get("tools") 
        tool_config_for_call = merged_generation_kwargs.get("tool_config")
        request_options = {"timeout": merged_generation_kwargs.get("request_timeout", 600)}

        try:
            response = await self._client.generate_content_async(
                contents=gemini_formatted_messages,
                generation_config=generation_config_for_call,
                safety_settings=final_safety_settings,
                request_options=request_options,
                tools=gemini_tools_for_call,
                tool_config=tool_config_for_call
            )
            return self._create_chat_result(response)
        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Google API error during generation: {e}")
            if isinstance(e, google_exceptions.InvalidArgument):
                 error_message = f"Invalid argument to Gemini API: {e}"
                 ai_message = AIMessage(content=error_message, additional_kwargs={"error": True, "type": "InvalidArgument"})
                 return ChatResult(generations=[ChatGeneration(message=ai_message)])
            raise
        except Exception as e: # Catch other unexpected errors
            logger.error(f"Unexpected error during Gemini generation: {e}", exc_info=True)
            ai_message = AIMessage(content=f"Unexpected error: {str(e)}", additional_kwargs={"error": True, "type": "Unexpected"})
            return ChatResult(generations=[ChatGeneration(message=ai_message)])

    @sleep_and_retry
    @limits(calls=RPM_LIMIT, period=60)
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=60),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception_type((
            google_exceptions.DeadlineExceeded,
            google_exceptions.ServiceUnavailable,
            google_exceptions.ResourceExhausted,
            google_exceptions.InternalServerError,
            RateLimitException
        )),
        reraise=True
    )
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # Ensure clients are initialized. _initialize_client handles both.
        # For Google SDK, system_instruction is passed during _initialize_client if it's the first system message.
        # For Portkey, system message is part of the messages list.
        # We call _initialize_client without system_instruction here;
        # if Google SDK path is taken, _convert_messages_to_gemini_format will extract it
        # and _initialize_client will be called again with it if needed.
        self._initialize_client(system_instruction=None) 

        # Updated logic to use OpenAI SDK pattern with Portkey for streaming
        if self.use_portkey_for_generate and self.async_openai_client_for_portkey:
            logger.info("Using OpenAI SDK pattern with Portkey Gateway for _astream")
            
            portkey_formatted_messages = self._convert_messages_to_portkey_format(messages)

            # Dynamically use self.model_name for the model parameter
            effective_model_name_for_portkey_openai = self.model_name
            logger.info(f"Using model name for Portkey OpenAI SDK streaming call: {effective_model_name_for_portkey_openai} (Original instance model_name: {self.model_name})")

            openai_params: Dict[str, Any] = {
                "model": effective_model_name_for_portkey_openai, 
                "messages": portkey_formatted_messages,
                "stream": True, # Crucial for streaming
            }
            if self.temperature is not None:
                openai_params["temperature"] = self.temperature
            if self.max_output_tokens is not None:
                openai_params["max_tokens"] = self.max_output_tokens # OpenAI SDK uses max_tokens
            if self.top_p is not None:
                openai_params["top_p"] = self.top_p
            if stop:
                openai_params["stop"] = stop
            
            # Gemini-specific parameters (top_k, safety_settings, candidate_count) are now
            # handled via the `config` in `default_headers` during client initialization.
            # No `override_params` are used here directly in the `create` call.

            try:
                logger.debug(f"OpenAI SDK to Portkey: chat.completions.create (streaming) params: {json.dumps(openai_params, indent=2)}")

                request_specific_headers_stream = None
                run_config_stream: Optional[RunnableConfig] = kwargs.get("config")
                if run_config_stream and isinstance(run_config_stream, dict):
                    user_metadata_from_config_stream = run_config_stream.get("metadata")
                    if user_metadata_from_config_stream and isinstance(user_metadata_from_config_stream, dict):
                        user_id_stream = user_metadata_from_config_stream.get("user_id")
                        if user_id_stream:
                            final_metadata_for_request_stream = {**user_metadata_from_config_stream}
                            if "_user" not in final_metadata_for_request_stream:
                                final_metadata_for_request_stream["_user"] = user_id_stream
                            
                            pk_api_key_for_req_stream = os.environ.get("PORTKEY_API_KEY")
                            google_virtual_key_for_req_stream = os.environ.get("GOOGLE_VIRTUAL_KEY_ID")

                            if pk_api_key_for_req_stream and google_virtual_key_for_req_stream and hasattr(self, '_portkey_gemini_config'):
                                request_specific_headers_stream = createHeaders(
                                    api_key=pk_api_key_for_req_stream,
                                    virtual_key=google_virtual_key_for_req_stream,
                                    metadata=final_metadata_for_request_stream,
                                    config=self._portkey_gemini_config
                                )
                                logger.info(f"Portkey _astream: Using request-specific headers for user_id: {user_id_stream}")
                            else:
                                logger.warning("Portkey _astream: Missing API keys or Portkey config for request-specific streaming headers.")
                
                if request_specific_headers_stream:
                    async_pk_response_stream = await self.async_openai_client_for_portkey.with_options(
                        headers=request_specific_headers_stream
                    ).chat.completions.create(**openai_params)
                else:
                    async_pk_response_stream = await self.async_openai_client_for_portkey.chat.completions.create(**openai_params)
                
                async for chunk in async_pk_response_stream:
                    chunk_text = ""
                    generation_info = {}
                    tool_call_chunks_for_langchain = []

                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            chunk_text = delta.content
                        
                        # Handle tool_call_chunks for streaming
                        if delta.tool_calls:
                            for tc_chunk in delta.tool_calls:
                                # OpenAI SDK streaming tool_calls are structured as:
                                # tc_chunk.index, tc_chunk.id, tc_chunk.function.name, tc_chunk.function.arguments
                                tool_call_chunks_for_langchain.append({
                                    "name": tc_chunk.function.name,
                                    "args": tc_chunk.function.arguments, # This is a string for partial args
                                    "id": tc_chunk.id,
                                    "index": tc_chunk.index
                                })
                        
                        if chunk.choices[0].finish_reason:
                            generation_info["finish_reason"] = chunk.choices[0].finish_reason
                        
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(
                                content=chunk_text, 
                                tool_call_chunks=tool_call_chunks_for_langchain if tool_call_chunks_for_langchain else []
                            ),
                            generation_info=generation_info if generation_info else None,
                        )
                        if run_manager and chunk_text:
                            await run_manager.on_llm_new_token(chunk_text)
            except Exception as e:
                logger.error(f"OpenAI SDK pattern to Portkey API error during _astream: {e}", exc_info=True)
                yield ChatGenerationChunk(message=AIMessageChunk(content=f"OpenAI SDK/Portkey Streaming Error: {str(e)}", additional_kwargs={"error": True}))
                return

        elif self.use_portkey_for_generate and not self.async_openai_client_for_portkey:
            logger.error("Portkey usage is enabled, but AsyncOpenAI client for Portkey is not initialized. Erroring out for streaming.")
            yield ChatGenerationChunk(message=AIMessageChunk(content="Configuration error: AsyncOpenAI client for Portkey not available for streaming.", additional_kwargs={"error": True}))
            return
                
        else: # Original Google SDK path
            logger.info("Using Google SDK client for _astream (Portkey not used or not configured for OpenAI SDK pattern).")
            system_instruction, gemini_formatted_messages = self._convert_messages_to_gemini_format(messages)
            self._initialize_client(system_instruction=system_instruction)

            gen_config_params = {
                "stop_sequences": stop,
                "candidate_count": 1,
                "max_output_tokens": kwargs.get("max_output_tokens", self.max_output_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "top_k": kwargs.get("top_k", self.top_k),
            }
            final_gen_config_params = {k: v for k, v in gen_config_params.items() if v is not None}

            # Merge model's generation_kwargs with per-call kwargs
            merged_generation_kwargs_stream = {**self.generation_kwargs, **kwargs}

            # Add response_mime_type and response_schema if present in merged_generation_kwargs for streaming
            if "response_mime_type" in merged_generation_kwargs_stream:
                final_gen_config_params["response_mime_type"] = merged_generation_kwargs_stream["response_mime_type"]
            if "response_schema" in merged_generation_kwargs_stream:
                final_gen_config_params["response_schema"] = merged_generation_kwargs_stream["response_schema"]
                
            generation_config_for_call = genai_types.GenerationConfig(**final_gen_config_params)
            
            effective_safety_settings_input = merged_generation_kwargs_stream.get('safety_settings', self.safety_settings)
            final_safety_settings = self._prepare_safety_settings(effective_safety_settings_input)

            gemini_tools_for_call_stream = merged_generation_kwargs_stream.get("tools")
            tool_config_for_call_stream = merged_generation_kwargs_stream.get("tool_config")
            request_options_stream = {"timeout": merged_generation_kwargs_stream.get("request_timeout", 600)}

            try:
                async_response_stream = await self._client.generate_content_async(
                    contents=gemini_formatted_messages,
                    generation_config=generation_config_for_call,
                    safety_settings=final_safety_settings,
                    stream=True,
                    request_options=request_options_stream,
                    tools=gemini_tools_for_call_stream,
                    tool_config=tool_config_for_call_stream,
                )

                async for chunk in async_response_stream:
                    chunk_text = ""
                    generation_info = {}
                    tool_call_chunks_for_langchain = [] # For Langchain AIMessageChunk

                    if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                        error_message = (
                            f"Prompt blocked during streaming due to {chunk.prompt_feedback.block_reason.name}. "
                            f"Safety ratings: {chunk.prompt_feedback.safety_ratings}"
                        )
                        logger.warning(error_message)
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(content=error_message, additional_kwargs={"blocked": True, "prompt_feedback": str(chunk.prompt_feedback)}),
                            generation_info={"finish_reason": "PROMPT_BLOCKED"}
                        )
                        return

                    if not chunk.candidates:
                        logger.warning("Stream chunk has no candidates.")
                        continue # Or yield an empty/error chunk

                    candidate = chunk.candidates[0]
                    
                    # Streaming tool calls with Gemini
                    # Gemini's `function_calls` in a streaming candidate are usually complete calls, not partial.
                    # Langchain's `tool_call_chunks` expects partial updates (name, args string, id, index).
                    # This requires careful adaptation. If Gemini streams full tool calls at once,
                    # we might send a single AIMessageChunk with `tool_calls` (not `tool_call_chunks`).
                    # For now, let's assume Gemini might stream parts of a tool call, or a full one.
                    
                    # Check for function calls in the chunk
                    # Based on Gemini's current Python SDK (google-generativeai >= 0.5.0),
                    # when streaming with tools, `chunk.function_calls` will contain a list of
                    # `genai_types.FunctionCall` objects if the model decides to call functions.
                    # These are typically sent as a complete set for that turn, not streamed token by token for the call itself.
                    if hasattr(candidate, 'function_calls') and candidate.function_calls:
                        for fc_idx, fc in enumerate(candidate.function_calls):
                            # Convert Gemini FunctionCall to Langchain ToolCallChunk format
                            # Since Gemini sends the whole call, args will be a dict, convert to JSON string for Langchain chunk
                            args_str = json.dumps(dict(fc.args)) if fc.args else "{}"
                            tool_call_chunks_for_langchain.append({
                                "name": fc.name.replace('-', '_'),
                                "args": args_str,
                                "id": f"call_{fc.name.replace('-', '_')}_{os.urandom(4).hex()}", # Generate a unique ID for this call
                                "index": fc_idx # Index of the tool call in this chunk
                            })
                    
                    # Extract text content if no tool calls or if text coexists
                    if not tool_call_chunks_for_langchain: # Only extract text if no tool calls were processed for this chunk
                        try:
                            if candidate.content and candidate.content.parts:
                                chunk_text = "".join(part.text for part in candidate.content.parts if hasattr(part, "text"))
                        except ValueError as e: # Typically means content was blocked
                            logger.warning(f"Error extracting text from stream chunk: {e}. Candidate: {candidate}")
                            generation_info["error"] = f"Content extraction error: {e}"
                            # Potentially yield an error chunk here
                    
                    if candidate.finish_reason:
                        generation_info["finish_reason"] = candidate.finish_reason.name
                    
                    # Removed manual addition of token_usage to generation_info.
                    # Relying on AIMessageChunk.usage_metadata if populated by underlying SDK/Langchain.
                    # if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata: # Usually at the end of stream
                    #     generation_info["token_usage"] = {
                    #         "prompt_token_count": chunk.usage_metadata.prompt_token_count,
                    #         "candidates_token_count": chunk.usage_metadata.candidates_token_count,
                    #         "total_token_count": chunk.usage_metadata.total_token_count,
                    #     }
                    
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=chunk_text, 
                            tool_call_chunks=tool_call_chunks_for_langchain if tool_call_chunks_for_langchain else []
                            # AIMessageChunk itself might get usage_metadata populated by the SDK wrapper
                        ),
                        generation_info=generation_info if generation_info else None, # Now mainly for finish_reason
                    )
                    if run_manager and chunk_text:
                        await run_manager.on_llm_new_token(chunk_text)

            except google_exceptions.GoogleAPIError as e:
                logger.error(f"Google API error during streaming: {e}")
                if isinstance(e, google_exceptions.InvalidArgument):
                     error_message = f"Invalid argument to Gemini API (streaming): {e}"
                     yield ChatGenerationChunk(message=AIMessageChunk(content=error_message, additional_kwargs={"error": True, "type": "InvalidArgument"}))
                     return
                raise
            except Exception as e:
                logger.error(f"Unexpected error during Gemini streaming: {e}", exc_info=True)
                yield ChatGenerationChunk(message=AIMessageChunk(content=f"Unexpected error: {str(e)}", additional_kwargs={"error": True, "type": "Unexpected"}))
                return

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Synchronous wrapper for _agenerate
        # Using Langchain's recommended way if available, or simple asyncio.run
        # This is a common pattern for libraries that are async-first.
        try:
            # Check if an event loop is already running.
            loop = asyncio.get_running_loop()
            # If so, schedule the coroutine and wait for its result.
            # This is a simplified approach. More robust solutions might use
            # `asyncio.run_coroutine_threadsafe` if called from another thread,
            # or ensure the sync call doesn't block an outer async context.
            # For Langchain's typical use, this often suffices.
            if loop.is_running():
                 # This can be problematic if _generate is called from an async context that
                 # isn't expecting to be blocked. Langchain usually calls _agenerate directly then.
                 # A common pattern is to use a thread pool executor for true non-blocking sync calls from async.
                 # For now, simple blocking call:
                 future = asyncio.run_coroutine_threadsafe(self._agenerate(messages, stop, run_manager, **kwargs), loop)
                 return future.result() # Blocks until result
            else: # Should not happen if get_running_loop() succeeded without error
                 return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))
        except RuntimeError: # No event loop is running
            return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))


    @classmethod
    def _recursively_clean_schema(cls, schema_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(schema_dict, dict):
            return schema_dict

        # Keywords to remove from the current level of the schema
        keys_to_remove = ["title", "anyOf", "allOf", "oneOf", "not"] 
        for key_to_remove in keys_to_remove:
            if key_to_remove in schema_dict:
                del schema_dict[key_to_remove]
        
        # Remove 'default' from direct properties of the current schema_dict
        if "properties" in schema_dict and isinstance(schema_dict["properties"], dict):
            for _prop_name, prop_details_val in schema_dict["properties"].items():
                 if isinstance(prop_details_val, dict) and "default" in prop_details_val:
                    del prop_details_val["default"]

        # Recursively clean values that are dicts or lists of dicts
        for key in list(schema_dict.keys()): 
            value = schema_dict.get(key) # Use .get() in case a key was deleted by the loop above
            if isinstance(value, dict):
                cls._recursively_clean_schema(value) 
            elif isinstance(value, list):
                new_list = []
                for item_idx, item in enumerate(value): # Corrected variable name
                    if isinstance(item, dict):
                        new_list.append(cls._recursively_clean_schema(item))
                    else:
                        new_list.append(item) 
                schema_dict[key] = new_list # Update the list in the dictionary
            
        return schema_dict

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # Synchronous wrapper for _astream
        async_gen = self._astream(messages, stop, run_manager, **kwargs)
        
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Iterating an async generator from a sync method in a running loop is tricky.
                # This is a simplified blocking iterator.
                while True:
                    try:
                        future = asyncio.run_coroutine_threadsafe(async_gen.__anext__(), loop)
                        yield future.result()
                    except StopAsyncIteration:
                        break
            else: # Should not happen
                 # Fallback to asyncio.run for each item (less efficient)
                while True:
                    try:
                        yield asyncio.run(async_gen.__anext__())
                    except StopAsyncIteration:
                        break
        except RuntimeError: # No event loop running
            while True:
                try:
                    yield asyncio.run(async_gen.__anext__())
                except StopAsyncIteration:
                    break

    def _convert_tools(self, tools: Sequence[Union[Dict[str, Any], Type[PydanticV2BaseModel], Callable, BaseTool]]) -> List[Dict[str, Any]]:
        """
        Converts Langchain tools to the format expected by Gemini (list of FunctionDeclaration dicts).
        """
        if not tools:
            return []
        
        processed_tools = []
        for tool_spec in tools:
            function_declaration = None
            if isinstance(tool_spec, BaseTool):
                # For BaseTool, args_schema should be a Pydantic model.
                schema = tool_spec.args_schema.model_json_schema() if tool_spec.args_schema else {"type": "OBJECT", "properties": {}}
                schema = self._recursively_clean_schema(schema) # Apply recursive cleaning
                
                function_declaration = {
                    "name": tool_spec.name.replace('-', '_'),
                    "description": tool_spec.description,
                    "parameters": schema
                }
            elif isinstance(tool_spec, type) and issubclass(tool_spec, PydanticV2BaseModel):
                # Pydantic model directly
                schema = tool_spec.model_json_schema()
                schema = self._recursively_clean_schema(schema) # Apply recursive cleaning

                function_declaration = {
                    "name": tool_spec.__name__.replace('-', '_'), # Use class name as Pydantic schema title was removed
                    "description": schema.get("description", ""), # Description is usually fine
                    "parameters": schema
                }
            elif callable(tool_spec) and hasattr(tool_spec, "lc_name"): # Langchain @tool decorated
                # Use convert_to_openai_tool as a helper, then adapt its output
                openai_tool_format = convert_to_openai_tool(tool_spec)
                parameters_schema = openai_tool_format["function"]["parameters"]
                parameters_schema = self._recursively_clean_schema(parameters_schema) # Apply recursive cleaning
                function_declaration = {
                    "name": openai_tool_format["function"]["name"].replace('-', '_'),
                    "description": openai_tool_format["function"]["description"],
                    "parameters": parameters_schema
                }
            elif isinstance(tool_spec, dict) and "type" in tool_spec and tool_spec["type"] == "function": # Already OpenAI format
                 # Adapt OpenAI function tool format to Gemini FunctionDeclaration
                parameters_schema = tool_spec["function"]["parameters"]
                parameters_schema = self._recursively_clean_schema(parameters_schema) # Apply recursive cleaning
                function_declaration = {
                    "name": tool_spec["function"]["name"].replace('-', '_'),
                    "description": tool_spec["function"]["description"],
                    "parameters": parameters_schema
                }
            else:
                raise ValueError(f"Unsupported tool type: {type(tool_spec)}. Gemini requires FunctionDeclaration-like format.")
            
            if function_declaration:
                processed_tools.append(function_declaration)
        return processed_tools

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[PydanticV2BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[str, dict, Literal["any", "auto", "none"], bool]] = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]: # Adjusted LanguageModelInput to Any for broader compatibility
        """Bind tools and tool_choice to this chat model for Gemini.

        Args:
            tools: A list of tools to bind to the model.
            tool_choice: How to use tools. See Gemini API documentation.
                - "ANY", "AUTO": Model predicts a tool call or generates content.
                - "NONE": Model generates content, no tool calls.
                - dict (OpenAI-like): {"function_call": {"name": "tool_name"}} to force a specific tool.
                - str (tool name): Force a specific tool.
            kwargs: Other parameters to bind to the model.
        """
        # Determine how to get the current model's configuration based on Pydantic version.
        # Since BaseChatModel is Pydantic V1, GeminiChatModel will also behave as V1.
        excluded_from_dump = set() # For Pydantic V1, field-level exclude=True is primary. For V2, this can be used.

        if hasattr(self, "model_dump"):
            # Pydantic V2 path (should not be taken if BaseChatModel is V1)
            current_model_config = self.model_dump(exclude=excluded_from_dump, exclude_none=True)
        else:
            # Pydantic V1 path
            # For Pydantic V1, ensure private fields like _client are excluded if not already by Field(exclude=True)
            # The `exclude` parameter in `dict()` takes a set of field names.
            # Fields defined with `Field(..., exclude=True)` are automatically excluded by `dict()`.
            current_model_config = self.dict(exclude=excluded_from_dump, exclude_none=True)

        # Combine current config with any additional kwargs passed to bind_tools
        # kwargs passed here should override existing model config for the new instance
        bound_kwargs = {**current_model_config, **kwargs}

        # Prepare tools in the format Gemini expects (list of FunctionDeclarations)
        if tools:
            processed_gemini_tools = self._convert_tools(tools)
            
            gen_kwargs = bound_kwargs.get("generation_kwargs", {}).copy()
           
            gen_kwargs["tools"] = [{"function_declarations": processed_gemini_tools}] # Gemini nests this
            bound_kwargs["generation_kwargs"] = gen_kwargs
        else: # Ensure if no tools are passed, any existing tools in generation_kwargs are cleared
            gen_kwargs = bound_kwargs.get("generation_kwargs", {}).copy()
            if "tools" in gen_kwargs:
                del gen_kwargs["tools"]
            bound_kwargs["generation_kwargs"] = gen_kwargs


        # Handle tool_choice for Gemini
        # Gemini uses 'tool_config' for tool_choice.
        # tool_config: { function_calling_config: { mode: "ANY" | "NONE" | "AUTO", allowed_function_names: ["tool1"] } }
        if tool_choice is not None:
            gen_kwargs = bound_kwargs.get("generation_kwargs", {}).copy()
            tool_config_val = {} # This will be the value for "tool_config"
            fcc_val = {} # This will be the value for "function_calling_config"

            if isinstance(tool_choice, str):
                if tool_choice.upper() in ["ANY", "AUTO"]: # "AUTO" is default, "ANY" means model can choose
                    fcc_val = {"mode": "ANY"} # Gemini uses ANY for model choice
                elif tool_choice.upper() == "NONE":
                    fcc_val = {"mode": "NONE"}
                else: # Assumed to be a specific tool name to force (or allow)
                    fcc_val = {"mode": "ANY", "allowed_function_names": [tool_choice.replace('-', '_')]}
            elif isinstance(tool_choice, dict) and "function_call" in tool_choice and "name" in tool_choice["function_call"]:
                 # Adapting OpenAI-like forced tool choice to Gemini
                 tool_name = tool_choice["function_call"]["name"].replace('-', '_')
                 fcc_val = {"mode": "ANY", "allowed_function_names": [tool_name]}
            elif tool_choice is True: # equivalent to "any" or "auto"
                fcc_val = {"mode": "ANY"}
            elif tool_choice is False: # equivalent to "none"
                fcc_val = {"mode": "NONE"}
            
            if fcc_val: # If a valid function_calling_config was derived
                tool_config_val = {"function_calling_config": fcc_val}
                # Merge with existing tool_config if any, or set it
                existing_tool_config = gen_kwargs.get("tool_config", {})
                existing_tool_config.update(tool_config_val) # fcc_val is nested
                gen_kwargs["tool_config"] = existing_tool_config
                bound_kwargs["generation_kwargs"] = gen_kwargs
        else: # If tool_choice is explicitly None, ensure it's cleared from generation_kwargs if set
            gen_kwargs = bound_kwargs.get("generation_kwargs", {}).copy()
            if "tool_config" in gen_kwargs:
                # Clearing it relies on Gemini's default behavior (usually "AUTO" which is like "ANY")
                del gen_kwargs["tool_config"]
            bound_kwargs["generation_kwargs"] = gen_kwargs

        # Create a new instance of the model with the updated configuration
        return self.__class__(**bound_kwargs)


    @property
    def _llm_type(self) -> str:
        return "gemini-chat-model"
