"""Shared utility functions used in the project.

Functions:
    format_docs: Convert documents to an xml-formatted string.
    load_chat_model: Load a chat model from a model name.
"""

import logging # Added for logging
from typing import Optional, Dict, Any # Added Dict, Any
from dotenv import load_dotenv # Added for .env loading

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

# from custom_providers.gemini_chat_model import GeminiChatModel # Old model
from custom_providers.production_gemini_chat_model import ProductionGeminiChatModel # New production-ready model

logger = logging.getLogger(__name__) # Added logger instance

load_dotenv() # Load environment variables from .env file

def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


import os # Ensure os is imported for getenv
from retrieval_graph.configuration import AgentConfiguration # Corrected import path

def load_chat_model(fully_specified_name: str, agent_config: AgentConfiguration) -> BaseChatModel:
    """Load a chat model from a fully specified name, configured by AgentConfiguration.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
        agent_config (AgentConfiguration): The agent's configuration object.
    """
    if "/" in fully_specified_name:
        provider, model_name_part = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = "" 
        model_name_part = fully_specified_name

    if provider == "gemini_custom":
        persona = agent_config.agent_persona
        logger.info(f"Loading Production Gemini chat model for persona: {persona}. Specified model part: '{model_name_part}'.")

        target_config_id: Optional[str] = None
        target_metadata: Dict[str, Any] = {"persona": persona} # Start with persona in metadata

        if persona == "المستشار":
            target_config_id = os.getenv("ALMUSTASHAR_PORTKEY_CONFIG_ID")
            target_metadata["model_version"] = "gemini-2.0-flash" # Or actual model used by this config
            logger.info(f"Persona 'المستشار': Using Portkey Config ID: {target_config_id}")
        elif persona == "الذكي":
            target_config_id = os.getenv("ALZAKI_PORTKEY_CONFIG_ID")
            target_metadata["model_version"] = "gemini-1.5-flash" # Or actual model used by this config
            logger.info(f"Persona 'الذكي': Using Portkey Config ID: {target_config_id}")
        elif persona == "الغبي":
            target_config_id = os.getenv("ALZAKI_PORTKEY_CONFIG_ID") # Use same config as "الذكي"
            target_metadata["model_version"] = "gemini-1.5-flash" # Ensure consistency with "الذكي"
            logger.info(f"Persona 'الغبي': Using Portkey Config ID for 'الذكي': {target_config_id}")
        else: # Default or unknown persona
            logger.warning(f"Unknown or default persona '{persona}'. Using default fallback Portkey Config ID.")
            target_config_id = os.getenv("FALLBACK_CONFIG_ID") # Default fallback
            target_metadata["persona"] = "default_fallback" # Clarify metadata

        if not target_config_id:
            # This is a critical issue if a specific persona config ID was expected but not found
            logger.error(f"Portkey Config ID for persona '{persona}' is not set in environment variables. FALLBACK_CONFIG_ID will be attempted if available, or init will fail.")
            # Rely on ProductionGeminiChatModel's internal fallback_config_id if target_config_id remains None
            # or raise an error if strict persona config is required. For now, let it try to use its internal fallback.

        # Instantiate ProductionGeminiChatModel with the determined target_config_id and metadata
        # It will use its own os.getenv("PORTKEY_API_KEY") internally.
        # Other params like fallback_config_id, loadbalance_config_id are still available in ProductionGeminiChatModel
        # if target_config_id is None (e.g. for a truly default/unspecified persona).
        return ProductionGeminiChatModel(
            target_config_id=target_config_id,
            target_metadata=target_metadata
            # model_override could be passed here if model_name_part was intended for that,
            # but Portkey's advice is to manage model via Config ID.
        )
    else:
        # Fallback to existing LangChain init_chat_model for other providers
        logger.info(f"Loading chat model '{model_name_part}' using Langchain's init_chat_model for provider '{provider}'.")
        return init_chat_model(model_name_part, model_provider=provider)


def get_knowledge_supabase_client():
    """Initializes and returns a Supabase client for the knowledge database.

    Reads KNOWLEDGE_SUPABASE_URL and KNOWLEDGE_SUPABASE_KEY from environment variables.

    Returns:
        supabase.Client: An initialized Supabase client.

    Raises:
        ValueError: If KNOWLEDGE_SUPABASE_URL or KNOWLEDGE_SUPABASE_KEY are not set.
    """
    from supabase import create_client, Client as SupabaseClient # Local import
    from .configuration import BaseConfiguration # Local import
    import os

    config = BaseConfiguration() # To potentially load from .env if not already loaded by main app
    
    url = os.getenv("KNOWLEDGE_SUPABASE_URL", config.knowledge_supabase_url)
    key = os.getenv("KNOWLEDGE_SUPABASE_KEY", config.knowledge_supabase_key)

    if not url or not key:
        error_msg = "KNOWLEDGE_SUPABASE_URL and KNOWLEDGE_SUPABASE_KEY must be set in environment variables or BaseConfiguration."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Initializing Supabase client for knowledge database.")
    try:
        supabase_client: SupabaseClient = create_client(url, key)
        return supabase_client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        raise
