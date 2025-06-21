import logging
import os
import asyncio # Added for asyncio.to_thread
from typing import Any, Dict, List, Optional
from supabase import create_client, Client as SupabaseClient # Ensure Client is imported for type hinting
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from retrieval_graph.state import AgentState
from retrieval_graph.configuration import AgentConfiguration # For accessing potential Supabase creds

logger = logging.getLogger(__name__)

def escape_like_wildcards(text: str, escape_char: str = '\\\\') -> str:
    """Escapes ILIKE wildcard characters (% and _) in a string."""
    if not text:
        return text
    text = text.replace(escape_char, escape_char * 2) # Escape the escape character itself
    text = text.replace('%', f'{escape_char}%')
    text = text.replace('_', f'{escape_char}_')
    return text

# KNOWLEDGE_DB_PROJECT_ID is not directly used if we connect via URL and KEY,
# but good to keep for reference or if other Supabase interactions need it.
# KNOWLEDGE_DB_PROJECT_ID = "lobmeaosjgmxeyjxlvhf" 
# MCP_SERVER_NAME is no longer needed for this direct approach.
# MCP_SERVER_NAME = "github.com/supabase-community/supabase-mcp" 

async def execute_direct_supabase_lookup_node(state: AgentState, *, config: RunnableConfig) -> Dict[str, Any]:
    """
    Performs a direct lookup for a specific article using Supabase client,
    constructs the query, executes it, and processes the result into documents.
    """
    node_name = "execute_direct_supabase_lookup_node"
    logger.info(f"--- Executing {node_name} ---")
    query_analysis = state.query_analysis_result
    processed_documents: List[Document] = []
    
    default_error_return = {
        "retrieved_documents": [], # Ensure this key is always present
        "direct_lookup_mcp_args": None, # Clear this as it's no longer used
        "direct_lookup_mcp_response": None, # Clear this as it's no longer used
        "error_message": f"{node_name}: Could not complete direct lookup.",
        "error_node": node_name
    }

    if not (query_analysis and query_analysis.article_number):
        logger.warning(f"[{node_name}] Missing article number in query_analysis_result. Cannot perform lookup.")
        default_error_return["error_message"] = f"{node_name}: Missing article number for direct lookup."
        return default_error_return

    try:
        article_num_int = int(query_analysis.article_number)
    except ValueError:
        logger.warning(f"[{node_name}] Invalid article number format: {query_analysis.article_number}")
        default_error_return["error_message"] = f"{node_name}: Invalid article number format '{query_analysis.article_number}'."
        return default_error_return

    law_name_str = query_analysis.law_name
    law_name_to_search = law_name_str.strip() if law_name_str else None # Strip whitespace

    if law_name_to_search == "قانون العمل اليمني": # Specific adjustment based on previous logic
        law_name_to_search = "قانون العمل"
        logger.info(f"[{node_name}] Adjusted law name for search from '{law_name_str}' to '{law_name_to_search}'.")
    elif law_name_to_search: # law_name_to_search is already stripped here
        logger.info(f"[{node_name}] Using extracted and stripped law name for search: '{law_name_to_search}'")
    
    # Escape ILIKE wildcards in the final search term
    if law_name_to_search:
        law_name_to_search_escaped = escape_like_wildcards(law_name_to_search)
        if law_name_to_search_escaped != law_name_to_search:
            logger.info(f"[{node_name}] Escaped law name for ILIKE: '{law_name_to_search_escaped}' from '{law_name_to_search}'")
        law_name_to_search = law_name_to_search_escaped # Use the escaped version for the query

    logger.info(f"[{node_name}] Preparing direct lookup for article: {article_num_int}" + (f" of law: {law_name_to_search}" if law_name_to_search else ""))

    # Initialize Supabase client
    app_config = AgentConfiguration.from_runnable_config(config)
    # Prefer credentials from AgentConfiguration if available, else fallback to env vars
    supabase_url = getattr(app_config, 'knowledge_base_supabase_url', os.getenv("USERS_SUPABASE_URL"))
    supabase_key = getattr(app_config, 'knowledge_base_supabase_key', os.getenv("SUPABASE_SERVICE_ROLE_KEY"))

    if not supabase_url or not supabase_key:
        logger.error(f"[{node_name}] Supabase URL or Key not configured.")
        default_error_return["error_message"] = f"{node_name}: Supabase credentials not configured."
        return default_error_return

    try:
        supabase_client: SupabaseClient = create_client(supabase_url, supabase_key)
        
        query_builder = supabase_client.table('law_articles').select('*, law!inner(law_name, law_year)' if law_name_to_search else '*, law(law_name, law_year)')
        query_builder = query_builder.eq('article_number', article_num_int)

        if law_name_to_search:
            # Ensure 'law.law_name' is the correct path for ilike with joined table.
            # If 'law' is a foreign table, the client handles the join syntax.
            # The select 'law!inner(law_name, law_year)' implies 'law' is the relationship name.
            query_builder = query_builder.ilike('law.law_name', f'%{law_name_to_search}%')
        
        query_builder = query_builder.limit(1)
        
        logger.debug(f"[{node_name}] Executing Supabase query...")
        # Assuming query_builder.execute() might be synchronous based on the TypeError
        response = await asyncio.to_thread(query_builder.execute)
        # If using supabase-py v2 with a properly initialized async client,
        # and execute() is indeed async, then `await query_builder.execute()` would be correct.
        # The TypeError suggests it's not awaitable as called.

        if response.data:
            logger.info(f"[{node_name}] Direct lookup successful. Found {len(response.data)} item(s).")
            item = response.data[0]
            
            # Replicate logic from LawArticlesAdapter to flatten 'law' object if present
            law_info = item.pop('law', None) # Remove 'law' key, get its value
            if isinstance(law_info, dict):
                item['law_name'] = law_info.get('law_name')
                item['law_year'] = law_info.get('law_year')
            
            doc_content = item.get("processed_text") or item.get("article_text", "")
            metadata = {
                "source": "Supabase Direct Lookup", # Updated source
                "article_id": str(item.get("article_id")) if item.get("article_id") else None,
                "law_name": item.get("law_name"),
                "law_year": str(item.get("law_year")) if item.get("law_year") else None,
                "article_number": str(item.get("article_number")) if item.get("article_number") else None,
            }
            metadata = {k: v for k, v in metadata.items() if v is not None}
            processed_documents.append(Document(page_content=doc_content, metadata=metadata))
            logger.info(f"[{node_name}] Processed direct lookup result into document: article_id {metadata.get('article_id')}")
        else:
            logger.info(f"[{node_name}] Direct lookup returned no data (document not found).")

    except Exception as e:
        logger.error(f"[{node_name}] Error during Supabase query execution or processing: {e}", exc_info=True)
        default_error_return["error_message"] = f"{node_name}: Supabase query failed: {str(e)}"
        # Return default_error_return which already has retrieved_documents as empty list
        return default_error_return

    return {
        "retrieved_documents": processed_documents,
        "direct_lookup_mcp_args": None, # Ensure these are cleared
        "direct_lookup_mcp_response": None,
        "error_message": None,
        "error_node": None
    }

# The old process_direct_lookup_result_node is no longer needed as its logic
# has been incorporated into execute_direct_supabase_lookup_node.
# It can be safely removed from this file and from the graph definition.
