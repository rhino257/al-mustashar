"""Custom nodes for generating embeddings."""

import logging
import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from retrieval_graph.state import AgentState
from shared.configuration import BaseConfiguration
from shared.retrieval import make_text_encoder
from custom_providers.matryoshka_arabic_embeddings import MatryoshkaArabicEmbeddings
from custom_providers.bge_m3_law_embeddings import BgeM3LawEmbeddings

logger = logging.getLogger(__name__)

async def generate_multi_embeddings_node(state: AgentState, *, config: RunnableConfig) -> Dict[str, Any]:
    """
    Generates multiple embeddings for the query, specifically for Matryoshka and BGE-M3-Law models.
    This node is intended for the 'المستشار' persona's meta-hybrid retrieval strategy.
    """
    node_name = "generate_multi_embeddings_node"
    logger.info(f"--- Executing {node_name} ---")
    
    try:
        text_to_embed = state.text_for_embedding
        if not text_to_embed:
            logger.error(f"[{node_name}] No text available for embedding (state.text_for_embedding is empty).")
            return {
                "matryoshka_embedding": None,
                "bge_embedding": None,
                "error_message": "No text available for embedding.",
                "error_node": node_name
            }

        logger.info(f"[{node_name}] Text to embed: '{text_to_embed}'")

        # Initialize embedders
        matryoshka_embedder = MatryoshkaArabicEmbeddings(normalize_text=True)
        bge_m3_law_embedder = BgeM3LawEmbeddings(normalize_text=True)

        # Generate embeddings in parallel
        logger.info(f"[{node_name}] Generating Matryoshka and BGE embeddings in parallel.")
        matryoshka_task = matryoshka_embedder.aembed_query(text_to_embed)
        bge_task = bge_m3_law_embedder.aembed_query(text_to_embed)
        
        results = await asyncio.gather(matryoshka_task, bge_task)
        
        matryoshka_embedding = results[0]
        bge_embedding = results[1]
        
        logger.info(f"[{node_name}] Successfully generated both embeddings.")

        return {
            "query_embedding": matryoshka_embedding,
            "bge_embedding": bge_embedding,
        }

    except Exception as e:
        logger.error(f"[{node_name}] An unexpected error occurred: {e}", exc_info=True)
        return {
            "matryoshka_embedding": None,
            "bge_embedding": None,
            "error_message": f"Unexpected error in node: {str(e)}",
            "error_node": node_name
        }
