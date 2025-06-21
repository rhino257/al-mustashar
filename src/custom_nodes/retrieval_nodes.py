"""Custom nodes for retrieval processes in the RAG agent."""

import logging
import os
import time 
import asyncio # Ensure asyncio is imported for to_thread
import json # Ensure json is imported
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv, find_dotenv, dotenv_values 

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage 
from langchain_core.runnables import RunnableConfig
from retrieval_graph.state import AgentState
from retrieval_graph.configuration import AgentConfiguration # For reranker_node config
from shared.configuration import BaseConfiguration # For other nodes
from custom_providers.openai_custom_embeddings import OpenAICustomEmbeddings
# Import new embedding providers
from custom_providers.matryoshka_arabic_embeddings import MatryoshkaArabicEmbeddings
from custom_providers.bge_m3_law_embeddings import BgeM3LawEmbeddings
from shared.retrieval import make_text_encoder

from shared.utils import get_knowledge_supabase_client 
from .law_articles_adapter import LawArticlesAdapter

# Import Portkey SDK for the reranker_node
from portkey_ai import Portkey

logger = logging.getLogger(__name__)


async def generate_query_embedding_node(state: AgentState, *, config: RunnableConfig) -> Dict[str, Any]:
    """
    Generates a query embedding using the OpenAICustomEmbeddings provider.
    NOTE: This node might need future updates if the primary embedding for 'الذكي'
    is Matryoshka and app_config.embedding_model needs to reflect that for make_text_encoder.
    """
    node_name = "generate_query_embedding_node"
    logger.info(f"--- Executing {node_name} ---")
    start_time = time.perf_counter()
    
    try:
        app_config = BaseConfiguration.from_runnable_config(config)
        
        text_to_embed = getattr(state, 'text_for_embedding', None)
        
        if not text_to_embed:
            logger.warning(f"[{node_name}] No text to embed (text_for_embedding is empty or None).")
            current_messages = getattr(state, 'messages', [])
            if current_messages and isinstance(current_messages[-1], HumanMessage):
                logger.info(f"[{node_name}] Falling back to user_query for embedding.")
                text_to_embed = current_messages[-1].content
            
            if not text_to_embed: 
                logger.error(f"[{node_name}] No text available for embedding.")
                return {"query_embedding": None, "error_message": "No text available for embedding.", "error_node": node_name}

        logger.info(f"[{node_name}] Text to embed: '{text_to_embed}'")

        try:
            embedding_provider = make_text_encoder(app_config.embedding_model)
            logger.info(f"[{node_name}] Successfully loaded embedding provider: {app_config.embedding_model} of type {type(embedding_provider)}")
        except Exception as e:
            logger.error(f"[{node_name}] Error initializing embedding provider: {e}", exc_info=True)
            return {"query_embedding": None, "error_message": f"Failed to initialize embedding provider: {str(e)}", "error_node": node_name}

        try:
            logger.info(f"[{node_name}] Generating query embedding...")
            
            portkey_metadata_for_embedding: Optional[Dict[str, Any]] = None
            user_id_val = getattr(state, "user_id", None)
            if user_id_val:
                portkey_metadata_for_embedding = {"user_id": str(user_id_val), "_user": str(user_id_val)}
            
            if isinstance(embedding_provider, OpenAICustomEmbeddings):
                query_embedding = await embedding_provider.aembed_query(text_to_embed, portkey_metadata_override=portkey_metadata_for_embedding)
            else:
                query_embedding = await embedding_provider.aembed_query(text_to_embed)
                
            logger.info(f"[{node_name}] Successfully generated query embedding.")
            return {"query_embedding": query_embedding}
        except Exception as e:
            logger.error(f"[{node_name}] Error generating query embedding: {e}", exc_info=True)
            return {"query_embedding": None, "error_message": f"Failed to generate query embedding: {str(e)}", "error_node": node_name}

    except Exception as e:
        logger.error(f"[{node_name}] An unexpected error occurred: {e}", exc_info=True)
        return {"query_embedding": None, "error_message": f"Unexpected error in node: {str(e)}", "error_node": node_name}
    finally:
        end_time = time.perf_counter()
        logger.info(f"--- {node_name} execution time: {end_time - start_time:.4f} seconds ---")


async def supabase_hybrid_retriever_node(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, Any]:
    node_name = "supabase_hybrid_retriever_node"
    logger.info(f"--- Executing {node_name} ---")
    start_time = time.perf_counter()

    try:
        app_config = AgentConfiguration.from_runnable_config(config)
        agent_persona = app_config.agent_persona
        
        # Access state attributes directly using getattr for safety
        query_embedding: Optional[List[float]] = getattr(state, "query_embedding", None)
        search_keywords_list: Optional[List[str]] = getattr(state, "search_keywords", [])
        keyword_query: Optional[str] = " ".join(search_keywords_list) if search_keywords_list else None

        if query_embedding is None and keyword_query is None:
            logger.error(f"[{node_name}] Both primary query embedding and keywords are missing for persona '{agent_persona}'.")
            return {"retrieved_documents": [], "error_message": "Missing primary embedding and keywords for search.", "error_node": node_name}

        try:
            supabase_client = get_knowledge_supabase_client()
            adapter = LawArticlesAdapter(db_client=supabase_client)
        except Exception as e:
            logger.error(f"[{node_name}] Error initializing Supabase client or adapter: {e}", exc_info=True)
            return {"retrieved_documents": [], "error_message": f"Failed to initialize Supabase client/adapter: {str(e)}", "error_node": node_name}

        final_retrieved_documents: List[Document] = []
        
        if agent_persona == "المستشار":
            logger.info(f"[{node_name}] Processing 'المستشار' persona retrieval strategy (Meta-Hybrid).")
            
            matryoshka_embedding = getattr(state, "query_embedding", None)
            bge_embedding = getattr(state, "bge_embedding", None)

            if not matryoshka_embedding or not bge_embedding:
                logger.error(f"[{node_name}] Missing one or both embeddings for 'المستشار' meta-hybrid strategy.")
                return {"retrieved_documents": [], "error_message": "Missing embeddings for meta-hybrid strategy.", "error_node": node_name}

            retrieval_limit = app_config.almustashar_synthesis_limit
            
            final_retrieved_documents = await adapter.meta_hybrid_search(
                matryoshka_embedding=matryoshka_embedding,
                bge_embedding=bge_embedding,
                keyword_query=keyword_query,
                filters={'threshold': app_config.MATRYOSHKA_RPC_SEMANTIC_THRESHOLD, 'bge_threshold': app_config.BGE_RPC_SEMANTIC_THRESHOLD},
                limit=retrieval_limit,
                rrf_k_val=app_config.DEFAULT_RRF_K_VAL,
                meta_rrf_k_val=app_config.DEFAULT_RRF_K_VAL
            )
            logger.info(f"[{node_name}] Meta-Hybrid search returned {len(final_retrieved_documents)} documents for 'المستشار'.")

        elif agent_persona == "الذكي":
            logger.info(f"[{node_name}] Processing 'الذكي' persona retrieval strategy (Simplified: Matryoshka Only).")
            matryoshka_docs: List[Document] = []

            if query_embedding: 
                logger.info(f"[{node_name}] Calling Matryoshka Hybrid RPC for 'الذكي' persona.")
                retrieval_limit = app_config.alzaki_synthesis_limit
                logger.info(f"[{node_name}] Using retrieval limit for Matryoshka (الذكي): {retrieval_limit}")

                matryoshka_docs = await adapter.hybrid_search_articles_smart_matryoshka(
                    query_embedding=query_embedding,
                    keyword_query=keyword_query,
                    filters={'threshold': app_config.MATRYOSHKA_RPC_SEMANTIC_THRESHOLD},
                    limit=retrieval_limit, 
                    rrf_k_val=app_config.DEFAULT_RRF_K_VAL
                )
                logger.info(f"[{node_name}] Matryoshka RPC returned {len(matryoshka_docs)} documents for 'الذكي'.")
                final_retrieved_documents = matryoshka_docs
            else:
                logger.warning(f"[{node_name}] Matryoshka (primary) query embedding missing for 'الذكي' persona. No documents retrieved.")
                final_retrieved_documents = []
        
        elif agent_persona == "الغبي":
            logger.info(f"[{node_name}] Processing 'الغبي' persona retrieval strategy (Matryoshka Semantic V2 Only).")
            if query_embedding:
                logger.info(f"[{node_name}] Calling Matryoshka Semantic V2 RPC for 'الغبي' persona.")
                semantic_threshold = app_config.MATRYOSHKA_RPC_SEMANTIC_THRESHOLD 
                retrieval_limit = app_config.alghabi_synthesis_limit
                logger.info(f"[{node_name}] Using retrieval limit for Matryoshka Semantic (الغبي): {retrieval_limit}, Threshold: {semantic_threshold}")

                final_retrieved_documents = await adapter.semantic_search_articles_matryoshka_v2(
                    query_embedding=query_embedding,
                    filters={'threshold': semantic_threshold},
                    limit=retrieval_limit
                )
                logger.info(f"[{node_name}] Matryoshka Semantic V2 RPC returned {len(final_retrieved_documents)} documents for 'الغبي'.")
            else:
                logger.warning(f"[{node_name}] Matryoshka query embedding missing for 'الغبي' persona. No documents retrieved.")
                final_retrieved_documents = []
        
        else: 
            logger.warning(f"[{node_name}] Persona '{agent_persona}' not recognized. Returning no documents.")
            final_retrieved_documents = [] 

        processed_docs: List[Document] = []
        for doc in final_retrieved_documents: 
            new_meta = doc.metadata.copy()
            if 'meta_rrf_score' in new_meta: new_meta['score'] = new_meta['meta_rrf_score']
            elif 'rrf_score' in new_meta: new_meta['score'] = new_meta['rrf_score']
            elif 'similarity' in new_meta: new_meta['score'] = new_meta['similarity'] # For semantic search results
            elif 'score' not in new_meta: new_meta['score'] = 0.0 
            processed_docs.append(Document(page_content=doc.page_content, metadata=new_meta))
        
        processed_docs.sort(key=lambda d: d.metadata.get('score', 0.0), reverse=True)
        
        overall_limit = app_config.hybrid_search_match_count 
        # Apply persona-specific synthesis limits if they are more restrictive than the general hybrid_search_match_count
        if agent_persona == "المستشار":
            overall_limit = min(overall_limit, app_config.almustashar_synthesis_limit)
        elif agent_persona == "الذكي":
            overall_limit = min(overall_limit, app_config.alzaki_synthesis_limit)
        elif agent_persona == "الغبي":
            overall_limit = min(overall_limit, app_config.alghabi_synthesis_limit)
            
        final_limited_documents = processed_docs[:overall_limit]
        
        logger.info(f"[{node_name}] Final documents after processing & limit ({overall_limit}): {len(final_limited_documents)} for persona '{agent_persona}'.")

        if final_limited_documents:
            logger.info(f"--- [{node_name}] Final Retrieved Documents for Persona '{agent_persona}' (Top {len(final_limited_documents)}) ---")
            for i, doc_to_log in enumerate(final_limited_documents):
                logger.info(f"Doc [{i+1}/{len(final_limited_documents)}]: {doc_to_log.page_content}")
                logger.info(f"Metadata: {doc_to_log.metadata}")
            logger.info(f"--- [{node_name}] End of Final Retrieved Documents ---")

        return {"retrieved_documents": final_limited_documents}

    except Exception as e:
        logger.error(f"[{node_name}] An unexpected error occurred: {e}", exc_info=True)
        return {"retrieved_documents": [], "error_message": f"Unexpected error in Supabase hybrid retriever: {str(e)}", "error_node": node_name}
    finally:
        end_time = time.perf_counter()
        logger.info(f"--- {node_name} execution time: {end_time - start_time:.4f} seconds ---")
