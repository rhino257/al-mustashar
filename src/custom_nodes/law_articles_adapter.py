"""Adapter for interacting with Law Articles data source, specifically for Supabase hybrid search."""

import logging
import traceback
import asyncio
from typing import List, Optional, Any, Dict

from supabase import Client as SupabaseClient
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Assuming a default embedding dimension if not provided, e.g., for OpenAI's ada-002
DEFAULT_EMBEDDING_DIM = 1536

def map_db_result_to_document(db_result: Dict[str, Any], item_type: str) -> Document:
    """Maps a database result from a hybrid search RPC to a LangChain Document."""
    # For articles, content is in 'processed_text'.
    # For comments, the RPC hybrid_search_comments_rrf aliases COALESCE(c.processed_text, c.content) AS content.
    page_content = db_result.get("processed_text") if item_type == "article" else db_result.get("content", "")
    if page_content is None: # Ensure page_content is not None
        page_content = ""

    # Prepare metadata, including all fields from the result and prioritizing specific ones
    metadata = {k: v for k, v in db_result.items() if v is not None} # Exclude None values from db_result
    metadata["source"] = f"supabase_hybrid_search_{item_type}"
    metadata["item_type"] = item_type
    
    # Ensure common fields are present, even if None, and add item-specific fields
    if item_type == "article":
        metadata.setdefault("law_name", db_result.get("law_name"))
        # metadata.setdefault("law_year", db_result.get("law_year")) # law_year removed from RPC output
        metadata.setdefault("article_number", db_result.get("article_number"))
    elif item_type == "comment":
        metadata.setdefault("title", db_result.get("title"))
        metadata.setdefault("author", db_result.get("author"))
        # created_at is already included if present in db_result and not None
    
    metadata.setdefault("score", db_result.get("rrf_score", db_result.get("score"))) # Prefer rrf_score

    # Clean metadata again to remove any keys that ended up with None from setdefault if db_result.get was None
    metadata = {k: v for k, v in metadata.items() if v is not None}

    return Document(page_content=page_content, metadata=metadata)

def _client_side_reciprocal_rank_fusion_adapted(
    list_of_ranked_doc_lists: List[List[Document]], 
    k: int = 60,
    id_field: str = "article_id" # or a more generic unique ID field in metadata
) -> List[Document]:
    """
    Performs client-side Reciprocal Rank Fusion on lists of LangChain Document objects.
    Adds 'meta_rrf_score' to the metadata of the fused documents and also sets 'score'.
    """
    fused_scores: Dict[str, float] = {} 
    doc_map: Dict[str, Document] = {} 

    for ranked_doc_list in list_of_ranked_doc_lists:
        if not ranked_doc_list:
            continue
        for rank, doc in enumerate(ranked_doc_list, 1): 
            doc_id = doc.metadata.get(id_field)
            if not doc_id:
                logger.warning(f"Document missing '{id_field}' in metadata, cannot fuse: {doc.page_content[:50]}...")
                continue
            
            doc_id = str(doc_id) 

            if doc_id not in doc_map:
                doc_map[doc_id] = doc 
            else:
                current_score = doc.metadata.get('score', 0.0)
                existing_score = doc_map[doc_id].metadata.get('score', 0.0)
                if isinstance(current_score, (int, float)) and isinstance(existing_score, (int, float)) and current_score > existing_score:
                    doc_map[doc_id] = doc
                elif len(doc.page_content) > len(doc_map[doc_id].page_content): 
                     doc_map[doc_id] = doc

            score_contribution = 1.0 / (k + rank)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + score_contribution
            
    reranked_results_with_scores = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    
    final_fused_docs: List[Document] = []
    for doc_id, score in reranked_results_with_scores:
        doc_detail = doc_map.get(doc_id)
        if doc_detail:
            new_metadata = doc_detail.metadata.copy()
            new_metadata['meta_rrf_score'] = score 
            new_metadata['score'] = score 
            final_fused_docs.append(Document(page_content=doc_detail.page_content, metadata=new_metadata))
            
    return final_fused_docs

class LawArticlesAdapter:
    """
    Adapter for performing hybrid search on Law Articles stored in Supabase
    via an RPC function.
    """
    def __init__(self, db_client: SupabaseClient):
        self.db = db_client

    async def hybrid_search_articles(
        self,
        query_embedding: List[float], 
        keyword_query: Optional[str],
        filters: Dict[str, Any], 
        limit: int,
        rrf_k_val: int = 60,
        filter_law_categories: Optional[List[str]] = None,
        filter_article_tags: Optional[List[str]] = None,
        filter_sharia_influence: Optional[bool] = None,
        metadata_filter_logic: str = 'AND', 
        metadata_score_weight: float = 0.3, 
        **kwargs 
    ) -> List[Document]:
        """
        Performs hybrid search for LAW ARTICLES using the 'hybrid_search_articles_with_metadata_filters_rrf' RPC function.
        Includes configurable RRF K value, new metadata filters, and logic control.
        This is the reverted 12-parameter version.
        """
        try:
            active_query_embedding = query_embedding if query_embedding is not None else [0.0] * DEFAULT_EMBEDDING_DIM
            
            rpc_params = {
                'p_query_embedding': active_query_embedding,
                'p_keyword_query': keyword_query,
                'p_match_threshold': filters.get('threshold', 0.5),
                'p_match_count': limit,
                'p_filter_article_number': filters.get('article_number'),
                'p_filter_sharia_influence': filter_sharia_influence,
                'p_filter_law_categories': filter_law_categories if filter_law_categories else [],
                'p_filter_article_tags': filter_article_tags if filter_article_tags else [],
                'p_rrf_k_val': rrf_k_val,
                'p_metadata_filter_logic': metadata_filter_logic,
                'p_metadata_score_weight': metadata_score_weight, 
            }
            rpc_params['p_effective_limit_multiplier'] = kwargs.get('p_effective_limit_multiplier', 15) 

            rpc_params = {k: v for k, v in rpc_params.items() if v is not None or k in ['p_filter_law_categories', 'p_filter_article_tags']}

            function_name = 'hybrid_search_articles_with_metadata_filters_rrf'
            logger.info(f"Calling Supabase RPC for articles (REVERTED 12-param version): {function_name} with params: {rpc_params}")
            response = self.db.rpc(function_name, rpc_params).execute()

            if response.data:
                logger.info(f"Received {len(response.data)} results from article hybrid search.")
                return [map_db_result_to_document(item, "article") for item in response.data if isinstance(item, dict)]
            else:
                logger.info("No results from article hybrid search.")
                return []
        except Exception as e:
            logger.error(f"Error during LawArticlesAdapter article hybrid search: {e}\n{traceback.format_exc()}")
            return []

    async def hybrid_search_comments(
        self,
        query_embedding: List[float],
        keyword_query: Optional[str],
        filters: Dict[str, Any], 
        limit: int,
        rrf_k_val: int = 60,
        filter_tags: Optional[List[str]] = None,
        metadata_filter_logic: str = 'AND',
        metadata_score_weight: float = 0.1, 
        **kwargs 
    ) -> List[Document]:
        """
        Performs hybrid search for COMMENTS using the 'hybrid_search_comments_with_tag_filters_rrf' RPC function.
        Includes configurable RRF K value and new metadata filters for tags.
        """
        try:
            active_query_embedding = query_embedding if query_embedding is not None else [0.0] * DEFAULT_EMBEDDING_DIM
            
            rpc_params = {
                'p_query_embedding': active_query_embedding,
                'p_keyword_query': keyword_query,
                'p_match_threshold': filters.get('threshold', 0.5),
                'p_match_count': limit,
                'p_rrf_k_val': rrf_k_val,
                'p_filter_tags': filter_tags if filter_tags else [], 
                'p_metadata_filter_logic': metadata_filter_logic,
                'p_metadata_score_weight': metadata_score_weight
            }

            rpc_params = {
                k: v for k, v in rpc_params.items() 
                if v is not None or (k == 'p_filter_tags' and isinstance(v, list))
            }

            function_name = 'hybrid_search_comments_with_tag_filters_rrf' 
            logger.info(f"Calling Supabase RPC for comments: {function_name} with params: {rpc_params}")
            response = self.db.rpc(function_name, rpc_params).execute()

            if response.data:
                logger.info(f"Received {len(response.data)} results from comment hybrid search (with filters).")
                return [map_db_result_to_document(item, "comment") for item in response.data if isinstance(item, dict)]
            else:
                logger.info("No results from comment hybrid search (with filters).")
                return []
        except Exception as e:
            logger.error(f"Error during LawArticlesAdapter comment hybrid search (with filters): {e}\\n{traceback.format_exc()}")
            return []

    async def hybrid_search_articles_smart(
        self,
        query_embedding: List[float],
        keyword_query: Optional[str],
        filters: Dict[str, Any], 
        limit: int, 
        rrf_k_val: int = 60,
        **kwargs 
    ) -> List[Document]:
        """
        Performs hybrid search for LAW ARTICLES using the 'hybrid_search_articles_smart_rrf' RPC function.
        This version is for the 'الذكي' (Smart) persona and does not include metadata in RRF.
        """
        try:
            active_query_embedding = query_embedding if query_embedding is not None else [0.0] * DEFAULT_EMBEDDING_DIM
            
            rpc_params = {
                'p_query_embedding': active_query_embedding,
                'p_keyword_query': keyword_query, 
                'p_match_threshold': filters.get('threshold', 0.5),
                'p_match_count': limit,
                'p_rrf_k_val': rrf_k_val
            }
            if keyword_query is None:
                rpc_params.pop('p_keyword_query', None)

            function_name = 'hybrid_search_articles_smart_rrf'
            logger.info(f"Calling Supabase RPC for 'الذكي' articles: {function_name} with params: {rpc_params}")
            response = self.db.rpc(function_name, rpc_params).execute()

            if response.data:
                logger.info(f"Received {len(response.data)} results from '{function_name}'.")
                return [map_db_result_to_document(item, "article") for item in response.data if isinstance(item, dict)]
            else:
                logger.info(f"No results from '{function_name}'.")
                return []
        except Exception as e:
            logger.error(f"Error during LawArticlesAdapter '{function_name}' search: {e}\n{traceback.format_exc()}")
            return []

    async def hybrid_search_articles_smart_matryoshka(
        self,
        query_embedding: List[float], # Expects Matryoshka embedding
        keyword_query: Optional[str],
        filters: Dict[str, Any], 
        limit: int, 
        rrf_k_val: int = 60,
        **kwargs 
    ) -> List[Document]:
        """
        Performs hybrid search for LAW ARTICLES using the 'hybrid_search_articles_smart_rrf_matryoshka' RPC function.
        Tailored for Matryoshka embeddings.
        """
        try:
            active_query_embedding = query_embedding if query_embedding is not None else [0.0] * 768 # Matryoshka dim
            
            rpc_params = {
                'p_query_embedding': active_query_embedding,
                'p_keyword_query': keyword_query, 
                'p_match_threshold': filters.get('threshold', 0.5), # Default or from config
                'p_match_count': limit,
                'p_rrf_k_val': rrf_k_val
            }
            if keyword_query is None:
                rpc_params.pop('p_keyword_query', None)

            function_name = 'hybrid_search_articles_smart_rrf_matryoshka'
            logger.info(f"Calling Supabase RPC for Matryoshka hybrid articles: {function_name} with params: {rpc_params}")
            response = self.db.rpc(function_name, rpc_params).execute()

            if response.data:
                logger.info(f"Received {len(response.data)} results from '{function_name}'.")
                return [map_db_result_to_document(item, "article") for item in response.data if isinstance(item, dict)]
            else:
                logger.info(f"No results from '{function_name}'.")
                return []
        except Exception as e:
            logger.error(f"Error during LawArticlesAdapter '{function_name}' search: {e}\n{traceback.format_exc()}", exc_info=True)
            return []

    async def hybrid_search_articles_smart_bge(
        self,
        query_embedding: List[float], # Expects BGE-M3-Law embedding
        keyword_query: Optional[str],
        filters: Dict[str, Any], 
        limit: int, 
        rrf_k_val: int = 60,
        **kwargs 
    ) -> List[Document]:
        """
        Performs hybrid search for LAW ARTICLES using the 'hybrid_search_articles_smart_rrf_bge' RPC function.
        Tailored for BGE-M3-Law embeddings.
        """
        try:
            # BGE-M3 embeddings are typically 1024 dimensional.
            # The embedding provider should handle the correct dimension.
            active_query_embedding = query_embedding if query_embedding is not None else [0.0] * 1024 # Default BGE-M3 dim
            
            rpc_params = {
                'p_query_embedding': active_query_embedding,
                'p_keyword_query': keyword_query, 
                'p_match_threshold': filters.get('threshold', 0.25), # Default BGE threshold from test script
                'p_match_count': limit,
                'p_rrf_k_val': rrf_k_val
            }
            if keyword_query is None:
                rpc_params.pop('p_keyword_query', None)

            function_name = 'hybrid_search_articles_smart_rrf_bge'
            logger.info(f"Calling Supabase RPC for BGE-M3-Law hybrid articles: {function_name} with params: {rpc_params}")
            response = self.db.rpc(function_name, rpc_params).execute()

            if response.data:
                logger.info(f"Received {len(response.data)} results from '{function_name}'.")
                return [map_db_result_to_document(item, "article") for item in response.data if isinstance(item, dict)]
            else:
                logger.info(f"No results from '{function_name}'.")
                return []
        except Exception as e:
            logger.error(f"Error during LawArticlesAdapter '{function_name}' search: {e}\n{traceback.format_exc()}")
            return []

    async def hybrid_search_articles_meta_matryoshka(
        self,
        query_embedding: List[float], 
        keyword_query: Optional[str],
        filters: Dict[str, Any], 
        limit: int,
        rrf_k_val: int = 60,
        **kwargs 
    ) -> List[Document]:
        """
        Performs hybrid search for LAW ARTICLES using the 'hybrid_search_articles_smart_rrf_matryoshka' RPC function.
        This is for the 'المستشار' persona's Matryoshka retrieval part.
        """
        logger.info("Routing 'المستشار' to use 'hybrid_search_articles_smart_matryoshka' for simplicity.")
        return await self.hybrid_search_articles_smart_matryoshka(
            query_embedding=query_embedding,
            keyword_query=keyword_query,
            filters=filters,
            limit=limit,
            rrf_k_val=rrf_k_val,
            **kwargs
        )

    async def hybrid_search_articles_meta_bge(
        self,
        query_embedding: List[float], 
        keyword_query: Optional[str],
        filters: Dict[str, Any], 
        limit: int,
        rrf_k_val: int = 60,
        filter_law_categories: Optional[List[str]] = None,
        filter_article_tags: Optional[List[str]] = None,
        filter_sharia_influence: Optional[bool] = None,
        metadata_filter_logic: str = 'AND', 
        metadata_score_weight: float = 0.3, 
        **kwargs 
    ) -> List[Document]:
        """
        Performs hybrid search for LAW ARTICLES using the 'hybrid_search_articles_meta_rrf_bge' RPC function.
        This is for the 'المستشار' persona's BGE-M3-Law retrieval part.
        """
        try:
            active_query_embedding = query_embedding if query_embedding is not None else [0.0] * 1024 # BGE-M3 dim
            
            rpc_params = {
                'p_query_embedding': active_query_embedding,
                'p_keyword_query': keyword_query,
                'p_match_threshold': filters.get('threshold', 0.25), # BGE has a different optimal threshold
                'p_match_count': limit,
                'p_filter_article_number': filters.get('article_number'),
                'p_filter_sharia_influence': filter_sharia_influence,
                'p_filter_law_categories': filter_law_categories if filter_law_categories else [],
                'p_filter_article_tags': filter_article_tags if filter_article_tags else [],
                'p_rrf_k_val': rrf_k_val,
                'p_metadata_filter_logic': metadata_filter_logic,
                'p_metadata_score_weight': metadata_score_weight, 
            }
            rpc_params['p_effective_limit_multiplier'] = kwargs.get('p_effective_limit_multiplier', 5) 

            # Clean params: remove None values and empty lists to avoid issues with RPC calls
            rpc_params = {k: v for k, v in rpc_params.items() if v is not None and v != []}

            function_name = 'hybrid_search_articles_meta_rrf_bge'
            logger.info(f"Calling Supabase RPC for 'المستشار' (BGE part): {function_name} with params: {rpc_params}")
            response = self.db.rpc(function_name, rpc_params).execute()

            if response.data:
                logger.info(f"Received {len(response.data)} results from '{function_name}'.")
                return [map_db_result_to_document(item, "article") for item in response.data if isinstance(item, dict)]
            else:
                logger.info(f"No results from '{function_name}'.")
                return []
        except Exception as e:
            logger.error(f"Error during LawArticlesAdapter '{function_name}' search: {e}\n{traceback.format_exc()}")
            return []

    async def semantic_search_articles_basic(
        self,
        query_embedding: List[float],
        filters: Dict[str, Any], 
        limit: int, 
        **kwargs
    ) -> List[Document]:
        """
        Performs semantic search only for LAW ARTICLES using the 'semantic_search_articles_basic' RPC function.
        This version is for the 'الغبي' (Simple) persona.
        """
        try:
            active_query_embedding = query_embedding if query_embedding is not None else [0.0] * DEFAULT_EMBEDDING_DIM
            
            rpc_params = {
                'p_query_embedding': active_query_embedding,
                'p_match_threshold': filters.get('threshold', 0.5),
                'p_match_count': limit
            }

            function_name = 'semantic_search_articles_basic'
            logger.info(f"Calling Supabase RPC for 'الغبي' articles: {function_name} with params: {rpc_params}")
            response = self.db.rpc(function_name, rpc_params).execute()

            if response.data:
                logger.info(f"Received {len(response.data)} results from '{function_name}'.")
                adapted_data = []
                for item in response.data:
                    if isinstance(item, dict):
                        item['score'] = item.get('similarity', 0.0) 
                        adapted_data.append(item)
                return [map_db_result_to_document(item, "article") for item in adapted_data]
            else:
                logger.info(f"No results from '{function_name}'.")
                return []
        except Exception as e:
            logger.error(f"Error during LawArticlesAdapter '{function_name}' search: {e}\n{traceback.format_exc()}")
            return []

    async def semantic_search_articles_matryoshka_v2(
        self,
        query_embedding: List[float], # Expects Matryoshka embedding
        filters: Dict[str, Any], 
        limit: int, 
        **kwargs
    ) -> List[Document]:
        """
        Performs semantic search for LAW ARTICLES using the 'semantic_search_articles_matryoshka_v2' RPC function.
        Tailored for Matryoshka embeddings.
        """
        try:
            # Matryoshka embeddings are 768 dimensional.
            # The embedding provider should handle the correct dimension for the input vector.
            active_query_embedding = query_embedding if query_embedding is not None else [0.0] * 768 
            
            rpc_params = {
                'p_query_embedding': active_query_embedding,
                'p_match_threshold': filters.get('threshold', 0.5), # Default or from config
                'p_match_count': limit
            }

            function_name = 'semantic_search_articles_matryoshka_v2'
            logger.info(f"Calling Supabase RPC for Matryoshka semantic articles (v2): {function_name} with params: {rpc_params}")
            response = self.db.rpc(function_name, rpc_params).execute()

            if response.data:
                logger.info(f"Received {len(response.data)} results from '{function_name}'.")
                # map_db_result_to_document handles 'similarity' if present and maps to 'score'
                return [map_db_result_to_document(item, "article") for item in response.data if isinstance(item, dict)]
            else:
                logger.info(f"No results from '{function_name}'.")
                return []
        except Exception as e:
            logger.error(f"Error during LawArticlesAdapter '{function_name}' search: {e}\n{traceback.format_exc()}")
            return []

    async def meta_hybrid_search(
        self,
        matryoshka_embedding: List[float],
        bge_embedding: List[float],
        keyword_query: Optional[str],
        filters: Dict[str, Any],
        limit: int,
        rrf_k_val: int = 60,
        meta_rrf_k_val: int = 60,
        **kwargs
    ) -> List[Document]:
        """
        Performs a meta-hybrid search by calling two different hybrid search RPCs
        and then fusing the results using a client-side RRF.
        """
        logger.info("--- Performing Meta-Hybrid Search ---")

        # Call both hybrid search functions concurrently
        matryoshka_task = self.hybrid_search_articles_smart_matryoshka(
            query_embedding=matryoshka_embedding,
            keyword_query=keyword_query,
            filters=filters,
            limit=limit,
            rrf_k_val=rrf_k_val,
            **kwargs
        )
        bge_task = self.hybrid_search_articles_smart_bge(
            query_embedding=bge_embedding,
            keyword_query=keyword_query,
            filters=filters,
            limit=limit,
            rrf_k_val=rrf_k_val,
            **kwargs
        )

        results = await asyncio.gather(matryoshka_task, bge_task)
        matryoshka_docs, bge_docs = results

        logger.info(f"Matryoshka RPC returned {len(matryoshka_docs)} docs.")
        logger.info(f"BGE RPC returned {len(bge_docs)} docs.")

        # Fuse the results
        fused_docs = _client_side_reciprocal_rank_fusion_adapted(
            [matryoshka_docs, bge_docs],
            k=meta_rrf_k_val
        )
        logger.info(f"Meta-Hybrid RRF resulted in {len(fused_docs)} fused documents.")

        if fused_docs:
            logger.info(f"--- Top 5 Meta-Hybrid Fused Docs ---")
            for i, doc in enumerate(fused_docs[:5]):
                logger.info(f"  [{i+1}] ID: {doc.metadata.get('article_id')}, Score: {doc.metadata.get('score'):.4f}, Law: {doc.metadata.get('law_name')}, Article: {doc.metadata.get('article_number')}")
            logger.info(f"------------------------------------")

        return fused_docs

    async def hybrid_search_comments_smart_rrf(
        self,
        query_embedding: List[float],
        keyword_query: Optional[str],
        filters: Dict[str, Any], 
        limit: int,
        rrf_k_val: int = 60,
        filter_tags: Optional[List[str]] = None,
        metadata_filter_logic: str = 'AND', 
        **kwargs 
    ) -> List[Document]:
        """
        Performs hybrid search for COMMENTS using the 'hybrid_search_comments_rrf' RPC function.
        This version is for the 'الذكي' (Smart) persona and does not include metadata score weighting in RRF.
        It assumes the RPC 'hybrid_search_comments_rrf' handles RRF internally without explicit metadata weight.
        """
        try:
            active_query_embedding = query_embedding if query_embedding is not None else [0.0] * DEFAULT_EMBEDDING_DIM
            
            rpc_params = {
                'query_embedding': active_query_embedding, # Changed from p_query_embedding to match DB
                'keyword_query': keyword_query,
                'match_threshold': filters.get('threshold', 0.5),
                'match_count': limit,
                'p_k_val': rrf_k_val, # Changed from p_rrf_k_val to p_k_val
                # 'p_filter_tags' and 'p_metadata_filter_logic' are removed as they are not in the DB function signature
            }

            if keyword_query is None:
                rpc_params.pop('keyword_query', None) # Ensure correct key is popped
            
            # Clean rpc_params: remove None values.
            # The DB function has defaults for keyword_query and p_k_val, so sending None or omitting them is fine.
            rpc_params = {k: v for k, v in rpc_params.items() if v is not None}


            function_name = 'hybrid_search_comments_smart_rrf' 
            logger.info(f"Calling Supabase RPC for 'الذكي' comments: {function_name} with params: {rpc_params}")
            response = self.db.rpc(function_name, rpc_params).execute()

            if response.data:
                logger.info(f"Received {len(response.data)} results from comment hybrid search ('{function_name}').")
                return [map_db_result_to_document(item, "comment") for item in response.data if isinstance(item, dict)]
            else:
                logger.info(f"No results from comment hybrid search ('{function_name}').")
                return []
        except Exception as e:
            logger.error(f"Error during LawArticlesAdapter comment hybrid search ('{function_name}'): {e}\\n{traceback.format_exc()}")
            return []
