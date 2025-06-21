"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os # Added import for os.getenv
from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config


@dataclass(kw_only=True)
class BaseConfiguration:
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including embedding model selection, retriever provider choice, and search parameters.
    """

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="openai_custom_embeddings/text-embedding-ada-002", # Reverted to text-embedding-ada-002
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    retriever_provider: Annotated[
        Literal["elastic-local", "elastic", "mongodb", "supabase_hybrid"], # Removed "pinecone", added "supabase_hybrid"
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="supabase_hybrid", # Changed default to supabase_hybrid
        metadata={
            "description": "The vector store provider to use for retrieval. Options are 'elastic', 'mongodb', or 'supabase_hybrid'."
        },
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Additional keyword arguments to pass to the search function of the retriever."
        },
    )

    # Pinecone fields removed
    # pinecone_top_k: int = field(
    #     default=10,
    #     metadata={
    #         "description": "The number of results to retrieve from Pinecone."
    #     },
    # )

    # Knowledge Supabase
    knowledge_supabase_url: Optional[str] = field(
        default=None,
        metadata={"description": "URL for the Knowledge Supabase instance."},
    )
    knowledge_supabase_key: Optional[str] = field(
        default=None,
        metadata={"description": "Service role key for the Knowledge Supabase instance."},
    )

    # Hybrid Search Parameters
    hybrid_search_threshold: float = field(
        default=0.55, # Increased from 0.5 to make semantic search stricter
        metadata={"description": "Similarity threshold for Supabase hybrid search."},
    )
    hybrid_search_match_count: int = field(
        default=20, # Changed to 20 as per user request for performance testing
        metadata={"description": "Number of matches to retrieve from Supabase hybrid search."},
    )
    retriever_per_source_limit: int = field(
        default=20, 
        metadata={"description": "Number of matches to retrieve per source in Supabase hybrid search (e.g., for articles, for comments)."}
    )

    PORTKEY_API_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv("PORTKEY_API_KEY"),
        metadata={"description": "API key for Portkey.ai service."},
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of IndexConfiguration with the specified configuration.
        """
        import logging # Temporary import for logging
        logger = logging.getLogger(__name__) # Temporary logger

        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        
        logger.info(f"BaseConfiguration.from_runnable_config called for class {cls.__name__}.")
        logger.info(f"Received configurable: {configurable}")

        _fields = {f.name for f in fields(cls) if f.init}
        
        # Log which fields are being populated
        final_kwargs = {k: v for k, v in configurable.items() if k in _fields}
        logger.info(f"Populating {cls.__name__} with kwargs: {final_kwargs}")
        
        return cls(**final_kwargs)


T = TypeVar("T", bound=BaseConfiguration)
