"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os # Ensure os is imported for getenv
from dataclasses import dataclass, field
from typing import Annotated, Optional

from retrieval_graph import prompts
from shared.configuration import BaseConfiguration


@dataclass(kw_only=True)
class AgentConfiguration(BaseConfiguration):
    """The configuration for the agent."""

    # models

    agent_persona: str = field(
        default="almustashar",
        metadata={
            "description": "The agent persona to use (e.g., 'almustashar', 'alzaki'). Determines underlying model and behavior."
        }
    )

    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gemini_custom/gemini-2.0-flash", # Changed to Gemini Pro
        metadata={
            "description": "The language model used for processing and refining queries. Should be in the form: provider/model-name."
        },
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gemini_custom/gemini-2.0-flash", # Changed as per user request
        metadata={
            "description": "The language model used for generating responses. Should be in the form: provider/model-name."
        },
    )

    # prompts

    router_system_prompt: str = field(
        default=prompts.ROUTER_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for classifying user questions to route them to the correct node."
        },
    )

    more_info_system_prompt: str = field(
        default=prompts.MORE_INFO_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for asking for more information from the user."
        },
    )

    general_system_prompt: str = field(
        default=prompts.GENERAL_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for responding to general questions."
        },
    )

    research_plan_system_prompt: str = field(
        default=prompts.RESEARCH_PLAN_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for generating a research plan based on the user's question."
        },
    )

    generate_queries_system_prompt: str = field(
        default=prompts.GENERATE_QUERIES_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used by the researcher to generate queries based on a step in the research plan."
        },
    )

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating responses."},
    )

    reranker_model_name: Annotated[Optional[str], {"__template_metadata__": {"kind": "llm"}}] = field(
        default=None, # Or you can set a default model e.g., "gemini_custom/gemini-2.0-flash"
        metadata={
            "description": "The language model used for reranking documents. If None, query_model will be used. Should be in the form: provider/model-name."
        },
    )

    reranker_top_k: int = field(
        default=10,
        metadata={
            "description": "The number of documents to return after reranking."
        }
    )

    retriever_per_source_limit: int = field(
        default=20,
        metadata={
            "description": "The number of documents to retrieve from each source (e.g., articles, comments) in the hybrid retriever."
        }
    )

    # Cohere ReRanker specific configurations
    COHERE_PORTKEY_VIRTUAL_KEY_ID: Optional[str] = field(
        default_factory=lambda: os.getenv("COHERE_PORTKEY_VIRTUAL_KEY_ID", "cohere-api-ab1b58"),
        metadata={"description": "Portkey Virtual Key ID for Cohere ReRank service."},
    )
    COHERE_RERANK_MODEL_NAME: str = field(
        default="rerank-multilingual-v3.0", # Example, use the actual model you intend
        metadata={"description": "The specific Cohere ReRank model name to be used via Portkey."},
    )
    MAX_DOCS_FOR_RERANKING: int = field(
        default=50,
        metadata={"description": "Maximum number of documents to send to the Cohere reranker from the initial retrieval."},
    )
    # reranker_top_k is already defined above, it specifies how many docs to *return* from reranker

    DEFAULT_RRF_K_VAL: int = field(
        default=60,
        metadata={"description": "Default K value for Reciprocal Rank Fusion (RRF) in Supabase RPC functions."},
    )

    metadata_score_weight_articles: float = field(
        default=0.6, # Tuned value, was 0.3 in original RPC default
        metadata={"description": "Weight for metadata score in RRF for article search."}
    )

    metadata_score_weight_comments: float = field(
        default=0.1, 
        metadata={"description": "Weight for metadata match score in RRF for comment search."}
    )

    # Persona-specific limits for documents sent to synthesis
    almustashar_synthesis_limit: int = field(
        default=20,
        metadata={"description": "Number of documents to use for synthesis for the 'المستشار' persona."}
    )
    alzaki_synthesis_limit: int = field(
        default=10,
        metadata={"description": "Number of documents to use for synthesis for the 'الذكي' persona."}
    )
    alghabi_synthesis_limit: int = field(
        default=5,
        metadata={"description": "Number of documents to use for synthesis for the 'الغبي' persona."}
    )

    # --- Configurations for Meta-Hybrid Retrieval (الذكي Persona) ---
    BGE_M3_LAW_EMBEDDING_MODEL_NAME: Optional[str] = field(
        default="bge_m3_law/default", # Placeholder, adjust if BGE model is loaded differently
        metadata={
            "description": "Identifier for the BGE-M3-Law embedding model, used if meta-hybrid strategy requires it."
        }
    )

    MATRYOSHKA_RPC_SEMANTIC_THRESHOLD: float = field(
        default=0.5,
        metadata={"description": "Semantic threshold for the Matryoshka Hybrid RPC."}
    )

    BGE_RPC_SEMANTIC_THRESHOLD: float = field(
        default=0.25,
        metadata={"description": "Semantic threshold for the BGE-M3-Law Hybrid RPC."}
    )

    META_RRF_K_VAL: int = field(
        default=60,
        metadata={"description": "K value for client-side Reciprocal Rank Fusion in Meta-Hybrid strategy."}
    )
    
    RETRIEVER_PER_SOURCE_LIMIT_META: int = field(
        default=10,
        metadata={"description": "Number of documents to retrieve from each source RPC (Matryoshka, BGE) before client-side RRF in Meta-Hybrid strategy."}
    )
    # --- End of Meta-Hybrid Configurations ---

    # Credentials for the Knowledge Base Supabase (for direct lookup)
    knowledge_base_supabase_url: Optional[str] = field(
        default_factory=lambda: os.getenv("KNOWLEDGE_SUPABASE_URL", os.getenv("USERS_SUPABASE_URL")),
        metadata={"description": "Supabase URL for the knowledge base. Defaults to KNOWLEDGE_SUPABASE_URL env var, then USERS_SUPABASE_URL env var."}
    )
    knowledge_base_supabase_key: Optional[str] = field(
        default_factory=lambda: os.getenv("KNOWLEDGE_SUPABASE_KEY", os.getenv("SUPABASE_SERVICE_ROLE_KEY")),
        metadata={"description": "Supabase Service Role Key for the knowledge base. Defaults to KNOWLEDGE_SUPABASE_KEY env var, then SUPABASE_SERVICE_ROLE_KEY env var."}
    )
