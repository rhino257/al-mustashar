from __future__ import annotations # Add this for deferred evaluation of type hints

"""State management for the retrieval graph.

This module defines the state structures used in the retrieval graph. It includes
definitions for agent state, input state, and router classification schema.
"""

from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict, List, Optional, AsyncGenerator, Dict, Any # Added Dict, Any

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

# Attempt to resolve Pylance error for YemeniLegalQueryAnalysis
from shared.models import YemeniLegalQueryAnalysis # Updated import
from shared.state import reduce_docs


# Optional, the InputState is a restricted version of the State that is used to
# define a narrower interface to the outside world vs. what is maintained
# internally.
@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. It serves as
    a restricted version of the full State, providing a narrower interface
    to the outside world compared to what is maintained internally.
    """
    user_query: str
    """The original user query from the API request."""
    chat_id: str
    """The ID of the chat session."""
    user_id: str
    """The ID of the authenticated user."""
    user_name: Optional[str] = None
    """The name of the authenticated user."""
    phone_number: Optional[str] = None
    """The phone number of the authenticated user."""
    use_reranker: bool = False # Add use_reranker to InputState
    """Flag to indicate whether reranking should be used, from API input."""
    messages: Annotated[list[AnyMessage], add_messages]
    """Messages track the primary execution state of the agent.

    Typically accumulates a pattern of Human/AI/Human/AI messages; if
    you were to combine this template with a tool-calling ReAct agent pattern,
    it may look like this:

    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
         information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    
        (... repeat steps 2 and 3 as needed ...)
    4. AIMessage without .tool_calls - agent responding in unstructured
        format to the user.

    5. HumanMessage - user responds with the next conversational turn.

        (... repeat steps 2-5 as needed ... )
    
    Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`."""


class Router(TypedDict):
    """Classify user query."""

    logic: str
    type: Literal["more-info", "langchain", "general"]


# This is the primary state of your agent, where you can store any information


@dataclass(kw_only=True)
class AgentState(InputState):
    """State of the retrieval graph / agent."""

    router: Router = field(default_factory=lambda: Router(type="general", logic=""))
    """The router's classification of the user's query."""
    steps: list[str] = field(default_factory=list)
    """A list of steps in the research plan."""
    documents: Annotated[list[Document], reduce_docs] = field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""
    user_query: str # Add user_query to AgentState
    """The original user query from the API request."""
    chat_id: str # Add chat_id to AgentState
    """The ID of the chat session."""
    user_id: str # Add user_id to AgentState
    """The ID of the authenticated user."""
    user_name: Optional[str] = None # Add user_name to AgentState
    """The name of the authenticated user."""
    phone_number: Optional[str] = None # Add phone_number to AgentState
    """The phone number of the authenticated user."""
    query_embedding: Optional[List[float]] = field(default=None) # Keep kw_only=True behavior
    """Stores the query embedding generated for semantic search (this will typically be the HyDE embedding)."""
    original_query_embedding: Optional[List[float]] = field(default=None)
    """Stores the embedding of the original, raw user query for multi-query strategies."""

    # --- Fields for Yemeni Legal Query Analysis ---
    query_analysis_result: Optional[YemeniLegalQueryAnalysis] = field(default=None) # Changed from string literal
    """Stores the full structured output from the query comprehension node."""
    query_classification: Optional[
        Literal[
            "conversational",
            "legal_query_direct_lookup",
            "legal_query_conceptual_search",
            "other",
        ]
    ] = field(default=None)
    """Classification of the user's query based on Yemeni legal context."""
    text_for_embedding: Optional[str] = field(default=None)
    """The text to be used for generating embeddings (HyDE answer or raw query)."""
    identified_law_name: Optional[str] = field(default=None)
    """The name of the law identified in the query, if any."""
    identified_article_number: Optional[str] = field(default=None)
    """The article number identified in the query, if any."""
    search_keywords: List[str] = field(default_factory=list)
    """Keywords extracted from the query for potential keyword search."""
    query_intent: Optional[
        Literal["specific_article_lookup", "conceptual_search", "unknown"]
    ] = field(default=None)
    """The specific intent behind a legal query."""

    # --- Fields for Meta-Hybrid Retrieval ('المستشار' Persona) ---
    matryoshka_embedding: Optional[List[float]] = field(default=None)
    """Stores the Matryoshka embedding for the meta-hybrid strategy."""
    bge_embedding: Optional[List[float]] = field(default=None)
    """Stores the BGE-M3-Law embedding for the meta-hybrid strategy."""
    # --- End Fields for Meta-Hybrid Retrieval ---

    error_message: Optional[str] = field(default=None)
    """Stores error messages from graph nodes."""
    error_node: Optional[str] = field(default=None)
    """Stores the name of the node where an error occurred."""
    # --- End Fields for Yemeni Legal Query Analysis ---

    # --- Fields for Retrieval and Reranking ---
    retrieved_documents: Optional[List[Document]] = field(default_factory=list)
    """Stores documents retrieved by the primary retriever."""
    reranked_documents: Optional[List[Document]] = field(default_factory=list)
    """Stores documents after they have been reranked."""
    use_reranker: bool = False # Populated from API, controls reranking logic
    """Flag to indicate whether reranking should be used."""
    preliminary_answer: Optional[str] = None # Optional, for reranker prompt
    """An optional preliminary answer that can be used to guide reranking."""
    final_documents_for_synthesis: Optional[List[Document]] = field(default_factory=list)
    """The final set of documents (either retrieved or reranked) to be used for answer synthesis."""
    # --- End Fields for Retrieval and Reranking ---

    # --- Fields for Response Generation ---
    final_answer: Optional[str] = field(default=None)
    """Stores the complete synthesized ARABIC answer for legal queries."""
    llm_stream_final_answer: Optional[AsyncGenerator] = field(default=None)
    """Stores the stream generator for the final_answer if streaming is used."""
    conversational_response: Optional[str] = field(default=None)
    """Stores the complete ARABIC conversational answer."""
    llm_stream_conversational_response: Optional[AsyncGenerator] = field(default=None)
    """Stores the stream generator for the conversational_response if streaming is used."""
    llm_output_stream: Optional[AsyncGenerator] = field(default=None) # Added as per Step 1
    """Stores the raw LLM output stream from a node, if applicable."""
    # --- End Fields for Response Generation ---

    # --- Fields for Direct Lookup MCP Call ---
    direct_lookup_mcp_args: Optional[Dict[str, Any]] = field(default=None)
    """Stores arguments for the MCP tool call for direct Supabase lookup (e.g., query, project_id)."""
    direct_lookup_mcp_response: Optional[List[Dict[str, Any]]] = field(default=None)
    """Stores the raw response data from the execute_sql MCP tool for direct lookup."""
    # --- End Fields for Direct Lookup MCP Call ---

    # Feel free to add additional attributes to your state as needed.
    # Common examples include retrieved documents, extracted entities, API connections, etc.
