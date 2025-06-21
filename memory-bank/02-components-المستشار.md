# Key Software Components: "المستشار"
## 1. Custom Providers
 - **`GeminiChatModel`**:
   - Location: `src/custom_providers/gemini_chat_model.py`.
   - Purpose: Adapts Gemini API for chat interactions, including async methods, retry/rate-limiting, LangChain BaseMessage handling.
   - Status: Implemented and Verified.
 - **`OpenAICustomEmbeddings`**:
   - Location: `src/custom_providers/openai_custom_embeddings.py`.
   - Purpose: Handles OpenAI embedding generation with async client, chunking, and retry logic.
   - Status: Implemented and Verified.

 ## 2. LangGraph Nodes (Initial Set from Handoff Report)
 - **`generate_query_embedding_node`**:
   - Location: `src/custom_nodes/retrieval_nodes.py`.
   - Purpose: Generates embedding for `state.text_for_embedding` (which is derived from HyDE or the raw query by the comprehension node) and stores in `state.query_embedding`.
   - Dependencies: `OpenAICustomEmbeddings` (via `make_text_encoder`).
   - Status: Updated and Verified.
 - **`pinecone_semantic_retriever_node`**:
   - Location: `src/custom_nodes/retrieval_nodes.py`.
   - Purpose: Queries Pinecone using `state.query_embedding`. Extracts text snippets from metadata and populates `AgentState.documents`.
   - Status: Implemented and Verified.

 ## 3. Configuration
 - `.env` file: Stores API keys (GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, OPENAI_API_KEY).
 - `src/shared/utils.py` (`load_chat_model`): Loads `GeminiChatModel`.
 - `src/shared/retrieval.py` (`make_text_encoder`): Loads `OpenAICustomEmbeddings`.

 ## 4. Agent State (`src/retrieval_graph/state.py`)
 - Key fields added:
   - `query_embedding: Optional[List[float]] = None` (Implemented)
   - `query_analysis_result: Optional[YemeniLegalQueryAnalysis]` (Implemented)
   - `query_classification: Optional[Literal["conversational", ...]]` (Implemented)
   - `text_for_embedding: Optional[str]` (Implemented, populated by comprehension node)
   - `identified_law_name: Optional[str]` (Implemented)
   - `identified_article_number: Optional[str]` (Implemented)
   - `search_keywords: List[str]` (Implemented)
   - `query_intent: Optional[Literal["specific_article_lookup", ...]]` (Implemented)
   - `error_message: Optional[str]` (Implemented)
   - `error_node: Optional[str]` (Implemented)
   - **New fields added for Direct Lookup (May 28):**
     - `direct_lookup_mcp_args: Optional[Dict[str, Any]]` (Implemented)
     - `direct_lookup_mcp_response: Optional[List[Dict[str, Any]]]` (Implemented)

## 5. Query Comprehension Components (NEW - Task 13)
 - **`understand_yemeni_legal_query_node`**:
   - Location: `src/custom_nodes/comprehension_nodes.py`.
   - Purpose: Performs in-depth analysis of user queries related to Yemeni law.
   - Key functionalities: Query classification (conversational, legal_query_direct_lookup, legal_query_conceptual_search, other), hypothetical answer generation (HyDE) in Arabic, extraction of law name and article number, generation of Arabic search keywords, and intent determination.
   - Input: `AgentState` (user query from messages, chat history).
   - Output: Updates `AgentState` with `YemeniLegalQueryAnalysis` results and derived fields.
   - Status: Implemented and Verified.
 - **`YemeniLegalQueryAnalysis` Pydantic Model**:
   - Location: `src/custom_nodes/comprehension_nodes.py`.
   - Purpose: Defines the structured output expected from the LLM for query analysis.
   - Key fields: `classification`, `raw_query`, `hypothetical_answer_for_embedding`, `intent`, `law_name`, `article_number`, `keywords_for_search`.
   - Status: Implemented and Verified.

## 6. Response Generation Components (NEW - Task 14)
 - **`synthesize_yemeni_legal_answer_node`**:
   - Location: `src/custom_nodes/response_nodes.py`.
   - Purpose: Generates coherent, cited, ARABIC legal answers from retrieved documents, considering chat history and persona. Uses the synthesis prompt.
   - Input: `AgentState` (retrieved documents, query analysis, chat history).
   - Output: Updates `AgentState` with `final_answer` and/or `llm_stream_final_answer`.
   - Status: Implemented.
 - **`handle_conversational_query_node`**:
   - Location: `src/custom_nodes/response_nodes.py`.
   - Purpose: Generates ARABIC conversational responses based on chat history and specific persona instructions. Uses the conversational prompt.
   - Input: `AgentState` (user query, chat history, query classification).
   - Output: Updates `AgentState` with `conversational_response` and/or `llm_stream_conversational_response`.
   - Status: Implemented.
 - **Synthesis Prompt**:
   - Location: `src/prompts/synthesis_prompts.py` (variable `SYNTHESIS_PROMPT_ARABIC`).
   - Purpose: Provides detailed instructions to the LLM for generating comprehensive and cited ARABIC legal answers.
   - Status: Created.
 - **Conversational Prompt**:
   - Location: `src/prompts/conversational_prompts.py` (variable `CONVERSATIONAL_PROMPT_ARABIC`).
   - Purpose: Provides instructions and persona details to the LLM for handling general conversational queries in ARABIC.
   - Status: Created.

## 7. Direct Lookup Components (NEW - Task for May 28)
 - **`prepare_direct_lookup_node`**:
   - Location: `src/custom_nodes/direct_lookup_node.py`.
   - Purpose: Prepares arguments (SQL query, project ID) for a direct article lookup using the `execute_sql` tool of the official Supabase MCP server. Constructs SQL query based on `law_name` and `article_number` from `AgentState.query_analysis_result`, including basic SQL string escaping for `law_name`.
   - Input: `AgentState` (specifically `query_analysis_result`).
   - Output: Updates `AgentState.direct_lookup_mcp_args` with the arguments for the MCP tool call. Clears `AgentState.documents`.
   - Status: Implemented.
 - **`process_direct_lookup_result_node`**:
   - Location: `src/custom_nodes/direct_lookup_node.py`.
   - Purpose: Processes the raw JSON response (expected in `AgentState.direct_lookup_mcp_response`) from the `execute_sql` MCP tool. Maps the database row(s) to LangChain `Document` objects.
   - Input: `AgentState` (specifically `direct_lookup_mcp_response`).
   - Output: Updates `AgentState.documents` with the retrieved document(s). Clears `direct_lookup_mcp_args` and `direct_lookup_mcp_response`.
   - Status: Implemented.

## 8. Retrieval Components (Updates for Hybrid Search - June 9, 2025)
 - **Supabase RPC Function `public.hybrid_search_comments_with_tag_filters_rrf`**:
   - Purpose: Performs hybrid search (semantic + keyword + RRF) on the `public.comments` table, with added support for filtering by `suggested_tags` based on `p_filter_tags` and `p_metadata_filter_logic` parameters.
   - Status: Created (Manually by user, after several tool-based attempts).
 - **`LawArticlesAdapter` (`src/custom_nodes/law_articles_adapter.py`)**:
   - **`hybrid_search_comments` method**: Updated to accept `filter_tags`, `metadata_filter_logic`, and `metadata_score_weight` parameters. It now calls the `hybrid_search_comments_with_tag_filters_rrf` Supabase RPC to perform hybrid search on comments with tag-based metadata filtering.
   - Status: Updated.
 - **`supabase_hybrid_retriever_node` (`src/custom_nodes/retrieval_nodes.py`)**:
   - Updated to extract `identified_tags` and `filter_logic` from `AgentState.query_analysis_result`.
   - Passes these extracted values and a configured `metadata_score_weight_comments` to the updated `hybrid_search_comments` method of the `LawArticlesAdapter`.
   - Status: Updated.
 - **`AgentConfiguration` (`src/retrieval_graph/configuration.py`)**:
    - Added `metadata_score_weight_comments: float` field to allow configuration of the metadata score weight specifically for comment retrieval.
    - Status: Updated.
