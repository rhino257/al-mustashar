# Metadata Generation PoC - Progress Log

**Last Updated:** 2025-06-07

**Objective:**
To develop and test a Proof-of-Concept (PoC) script for automatically generating metadata (categories, Sharia influence assessment for laws; topical tags for law articles) for Yemeni legal documents stored in the Supabase "Knowledge Database". This metadata aims to enhance the retrieval accuracy and relevance of the "Almustashar" RAG agent.

**Key Technologies/Components Involved:**
*   Python script: `scripts/db_processing/01_populate_metadata_poc.py`
*   LLM: Google `gemini-2.0-flash` (accessed via Portkey.ai)
*   Pydantic models: `LawClassificationOutput`, `ArticleTaggingOutput` for structured LLM output.
*   Supabase Python client for database interaction.
*   `tqdm` for progress display.

**Summary of Work Done (Chronological):**

1.  **Initial Script Development & Setup:**
    *   Created the base PoC script to fetch laws and articles.
    *   Configured to use `gemini-pro` initially, later changed to `gemini-2.0-flash` based on user preference for PoC.
    *   Defined Pydantic models for desired structured output from the LLM.

2.  **Iterative Debugging & Refinement:**
    *   **Import Errors:** Resolved several Python import errors related to relative imports within the `scripts` directory structure by adding `__init__.py` files to `scripts/` and `scripts/db_processing/`, and adjusting import statements to be absolute from the project root.
    *   **LLM Initialization:** Corrected `ProductionGeminiChatModel` instantiation. It was initially passed a `config` object, but its `__init__` expects individual parameters or loads from environment variables. Changed to `ProductionGeminiChatModel()`.
    *   **API Call Issues (Schema Compatibility):**
        *   Encountered `AttributeError` because the script was calling a non-existent `ainvoke_with_tool_by_name` method. Refactored LLM call functions (`llm_law_classification`, `llm_article_tagging`) to use the existing `chat_completion_with_tools` method, including formatting Pydantic models into JSON tool schemas and parsing the LLM's tool call responses.
        *   Encountered `openai.BadRequestError` from Google API due to schema incompatibility with `Optional` fields in Pydantic models generating `anyOf` with other properties (like `title`, `description`). Resolved by changing `Optional` fields in `LawClassificationOutput` and `ArticleTaggingOutput` (e.g., `confidence_score`, `reasoning`, `sharia_influence`) to have non-`None` default values (e.g., `0.0`, `""`, `False`), which simplifies the generated JSON schema.
    *   **Progress Display:** Added `tqdm` progress bars for processing loops.
    *   **Output Verbosity:** Reduced console output during script runs by commenting out detailed suggestion printing.
    *   **Conditional Review Flags:** Implemented logic to set `*_needs_review` flags conditionally based on LLM confidence and absence of errors, rather than always `TRUE`.
    *   **Full Database Processing:** Modified script to process all eligible records by default.

3.  **Prompt Engineering & Vocabulary Expansion:**
    *   **Initial Vocabulary:** Defined `FINALIZED_LAW_CATEGORIES` and `INITIAL_ARTICLE_TAG_VOCABULARY`.
    *   **Testing & Analysis:**
        *   Ran the script on a sample of 3 laws/articles.
        *   Ran the script on a larger sample of 20 laws/articles.
        *   Analyzed LLM output for quality, relevance, and adherence to constraints.
    *   **Refinements Based on Analysis:**
        *   Expanded `INITIAL_ARTICLE_TAG_VOCABULARY` with good new tags suggested by the LLM.
        *   Made the system prompt for `llm_law_classification` stricter to enforce the use of only predefined categories and added Python-side validation.
        *   Refined the system prompt for `llm_article_tagging` to encourage a balance of using the vocabulary and suggesting new relevant tags, and to aim for appropriate specificity. Explicitly requested reasoning in Arabic.

**Current Status of the PoC Script (`01_populate_metadata_poc.py`):**
*   The script is functional and can process all eligible laws and articles.
*   It uses `gemini-2.0-flash` to suggest classifications and tags.
*   It writes these suggestions to the `suggested_*` columns in the database.
*   It conditionally sets `*_needs_review` flags.
*   The latest test run (full processing) is currently in progress or recently completed by the user. The script has been updated to request reasoning in Arabic.

**Assessment of Last Full Test Run (Before Arabic Reasoning Prompt Update):**
*   **Overall Score:** 85/100
*   **Law Classification:** Good (approx. 88/100). One instance of hallucinated category (though Python validation should now catch this). Sharia influence assessment mostly reasonable.
*   **Article Tagging:** Very good (approx. 82/100). Tags generally relevant. Some tendency for overly general "interpretation/proof" tags for foundational articles. Expanded vocabulary was used well.
*   **Issue Noted:** LLM reasoning was sometimes in English, not consistently Arabic. (This has since been addressed by user manually updating the script's prompt for Arabic reasoning).

---

## Phase 2: RAG Agent Enhancement (Utilizing New Metadata) - Detailed Plan

**(Added: 2025-06-07)**

**Overall Goal:** Transform the "Almustashar" RAG agent to intelligently use the new rich metadata (`classification`, `sharia_influence`, `tags` - once reviewed and finalized) for significantly improved retrieval accuracy, relevance, and the ability to handle complex, multi-faceted legal queries.

**Core Principles for a "Powerful" Agent:**
*   **Deep Query Understanding:** Go beyond simple keyword extraction. Understand the user's *intent*, the specific legal *concepts* they're interested in, and any *contextual constraints*.
*   **Flexible Retrieval Strategy:** Dynamically choose or combine strategies (semantic search, keyword search on tags, filtering by category) based on the query analysis.
*   **Multi-faceted Filtering:** Allow for combining multiple filters (e.g., specific category AND specific tags AND Sharia influence).
*   **Contextual Re-ranking (Optional but Powerful):** After initial retrieval, potentially re-rank results based on how well they match the *full context* of the query, including all identified metadata aspects.
*   **Transparency (Optional):** For complex queries, the agent could briefly explain *why* certain documents were retrieved.

**Detailed Steps for Phase 2:**

1.  **Pydantic Model Enhancement (`YemeniLegalQueryAnalysis`):**
    *   **Location:** `src/shared/models.py` (or current definition location).
    *   **Add Fields:**
        *   `identified_law_categories: Optional[List[str]] = Field(default_factory=list, description="Law categories identified from the user query, matching the predefined taxonomy.")`
        *   `identified_tags: Optional[List[str]] = Field(default_factory=list, description="Potential topical tags identified from the user query.")`
        *   `query_intent_details: Optional[Dict[str, Any]] = Field(default=None, description="Structured details about the user's specific intent, e.g., seeking definition, conditions, penalties, procedure.")`
        *   `filter_logic: Optional[Literal['AND', 'OR']] = Field(default='AND', description="Logic to apply between different filter types (categories, tags). Default to AND for precision.")`
        *   `confidence_scores: Optional[Dict[str, float]] = Field(default_factory=dict, description="Confidence scores for identified categories, tags, or intent.")`

2.  **Advanced Query Understanding Node (`understand_yemeni_legal_query_node`):**
    *   **Location:** `src/custom_nodes/comprehension_nodes.py`.
    *   **Prompt Engineering:**
        *   Instruct the LLM for thorough analysis: identify main legal concepts for `identified_tags`, determine relevant `identified_law_categories` (from `FINALIZED_LAW_CATEGORIES`), infer `query_intent_details`, suggest `filter_logic`, and provide `confidence_scores`.
        *   Consider few-shot examples for complex query breakdown.
    *   **Output:** Populate the enhanced `YemeniLegalQueryAnalysis` model.

3.  **Dynamic Retrieval Strategy & Filtering Logic (Retrieval Node):**
    *   **Location:** `src/custom_nodes/retrieval_nodes.py` (e.g., `supabase_hybrid_retriever_node`).
    *   **Input:** Rich `query_analysis_result` from `AgentState`.
    *   **Decision Logic:**
        *   Prioritize direct lookup if `identified_law_name` and `identified_article_number` are high confidence.
        *   Use `identified_law_categories` and `identified_tags` for primary/secondary filtering, applying `filter_logic`.
        *   Consider `filter_sharia_influence` if query context implies it.
        *   Implement fallback to broader search if initial filtered results are insufficient.
    *   **Iterative Retrieval (Advanced - Future):** For very complex queries, consider multi-step retrieval (e.g., filter laws by category, then articles by tags).

4.  **Supabase RPC Function Enhancement (e.g., `hybrid_search_with_scores`):**
    *   **Parameters:** Add optional `filter_categories TEXT[]`, `filter_tags TEXT[]`, `filter_sharia_influence BOOLEAN`.
    *   **SQL Logic:** Modify the RPC's SQL to incorporate these filters in the `WHERE` clause *before* vector search.
        *   `law.classification && filter_categories` (array overlap) or `law.classification @> filter_categories` (array contains all).
        *   `law_articles.tags && filter_tags` (array overlap).
        *   `law.sharia_influence = filter_sharia_influence`.
    *   Consider how to combine multiple active filters (e.g., using `AND` logic).
    *   Ensure RPC returns necessary metadata for the adapter node.

5.  **`LawArticleAdapterNode` Enhancement (`src/custom_nodes/law_articles_adapter.py`):**
    *   Include `law.classification`, `law.sharia_influence`, and `law_articles.tags` in the formatted context string for each retrieved document to aid the synthesis LLM.

6.  **Synthesis Node (`generate_response_node` in `src/custom_nodes/response_nodes.py`):**
    *   The synthesis prompt can be subtly guided by `query_intent_details` to tailor the answer structure (e.g., focus on listing conditions if that was the intent).

**Implementation Order for Phase 2:**
*   Step 1: Update `YemeniLegalQueryAnalysis` model.
*   Step 2: Update Query Understanding Node and its prompt.
*   Step 3 & 4 (Iterative): Enhance RPC function and the Retrieval Node to use the new filters. This will likely be an iterative process.
*   Step 5 & 6: Update Adapter and Synthesis nodes.

---
**(Previous content of 05-progress-log-المستشار.md continues below this new section if any)**
