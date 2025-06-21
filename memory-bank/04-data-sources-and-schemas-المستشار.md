# Data Sources and Schemas: "المستشار"
## 1. Primary Data Source: Yemeni Law
 - **Initial Stage:** Text snippets/summaries retrieved directly from Pinecone metadata.
 - **Future Enhancement:** Full text fetched from Supabase (deferred).
 - **Long-term:** Content primarily sourced and cross-referenced via the Yemen Legal Knowledge Graph (YLKG) and Versioned Knowledge Store.

 ## 2. Vector Database: Pinecone
 - Stores embeddings of Yemeni legal text.
 - Metadata includes document source, type, and potentially other filterable fields.
 - `pinecone_top_k`: Currently set to 10 in `BaseConfiguration` (src/shared/configuration.py).

 ## 3. Metadata Database: Supabase
 - Initial use: Direct lookup for specific articles (e.g., `SupabaseDirectLookupNode` - deferred).
 - Future use: Will likely form part of the Data & Learning Infrastructure (DLI) for storing interaction logs, annotations, etc.

 ## 4. `AgentState` Schema (`src/retrieval_graph/state.py` - Key Fields)
 - `user_query: str`
 - `messages: Annotated[list[AnyMessage], add_messages]`
 - `documents: Annotated[list[Document], reduce_docs]`
 - `query_embedding: Optional[List[float]] = None` (Implemented)
 - **New fields added in Task 13 (Implemented):**
   - `query_analysis_result: Optional[YemeniLegalQueryAnalysis]` (Full structured output from comprehension node)
   - `query_classification: Optional[Literal["conversational", "legal_query_direct_lookup", "legal_query_conceptual_search", "other"]]`
   - `text_for_embedding: Optional[str]` (Populated by comprehension node with HyDE or raw query)
   - `identified_law_name: Optional[str]`
   - `identified_article_number: Optional[str]`
   - `search_keywords: List[str]`
   - `query_intent: Optional[Literal["specific_article_lookup", "conceptual_search", "unknown"]]`
   - `error_message: Optional[str]`
   - `error_node: Optional[str]`
 - **New fields added in Task 14 (Implemented):**
   - `final_answer: Optional[str]` (Stores the complete synthesized ARABIC answer for legal queries)
   - `llm_stream_final_answer: Optional[AsyncGenerator]` (Stores the stream generator for the final_answer)
   - `conversational_response: Optional[str]` (Stores the complete ARABIC conversational answer)
   - `llm_stream_conversational_response: Optional[AsyncGenerator]` (Stores the stream generator for the conversational_response)
 - The old `Router` TypedDict (`logic`, `type`) in `AgentState` is still present but is superseded by the new query comprehension fields for the primary legal RAG flow.

 ## 5. `YemeniLegalQueryAnalysis` Pydantic Model (`src/custom_nodes/comprehension_nodes.py`)
 - This model defines the structured output from the `understand_yemeni_legal_query_node`.
 - Key fields: `classification`, `raw_query`, `hypothetical_answer_for_embedding`, `intent`, `law_name`, `article_number`, `keywords_for_search`.

 ## 6. Yemen Legal Knowledge Graph (YLKG) Schema (High-Level - To Be Detailed)
 - Entity Types: Law, Article, Case, LegalConcept, Person, Organization, Date, Jurisdiction.
 - Relationship Types: Cites, Amends, Interprets, AppliesTo, RuledInFavorOf, etc.

 ## 7. Traceability-Centric Database (DLI) Schema (High-Level - To Be Detailed)
 - Core Tables: Interactions, Queries, AgentSteps, RetrievalEvents, KGQueryEvents, ReasoningEvents, VerificationEvents, SynthesisOutputs, UserFeedback, Annotations.
 - Focus: Logging every step of an interaction for analysis, debugging, and retraining.
