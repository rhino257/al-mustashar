# Architecture Documentation: "المستشار"
## 1. Overall Architecture Vision
 Agentic Neuro-Symbolic RAG with Multi-Layered Verification & Learning.

 ## 2. Key Architectural Modules
 1.  **Intelligent Agent Controller (Orchestrator):**
     - LLM-based (initially Gemini, subject to benchmarking).
     - Handles intent recognition, multi-step reasoning orchestration, tool calling, conversational state.
 2.  **Advanced Retrieval Engine (ARE):**
     - Hybrid Search (Vector, Keyword, KG-augmented).
     - Multi-Hop & Recursive Retrieval capabilities.
     - Context-Aware Query Expansion.
     - Sources: Versioned Yemen Legal Corpus (Vector DB), Yemen Legal Knowledge Graph.
 3.  **Yemen Legal Knowledge Graph (YLKG):**
     - Graph Database storing Yemeni legal entities & relationships.
     - Aids retrieval, reasoning, and verification.
 4.  **Hybrid Reasoning Module (HRM):**
     - LLM-driven legal analysis (e.g., IRAC, CoT).
     - Symbolic Rule Engine for deterministic legal logic.
     - KG-based inference.
 5.  **Multi-Layered Verification & Confidence Module (MVCM):**
     - Source Grounding & Attribution Check.
     - Citation Accuracy Check.
     - Temporal Validity Filter.
     - Fact-Consistency Verification.
     - Confidence Scoring & Calibration.
 6.  **Advanced Synthesis Module (ASM):**
     - Generates final, verified, cited, and professionally toned Arabic responses.
     - Transparently communicates uncertainty.
 7.  **Data & Learning Infrastructure (DLI):**
     - Traceability-Centric Database for all interactions.
     - Versioned Knowledge Store for legal documents.
     - Annotation Platform.
     - Feedback-Driven Retraining Pipeline.

 ## 3. Current LangGraph Implementation (High-Level)
 - Built upon the "LangGraph RAG Research Agent Template."
 - Core State: `AgentState` in `src/retrieval_graph/state.py`.
 - Key Nodes (Current/Planned Initial):
     - `understand_yemeni_legal_query_node` (in `src/custom_nodes/comprehension_nodes.py`): The new entry point for query processing. Performs detailed analysis of Yemeni legal queries, including classification, hypothetical answer generation (HyDE), entity extraction (law name, article number), and keyword generation. Uses `route_yemeni_legal_query` for conditional routing.
     - `analyze_and_route_query` (in `src/retrieval_graph/graph.py`): Original template router, now largely superseded by `understand_yemeni_legal_query_node` for primary legal query flow. Kept for reference or potential non-legal routing.
     - `generate_query_embedding_node` (in `src/custom_nodes/retrieval_nodes.py`): Generates embeddings based on `state.text_for_embedding` (from HyDE or raw query).
     - `pinecone_semantic_retriever_node` (in `src/custom_nodes/retrieval_nodes.py`): Retrieves documents from Pinecone using the generated embedding.
     - `SynthesizeAnswerNode` (to be implemented): For generating answers based on retrieved documents.
     - `ConversationalResponseNode` (to be implemented): For handling conversational queries.
     - `SupabaseDirectLookupNode` (planned, deferred): For direct lookups of specific laws/articles.
